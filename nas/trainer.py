"""
GraphNAS 训练器：对候选架构做短训练并打分（事件级动态图）。
"""

import json
import random
import subprocess
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from data.public_dataset import load_public_dataset
from data.synthetic import generate_synthetic_data, init_dynamic_graph_state
from data.temporal_partition import build_partition_plan
from models.factory import build_model
from models.training import evaluate_ranking_metrics, evaluate_recall_by_type, train_model, train_model_ce
from nas.ray_pipeline import RayPipelineExecutor
from nas.search_space import canonical_config_signature, sanitize_config


class GraphNASTrainer:
    """执行候选架构评估。"""

    def __init__(self, base_config: Dict):
        self.base_config = base_config

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _sample_unique_arch(
        self,
        controller,
        seen_signatures: Set[str],
        max_attempts: int = 64,
    ) -> Tuple[Dict, Optional[torch.Tensor]]:
        attempts = 0

        while attempts < max_attempts:
            if hasattr(controller, "sample_arch_with_logprob"):
                arch, logprob = controller.sample_arch_with_logprob()
            else:
                arch = controller.sample_arch()
                logprob = None

            arch = sanitize_config(arch)
            signature = canonical_config_signature(arch)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                return arch, logprob
            attempts += 1

        # Exhausted unique attempts, return one valid sample to keep the run progressing.
        if hasattr(controller, "sample_arch_with_logprob"):
            arch, logprob = controller.sample_arch_with_logprob()
        else:
            arch = controller.sample_arch()
            logprob = None
        arch = sanitize_config(arch)
        return arch, logprob

    def _sample_unique_arch_batch(
        self,
        controller,
        batch_size: int,
        seen_signatures: Set[str],
    ) -> List[Tuple[Dict, Optional[torch.Tensor]]]:
        samples: List[Tuple[Dict, Optional[torch.Tensor]]] = []
        for _ in range(batch_size):
            arch, logprob = self._sample_unique_arch(controller, seen_signatures)
            samples.append((arch, logprob))
        return samples

    def _prepare_data(self):
        dataset_name = self.base_config.get("dataset", "synthetic")
        feature_dim = self.base_config["feature_dim"]
        max_events = self.base_config.get("max_events", 0)
        train_ratio = float(self.base_config.get("train_ratio", 0.7))
        val_ratio = float(self.base_config.get("val_ratio", 0.1))
        partition_size = int(self.base_config.get("partition_size", 0))
        partition_strategy = self.base_config.get("partition_strategy", "count")

        if train_ratio <= 0 or val_ratio < 0:
            raise ValueError("train_ratio must be > 0 and val_ratio must be >= 0")
        if train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be < 1")

        if dataset_name == "synthetic":
            interactions, user_type_prefs, item_type = generate_synthetic_data(
                num_users=self.base_config["num_users"],
                num_items=self.base_config["num_items"],
                num_interactions=self.base_config["num_interactions"],
                feature_dim=feature_dim,
                seed=self.base_config.get("seed", 42),
            )
        else:
            interactions, num_users, num_items = load_public_dataset(
                dataset_name=dataset_name,
                dataset_dir=self.base_config.get("dataset_dir", "data/public"),
                feature_dim=feature_dim,
                max_events=max_events,
                local_data_path=self.base_config.get("local_data_path", ""),
            )
            self.base_config["num_users"] = num_users
            self.base_config["num_items"] = num_items
            item_type = np.zeros(num_items, dtype=np.int64)
            user_type_prefs = {uid: {0} for uid in range(num_users)}

        interactions = sorted(interactions, key=lambda x: x.timestamp)
        total_events = len(interactions)
        train_end = int(total_events * train_ratio)
        val_end = int(total_events * (train_ratio + val_ratio))

        train_end = max(1, min(train_end, total_events - 2))
        val_end = max(train_end + 1, min(val_end, total_events - 1))

        train_data = interactions[:train_end]
        val_data = interactions[train_end:val_end]
        test_data = interactions[val_end:]

        partition_plan = build_partition_plan(
            train_interactions=train_data,
            val_interactions=val_data,
            test_interactions=test_data,
            partition_size=partition_size if partition_size > 0 else None,
            strategy=partition_strategy,
        )

        graph_template = init_dynamic_graph_state(
            num_users=self.base_config["num_users"],
            num_items=self.base_config["num_items"],
            max_neighbors=self.base_config.get("max_neighbors", 20),
        )
        return train_data, val_data, test_data, user_type_prefs, item_type, graph_template, partition_plan

    def _train_and_eval(
        self,
        config: Dict,
        train_data,
        eval_data,
        user_type_prefs,
        item_type,
        graph_template,
        epochs: int,
        trial_seed: int,
        train_partitions=None,
        eval_partitions=None,
    ) -> Dict[str, float]:
        self._set_seed(trial_seed)
        model = build_model(config)
        model_name = config.get("model", "temporal_event_gnn_jodie")
        graph_ctx = None if model_name == "jodie_rnn" else graph_template

        dataset_name = self.base_config.get("dataset", "synthetic")
        if dataset_name == "synthetic":
            train_model(
                model,
                train_data,
                num_epochs=epochs,
                lr=config.get("lr", 1e-3),
                neg_sample_size=config.get("neg_sample_size", 5),
                graph_ctx=graph_ctx,
                seed=trial_seed,
                partitions=train_partitions,
            )
            value = evaluate_recall_by_type(
                model,
                eval_data,
                item_type,
                user_type_prefs,
                k=config.get("k", 10),
                graph_ctx=graph_ctx,
                partitions=eval_partitions,
            )
            return {
                "mrr": float(value),
                "recall_at_k": float(value),
            }

        train_model_ce(
            model,
            train_data,
            num_epochs=epochs,
            lr=config.get("lr", 1e-3),
            graph_ctx=graph_ctx,
            seed=trial_seed,
            partitions=train_partitions,
        )
        return evaluate_ranking_metrics(
            model,
            eval_data,
            k=config.get("k", 10),
            graph_ctx=graph_ctx,
            partitions=eval_partitions,
        )

    def _selection_score(self, config: Dict, metrics: Dict[str, float]) -> float:
        if self.base_config.get("dataset", "synthetic") == "synthetic":
            return float(metrics["recall_at_k"])
        selection_metric = config.get("selection_metric", "mrr")
        if selection_metric not in {"mrr", "recall_at_k"}:
            raise ValueError(f"Unsupported selection_metric: {selection_metric}")
        return float(metrics[selection_metric])

    def _distribution_metadata(self, train_data, val_data, test_data) -> Dict:
        return {
            "dataset": self.base_config.get("dataset", "synthetic"),
            "max_events": int(self.base_config.get("max_events", 0)),
            "feature_dim": int(self.base_config.get("feature_dim", 8)),
            "lr": float(self.base_config.get("lr", 1e-3)),
            "train_ratio": float(self.base_config.get("train_ratio", 0.7)),
            "val_ratio": float(self.base_config.get("val_ratio", 0.1)),
            "k": int(self.base_config.get("k", 10)),
            "selection_metric": self.base_config.get("selection_metric", "mrr"),
            "seed": int(self.base_config.get("seed", 42)),
            "num_train_events": len(train_data),
            "num_val_events": len(val_data),
            "num_test_events": len(test_data),
        }

    def _family_balanced_candidates(
        self,
        coarse_sorted: List[Dict],
        rerank_top_k: int,
        min_per_model: int,
    ) -> List[Dict]:
        selected: List[Dict] = []
        used_ids = set()

        model_families = sorted({row["config"].get("model", "unknown") for row in coarse_sorted})
        for family in model_families:
            family_rows = [r for r in coarse_sorted if r["config"].get("model", "unknown") == family]
            for row in family_rows[: max(0, min_per_model)]:
                row_id = id(row)
                if row_id in used_ids:
                    continue
                selected.append(row)
                used_ids.add(row_id)
                if len(selected) >= rerank_top_k:
                    return selected

        for row in coarse_sorted:
            row_id = id(row)
            if row_id in used_ids:
                continue
            selected.append(row)
            used_ids.add(row_id)
            if len(selected) >= rerank_top_k:
                break

        return selected

    def _evaluate_arch_multi_seed(
        self,
        arch_config: Dict,
        train_data,
        eval_data,
        user_type_prefs,
        item_type,
        graph_template,
        epochs: int,
        eval_seeds: Optional[List[int]],
        default_seed: int,
        phase: str,
        eval_split: str,
    ) -> Dict:
        seeds = eval_seeds if eval_seeds else [default_seed]
        per_seed_results = []

        for seed in seeds:
            single = self.evaluate_arch(
                arch_config=arch_config,
                train_data=train_data,
                eval_data=eval_data,
                user_type_prefs=user_type_prefs,
                item_type=item_type,
                graph_template=graph_template,
                epochs=epochs,
                trial_seed=int(seed),
                phase=phase,
                eval_split=eval_split,
            )
            per_seed_results.append(single)

        mean_score = float(np.mean([r["score"] for r in per_seed_results]))
        mean_mrr = float(np.mean([r["mrr"] for r in per_seed_results]))
        mean_recall = float(np.mean([r["recall_at_k"] for r in per_seed_results]))
        mean_time = float(np.mean([r["time_sec"] for r in per_seed_results]))

        merged = dict(per_seed_results[0])
        merged["seed"] = int(seeds[0])
        merged["seed_list"] = [int(s) for s in seeds]
        merged["score"] = mean_score
        merged["mrr"] = mean_mrr
        merged["recall_at_k"] = mean_recall
        merged["time_sec"] = round(mean_time, 4)
        if eval_split == "val":
            merged["val_score"] = mean_score
        if eval_split == "test":
            merged["test_score"] = mean_score
        merged["per_seed_metrics"] = [
            {
                "seed": int(r["seed"]),
                "score": float(r["score"]),
                "mrr": float(r["mrr"]),
                "recall_at_k": float(r["recall_at_k"]),
                "time_sec": float(r["time_sec"]),
            }
            for r in per_seed_results
        ]
        return merged

    def evaluate_arch(
        self,
        arch_config: Dict,
        train_data,
        eval_data,
        user_type_prefs,
        item_type,
        graph_template,
        epochs: int,
        trial_seed: int,
        phase: str,
        eval_split: str,
        train_partitions=None,
        eval_partitions=None,
    ) -> Dict:
        config = dict(self.base_config)
        config.update(arch_config)
        graph_template["max_neighbors"] = config.get("max_neighbors", graph_template["max_neighbors"])

        start = time.time()
        metrics = self._train_and_eval(
            config=config,
            train_data=train_data,
            eval_data=eval_data,
            user_type_prefs=user_type_prefs,
            item_type=item_type,
            graph_template=graph_template,
            epochs=epochs,
            trial_seed=trial_seed,
            train_partitions=train_partitions,
            eval_partitions=eval_partitions,
        )
        elapsed = time.time() - start

        model = build_model(config)
        params = sum(p.numel() for p in model.parameters())
        score = self._selection_score(config, metrics)

        return {
            "config": config,
            "phase": phase,
            "eval_split": eval_split,
            "seed": trial_seed,
            "score": float(score),
            "val_score": float(score) if eval_split == "val" else None,
            "test_score": float(score) if eval_split == "test" else None,
            "mrr": float(metrics["mrr"]),
            "recall_at_k": float(metrics["recall_at_k"]),
            "params": int(params),
            "time_sec": round(elapsed, 4),
        }

    def evaluate_arch_pipeline(
        self,
        arch_configs: List[Dict],
        partition_plan,
        user_type_prefs,
        item_type,
        phase: str,
        eval_split: str,
        epochs: int,
    ) -> List[Dict]:
        executor = RayPipelineExecutor(self.base_config, partition_plan)
        start = time.time()
        pipeline_results = executor.run(
            arch_configs,
            user_type_prefs=user_type_prefs,
            item_type=item_type,
            num_train_epochs=epochs,
        )
        elapsed = time.time() - start

        formatted = []
        for row in pipeline_results:
            config = dict(self.base_config)
            config.update(row["config"])
            model = build_model(config)
            params = sum(p.numel() for p in model.parameters())
            formatted.append(
                {
                    "config": config,
                    "phase": phase,
                    "eval_split": eval_split,
                    "seed": int(self.base_config.get("seed", 42)) + row["trial_id"],
                    "score": float(row["score"]),
                    "val_score": float(row["score"]) if eval_split == "val" else None,
                    "test_score": float(row["score"]) if eval_split == "test" else None,
                    "mrr": float(row["mrr"]),
                    "recall_at_k": float(row["recall_at_k"]),
                    "params": int(params),
                    "time_sec": round(elapsed / max(len(pipeline_results), 1), 4),
                }
            )
        return formatted

    def search_pipeline(
        self,
        controller,
        coarse_trials: int,
        architectures_per_step: int,
        coarse_epochs: int,
        rerank_top_k: int = 0,
        rerank_epochs: int = 1,
        family_balanced_rerank: bool = False,
        family_balance_per_model: int = 1,
    ) -> Tuple[Dict, List[Dict]]:
        print(f"\n{'='*70}", flush=True)
        print(f"[GraphNAS] Starting search pipeline", flush=True)
        print(f"[GraphNAS] Coarse trials: {coarse_trials}, epochs: {coarse_epochs}", flush=True)
        print(f"[GraphNAS] Rerank top-k: {rerank_top_k}, epochs: {rerank_epochs}", flush=True)
        print(f"{'='*70}\n", flush=True)
        
        # 启动效率监控（如果启用）
        monitor_process = None
        if self.base_config.get("enable_efficiency_monitor", False) and self.base_config.get("pipeline_trace_log_path"):
            trace_file = self.base_config.get("pipeline_trace_log_path")
            interval = self.base_config.get("efficiency_monitor_interval", 10)
            
            try:
                print(f"[Efficiency Monitor] Starting efficiency monitor (interval: {interval}s)", flush=True)
                monitor_process = subprocess.Popen(
                    ["python", "tools/monitor_pipeline_efficiency.py", trace_file, str(interval)],
                )
                print(f"[Efficiency Monitor] Monitor process started (PID: {monitor_process.pid})", flush=True)
            except Exception as e:
                print(f"[Efficiency Monitor] ⚠️ Failed to start monitor: {e}", flush=True)
                monitor_process = None
        
        train_data, val_data, test_data, user_type_prefs, item_type, graph_template, partition_plan = self._prepare_data()
        results: List[Dict] = []
        total_generated = 0
        seen_signatures: Set[str] = set()

        while total_generated < coarse_trials:
            batch_size = min(architectures_per_step, coarse_trials - total_generated)
            print(f"[Coarse Phase] Sampling batch {total_generated//batch_size + 1}: {batch_size} architectures", flush=True)
            
            samples = self._sample_unique_arch_batch(
                controller=controller,
                batch_size=batch_size,
                seen_signatures=seen_signatures,
            )
            arch_batch = [arch for arch, _ in samples]
            logprobs = [logprob for _, logprob in samples]

            print(f"[Coarse Phase] Evaluating architectures {total_generated+1}-{total_generated+len(arch_batch)}/{coarse_trials}", flush=True)
            batch_results = self.evaluate_arch_pipeline(
                arch_configs=arch_batch,
                partition_plan=partition_plan,
                user_type_prefs=user_type_prefs,
                item_type=item_type,
                phase="coarse_pipeline",
                eval_split="val",
                epochs=coarse_epochs,
            )
            results.extend(batch_results)

            batch_samples = [
                (logprob, result["score"])
                for logprob, result in zip(logprobs, batch_results)
                if logprob is not None
            ]
            if batch_samples and hasattr(controller, "reinforce_step_batch"):
                controller.reinforce_step_batch(batch_samples)
            else:
                for logprob, result in batch_samples:
                    if hasattr(controller, "reinforce_step"):
                        controller.reinforce_step(logprob, result["score"])

            total_generated += len(batch_results)
            print(f"[Coarse Phase] Progress: {total_generated}/{coarse_trials} trials completed", flush=True)

        coarse_sorted = sorted(results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)
        selected = coarse_sorted[0]

        if rerank_top_k > 0:
            if family_balanced_rerank:
                rerank_candidates = self._family_balanced_candidates(
                    coarse_sorted=coarse_sorted,
                    rerank_top_k=rerank_top_k,
                    min_per_model=family_balance_per_model,
                )
            else:
                rerank_candidates = coarse_sorted[:rerank_top_k]

            rerank_configs = [row["config"] for row in rerank_candidates]
            print(f"[Rerank Phase] Evaluating {len(rerank_configs)} candidates with {rerank_epochs} epochs", flush=True)
            rerank_results = self.evaluate_arch_pipeline(
                arch_configs=rerank_configs,
                partition_plan=partition_plan,
                user_type_prefs=user_type_prefs,
                item_type=item_type,
                phase="rerank_pipeline",
                eval_split="val",
                epochs=rerank_epochs,
            )
            results.extend(rerank_results)
            selected = sorted(rerank_results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)[0]

        best = selected
        best["distribution_metadata"] = self._distribution_metadata(train_data, val_data, test_data)
        
        # 停止效率监控
        if monitor_process is not None:
            try:
                monitor_process.terminate()
                monitor_process.wait(timeout=5)
                print(f"[Efficiency Monitor] Monitor process stopped", flush=True)
                
                # 生成效率报告
                trace_file = self.base_config.get("pipeline_trace_log_path")
                if trace_file:
                    efficiency_log = trace_file.replace("pipeline_trace_", "efficiency_log_").replace(".log", ".csv")
                    report_file = efficiency_log.replace(".csv", "_report.txt")
                    try:
                        subprocess.run(
                            ["python", "tools/visualize_efficiency_log.py", efficiency_log, "--export", report_file],
                            timeout=10,
                            check=False,
                        )
                        print(f"[Efficiency Monitor] Report saved to: {report_file}", flush=True)
                    except Exception as e:
                        print(f"[Efficiency Monitor] ⚠️ Failed to generate report: {e}", flush=True)
            except Exception as e:
                print(f"[Efficiency Monitor] ⚠️ Error stopping monitor: {e}", flush=True)
        
        return best, results

    def search(
        self,
        controller,
        coarse_trials: int,
        coarse_epochs: int,
        rerank_top_k: int = 0,
        rerank_epochs: int = 1,
        eval_seeds: Optional[List[int]] = None,
        family_balanced_rerank: bool = False,
        family_balance_per_model: int = 1,
    ) -> Tuple[Dict, List[Dict]]:
        train_data, val_data, test_data, user_type_prefs, item_type, graph_template, partition_plan = self._prepare_data()
        results: List[Dict] = []
        seen_signatures: Set[str] = set()

        for trial in range(coarse_trials):
            arch, logprob = self._sample_unique_arch(
                controller=controller,
                seen_signatures=seen_signatures,
            )

            trial_seed = int(self.base_config.get("seed", 42)) + trial
            result = self._evaluate_arch_multi_seed(
                arch_config=arch,
                train_data=train_data,
                eval_data=val_data,
                user_type_prefs=user_type_prefs,
                item_type=item_type,
                graph_template=graph_template,
                epochs=coarse_epochs,
                eval_seeds=eval_seeds,
                default_seed=trial_seed,
                phase="coarse",
                eval_split="val",
            )
            results.append(result)

            if logprob is not None and hasattr(controller, "reinforce_step"):
                controller.reinforce_step(logprob, result["score"])

            print(
                f"[Coarse {trial + 1}/{coarse_trials}] "
                f"model={result['config'].get('model', 'unknown')} "
                f"agg={result['config'].get('event_agg', 'na')} "
                f"memory={result['config'].get('memory_cell', 'na')} "
                f"time_proj={result['config'].get('time_proj', 'na')} "
                f"val_score={result['score']:.4f}"
            )

        coarse_sorted = sorted(results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)
        selected = coarse_sorted[0]

        if rerank_top_k > 0:
            if family_balanced_rerank:
                rerank_candidates = self._family_balanced_candidates(
                    coarse_sorted=coarse_sorted,
                    rerank_top_k=rerank_top_k,
                    min_per_model=family_balance_per_model,
                )
            else:
                rerank_candidates = coarse_sorted[:rerank_top_k]
            rerank_results = []
            for idx, candidate in enumerate(rerank_candidates):
                rerank_seed = int(self.base_config.get("seed", 42)) + 10000 + idx
                rerank_result = self._evaluate_arch_multi_seed(
                    arch_config=candidate["config"],
                    train_data=train_data,
                    eval_data=val_data,
                    user_type_prefs=user_type_prefs,
                    item_type=item_type,
                    graph_template=graph_template,
                    epochs=rerank_epochs,
                    eval_seeds=eval_seeds,
                    default_seed=rerank_seed,
                    phase="rerank",
                    eval_split="val",
                )
                rerank_results.append(rerank_result)
                print(
                    f"[Rerank {idx + 1}/{len(rerank_candidates)}] "
                    f"model={rerank_result['config'].get('model', 'unknown')} "
                    f"val_score={rerank_result['score']:.4f}"
                )

            results.extend(rerank_results)
            selected = sorted(rerank_results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)[0]

        final_seed = int(self.base_config.get("seed", 42)) + 20000
        final_train_data = train_data + val_data
        final_result = self._evaluate_arch_multi_seed(
            arch_config=selected["config"],
            train_data=final_train_data,
            eval_data=test_data,
            user_type_prefs=user_type_prefs,
            item_type=item_type,
            graph_template=graph_template,
            epochs=rerank_epochs if rerank_top_k > 0 else coarse_epochs,
            eval_seeds=eval_seeds,
            default_seed=final_seed,
            phase="final",
            eval_split="test",
        )

        final_result["selected_val_score"] = float(selected["score"])
        final_result["distribution_metadata"] = self._distribution_metadata(train_data, val_data, test_data)
        results.append(final_result)
        return final_result, results
