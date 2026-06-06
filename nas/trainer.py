"""
GraphNAS 训练器：对候选架构做短训练并打分（事件级动态图）。
"""

import csv
import json
import os
import random
import subprocess
import atexit
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
from nas.data_parallel_executor import DataParallelExecutor
from nas.ray_pipeline import RayPipelineExecutor
from nas.search_space import canonical_config_signature, sanitize_config


class GraphNASTrainer:
    """执行候选架构评估。"""

    def __init__(self, base_config: Dict):
        self.base_config = base_config

    def _time_budget_reached(self, search_start_time: Optional[float], time_budget_sec: float) -> bool:
        return bool(time_budget_sec > 0 and search_start_time is not None and (time.time() - search_start_time) >= time_budget_sec)

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
        device = torch.device(self.base_config.get("device", "cpu"))
        model = model.to(device)
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
                batch_training=self.base_config.get("batch_training", False),
                batch_size=self.base_config.get("train_batch_size", 32),
                batch_mode=self.base_config.get("batch_mode", "serial"),
                tgn_loss_mode=self.base_config.get("tgn_loss_mode", "all"),
                tgn_window_size=self.base_config.get("tgn_window_size", 10.0),
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
            batch_training=self.base_config.get("batch_training", False),
            batch_size=self.base_config.get("train_batch_size", 32),
            batch_mode=self.base_config.get("batch_mode", "serial"),
            tgn_loss_mode=self.base_config.get("tgn_loss_mode", "all"),
            tgn_window_size=self.base_config.get("tgn_window_size", 10.0),
        )
        return evaluate_ranking_metrics(
            model,
            eval_data,
            k=config.get("k", 10),
            graph_ctx=graph_ctx,
            partitions=eval_partitions,
            frozen=self.base_config.get("eval_frozen", False),
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
        executor=None,
        time_budget_sec: float = 0.0,
        search_start_time: float = None,
    ) -> List[Dict]:
        own_executor = executor is None
        if own_executor:
            executor = RayPipelineExecutor(self.base_config, partition_plan)
        start = time.time()
        pipeline_results = executor.run(
            arch_configs,
            user_type_prefs=user_type_prefs,
            item_type=item_type,
            num_train_epochs=epochs,
            eval_split=eval_split,
            time_budget_sec=time_budget_sec,
            search_start_time=search_start_time,
        )
        elapsed = time.time() - start
        if own_executor:
            executor.shutdown()

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

    def _search_pipeline_async(
        self,
        controller,
        pipeline_executor,
        coarse_trials: int,
        architectures_per_step: int,
        coarse_epochs: int,
        seen_signatures,
        user_type_prefs,
        item_type,
        partition_plan,
        timing_log_path: str,
        search_start_time: float,
        time_budget_sec: float,
    ):
        """
        Smart 异步搜索循环：
        - 直接用均匀成本计算最优分配（partition 均等切分，profiling 无意义）
        - controller 持续采样，pipeline 持续训练，GPU 始终满载
        - 累积 arch_per_step 个结果后用 off-policy RL 更新 controller
        """
        from nas.config_optimizer import ConfigOptimizer

        # 如果用户显式指定了 stage 数和 worker 分配，直接使用，不走 auto_allocate
        user_stages = self.base_config.get("num_pipeline_stages")
        user_workers = str(self.base_config.get("pipeline_stage_train_workers", "")).strip()
        if user_stages and user_workers:
            print(f"[Smart] Using manual allocation: stages={user_stages}, workers={user_workers}", flush=True)
        else:
            # 自动计算最优分配
            gpu_list = self.base_config.get("gpu_list", "0,1,2")
            gpu_count = len(ConfigOptimizer.parse_gpu_list(gpu_list))
            num_events = int(self.base_config.get("max_events", 0)) or int(sum(
                len(p.interactions) for p in partition_plan.get_split_partitions("train")))
            # 从搜索空间获取最大 embedding_dim 和 max_neighbors
            from nas.search_space import get_search_space
            ss = get_search_space(self.base_config.get("space", "rnn_only"))
            max_emb = max(ss.get("embedding_dim", [128]))
            max_nbr = max(ss.get("max_neighbors", [0]))
            auto_cfg = ConfigOptimizer.auto_allocate_config_advanced(
                gpu_count=gpu_count,
                num_events=num_events,
                num_partitions=0,
                epochs=coarse_epochs,
                num_users=int(self.base_config.get("num_users", 0)),
                num_items=int(self.base_config.get("num_items", 0)),
                max_embedding_dim=max_emb,
                max_neighbors=max_nbr,
            )
            print(f"[Smart] Allocation:\n{auto_cfg['info']}", flush=True)
            for k in ("num_pipeline_stages", "pipeline_stage_train_workers",
                      "pipeline_stage_eval_workers", "partition_size"):
                self.base_config[k] = auto_cfg[k]
        pipeline_executor.__init__(self.base_config, partition_plan)

        # ── 启动持久化 pool ──
        eval_kwargs = {
            "eval_split": "val",
            "item_type": item_type,
            "user_type_prefs": user_type_prefs,
            "k": int(self.base_config.get("k", 10)),
        }
        pipeline_executor.start_persistent_pool(eval_kwargs)

        results = []
        total_workers = sum(len(p) for p in pipeline_executor._pool_workers)
        pending_logprobs = {}
        update_buffer = []
        total_submitted = 0
        remaining = coarse_trials
        cumulative_best = 0.0

        # 预填充：提交 2×arch_per_step 个架构
        prefill = min(architectures_per_step * 2, remaining)
        while total_submitted < prefill:
            samples = self._sample_unique_arch_batch(controller, 1, seen_signatures)
            arch, logprob = samples[0]
            tid = pipeline_executor.submit_arch(arch)
            pending_logprobs[tid] = logprob.clone() if logprob is not None else None
            total_submitted += 1

        # 主循环
        last_print = time.time()
        while len(results) < coarse_trials:
            if time_budget_sec > 0 and (time.time() - search_start_time) >= time_budget_sec:
                print("[Smart] Time budget reached, draining pipeline...", flush=True)
                break

            completed = pipeline_executor.poll_completed(timeout=0.05)
            # 如果 in_flight 为空但 pending 不为空，主动 drain
            if not pipeline_executor._pool_in_flight:
                pipeline_executor._drain_pool()
            now = time.time()
            for r in completed:
                tid = r["trial_id"]
                pending_logprobs.pop(tid, None)  # 旧 logprob 不再使用
                results.append(r)
                cumulative_best = max(cumulative_best, r["score"])
                update_buffer.append((r["config"], r["score"]))
                with open(timing_log_path, "a", newline="", encoding="utf-8") as f:
                    import csv as _csv
                    trial_end_time = now - search_start_time
                    trial_duration = r.get("time_sec", 0)
                    trial_start_time = max(0, trial_end_time - trial_duration)
                    _csv.writer(f).writerow([
                        len(results) - 1, "pipeline_smart",
                        round(trial_start_time, 3),
                        round(trial_end_time, 3),
                        round(trial_duration, 3),
                        round(r["score"], 6), round(r["mrr"], 6),
                        round(r["recall_at_k"], 6), round(cumulative_best, 6),
                        r["config"].get("model", "unknown"),
                    ])

            # 累积够 arch_per_step 个结果后用 off-policy 方式更新 controller
            if len(update_buffer) >= architectures_per_step:
                if hasattr(controller, "compute_logprob") and hasattr(controller, "reinforce_step"):
                    for arch_cfg, sc in update_buffer:
                        logprob = controller.compute_logprob(arch_cfg)
                        controller.reinforce_step(logprob, sc)
                else:
                    for _, sc in update_buffer:
                        controller.reward_baseline = 0.9 * controller.reward_baseline + 0.1 * sc
                update_buffer.clear()

            # 补充提交新架构，保持 pipeline 满载
            headroom = total_workers - len(pipeline_executor._pool_in_flight)
            while headroom > 0 and total_submitted < remaining:
                s = self._sample_unique_arch_batch(controller, 1, seen_signatures)
                arch, logprob = s[0]
                tid = pipeline_executor.submit_arch(arch)
                pending_logprobs[tid] = logprob.clone() if logprob is not None else None
                total_submitted += 1
                headroom -= 1

            in_flight = len(pipeline_executor._pool_in_flight)
            now_t = time.time()
            if completed or now_t - last_print >= 10:
                print(f"[Smart] Progress: {len(results)}/{coarse_trials} completed, "
                      f"in_flight={in_flight}, submitted={total_submitted}", flush=True)
                last_print = now_t

        # flush 剩余 update_buffer
        if update_buffer and hasattr(controller, "compute_logprob") and hasattr(controller, "reinforce_step"):
            for arch_cfg, sc in update_buffer:
                logprob = controller.compute_logprob(arch_cfg)
                controller.reinforce_step(logprob, sc)

        pipeline_executor.shutdown_persistent_pool()
        return results

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
        time_budget_sec: float = 0.0,
    ) -> Tuple[Dict, List[Dict]]:
        print(f"\n{'='*70}", flush=True)
        print(f"[GraphNAS] Starting search pipeline", flush=True)
        print(f"[GraphNAS] Coarse trials: {coarse_trials}, epochs: {coarse_epochs}", flush=True)
        print(f"[GraphNAS] Rerank top-k: {rerank_top_k}, epochs: {rerank_epochs}", flush=True)
        print(f"{'='*70}\n", flush=True)
        
        # ──── 阶段 1: 初始自动化配置（快速启发式） ────
        cost_model = None
        if self.base_config.get("enable_auto_pipeline_config", False):
            from nas.config_optimizer import ConfigOptimizer
            
            gpu_list = self.base_config.get("gpu_list", "0,1,2")
            gpu_count = len(ConfigOptimizer.parse_gpu_list(gpu_list))
            
            # 预计数据大小（需要快速估计，不能完整加载）
            max_events = self.base_config.get("max_events", 0)
            if max_events <= 0:
                dataset_name = self.base_config.get("dataset", "synthetic")
                if dataset_name == "synthetic":
                    num_interactions = self.base_config.get("num_interactions", 3000)
                    estimated_events = num_interactions * 5  # 粗略估计
                else:
                    estimated_events = 50000  # 保守估计
            else:
                estimated_events = max_events
            
            # 调用启发式优化器
            auto_config = ConfigOptimizer.auto_allocate_config_advanced(
                gpu_count=gpu_count,
                num_events=estimated_events,
                num_partitions=0,
                architectures_per_step=architectures_per_step,
                coarse_trials=coarse_trials,
                epochs=coarse_epochs,
            )
            
            print(f"\n[Auto-Config Phase 1] 启发式配置 (GPU数={gpu_count}):")
            print(auto_config['info'])
            print()
            
            self.base_config["num_pipeline_stages"] = auto_config["num_pipeline_stages"]
            self.base_config["pipeline_stage_train_workers"] = auto_config["pipeline_stage_train_workers"]
            self.base_config["pipeline_stage_eval_workers"] = auto_config["pipeline_stage_eval_workers"]
            if auto_config["partition_size"] > 0:
                self.base_config["partition_size"] = auto_config["partition_size"]
            architectures_per_step = auto_config["architectures_per_step"]
        
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
                pid = monitor_process.pid
                print(f"[Efficiency Monitor] Monitor process started (PID: {pid})", flush=True)
                # Auto-stop monitor on any exit (crash or normal)
                atexit.register(lambda p=monitor_process: (
                    p.terminate(), p.wait(timeout=5)
                ) if p is not None and p.poll() is None else None)
            except Exception as e:
                print(f"[Efficiency Monitor] ⚠️ Failed to start monitor: {e}", flush=True)
                monitor_process = None
        
        train_data, val_data, test_data, user_type_prefs, item_type, graph_template, partition_plan = self._prepare_data()
        results: List[Dict] = []
        total_generated = 0
        seen_signatures: Set[str] = set()

        # ──── 阶段 2: 成本感知的 DP 优化（基于实际 partition 数据） ────
        if self.base_config.get("enable_auto_pipeline_config", False) and partition_plan:
            try:
                from nas.config_optimizer import ConfigOptimizer, CostModel
                
                # 估计每个 partition 的成本
                gpu_list = self.base_config.get("gpu_list", "0,1,2")
                gpu_count = len(ConfigOptimizer.parse_gpu_list(gpu_list))
                
                # 收集 partition 成本信息
                train_partitions = partition_plan.get_split_partitions("train")
                if train_partitions:
                    partition_costs = []
                    seen_users = set()
                    seen_items = set()
                    
                    for partition in sorted(train_partitions, key=lambda p: (float(p.start_ts), p.partition_id)):
                        interactions = partition.interactions
                        if not interactions:
                            partition_costs.append(1.0)
                            continue
                        
                        unique_users = {inter.user_id for inter in interactions}
                        unique_items = {inter.item_id for inter in interactions}
                        new_users = unique_users.difference(seen_users)
                        new_items = unique_items.difference(seen_items)
                        span = max(float(partition.end_ts) - float(partition.start_ts), 0.0)
                        
                        cost = float(len(interactions))
                        cost += self.base_config.get("stage_balance_user_weight", 0.25) * float(len(unique_users) + len(new_users))
                        cost += self.base_config.get("stage_balance_item_weight", 0.25) * float(len(unique_items) + len(new_items))
                        cost += self.base_config.get("stage_balance_span_weight", 0.0) * span
                        partition_costs.append(max(cost, 1.0))
                        
                        seen_users.update(unique_users)
                        seen_items.update(unique_items)
                    
                    # 使用 DP 优化得到更好的 stage 配置
                    num_stages = self.base_config.get("num_pipeline_stages", 2)
                    cost_model = CostModel(
                        user_weight=self.base_config.get("stage_balance_user_weight", 0.25),
                        item_weight=self.base_config.get("stage_balance_item_weight", 0.25),
                        span_weight=self.base_config.get("stage_balance_span_weight", 0.0),
                    )
                    
                    grouping = cost_model.optimize_partition_grouping(partition_costs, num_stages)
                    if grouping:
                        print(f"\n[Auto-Config Phase 2] 成本感知 DP 优化:")
                        print(f"  Total partitions: {len(partition_costs)}")
                        print(f"  Total cost: {sum(partition_costs):.0f}")
                        print(f"  Optimized grouping:")
                        imbalance_info = []
                        for i, (start_idx, end_idx) in enumerate(grouping):
                            stage_cost = sum(partition_costs[start_idx:end_idx])
                            imbalance_info.append(f"    Stage {i+1}: partitions {start_idx}-{end_idx-1} (cost={stage_cost:.0f})")
                        print('\n'.join(imbalance_info))
                        print()
            except Exception as e:
                print(f"\n[Auto-Config Phase 2] ⚠️ DP 优化失败: {e}", flush=True)

        # Create pipeline executor once for entire search (reuse across batches)
        pipeline_executor = RayPipelineExecutor(self.base_config, partition_plan)

        # 计时日志初始化
        search_start_time = time.time()
        output_dir = self.base_config.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        timing_log_path = os.path.join(output_dir, "timing_log.csv")
        with open(timing_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_id", "mode", "start_time_s", "end_time_s",
                             "duration_s", "score", "mrr", "recall_at_k",
                             "cumulative_best_score", "model"])
        print(f"[Timing] Timing log: {timing_log_path}", flush=True)

        cumulative_best = 0.0
        pipeline_mode = str(self.base_config.get("pipeline_mode", "naive")).strip().lower()

        if pipeline_mode == "smart":
            # ── Smart 模式：异步 controller + pipeline，先 profiling 再最优分配 ──
            results = self._search_pipeline_async(
                controller=controller,
                pipeline_executor=pipeline_executor,
                coarse_trials=coarse_trials,
                architectures_per_step=architectures_per_step,
                coarse_epochs=coarse_epochs,
                seen_signatures=seen_signatures,
                user_type_prefs=user_type_prefs,
                item_type=item_type,
                partition_plan=partition_plan,
                timing_log_path=timing_log_path,
                search_start_time=search_start_time,
                time_budget_sec=time_budget_sec,
            )
            total_generated = len(results)
        else:
            while total_generated < coarse_trials:
                if time_budget_sec > 0 and (time.time() - search_start_time) >= time_budget_sec:
                    print(f"[Coarse Phase] Time budget {time_budget_sec:.0f}s reached after {total_generated} trials, stopping.", flush=True)
                    break
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
                batch_start = time.time()
                batch_results = self.evaluate_arch_pipeline(
                    arch_configs=arch_batch,
                    partition_plan=partition_plan,
                    user_type_prefs=user_type_prefs,
                    item_type=item_type,
                    phase="coarse_pipeline",
                    eval_split="val",
                    epochs=coarse_epochs,
                    executor=pipeline_executor,
                    time_budget_sec=time_budget_sec,
                    search_start_time=search_start_time,
                )
                batch_end = time.time()
                results.extend(batch_results)

                with open(timing_log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for i, result in enumerate(batch_results):
                        trial_id = total_generated + i
                        cumulative_best = max(cumulative_best, result["score"])
                        writer.writerow([
                            trial_id,
                            "pipeline",
                            round(batch_start - search_start_time, 3),
                            round(batch_end - search_start_time, 3),
                            round(batch_end - batch_start, 3),
                            round(result["score"], 6),
                            round(result["mrr"], 6),
                            round(result["recall_at_k"], 6),
                            round(cumulative_best, 6),
                            result["config"].get("model", "unknown"),
                        ])

                batch_samples = [
                    (logprob, result["score"])
                    for logprob, result in zip(logprobs, batch_results)
                    if logprob is not None
                ]
                if batch_samples and hasattr(controller, "reinforce_step_batch"):
                    controller.reinforce_step_batch(batch_samples)
                else:
                    for logprob, score in batch_samples:
                        if hasattr(controller, "reinforce_step"):
                            controller.reinforce_step(logprob, score)

                total_generated += len(batch_results)
                print(f"[Coarse Phase] Progress: {total_generated}/{coarse_trials} trials completed", flush=True)

        coarse_sorted = sorted(results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)
        selected = coarse_sorted[0]

        if self._time_budget_reached(search_start_time, time_budget_sec):
            print(f"[Coarse Phase] Time budget {time_budget_sec:.0f}s reached before rerank/final test, returning best coarse result.", flush=True)
            best = selected
            best["distribution_metadata"] = self._distribution_metadata(train_data, val_data, test_data)

            if monitor_process is not None:
                try:
                    monitor_process.terminate()
                    monitor_process.wait(timeout=5)
                    print(f"[Efficiency Monitor] Monitor process stopped", flush=True)

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

            pipeline_executor.shutdown()
            return best, results

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
                time_budget_sec=time_budget_sec,
                search_start_time=search_start_time,
            )
            results.extend(rerank_results)
            selected = sorted(rerank_results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)[0]

        # Final evaluation on test set (train on train+val, evaluate on test) — matches JODIE paper protocol
        if self._time_budget_reached(search_start_time, time_budget_sec):
            print(f"[Coarse Phase] Time budget {time_budget_sec:.0f}s reached before final test, returning best available result.", flush=True)
            best = selected
            best["distribution_metadata"] = self._distribution_metadata(train_data, val_data, test_data)

            if monitor_process is not None:
                try:
                    monitor_process.terminate()
                    monitor_process.wait(timeout=5)
                    print(f"[Efficiency Monitor] Monitor process stopped", flush=True)

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

            pipeline_executor.shutdown()
            return best, results

        final_train_data = train_data + val_data
        final_partition_plan = build_partition_plan(
            train_interactions=final_train_data,
            val_interactions=[],
            test_interactions=test_data,
            partition_size=int(self.base_config.get("partition_size", 0)) if int(self.base_config.get("partition_size", 0)) > 0 else None,
            strategy=self.base_config.get("partition_strategy", "count"),
        )
        final_epochs = rerank_epochs if rerank_top_k > 0 else coarse_epochs
        print(f"[Final Test] Evaluating best architecture on test set (fit=train+val, test=test, epochs={final_epochs})", flush=True)
        final_test_result = self.evaluate_arch_pipeline(
            arch_configs=[selected["config"]],
            partition_plan=final_partition_plan,
            user_type_prefs=user_type_prefs,
            item_type=item_type,
            phase="final_pipeline",
            eval_split="test",
            epochs=final_epochs,
            executor=None,
            time_budget_sec=0.0,
            search_start_time=None,
        )[0]
        selected["selected_val_score"] = float(selected["score"])
        selected["val_score"] = float(selected["score"])
        selected["val_mrr"] = float(selected.get("mrr", selected["score"]))
        selected["val_recall_at_k"] = float(selected.get("recall_at_k", 0.0))
        selected["test_score"] = float(final_test_result["score"])
        selected["test_mrr"] = float(final_test_result["mrr"])
        selected["test_recall_at_k"] = float(final_test_result["recall_at_k"])
        # Pipeline Final Test应使用与Serial一致的seed计算: base_seed + 20000
        selected["seed"] = int(self.base_config.get("seed", 42)) + 20000
        # score/mrr/recall_at_k stay as val scores for fair NAS comparison

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

        # Shutdown pipeline executor
        pipeline_executor.shutdown()

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
        time_budget_sec: float = 0.0,
    ) -> Tuple[Dict, List[Dict]]:
        train_data, val_data, test_data, user_type_prefs, item_type, graph_template, partition_plan = self._prepare_data()
        results: List[Dict] = []
        seen_signatures: Set[str] = set()

        # 计时日志初始化
        search_start_time = time.time()
        output_dir = self.base_config.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        timing_log_path = os.path.join(output_dir, "timing_log.csv")
        with open(timing_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_id", "mode", "start_time_s", "end_time_s",
                             "duration_s", "score", "mrr", "recall_at_k",
                             "cumulative_best_score", "model"])
        print(f"[Timing] Timing log: {timing_log_path}", flush=True)

        cumulative_best = 0.0

        for trial in range(coarse_trials):
            if time_budget_sec > 0 and (time.time() - search_start_time) >= time_budget_sec:
                print(f"[Coarse] Time budget {time_budget_sec:.0f}s reached after {trial} trials, stopping.", flush=True)
                break
            arch, logprob = self._sample_unique_arch(
                controller=controller,
                seen_signatures=seen_signatures,
            )

            trial_seed = int(self.base_config.get("seed", 42)) + trial
            trial_start = time.time()
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
            trial_end = time.time()
            results.append(result)

            # 写计时日志
            cumulative_best = max(cumulative_best, result["score"])
            with open(timing_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    trial,
                    "serial",
                    round(trial_start - search_start_time, 3),
                    round(trial_end - search_start_time, 3),
                    round(trial_end - trial_start, 3),
                    round(result["score"], 6),
                    round(result["mrr"], 6),
                    round(result["recall_at_k"], 6),
                    round(cumulative_best, 6),
                    result["config"].get("model", "unknown"),
                ])

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

        if self._time_budget_reached(search_start_time, time_budget_sec):
            print(f"[Coarse] Time budget {time_budget_sec:.0f}s reached before rerank/final test, returning best coarse result.", flush=True)
            final_result = dict(selected)
            final_result["selected_val_score"] = float(selected["score"])
            final_result["val_score"] = float(selected["score"])
            final_result["test_score"] = None
            final_result["distribution_metadata"] = self._distribution_metadata(train_data, val_data, test_data)
            results.append(final_result)
            return final_result, results

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
        final_result["val_score"] = float(selected["score"])
        final_result["val_mrr"] = float(selected.get("mrr", selected["score"]))
        final_result["val_recall_at_k"] = float(selected.get("recall_at_k", 0.0))
        final_result["test_score"] = float(final_result["score"])
        final_result["test_mrr"] = float(final_result["mrr"])
        final_result["test_recall_at_k"] = float(final_result["recall_at_k"])
        # score/mrr/recall_at_k stay as val scores for fair NAS comparison
        final_result["score"] = float(selected["score"])
        final_result["mrr"] = float(selected.get("mrr", selected["score"]))
        final_result["recall_at_k"] = float(selected.get("recall_at_k", 0.0))
        final_result["distribution_metadata"] = self._distribution_metadata(train_data, val_data, test_data)
        results.append(final_result)
        return final_result, results

    def search_data_parallel(
        self,
        controller,
        coarse_trials: int,
        coarse_epochs: int,
        num_workers: int = 3,
        time_budget_sec: float = 0.0,
    ) -> Tuple[Dict, List[Dict]]:
        print(f"\n{'='*70}", flush=True)
        print(f"[GraphNAS] Starting data-parallel search", flush=True)
        print(f"[GraphNAS] Coarse trials: {coarse_trials}, epochs: {coarse_epochs}, workers: {num_workers}", flush=True)
        print(f"{'='*70}\n", flush=True)

        train_data, val_data, test_data, user_type_prefs, item_type, graph_template, partition_plan = self._prepare_data()
        results: List[Dict] = []
        seen_signatures: Set[str] = set()

        search_start_time = time.time()
        output_dir = self.base_config.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        timing_log_path = os.path.join(output_dir, "timing_log.csv")
        with open(timing_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_id", "mode", "start_time_s", "end_time_s",
                             "duration_s", "score", "mrr", "recall_at_k",
                             "cumulative_best_score", "model"])
        print(f"[Timing] Timing log: {timing_log_path}", flush=True)

        executor = DataParallelExecutor(self.base_config, partition_plan, num_workers=num_workers)
        cumulative_best = 0.0
        search_start = time.time()

        for trial_idx in range(coarse_trials):
            if time_budget_sec > 0 and (time.time() - search_start_time) >= time_budget_sec:
                print(f"[DataParallel] Time budget {time_budget_sec:.0f}s reached after {trial_idx} trials, stopping.", flush=True)
                break
            # Sample THEN evaluate THEN update RL — same order as serial search to avoid
            # inplace-modification of the controller's computation graph between sample and backward.
            arch, logprob = self._sample_unique_arch(
                controller=controller,
                seen_signatures=seen_signatures,
            )

            raw_list = executor.run([arch], user_type_prefs=user_type_prefs,
                                    item_type=item_type, num_train_epochs=coarse_epochs)
            raw = raw_list[0]

            config = dict(self.base_config)
            config.update(raw["config"])
            model_obj = build_model(config)
            params = sum(p.numel() for p in model_obj.parameters())

            trial_end_rel = time.time() - search_start
            result = {
                "config": config,
                "phase": "coarse_dp",
                "eval_split": "val",
                "seed": int(self.base_config.get("seed", 42)) + trial_idx,
                "score": float(raw["score"]),
                "val_score": float(raw["score"]),
                "test_score": None,
                "mrr": float(raw["mrr"]),
                "recall_at_k": float(raw["recall_at_k"]),
                "params": int(params),
                "time_sec": float(raw["time_sec"]),
            }
            results.append(result)

            cumulative_best = max(cumulative_best, result["score"])
            with open(timing_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    trial_idx, "data_parallel",
                    round(trial_end_rel - result["time_sec"], 3),
                    round(trial_end_rel, 3),
                    round(result["time_sec"], 3),
                    round(result["score"], 6),
                    round(result["mrr"], 6),
                    round(result["recall_at_k"], 6),
                    round(cumulative_best, 6),
                    result["config"].get("model", "unknown"),
                ])

            if logprob is not None and hasattr(controller, "reinforce_step"):
                controller.reinforce_step(logprob, result["score"])

            print(
                f"[DataParallel {trial_idx + 1}/{coarse_trials}] "
                f"model={result['config'].get('model', 'unknown')} "
                f"agg={result['config'].get('event_agg', 'na')} "
                f"memory={result['config'].get('memory_cell', 'na')} "
                f"val_score={result['score']:.4f}"
            )

        # Kill Ray workers so pipeline can have full resources
        executor.shutdown()
        print("[DataParallel] Workers shut down.", flush=True)

        coarse_sorted = sorted(results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)
        selected = coarse_sorted[0]

        if self._time_budget_reached(search_start_time, time_budget_sec):
            print(f"[DataParallel] Time budget {time_budget_sec:.0f}s reached before final test, returning best coarse result.", flush=True)
            final_result = dict(selected)
            final_result["selected_val_score"] = float(selected["score"])
            final_result["val_score"] = float(selected["score"])
            final_result["test_score"] = None
            final_result["distribution_metadata"] = self._distribution_metadata(train_data, val_data, test_data)
            results.append(final_result)
            return final_result, results

        # Final evaluation on test set
        final_seed = int(self.base_config.get("seed", 42)) + 20000
        final_train_data = train_data + val_data
        final_result = self._evaluate_arch_multi_seed(
            arch_config=selected["config"],
            train_data=final_train_data,
            eval_data=test_data,
            user_type_prefs=user_type_prefs,
            item_type=item_type,
            graph_template=graph_template,
            epochs=coarse_epochs,
            eval_seeds=None,
            default_seed=final_seed,
            phase="final_dp",
            eval_split="test",
        )

        final_result["selected_val_score"] = float(selected["score"])
        final_result["val_score"] = float(selected["score"])
        final_result["val_mrr"] = float(selected.get("mrr", selected["score"]))
        final_result["val_recall_at_k"] = float(selected.get("recall_at_k", 0.0))
        final_result["test_score"] = float(final_result["score"])
        final_result["test_mrr"] = float(final_result["mrr"])
        final_result["test_recall_at_k"] = float(final_result["recall_at_k"])
        # score/mrr/recall_at_k stay as val scores for fair NAS comparison
        final_result["score"] = float(selected["score"])
        final_result["mrr"] = float(selected.get("mrr", selected["score"]))
        final_result["recall_at_k"] = float(selected.get("recall_at_k", 0.0))
        final_result["distribution_metadata"] = self._distribution_metadata(train_data, val_data, test_data)
        results.append(final_result)
        return final_result, results
