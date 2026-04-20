"""
GraphNAS 训练器：对候选架构做短训练并打分（事件级动态图）。
"""

import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from data.public_dataset import load_public_dataset
from data.synthetic import generate_synthetic_data, init_dynamic_graph_state
from models.factory import build_model
from models.training import evaluate_ranking_metrics, evaluate_recall_by_type, train_model, train_model_ce


class GraphNASTrainer:
    """执行候选架构评估。"""

    def __init__(self, base_config: Dict):
        self.base_config = base_config

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _prepare_data(self):
        dataset_name = self.base_config.get("dataset", "synthetic")
        feature_dim = self.base_config["feature_dim"]
        max_events = self.base_config.get("max_events", 0)
        train_ratio = float(self.base_config.get("train_ratio", 0.7))
        val_ratio = float(self.base_config.get("val_ratio", 0.1))

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

        total_events = len(interactions)
        train_end = int(total_events * train_ratio)
        val_end = int(total_events * (train_ratio + val_ratio))

        train_end = max(1, min(train_end, total_events - 2))
        val_end = max(train_end + 1, min(val_end, total_events - 1))

        train_data = interactions[:train_end]
        val_data = interactions[train_end:val_end]
        test_data = interactions[val_end:]

        graph_template = init_dynamic_graph_state(
            num_users=self.base_config["num_users"],
            num_items=self.base_config["num_items"],
            max_neighbors=self.base_config.get("max_neighbors", 20),
        )
        return train_data, val_data, test_data, user_type_prefs, item_type, graph_template

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
            )
            value = evaluate_recall_by_type(
                model,
                eval_data,
                item_type,
                user_type_prefs,
                k=config.get("k", 10),
                graph_ctx=graph_ctx,
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
        )
        return evaluate_ranking_metrics(
            model,
            eval_data,
            k=config.get("k", 10),
            graph_ctx=graph_ctx,
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

    def search(
        self,
        controller,
        coarse_trials: int,
        coarse_epochs: int,
        rerank_top_k: int = 0,
        rerank_epochs: int = 1,
    ) -> Tuple[Dict, List[Dict]]:
        train_data, val_data, test_data, user_type_prefs, item_type, graph_template = self._prepare_data()
        results: List[Dict] = []

        for trial in range(coarse_trials):
            if hasattr(controller, "sample_arch_with_logprob"):
                arch, logprob = controller.sample_arch_with_logprob()
            else:
                arch = controller.sample_arch()
                logprob = None

            trial_seed = int(self.base_config.get("seed", 42)) + trial
            result = self.evaluate_arch(
                arch_config=arch,
                train_data=train_data,
                eval_data=val_data,
                user_type_prefs=user_type_prefs,
                item_type=item_type,
                graph_template=graph_template,
                epochs=coarse_epochs,
                trial_seed=trial_seed,
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
                f"val_score={result['score']:.4f}"
            )

        coarse_sorted = sorted(results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)
        selected = coarse_sorted[0]

        if rerank_top_k > 0:
            rerank_candidates = coarse_sorted[:rerank_top_k]
            rerank_results = []
            for idx, candidate in enumerate(rerank_candidates):
                rerank_seed = int(self.base_config.get("seed", 42)) + 10000 + idx
                rerank_result = self.evaluate_arch(
                    arch_config=candidate["config"],
                    train_data=train_data,
                    eval_data=val_data,
                    user_type_prefs=user_type_prefs,
                    item_type=item_type,
                    graph_template=graph_template,
                    epochs=rerank_epochs,
                    trial_seed=rerank_seed,
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
        final_result = self.evaluate_arch(
            arch_config=selected["config"],
            train_data=final_train_data,
            eval_data=test_data,
            user_type_prefs=user_type_prefs,
            item_type=item_type,
            graph_template=graph_template,
            epochs=rerank_epochs if rerank_top_k > 0 else coarse_epochs,
            trial_seed=final_seed,
            phase="final",
            eval_split="test",
        )

        final_result["selected_val_score"] = float(selected["score"])
        final_result["distribution_metadata"] = self._distribution_metadata(train_data, val_data, test_data)
        results.append(final_result)
        return final_result, results
