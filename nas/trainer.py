"""
GraphNAS 训练器：对候选架构做短训练并打分（事件级动态图）。
"""

import time
from typing import Dict, List, Tuple

from data.synthetic import generate_synthetic_data, init_dynamic_graph_state
from models.factory import build_model
from models.training import evaluate_recall_by_type, train_model


class GraphNASTrainer:
    """执行候选架构评估。"""

    def __init__(self, base_config: Dict):
        self.base_config = base_config

    def _prepare_data(self):
        interactions, user_type_prefs, item_type = generate_synthetic_data(
            num_users=self.base_config["num_users"],
            num_items=self.base_config["num_items"],
            num_interactions=self.base_config["num_interactions"],
            feature_dim=self.base_config["feature_dim"],
            seed=self.base_config.get("seed", 42),
        )
        split = int(len(interactions) * 0.8)
        train_data = interactions[:split]
        test_data = interactions[split:]

        graph_template = init_dynamic_graph_state(
            num_users=self.base_config["num_users"],
            num_items=self.base_config["num_items"],
            max_neighbors=self.base_config.get("max_neighbors", 20),
        )
        return train_data, test_data, user_type_prefs, item_type, graph_template

    def evaluate_arch(self, arch_config: Dict, epochs_per_trial: int = 1) -> Dict:
        config = dict(self.base_config)
        config.update(arch_config)

        train_data, test_data, user_type_prefs, item_type, graph_template = self._prepare_data()
        graph_template["max_neighbors"] = config.get("max_neighbors", graph_template["max_neighbors"])

        model = build_model(config)

        start = time.time()
        train_model(
            model,
            train_data,
            num_epochs=epochs_per_trial,
            lr=config.get("lr", 1e-3),
            neg_sample_size=config.get("neg_sample_size", 5),
            graph_ctx=graph_template,
        )

        score = evaluate_recall_by_type(
            model,
            test_data,
            item_type,
            user_type_prefs,
            k=config.get("k", 10),
            graph_ctx=graph_template,
        )
        elapsed = time.time() - start
        params = sum(p.numel() for p in model.parameters())

        return {
            "config": config,
            "score": float(score),
            "params": int(params),
            "time_sec": round(elapsed, 4),
        }

    def search(self, controller, trials: int, epochs_per_trial: int) -> Tuple[Dict, List[Dict]]:
        results = []

        for trial in range(trials):
            if hasattr(controller, "sample_arch_with_logprob"):
                arch, logprob = controller.sample_arch_with_logprob()
            else:
                arch = controller.sample_arch()
                logprob = None

            result = self.evaluate_arch(arch, epochs_per_trial=epochs_per_trial)
            results.append(result)

            if logprob is not None and hasattr(controller, "reinforce_step"):
                controller.reinforce_step(logprob, result["score"])

            print(
                f"[Trial {trial + 1}/{trials}] "
                f"agg={result['config']['event_agg']} memory={result['config']['memory_cell']} "
                f"time_proj={result['config']['time_proj']} score={result['score']:.4f}"
            )

        best = sorted(results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)[0]
        return best, results
