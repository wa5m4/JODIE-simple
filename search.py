"""
GraphNAS 搜索入口：搜索事件级时序 GNN-JODIE。
"""

import argparse
import csv
import json
import os

from nas.controller import RLGraphNASController, RandomGraphNASController
from nas.search_space import get_small_search_space
from nas.trainer import GraphNASTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="GraphNAS search for event-level temporal GNN JODIE")
    parser.add_argument("--space", choices=["small"], default="small")
    parser.add_argument("--search-mode", choices=["random", "rl"], default="rl")
    parser.add_argument("--controller-lr", type=float, default=1e-2)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--epochs-per-trial", type=int, default=1)
    parser.add_argument("--num-users", type=int, default=500)
    parser.add_argument("--num-items", type=int, default=1000)
    parser.add_argument("--num-interactions", type=int, default=3000)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg-sample-size", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def save_results(best, results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    best_path = os.path.join(output_dir, "best_arch.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "leaderboard.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "score", "params", "time_sec", "model", "config_json"])
        sorted_results = sorted(results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)
        for idx, row in enumerate(sorted_results, start=1):
            writer.writerow(
                [
                    idx,
                    row["score"],
                    row["params"],
                    row["time_sec"],
                    row["config"].get("model", "unknown"),
                    json.dumps(row["config"], ensure_ascii=False),
                ]
            )

    print(f"Best architecture saved to: {best_path}")
    print(f"Leaderboard saved to: {csv_path}")


def main():
    args = parse_args()

    if args.space != "small":
        raise ValueError("Only small search space is implemented in this version.")

    search_space = get_small_search_space()
    if args.search_mode == "rl":
        controller = RLGraphNASController(search_space, seed=args.seed, lr=args.controller_lr)
    else:
        controller = RandomGraphNASController(search_space, seed=args.seed)

    base_config = {
        "num_users": args.num_users,
        "num_items": args.num_items,
        "num_interactions": args.num_interactions,
        "feature_dim": args.feature_dim,
        "lr": args.lr,
        "neg_sample_size": args.neg_sample_size,
        "k": args.k,
        "seed": args.seed,
    }

    trainer = GraphNASTrainer(base_config)
    best, results = trainer.search(
        controller=controller,
        trials=args.trials,
        epochs_per_trial=args.epochs_per_trial,
    )

    save_results(best, results, args.output_dir)
    print(f"Search mode: {args.search_mode}")
    print(f"Best score Recall@{args.k}: {best['score']:.4f}")
    print(f"Best model family: {best['config'].get('model', 'temporal_event_gnn_jodie')}")


if __name__ == "__main__":
    main()
