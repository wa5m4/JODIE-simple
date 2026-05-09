"""
MemShare-DP baseline search script.

Mirrors search.py's data_parallel branch but uses MemShareDPExecutor,
which applies MemShare's Hotspot Memory Sharing (VLDB 2025) on top of
standard data parallelism:
  - Hot nodes (top-k% by frequency) use smooth aggregation (weighted avg)
  - Cold nodes use max-timestamp merge (same as plain data parallel)
"""

import argparse
import csv
import json
import os
import time

import torch

from nas.controller import RLGraphNASController, RandomGraphNASController
from nas.memshare_dp_executor import MemShareDPExecutor
from nas.search_space import get_search_space, canonical_config_signature, sanitize_config
from data.public_dataset import load_public_dataset
from data.synthetic import generate_synthetic_data
from data.temporal_partition import build_partition_plan
from models.factory import build_model
from models.training import evaluate_ranking_metrics, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="MemShare-DP baseline search")
    parser.add_argument("--space", choices=["small", "paper_compare", "rnn_only"], default="small")
    parser.add_argument("--search-mode", choices=["random", "rl"], default="rl")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--epochs-per-trial", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-gpus", type=float, default=1.0)
    parser.add_argument("--visible-gpus", type=str, default="0,1,2")
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--hot-node-ratio", type=float, default=0.1,
                        help="Top-k fraction of users/items treated as hot nodes (MemShare).")
    parser.add_argument("--dataset", choices=["synthetic", "wikipedia", "reddit", "public_csv"], default="synthetic")
    parser.add_argument("--dataset-dir", type=str, default="data/public")
    parser.add_argument("--local-data-path", type=str, default="")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--partition-size", type=int, default=0)
    parser.add_argument("--num-users", type=int, default=500)
    parser.add_argument("--num-items", type=int, default=1000)
    parser.add_argument("--num-interactions", type=int, default=3000)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg-sample-size", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--selection-metric", choices=["mrr", "recall_at_k"], default="mrr")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller-lr", type=float, default=1e-2)
    parser.add_argument("--time-budget-sec", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="outputs/memshare_dp")
    return parser.parse_args()


def _load_data(args):
    if args.dataset == "synthetic":
        import random, numpy as np
        random.seed(args.seed); np.random.seed(args.seed)
        interactions = generate_synthetic_data(
            num_users=args.num_users, num_items=args.num_items,
            num_interactions=args.num_interactions, feature_dim=args.feature_dim,
        )
    else:
        interactions = load_public_dataset(
            dataset=args.dataset, dataset_dir=args.dataset_dir,
            local_data_path=args.local_data_path, max_events=args.max_events,
        )

    n = len(interactions)
    train_end = int(n * args.train_ratio)
    val_end = int(n * (args.train_ratio + args.val_ratio))
    train_data = interactions[:train_end]
    val_data = interactions[train_end:val_end]
    test_data = interactions[val_end:]

    partition_size = args.partition_size if args.partition_size > 0 else max(1, len(train_data))
    micro_batch_size = args.micro_batch_size if args.micro_batch_size > 0 else max(1, n // 100)

    partition_plan = build_partition_plan(
        train_data, val_data, test_data, partition_size=partition_size,
    )
    return train_data, val_data, test_data, partition_plan, micro_batch_size


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    train_data, val_data, test_data, partition_plan, micro_batch_size = _load_data(args)

    base_config = {
        "dataset": args.dataset,
        "dataset_dir": args.dataset_dir,
        "local_data_path": args.local_data_path,
        "num_users": args.num_users,
        "num_items": args.num_items,
        "feature_dim": args.feature_dim,
        "lr": args.lr,
        "neg_sample_size": args.neg_sample_size,
        "k": args.k,
        "selection_metric": args.selection_metric,
        "device": device,
        "seed": args.seed,
        "data_parallel_visible_gpus": args.visible_gpus,
        "data_parallel_worker_gpus": args.worker_gpus,
        "data_parallel_micro_batch_size": micro_batch_size,
        "output_dir": args.output_dir,
    }

    search_space = get_search_space(args.space)
    if args.search_mode == "rl":
        controller = RLGraphNASController(search_space, seed=args.seed, lr=args.controller_lr)
    else:
        controller = RandomGraphNASController(search_space, seed=args.seed)

    executor = MemShareDPExecutor(
        base_config, partition_plan,
        num_workers=args.workers,
        hot_node_ratio=args.hot_node_ratio,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    timing_log = os.path.join(args.output_dir, "timing_log.csv")
    with open(timing_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["trial_id", "mode", "start_s", "end_s", "duration_s",
                                 "score", "mrr", "recall_at_k", "cumulative_best", "model"])

    results = []
    seen: set = set()
    search_start = time.time()
    cumulative_best = 0.0

    for trial_idx in range(args.trials):
        if args.time_budget_sec > 0 and (time.time() - search_start) >= args.time_budget_sec:
            print(f"[MemShare-DP] Time budget reached after {trial_idx} trials.", flush=True)
            break

        # Sample unique architecture
        for _ in range(50):
            arch = controller.sample_arch()
            arch = sanitize_config(arch, search_space)
            sig = canonical_config_signature(arch)
            if sig not in seen:
                seen.add(sig)
                break
        logprob = getattr(controller, "_last_logprob", None)

        t0 = time.time() - search_start
        raw_list = executor.run([arch], num_train_epochs=args.epochs_per_trial)
        raw = raw_list[0]
        t1 = time.time() - search_start

        config = dict(base_config)
        config.update(raw["config"])
        params = sum(p.numel() for p in build_model(config).parameters())

        result = {
            "config": config,
            "score": float(raw["score"]),
            "mrr": float(raw["mrr"]),
            "recall_at_k": float(raw["recall_at_k"]),
            "params": int(params),
            "time_sec": float(raw["time_sec"]),
        }
        results.append(result)
        cumulative_best = max(cumulative_best, result["score"])

        with open(timing_log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                trial_idx, "memshare_dp", round(t0, 3), round(t1, 3),
                round(result["time_sec"], 3), round(result["score"], 6),
                round(result["mrr"], 6), round(result["recall_at_k"], 6),
                round(cumulative_best, 6), config.get("model", "unknown"),
            ])

        if hasattr(controller, "reinforce_step") and logprob is not None:
            controller.reinforce_step(logprob, result["score"])

        print(f"[MemShare-DP {trial_idx+1}/{args.trials}] "
              f"model={config.get('model','?')} val_score={result['score']:.4f}", flush=True)

    executor.shutdown()

    best = max(results, key=lambda x: x["score"])
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "best_arch.json"), "w") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    print(f"\nBest val score: {best['score']:.4f}")
    print(f"Best model: {best['config'].get('model', '?')}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
