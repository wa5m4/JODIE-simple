"""
GraphNAS 搜索入口：搜索事件级时序 GNN-JODIE。
"""

import argparse
import csv
import json
import os
import time

from nas.controller import RLGraphNASController, RandomGraphNASController
from nas.search_space import get_search_space
from nas.trainer import GraphNASTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="GraphNAS search for event-level temporal GNN JODIE")
    parser.add_argument("--space", choices=["small", "paper_compare"], default="small", help="Search space name.")
    parser.add_argument("--search-mode", choices=["random", "rl"], default="rl", help="Architecture search mode.")
    parser.add_argument("--execution-mode", choices=["serial", "ray_pipeline", "data_parallel"], default="serial", help="Execution backend.")
    parser.add_argument("--data-parallel-workers", type=int, default=3, help="Number of Ray workers for data-parallel intra-trial parallelism.")
    parser.add_argument("--data-parallel-worker-gpus", type=float, default=1.0, help="GPU fraction for each data-parallel worker.")
    parser.add_argument("--data-parallel-visible-gpus", type=str, default="0,1,2", help="CUDA_VISIBLE_DEVICES for data-parallel Ray workers.")
    parser.add_argument("--controller-lr", type=float, default=1e-2, help="Learning rate for RL controller.")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "wikipedia", "reddit", "public_csv"],
        default="synthetic",
        help="Dataset source.",
    )
    parser.add_argument("--dataset-dir", type=str, default="data/public", help="Directory for public dataset files.")
    parser.add_argument(
        "--local-data-path",
        type=str,
        default="",
        help="Local CSV path. Overrides dataset-dir and auto-download when provided.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--max-events", type=int, default=0, help="Use first N events. 0 means all events.")
    parser.add_argument("--trials", type=int, default=6, help="Legacy alias of coarse-trials when coarse-trials is not set.")
    parser.add_argument("--epochs-per-trial", type=int, default=1, help="Legacy alias of coarse-epochs when coarse-epochs is not set.")
    parser.add_argument("--coarse-trials", type=int, default=0, help="Stage-1 number of sampled architectures. 0 uses --trials.")
    parser.add_argument("--coarse-epochs", type=int, default=0, help="Stage-1 epochs per architecture. 0 uses --epochs-per-trial.")
    parser.add_argument("--rerank-top-k", type=int, default=0, help="Stage-2 rerank top-k candidates. 0 disables stage-2.")
    parser.add_argument("--rerank-epochs", type=int, default=0, help="Stage-2 epochs per candidate. 0 uses coarse-epochs.")
    parser.add_argument("--architectures-per-step", type=int, default=2, help="Number of architectures evaluated concurrently in pipeline mode.")
    parser.add_argument("--partition-size", type=int, default=0, help="Temporal partition size. 0 keeps one partition per split.")
    parser.add_argument("--partition-strategy", choices=["count"], default="count", help="Temporal partition strategy.")
    parser.add_argument("--num-pipeline-stages", type=int, default=2, help="Number of pipeline worker stages.")
    parser.add_argument("--pipeline-worker-gpus", type=float, default=0.0, help="GPU resources requested by each pipeline worker stage.")
    parser.add_argument("--pipeline-worker-cpus", type=float, default=1.0, help="CPU resources requested by each pipeline worker stage.")
    parser.add_argument(
        "--pipeline-stage-train-workers",
        type=str,
        default="",
        help="Train stage worker counts. Use one int for all stages (e.g. 2) or comma-separated per stage (e.g. 2,1,1). Empty defaults to 1 per stage.",
    )
    parser.add_argument(
        "--pipeline-stage-eval-workers",
        type=str,
        default="",
        help="Eval stage worker counts. Use one int for all stages or comma-separated per stage. Empty defaults to 1 per stage.",
    )
    parser.add_argument(
        "--stage-balance-strategy",
        choices=["cost", "count"],
        default="cost",
        help="Pipeline stage partitioning strategy. cost=cost-aware contiguous split, count=even partition count split.",
    )
    parser.add_argument(
        "--stage-balance-user-weight",
        type=float,
        default=0.25,
        help="User diversity weight in partition cost estimation when stage-balance-strategy=cost.",
    )
    parser.add_argument(
        "--stage-balance-item-weight",
        type=float,
        default=0.25,
        help="Item diversity weight in partition cost estimation when stage-balance-strategy=cost.",
    )
    parser.add_argument(
        "--stage-balance-span-weight",
        type=float,
        default=0.0,
        help="Time-span weight in partition cost estimation when stage-balance-strategy=cost.",
    )
    parser.add_argument("--ray-address", type=str, default="", help="Ray cluster address. Empty means start local Ray runtime.")
    parser.add_argument("--pipeline-trace", action="store_true", help="Print per-trial per-stage pipeline dispatch/complete timestamps.")
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default="",
        help="Comma-separated seeds for multi-seed averaging per architecture, e.g. 42,43,44. Empty means single-seed.",
    )
    parser.add_argument(
        "--family-balanced-rerank",
        action="store_true",
        help="Enable family-balanced rerank candidate selection to avoid one model family dominating top-k.",
    )
    parser.add_argument(
        "--family-balance-per-model",
        type=int,
        default=1,
        help="Minimum number of candidates per model family when family-balanced-rerank is enabled.",
    )
    parser.add_argument("--num-users", type=int, default=500, help="Synthetic-only: number of users.")
    parser.add_argument("--num-items", type=int, default=1000, help="Synthetic-only: number of items.")
    parser.add_argument("--num-interactions", type=int, default=3000, help="Synthetic-only: number of interactions.")
    parser.add_argument("--feature-dim", type=int, default=8, help="Input feature dimension.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Model training learning rate.")
    parser.add_argument("--neg-sample-size", type=int, default=5, help="Negative sample size for BPR training.")
    parser.add_argument("--k", type=int, default=10, help="Recall@K metric K.")
    parser.add_argument(
        "--selection-metric",
        choices=["mrr", "recall_at_k"],
        default="mrr",
        help="Primary score used to rank architectures on public datasets.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto' (cuda if available), 'cuda', 'cpu'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for search outputs.")
    parser.add_argument("--enable-efficiency-monitor", action="store_true", help="Enable real-time efficiency monitoring during pipeline execution.")
    parser.add_argument("--efficiency-monitor-interval", type=int, default=10, help="Efficiency monitoring sampling interval in seconds (when enabled).")
    return parser.parse_args()


def save_results(best, results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    best_path = os.path.join(output_dir, "best_arch.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "leaderboard.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "phase", "eval_split", "score", "val_score", "test_score", "mrr", "recall_at_k", "params", "time_sec", "model", "config_json"])
        sorted_results = sorted(results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)
        for idx, row in enumerate(sorted_results, start=1):
            writer.writerow(
                [
                    idx,
                    row.get("phase", "na"),
                    row.get("eval_split", "na"),
                    row["score"],
                    row.get("val_score"),
                    row.get("test_score"),
                    row.get("mrr"),
                    row.get("recall_at_k"),
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

    search_space = get_search_space(args.space)
    if args.search_mode == "rl":
        controller = RLGraphNASController(search_space, seed=args.seed, lr=args.controller_lr)
    else:
        controller = RandomGraphNASController(search_space, seed=args.seed)

    pipeline_trace_log_path = ""
    if args.execution_mode == "ray_pipeline":
        os.makedirs(args.output_dir, exist_ok=True)
        run_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        pipeline_trace_log_path = os.path.join(args.output_dir, f"pipeline_trace_{run_tag}.log")
        with open(pipeline_trace_log_path, "w", encoding="utf-8") as f:
            f.write(f"# pipeline trace log\n")
            f.write(f"# created_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        print(f"Pipeline trace log file: {pipeline_trace_log_path}")

    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    base_config = {
        "dataset": args.dataset,
        "dataset_dir": args.dataset_dir,
        "local_data_path": args.local_data_path,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "max_events": args.max_events,
        "num_users": args.num_users,
        "num_items": args.num_items,
        "num_interactions": args.num_interactions,
        "feature_dim": args.feature_dim,
        "lr": args.lr,
        "neg_sample_size": args.neg_sample_size,
        "k": args.k,
        "selection_metric": args.selection_metric,
        "device": device,
        "seed": args.seed,
        "partition_size": args.partition_size,
        "partition_strategy": args.partition_strategy,
        "num_pipeline_stages": args.num_pipeline_stages,
        "pipeline_worker_gpus": args.pipeline_worker_gpus,
        "pipeline_worker_cpus": args.pipeline_worker_cpus,
        "pipeline_stage_train_workers": args.pipeline_stage_train_workers,
        "pipeline_stage_eval_workers": args.pipeline_stage_eval_workers,
        "stage_balance_strategy": args.stage_balance_strategy,
        "stage_balance_user_weight": args.stage_balance_user_weight,
        "stage_balance_item_weight": args.stage_balance_item_weight,
        "stage_balance_span_weight": args.stage_balance_span_weight,
        "ray_address": args.ray_address,
        "pipeline_trace": args.pipeline_trace,
        "pipeline_trace_log_path": pipeline_trace_log_path,
        "output_dir": args.output_dir,
        "enable_efficiency_monitor": args.enable_efficiency_monitor,
        "efficiency_monitor_interval": args.efficiency_monitor_interval,
        "data_parallel_workers": args.data_parallel_workers,
        "data_parallel_worker_gpus": args.data_parallel_worker_gpus,
        "data_parallel_visible_gpus": args.data_parallel_visible_gpus,
    }

    trainer = GraphNASTrainer(base_config)
    coarse_trials = args.coarse_trials if args.coarse_trials > 0 else args.trials
    coarse_epochs = args.coarse_epochs if args.coarse_epochs > 0 else args.epochs_per_trial
    rerank_epochs = args.rerank_epochs if args.rerank_epochs > 0 else coarse_epochs
    eval_seeds = [int(x.strip()) for x in args.eval_seeds.split(",") if x.strip()] if args.eval_seeds else None

    if args.execution_mode == "ray_pipeline":
        best, results = trainer.search_pipeline(
            controller=controller,
            coarse_trials=coarse_trials,
            architectures_per_step=args.architectures_per_step,
            coarse_epochs=coarse_epochs,
            rerank_top_k=args.rerank_top_k,
            rerank_epochs=rerank_epochs,
            family_balanced_rerank=args.family_balanced_rerank,
            family_balance_per_model=args.family_balance_per_model,
        )
    elif args.execution_mode == "data_parallel":
        best, results = trainer.search_data_parallel(
            controller=controller,
            coarse_trials=coarse_trials,
            coarse_epochs=coarse_epochs,
            num_workers=args.data_parallel_workers,
        )
    else:
        best, results = trainer.search(
            controller=controller,
            coarse_trials=coarse_trials,
            coarse_epochs=coarse_epochs,
            rerank_top_k=args.rerank_top_k,
            rerank_epochs=rerank_epochs,
            eval_seeds=eval_seeds,
            family_balanced_rerank=args.family_balanced_rerank,
            family_balance_per_model=args.family_balance_per_model,
        )

    save_results(best, results, args.output_dir)
    print(f"Search mode: {args.search_mode}")
    print(f"Best selection score: {best.get('selected_val_score', best['score']):.4f}")
    print(f"Best test score: {best['score']:.4f}")
    print(f"Best model family: {best['config'].get('model', 'temporal_event_gnn_jodie')}")


if __name__ == "__main__":
    main()
