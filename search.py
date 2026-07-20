"""
GraphNAS 搜索入口：搜索事件级时序 GNN-JODIE 最优架构。

支持三种执行模式:
  - serial:        单机串行搜索
  - data_parallel: Ray 数据并行搜索
  - ray_pipeline:  Ray 流水线并行搜索

用法:
  # 冒烟测试
  python search.py --space small --execution-mode serial --trials 2 --epochs-per-trial 1

  # 完整搜索
  python search.py --space rnn_only --execution-mode ray_pipeline \
      --dataset public_csv --local-data-path data/public/mooc.csv \
      --coarse-trials 32 --coarse-epochs 4 --rerank-top-k 8
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from jodie.nas.controller import RLGraphNASController, RandomGraphNASController
from jodie.nas.search_space import get_search_space
from jodie.nas.trainer import GraphNASTrainer


# ---------------------------------------------------------------------------
# 配置 dataclass — 集中管理所有参数及其默认值
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    """NAS 搜索的完整配置。"""
    # 搜索
    space: str = "small"
    search_mode: str = "rl"
    execution_mode: str = "serial"

    # 阶段参数
    coarse_trials: int = 0
    coarse_epochs: int = 0
    trials: int = 6          # 向后兼容 (coarse_trials 未设置时使用)
    epochs_per_trial: int = 1
    rerank_top_k: int = 0
    rerank_epochs: int = 0

    # 控制器
    controller_lr: float = 1e-2

    # 数据
    dataset: str = "synthetic"
    dataset_dir: str = "data/public"
    local_data_path: str = ""
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    max_events: int = 0

    # 合成数据
    num_users: int = 500
    num_items: int = 1000
    num_interactions: int = 3000

    # 训练
    feature_dim: int = 8
    lr: float = 1e-3
    neg_sample_size: int = 5
    k: int = 10
    selection_metric: str = "mrr"
    batch_training: bool = False
    train_batch_size: int = 32
    batch_mode: str = "tbatch"
    tgn_loss_mode: str = "all"
    tgn_window_size: float = 10.0
    eval_frozen: bool = False

    # 设备
    device: str = "auto"
    seed: int = 42
    output_dir: str = "outputs"

    # 时间分区
    partition_size: int = 0
    partition_strategy: str = "count"
    partition_overlap_ratio: float = 0.0

    # 流水线
    architectures_per_step: int = 2
    num_pipeline_stages: int = 2
    pipeline_worker_gpus: float = 0.0
    pipeline_worker_cpus: float = 1.0
    pipeline_stage_train_workers: str = ""
    pipeline_stage_eval_workers: str = ""
    stage_balance_strategy: str = "cost"
    stage_balance_user_weight: float = 0.25
    stage_balance_item_weight: float = 0.25
    stage_balance_span_weight: float = 0.0
    pipeline_mode: str = "naive"
    pipeline_trace: bool = False

    # Ray 设置
    ray_address: str = ""

    # GPU 自动配置
    gpu_list: str = "0,1,2"
    enable_auto_pipeline_config: bool = False

    # 数据并行
    data_parallel_workers: int = 3
    data_parallel_worker_gpus: float = 1.0
    data_parallel_visible_gpus: str = "0,1,2"
    data_parallel_sync_level: str = "micro_batch"
    data_parallel_micro_batch_size: int = 0

    # 评估
    eval_seeds_str: str = ""
    family_balanced_rerank: bool = False
    family_balance_per_model: int = 1

    # 效率监控
    enable_efficiency_monitor: bool = False
    efficiency_monitor_interval: int = 10

    # 时间预算
    time_budget_sec: float = 0.0

    @property
    def eval_seeds(self) -> Optional[List[int]]:
        if not self.eval_seeds_str:
            return None
        return [int(x.strip()) for x in self.eval_seeds_str.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# CLI 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> SearchConfig:
    parser = argparse.ArgumentParser(description="GraphNAS search for event-level temporal GNN JODIE")

    # 核心参数
    parser.add_argument("--space", choices=["small", "paper_compare", "rnn_only", "mixed"], default="small")
    parser.add_argument("--search-mode", choices=["random", "rl"], default="rl")
    parser.add_argument("--execution-mode", choices=["serial", "ray_pipeline", "data_parallel"], default="serial")

    # 阶段
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--epochs-per-trial", type=int, default=1)
    parser.add_argument("--coarse-trials", type=int, default=0)
    parser.add_argument("--coarse-epochs", type=int, default=0)
    parser.add_argument("--rerank-top-k", type=int, default=0)
    parser.add_argument("--rerank-epochs", type=int, default=0)

    # 控制器
    parser.add_argument("--controller-lr", type=float, default=1e-2)

    # 数据
    parser.add_argument("--dataset", choices=["synthetic", "wikipedia", "reddit", "public_csv"], default="synthetic")
    parser.add_argument("--dataset-dir", type=str, default="data/public")
    parser.add_argument("--local-data-path", type=str, default="")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-events", type=int, default=0)

    # 合成数据
    parser.add_argument("--num-users", type=int, default=500)
    parser.add_argument("--num-items", type=int, default=1000)
    parser.add_argument("--num-interactions", type=int, default=3000)

    # 训练
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg-sample-size", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--selection-metric", choices=["mrr", "recall_at_k"], default="mrr")
    parser.add_argument("--batch-training", action="store_true")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--batch-mode", type=str, default="tbatch", choices=["serial", "tbatch", "tgn"])
    parser.add_argument("--tgn-loss-mode", type=str, default="all", choices=["all", "last"])
    parser.add_argument("--tgn-window-size", type=float, default=10.0)
    parser.add_argument("--eval-frozen", type=str, default="false", choices=["true", "false"])

    # 设备
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")

    # 时间分区
    parser.add_argument("--partition-size", type=int, default=0)
    parser.add_argument("--partition-strategy", choices=["count"], default="count")
    parser.add_argument("--partition-overlap-ratio", type=float, default=0.0)

    # 流水线
    parser.add_argument("--architectures-per-step", type=int, default=2)
    parser.add_argument("--num-pipeline-stages", type=int, default=2)
    parser.add_argument("--pipeline-worker-gpus", type=float, default=0.0)
    parser.add_argument("--pipeline-worker-cpus", type=float, default=1.0)
    parser.add_argument("--pipeline-stage-train-workers", type=str, default="")
    parser.add_argument("--pipeline-stage-eval-workers", type=str, default="")
    parser.add_argument("--stage-balance-strategy", choices=["cost", "count"], default="cost")
    parser.add_argument("--stage-balance-user-weight", type=float, default=0.25)
    parser.add_argument("--stage-balance-item-weight", type=float, default=0.25)
    parser.add_argument("--stage-balance-span-weight", type=float, default=0.0)
    parser.add_argument("--pipeline-mode", type=str, default="naive", choices=["naive", "smart"])
    parser.add_argument("--pipeline-trace", action="store_true")

    # Ray 设置
    parser.add_argument("--ray-address", type=str, default="")

    # GPU 设置
    parser.add_argument("--gpu-list", type=str, default="0,1,2")
    parser.add_argument("--enable-auto-pipeline-config", action="store_true")

    # 数据并行
    parser.add_argument("--data-parallel-workers", type=int, default=3)
    parser.add_argument("--data-parallel-worker-gpus", type=float, default=1.0)
    parser.add_argument("--data-parallel-visible-gpus", type=str, default="0,1,2")
    parser.add_argument("--data-parallel-sync-level", choices=["partition", "tbatch", "micro_batch"], default="micro_batch")
    parser.add_argument("--data-parallel-micro-batch-size", type=int, default=0)

    # 评估种子
    parser.add_argument("--eval-seeds", type=str, default="")
    parser.add_argument("--family-balanced-rerank", action="store_true")
    parser.add_argument("--family-balance-per-model", type=int, default=1)

    # 效率监控
    parser.add_argument("--enable-efficiency-monitor", action="store_true")
    parser.add_argument("--efficiency-monitor-interval", type=int, default=10)

    # 时间预算
    parser.add_argument("--time-budget-sec", type=float, default=0.0)

    args = parser.parse_args()

    # 构建 dataclass
    return SearchConfig(
        space=args.space,
        search_mode=args.search_mode,
        execution_mode=args.execution_mode,
        coarse_trials=args.coarse_trials,
        coarse_epochs=args.coarse_epochs,
        trials=args.trials,
        epochs_per_trial=args.epochs_per_trial,
        rerank_top_k=args.rerank_top_k,
        rerank_epochs=args.rerank_epochs,
        controller_lr=args.controller_lr,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        local_data_path=args.local_data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_events=args.max_events,
        num_users=args.num_users,
        num_items=args.num_items,
        num_interactions=args.num_interactions,
        feature_dim=args.feature_dim,
        lr=args.lr,
        neg_sample_size=args.neg_sample_size,
        k=args.k,
        selection_metric=args.selection_metric,
        batch_training=args.batch_training,
        train_batch_size=args.train_batch_size,
        batch_mode=args.batch_mode,
        tgn_loss_mode=args.tgn_loss_mode,
        tgn_window_size=args.tgn_window_size,
        eval_frozen=args.eval_frozen == "true",
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        partition_size=args.partition_size,
        partition_strategy=args.partition_strategy,
        partition_overlap_ratio=args.partition_overlap_ratio,
        architectures_per_step=args.architectures_per_step,
        num_pipeline_stages=args.num_pipeline_stages,
        pipeline_worker_gpus=args.pipeline_worker_gpus,
        pipeline_worker_cpus=args.pipeline_worker_cpus,
        pipeline_stage_train_workers=args.pipeline_stage_train_workers,
        pipeline_stage_eval_workers=args.pipeline_stage_eval_workers,
        stage_balance_strategy=args.stage_balance_strategy,
        stage_balance_user_weight=args.stage_balance_user_weight,
        stage_balance_item_weight=args.stage_balance_item_weight,
        stage_balance_span_weight=args.stage_balance_span_weight,
        pipeline_mode=args.pipeline_mode,
        pipeline_trace=args.pipeline_trace,
        ray_address=args.ray_address,
        gpu_list=args.gpu_list,
        enable_auto_pipeline_config=args.enable_auto_pipeline_config,
        data_parallel_workers=args.data_parallel_workers,
        data_parallel_worker_gpus=args.data_parallel_worker_gpus,
        data_parallel_visible_gpus=args.data_parallel_visible_gpus,
        data_parallel_sync_level=args.data_parallel_sync_level,
        data_parallel_micro_batch_size=args.data_parallel_micro_batch_size if args.data_parallel_micro_batch_size > 0 else max(1, args.max_events // 100),
        eval_seeds_str=args.eval_seeds,
        family_balanced_rerank=args.family_balanced_rerank,
        family_balance_per_model=args.family_balance_per_model,
        enable_efficiency_monitor=args.enable_efficiency_monitor,
        efficiency_monitor_interval=args.efficiency_monitor_interval,
        time_budget_sec=args.time_budget_sec,
    )


# ---------------------------------------------------------------------------
# 结果保存
# ---------------------------------------------------------------------------

def save_results(best: dict, results: list, output_dir: str) -> None:
    """保存最佳架构和 leaderboard 到输出目录。"""
    os.makedirs(output_dir, exist_ok=True)

    best_path = os.path.join(output_dir, "best_arch.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "leaderboard.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "phase", "eval_split", "score", "val_score", "test_score",
            "mrr", "recall_at_k", "params", "time_sec", "model", "config_json"
        ])
        sorted_results = sorted(
            results,
            key=lambda x: (x["score"], -x["params"], -x["time_sec"]),
            reverse=True,
        )
        for idx, row in enumerate(sorted_results, start=1):
            writer.writerow([
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
            ])

    print(f"Best architecture saved to: {best_path}")
    print(f"Leaderboard saved to: {csv_path}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args()

    # 搜索空间
    search_space = get_search_space(cfg.space)

    # 控制器
    if cfg.search_mode == "rl":
        controller = RLGraphNASController(search_space, seed=cfg.seed, lr=cfg.controller_lr)
    else:
        controller = RandomGraphNASController(search_space, seed=cfg.seed)

    # 流水线追踪日志
    pipeline_trace_log_path = ""
    if cfg.execution_mode == "ray_pipeline":
        os.makedirs(cfg.output_dir, exist_ok=True)
        run_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        pipeline_trace_log_path = os.path.join(cfg.output_dir, f"pipeline_trace_{run_tag}.log")
        with open(pipeline_trace_log_path, "w", encoding="utf-8") as f:
            f.write(f"# pipeline trace log\n")
            f.write(f"# created_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        print(f"Pipeline trace log file: {pipeline_trace_log_path}")

    # 设备
    if cfg.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device

    # 构建 trainer 配置
    base_config = {
        "dataset": cfg.dataset,
        "dataset_dir": cfg.dataset_dir,
        "local_data_path": cfg.local_data_path,
        "train_ratio": cfg.train_ratio,
        "val_ratio": cfg.val_ratio,
        "max_events": cfg.max_events,
        "num_users": cfg.num_users,
        "num_items": cfg.num_items,
        "num_interactions": cfg.num_interactions,
        "feature_dim": cfg.feature_dim,
        "lr": cfg.lr,
        "neg_sample_size": cfg.neg_sample_size,
        "k": cfg.k,
        "selection_metric": cfg.selection_metric,
        "device": device,
        "seed": cfg.seed,
        "partition_size": cfg.partition_size,
        "partition_strategy": cfg.partition_strategy,
        "num_pipeline_stages": cfg.num_pipeline_stages,
        "pipeline_worker_gpus": cfg.pipeline_worker_gpus,
        "pipeline_worker_cpus": cfg.pipeline_worker_cpus,
        "pipeline_stage_train_workers": cfg.pipeline_stage_train_workers,
        "pipeline_stage_eval_workers": cfg.pipeline_stage_eval_workers,
        "stage_balance_strategy": cfg.stage_balance_strategy,
        "stage_balance_user_weight": cfg.stage_balance_user_weight,
        "stage_balance_item_weight": cfg.stage_balance_item_weight,
        "stage_balance_span_weight": cfg.stage_balance_span_weight,
        "ray_address": cfg.ray_address,
        "pipeline_trace": cfg.pipeline_trace,
        "pipeline_trace_log_path": pipeline_trace_log_path,
        "output_dir": cfg.output_dir,
        "enable_efficiency_monitor": cfg.enable_efficiency_monitor,
        "efficiency_monitor_interval": cfg.efficiency_monitor_interval,
        "data_parallel_workers": cfg.data_parallel_workers,
        "data_parallel_worker_gpus": cfg.data_parallel_worker_gpus,
        "data_parallel_visible_gpus": cfg.data_parallel_visible_gpus,
        "data_parallel_sync_level": cfg.data_parallel_sync_level,
        "data_parallel_micro_batch_size": cfg.data_parallel_micro_batch_size,
        "gpu_list": cfg.gpu_list,
        "enable_auto_pipeline_config": cfg.enable_auto_pipeline_config,
        "pipeline_mode": cfg.pipeline_mode,
        "batch_training": cfg.batch_training,
        "train_batch_size": cfg.train_batch_size,
        "batch_mode": cfg.batch_mode,
        "tgn_loss_mode": cfg.tgn_loss_mode,
        "tgn_window_size": cfg.tgn_window_size,
        "eval_frozen": cfg.eval_frozen,
    }

    trainer = GraphNASTrainer(base_config)

    coarse_trials = cfg.coarse_trials if cfg.coarse_trials > 0 else cfg.trials
    coarse_epochs = cfg.coarse_epochs if cfg.coarse_epochs > 0 else cfg.epochs_per_trial
    rerank_epochs = cfg.rerank_epochs if cfg.rerank_epochs > 0 else coarse_epochs

    # 执行搜索
    if cfg.execution_mode == "ray_pipeline":
        best, results = trainer.search_pipeline(
            controller=controller,
            coarse_trials=coarse_trials,
            architectures_per_step=cfg.architectures_per_step,
            coarse_epochs=coarse_epochs,
            rerank_top_k=cfg.rerank_top_k,
            rerank_epochs=rerank_epochs,
            family_balanced_rerank=cfg.family_balanced_rerank,
            family_balance_per_model=cfg.family_balance_per_model,
            time_budget_sec=cfg.time_budget_sec,
        )
    elif cfg.execution_mode == "data_parallel":
        best, results = trainer.search_data_parallel(
            controller=controller,
            coarse_trials=coarse_trials,
            coarse_epochs=coarse_epochs,
            num_workers=cfg.data_parallel_workers,
            rerank_top_k=cfg.rerank_top_k,
            rerank_epochs=rerank_epochs,
            time_budget_sec=cfg.time_budget_sec,
        )
    else:
        best, results = trainer.search(
            controller=controller,
            coarse_trials=coarse_trials,
            coarse_epochs=coarse_epochs,
            rerank_top_k=cfg.rerank_top_k,
            rerank_epochs=rerank_epochs,
            eval_seeds=cfg.eval_seeds,
            family_balanced_rerank=cfg.family_balanced_rerank,
            family_balance_per_model=cfg.family_balance_per_model,
            time_budget_sec=cfg.time_budget_sec,
        )

    save_results(best, results, cfg.output_dir)
    print(f"Search mode: {cfg.search_mode}")
    print(f"Best selection score: {best.get('selected_val_score', best['score']):.4f}")
    print(f"Best test score: {best['score']:.4f}")
    print(f"Best model family: {best['config'].get('model', 'temporal_event_gnn_jodie')}")


if __name__ == "__main__":
    main()
