#!/usr/bin/env python3
"""快速验证方案 C：Pipeline 训练 + 全数据评估，单架构测试"""

import os
import sys
import time
import json
import torch
import numpy as np

# 设置环境
os.environ["RAY_DISABLE_IMPORT_HOOK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 用空闲的 GPU2

from jodie.data.public_dataset import load_public_dataset
from jodie.data.temporal_partition import build_partition_plan
from jodie.models.factory import build_model
from jodie.nas.ray_pipeline import RayPipelineExecutor
from jodie.training.metrics import evaluate_ranking_metrics


def main():
    # 基本配置（匹配 run_all.py）
    BASE_CONFIG = {
        "dataset": "public_csv",
        "dataset_dir": "data/public",
        "local_data_path": "data/public/mooc.csv",
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "max_events": 20000,
        "feature_dim": 4,
        "lr": 0.001,
        "neg_sample_size": 5,
        "k": 10,
        "selection_metric": "mrr",
        "device": "cuda",
        "seed": 42,
        "partition_size": 500,
        "partition_strategy": "count",
        "partition_overlap_ratio": 0.0,
        "num_pipeline_stages": 1,  # 简化：单阶段测试
        "pipeline_worker_gpus": 1.0,
        "pipeline_worker_cpus": 1.0,
        "pipeline_stage_train_workers": "1",  # 单worker
        "pipeline_stage_eval_workers": "1",
        "stage_balance_strategy": "cost",
        "stage_balance_user_weight": 0.25,
        "stage_balance_item_weight": 0.25,
        "stage_balance_span_weight": 0.0,
        "pipeline_mode": "naive",
        "gpu_list": "2",
        "batch_mode": "tbatch",
    }

    # 测试两个架构：Serial最优 (all off) vs Pipeline最优 (all on)
    arch_all_off = {
        "model": "jodie_rnn",
        "embedding_dim": 128,
        "memory_cell": "rnn",
        "time_proj": "off",
        "use_static_embeddings": "off",
        "normalize_state": "off",
    }
    arch_all_on = {
        "model": "jodie_rnn",
        "embedding_dim": 128,
        "memory_cell": "rnn",
        "time_proj": "linear",
        "use_static_embeddings": "on",
        "normalize_state": "on",
    }

    print("=" * 60)
    print("方案 C 验证测试")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    interactions, num_users, num_items = load_public_dataset(
        dataset_name=BASE_CONFIG["dataset"],
        dataset_dir=BASE_CONFIG.get("dataset_dir", "data/public"),
        feature_dim=BASE_CONFIG["feature_dim"],
        max_events=BASE_CONFIG.get("max_events", 0),
        local_data_path=BASE_CONFIG.get("local_data_path", ""),
    )
    BASE_CONFIG["num_users"] = num_users
    BASE_CONFIG["num_items"] = num_items

    # 按时间排序后切分（与 trainer._prepare_data 一致）
    interactions = sorted(interactions, key=lambda x: x.timestamp)
    total_events = len(interactions)
    train_ratio = float(BASE_CONFIG.get("train_ratio", 0.7))
    val_ratio = float(BASE_CONFIG.get("val_ratio", 0.1))
    train_end = int(total_events * train_ratio)
    val_end = int(total_events * (train_ratio + val_ratio))
    train_end = max(1, min(train_end, total_events - 2))
    val_end = max(train_end + 1, min(val_end, total_events - 1))

    train_data = interactions[:train_end]
    val_data = interactions[train_end:val_end]
    test_data = interactions[val_end:]

    item_type = np.zeros(num_items, dtype=np.int64)
    user_type_prefs = {uid: {0} for uid in range(num_users)}
    graph_template = None

    print(f"  Train interactions: {len(train_data)}")
    print(f"  Val interactions: {len(val_data)}")
    print(f"  Test interactions: {len(test_data)}")

    # 构建分区计划
    print("\n[2/4] 构建分区计划...")
    partition_plan = build_partition_plan(
        train_data,
        val_data,
        test_data,
        partition_size=BASE_CONFIG["partition_size"],
        strategy=BASE_CONFIG["partition_strategy"],
        overlap_ratio=BASE_CONFIG["partition_overlap_ratio"],
    )
    train_partitions = partition_plan.get_split_partitions("train")
    print(f"  Train partitions: {len(train_partitions)}")

    # 创建 executor
    executor = RayPipelineExecutor(BASE_CONFIG, partition_plan)

    # 测试两个架构
    arch_configs = [arch_all_off, arch_all_on]

    print("\n[3/4] Pipeline 训练 + 全数据评估...")
    start = time.time()

    trained_payloads = executor.run_train_only(arch_configs, num_train_epochs=2)

    print(f"\n  训练完成，耗时 {time.time() - start:.1f}s")

    print("\n[4/4] 全数据评估结果:")
    print("-" * 60)

    for i, payload in enumerate(trained_payloads):
        config = dict(BASE_CONFIG)
        config.update(payload.arch_config)
        model = build_model(config)
        model = model.to("cuda")

        # 加载训练后的模型
        model.load_state_dict(payload.model_state_dict)
        if payload.runtime_state is not None:
            model.import_runtime_state(payload.runtime_state)

        # 全数据验证集评估
        val_metrics = evaluate_ranking_metrics(
            model, val_data, k=10, partitions=None  # partitions=None = 全数据
        )
        # 全数据测试集评估
        test_metrics = evaluate_ranking_metrics(
            model, test_data, k=10, partitions=None
        )

        arch_desc = (
            "all OFF (Serial最优)"
            if payload.arch_config.get("time_proj") == "off"
            else "all ON (Pipeline旧最优)"
        )
        params = sum(p.numel() for p in model.parameters())

        print(f"\n  架构 #{i}: {arch_desc}")
        print(f"    参数量: {params:,}")
        print(f"    val_mrr: {val_metrics['mrr']:.4f}  |  val_recall@10: {val_metrics['recall_at_k']:.4f}")
        print(f"    test_mrr: {test_metrics['mrr']:.4f}  |  test_recall@10: {test_metrics['recall_at_k']:.4f}")

    executor.shutdown()
    import ray
    if ray.is_initialized():
        ray.shutdown()
    print("\n✅ 方案 C 验证完成")


if __name__ == "__main__":
    main()
