#!/usr/bin/env python3
"""
对比冻结vs非冻结评估模式在MOOC数据集上的性能
量化测试时embedding更新的影响
"""
import argparse
import json
import time
from pathlib import Path

import torch

from data.public_dataset import load_public_dataset
from models.hybrid_jodie import build_model
from models.training import train_model, evaluate_ranking_metrics
from data.temporal_partition import build_temporal_partitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="jodie_rnn")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--time-proj", type=str, default="linear")
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/frozen_comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("冻结 vs 非冻结评估模式对比")
    print("="*80)
    print(f"模型: {args.model}, {args.embedding_dim}-dim, time_proj={args.time_proj}")
    print(f"数据: MOOC, max_events={args.max_events}")
    print(f"训练: {args.epochs} epochs, seed={args.seed}")
    print()

    # 加载数据
    print("加载数据...")
    all_interactions, num_users, num_items = load_public_dataset(
        dataset_name="public_csv",
        dataset_dir="data/public",
        feature_dim=8,
        max_events=args.max_events,
        local_data_path="data/public/mooc.csv",
    )

    all_interactions.sort(key=lambda x: x.timestamp)
    n_train = int(len(all_interactions) * 0.7)
    n_val = int(len(all_interactions) * 0.1)

    train_data = all_interactions[:n_train]
    val_data = all_interactions[n_train:n_train + n_val]
    test_data = all_interactions[n_train + n_val:]
    final_train_data = train_data + val_data

    print(f"数据集: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    print(f"最终训练: {len(final_train_data)} (train+val)")
    print(f"用户数: {num_users}, 物品数: {num_items}")
    print()

    # 构建模型
    print("构建模型...")
    config = {
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "time_proj": args.time_proj,
        "num_users": num_users,
        "num_items": num_items,
        "feature_dim": 8,
        "lr": 0.001,
        "neg_sample_size": 5,
        "k": 10,
        "device": "cuda",
        "seed": args.seed,
    }
    model = build_model(config)
    print(f"模型参数: {sum(p.numel() for p in model.parameters())}")
    print()

    # 训练
    print(f"训练 ({args.epochs} epochs)...")
    torch.manual_seed(args.seed)
    partition_plan = build_temporal_partitions(
        final_train_data, split="train", partition_size=1000
    )

    start_time = time.time()
    train_model(
        model=model,
        interactions=final_train_data,
        num_epochs=args.epochs,
        lr=config["lr"],
        neg_sample_size=config["neg_sample_size"],
        seed=config["seed"],
        partitions=partition_plan,
        batch_size=32,
        batch_mode="tbatch",
    )
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f}s")
    print()

    # 评估：非冻结模式
    print("="*80)
    print("评估1: 非冻结模式 (frozen=False)")
    print("="*80)
    print("允许测试时embedding更新（当前默认行为）")
    print()

    start_time = time.time()
    metrics_unfrozen = evaluate_ranking_metrics(
        model, test_data, k=10, frozen=False
    )
    eval_time_unfrozen = time.time() - start_time

    print(f"结果:")
    print(f"  Test MRR: {metrics_unfrozen['mrr']:.4f}")
    print(f"  Test Recall@10: {metrics_unfrozen['recall_at_k']:.4f}")
    print(f"  评估时间: {eval_time_unfrozen:.2f}s")
    print()

    # 评估：冻结模式
    print("="*80)
    print("评估2: 冻结模式 (frozen=True)")
    print("="*80)
    print("阻止测试时embedding更新（标准离线评估）")
    print()

    start_time = time.time()
    metrics_frozen = evaluate_ranking_metrics(
        model, test_data, k=10, frozen=True
    )
    eval_time_frozen = time.time() - start_time

    print(f"结果:")
    print(f"  Test MRR: {metrics_frozen['mrr']:.4f}")
    print(f"  Test Recall@10: {metrics_frozen['recall_at_k']:.4f}")
    print(f"  评估时间: {eval_time_frozen:.2f}s")
    print()

    # 对比分析
    print("="*80)
    print("对比分析")
    print("="*80)

    mrr_diff = metrics_unfrozen['mrr'] - metrics_frozen['mrr']
    mrr_pct = (mrr_diff / metrics_frozen['mrr'] * 100) if metrics_frozen['mrr'] > 0 else 0
    recall_diff = metrics_unfrozen['recall_at_k'] - metrics_frozen['recall_at_k']
    recall_pct = (recall_diff / metrics_frozen['recall_at_k'] * 100) if metrics_frozen['recall_at_k'] > 0 else 0

    print(f"MRR:")
    print(f"  非冻结: {metrics_unfrozen['mrr']:.4f}")
    print(f"  冻结:   {metrics_frozen['mrr']:.4f}")
    print(f"  差异:   {mrr_diff:+.4f} ({mrr_pct:+.1f}%)")
    print()
    print(f"Recall@10:")
    print(f"  非冻结: {metrics_unfrozen['recall_at_k']:.4f}")
    print(f"  冻结:   {metrics_frozen['recall_at_k']:.4f}")
    print(f"  差异:   {recall_diff:+.4f} ({recall_pct:+.1f}%)")
    print()

    if mrr_diff > 0.01 or recall_diff > 0.01:
        print("⚠️  测试时embedding更新显著提升了性能")
        print("    这表明存在测试时信息泄露")
    else:
        print("✅ 两种模式性能接近，测试时更新影响较小")
    print()

    # 保存结果
    results = {
        "config": config,
        "train_time": train_time,
        "unfrozen": {
            "mrr": float(metrics_unfrozen['mrr']),
            "recall_at_k": float(metrics_unfrozen['recall_at_k']),
            "eval_time": eval_time_unfrozen,
        },
        "frozen": {
            "mrr": float(metrics_frozen['mrr']),
            "recall_at_k": float(metrics_frozen['recall_at_k']),
            "eval_time": eval_time_frozen,
        },
        "difference": {
            "mrr": float(mrr_diff),
            "mrr_pct": float(mrr_pct),
            "recall_at_k": float(recall_diff),
            "recall_at_k_pct": float(recall_pct),
        }
    }

    result_file = output_dir / "comparison_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
