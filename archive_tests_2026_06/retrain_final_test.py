#!/usr/bin/env python3
"""
Final Test Retrain Script

使用与search.py相同的final test逻辑，对指定架构进行标准化重训练。
确保所有策略的retrain结果可比较。

核心逻辑来自 nas/trainer.py:930-954
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from nas.trainer import GraphNASTrainer


def main():
    parser = argparse.ArgumentParser(description="使用final test逻辑重训练架构")
    parser.add_argument("--best-arch-json", required=True, help="best_arch.json路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--dataset", default="public_csv")
    parser.add_argument("--local-data-path", default="data/public/mooc.csv")
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument("--seed", type=int, help="基础seed（将+20000作为final seed）")
    parser.add_argument("--epochs", type=int, default=3, help="训练epochs")

    args = parser.parse_args()

    # 读取best_arch.json
    with open(args.best_arch_json) as f:
        best_arch_data = json.load(f)

    arch_config = best_arch_data["config"]

    # 提取关键配置
    base_seed = args.seed if args.seed else arch_config.get("seed", 42)
    final_seed = base_seed + 20000  # 与search.py保持一致

    # 构建base_config（从arch_config中提取必需字段）
    base_config = {
        "dataset": args.dataset,
        "local_data_path": args.local_data_path,
        "max_events": args.max_events,
        "seed": base_seed,
        "train_ratio": arch_config.get("train_ratio", 0.7),
        "val_ratio": arch_config.get("val_ratio", 0.1),
        "lr": arch_config.get("lr", 0.001),
        "neg_sample_size": arch_config.get("neg_sample_size", 5),
        "k": arch_config.get("k", 10),
        "device": arch_config.get("device", "cuda"),
        "selection_metric": arch_config.get("selection_metric", "mrr"),
        "batch_mode": arch_config.get("batch_mode", "tbatch"),
        "num_users": arch_config.get("num_users", 1435),
        "num_items": arch_config.get("num_items", 21),
        "feature_dim": arch_config.get("feature_dim", 8),
        "embedding_dim": arch_config.get("embedding_dim", 128),
    }

    print("=" * 70)
    print("Final Test Retrain")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  架构: {arch_config.get('model')} / {arch_config.get('time_proj')} / {arch_config.get('use_static_embeddings')}")
    print(f"  Base seed: {base_seed}")
    print(f"  Final seed: {final_seed} (base_seed + 20000)")
    print(f"  Epochs: {args.epochs}")
    print(f"  输出: {args.output_dir}")
    print()

    # 创建trainer
    trainer = GraphNASTrainer(base_config=base_config)

    # 准备数据
    train_data, val_data, test_data, user_type_prefs, item_type, graph_template, partition_plan = trainer._prepare_data()

    # Final test逻辑（与nas/trainer.py:930-954完全一致）
    final_train_data = train_data + val_data

    print(f"[Final Test] Evaluating architecture on test set using Serial training")
    print(f"  Training data: train+val ({len(final_train_data)} events)")
    print(f"  Test data: {len(test_data)} events")
    print(f"  Epochs: {args.epochs}")
    print(f"  Seed: {final_seed}")
    print()

    # 调用相同的评估方法
    final_test_result = trainer._evaluate_arch_multi_seed(
        arch_config=arch_config,
        train_data=final_train_data,
        eval_data=test_data,
        user_type_prefs=user_type_prefs,
        item_type=item_type,
        graph_template=graph_template,
        epochs=args.epochs,
        eval_seeds=None,
        default_seed=final_seed,
        phase="final",
        eval_split="test",
    )

    # 构建结果
    result = {
        "config": arch_config,
        "phase": "final",
        "eval_split": "test",
        "seed": final_seed,
        "base_seed": base_seed,
        "test_mrr": final_test_result["mrr"],
        "test_recall_at_k": final_test_result["recall_at_k"],
        "test_score": final_test_result["score"],
        "time_sec": final_test_result["time_sec"],
        "per_seed_metrics": final_test_result.get("per_seed_metrics", []),
    }

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "best_arch.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 70)
    print("Retrain完成")
    print("=" * 70)
    print(f"\n结果:")
    print(f"  Test MRR: {result['test_mrr']:.6f}")
    print(f"  Test Recall@10: {result['test_recall_at_k']:.4f}")
    print(f"  Time: {result['time_sec']:.2f}s")
    print(f"\n保存到: {output_file}")


if __name__ == "__main__":
    main()
