"""
训练单个架构并评估。
"""

import argparse
import json
import time
from pathlib import Path
import torch
import numpy as np

from data.synthetic import generate_synthetic_data
from data.temporal_partition import build_temporal_partitions
from models.factory import build_model
from models.training import train_model, train_model_ce, evaluate_ranking_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--memory-cell", type=str, default="rnn")
    parser.add_argument("--time-proj", type=str, default="off")
    parser.add_argument("--use-static-embeddings", type=str, default="off")
    parser.add_argument("--normalize-state", type=str, default="off")
    parser.add_argument("--partition-size", type=int, default=0)
    parser.add_argument("--event-agg", type=str, default="none")
    parser.add_argument("--max-neighbors", type=int, default=0)
    parser.add_argument("--batch-mode", type=str, default="tbatch")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="public_csv")
    parser.add_argument("--local-data-path", type=str, required=True)
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-frozen", type=str, default="false", choices=["true", "false"], help="Evaluation mode: true=offline (frozen embeddings), false=online (update embeddings).")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建配置
    config = {
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "memory_cell": args.memory_cell,
        "time_proj": args.time_proj,
        "use_static_embeddings": args.use_static_embeddings,
        "normalize_state": args.normalize_state,
        "event_agg": args.event_agg,
        "max_neighbors": args.max_neighbors,
        "batch_mode": args.batch_mode,
        "train_batch_size": args.train_batch_size,
        "dataset": args.dataset,
        "local_data_path": args.local_data_path,
        "max_events": args.max_events,
        "seed": args.seed,
        "lr": 0.001,
        "neg_sample_size": 5,
        "k": 10,
        "device": "cuda",
    }

    print(f"训练配置: {json.dumps(config, indent=2)}")

    # 加载数据
    print("\n加载数据...")
    from data.public_dataset import load_public_dataset

    all_interactions, num_users, num_items = load_public_dataset(
        dataset_name=args.dataset,
        dataset_dir="data/public",
        feature_dim=8,
        max_events=args.max_events,
        local_data_path=args.local_data_path,
    )

    # 按时间排序并分割
    all_interactions.sort(key=lambda x: x.timestamp)

    train_ratio = 0.7
    val_ratio = 0.1

    n_train = int(len(all_interactions) * train_ratio)
    n_val = int(len(all_interactions) * val_ratio)

    train_data = all_interactions[:n_train]
    val_data = all_interactions[n_train:n_train + n_val]
    test_data = all_interactions[n_train + n_val:]

    final_train_data = train_data + val_data

    config["num_users"] = num_users
    config["num_items"] = num_items
    config["feature_dim"] = 8

    print(f"数据集: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    print(f"最终训练: {len(final_train_data)} (train+val), 测试: {len(test_data)}")
    print(f"用户数: {config['num_users']}, 物品数: {config['num_items']}")

    # 设置种子（必须在构建模型前！）
    print(f"\n设置种子: {config['seed']}")
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # 构建模型
    print("构建模型...")
    model = build_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters())}")

    # 训练
    print(f"\n开始训练 ({args.epochs} epochs)...")
    start_time = time.time()

    # 根据partition_size参数决定是否使用partition
    if args.partition_size > 0:
        partition_plan = build_temporal_partitions(
            final_train_data, split="train", partition_size=args.partition_size
        )
    else:
        partition_plan = None

    # 使用train_model_ce (与NAS一致)
    train_model_ce(
        model=model,
        interactions=final_train_data,
        num_epochs=args.epochs,
        lr=config["lr"],
        graph_ctx=None,
        seed=config["seed"],
        partitions=partition_plan,
        batch_training=False,
        batch_size=config["train_batch_size"],
        batch_mode=config["batch_mode"],
        tgn_loss_mode=config.get("tgn_loss_mode", "all"),
        tgn_window_size=config.get("tgn_window_size", 10.0),
    )

    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f}s")

    # 评估
    print("\n评估...")
    metrics = evaluate_ranking_metrics(
        model, test_data, k=config["k"], graph_ctx=None,
        frozen=args.eval_frozen == "true"
    )

    result = {
        "config": config,
        "test_mrr": float(metrics["mrr"]),
        "test_recall_at_k": float(metrics["recall_at_k"]),
        "time_sec": train_time,
    }

    # 保存结果
    result_file = output_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n结果:")
    print(f"  Test MRR: {result['test_mrr']:.4f}")
    print(f"  Test Recall@{config['k']}: {result['test_recall_at_k']:.4f}")
    print(f"  训练时间: {result['time_sec']:.2f}s")
    print(f"\n结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
