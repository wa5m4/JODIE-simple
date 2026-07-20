"""
训练单个架构并评估。

用法:
    python train.py --model jodie_rnn --embedding-dim 32 \
        --local-data-path data/public/mooc.csv --output-dir outputs/my_run
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from jodie.data.public_dataset import load_public_dataset
from jodie.data.temporal_partition import build_temporal_partitions
from jodie.models.factory import build_model
from jodie.training.loops import train_model_ce
from jodie.training.metrics import evaluate_ranking_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single architecture and evaluate.")
    parser.add_argument("--model", type=str, required=True, help="Model type (e.g. jodie_rnn, temporal_event_gnn_jodie).")
    parser.add_argument("--embedding-dim", type=int, required=True, help="Embedding dimension.")
    parser.add_argument("--memory-cell", type=str, default="rnn", help="Memory cell type (rnn, gru, lstm).")
    parser.add_argument("--time-proj", type=str, default="off", help="Time projection: linear, mlp, off.")
    parser.add_argument("--use-static-embeddings", type=str, default="off", help="Use static embeddings: on, off.")
    parser.add_argument("--normalize-state", type=str, default="off", help="Normalize state: on, off.")
    parser.add_argument("--partition-size", type=int, default=0, help="Partition size (0 = no partitioning).")
    parser.add_argument("--event-agg", type=str, default="none", help="Event aggregation method.")
    parser.add_argument("--max-neighbors", type=int, default=0, help="Max neighbors for GNN.")
    parser.add_argument("--batch-mode", type=str, default="tbatch", choices=["serial", "tbatch", "tgn"])
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="public_csv")
    parser.add_argument("--local-data-path", type=str, required=True, help="Path to local CSV dataset.")
    parser.add_argument("--max-events", type=int, default=20000, help="Max events to use.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-frozen", type=str, default="false", choices=["true", "false"],
                        help="Evaluation mode: true=offline, false=online.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    print(f"Training config: {json.dumps(config, indent=2)}")

    # 加载数据
    print("\nLoading data...")
    all_interactions, num_users, num_items = load_public_dataset(
        dataset_name=args.dataset,
        dataset_dir="data/public",
        feature_dim=8,
        max_events=args.max_events,
        local_data_path=args.local_data_path,
    )

    # 按时序分割
    all_interactions.sort(key=lambda x: x.timestamp)
    train_ratio, val_ratio = 0.7, 0.1
    n_train = int(len(all_interactions) * train_ratio)
    n_val = int(len(all_interactions) * val_ratio)

    train_data = all_interactions[:n_train]
    val_data = all_interactions[n_train:n_train + n_val]
    test_data = all_interactions[n_train + n_val:]
    final_train_data = train_data + val_data

    config["num_users"] = num_users
    config["num_items"] = num_items
    config["feature_dim"] = 8

    print(f"Dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    print(f"Final train: {len(final_train_data)} (train+val), test: {len(test_data)}")
    print(f"Users: {config['num_users']}, Items: {config['num_items']}")

    # 种子
    print(f"\nSetting seed: {config['seed']}")
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # 构建模型
    print("Building model...")
    model = build_model(config)
    device = torch.device(config["device"])
    model = model.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # 训练
    print(f"\nTraining ({args.epochs} epochs)...")
    start_time = time.time()

    partitions = None
    if args.partition_size > 0:
        partitions = build_temporal_partitions(
            final_train_data, split="train", partition_size=args.partition_size
        )

    train_model_ce(
        model=model,
        interactions=final_train_data,
        num_epochs=args.epochs,
        lr=config["lr"],
        graph_ctx=None,
        seed=config["seed"],
        partitions=partitions,
        batch_training=False,
        batch_size=config["train_batch_size"],
        batch_mode=config["batch_mode"],
        tgn_loss_mode=config.get("tgn_loss_mode", "all"),
        tgn_window_size=config.get("tgn_window_size", 10.0),
    )

    train_time = time.time() - start_time
    print(f"Training done: {train_time:.2f}s")

    # 评估
    print("\nEvaluating...")
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

    result_file = output_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults:")
    print(f"  Test MRR: {result['test_mrr']:.4f}")
    print(f"  Test Recall@{config['k']}: {result['test_recall_at_k']:.4f}")
    print(f"  Train time: {result['time_sec']:.2f}s")
    print(f"  Saved to: {result_file}")


if __name__ == "__main__":
    main()
