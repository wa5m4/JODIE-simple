#!/usr/bin/env python3
"""
步骤2: 在统一条件下重新训练四组架构
使用 serial + tbatch 模式，不进行架构搜索
"""
import json
import torch
import time
from pathlib import Path
from data.public_dataset import load_public_dataset
from models.factory import build_model
from models.training import train_model, evaluate_ranking_metrics
from data.temporal_partition import build_temporal_partitions

def train_fixed_architecture(config, dataset_name, local_data_path, max_events, epochs, device, partition_size):
    """Train a single fixed architecture"""
    print(f"  Loading data...")

    # Load dataset
    interactions, num_users, num_items = load_public_dataset(
        dataset_name=dataset_name,
        dataset_dir="data/public",
        feature_dim=config.get("feature_dim", 8),
        max_events=max_events,
        local_data_path=local_data_path
    )

    # Split data
    train_ratio = 0.7
    val_ratio = 0.15
    train_end = int(len(interactions) * train_ratio)
    val_end = int(len(interactions) * (train_ratio + val_ratio))

    train_data = interactions[:train_end]
    val_data = interactions[train_end:val_end]
    test_data = interactions[val_end:]

    print(f"  Data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    print(f"  Users: {num_users}, Items: {num_items}")

    # Create partitions
    train_partitions = build_temporal_partitions(train_data, split="train", partition_size=partition_size) if partition_size > 0 else None
    val_partitions = build_temporal_partitions(val_data, split="val", partition_size=partition_size) if partition_size > 0 else None
    test_partitions = build_temporal_partitions(test_data, split="test", partition_size=partition_size) if partition_size > 0 else None

    # Build model
    model_config = config.copy()
    model_config["num_users"] = num_users
    model_config["num_items"] = num_items
    model = build_model(model_config).to(device)

    # Train
    print(f"  Training for {epochs} epochs...")
    start_time = time.time()

    train_model(
        model=model,
        interactions=train_data,
        num_epochs=epochs,
        partitions=train_partitions,
        batch_mode="tbatch",
        batch_size=config.get("train_batch_size", 32)
    )

    train_time = time.time() - start_time

    # Evaluate
    print(f"  Evaluating...")
    val_metrics = evaluate_ranking_metrics(model, val_data, k=10, partitions=val_partitions)
    test_metrics = evaluate_ranking_metrics(model, test_data, k=10, partitions=test_partitions)

    return {
        "val_mrr": val_metrics["mrr"],
        "val_recall": val_metrics["recall_at_k"],
        "test_mrr": test_metrics["mrr"],
        "test_recall": test_metrics["recall_at_k"],
        "train_time": train_time
    }

def main():
    # Load architectures
    with open("arch_reeval_configs.json") as f:
        architectures = json.load(f)

    # Training parameters (same as original experiments)
    dataset_name = "public_csv"
    local_data_path = "data/public/mooc.csv"
    max_events = 20000
    epochs = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Fixed training parameters
    partition_size = 1000
    train_batch_size = 32

    print("=" * 120)
    print("步骤2: 统一条件下重新训练四组架构")
    print("=" * 120)
    print(f"模式: Serial + T-Batch")
    print(f"数据集: {dataset_name}")
    print(f"Max Events: {max_events}")
    print(f"Epochs: {epochs}")
    print(f"Partition Size: {partition_size}")
    print(f"Batch Size: {train_batch_size}")
    print(f"Device: {device}")
    print("=" * 120)
    print()

    results = {}

    for name in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]:
        print(f"[{name}] 开始训练...")
        arch = architectures[name]

        # Build config
        config = arch["params"].copy()
        config["train_batch_size"] = train_batch_size

        try:
            result = train_fixed_architecture(
                config=config,
                dataset_name=dataset_name,
                local_data_path=local_data_path,
                max_events=max_events,
                epochs=epochs,
                device=device,
                partition_size=partition_size
            )

            results[name] = {
                "retrained": result,
                "original": {
                    "val_mrr": arch["original_val_mrr"],
                    "test_mrr": arch["original_test_mrr"],
                    "test_recall": arch["original_test_recall"]
                }
            }

            print(f"[{name}] 完成!")
            print(f"  Test MRR: {result['test_mrr']:.4f}")
            print(f"  Test Recall@10: {result['test_recall']:.4f}")
            print(f"  训练时间: {result['train_time']:.1f}秒")
            print()

        except Exception as e:
            print(f"[{name}] 训练失败: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Save results
    output_file = "arch_reeval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 120)
    print("步骤3: 对比结果")
    print("=" * 120)
    print()

    # Comparison table
    print(f"{'搜索方法':<20} | {'原始Test MRR':<15} | {'重训Test MRR':<15} | {'差异':<12} | {'重训时间(秒)':<15}")
    print("-" * 120)

    for name in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]:
        if name in results:
            orig_mrr = results[name]["original"]["test_mrr"]
            new_mrr = results[name]["retrained"]["test_mrr"]
            diff = new_mrr - orig_mrr
            train_time = results[name]["retrained"]["train_time"]
            print(f"{name:<20} | {orig_mrr:<15.4f} | {new_mrr:<15.4f} | {diff:+<12.4f} | {train_time:<15.1f}")

    print()
    print("=" * 120)
    print("步骤4: 分析")
    print("=" * 120)
    print()

    # Find best retrained architecture
    best_name = max(results.keys(), key=lambda x: results[x]["retrained"]["test_mrr"])
    best_mrr = results[best_name]["retrained"]["test_mrr"]

    print(f"✅ 在统一条件下，最佳架构来自: {best_name}")
    print(f"   Test MRR: {best_mrr:.4f}")
    print()

    # Compare serial vs pipeline_smart
    if "serial" in results and "pipeline_smart" in results:
        serial_mrr = results["serial"]["retrained"]["test_mrr"]
        smart_mrr = results["pipeline_smart"]["retrained"]["test_mrr"]
        gap = serial_mrr - smart_mrr
        gap_pct = gap / serial_mrr * 100

        print(f"Serial vs Pipeline Smart 差距:")
        print(f"  Serial: {serial_mrr:.4f}")
        print(f"  Pipeline Smart: {smart_mrr:.4f}")
        print(f"  差距: {gap:.4f} ({gap_pct:.1f}%)")
        print()

        # Compare with original gap
        orig_serial = results["serial"]["original"]["test_mrr"]
        orig_smart = results["pipeline_smart"]["original"]["test_mrr"]
        orig_gap = orig_serial - orig_smart
        orig_gap_pct = orig_gap / orig_serial * 100

        print(f"与原始差距对比:")
        print(f"  原始差距: {orig_gap:.4f} ({orig_gap_pct:.1f}%)")
        print(f"  重训差距: {gap:.4f} ({gap_pct:.1f}%)")

        if abs(gap) < abs(orig_gap):
            print(f"  ✅ 差距缩小了 {abs(orig_gap - gap):.4f}")
        else:
            print(f"  ⚠️ 差距扩大了 {abs(gap - orig_gap):.4f}")

    print()
    print(f"✅ 结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
