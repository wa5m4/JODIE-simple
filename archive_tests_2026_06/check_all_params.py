#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

# Find all best_arch.json files in outputs/
outputs_dir = Path("outputs")
all_experiments = []

for exp_dir in sorted(outputs_dir.rglob("best_arch.json")):
    rel_path = exp_dir.relative_to(outputs_dir)
    with open(exp_dir) as f:
        data = json.load(f)
        config = data.get("config", {})

        all_experiments.append({
            "path": str(rel_path.parent),
            "partition_size": config.get("partition_size", "N/A"),
            "partition_strategy": config.get("partition_strategy", "N/A"),
            "max_events": config.get("max_events", "N/A"),
            "batch_mode": config.get("batch_mode", "N/A"),
            "train_batch_size": config.get("train_batch_size", "N/A"),
            "execution_mode": str(rel_path.parent).split("/")[-1] if "/" in str(rel_path.parent) else "N/A",
        })

# Group by top-level directory
by_dir = defaultdict(list)
for exp in all_experiments:
    top_dir = exp["path"].split("/")[0]
    by_dir[top_dir].append(exp)

print("=" * 150)
print("outputs/ 下所有实验的参数对比")
print("=" * 150)
print()

for top_dir in sorted(by_dir.keys()):
    exps = by_dir[top_dir]
    print(f"\n【{top_dir}】({len(exps)} 个实验)")
    print("-" * 150)
    print(f"{'实验路径':<50} | {'partition_size':<15} | {'batch_mode':<12} | {'train_batch_size':<15} | {'max_events':<12}")
    print("-" * 150)

    # Check for inconsistencies
    partition_sizes = set(e["partition_size"] for e in exps)
    batch_modes = set(e["batch_mode"] for e in exps)

    for exp in exps:
        psize_marker = " ⚠️" if exp["partition_size"] == 0 else ""
        batch_marker = " ⚠️" if exp["batch_mode"] == "N/A" else ""

        print(f"{exp['path']:<50} | {str(exp['partition_size']):<15}{psize_marker} | {str(exp['batch_mode']):<12}{batch_marker} | {str(exp['train_batch_size']):<15} | {str(exp['max_events']):<12}")

    # Summary for this directory
    print()
    if len(partition_sizes) > 1:
        print(f"  ⚠️ partition_size 不一致: {partition_sizes}")
    if 0 in partition_sizes or "N/A" in partition_sizes:
        count = sum(1 for e in exps if e["partition_size"] in [0, "N/A"])
        print(f"  ⚠️ {count}/{len(exps)} 个实验 partition_size=0 或缺失")
    if "N/A" in batch_modes:
        count = sum(1 for e in exps if e["batch_mode"] == "N/A")
        print(f"  ⚠️ {count}/{len(exps)} 个实验缺少 batch_mode 参数")

print()
print("=" * 150)
print("总体发现:")
print("=" * 150)

# Overall statistics
all_partition_sizes = [e["partition_size"] for e in all_experiments]
zero_count = sum(1 for p in all_partition_sizes if p == 0)
na_count = sum(1 for p in all_partition_sizes if p == "N/A")

print(f"总实验数: {len(all_experiments)}")
print(f"partition_size=0: {zero_count} 个")
print(f"partition_size=N/A: {na_count} 个")
print(f"partition_size>0: {len(all_experiments) - zero_count - na_count} 个")
