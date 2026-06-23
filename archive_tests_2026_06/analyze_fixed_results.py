#!/usr/bin/env python3
import json
from pathlib import Path

base_dir = Path("outputs/full_cross_experiment_fixed")
experiments = ["serial_tbatch", "data_parallel_tbatch", "pipeline_naive_tbatch", "pipeline_smart_tbatch"]

print("=" * 100)
print("Step 4: 修复后的 tbatch 实验结果汇总")
print("=" * 100)
print()

results = []
for exp_name in experiments:
    best_arch_path = base_dir / exp_name / "best_arch.json"
    if best_arch_path.exists():
        with open(best_arch_path) as f:
            data = json.load(f)
            config = data.get("config", {})

            results.append({
                "name": exp_name.replace("_tbatch", ""),
                "test_mrr": data.get("test_mrr", 0),
                "test_recall": data.get("test_recall_at_k", 0),
                "val_mrr": data.get("val_mrr", 0),
                "val_recall": data.get("val_recall_at_k", 0),
                "time_sec": data.get("time_sec", 0),
                "partition_size": config.get("partition_size", "N/A"),
            })

# Print table
print(f"{'并行模式':<20} | {'Test MRR':<10} | {'Test Recall':<12} | {'Val MRR':<10} | {'Val Recall':<12} | {'训练时间(秒)':<12} | {'partition_size':<15}")
print("-" * 100)

for r in results:
    print(f"{r['name']:<20} | {r['test_mrr']:<10.4f} | {r['test_recall']:<12.4f} | {r['val_mrr']:<10.4f} | {r['val_recall']:<12.4f} | {r['time_sec']:<12.0f} | {r['partition_size']:<15}")

print()
print("=" * 100)
print("Step 5: 对比分析")
print("=" * 100)
print()

# Load original (broken) results for comparison
original_dir = Path("outputs/full_cross_experiment")
original_results = {}
for exp_name in experiments:
    orig_path = original_dir / exp_name / "best_arch.json"
    if orig_path.exists():
        with open(orig_path) as f:
            data = json.load(f)
            original_results[exp_name.replace("_tbatch", "")] = {
                "test_mrr": data.get("test_mrr", 0),
            }

print("【修复前后对比】(partition_size: 0 → 1000)")
print("-" * 100)
print(f"{'模式':<20} | {'修复前 MRR':<12} | {'修复后 MRR':<12} | {'变化':<12} | {'变化率':<10}")
print("-" * 100)

for r in results:
    name = r['name']
    if name in original_results:
        old_mrr = original_results[name]['test_mrr']
        new_mrr = r['test_mrr']
        diff = new_mrr - old_mrr
        pct = (diff / old_mrr * 100) if old_mrr > 0 else 0
        print(f"{name:<20} | {old_mrr:<12.4f} | {new_mrr:<12.4f} | {diff:+<12.4f} | {pct:+<10.1f}%")

print()
print("【关键发现】")
print("-" * 100)

# Find best performer
best = max(results, key=lambda x: x['test_mrr'])
print(f"1. 最佳性能: {best['name']} (Test MRR={best['test_mrr']:.4f})")

# Compare with medium_4way_multiseed
print(f"\n2. 与 medium_4way_multiseed 对比:")
print(f"   - medium serial (partition_size=1000): Test MRR ≈ 0.6845")
print(f"   - fixed serial (partition_size=1000): Test MRR = {[r for r in results if r['name']=='serial'][0]['test_mrr']:.4f}")

# Parallel efficiency
serial_mrr = [r for r in results if r['name']=='serial'][0]['test_mrr']
for r in results:
    if r['name'] != 'serial':
        gap = (serial_mrr - r['test_mrr']) / serial_mrr * 100
        print(f"   - {r['name']} vs serial: {gap:.1f}% 性能差距")

print()
