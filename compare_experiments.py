#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

# Extract medium_4way_multiseed results
medium_dir = Path("outputs/medium_4way_multiseed")
medium_results = defaultdict(list)

for seed_dir in sorted(medium_dir.glob("seed_*")):
    seed = seed_dir.name
    for mode_dir in seed_dir.iterdir():
        if mode_dir.is_dir():
            best_arch = mode_dir / "best_arch.json"
            if best_arch.exists():
                with open(best_arch) as f:
                    data = json.load(f)
                    mode = mode_dir.name
                    # Medium experiments use test_score, not test_mrr
                    test_mrr = data.get("test_mrr") or data.get("test_score") or 0
                    test_recall = data.get("test_recall_at_k") or data.get("recall_at_k") or 0
                    medium_results[mode].append({
                        "seed": seed,
                        "test_mrr": test_mrr,
                        "test_recall": test_recall,
                    })

# Extract full_cross_experiment results
full_dir = Path("outputs/full_cross_experiment")
full_results = {}

for exp_dir in full_dir.iterdir():
    if exp_dir.is_dir():
        best_arch = exp_dir / "best_arch.json"
        if best_arch.exists():
            with open(best_arch) as f:
                data = json.load(f)
                full_results[exp_dir.name] = {
                    "test_mrr": data.get("test_mrr", 0),
                    "test_recall": data.get("test_recall_at_k", 0),
                }

# Print comparison
print("=" * 120)
print("medium_4way_multiseed vs full_cross_experiment 对比")
print("=" * 120)
print()

# Calculate averages for medium experiments
print("【medium_4way_multiseed 平均结果】(多种子平均)")
print("-" * 120)
for mode in sorted(medium_results.keys()):
    results = medium_results[mode]
    avg_mrr = sum(r["test_mrr"] for r in results) / len(results)
    avg_recall = sum(r["test_recall"] for r in results) / len(results)
    print(f"{mode:<30} | Test MRR: {avg_mrr:.4f} | Test Recall: {avg_recall:.4f} | Seeds: {len(results)}")

print()
print("【full_cross_experiment 结果】(单种子, seed=42)")
print("-" * 120)

# Map full experiment names to medium names for comparison
mode_mapping = {
    "serial_serial": "serial",
    "data_parallel_serial": "data_parallel",
    "pipeline_naive_serial": "pipeline_naive",
    "pipeline_smart_serial": "pipeline_smart",
}

for full_name in sorted(full_results.keys()):
    if full_name.endswith("_serial"):  # Only compare serial batch mode
        result = full_results[full_name]
        print(f"{full_name:<30} | Test MRR: {result['test_mrr']:.4f} | Test Recall: {result['test_recall']:.4f}")

print()
print("=" * 120)
print("【差异分析】")
print("=" * 120)
print()

for full_name, medium_name in mode_mapping.items():
    if full_name in full_results and medium_name in medium_results:
        full_mrr = full_results[full_name]["test_mrr"]
        medium_mrrs = [r["test_mrr"] for r in medium_results[medium_name]]
        medium_avg = sum(medium_mrrs) / len(medium_mrrs)
        diff = full_mrr - medium_avg
        diff_pct = (diff / medium_avg) * 100 if medium_avg > 0 else 0

        print(f"{full_name} vs {medium_name}:")
        print(f"  Full:   {full_mrr:.4f}")
        print(f"  Medium: {medium_avg:.4f} (avg of {len(medium_mrrs)} seeds)")
        print(f"  差异:   {diff:+.4f} ({diff_pct:+.1f}%)")
        print()
