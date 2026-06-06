#!/usr/bin/env python3
import json
import csv
import os
from pathlib import Path

base_dir = Path("outputs/full_cross_experiment")
experiments = [
    "serial_serial", "serial_tbatch", "serial_tgn_last", "serial_tgn_all",
    "data_parallel_serial", "data_parallel_tbatch", "data_parallel_tgn_last", "data_parallel_tgn_all",
    "pipeline_naive_serial", "pipeline_naive_tbatch", "pipeline_naive_tgn_last", "pipeline_naive_tgn_all",
    "pipeline_smart_serial", "pipeline_smart_tbatch", "pipeline_smart_tgn_last", "pipeline_smart_tgn_all",
]

results = []

for exp_name in experiments:
    exp_dir = base_dir / exp_name
    if not exp_dir.exists():
        results.append({
            "name": exp_name,
            "status": "MISSING",
            "best_arch": "N/A",
            "val_mrr_range": "N/A",
            "best_trial": "N/A",
            "timing_range": "N/A",
            "anomaly": "异常",
            "note": "实验目录不存在"
        })
        continue

    # Read best_arch.json
    best_arch_path = exp_dir / "best_arch.json"
    if not best_arch_path.exists():
        results.append({
            "name": exp_name,
            "status": "INCOMPLETE",
            "best_arch": "N/A",
            "val_mrr_range": "N/A",
            "best_trial": "N/A",
            "timing_range": "N/A",
            "anomaly": "异常",
            "note": "best_arch.json不存在"
        })
        continue

    with open(best_arch_path) as f:
        best_arch = json.load(f)

    config = best_arch.get('config', {})
    arch_str = f"{config.get('model', 'N/A')}/{config.get('embedding_dim', 'N/A')}/{config.get('memory_cell', 'N/A')}"
    best_trial_id = best_arch.get('rank', 'N/A')
    test_mrr = best_arch.get('test_mrr', 'N/A')

    # Read leaderboard.csv (only coarse phase trials)
    leaderboard_path = exp_dir / "leaderboard.csv"
    val_mrrs = []
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                phase = row.get('phase', '')
                if phase.startswith('coarse'):  # coarse, coarse_dp, coarse_pipeline
                    try:
                        val_mrrs.append(float(row.get('mrr', row.get('score', 0))))
                    except:
                        pass

    if val_mrrs:
        val_mrr_range = f"{min(val_mrrs):.4f}-{max(val_mrrs):.4f}"
        # Check for anomalies: too many identical values
        unique_mrrs = len(set(val_mrrs))
        mrr_anomaly = unique_mrrs < len(val_mrrs) * 0.5  # Less than 50% unique
    else:
        val_mrr_range = "N/A"
        mrr_anomaly = True

    # Read timing_log.csv
    timing_path = exp_dir / "timing_log.csv"
    durations = []
    if timing_path.exists():
        with open(timing_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    durations.append(float(row.get('duration_s', 0)))
                except:
                    pass

    if durations:
        timing_range = f"{min(durations):.0f}-{max(durations):.0f}"
        # Check for timing anomalies
        avg_duration = sum(durations) / len(durations)
        timing_anomaly = any(d < avg_duration * 0.3 or d > avg_duration * 3 for d in durations)
    else:
        timing_range = "N/A"
        timing_anomaly = True

    # Determine anomaly status
    anomalies = []
    if mrr_anomaly:
        anomalies.append("MRR分布异常")
    if timing_anomaly:
        anomalies.append("耗时异常")
    if len(val_mrrs) != 27:
        anomalies.append(f"trial数={len(val_mrrs)}≠27")

    if anomalies:
        anomaly_flag = "警告" if len(anomalies) == 1 else "异常"
        note = "; ".join(anomalies)
    else:
        anomaly_flag = "正常"
        note = ""

    results.append({
        "name": exp_name,
        "best_arch": arch_str,
        "val_mrr_range": val_mrr_range,
        "best_trial": best_trial_id,
        "timing_range": timing_range,
        "anomaly": anomaly_flag,
        "note": note
    })

# Print table
print(f"{'实验名称':<30} | {'最佳架构':<40} | {'Val MRR范围':<20} | {'最佳Trial':<10} | {'耗时范围(秒)':<15} | {'状态':<6} | {'说明'}")
print("-" * 160)
for r in results:
    print(f"{r['name']:<30} | {r['best_arch']:<40} | {r['val_mrr_range']:<20} | {str(r['best_trial']):<10} | {r['timing_range']:<15} | {r['anomaly']:<6} | {r['note']}")
