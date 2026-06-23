#!/usr/bin/env python3
import json
import csv
from pathlib import Path
from collections import Counter

base_dir = Path("outputs/full_cross_experiment")
experiments = [
    "serial_serial", "serial_tbatch", "serial_tgn_last", "serial_tgn_all",
    "data_parallel_serial", "data_parallel_tbatch", "data_parallel_tgn_last", "data_parallel_tgn_all",
    "pipeline_naive_serial", "pipeline_naive_tbatch", "pipeline_naive_tgn_last", "pipeline_naive_tgn_all",
    "pipeline_smart_serial", "pipeline_smart_tbatch", "pipeline_smart_tgn_last", "pipeline_smart_tgn_all",
]

print("=" * 180)
print("16组实验完整性诊断报告")
print("=" * 180)
print()

results = []

for exp_name in experiments:
    exp_dir = base_dir / exp_name
    if not exp_dir.exists():
        continue

    # Read best_arch.json
    best_arch_path = exp_dir / "best_arch.json"
    if not best_arch_path.exists():
        continue

    with open(best_arch_path) as f:
        best_arch = json.load(f)

    config = best_arch.get('config', {})
    arch_str = f"{config.get('model', 'N/A')}/{config.get('embedding_dim', 'N/A')}/{config.get('memory_cell', 'N/A')}"
    val_mrr = best_arch.get('val_mrr', 0)
    test_mrr = best_arch.get('test_mrr', 0)

    # Read leaderboard.csv
    leaderboard_path = exp_dir / "leaderboard.csv"
    val_mrrs = []
    model_counts = Counter()

    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                phase = row.get('phase', '')
                if phase.startswith('coarse'):
                    try:
                        val_mrrs.append(float(row.get('mrr', 0)))
                        model_counts[row.get('model', 'unknown')] += 1
                    except:
                        pass

    # Read timing_log.csv
    timing_path = exp_dir / "timing_log.csv"
    durations = []
    if timing_path.exists():
        with open(timing_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dur = float(row.get('duration_s', 0))
                    # If duration_s is 0, calculate from end_time_s - start_time_s
                    if dur == 0:
                        end_t = float(row.get('end_time_s', 0))
                        start_t = float(row.get('start_time_s', 0))
                        dur = end_t - start_t
                    durations.append(dur)
                except:
                    pass

    # Calculate statistics
    if val_mrrs:
        val_min, val_max = min(val_mrrs), max(val_mrrs)
        val_range = val_max - val_min
        unique_count = len(set(val_mrrs))
    else:
        val_min = val_max = val_range = unique_count = 0

    if durations:
        time_min, time_max = min(durations), max(durations)
        time_avg = sum(durations) / len(durations)
    else:
        time_min = time_max = time_avg = 0

    # Detect anomalies
    anomalies = []
    if len(val_mrrs) != 27:
        anomalies.append(f"trial数={len(val_mrrs)}≠27")
    if unique_count < len(val_mrrs) * 0.5:
        anomalies.append("MRR重复值过多")
    if val_range < 0.1:
        anomalies.append(f"MRR变化范围过小({val_range:.4f})")
    if durations and any(d < time_avg * 0.3 for d in durations):
        anomalies.append("存在异常短耗时trial")

    # Val-Test gap
    val_test_gap = test_mrr - val_mrr
    if abs(val_test_gap) > 0.15:
        anomalies.append(f"Val-Test差距大({val_test_gap:+.3f})")

    results.append({
        "name": exp_name,
        "arch": arch_str,
        "val_mrr": val_mrr,
        "test_mrr": test_mrr,
        "val_range": f"{val_min:.4f}-{val_max:.4f}",
        "val_span": val_range,
        "trials": len(val_mrrs),
        "unique": unique_count,
        "time_range": f"{time_min:.0f}-{time_max:.0f}",
        "time_avg": time_avg,
        "models": dict(model_counts),
        "anomalies": anomalies
    })

# Print table
print(f"{'实验名称':<30} | {'最佳架构':<25} | {'Val MRR':<8} | {'Test MRR':<8} | {'Val范围':<20} | {'Trials':<7} | {'唯一值':<6} | {'耗时范围':<12} | {'状态'}")
print("-" * 180)

for r in results:
    status = "异常" if r['anomalies'] else "正常"
    print(f"{r['name']:<30} | {r['arch']:<25} | {r['val_mrr']:<8.4f} | {r['test_mrr']:<8.4f} | {r['val_range']:<20} | {r['trials']:<7} | {r['unique']:<6} | {r['time_range']:<12} | {status}")

print()
print("=" * 180)
print("异常详情")
print("=" * 180)

for r in results:
    if r['anomalies']:
        print(f"\n{r['name']}:")
        for anomaly in r['anomalies']:
            print(f"  - {anomaly}")

print()
print("=" * 180)
print("模型选择分布")
print("=" * 180)

for r in results:
    print(f"\n{r['name']}:")
    for model, count in r['models'].items():
        print(f"  {model}: {count}次 ({count/r['trials']*100:.1f}%)")
