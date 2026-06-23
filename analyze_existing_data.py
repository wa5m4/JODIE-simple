#!/usr/bin/env python3
"""分析三因素实验的现有数据"""

import json
import csv
from pathlib import Path

# 现有实验映射
EXPERIMENTS = {
    'B1_serial': 'outputs/50k_comparison/seed_42/serial',
    'B2_data_parallel': 'outputs/50k_comparison/seed_42/data_parallel_improved',
    'E1_1s_20%_async': 'outputs/50k_comparison/seed_42/smart_1stage',
    'E3_2s_20%_async': 'outputs/50k_comparison/seed_42/smart_overlap20',
    'E4_2s_0%_async': 'outputs/50k_comparison/seed_42/naive_no_overlap',
    'E6_3s_0%_async': 'outputs/50k_comparison/seed_42/naive_3stages',
}

def load_result(path):
    best = Path(path) / "best_arch.json"
    board = Path(path) / "leaderboard.csv"
    if not best.exists() or not board.exists():
        return None

    with open(best) as f:
        arch = json.load(f)

    off_off_vals = []
    with open(board) as f:
        for row in csv.DictReader(f):
            cfg = json.loads(row['config_json'])
            if cfg.get('time_proj') == 'off' and cfg.get('use_static_embeddings') == 'off':
                off_off_vals.append(float(row['mrr']))

    return {
        'selected': f"{arch['config']['time_proj']}/{arch['config']['use_static_embeddings']}",
        'test_mrr': arch.get('test_mrr', arch.get('test_score')),
        'off_off_val': max(off_off_vals) if off_off_vals else None,
    }

print("="*90)
print("Pipeline NAS 三因素实验分析")
print("="*90)
print()

print("【基准组】")
print(f"{'实验':<25} {'选出架构':<15} {'Test MRR':<12} {'off/off Val':<12} {'正确'}")
print("-"*90)

for exp_id in ['B1_serial', 'B2_data_parallel']:
    r = load_result(EXPERIMENTS[exp_id]) if Path(EXPERIMENTS[exp_id]).exists() else None
    if r:
        ok = "✅" if r['selected'] == "off/off" else "❌"
        off_val = f"{r['off_off_val']:.4f}" if r['off_off_val'] else "N/A"
        print(f"{exp_id:<25} {r['selected']:<15} {r['test_mrr']:<12.4f} {off_val:<12} {ok}")

print()
print("【因素A：Stage划分】(固定20% overlap, 异步)")
print(f"{'实验':<25} {'选出架构':<15} {'Test MRR':<12} {'off/off Val':<12} {'正确'}")
print("-"*90)

for exp_id in ['E1_1s_20%_async', 'E3_2s_20%_async']:
    r = load_result(EXPERIMENTS[exp_id]) if Path(EXPERIMENTS[exp_id]).exists() else None
    if r:
        ok = "✅" if r['selected'] == "off/off" else "❌"
        off_val = f"{r['off_off_val']:.4f}" if r['off_off_val'] else "N/A"
        print(f"{exp_id:<25} {r['selected']:<15} {r['test_mrr']:<12.4f} {off_val:<12} {ok}")
print(f"{'E5_3s_20%_async':<25} {'待补充':<15} {'N/A':<12} {'N/A':<12} {'⏳'}")

print()
print("【因素B：Overlap】(固定2 stages, 异步)")
print(f"{'实验':<25} {'选出架构':<15} {'Test MRR':<12} {'off/off Val':<12} {'正确'}")
print("-"*90)

for exp_id in ['E4_2s_0%_async', 'E3_2s_20%_async']:
    r = load_result(EXPERIMENTS[exp_id]) if Path(EXPERIMENTS[exp_id]).exists() else None
    if r:
        ok = "✅" if r['selected'] == "off/off" else "❌"
        off_val = f"{r['off_off_val']:.4f}" if r['off_off_val'] else "N/A"
        print(f"{exp_id:<25} {r['selected']:<15} {r['test_mrr']:<12.4f} {off_val:<12} {ok}")

print()
print("【交叉验证：3 stages】")
print(f"{'实验':<25} {'选出架构':<15} {'Test MRR':<12} {'off/off Val':<12} {'正确'}")
print("-"*90)

for exp_id in ['E6_3s_0%_async']:
    r = load_result(EXPERIMENTS[exp_id]) if Path(EXPERIMENTS[exp_id]).exists() else None
    if r:
        ok = "✅" if r['selected'] == "off/off" else "❌"
        off_val = f"{r['off_off_val']:.4f}" if r['off_off_val'] else "N/A"
        print(f"{exp_id:<25} {r['selected']:<15} {r['test_mrr']:<12.4f} {off_val:<12} {ok}")
print(f"{'E5_3s_20%_async':<25} {'待补充':<15} {'N/A':<12} {'N/A':<12} {'⏳'}")

print()
print("="*90)
print("需要补充的异步实验:")
print("  E2: 1 stage + 0% overlap")
print("  E5: 3 stages + 20% overlap")
print()
print("运行补充实验: bash run_missing_experiments.sh")
