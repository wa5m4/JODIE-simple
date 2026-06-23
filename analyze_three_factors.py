#!/usr/bin/env python3
"""三因素实验完整分析脚本"""

import json
import csv
from pathlib import Path

BASE = Path("outputs/three_factor_test")

def load_result(path):
    best = path / "best_arch.json"
    board = path / "leaderboard.csv"
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
        'arch': f"{arch['config']['time_proj']}/{arch['config']['use_static_embeddings']}",
        'test': arch.get('test_mrr', arch.get('test_score')),
        'off_off_val': max(off_off_vals) if off_off_vals else None,
    }

print("="*100)
print("Pipeline NAS 三因素完全交叉实验分析")
print("="*100)
print()

# 基准组
print("【基准组】")
print(f"{'实验':<20} {'架构':<15} {'Test MRR':<12} {'off/off Val':<12} {'正确'}")
print("-"*100)
for name in ['serial', 'data_parallel']:
    r = load_result(BASE / 'baseline' / name)
    if r:
        ok = "✅" if r['arch'] == 'off/off' else "❌"
        val = f"{r['off_off_val']:.4f}" if r['off_off_val'] else "N/A"
        print(f"{name:<20} {r['arch']:<15} {r['test']:<12.4f} {val:<12} {ok}")

# 异步实验矩阵
print()
print("【异步实验】(pipeline-mode=smart)")
print(f"{'实验':<25} {'架构':<15} {'Test MRR':<12} {'off/off Val':<12} {'正确'}")
print("-"*100)

for stages, overlap in [(1,0), (1,20), (2,0), (2,20), (3,0), (3,20)]:
    name = f"{stages}stage_overlap{overlap}"
    path = BASE / 'async' / name
    r = load_result(path)
    if r:
        ok = "✅" if r['arch'] == 'off/off' else "❌"
        val = f"{r['off_off_val']:.4f}" if r['off_off_val'] else "N/A"
        print(f"{name:<25} {r['arch']:<15} {r['test']:<12.4f} {val:<12} {ok}")
    else:
        print(f"{name:<25} {'待补充':<15} {'N/A':<12} {'N/A':<12} {'⏳'}")

# 同步实验矩阵
print()
print("【同步实验】(pipeline-mode=naive)")
print(f"{'实验':<25} {'架构':<15} {'Test MRR':<12} {'off/off Val':<12} {'正确'}")
print("-"*100)

for stages, overlap in [(1,0), (1,20), (2,0), (2,20), (3,0), (3,20)]:
    name = f"{stages}stage_overlap{overlap}"
    path = BASE / 'sync' / name
    r = load_result(path)
    if r:
        ok = "✅" if r['arch'] == 'off/off' else "❌"
        val = f"{r['off_off_val']:.4f}" if r['off_off_val'] else "N/A"
        print(f"{name:<25} {r['arch']:<15} {r['test']:<12.4f} {val:<12} {ok}")
    else:
        print(f"{name:<25} {'待补充':<15} {'N/A':<12} {'N/A':<12} {'⏳'}")

print()
print("="*100)

# 统计需要补充的实验
missing = []
for mode in ['async', 'sync']:
    for stages, overlap in [(1,0), (1,20), (2,0), (2,20), (3,0), (3,20)]:
        name = f"{stages}stage_overlap{overlap}"
        if not (BASE / mode / name / "best_arch.json").exists():
            missing.append(f"{mode}/{name}")

if missing:
    print(f"需补充实验 ({len(missing)}/12): {', '.join(missing)}")
    print()
    print("运行: bash run_missing_experiments.sh")
else:
    print("所有实验已完成!")
