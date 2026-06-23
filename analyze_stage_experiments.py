#!/usr/bin/env python3
"""简化版Pipeline NAS实验分析：只测试Stage数量的影响"""

import json
import csv
from pathlib import Path

# 实验映射
EXPERIMENTS = {
    'B1_serial': 'outputs/50k_comparison/seed_42/serial',
    'B2_data_parallel': 'outputs/50k_comparison/seed_42/data_parallel',
    'S1_1stage': 'outputs/50k_comparison/seed_42/smart_1stage',
    'S2_2stages': 'outputs/50k_comparison/seed_42/smart_overlap20',
    'S3_3stages': 'outputs/stage_test/S3_3stages',
    'S4_4stages': 'outputs/stage_test/S4_4stages',
}

def load_result(exp_path):
    """加载实验结果"""
    best_arch = Path(exp_path) / "best_arch.json"
    leaderboard = Path(exp_path) / "leaderboard.csv"

    if not best_arch.exists() or not leaderboard.exists():
        return None

    with open(best_arch) as f:
        arch = json.load(f)

    # 找出off/off架构的最佳Val MRR
    off_off_scores = []
    with open(leaderboard) as f:
        for row in csv.DictReader(f):
            cfg = json.loads(row['config_json'])
            if cfg.get('time_proj') == 'off' and cfg.get('use_static_embeddings') == 'off':
                off_off_scores.append(float(row['mrr']))

    return {
        'selected': f"{arch['config']['time_proj']}/{arch['config']['use_static_embeddings']}",
        'test_mrr': arch.get('test_mrr', arch.get('test_score')),
        'off_off_val': max(off_off_scores) if off_off_scores else None,
    }

print("="*80)
print("Pipeline Stage数量验证实验分析")
print("="*80)
print()

print("【基准组】")
print(f"{'实验':<20} {'选出架构':<15} {'Test MRR':<12} {'off/off Val':<12} {'状态'}")
print("-"*80)

for exp_id in ['B1_serial', 'B2_data_parallel']:
    path = EXPERIMENTS[exp_id]
    result = load_result(path) if Path(path).exists() else None

    if result:
        is_correct = "✅" if result['selected'] == "off/off" else "❌"
        off_off = f"{result['off_off_val']:.4f}" if result['off_off_val'] else "N/A"
        print(f"{exp_id:<20} {result['selected']:<15} {result['test_mrr']:<12.4f} {off_off:<12} {is_correct}")
    else:
        print(f"{exp_id:<20} {'待补充':<15} {'N/A':<12} {'N/A':<12} {'⏳'}")

print()
print("【Stage数量测试】(固定Overlap=20%, RL)")
print(f"{'实验':<20} {'选出架构':<15} {'Test MRR':<12} {'off/off Val':<12} {'状态'}")
print("-"*80)

for exp_id in ['S1_1stage', 'S2_2stages', 'S3_3stages', 'S4_4stages']:
    path = EXPERIMENTS[exp_id]
    result = load_result(path) if Path(path).exists() else None

    if result:
        is_correct = "✅" if result['selected'] == "off/off" else "❌"
        off_off = f"{result['off_off_val']:.4f}" if result['off_off_val'] else "N/A"
        print(f"{exp_id:<20} {result['selected']:<15} {result['test_mrr']:<12.4f} {off_off:<12} {is_correct}")
    else:
        print(f"{exp_id:<20} {'待补充':<15} {'N/A':<12} {'N/A':<12} {'⏳'}")

print()
print("="*80)

# 统计需要补充的实验
missing = [exp_id for exp_id, path in EXPERIMENTS.items()
           if not Path(path).exists() or not (Path(path)/"best_arch.json").exists()]

if missing:
    print(f"需要补充的实验 ({len(missing)}个): {', '.join(missing)}")
    print()
    print("运行补充实验: bash run_stage_experiments.sh")
else:
    print("所有实验已完成!")
