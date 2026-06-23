#!/usr/bin/env python3
"""
Pipeline NAS综合实验结果分析脚本 (复用已有数据)
分析outputs/50k_comparison/seed_42/中的现有实验 + 补充新实验
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict

# 实验映射：将现有数据映射到实验计划中的ID
EXISTING_EXPERIMENTS = {
    # 第一部分：基准组
    'B1_serial': 'outputs/50k_comparison/seed_42/serial',
    'B2_data_parallel': 'outputs/50k_comparison/seed_42/data_parallel',  # 需要确认

    # 第二部分：Stage数量 (固定overlap=20%)
    'S1_1stage_20overlap': 'outputs/50k_comparison/seed_42/smart_1stage',
    'S2_2stages_20overlap': 'outputs/50k_comparison/seed_42/smart_overlap20',
    # S3, S4 需要补充

    # 第三部分：Overlap (固定2stages)
    # O1 (0%), O2 (10%) 需要补充
    'O3_2stages_20overlap': 'outputs/50k_comparison/seed_42/smart_overlap20',  # 与S2相同

    # 第四部分：交叉验证
    # C1: 1stage + 0% 需要补充
    'C2_3stages_0overlap': 'outputs/50k_comparison/seed_42/naive_3stages',
    # C3, C4 需要补充
}

# 需要补充的实验
NEW_EXPERIMENTS = {
    'S3_3stages_20overlap': {
        'stages': 3,
        'overlap': 0.2,
        'description': '3 stages + 20% overlap'
    },
    'S4_4stages_20overlap': {
        'stages': 4,
        'overlap': 0.2,
        'description': '4 stages + 20% overlap'
    },
    'O1_2stages_0overlap': {
        'stages': 2,
        'overlap': 0.0,
        'description': '2 stages + 0% overlap'
    },
    'O2_2stages_10overlap': {
        'stages': 2,
        'overlap': 0.1,
        'description': '2 stages + 10% overlap'
    },
    'C1_1stage_0overlap': {
        'stages': 1,
        'overlap': 0.0,
        'description': '1 stage + 0% overlap'
    },
    'C3_3stages_20overlap': {
        'stages': 3,
        'overlap': 0.2,
        'description': '3 stages + 20% overlap'
    },
    'C4_4stages_10overlap': {
        'stages': 4,
        'overlap': 0.1,
        'description': '4 stages + 10% overlap'
    },
    'M1_2stages_random': {
        'stages': 2,
        'overlap': 0.2,
        'search_mode': 'random',
        'description': '2 stages + 20% overlap + Random search'
    }
}

def load_experiment_result(exp_path):
    """加载单个实验的结果"""
    exp_path = Path(exp_path)
    best_arch_path = exp_path / "best_arch.json"
    leaderboard_path = exp_path / "leaderboard.csv"

    if not best_arch_path.exists() or not leaderboard_path.exists():
        return None

    # 加载最佳架构
    with open(best_arch_path) as f:
        best_arch = json.load(f)

    # 分析leaderboard中off/off架构的表现
    off_off_archs = []
    total_archs = 0
    with open(leaderboard_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_archs += 1
            config = json.loads(row['config_json'])
            if config.get('time_proj') == 'off' and config.get('use_static_embeddings') == 'off':
                off_off_archs.append({
                    'memory_cell': config.get('memory_cell'),
                    'emb_dim': config.get('embedding_dim'),
                    'val_mrr': float(row['mrr']),
                    'rank': int(row['rank'])
                })

    # 找出最佳off/off架构
    best_off_off = max(off_off_archs, key=lambda x: x['val_mrr']) if off_off_archs else None

    # 判断选出的架构
    selected_arch = {
        'time_proj': best_arch['config'].get('time_proj'),
        'use_static': best_arch['config'].get('use_static_embeddings'),
        'memory_cell': best_arch['config'].get('memory_cell'),
        'emb_dim': best_arch['config'].get('embedding_dim'),
    }

    return {
        'selected_arch': selected_arch,
        'selected_is_off_off': (selected_arch['time_proj'] == 'off' and
                                selected_arch['use_static'] == 'off'),
        'test_mrr': best_arch.get('test_mrr', best_arch.get('test_score')),
        'val_mrr': best_arch.get('val_mrr', best_arch.get('val_score')),
        'off_off_best': best_off_off,
        'off_off_count': len(off_off_archs),
        'total_archs': total_archs,
    }

def analyze_existing_experiments():
    """分析已有实验"""
    results = {}

    for exp_id, exp_path in EXISTING_EXPERIMENTS.items():
        if Path(exp_path).exists():
            result = load_experiment_result(exp_path)
            if result:
                results[exp_id] = result
                results[exp_id]['status'] = 'completed'
            else:
                results[exp_id] = {'status': 'failed'}
        else:
            results[exp_id] = {'status': 'missing'}

    return results

def print_analysis(results):
    """打印分析结果"""
    print("="*100)
    print("Pipeline NAS 综合实验结果分析")
    print("="*100)
    print()

    print("【第一部分：基准组】")
    print("-"*100)
    baseline_exps = ['B1_serial', 'B2_data_parallel']
    print(f"{'实验ID':<25} {'选出架构':<15} {'Test MRR':<12} {'off/off最佳Val':<15} {'状态':<10}")
    print("-"*100)

    for exp_id in baseline_exps:
        if exp_id in results and results[exp_id]['status'] == 'completed':
            r = results[exp_id]
            arch_str = f"{r['selected_arch']['time_proj']}/{r['selected_arch']['use_static']}"
            off_off_str = f"{r['off_off_best']['val_mrr']:.4f}" if r['off_off_best'] else "N/A"
            status = "✅" if r['selected_is_off_off'] else "❌"
            print(f"{exp_id:<25} {arch_str:<15} {r['test_mrr']:<12.4f} {off_off_str:<15} {status}")
        else:
            status = results.get(exp_id, {}).get('status', 'unknown')
            print(f"{exp_id:<25} {'N/A':<15} {'N/A':<12} {'N/A':<15} {status}")
    print()

    print("【第二部分：Stage数量实验】(固定Overlap=20%)")
    print("-"*100)
    stage_exps = ['S1_1stage_20overlap', 'S2_2stages_20overlap', 'S3_3stages_20overlap', 'S4_4stages_20overlap']
    print(f"{'实验ID':<25} {'选出架构':<15} {'Test MRR':<12} {'off/off最佳Val':<15} {'状态':<10}")
    print("-"*100)

    for exp_id in stage_exps:
        if exp_id in results and results[exp_id]['status'] == 'completed':
            r = results[exp_id]
            arch_str = f"{r['selected_arch']['time_proj']}/{r['selected_arch']['use_static']}"
            off_off_str = f"{r['off_off_best']['val_mrr']:.4f}" if r['off_off_best'] else "N/A"
            status = "✅" if r['selected_is_off_off'] else "❌"
            print(f"{exp_id:<25} {arch_str:<15} {r['test_mrr']:<12.4f} {off_off_str:<15} {status}")
        else:
            print(f"{exp_id:<25} {'待补充':<15} {'N/A':<12} {'N/A':<15} {'⏳'}")
    print()

    print("【第三部分：Overlap比例实验】(固定Stage=2)")
    print("-"*100)
    overlap_exps = ['O1_2stages_0overlap', 'O2_2stages_10overlap', 'O3_2stages_20overlap']
    print(f"{'实验ID':<25} {'选出架构':<15} {'Test MRR':<12} {'off/off最佳Val':<15} {'状态':<10}")
    print("-"*100)

    for exp_id in overlap_exps:
        if exp_id in results and results[exp_id]['status'] == 'completed':
            r = results[exp_id]
            arch_str = f"{r['selected_arch']['time_proj']}/{r['selected_arch']['use_static']}"
            off_off_str = f"{r['off_off_best']['val_mrr']:.4f}" if r['off_off_best'] else "N/A"
            status = "✅" if r['selected_is_off_off'] else "❌"
            print(f"{exp_id:<25} {arch_str:<15} {r['test_mrr']:<12.4f} {off_off_str:<15} {status}")
        else:
            print(f"{exp_id:<25} {'待补充':<15} {'N/A':<12} {'N/A':<15} {'⏳'}")
    print()

    print("【第四部分：Stage×Overlap交叉验证】")
    print("-"*100)
    cross_exps = ['C1_1stage_0overlap', 'C2_3stages_0overlap', 'C3_3stages_20overlap', 'C4_4stages_10overlap']
    print(f"{'实验ID':<25} {'选出架构':<15} {'Test MRR':<12} {'off/off最佳Val':<15} {'状态':<10}")
    print("-"*100)

    for exp_id in cross_exps:
        if exp_id in results and results[exp_id]['status'] == 'completed':
            r = results[exp_id]
            arch_str = f"{r['selected_arch']['time_proj']}/{r['selected_arch']['use_static']}"
            off_off_str = f"{r['off_off_best']['val_mrr']:.4f}" if r['off_off_best'] else "N/A"
            status = "✅" if r['selected_is_off_off'] else "❌"
            print(f"{exp_id:<25} {arch_str:<15} {r['test_mrr']:<12.4f} {off_off_str:<15} {status}")
        else:
            print(f"{exp_id:<25} {'待补充':<15} {'N/A':<12} {'N/A':<15} {'⏳'}")
    print()

    print("="*100)
    print("【关键发现】")
    print("="*100)

    # 统计已完成的实验中正确的配置
    correct_configs = []
    for exp_id, result in results.items():
        if result['status'] == 'completed' and result['selected_is_off_off']:
            correct_configs.append(exp_id)

    if correct_configs:
        print(f"✅ 正确选出off/off的配置: {', '.join(correct_configs)}")
    else:
        print("❌ 暂无配置正确选出off/off")

    print()

    # 需要补充的实验列表
    needed = []
    for exp_id in NEW_EXPERIMENTS:
        if exp_id not in results or results[exp_id]['status'] != 'completed':
            needed.append(exp_id)

    if needed:
        print(f"⏳ 需要补充的实验 ({len(needed)}个): {', '.join(needed)}")
        print()
        print("运行补充实验: bash run_supplementary_experiments.sh")

def generate_supplementary_script():
    """生成补充实验的脚本"""
    script_lines = [
        "#!/bin/bash",
        "",
        "# 补充实验脚本 (仅运行缺失的实验)",
        "",
        "set -e",
        "",
        'DATASET="public_csv"',
        'MAX_EVENTS=50000',
        'SEED=42',
        'TRIALS=50',
        'SPACE="rnn_only"',
        'PARTITION_SIZE=12500',
        'BASE_OUTPUT="outputs/comprehensive_experiment"',
        "",
        'echo "=== 运行补充实验 ==="',
        'echo ""',
        "",
        'mkdir -p "$BASE_OUTPUT"',
        "",
    ]

    for exp_id, config in NEW_EXPERIMENTS.items():
        stages = config['stages']
        overlap = config['overlap']
        search_mode = config.get('search_mode', 'rl')

        script_lines.extend([
            f"# {exp_id}: {config['description']}",
            f'echo "[{exp_id}] {config["description"]}..."',
            "python search.py \\",
            '  --dataset "$DATASET" \\',
            '  --max-events $MAX_EVENTS \\',
            '  --seed $SEED \\',
            '  --space "$SPACE" \\',
            '  --coarse-trials $TRIALS \\',
            '  --coarse-epochs 1 \\',
            '  --execution-mode ray_pipeline \\',
            f'  --search-mode {search_mode} \\',
            f'  --num-pipeline-stages {stages} \\',
            '  --partition-size $PARTITION_SIZE \\',
            f'  --partition-overlap-ratio {overlap} \\',
            '  --pipeline-mode smart \\',
            '  --architectures-per-step 2 \\',
            f'  --output-dir "$BASE_OUTPUT/{exp_id}" \\',
            f'  > "$BASE_OUTPUT/{exp_id}.log" 2>&1 &',
            "",
        ])

    script_lines.extend([
        'echo "等待所有实验完成..."',
        'wait',
        'echo "补充实验完成!"',
        'echo ""',
        'echo "运行分析: python analyze_comprehensive_experiments_v2.py"',
    ])

    return "\n".join(script_lines)

if __name__ == "__main__":
    print("正在分析已有实验...")
    results = analyze_existing_experiments()

    print_analysis(results)

    # 生成补充实验脚本
    script_content = generate_supplementary_script()
    with open("run_supplementary_experiments.sh", "w") as f:
        f.write(script_content)

    print()
    print("补充实验脚本已生成: run_supplementary_experiments.sh")
