#!/usr/bin/env python3
"""
Pipeline NAS综合实验结果分析脚本
"""

import json
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

def load_experiment_result(exp_dir):
    """加载单个实验的结果"""
    best_arch_path = Path(exp_dir) / "best_arch.json"
    leaderboard_path = Path(exp_dir) / "leaderboard.csv"

    if not best_arch_path.exists() or not leaderboard_path.exists():
        return None

    # 加载最佳架构
    with open(best_arch_path) as f:
        best_arch = json.load(f)

    # 分析leaderboard中off/off架构的表现
    off_off_scores = []
    with open(leaderboard_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = json.loads(row['config_json'])
            if config.get('time_proj') == 'off' and config.get('use_static_embeddings') == 'off':
                off_off_scores.append({
                    'memory_cell': config.get('memory_cell'),
                    'emb_dim': config.get('embedding_dim'),
                    'val_mrr': float(row['mrr']),
                    'rank': int(row['rank'])
                })

    result = {
        'selected_arch': {
            'time_proj': best_arch['config'].get('time_proj'),
            'use_static': best_arch['config'].get('use_static_embeddings'),
            'memory_cell': best_arch['config'].get('memory_cell'),
            'emb_dim': best_arch['config'].get('embedding_dim'),
        },
        'test_mrr': best_arch.get('test_mrr', best_arch.get('test_score')),
        'val_mrr': best_arch.get('val_mrr', best_arch.get('val_score')),
        'off_off_best': max(off_off_scores, key=lambda x: x['val_mrr']) if off_off_scores else None,
        'off_off_count': len(off_off_scores)
    }

    return result

def analyze_experiments(base_dir):
    """分析所有实验结果"""
    base_path = Path(base_dir)

    experiments = {
        # 第一部分：基准组
        'baseline': {
            'B1_serial': 'B1_serial_baseline',
            'B2_data_parallel': 'B2_data_parallel',
        },
        # 第二部分：Stage数量
        'stages': {
            '1-stage': 'S1_pipeline_1stage_overlap0.2',
            '2-stages': 'S2_pipeline_2stage_overlap0.2',
            '3-stages': 'S3_pipeline_3stage_overlap0.2',
            '4-stages': 'S4_pipeline_4stage_overlap0.2',
        },
        # 第三部分：Overlap比例
        'overlap': {
            '0%': 'O1_pipeline_2stage_overlap0.0',
            '10%': 'O2_pipeline_2stage_overlap0.1',
            '20%': 'O3_pipeline_2stage_overlap0.2',
        },
        # 第四部分：交叉验证
        'cross': {
            '1s+0%': 'C1_1stage_no_overlap',
            '3s+0%': 'C2_3stages_no_overlap',
            '3s+20%': 'C3_3stages_overlap20',
            '4s+10%': 'C4_4stages_overlap10',
        },
        # 第五部分：搜索模式
        'search_mode': {
            'Random': 'M1_2stage_random',
            'RL': 'S2_pipeline_2stage_overlap0.2',  # 与S2共用
        }
    }

    results = {}

    for category, exps in experiments.items():
        results[category] = {}
        for name, dirname in exps.items():
            exp_path = base_path / dirname
            result = load_experiment_result(exp_path)
            results[category][name] = result

    return results

def print_results(results):
    """打印分析结果"""

    print("="*80)
    print("Pipeline NAS 综合实验结果分析")
    print("="*80)
    print()

    # 第一部分：基准组
    print("【第一部分：基准组】")
    print("-" * 80)
    print(f"{'实验':<20} {'选出架构':<20} {'Test MRR':<12} {'off/off最佳Val MRR':<20}")
    print("-" * 80)

    for name, result in results['baseline'].items():
        if result:
            arch_str = f"{result['selected_arch']['time_proj']}/{result['selected_arch']['use_static']}"
            off_off_val = result['off_off_best']['val_mrr'] if result['off_off_best'] else 'N/A'
            print(f"{name:<20} {arch_str:<20} {result['test_mrr']:<12.4f} {off_off_val}")
        else:
            print(f"{name:<20} {'FAILED':<20}")
    print()

    # 第二部分：Stage数量实验
    print("【第二部分：Stage数量实验】(固定Overlap=20%)")
    print("-" * 80)
    print(f"{'Stage配置':<20} {'选出架构':<20} {'Test MRR':<12} {'off/off最佳Val':<15} {'是否正确':<10}")
    print("-" * 80)

    for name, result in results['stages'].items():
        if result:
            arch_str = f"{result['selected_arch']['time_proj']}/{result['selected_arch']['use_static']}"
            off_off_val = f"{result['off_off_best']['val_mrr']:.4f}" if result['off_off_best'] else 'N/A'
            is_correct = "✅" if arch_str == "off/off" else "❌"
            print(f"{name:<20} {arch_str:<20} {result['test_mrr']:<12.4f} {off_off_val:<15} {is_correct:<10}")
        else:
            print(f"{name:<20} {'FAILED':<20}")
    print()

    # 第三部分：Overlap比例实验
    print("【第三部分：Overlap比例实验】(固定Stage=2)")
    print("-" * 80)
    print(f"{'Overlap配置':<20} {'选出架构':<20} {'Test MRR':<12} {'off/off最佳Val':<15} {'是否正确':<10}")
    print("-" * 80)

    for name, result in results['overlap'].items():
        if result:
            arch_str = f"{result['selected_arch']['time_proj']}/{result['selected_arch']['use_static']}"
            off_off_val = f"{result['off_off_best']['val_mrr']:.4f}" if result['off_off_best'] else 'N/A'
            is_correct = "✅" if arch_str == "off/off" else "❌"
            print(f"{name:<20} {arch_str:<20} {result['test_mrr']:<12.4f} {off_off_val:<15} {is_correct:<10}")
        else:
            print(f"{name:<20} {'FAILED':<20}")
    print()

    # 第四部分：交叉验证
    print("【第四部分：Stage×Overlap交叉验证】")
    print("-" * 80)
    print(f"{'配置':<20} {'选出架构':<20} {'Test MRR':<12} {'off/off最佳Val':<15} {'是否正确':<10}")
    print("-" * 80)

    for name, result in results['cross'].items():
        if result:
            arch_str = f"{result['selected_arch']['time_proj']}/{result['selected_arch']['use_static']}"
            off_off_val = f"{result['off_off_best']['val_mrr']:.4f}" if result['off_off_best'] else 'N/A'
            is_correct = "✅" if arch_str == "off/off" else "❌"
            print(f"{name:<20} {arch_str:<20} {result['test_mrr']:<12.4f} {off_off_val:<15} {is_correct:<10}")
        else:
            print(f"{name:<20} {'FAILED':<20}")
    print()

    # 第五部分：搜索模式对比
    print("【第五部分：搜索模式对比】")
    print("-" * 80)
    print(f"{'搜索模式':<20} {'选出架构':<20} {'Test MRR':<12} {'是否正确':<10}")
    print("-" * 80)

    for name, result in results['search_mode'].items():
        if result:
            arch_str = f"{result['selected_arch']['time_proj']}/{result['selected_arch']['use_static']}"
            is_correct = "✅" if arch_str == "off/off" else "❌"
            print(f"{name:<20} {arch_str:<20} {result['test_mrr']:<12.4f} {is_correct:<10}")
        else:
            print(f"{name:<20} {'FAILED':<20}")
    print()

    # 汇总分析
    print("="*80)
    print("【关键发现】")
    print("="*80)

    # 分析Stage数量的影响
    stage_correct = []
    for name, result in results['stages'].items():
        if result:
            arch_str = f"{result['selected_arch']['time_proj']}/{result['selected_arch']['use_static']}"
            if arch_str == "off/off":
                stage_correct.append(name)

    if stage_correct:
        print(f"✓ 正确选出off/off的Stage配置: {', '.join(stage_correct)}")
    else:
        print("✗ 所有Stage配置都未能正确选出off/off")

    # 分析Overlap的影响
    overlap_correct = []
    for name, result in results['overlap'].items():
        if result:
            arch_str = f"{result['selected_arch']['time_proj']}/{result['selected_arch']['use_static']}"
            if arch_str == "off/off":
                overlap_correct.append(name)

    if overlap_correct:
        print(f"✓ 正确选出off/off的Overlap配置: {', '.join(overlap_correct)}")
    else:
        print("✗ 所有Overlap配置都未能正确选出off/off")

    print()
    print("详细数据已保存到: outputs/comprehensive_experiment/analysis.json")

def save_results(results, output_path):
    """保存结果到JSON文件"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    base_dir = "outputs/comprehensive_experiment"

    if not Path(base_dir).exists():
        print(f"错误: 实验目录不存在: {base_dir}")
        print("请先运行: bash run_comprehensive_experiments.sh")
        sys.exit(1)

    print("正在分析实验结果...")
    results = analyze_experiments(base_dir)

    # 保存结果
    output_path = Path(base_dir) / "analysis.json"
    save_results(results, output_path)

    # 打印结果
    print_results(results)
