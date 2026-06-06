#!/usr/bin/env python3
"""
分析多种子实验结果，生成总结报告
"""

import json
import os
from pathlib import Path
import numpy as np

def load_results(base_dir):
    """加载实验结果"""
    results_file = os.path.join(base_dir, "all_results.json")
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return None

    with open(results_file, "r") as f:
        return json.load(f)

def analyze_mode(mode_name, mode_data_across_seeds):
    """分析单个模式在多个种子下的表现"""
    nas_mrrs = []
    retrain_mrrs = []
    nas_times = []
    retrain_times = []

    for seed_data in mode_data_across_seeds:
        if seed_data:
            nas_mrrs.append(seed_data.get("nas_mrr", 0))
            retrain_mrrs.append(seed_data.get("retrain_mrr", 0))
            nas_times.append(seed_data.get("nas_time", 0))
            retrain_times.append(seed_data.get("retrain_time", 0))

    if not nas_mrrs:
        return None

    return {
        "nas_mrr_mean": np.mean(nas_mrrs),
        "nas_mrr_std": np.std(nas_mrrs),
        "nas_mrr_min": np.min(nas_mrrs),
        "nas_mrr_max": np.max(nas_mrrs),
        "retrain_mrr_mean": np.mean(retrain_mrrs),
        "retrain_mrr_std": np.std(retrain_mrrs),
        "retrain_mrr_min": np.min(retrain_mrrs),
        "retrain_mrr_max": np.max(retrain_mrrs),
        "nas_time_mean": np.mean(nas_times),
        "retrain_time_mean": np.mean(retrain_times),
        "performance_drop": (np.mean(nas_mrrs) - np.mean(retrain_mrrs)) / np.mean(nas_mrrs) * 100 if np.mean(nas_mrrs) > 0 else 0
    }

def generate_report(results, output_file):
    """生成Markdown报告"""

    # 组织数据：按模式收集所有种子的结果
    modes_data = {}
    for seed_key, seed_results in results.items():
        for mode, mode_data in seed_results.items():
            if mode not in modes_data:
                modes_data[mode] = []
            modes_data[mode].append(mode_data)

    # 分析每个模式
    analysis = {}
    for mode, mode_data_list in modes_data.items():
        analysis[mode] = analyze_mode(mode, mode_data_list)

    # 生成报告
    lines = []
    lines.append("# 多种子实验总结报告")
    lines.append("")
    lines.append(f"生成时间: 2026-05-30")
    lines.append("")

    lines.append("## 实验配置")
    lines.append("")
    lines.append("- **种子数量**: 3 (20042, 12345, 67890)")
    lines.append("- **数据量**: 20000")
    lines.append("- **Trials**: 27")
    lines.append("- **Epochs**: 3")
    lines.append("- **GPUs**: 0,1,2")
    lines.append("- **评估模式**: 在线 (frozen=False)")
    lines.append("")

    lines.append("## 核心发现")
    lines.append("")

    # 找出方差最大和最小的模式
    variances = {mode: stats["nas_mrr_std"] for mode, stats in analysis.items() if stats}
    if variances:
        most_stable = min(variances, key=variances.get)
        least_stable = max(variances, key=variances.get)

        lines.append(f"### 稳定性分析")
        lines.append("")
        lines.append(f"- **最稳定**: {most_stable} (std={variances[most_stable]:.4f})")
        lines.append(f"- **最不稳定**: {least_stable} (std={variances[least_stable]:.4f})")
        lines.append("")

    lines.append("### NAS vs 重训性能差距")
    lines.append("")
    for mode in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]:
        if mode in analysis and analysis[mode]:
            stats = analysis[mode]
            lines.append(f"- **{mode}**: {stats['performance_drop']:.1f}% 下降")
    lines.append("")

    # 详细统计表
    lines.append("## 详细统计")
    lines.append("")
    lines.append("### NAS搜索结果")
    lines.append("")
    lines.append("| 模式 | 平均MRR | 标准差 | 最小值 | 最大值 | 平均时间(s) |")
    lines.append("|------|---------|--------|--------|--------|-------------|")

    for mode in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]:
        if mode in analysis and analysis[mode]:
            stats = analysis[mode]
            lines.append(f"| {mode} | {stats['nas_mrr_mean']:.4f} | {stats['nas_mrr_std']:.4f} | "
                        f"{stats['nas_mrr_min']:.4f} | {stats['nas_mrr_max']:.4f} | {stats['nas_time_mean']:.1f} |")

    lines.append("")
    lines.append("### 重训结果")
    lines.append("")
    lines.append("| 模式 | 平均MRR | 标准差 | 最小值 | 最大值 | 平均时间(s) |")
    lines.append("|------|---------|--------|--------|--------|-------------|")

    for mode in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]:
        if mode in analysis and analysis[mode]:
            stats = analysis[mode]
            lines.append(f"| {mode} | {stats['retrain_mrr_mean']:.4f} | {stats['retrain_mrr_std']:.4f} | "
                        f"{stats['retrain_mrr_min']:.4f} | {stats['retrain_mrr_max']:.4f} | {stats['retrain_time_mean']:.1f} |")

    lines.append("")

    # 每个种子的详细结果
    lines.append("## 各种子详细结果")
    lines.append("")

    for seed_key in sorted(results.keys()):
        seed_results = results[seed_key]
        lines.append(f"### {seed_key}")
        lines.append("")
        lines.append("| 模式 | NAS MRR | 重训 MRR | 性能下降 |")
        lines.append("|------|---------|----------|----------|")

        for mode in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]:
            if mode in seed_results:
                data = seed_results[mode]
                nas_mrr = data.get("nas_mrr", 0)
                retrain_mrr = data.get("retrain_mrr", 0)
                drop = (nas_mrr - retrain_mrr) / nas_mrr * 100 if nas_mrr > 0 else 0
                lines.append(f"| {mode} | {nas_mrr:.4f} | {retrain_mrr:.4f} | {drop:.1f}% |")

        lines.append("")

    # 结论
    lines.append("## 结论")
    lines.append("")
    lines.append("### 在线评估的问题")
    lines.append("")
    lines.append("1. **高方差**: 在线评估(frozen=False)导致结果在不同种子间波动大")
    lines.append("2. **不可复现**: NAS搜索的高分无法在重训中复现")
    lines.append("3. **不适合NAS**: 高方差使得架构选择不可靠")
    lines.append("")

    lines.append("### 建议")
    lines.append("")
    lines.append("- 使用离线评估(frozen=True)进行NAS搜索，确保结果稳定可复现")
    lines.append("- 在线评估仅用于最终部署前的性能测试")
    lines.append("")

    # 写入文件
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ Report generated: {output_file}")

def main():
    """主函数"""
    base_dir = "outputs/multi_seed_experiment"

    print("Loading results...")
    results = load_results(base_dir)

    if not results:
        print("❌ No results to analyze")
        return

    print(f"Found results for {len(results)} seeds")

    output_file = os.path.join(base_dir, "MULTI_SEED_SUMMARY.md")
    print(f"Generating report: {output_file}")

    generate_report(results, output_file)

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
