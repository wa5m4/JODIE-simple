#!/usr/bin/env python3
"""
Efficiency Log Visualization Tool

可视化实时效率监控日志，显示趋势和统计
"""

import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def read_efficiency_log(log_file: str) -> List[Dict]:
    """读取效率日志 CSV 文件"""
    
    rows = []
    try:
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换数值类型
                row_converted = {
                    'timestamp': row['timestamp'],
                    'elapsed_time_s': float(row['elapsed_time_s']),
                    'wall_time': float(row['wall_time']),
                    'avg_concurrent_gpus': float(row['avg_concurrent_gpus']),
                    'gpu_util_ratio': float(row['gpu_util_ratio']),
                    'gpu_efficiency': float(row['gpu_efficiency']),
                    'avg_stage_util': float(row['avg_stage_util']),
                    'max_stage_util': float(row['max_stage_util']),
                    'trial_throughput': float(row['trial_throughput']),
                    'pipeline_speedup': float(row['pipeline_speedup']),
                    'speedup_efficiency': float(row['speedup_efficiency']),
                    'num_completed_tasks': int(row['num_completed_tasks']),
                }
                rows.append(row_converted)
    except FileNotFoundError:
        print(f"❌ 日志文件不存在: {log_file}")
        return []
    
    return rows


def print_summary(rows: List[Dict]):
    """打印摘要统计"""
    
    if not rows:
        print("❌ 没有数据")
        return
    
    print("\n" + "="*80)
    print("效率监控日志汇总")
    print("="*80 + "\n")
    
    print(f"📊 数据点数: {len(rows)}")
    print(f"⏱️  监控周期: {rows[0]['timestamp']} ~ {rows[-1]['timestamp']}")
    print(f"   总耗时: {rows[-1]['elapsed_time_s']:.2f}s")
    
    # 提取各项指标
    gpu_utils = [r['gpu_util_ratio'] for r in rows]
    gpu_effs = [r['gpu_efficiency'] for r in rows]
    stage_utils = [r['avg_stage_util'] for r in rows]
    throughputs = [r['trial_throughput'] for r in rows]
    speedups = [r['pipeline_speedup'] for r in rows]
    speedup_effs = [r['speedup_efficiency'] for r in rows]
    
    print(f"\n📈 关键指标统计:")
    print(f"   GPU 利用率:        {min(gpu_utils):.1%} ~ {max(gpu_utils):.1%} (平均 {sum(gpu_utils)/len(gpu_utils):.1%})")
    print(f"   GPU 效率:          {min(gpu_effs):.1%} ~ {max(gpu_effs):.1%} (平均 {sum(gpu_effs)/len(gpu_effs):.1%})")
    print(f"   Stage 平均利用率:  {min(stage_utils):.1%} ~ {max(stage_utils):.1%} (平均 {sum(stage_utils)/len(stage_utils):.1%})")
    print(f"   Stage 最大利用率:  {sum(r['max_stage_util'] for r in rows)/len(rows):.1%}")
    print(f"   Trial 吞吐量:      {min(throughputs):.3f} ~ {max(throughputs):.3f} trials/sec (平均 {sum(throughputs)/len(throughputs):.3f})")
    print(f"   Pipeline Speedup:  {min(speedups):.2f}x ~ {max(speedups):.2f}x (平均 {sum(speedups)/len(speedups):.2f}x)")
    print(f"   Speedup 效率:      {min(speedup_effs):.1%} ~ {max(speedup_effs):.1%} (平均 {sum(speedup_effs)/len(speedup_effs):.1%})")


def print_timeline(rows: List[Dict]):
    """打印时间线图表"""
    
    if not rows:
        return
    
    print(f"\n{'='*80}")
    print("时间线走势图")
    print(f"{'='*80}\n")
    
    # GPU 利用率走势
    print("📊 GPU 利用率趋势:")
    print("-" * 80)
    for i, row in enumerate(rows):
        ratio = row['gpu_util_ratio']
        bar_len = int(ratio * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {i+1:3d}. [{bar}] {ratio:.1%}")
    
    # Pipeline Speedup 走势
    print("\n📊 Pipeline Speedup 走势:")
    print("-" * 80)
    max_speedup = max(r['pipeline_speedup'] for r in rows)
    for i, row in enumerate(rows):
        speedup = row['pipeline_speedup']
        bar_len = int((speedup / max_speedup) * 30)
        bar = "█" * bar_len
        print(f"  {i+1:3d}. [{bar:<30}] {speedup:.2f}x")
    
    # Stage 利用率走势
    print("\n📊 Stage 平均利用率走势:")
    print("-" * 80)
    for i, row in enumerate(rows):
        util = row['avg_stage_util']
        bar_len = int(util * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {i+1:3d}. [{bar}] {util:.1%}")


def print_detailed_table(rows: List[Dict], limit: int = None):
    """打印详细表格"""
    
    if not rows:
        return
    
    print(f"\n{'='*80}")
    print("详细数据表")
    print(f"{'='*80}\n")
    
    # 表头
    print(f"{'#':<4} {'GPU利用率':<12} {'GPU效率':<12} {'Speedup':<12} {'Speedup效率':<12} {'完成任务':<12}")
    print("-" * 80)
    
    rows_to_show = rows if limit is None else rows[-limit:]
    
    for i, row in enumerate(rows_to_show):
        print(f"{i+1:<4} "
             f"{row['gpu_util_ratio']:>10.1%}  "
             f"{row['gpu_efficiency']:>10.1%}  "
             f"{row['pipeline_speedup']:>10.2f}x "
             f"{row['speedup_efficiency']:>10.1%}  "
             f"{row['num_completed_tasks']:>10}")


def export_to_summary_file(rows: List[Dict], output_file: str):
    """导出摘要到文件"""
    
    if not rows:
        return
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Pipeline 效率监控最终报告\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"监控周期: {rows[0]['timestamp']} ~ {rows[-1]['timestamp']}\n")
        f.write(f"数据点数: {len(rows)}\n")
        f.write(f"总监控时长: {rows[-1]['elapsed_time_s']:.2f}s\n\n")
        
        gpu_utils = [r['gpu_util_ratio'] for r in rows]
        gpu_effs = [r['gpu_efficiency'] for r in rows]
        speedups = [r['pipeline_speedup'] for r in rows]
        speedup_effs = [r['speedup_efficiency'] for r in rows]
        
        f.write("关键指标最终统计:\n")
        f.write("-" * 80 + "\n")
        f.write(f"GPU 利用率:     {sum(gpu_utils)/len(gpu_utils):.1%} (最小: {min(gpu_utils):.1%}, 最大: {max(gpu_utils):.1%})\n")
        f.write(f"GPU 效率:       {sum(gpu_effs)/len(gpu_effs):.1%} (最小: {min(gpu_effs):.1%}, 最大: {max(gpu_effs):.1%})\n")
        f.write(f"Pipeline Speedup: {sum(speedups)/len(speedups):.2f}x (最小: {min(speedups):.2f}x, 最大: {max(speedups):.2f}x)\n")
        f.write(f"Speedup 效率:   {sum(speedup_effs)/len(speedup_effs):.1%} (最小: {min(speedup_effs):.1%}, 最大: {max(speedup_effs):.1%})\n")
        f.write(f"完成任务数:     {rows[-1]['num_completed_tasks']}\n")
    
    print(f"\n✅ 摘要已导出到: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("用法: python visualize_efficiency_log.py <efficiency_log_file> [--limit N] [--export FILE]")
        print("例子: python visualize_efficiency_log.py outputs/efficiency_log_*.csv --limit 20")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    # 解析参数
    limit = None
    export_file = None
    
    if '--limit' in sys.argv:
        idx = sys.argv.index('--limit')
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])
    
    if '--export' in sys.argv:
        idx = sys.argv.index('--export')
        if idx + 1 < len(sys.argv):
            export_file = sys.argv[idx + 1]
    
    print(f"📂 读取日志文件: {log_file}")
    rows = read_efficiency_log(log_file)
    
    if rows:
        print_summary(rows)
        print_timeline(rows)
        print_detailed_table(rows, limit=limit)
        
        if export_file:
            export_to_summary_file(rows, export_file)


if __name__ == '__main__':
    main()
