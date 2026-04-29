#!/bin/bash

# 效率对比脚本
# 对比单 GPU vs 多 GPU 的效率数据

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  效率对比分析                                                  ║"
echo "║  单 GPU vs 多 GPU                                              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

SINGLE_CSV="outputs_single_gpu_test/efficiency_log_"*.csv
MULTI_CSV="outputs_multi_gpu_test/efficiency_log_"*.csv

echo "📁 数据源:"
echo "   单 GPU: $SINGLE_CSV"
echo "   多 GPU: $MULTI_CSV"
echo ""

# 检查文件是否存在
if [ ! -f $(echo $SINGLE_CSV | sed 's/\*/[^/]*/' | cut -d' ' -f1) ]; then
    echo "❌ 错误: 单 GPU 数据文件不存在"
    echo "   请先运行: bash test_single_gpu.sh"
    exit 1
fi

if [ ! -f $(echo $MULTI_CSV | sed 's/\*/[^/]*/' | cut -d' ' -f1) ]; then
    echo "❌ 错误: 多 GPU 数据文件不存在"
    echo "   请先运行: bash test_multi_gpu.sh"
    exit 1
fi

echo "✅ 文件已找到，开始对比分析..."
echo ""

# 提取单 GPU 数据
python3 << 'EOF'
import csv
import os
import glob

# 寻找 CSV 文件
single_files = sorted(glob.glob("outputs_single_gpu_test/efficiency_log_*.csv"))
multi_files = sorted(glob.glob("outputs_multi_gpu_test/efficiency_log_*.csv"))

def analyze_csv(filepath):
    """分析 CSV 文件，提取关键指标"""
    metrics = {
        'gpu_util': [],
        'gpu_efficiency': [],
        'speedup': [],
        'throughput': []
    }
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    metrics['gpu_util'].append(float(row.get('gpu_util_ratio', 0)))
                    metrics['gpu_efficiency'].append(float(row.get('gpu_efficiency', 0)))
                    metrics['speedup'].append(float(row.get('pipeline_speedup', 0)))
                    metrics['throughput'].append(float(row.get('trial_throughput', 0)))
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        print(f"   ⚠️  读取文件出错: {e}")
        return None
    
    if not metrics['gpu_util']:
        return None
    
    return metrics

def print_stats(label, metrics):
    """打印统计信息"""
    if not metrics:
        print(f"   ❌ {label}: 数据为空")
        return
    
    print(f"\n   {label}:")
    print(f"      GPU 利用率: avg={sum(metrics['gpu_util'])/len(metrics['gpu_util']):.1%}, "
          f"min={min(metrics['gpu_util']):.1%}, max={max(metrics['gpu_util']):.1%}")
    print(f"      GPU 效率:   avg={sum(metrics['gpu_efficiency'])/len(metrics['gpu_efficiency']):.1%}, "
          f"min={min(metrics['gpu_efficiency']):.1%}, max={max(metrics['gpu_efficiency']):.1%}")
    if any(metrics['speedup']):
        print(f"      Speedup:   avg={sum(metrics['speedup'])/len(metrics['speedup']):.2f}x, "
              f"min={min(metrics['speedup']):.2f}x, max={max(metrics['speedup']):.2f}x")
    if any(metrics['throughput']):
        print(f"      吞吐量:    avg={sum(metrics['throughput'])/len(metrics['throughput']):.3f} trials/s")

print("📊 分析结果:")
print("=" * 60)

single_metrics = None
multi_metrics = None

if single_files:
    single_metrics = analyze_csv(single_files[-1])  # 使用最新的文件
    print_stats("单 GPU", single_metrics)

if multi_files:
    multi_metrics = analyze_csv(multi_files[-1])
    print_stats("多 GPU", multi_metrics)

# 对比分析
if single_metrics and multi_metrics:
    print("\n" + "=" * 60)
    print("\n📈 对比分析:")
    
    single_util = sum(single_metrics['gpu_util']) / len(single_metrics['gpu_util'])
    multi_util = sum(multi_metrics['gpu_util']) / len(multi_metrics['gpu_util'])
    
    single_eff = sum(single_metrics['gpu_efficiency']) / len(single_metrics['gpu_efficiency'])
    multi_eff = sum(multi_metrics['gpu_efficiency']) / len(multi_metrics['gpu_efficiency'])
    
    print(f"\n   GPU 利用率对比:")
    print(f"      单 GPU:  {single_util:.1%}")
    print(f"      多 GPU:  {multi_util:.1%}")
    if single_util > 0:
        ratio = multi_util / single_util
        print(f"      比例:   {ratio:.2f}x (预期 < 1.0)")
    
    print(f"\n   GPU 效率对比:")
    print(f"      单 GPU:  {single_eff:.1%}")
    print(f"      多 GPU:  {multi_eff:.1%}")
    
    print(f"\n   预期观察:")
    print(f"      ✓ 单 GPU 利用率应接近 100%")
    print(f"      ✓ 多 GPU 利用率应低于单 GPU（调度竞争）")
    print(f"      ✓ 多 GPU 可能有更高的 Speedup（并行收益）")

print("\n" + "=" * 60)
EOF

echo ""
echo "💡 建议:"
echo "   1. 查看完整的 CSV 数据: cat outputs_*/efficiency_log_*.csv"
echo "   2. 查看详细报告: cat outputs_*/efficiency_log_*_report.txt"
echo "   3. 根据结果调整 --pipeline-stage-train-workers 参数"
echo ""
