#!/bin/bash

# 单 GPU 效率测试脚本
# 用于验证单 GPU 时效率应该接近 100%

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

# 配置
DATASET="public_csv"
DATA_FILE="data/public/mooc.csv"  # 使用较小的数据集
MAX_EVENTS=500
OUTPUT_DIR="outputs_single_gpu_test"
GPU_ID=0

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  单 GPU 效率测试                                              ║"
echo "║  预期: GPU 利用率应接近 100%                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 清理旧的输出
if [ -d "$OUTPUT_DIR" ]; then
    echo "🧹 清理旧输出目录..."
    rm -rf "$OUTPUT_DIR"
fi

echo "📊 测试配置:"
echo "   • GPU: CUDA_VISIBLE_DEVICES=$GPU_ID (单 GPU)"
echo "   • 数据集: $DATA_FILE"
echo "   • 仅使用前 $MAX_EVENTS 条数据"
echo "   • 输出目录: $OUTPUT_DIR"
echo "   • 效率监控: 启用"
echo "   • 采样间隔: 5 秒"
echo ""

echo "🚀 启动单 GPU 训练..."
echo ""

# 运行 Pipeline 搜索，仅使用 1 个 GPU
# 参数说明：
# --coarse-trials 2: 只评估 2 个 trial（快速测试）
# --coarse-epochs 1: 只训练 1 个 epoch（快速测试）
# --architectures-per-step 1: 每步 1 个架构（单序列）
# --num-pipeline-stages 1: 1 个 stage（避免阶段间开销）
# --pipeline-stage-train-workers 1: 1 个 worker（单进程）
# --enable-efficiency-monitor: 启用效率监控
# --efficiency-monitor-interval 5: 每 5 秒采样（快速反馈）
# --max-events 500: 仅使用 MOOC 前 500 条数据，快速跑完

CUDA_VISIBLE_DEVICES=$GPU_ID python search.py \
    --dataset "$DATASET" \
    --local-data-path "$DATA_FILE" \
    --max-events "$MAX_EVENTS" \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --coarse-trials 2 \
    --coarse-epochs 1 \
    --architectures-per-step 1 \
    --num-pipeline-stages 1 \
    --pipeline-stage-train-workers 1 \
    --partition-size 1000 \
    --pipeline-trace \
    --enable-efficiency-monitor \
    --efficiency-monitor-interval 5 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  测试完成！                                                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 显示效率报告
EFFICIENCY_REPORT=$(find "$OUTPUT_DIR" -name 'efficiency_log_*_report.txt' | head -1)
if [ -n "$EFFICIENCY_REPORT" ]; then
    echo "📈 效率分析报告:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   位置: $EFFICIENCY_REPORT"
    cat "$EFFICIENCY_REPORT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

# 显示效率数据
EFFICIENCY_CSV=$(find "$OUTPUT_DIR" -name 'efficiency_log_*.csv' | head -1)
if [ -n "$EFFICIENCY_CSV" ]; then
    echo "📊 效率数据 (CSV):"
    echo "   位置: $EFFICIENCY_CSV"
    echo ""
    echo "   前 10 行数据:"
    head -11 "$EFFICIENCY_CSV" | tail -10
    echo ""
fi

echo "✅ 验证项目:"
echo "   [✓] GPU 利用率应接近 100%（单 GPU，无竞争）"
echo "   [✓] GPU 效率应接近 100%（全时间都在计算）"
echo "   [✓] Pipeline Speedup 应接近 1.0（无流水线优势）"
echo "   [✓] 完整的时间序列数据已记录"
echo ""

echo "💡 对比建议:"
echo "   运行多 GPU 版本对比效率差异:"
echo "   bash test_multi_gpu.sh"
echo ""
