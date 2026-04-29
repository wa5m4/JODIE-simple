#!/bin/bash

# 多 GPU 效率测试脚本
# 用于与单 GPU 结果对比，验证多 GPU 效率提升

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

# 配置
DATASET="public_csv"
DATA_FILE="data/public/mooc.csv"
MAX_EVENTS=500
OUTPUT_DIR="outputs_multi_gpu_test"
GPUS="0,1,2"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  多 GPU 效率测试（对比测试）                                  ║"
echo "║  预期: GPU 利用率 < 100%（调度竞争导致）                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 清理旧的输出
if [ -d "$OUTPUT_DIR" ]; then
    echo "🧹 清理旧输出目录..."
    rm -rf "$OUTPUT_DIR"
fi

echo "📊 测试配置:"
echo "   • GPU: CUDA_VISIBLE_DEVICES=$GPUS (3 个 GPU)"
echo "   • 数据集: $DATA_FILE"
echo "   • 仅使用前 $MAX_EVENTS 条数据"
echo "   • 输出目录: $OUTPUT_DIR"
echo "   • 效率监控: 启用"
echo "   • 采样间隔: 5 秒"
echo ""

echo "🚀 启动多 GPU 训练..."
echo ""

# 运行 Pipeline 搜索，使用 3 个 GPU
# 与单 GPU 版本使用相同的数据/trial 数量，但启用流水线并行
# --max-events 500: 仅使用 MOOC 前 500 条数据，便于快速对比
CUDA_VISIBLE_DEVICES=$GPUS python search.py \
    --dataset "$DATASET" \
    --local-data-path "$DATA_FILE" \
    --max-events "$MAX_EVENTS" \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --coarse-trials 2 \
    --coarse-epochs 1 \
    --architectures-per-step 3 \
    --num-pipeline-stages 2 \
    --pipeline-stage-train-workers 2,1 \
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
echo "   [?] GPU 利用率 (预期 < 单 GPU 版本)"
echo "   [?] GPU 效率 (预期 < 100%，因为调度开销)"
echo "   [?] Pipeline Speedup (预期 > 1.0)"
echo "   [?] 与单 GPU 版本对比"
echo ""
