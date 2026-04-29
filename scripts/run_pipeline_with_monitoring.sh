#!/bin/bash
# 
# 启动 Pipeline 并同时监控效率的脚本
# 用法: bash run_pipeline_with_monitoring.sh [output_dir] [interval_sec]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

OUTPUT_DIR="${1:-outputs_pipeline_monitored}"
INTERVAL="${2:-10}"

echo "🚀 启动 Pipeline 和效率监控"
echo "📁 输出目录: $OUTPUT_DIR"
echo "⏱️  监控间隔: ${INTERVAL}s"
echo "=============================================="

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 启动 Pipeline（后台运行）
echo "▶️  启动 Pipeline..."
CUDA_VISIBLE_DEVICES=0,1,2 python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --search-mode rl \
  --execution-mode ray_pipeline \
  --coarse-trials 6 \
  --coarse-epochs 1 \
  --rerank-top-k 3 \
  --rerank-epochs 3 \
  --architectures-per-step 6 \
  --partition-size 2000 \
  --num-pipeline-stages 2 \
  --pipeline-worker-gpus 1 \
  --pipeline-worker-cpus 2 \
  --pipeline-stage-train-workers 3,2 \
  --pipeline-stage-eval-workers 1 \
  --stage-balance-strategy cost \
  --stage-balance-user-weight 0.25 \
  --stage-balance-item-weight 0.25 \
  --max-events 20000 \
  --lr 3e-4 \
  --k 10 \
  --seed 42 \
  --pipeline-trace \
  --output-dir "$OUTPUT_DIR" &

PIPELINE_PID=$!
echo "✅ Pipeline 启动 (PID: $PIPELINE_PID)"

# 等待 trace 文件生成
echo "⏳ 等待 trace 文件生成..."
TRACE_FILE=""
for i in {1..30}; do
    sleep 1
    TRACE_FILE=$(find "$OUTPUT_DIR" -name "pipeline_trace_*.log" -type f 2>/dev/null | head -1)
    if [ -n "$TRACE_FILE" ]; then
        echo "✅ Trace 文件已生成: $TRACE_FILE"
        break
    fi
    echo "   尝试 $i/30..."
done

if [ -z "$TRACE_FILE" ]; then
    echo "❌ Trace 文件生成超时"
    kill $PIPELINE_PID 2>/dev/null || true
    exit 1
fi

# 启动效率监控
echo ""
echo "▶️  启动效率监控..."
python tools/monitor_pipeline_efficiency.py "$TRACE_FILE" "$INTERVAL" &

MONITOR_PID=$!
echo "✅ 效率监控启动 (PID: $MONITOR_PID)"

echo ""
echo "🔄 Pipeline 和监控正在运行..."
echo "   Pipeline PID: $PIPELINE_PID"
echo "   Monitor PID:  $MONITOR_PID"
echo ""
echo "按 Ctrl+C 停止监控（Pipeline 会继续运行）"
echo "=============================================="

# 等待 Pipeline 完成或用户中断
wait $PIPELINE_PID
PIPELINE_EXIT=$?

echo ""
echo "🛑 Pipeline 已完成 (exit code: $PIPELINE_EXIT)"

# 停止监控
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

# 查找生成的效率日志
EFFICIENCY_LOG=$(find "$OUTPUT_DIR" -name "efficiency_log_*.csv" -type f 2>/dev/null | head -1)

if [ -n "$EFFICIENCY_LOG" ]; then
    echo ""
    echo "📊 生成效率分析报告..."
    python tools/visualize_efficiency_log.py "$EFFICIENCY_LOG" --limit 30
    
    # 导出摘要
    SUMMARY_FILE="$OUTPUT_DIR/efficiency_summary.txt"
    python tools/visualize_efficiency_log.py "$EFFICIENCY_LOG" --export "$SUMMARY_FILE"
else
    echo "⚠️  未找到效率日志文件"
fi

echo ""
echo "✅ 完成！"
echo ""
