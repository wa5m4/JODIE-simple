#!/bin/bash
# ============================================================
# NAS 对比实验：单 GPU 串行 vs Pipeline 并行
# 两种模式使用相同的数据集、trial 数、seed，保证公平对比
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# -------- 可调参数 --------
DATASET="public_csv"
DATA_FILE="data/public/mooc.csv"
MAX_EVENTS=0           # 0 = 使用全量数据；调小可加快测试，例如 5000
TRIALS=12              # 两种模式都搜索这么多个架构
EPOCHS=1               # 每个 trial 训练 epoch 数
SEED=42
K=10
METRIC="mrr"

# Pipeline 并行参数
ARCH_PER_STEP=3        # 每批并发评估的架构数
NUM_STAGES=3           # pipeline stage 数
WORKER_GPUS=0.33       # 每个 stage 占用的 GPU 比例（3 个 stage 共享 1 GPU）
PARTITION_SIZE=2000    # 每个 partition 的事件数

OUTPUT_SERIAL="outputs/compare_serial"
OUTPUT_PIPELINE="outputs/compare_pipeline"

# -------- 打印配置 --------
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         NAS Baseline Comparison: Serial vs Pipeline                 ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Dataset  : $DATA_FILE"
echo "║  MaxEvents: $MAX_EVENTS (0=all)"
echo "║  Trials   : $TRIALS  |  Epochs: $EPOCHS  |  Seed: $SEED"
echo "║  Metric   : $METRIC  |  K: $K"
echo "║  Pipeline : arch_per_step=$ARCH_PER_STEP  stages=$NUM_STAGES  gpu_per_worker=$WORKER_GPUS"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# -------- 清理旧输出 --------
for DIR in "$OUTPUT_SERIAL" "$OUTPUT_PIPELINE"; do
    if [ -d "$DIR" ]; then
        echo "🧹 清理旧输出: $DIR"
        rm -rf "$DIR"
    fi
done
mkdir -p "$OUTPUT_SERIAL" "$OUTPUT_PIPELINE"

# ============================================================
# Step 1: 单 GPU 串行 Baseline
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶  [1/2] 单 GPU 串行搜索 (Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

SERIAL_START=$(date +%s%N)

python search.py \
    --dataset "$DATASET" \
    --local-data-path "$DATA_FILE" \
    --max-events "$MAX_EVENTS" \
    --search-mode rl \
    --execution-mode serial \
    --trials "$TRIALS" \
    --epochs-per-trial "$EPOCHS" \
    --seed "$SEED" \
    --k "$K" \
    --selection-metric "$METRIC" \
    --output-dir "$OUTPUT_SERIAL"

SERIAL_END=$(date +%s%N)
SERIAL_SEC=$(( (SERIAL_END - SERIAL_START) / 1000000000 ))

echo ""
echo "✅ 串行搜索完成，耗时: ${SERIAL_SEC}s"
echo ""

# ============================================================
# Step 2: Pipeline 并行
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶  [2/2] Pipeline 并行搜索 (Our Method)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PIPELINE_START=$(date +%s%N)

python search.py \
    --dataset "$DATASET" \
    --local-data-path "$DATA_FILE" \
    --max-events "$MAX_EVENTS" \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --trials "$TRIALS" \
    --epochs-per-trial "$EPOCHS" \
    --architectures-per-step "$ARCH_PER_STEP" \
    --num-pipeline-stages "$NUM_STAGES" \
    --pipeline-worker-gpus "$WORKER_GPUS" \
    --partition-size "$PARTITION_SIZE" \
    --stage-balance-strategy cost \
    --seed "$SEED" \
    --k "$K" \
    --selection-metric "$METRIC" \
    --pipeline-trace \
    --enable-efficiency-monitor \
    --efficiency-monitor-interval 10 \
    --output-dir "$OUTPUT_PIPELINE"

PIPELINE_END=$(date +%s%N)
PIPELINE_SEC=$(( (PIPELINE_END - PIPELINE_START) / 1000000000 ))

echo ""
echo "✅ Pipeline 搜索完成，耗时: ${PIPELINE_SEC}s"
echo ""

# ============================================================
# Step 3: 生成对比报告
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶  [3/3] 生成对比报告"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python tools/compare_results.py \
    --serial-dir "$OUTPUT_SERIAL" \
    --pipeline-dir "$OUTPUT_PIPELINE" \
    --serial-time "$SERIAL_SEC" \
    --pipeline-time "$PIPELINE_SEC" \
    --trials "$TRIALS"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  对比实验完成！                                                       ║"
echo "║  串行耗时  : ${SERIAL_SEC}s"
echo "║  Pipeline 耗时: ${PIPELINE_SEC}s"
if [ "$PIPELINE_SEC" -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $SERIAL_SEC / $PIPELINE_SEC" | bc 2>/dev/null || echo "N/A")
    echo "║  搜索加速比: ${SPEEDUP}x"
fi
echo "║  详细报告  : outputs/compare_report.txt"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
