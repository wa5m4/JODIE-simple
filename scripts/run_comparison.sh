#!/bin/bash
# ============================================================
# NAS 多种子对比实验：单 GPU 串行 vs Pipeline 并行
# 策略：固定时间预算 × 多种子，评估搜索质量的均值和方差
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# -------- 可调参数 --------
DATASET="public_csv"
DATA_FILE="data/public/mooc.csv"
MAX_EVENTS=20000       # 5% MOOC，时序特征真实，单 trial ~50s
SERIAL_TRIALS=15       # 串行：15 trials/seed
PIPELINE_TRIALS=30     # Pipeline：30 trials/seed（2x 吞吐）
EPOCHS=3               # 3 epoch/trial，评估更稳定
SEEDS=(42 43 44)       # 3 个种子，用于 mean±std 统计
K=10
METRIC="mrr"

# Pipeline 并行参数
ARCH_PER_STEP=3
NUM_STAGES=3
WORKER_GPUS=0.33
PARTITION_SIZE=500

# 输出根目录（含数据集标识）
DATASET_TAG="mooc_20k"
OUTPUT_ROOT="outputs/${DATASET_TAG}_multiseed"

# -------- 打印配置 --------
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         NAS Multi-Seed Comparison: Serial vs Pipeline               ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Dataset  : $DATA_FILE  (${MAX_EVENTS} events)"
echo "║  Epochs   : $EPOCHS/trial  |  Metric: $METRIC  |  K: $K"
echo "║  Seeds    : ${SEEDS[*]}  ($(( ${#SEEDS[@]} )) runs each)"
echo "║  Serial   : $SERIAL_TRIALS trials/seed"
echo "║  Pipeline : $PIPELINE_TRIALS trials/seed  (expect 2x throughput)"
echo "║  Output   : $OUTPUT_ROOT/"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$OUTPUT_ROOT"
# 清理上次留下的计时记录，避免追加到旧数据
rm -f "${OUTPUT_ROOT}/seed_times.csv"

# ============================================================
# 对每个 seed 运行完整对比实验
# ============================================================
SEED_IDX=0
for SEED in "${SEEDS[@]}"; do
    SEED_IDX=$(( SEED_IDX + 1 ))
    SEED_DIR="${OUTPUT_ROOT}/seed_${SEED}"
    OUTPUT_SERIAL="${SEED_DIR}/serial"
    OUTPUT_PIPELINE="${SEED_DIR}/pipeline"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶  Seed ${SEED_IDX}/${#SEEDS[@]}  (seed=${SEED})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for DIR in "$OUTPUT_SERIAL" "$OUTPUT_PIPELINE"; do
        [ -d "$DIR" ] && rm -rf "$DIR"
    done
    mkdir -p "$OUTPUT_SERIAL" "$OUTPUT_PIPELINE"

    # ── [1/3] 串行 Baseline
    echo ""
    echo "  [1/3] 串行搜索 (Baseline)  seed=${SEED}  ${SERIAL_TRIALS} trials"
    SERIAL_START=$(date +%s%N)

    python search.py \
        --dataset         "$DATASET" \
        --local-data-path "$DATA_FILE" \
        --max-events      "$MAX_EVENTS" \
        --search-mode     rl \
        --execution-mode  serial \
        --trials          "$SERIAL_TRIALS" \
        --epochs-per-trial "$EPOCHS" \
        --seed            "$SEED" \
        --k               "$K" \
        --selection-metric "$METRIC" \
        --output-dir      "$OUTPUT_SERIAL"

    SERIAL_END=$(date +%s%N)
    SERIAL_SEC=$(( (SERIAL_END - SERIAL_START) / 1000000000 ))
    echo "  ✅ 串行完成  ${SERIAL_SEC}s"

    # ── [2/3] Pipeline
    echo ""
    echo "  [2/3] Pipeline 搜索 (Ours)  seed=${SEED}  ${PIPELINE_TRIALS} trials"
    PIPELINE_START=$(date +%s%N)

    python search.py \
        --dataset               "$DATASET" \
        --local-data-path       "$DATA_FILE" \
        --max-events            "$MAX_EVENTS" \
        --search-mode           rl \
        --execution-mode        ray_pipeline \
        --trials                "$PIPELINE_TRIALS" \
        --epochs-per-trial      "$EPOCHS" \
        --architectures-per-step "$ARCH_PER_STEP" \
        --num-pipeline-stages   "$NUM_STAGES" \
        --pipeline-worker-gpus  "$WORKER_GPUS" \
        --partition-size        "$PARTITION_SIZE" \
        --stage-balance-strategy cost \
        --seed                  "$SEED" \
        --k                     "$K" \
        --selection-metric      "$METRIC" \
        --pipeline-trace \
        --enable-efficiency-monitor \
        --efficiency-monitor-interval 10 \
        --output-dir            "$OUTPUT_PIPELINE"

    PIPELINE_END=$(date +%s%N)
    PIPELINE_SEC=$(( (PIPELINE_END - PIPELINE_START) / 1000000000 ))
    echo "  ✅ Pipeline 完成  ${PIPELINE_SEC}s"

    # ── [3/3] 单 seed 详细报告
    echo ""
    echo "  [3/3] 生成 seed=${SEED} 详细报告"
    python tools/compare_results.py \
        --serial-dir      "$OUTPUT_SERIAL" \
        --pipeline-dir    "$OUTPUT_PIPELINE" \
        --serial-time     "$SERIAL_SEC" \
        --pipeline-time   "$PIPELINE_SEC" \
        --serial-trials   "$SERIAL_TRIALS" \
        --pipeline-trials "$PIPELINE_TRIALS" \
        --output          "${SEED_DIR}/report.txt"

    # 记录本 seed 计时，供汇总脚本读取
    echo "${SEED},${SERIAL_SEC},${PIPELINE_SEC}" >> "${OUTPUT_ROOT}/seed_times.csv"

    echo ""
    echo "  seed=${SEED} 完成  Serial ${SERIAL_SEC}s  Pipeline ${PIPELINE_SEC}s"
done

# ============================================================
# 多种子汇总报告
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶  生成多种子汇总报告"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python tools/aggregate_seeds.py \
    --root            "$OUTPUT_ROOT" \
    --seeds           "${SEEDS[*]}" \
    --serial-trials   "$SERIAL_TRIALS" \
    --pipeline-trials "$PIPELINE_TRIALS" \
    --output          "${OUTPUT_ROOT}/aggregate_report.txt"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  多种子对比实验完成！                                                 ║"
echo "║  Seeds   : ${SEEDS[*]}"
echo "║  Results : $OUTPUT_ROOT/"
echo "║  Summary : ${OUTPUT_ROOT}/aggregate_report.txt"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
