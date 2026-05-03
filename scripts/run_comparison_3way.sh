#!/bin/bash
# ============================================================
# NAS 三方对比：Serial(单GPU) vs Data-Parallel vs Pipeline
#
# 核心论点：
#   - Serial：单 GPU 串行，15 trials/seed（基准）
#   - Data-Parallel：同一架构内部数据并行（3 worker），
#     每 trial 约 1/3 时间，同等时间内仍评估 ~15 个架构
#   - Pipeline：架构级并行（3 stage），同等时间内评估 ~30 个架构
#
# 结论：NAS 瓶颈是"搜索多少架构"，而非"每个架构训多快"
#       → Pipeline 的架构级并行比数据并行更适合 NAS
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# -------- 可调参数 --------
DATASET="public_csv"
DATA_FILE="data/public/mooc.csv"
MAX_EVENTS=20000
SERIAL_TRIALS=15
DP_TRIALS=15           # Data-Parallel：相同架构数量（加速了单 trial 但不增加架构数）
PIPELINE_TRIALS=30     # Pipeline：同等时间内 2x 架构覆盖
EPOCHS=3
SEEDS=(42 43 44)
K=10
METRIC="mrr"

# Data-Parallel 参数
DP_WORKERS=3
DP_WORKER_GPUS=1.0     # 每个 worker 独占 1 块 GPU（worker 0→GPU0, 1→GPU1, 2→GPU2）
DP_PARTITION_SIZE=500  # 每个 partition 再切成 DP_WORKERS 份

# Pipeline 参数
ARCH_PER_STEP=3
NUM_STAGES=3
WORKER_GPUS=0.33
PIPELINE_PARTITION_SIZE=500

DATASET_TAG="mooc_20k"
OUTPUT_ROOT="outputs/${DATASET_TAG}_3way"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║       NAS 3-Way: Serial  vs  Data-Parallel  vs  Pipeline            ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Dataset       : $DATA_FILE  (${MAX_EVENTS} events)"
echo "║  Epochs/trial  : $EPOCHS   |  Metric: $METRIC  |  K: $K"
echo "║  Seeds         : ${SEEDS[*]}  ($(( ${#SEEDS[@]} )) runs each)"
echo "║  Serial        : $SERIAL_TRIALS trials/seed  (baseline)"
echo "║  Data-Parallel : $DP_TRIALS trials/seed  ($DP_WORKERS workers, ~${DP_WORKERS}x per-trial speedup)"
echo "║  Pipeline      : $PIPELINE_TRIALS trials/seed  (2x arch coverage)"
echo "║  Output        : $OUTPUT_ROOT/"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$OUTPUT_ROOT"
rm -f "${OUTPUT_ROOT}/seed_times.csv"

SEED_IDX=0
for SEED in "${SEEDS[@]}"; do
    SEED_IDX=$(( SEED_IDX + 1 ))
    SEED_DIR="${OUTPUT_ROOT}/seed_${SEED}"
    OUTPUT_SERIAL="${SEED_DIR}/serial"
    OUTPUT_DP="${SEED_DIR}/data_parallel"
    OUTPUT_PIPELINE="${SEED_DIR}/pipeline"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶  Seed ${SEED_IDX}/${#SEEDS[@]}  (seed=${SEED})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for DIR in "$OUTPUT_SERIAL" "$OUTPUT_DP" "$OUTPUT_PIPELINE"; do
        [ -d "$DIR" ] && rm -rf "$DIR"
    done
    mkdir -p "$OUTPUT_SERIAL" "$OUTPUT_DP" "$OUTPUT_PIPELINE"

    # ── [1/4] 串行 Baseline
    echo ""
    echo "  [1/4] 串行搜索 (Baseline)  seed=${SEED}  ${SERIAL_TRIALS} trials"
    SERIAL_START=$(date +%s%N)
    export CUDA_VISIBLE_DEVICES="0"

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

    # ── [2/4] Data-Parallel
    echo ""
    echo "  [2/4] 数据并行 (Data-Parallel)  seed=${SEED}  ${DP_TRIALS} trials  ${DP_WORKERS} workers"
    DP_START=$(date +%s%N)
    export CUDA_VISIBLE_DEVICES="0,1,2"

    python search.py \
        --dataset                    "$DATASET" \
        --local-data-path            "$DATA_FILE" \
        --max-events                 "$MAX_EVENTS" \
        --search-mode                rl \
        --execution-mode             data_parallel \
        --trials                     "$DP_TRIALS" \
        --epochs-per-trial           "$EPOCHS" \
        --partition-size             "$DP_PARTITION_SIZE" \
        --data-parallel-workers      "$DP_WORKERS" \
        --data-parallel-worker-gpus  "$DP_WORKER_GPUS" \
        --data-parallel-visible-gpus "0,1,2" \
        --device                     cuda \
        --seed                       "$SEED" \
        --k                          "$K" \
        --selection-metric           "$METRIC" \
        --output-dir                 "$OUTPUT_DP"

    DP_END=$(date +%s%N)
    DP_SEC=$(( (DP_END - DP_START) / 1000000000 ))
    echo "  ✅ Data-Parallel 完成  ${DP_SEC}s"

    # ── [3/4] Pipeline
    echo ""
    echo "  [3/4] Pipeline 搜索 (Ours)  seed=${SEED}  ${PIPELINE_TRIALS} trials"
    PIPELINE_START=$(date +%s%N)
    export CUDA_VISIBLE_DEVICES="0,1,2"

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
        --partition-size        "$PIPELINE_PARTITION_SIZE" \
        --stage-balance-strategy cost \
        --seed                  "$SEED" \
        --k                     "$K" \
        --selection-metric      "$METRIC" \
        --pipeline-trace \
        --output-dir            "$OUTPUT_PIPELINE"

    PIPELINE_END=$(date +%s%N)
    PIPELINE_SEC=$(( (PIPELINE_END - PIPELINE_START) / 1000000000 ))
    echo "  ✅ Pipeline 完成  ${PIPELINE_SEC}s"

    # ── [4/4] 单 seed 三方报告
    echo ""
    echo "  [4/4] 生成 seed=${SEED} 三方报告"
    python tools/compare_results_3way.py \
        --serial-dir      "$OUTPUT_SERIAL" \
        --dp-dir          "$OUTPUT_DP" \
        --pipeline-dir    "$OUTPUT_PIPELINE" \
        --serial-time     "$SERIAL_SEC" \
        --dp-time         "$DP_SEC" \
        --pipeline-time   "$PIPELINE_SEC" \
        --serial-trials   "$SERIAL_TRIALS" \
        --dp-trials       "$DP_TRIALS" \
        --pipeline-trials "$PIPELINE_TRIALS" \
        --output          "${SEED_DIR}/report_3way.txt"

    echo "${SEED},${SERIAL_SEC},${DP_SEC},${PIPELINE_SEC}" >> "${OUTPUT_ROOT}/seed_times.csv"
    echo ""
    echo "  seed=${SEED} 完成  Serial ${SERIAL_SEC}s  DataParallel ${DP_SEC}s  Pipeline ${PIPELINE_SEC}s"
done

# ── 多种子汇总
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶  生成三方多种子汇总报告"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python tools/aggregate_seeds_3way.py \
    --root            "$OUTPUT_ROOT" \
    --seeds           "${SEEDS[*]}" \
    --serial-trials   "$SERIAL_TRIALS" \
    --dp-trials       "$DP_TRIALS" \
    --pipeline-trials "$PIPELINE_TRIALS" \
    --output          "${OUTPUT_ROOT}/aggregate_report_3way.txt"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  三方对比实验完成！                                                   ║"
echo "║  Seeds   : ${SEEDS[*]}"
echo "║  Results : $OUTPUT_ROOT/"
echo "║  Summary : ${OUTPUT_ROOT}/aggregate_report_3way.txt"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
