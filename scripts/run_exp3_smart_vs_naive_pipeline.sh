#!/bin/bash
# ============================================================
# Pipeline 自动优化效果对比：Pipeline-Smart vs Pipeline-Naive
#
# Pipeline-Smart（本文方法）：
#   - cost 均衡分配 partition（DP 最优切割）
#   - 自动 worker 分配（前置 stage 多分）
#   - 大 batch（NUM_GPUS*3）减少 pipeline bubble
#
# Pipeline-Naive（baseline）：
#   - count 均等分配 partition（简单按数量切）
#   - 每 stage 固定 1 worker
#   - 最小 batch（NUM_GPUS）bubble 最大
#
# 用法：bash run_exp3_smart_vs_naive_pipeline.sh [GPU列表]
#   例：bash run_exp3_smart_vs_naive_pipeline.sh 0,1,2
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# -------- 可调参数 --------
DATASET="public_csv"
DATA_FILE="data/public/mooc.csv"
MAX_EVENTS=10000
TIME_BUDGET=600
TRIALS=999
EPOCHS=2
SEEDS=(42 43 44)
K=10
METRIC="mrr"
SEARCH_SPACE="rnn_only"

# GPU 配置
if [ -n "$1" ]; then
    GPU_LIST="$1"
    NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    NUM_GPUS=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
    GPU_LIST=$(python -c "print(','.join(str(i) for i in range($NUM_GPUS)))")
fi

PARTITION_SIZE=$(( MAX_EVENTS / 20 ))
PARTITION_SIZE=$(( PARTITION_SIZE < 100 ? 100 : PARTITION_SIZE ))

NUM_STAGES=$NUM_GPUS
WORKER_GPUS=1.0

# Smart 参数
SMART_ARCH_PER_STEP=$(( NUM_GPUS * 3 ))

# Naive 参数
NAIVE_ARCH_PER_STEP=$NUM_STAGES
NAIVE_STAGE_WORKERS="1"

OUTPUT_ROOT="outputs/exp3_smart_vs_naive"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         Pipeline 自动优化效果对比：Smart vs Naive                    ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Dataset       : $DATA_FILE  (${MAX_EVENTS} events)"
echo "║  Search space  : $SEARCH_SPACE"
echo "║  Epochs/trial  : $EPOCHS   |  Metric: $METRIC  |  K: $K"
echo "║  Seeds         : ${SEEDS[*]}  ($(( ${#SEEDS[@]} )) runs each)"
echo "║  Time budget   : ${TIME_BUDGET}s per method"
echo "║  GPUs          : $NUM_GPUS  ($GPU_LIST)"
echo "║  Smart         : $NUM_STAGES stages, $SMART_ARCH_PER_STEP archs/step, cost balance, auto workers"
echo "║  Naive         : $NUM_STAGES stages, $NAIVE_ARCH_PER_STEP archs/step, count balance, 1 worker/stage"
echo "║  Output        : $OUTPUT_ROOT/"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$OUTPUT_ROOT"
rm -f "${OUTPUT_ROOT}/seed_times.csv"

SEED_IDX=0
for SEED in "${SEEDS[@]}"; do
    SEED_IDX=$(( SEED_IDX + 1 ))
    SEED_DIR="${OUTPUT_ROOT}/seed_${SEED}"
    OUTPUT_SMART="${SEED_DIR}/smart"
    OUTPUT_NAIVE="${SEED_DIR}/naive"

    for DIR in "$OUTPUT_SMART" "$OUTPUT_NAIVE"; do
        [ -d "$DIR" ] && rm -rf "$DIR"
    done
    mkdir -p "$OUTPUT_SMART" "$OUTPUT_NAIVE"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶  Seed ${SEED_IDX}/${#SEEDS[@]}  (seed=${SEED})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── [1/3] Pipeline-Smart
    echo ""
    echo "  [1/3] Pipeline-Smart  seed=${SEED}"
    SMART_START=$(date +%s%N)
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"

    python search.py \
        --dataset               "$DATASET" \
        --local-data-path       "$DATA_FILE" \
        --max-events            "$MAX_EVENTS" \
        --space                 "$SEARCH_SPACE" \
        --search-mode           rl \
        --execution-mode        ray_pipeline \
        --trials                "$TRIALS" \
        --epochs-per-trial      "$EPOCHS" \
        --time-budget-sec       "$TIME_BUDGET" \
        --architectures-per-step "$SMART_ARCH_PER_STEP" \
        --num-pipeline-stages   "$NUM_STAGES" \
        --pipeline-worker-gpus  "$WORKER_GPUS" \
        --partition-size        "$PARTITION_SIZE" \
        --stage-balance-strategy cost \
        --device                cuda \
        --seed                  "$SEED" \
        --k                     "$K" \
        --selection-metric      "$METRIC" \
        --output-dir            "$OUTPUT_SMART"

    SMART_END=$(date +%s%N)
    SMART_SEC=$(( (SMART_END - SMART_START) / 1000000000 ))
    echo "  ✅ Pipeline-Smart 完成  ${SMART_SEC}s"

    # ── [2/3] Pipeline-Naive
    echo ""
    echo "  [2/3] Pipeline-Naive  seed=${SEED}"
    NAIVE_START=$(date +%s%N)
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"

    python search.py \
        --dataset               "$DATASET" \
        --local-data-path       "$DATA_FILE" \
        --max-events            "$MAX_EVENTS" \
        --space                 "$SEARCH_SPACE" \
        --search-mode           rl \
        --execution-mode        ray_pipeline \
        --trials                "$TRIALS" \
        --epochs-per-trial      "$EPOCHS" \
        --time-budget-sec       "$TIME_BUDGET" \
        --architectures-per-step "$NAIVE_ARCH_PER_STEP" \
        --num-pipeline-stages   "$NUM_STAGES" \
        --pipeline-worker-gpus  "$WORKER_GPUS" \
        --pipeline-stage-train-workers "$NAIVE_STAGE_WORKERS" \
        --partition-size        "$PARTITION_SIZE" \
        --stage-balance-strategy count \
        --device                cuda \
        --seed                  "$SEED" \
        --k                     "$K" \
        --selection-metric      "$METRIC" \
        --output-dir            "$OUTPUT_NAIVE"

    NAIVE_END=$(date +%s%N)
    NAIVE_SEC=$(( (NAIVE_END - NAIVE_START) / 1000000000 ))
    echo "  ✅ Pipeline-Naive 完成  ${NAIVE_SEC}s"

    # ── [3/3] 报告
    echo ""
    echo "  [3/3] 生成 seed=${SEED} 对比报告"
    python tools/compare_results_2way.py \
        --a-dir     "$OUTPUT_NAIVE" \
        --b-dir     "$OUTPUT_SMART" \
        --a-label   "Pipeline-Naive" \
        --b-label   "Pipeline-Smart" \
        --title     "Pipeline 自动优化效果对比" \
        --conclusion "Pipeline-Smart 通过 cost 均衡分配和自动 worker 分配，相比 Pipeline-Naive 提升搜索效率" \
        --output    "${SEED_DIR}/report_smart_vs_naive.txt"

    echo "${SEED},${SMART_SEC},${NAIVE_SEC}" >> "${OUTPUT_ROOT}/seed_times.csv"
    echo "  seed=${SEED} 完成  Smart ${SMART_SEC}s  Naive ${NAIVE_SEC}s"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶  生成多种子汇总报告"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python tools/aggregate_seeds_2way.py \
    --root      "$OUTPUT_ROOT" \
    --seeds     "${SEEDS[*]}" \
    --a-label   "Pipeline-Naive" \
    --b-label   "Pipeline-Smart" \
    --output    "${OUTPUT_ROOT}/aggregate_report.txt"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  实验完成！                                                           ║"
echo "║  Results : $OUTPUT_ROOT/"
echo "║  Summary : ${OUTPUT_ROOT}/aggregate_report.txt"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
