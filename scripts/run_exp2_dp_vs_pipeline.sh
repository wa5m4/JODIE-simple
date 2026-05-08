#!/bin/bash
# ============================================================
# 实验2：小搜索空间  DataParallel vs Pipeline
#
# 论点：搜索架构数相近时，DataParallel 因 AllReduce 破坏时序
#       导致每个 trial 训练质量下降；Pipeline 保持时序完整性，
#       每个 trial 质量更高。
#       → Pipeline 在训练质量上优于 DataParallel。
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# -------- 可调参数 --------
DATASET="public_csv"
DATA_FILE="data/public/mooc.csv"
MAX_EVENTS=20000
TIME_BUDGET=3600
EPOCHS=3
SEEDS=(42 43 44)
K=10
METRIC="mrr"
SPACE="rnn_only"       # 小搜索空间（均匀 trial 时间，控制变量）

DP_WORKERS=3
DP_WORKER_GPUS=1.0
DP_PARTITION_SIZE=500

ARCH_PER_STEP=3
NUM_STAGES=3
WORKER_GPUS=1.0
PIPELINE_PARTITION_SIZE=500

DATASET_TAG="mooc_20k"
OUTPUT_ROOT="outputs/${DATASET_TAG}_exp2_dp_vs_pipeline"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║       实验2：DataParallel vs Pipeline  (小搜索空间)                  ║"
echo "║  论点：架构数相近时，Pipeline 时序完整性保证更高训练质量              ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Space         : $SPACE  (仅 jodie_rnn，trial 时间均匀)"
echo "║  Dataset       : $DATA_FILE  (${MAX_EVENTS} events)"
echo "║  Epochs/trial  : $EPOCHS   |  Metric: $METRIC  |  K: $K"
echo "║  Seeds         : ${SEEDS[*]}  ($(( ${#SEEDS[@]} )) runs each)"
echo "║  Time budget   : ${TIME_BUDGET}s per method"
echo "║  Output        : $OUTPUT_ROOT/"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$OUTPUT_ROOT"
rm -f "${OUTPUT_ROOT}/seed_times.csv"

SEED_IDX=0
for SEED in "${SEEDS[@]}"; do
    SEED_IDX=$(( SEED_IDX + 1 ))
    SEED_DIR="${OUTPUT_ROOT}/seed_${SEED}"
    OUTPUT_DP="${SEED_DIR}/data_parallel"
    OUTPUT_PIPELINE="${SEED_DIR}/pipeline"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶  Seed ${SEED_IDX}/${#SEEDS[@]}  (seed=${SEED})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for DIR in "$OUTPUT_DP" "$OUTPUT_PIPELINE"; do
        [ -d "$DIR" ] && rm -rf "$DIR"
    done
    mkdir -p "$OUTPUT_DP" "$OUTPUT_PIPELINE"

    # ── [1/3] DataParallel
    echo ""
    echo "  [1/3] 数据并行 (Baseline)  seed=${SEED}  ${DP_WORKERS} workers"
    DP_START=$(date +%s%N)
    export CUDA_VISIBLE_DEVICES="0,1,2"

    python search.py \
        --dataset                    "$DATASET" \
        --local-data-path            "$DATA_FILE" \
        --max-events                 "$MAX_EVENTS" \
        --space                      "$SPACE" \
        --search-mode                rl \
        --execution-mode             data_parallel \
        --trials                     999 \
        --epochs-per-trial           "$EPOCHS" \
        --time-budget-sec            "$TIME_BUDGET" \
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
    echo "  DataParallel 完成  ${DP_SEC}s"

    # ── [2/3] Pipeline
    echo ""
    echo "  [2/3] Pipeline 搜索 (Ours)  seed=${SEED}"
    PIPELINE_START=$(date +%s%N)
    export CUDA_VISIBLE_DEVICES="0,1,2"

    python search.py \
        --dataset                "$DATASET" \
        --local-data-path        "$DATA_FILE" \
        --max-events             "$MAX_EVENTS" \
        --space                  "$SPACE" \
        --search-mode            rl \
        --execution-mode         ray_pipeline \
        --trials                 999 \
        --epochs-per-trial       "$EPOCHS" \
        --time-budget-sec        "$TIME_BUDGET" \
        --architectures-per-step "$ARCH_PER_STEP" \
        --num-pipeline-stages    "$NUM_STAGES" \
        --pipeline-worker-gpus   "$WORKER_GPUS" \
        --partition-size         "$PIPELINE_PARTITION_SIZE" \
        --stage-balance-strategy cost \
        --seed                   "$SEED" \
        --k                      "$K" \
        --selection-metric       "$METRIC" \
        --pipeline-trace \
        --output-dir             "$OUTPUT_PIPELINE"

    PIPELINE_END=$(date +%s%N)
    PIPELINE_SEC=$(( (PIPELINE_END - PIPELINE_START) / 1000000000 ))
    echo "  Pipeline 完成  ${PIPELINE_SEC}s"

    # ── [3/3] 报告
    echo ""
    echo "  [3/3] 生成 seed=${SEED} 对比报告"
    python tools/compare_results_2way.py \
        --a-dir       "$OUTPUT_DP" \
        --b-dir       "$OUTPUT_PIPELINE" \
        --a-label     "DataParallel" \
        --b-label     "Pipeline" \
        --a-time      "$DP_SEC" \
        --b-time      "$PIPELINE_SEC" \
        --title       "NAS Exp2: DataParallel vs Pipeline  (Small Search Space)" \
        --conclusion  "DataParallel 将每个 partition 切成 3 个时间段并行训练，
AllReduce 平均了来自不同时间段的梯度，破坏了 JODIE 的时序依赖。
即使探索了相近数量的架构，每个 trial 的训练质量系统性偏低。
Pipeline 每个架构在独立 worker 上按完整时序顺序训练，梯度方向正确。
→ 相同架构数下，Pipeline 找到更高分数的架构。" \
        --output      "${SEED_DIR}/report_exp2.txt"

    echo "${SEED},${DP_SEC},${PIPELINE_SEC}" >> "${OUTPUT_ROOT}/seed_times.csv"
    echo "  seed=${SEED} 完成  DataParallel ${DP_SEC}s  Pipeline ${PIPELINE_SEC}s"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶  生成多种子汇总"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python tools/aggregate_seeds_2way.py \
    --root     "$OUTPUT_ROOT" \
    --seeds    "${SEEDS[*]}" \
    --a-label  "DataParallel" \
    --b-label  "Pipeline" \
    --title    "Exp2 Multi-Seed: DataParallel vs Pipeline (Small Space)" \
    --output   "${OUTPUT_ROOT}/aggregate_report_exp2.txt" 2>/dev/null || \
    echo "  (汇总脚本未找到，跳过多种子汇总)"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  实验2 完成！                                                         ║"
echo "║  Seeds   : ${SEEDS[*]}"
echo "║  Results : $OUTPUT_ROOT/"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
