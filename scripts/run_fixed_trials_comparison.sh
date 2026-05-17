#!/bin/bash
# ============================================================
# 固定 trial 数对比：Serial vs Data-Parallel vs Pipeline-Naive vs Pipeline-Smart
# 目的：相同搜索量下，比较各方法的实际运行时间
#
# 用法：
#   bash scripts/run_fixed_trials_comparison.sh [OPTIONS]
#
# 选项：
#   --gpu-list GPU列表     (默认: 自动检测)
#   --space SPACE          (默认: rnn_only)
#   --max-events NUM       (默认: 20000)
#   --trials NUM           (默认: 9)
#   --epochs NUM           (默认: 3)
#   --seeds SEEDS          (默认: 42)
#   --output-dir DIR       (默认: outputs/fixed_trials_TIMESTAMP)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ──── 默认参数 ────
GPU_LIST=""
SEARCH_SPACE="rnn_only"
DATASET="public_csv"
DATA_FILE="data/public/mooc.csv"
MAX_EVENTS=20000
TRIALS=9
EPOCHS=3
SEEDS="42"
K=10
METRIC="mrr"
OUTPUT_DIR=""
RESUME_EXISTING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-list)   GPU_LIST="$2";   shift 2 ;;
        --space)      SEARCH_SPACE="$2"; shift 2 ;;
        --max-events) MAX_EVENTS="$2"; shift 2 ;;
        --trials)     TRIALS="$2";     shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --seeds)      SEEDS="$2";      shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --resume-existing) RESUME_EXISTING=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "$GPU_LIST" ]; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    NUM_GPUS=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
    GPU_LIST=$(python -c "print(','.join(str(i) for i in range($NUM_GPUS)))")
else
    NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
fi

if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="outputs/fixed_trials_${TIMESTAMP}"
fi

PARTITION_SIZE=$(( MAX_EVENTS / 20 ))
PARTITION_SIZE=$(( PARTITION_SIZE < 100 ? 100 : PARTITION_SIZE ))
NUM_STAGES=$NUM_GPUS
DP_WORKERS=$NUM_GPUS

IFS=',' read -ra SEEDS_ARRAY <<< "$SEEDS"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║     固定 Trial 数对比：Serial vs DP vs Pipeline-Naive vs Pipeline-Smart ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  数据集: $DATA_FILE ($MAX_EVENTS events)"
echo "║  搜索空间: $SEARCH_SPACE  |  Trials: $TRIALS  |  Epochs: $EPOCHS"
echo "║  GPUs: $NUM_GPUS ($GPU_LIST)  |  Seeds: $SEEDS"
echo "║  输出: $OUTPUT_DIR/"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$OUTPUT_DIR"
SUMMARY="${OUTPUT_DIR}/summary.csv"
echo "seed,method,trials,wall_time_s,best_score,avg_time_per_trial_s" > "$SUMMARY"

for SEED in "${SEEDS_ARRAY[@]}"; do
    SEED_DIR="${OUTPUT_DIR}/seed_${SEED}"
    mkdir -p "$SEED_DIR"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶  seed=${SEED}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── [1/4] Serial
    echo "  [1/4] Serial"
    if [[ "$RESUME_EXISTING" == true ]] && [[ -f "${SEED_DIR}/serial/leaderboard.csv" ]]; then
        echo "  ↪ 已存在，跳过"
        SERIAL_SEC=$(python3 -c "import csv; rows=list(csv.DictReader(open('${SEED_DIR}/serial/timing_log.csv'))); print(int(float(rows[-1]['end_time_s']))) if rows else print(0)" 2>/dev/null || echo 0)
        SERIAL_SCORE=$(python -c "import csv; rows=list(csv.DictReader(open('${SEED_DIR}/serial/leaderboard.csv'))); print(rows[0]['score'] if rows else 0)")
    else
        export CUDA_VISIBLE_DEVICES="0"
        T0=$(date +%s%N)
        python search.py \
            --dataset "$DATASET" --local-data-path "$DATA_FILE" \
            --max-events "$MAX_EVENTS" --space "$SEARCH_SPACE" \
            --search-mode rl --execution-mode serial \
            --trials "$TRIALS" --epochs-per-trial "$EPOCHS" \
            --partition-size "$PARTITION_SIZE" \
            --seed "$SEED" --k "$K" --selection-metric "$METRIC" \
            --device cuda --output-dir "${SEED_DIR}/serial"
        T1=$(date +%s%N)
        SERIAL_SEC=$(( (T1 - T0) / 1000000000 ))
        SERIAL_SCORE=$(python -c "import csv; rows=list(csv.DictReader(open('${SEED_DIR}/serial/leaderboard.csv'))); print(rows[0]['score'] if rows else 0)")
    fi
    echo "  ✅ Serial: ${SERIAL_SEC}s  best=${SERIAL_SCORE}"
    echo "$SEED,serial,$TRIALS,$SERIAL_SEC,$SERIAL_SCORE,$(( SERIAL_SEC / TRIALS ))" >> "$SUMMARY"

    # ── [2/4] Data-Parallel
    echo "  [2/4] Data-Parallel"
    if [[ "$RESUME_EXISTING" == true ]] && [[ -f "${SEED_DIR}/data_parallel/leaderboard.csv" ]]; then
        echo "  ↪ 已存在，跳过"
        DP_SEC=$(python3 -c "import csv; rows=list(csv.DictReader(open('${SEED_DIR}/data_parallel/timing_log.csv'))); print(int(float(rows[-1]['end_time_s']))) if rows else print(0)" 2>/dev/null || echo 0)
        DP_SCORE=$(python -c "import csv; rows=list(csv.DictReader(open('${SEED_DIR}/data_parallel/leaderboard.csv'))); print(rows[0]['score'] if rows else 0)")
    else
        export CUDA_VISIBLE_DEVICES="$GPU_LIST"
        T0=$(date +%s%N)
        python search.py \
            --dataset "$DATASET" --local-data-path "$DATA_FILE" \
            --max-events "$MAX_EVENTS" --space "$SEARCH_SPACE" \
            --search-mode rl --execution-mode data_parallel \
            --trials "$TRIALS" --epochs-per-trial "$EPOCHS" \
            --partition-size "$PARTITION_SIZE" \
            --data-parallel-workers "$DP_WORKERS" \
            --data-parallel-worker-gpus 1.0 \
            --data-parallel-visible-gpus "$GPU_LIST" \
            --data-parallel-sync-level micro_batch \
            --seed "$SEED" --k "$K" --selection-metric "$METRIC" \
            --device cuda --output-dir "${SEED_DIR}/data_parallel"
        T1=$(date +%s%N)
        DP_SEC=$(( (T1 - T0) / 1000000000 ))
        DP_SCORE=$(python -c "import csv; rows=list(csv.DictReader(open('${SEED_DIR}/data_parallel/leaderboard.csv'))); print(rows[0]['score'] if rows else 0)")
    fi
    echo "  ✅ Data-Parallel: ${DP_SEC}s  best=${DP_SCORE}"
    echo "$SEED,data_parallel,$TRIALS,$DP_SEC,$DP_SCORE,$(( DP_SEC / TRIALS ))" >> "$SUMMARY"

    # ── [3/4] Pipeline-Naive
    echo "  [3/4] Pipeline-Naive"
    if [[ "$RESUME_EXISTING" == true ]] && [[ -f "${SEED_DIR}/pipeline_naive/leaderboard.csv" ]]; then
        echo "  ↪ 已存在，跳过"
        NAIVE_SEC=$(python3 -c "import csv; rows=list(csv.DictReader(open('${SEED_DIR}/pipeline_naive/timing_log.csv'))); print(int(float(rows[-1]['end_time_s']))) if rows else print(0)" 2>/dev/null || echo 0)
        NAIVE_SCORE=$(python -c "import csv; rows=list(csv.DictReader(open('${SEED_DIR}/pipeline_naive/leaderboard.csv'))); print(rows[0]['score'] if rows else 0)")
    else
        export CUDA_VISIBLE_DEVICES="$GPU_LIST"
        T0=$(date +%s%N)
        python search.py \
            --dataset "$DATASET" --local-data-path "$DATA_FILE" \
            --max-events "$MAX_EVENTS" --space "$SEARCH_SPACE" \
            --search-mode rl --execution-mode ray_pipeline \
            --trials "$TRIALS" --epochs-per-trial "$EPOCHS" \
            --architectures-per-step "$NUM_STAGES" \
            --num-pipeline-stages "$NUM_STAGES" \
            --pipeline-worker-gpus 1.0 \
            --pipeline-stage-train-workers 1 \
            --partition-size "$PARTITION_SIZE" \
            --stage-balance-strategy count \
            --seed "$SEED" --k "$K" --selection-metric "$METRIC" \
            --device cuda --gpu-list "$GPU_LIST" \
            --pipeline-mode naive \
            --output-dir "${SEED_DIR}/pipeline_naive"
        T1=$(date +%s%N)
        NAIVE_SEC=$(( (T1 - T0) / 1000000000 ))
        NAIVE_SCORE=$(python -c "import csv; rows=list(csv.DictReader(open('${SEED_DIR}/pipeline_naive/leaderboard.csv'))); print(rows[0]['score'] if rows else 0)")
    fi
    echo "  ✅ Pipeline-Naive: ${NAIVE_SEC}s  best=${NAIVE_SCORE}"
    echo "$SEED,pipeline_naive,$TRIALS,$NAIVE_SEC,$NAIVE_SCORE,$(( NAIVE_SEC / TRIALS ))" >> "$SUMMARY"

    # ── [4/4] Pipeline-Smart
    echo "  [4/4] Pipeline-Smart"
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"
    T0=$(date +%s%N)
    python search.py \
        --dataset "$DATASET" --local-data-path "$DATA_FILE" \
        --max-events "$MAX_EVENTS" --space "$SEARCH_SPACE" \
        --search-mode rl --execution-mode ray_pipeline \
        --trials "$TRIALS" --epochs-per-trial     "$EPOCHS" \
        --pipeline-worker-gpus 0.0 \
        --seed "$SEED" --k "$K" --selection-metric "$METRIC" \
        --device cuda --gpu-list "$GPU_LIST" \
        --pipeline-mode smart \
        --output-dir "${SEED_DIR}/pipeline_smart"
    T1=$(date +%s%N)
    SMART_SEC=$(( (T1 - T0) / 1000000000 ))
    SMART_SCORE=$(python -c "import csv; rows=list(csv.DictReader(open('${SEED_DIR}/pipeline_smart/leaderboard.csv'))); print(rows[0]['score'] if rows else 0)")
    echo "  ✅ Pipeline-Smart: ${SMART_SEC}s  best=${SMART_SCORE}"
    echo "$SEED,pipeline_smart,$TRIALS,$SMART_SEC,$SMART_SCORE,$(( SMART_SEC / TRIALS ))" >> "$SUMMARY"

    # ── 打印本 seed 汇总
    echo ""
    echo "  seed=${SEED} 汇总（${TRIALS} trials）:"
    printf "  %-20s %8s  %10s  %12s\n" "方法" "时间(s)" "best_score" "avg_s/trial"
    printf "  %-20s %8s  %10s  %12s\n" "Serial"          "$SERIAL_SEC" "$SERIAL_SCORE" "$(( SERIAL_SEC / TRIALS ))"
    printf "  %-20s %8s  %10s  %12s\n" "Data-Parallel"   "$DP_SEC"     "$DP_SCORE"     "$(( DP_SEC / TRIALS ))"
    printf "  %-20s %8s  %10s  %12s\n" "Pipeline-Naive"  "$NAIVE_SEC"  "$NAIVE_SCORE"  "$(( NAIVE_SEC / TRIALS ))"
    printf "  %-20s %8s  %10s  %12s\n" "Pipeline-Smart"  "$SMART_SEC"  "$SMART_SCORE"  "$(( SMART_SEC / TRIALS ))"
    echo ""

    # ── 生成 seed 级报告
    echo "  [5/5] 生成 seed=${SEED} 报告"
    python tools/compare_fixed_trials.py \
        --serial-dir    "${SEED_DIR}/serial" \
        --dp-dir        "${SEED_DIR}/data_parallel" \
        --naive-dir     "${SEED_DIR}/pipeline_naive" \
        --smart-dir     "${SEED_DIR}/pipeline_smart" \
        --serial-time   "$SERIAL_SEC" \
        --dp-time       "$DP_SEC" \
        --naive-time    "$NAIVE_SEC" \
        --smart-time    "$SMART_SEC" \
        --trials        "$TRIALS" \
        --output        "${SEED_DIR}/report_fixed_trials.txt"
    echo ""
done

# ── 多种子汇总
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶  多种子汇总（见 summary.csv）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
import csv, sys
rows = list(csv.DictReader(open('${OUTPUT_DIR}/summary.csv')))
methods = ['serial','data_parallel','pipeline_naive','pipeline_smart']
labels  = ['Serial','DataParallel','Pipeline-Naive','Pipeline-Smart']
print(f'  {\"方法\":<20} {\"平均时间(s)\":>12}  {\"平均best_score\":>14}')
print(f'  {\"─\"*20}  {\"─\"*12}  {\"─\"*14}')
for m, l in zip(methods, labels):
    mrows = [r for r in rows if r['method']==m]
    if not mrows: continue
    avg_t = sum(float(r['wall_time_s']) for r in mrows)/len(mrows)
    avg_s = sum(float(r['best_score']) for r in mrows)/len(mrows)
    print(f'  {l:<20} {avg_t:>12.0f}  {avg_s:>14.4f}')
"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  完成！"
echo "║  CSV 汇总  : $SUMMARY"
echo "║  各 seed   : ${OUTPUT_DIR}/seed_*/report_fixed_trials.txt"
echo "╚══════════════════════════════════════════════════════════════════════╝"
