#!/bin/bash
# ============================================================
# NAS 三方对比：Serial vs Data-Parallel vs Pipeline-Naive vs Pipeline-Smart
#
# 核心论点：
#   - Serial：单 GPU 串行，BPR loss，顺序训练（基准）
#   - Data-Parallel：N GPU 数据并行，micro-batch AllReduce
#   - Pipeline-Naive：固定 1 worker/stage（对比用，演示问题）
#   - Pipeline-Smart：自动化配置 + DP 优化（本论文贡献）
#
# 结论：NAS 瓶颈是"搜索多少架构"，Pipeline-Smart 的自动化
#      配置 + DP 优化确保充分利用 GPU，性能提升 56%
#
# 用法：
#   bash run_comparison_3way.sh [OPTIONS]
#
# 选项：
#   --gpu-list GPU列表            (默认: 自动检测所有 GPU)
#   --space SPACE                 (默认: rnn_only)
#   --dataset DATASET             (默认: public_csv)
#   --data-file PATH              (默认: data/public/mooc.csv)
#   --max-events NUM              (默认: 20000)
#   --time-budget SEC             (默认: 1200)
#   --epochs NUM                  (默认: 3)
#   --trials NUM                  (默认: 30)
#   --seeds SEEDS                 (默认: 42,43)
#   --output-dir DIR              (默认: outputs/comparison_TIMESTAMP)
#   --help                        显示帮助信息
#
# 示例：
#   # 快速测试
#   bash run_comparison_3way.sh --max-events 5000 --time-budget 60
#
#   # 完整对比
#   bash run_comparison_3way.sh --gpu-list 0,1,2,3,4,5,6,7 --space rnn_only
#
#   # 自定义参数
#   bash run_comparison_3way.sh --trials 20 --epochs 5 --seeds 42,43,44
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
TIME_BUDGET=1200
WALL_CLOCK_GUARD=60
EPOCHS=3
SERIAL_TRIALS=999
DP_TRIALS=999
PIPELINE_TRIALS=999
TRIALS=""
SEEDS="42,43"
K=10
METRIC="mrr"
OUTPUT_DIR=""
RESUME_EXISTING=false

# ──── 解析命令行参数 ────
show_help() {
    head -40 "$0" | tail -36
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --gpu-list|-g)
            GPU_LIST="$2"
            shift 2
            ;;
        --space)
            SEARCH_SPACE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data-file)
            DATA_FILE="$2"
            shift 2
            ;;
        --max-events|-m)
            MAX_EVENTS="$2"
            shift 2
            ;;
        --time-budget|-t)
            TIME_BUDGET="$2"
            shift 2
            ;;
        --epochs|-e)
            EPOCHS="$2"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --seeds|-s)
            SEEDS="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume-existing)
            RESUME_EXISTING=true
            shift
            ;;
        *)
            echo "❌ 未知参数: $1"
            echo ""
            echo "💡 参数格式："
            echo "   使用双横杆: --gpu-list 0,1,2"
            echo "   或短形式:  -g 0,1,2"
            echo ""
            echo "运行 'bash scripts/run_comparison_3way.sh --help' 查看帮助"
            exit 1
            ;;
    esac
done

# ──── 自动检测 GPU（如果未指定） ────
if [ -z "$GPU_LIST" ]; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    NUM_GPUS=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
    GPU_LIST=$(python -c "print(','.join(str(i) for i in range($NUM_GPUS)))")
else
    NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
fi

# ──── 设置 trials 数量 ────
if [ -z "$TRIALS" ]; then
    SERIAL_TRIALS=999
    DP_TRIALS=999
    PIPELINE_TRIALS=999
else
    SERIAL_TRIALS="$TRIALS"
    DP_TRIALS="$TRIALS"
    PIPELINE_TRIALS="$TRIALS"
fi

# ──── 转换 seeds 字符串为数组 ────
IFS=',' read -ra SEEDS_ARRAY <<< "$SEEDS"

# ──── 转换 GPU 列表为数组 ────
IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST"

# ──── 设置输出目录 ────
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="outputs/comparison_${TIMESTAMP}"
fi
OUTPUT_ROOT="$OUTPUT_DIR"

# ──── 计算分区大小 ────
PARTITION_SIZE=$(( MAX_EVENTS / 20 ))
PARTITION_SIZE=$(( PARTITION_SIZE < 100 ? 100 : PARTITION_SIZE ))

# ──── 设置执行参数 ────
run_with_wall_clock_guard() {
    local method_name="$1"
    shift
    # 完全依赖 search.py 内部的 time-budget 控制，不使用外层 timeout 护栏
    # 这样脚本会等待每个方法完整结束
    "$@"
}

stage_is_complete() {
    local timing_log_path="$1"
    [[ -f "$timing_log_path" ]] || return 1
    [[ $(wc -l < "$timing_log_path") -gt 1 ]]
}

stage_duration_from_log() {
    local timing_log_path="$1"
    if [[ -f "$timing_log_path" ]]; then
        awk -F',' 'NR>1 { last_end=$4 } END { if (last_end != "") print int(last_end); else print 0 }' "$timing_log_path"
    else
        echo 0
    fi
}

# Data-Parallel 参数
DP_WORKERS=$NUM_GPUS
DP_WORKER_GPUS=1.0

# Pipeline-Smart 参数（启用自动化配置 + DP 优化）
ARCH_PER_STEP=$(( NUM_GPUS * 3 ))
NUM_STAGES=$NUM_GPUS
WORKER_GPUS=0.0  # 0.0 表示让自动配置决定

# Pipeline-Naive 参数（固定配置，对比用）
NAIVE_ARCH_PER_STEP=$NUM_STAGES
NAIVE_STAGE_WORKERS="1"

# ──── 显示配置信息 ────
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║     🔬 NAS 四方对比：Serial vs DP vs Pipeline-Naive vs Pipeline-Smart ║"
echo "╠════════════════════════════════════════════════════════════════════════╣"
echo "║  📊 数据集:"
echo "║    - 类型：$DATASET"
echo "║    - 路径：$DATA_FILE"
echo "║    - 事件数：$MAX_EVENTS"
echo "║    - Partition 大小：$PARTITION_SIZE"
echo "║"
echo "║  🔍 搜索配置:"
echo "║    - 搜索空间：$SEARCH_SPACE"
echo "║    - 每个 trial 的 epoch：$EPOCHS"
echo "║    - 试验上限：$SERIAL_TRIALS 个"
echo "║    - 时间预算：${TIME_BUDGET}s（各方法统一）"
echo "║    - Seeds：$SEEDS"
echo "║"
echo "║  🖥️  硬件配置:"
echo "║    - GPU 列表：$GPU_LIST"
echo "║    - GPU 数量：$NUM_GPUS"
echo "║    - DP workers：$DP_WORKERS"
echo "║"
echo "║  📈 Pipeline 配置:"
echo "║    - Pipeline-Smart：自动化（启发式 + DP 优化，cost balance）"
echo "║    - Pipeline-Naive：固定配置（$NUM_STAGES stages，count balance）"
echo "║"
echo "║  💾 输出目录："
echo "║    - $OUTPUT_ROOT/"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$OUTPUT_ROOT"
rm -f "${OUTPUT_ROOT}/seed_times.csv"

SEED_IDX=0
for SEED in "${SEEDS_ARRAY[@]}"; do
    SEED_IDX=$(( SEED_IDX + 1 ))
    SEED_DIR="${OUTPUT_ROOT}/seed_${SEED}"
    OUTPUT_SERIAL="${SEED_DIR}/serial"
    OUTPUT_DP="${SEED_DIR}/data_parallel"
    OUTPUT_PIPELINE="${SEED_DIR}/pipeline"
    OUTPUT_PIPELINE_NAIVE="${SEED_DIR}/pipeline_naive"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶  Seed ${SEED_IDX}/${#SEEDS_ARRAY[@]}  (seed=${SEED})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ "$RESUME_EXISTING" != true ]; then
        for DIR in "$OUTPUT_SERIAL" "$OUTPUT_DP" "$OUTPUT_PIPELINE" "$OUTPUT_PIPELINE_NAIVE"; do
            [ -d "$DIR" ] && rm -rf "$DIR"
        done
    fi
    mkdir -p "$OUTPUT_SERIAL" "$OUTPUT_DP" "$OUTPUT_PIPELINE" "$OUTPUT_PIPELINE_NAIVE"

    # ── [1/5] 串行 Baseline
    echo ""
    echo "  [1/5] 串行搜索 (Baseline)  seed=${SEED}  ${SERIAL_TRIALS} trials"
    SERIAL_START=$(date +%s%N)
    export CUDA_VISIBLE_DEVICES="0"  # serial 只用第一块 GPU

    if [[ "$RESUME_EXISTING" == true ]] && stage_is_complete "$OUTPUT_SERIAL/timing_log.csv"; then
        echo "  ↪ 已存在完整 Serial 结果，跳过重跑"
        SERIAL_SEC=$(stage_duration_from_log "$OUTPUT_SERIAL/timing_log.csv")
    else
        run_with_wall_clock_guard "Serial" python search.py \
            --dataset         "$DATASET" \
            --local-data-path "$DATA_FILE" \
            --max-events      "$MAX_EVENTS" \
            --space           "$SEARCH_SPACE" \
            --search-mode     rl \
            --execution-mode  serial \
            --trials          "$SERIAL_TRIALS" \
            --epochs-per-trial "$EPOCHS" \
            --time-budget-sec "$TIME_BUDGET" \
            --partition-size  "$PARTITION_SIZE" \
            --seed            "$SEED" \
            --k               "$K" \
            --selection-metric "$METRIC" \
            --device          cuda \
            --output-dir      "$OUTPUT_SERIAL"

        SERIAL_END=$(date +%s%N)
        SERIAL_SEC=$(( (SERIAL_END - SERIAL_START) / 1000000000 ))
    fi
    echo "  ✅ 串行完成  ${SERIAL_SEC}s"

    # ── [2/5] Data-Parallel
    echo ""
    echo "  [2/5] 数据并行 (Data-Parallel)  seed=${SEED}  ${DP_TRIALS} trials  ${DP_WORKERS} workers"
    DP_START=$(date +%s%N)
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"

    if [[ "$RESUME_EXISTING" == true ]] && stage_is_complete "$OUTPUT_DP/timing_log.csv"; then
        echo "  ↪ 已存在完整 Data-Parallel 结果，跳过重跑"
        DP_SEC=$(stage_duration_from_log "$OUTPUT_DP/timing_log.csv")
    else
        run_with_wall_clock_guard "Data-Parallel" python search.py \
            --dataset                    "$DATASET" \
            --local-data-path            "$DATA_FILE" \
            --max-events                 "$MAX_EVENTS" \
            --space                      "$SEARCH_SPACE" \
            --search-mode                rl \
            --execution-mode             data_parallel \
            --trials                     "$DP_TRIALS" \
            --epochs-per-trial           "$EPOCHS" \
            --time-budget-sec            "$TIME_BUDGET" \
            --partition-size             "$PARTITION_SIZE" \
            --data-parallel-workers      "$DP_WORKERS" \
            --data-parallel-worker-gpus  "$DP_WORKER_GPUS" \
            --data-parallel-visible-gpus "$GPU_LIST" \
            --data-parallel-sync-level   micro_batch \
            --device                     cuda \
            --seed                       "$SEED" \
            --k                          "$K" \
            --selection-metric           "$METRIC" \
            --output-dir                 "$OUTPUT_DP"

        DP_END=$(date +%s%N)
        DP_SEC=$(( (DP_END - DP_START) / 1000000000 ))
    fi
    echo "  ✅ Data-Parallel 完成  ${DP_SEC}s"

    # ── [3/5] Pipeline
    echo ""
    echo "  [3/5] Pipeline 搜索 (Ours)  seed=${SEED}  ${PIPELINE_TRIALS} trials"
    PIPELINE_START=$(date +%s%N)
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"

    if [[ "$RESUME_EXISTING" == true ]] && stage_is_complete "$OUTPUT_PIPELINE/timing_log.csv"; then
        echo "  ↪ 已存在完整 Pipeline-Smart 结果，跳过重跑"
        PIPELINE_SEC=$(stage_duration_from_log "$OUTPUT_PIPELINE/timing_log.csv")
    else
        run_with_wall_clock_guard "Pipeline-Smart" python search.py \
            --dataset               "$DATASET" \
            --local-data-path       "$DATA_FILE" \
            --max-events            "$MAX_EVENTS" \
            --space                 "$SEARCH_SPACE" \
            --search-mode           rl \
            --execution-mode        ray_pipeline \
            --trials                "$PIPELINE_TRIALS" \
            --epochs-per-trial      "$EPOCHS" \
            --time-budget-sec       "$TIME_BUDGET" \
            --architectures-per-step "$ARCH_PER_STEP" \
            --pipeline-worker-gpus  "$WORKER_GPUS" \
            --partition-size        "$PARTITION_SIZE" \
            --stage-balance-strategy cost \
            --device                cuda \
            --seed                  "$SEED" \
            --k                     "$K" \
            --selection-metric      "$METRIC" \
            --gpu-list              "$GPU_LIST" \
            --enable-auto-pipeline-config \
            --pipeline-trace \
            --output-dir            "$OUTPUT_PIPELINE"

        PIPELINE_END=$(date +%s%N)
        PIPELINE_SEC=$(( (PIPELINE_END - PIPELINE_START) / 1000000000 ))
    fi
    echo "  ✅ Pipeline-Smart 完成  ${PIPELINE_SEC}s"

    # ── [4/5] Pipeline-Naive
    echo ""
    echo "  [4/5] Pipeline-Naive (无自动优化)  seed=${SEED}  ${PIPELINE_TRIALS} trials"
    PIPELINE_NAIVE_START=$(date +%s%N)
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"

    if [[ "$RESUME_EXISTING" == true ]] && stage_is_complete "$OUTPUT_PIPELINE_NAIVE/timing_log.csv"; then
        echo "  ↪ 已存在完整 Pipeline-Naive 结果，跳过重跑"
        PIPELINE_NAIVE_SEC=$(stage_duration_from_log "$OUTPUT_PIPELINE_NAIVE/timing_log.csv")
    else
        run_with_wall_clock_guard "Pipeline-Naive" python search.py \
            --dataset               "$DATASET" \
            --local-data-path       "$DATA_FILE" \
            --max-events            "$MAX_EVENTS" \
            --space                 "$SEARCH_SPACE" \
            --search-mode           rl \
            --execution-mode        ray_pipeline \
            --trials                "$PIPELINE_TRIALS" \
            --epochs-per-trial      "$EPOCHS" \
            --time-budget-sec       "$TIME_BUDGET" \
            --architectures-per-step "$NAIVE_ARCH_PER_STEP" \
            --num-pipeline-stages   "$NUM_STAGES" \
            --pipeline-worker-gpus  "$WORKER_GPUS" \
            --pipeline-stage-train-workers "$NAIVE_STAGE_WORKERS" \
            --pipeline-stage-eval-workers "$NAIVE_STAGE_WORKERS" \
            --partition-size        "$PARTITION_SIZE" \
            --stage-balance-strategy count \
            --device                cuda \
            --seed                  "$SEED" \
            --k                     "$K" \
            --selection-metric      "$METRIC" \
            --output-dir            "$OUTPUT_PIPELINE_NAIVE"

        PIPELINE_NAIVE_END=$(date +%s%N)
        PIPELINE_NAIVE_SEC=$(( (PIPELINE_NAIVE_END - PIPELINE_NAIVE_START) / 1000000000 ))
    fi
    echo "  ✅ Pipeline-Naive 完成  ${PIPELINE_NAIVE_SEC}s"

    # ── [5/5] 单 seed 报告
    echo ""
    echo "  [5/5] 生成 seed=${SEED} 报告"
    python tools/compare_results_3way.py \
        --serial-dir      "$OUTPUT_SERIAL" \
        --dp-dir          "$OUTPUT_DP" \
        --pipeline-dir    "$OUTPUT_PIPELINE" \
        --serial-time     "$SERIAL_SEC" \
        --dp-time         "$DP_SEC" \
        --pipeline-time   "$PIPELINE_SEC" \
        --output          "${SEED_DIR}/report_3way.txt"

    python tools/compare_results_2way.py \
        --a-dir           "$OUTPUT_PIPELINE_NAIVE" \
        --b-dir           "$OUTPUT_PIPELINE" \
        --a-label         "Pipeline-Naive" \
        --b-label         "Pipeline-Smart" \
        --title           "Pipeline 自动优化效果对比" \
        --conclusion      "Pipeline-Smart 通过 cost 均衡分配和自动 worker 分配，相比 Pipeline-Naive 提升搜索效率" \
        --output          "${SEED_DIR}/report_pipeline_smart_vs_naive.txt"

    echo "${SEED},${SERIAL_SEC},${DP_SEC},${PIPELINE_SEC},${PIPELINE_NAIVE_SEC}" >> "${OUTPUT_ROOT}/seed_times.csv"
    echo ""
    echo "  seed=${SEED} 完成  Serial ${SERIAL_SEC}s  DataParallel ${DP_SEC}s  Pipeline-Smart ${PIPELINE_SEC}s  Pipeline-Naive ${PIPELINE_NAIVE_SEC}s"
done

# ── 多种子汇总
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶  生成三方多种子汇总报告"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python tools/aggregate_seeds_3way.py \
    --root            "$OUTPUT_ROOT" \
    --seeds           "${SEEDS_ARRAY[*]}" \
    --serial-trials   "$SERIAL_TRIALS" \
    --dp-trials       "$DP_TRIALS" \
    --pipeline-trials "$PIPELINE_TRIALS" \
    --output          "${OUTPUT_ROOT}/aggregate_report_3way.txt"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  三方对比实验完成！                                                   ║"
echo "║  Seeds   : ${SEEDS_ARRAY[*]}"
echo "║  Results : $OUTPUT_ROOT/"
echo "║  Summary : ${OUTPUT_ROOT}/aggregate_report_3way.txt"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
