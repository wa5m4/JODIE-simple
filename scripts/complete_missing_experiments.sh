#!/bin/bash
# 补齐缺失的实验：检测16组实验中哪些未完成，按顺序补跑

set -e

# 共用参数
DATASET="public_csv"
LOCAL_DATA_PATH="data/public/mooc.csv"
MAX_EVENTS=20000
TRIALS=27
EPOCHS=3
GPU_LIST="0,1,2"
SEED=42
SPACE="mixed"
BASE_OUTPUT="outputs/full_cross_experiment"

# 创建输出目录
mkdir -p "$BASE_OUTPUT"

# 记录总开始时间
TOTAL_START=$(date +%s)

echo "=========================================="
echo "补齐缺失实验"
echo "数据集: $DATASET ($LOCAL_DATA_PATH)"
echo "事件数: $MAX_EVENTS, Trials: $TRIALS, Epochs: $EPOCHS"
echo "GPU: $GPU_LIST, Seed: $SEED, Space: $SPACE"
echo "=========================================="
echo ""

# 存储结果的数组
declare -a RESULTS
declare -a MISSING_EXPERIMENTS

# 函数：检查实验是否完成
is_completed() {
    local output_name=$1
    local output_dir="$BASE_OUTPUT/$output_name"
    local best_arch="$output_dir/best_arch.json"

    if [ -f "$best_arch" ] && [ -s "$best_arch" ]; then
        return 0  # 已完成
    else
        return 1  # 未完成
    fi
}

# 函数：从timing_log.csv提取总耗时
extract_elapsed_time() {
    local output_name=$1
    local timing_log="$BASE_OUTPUT/$output_name/timing_log.csv"

    if [ ! -f "$timing_log" ]; then
        echo "0"
        return
    fi

    # 提取最后一行的end_time_s列（第4列），即为总耗时
    local elapsed=$(tail -n 1 "$timing_log" | cut -d',' -f4)

    # 如果提取失败或为空，返回0
    if [ -z "$elapsed" ]; then
        echo "0"
    else
        # 四舍五入到整数
        printf "%.0f" "$elapsed"
    fi
}

# 函数：运行单个实验
run_experiment() {
    local exec_mode=$1
    local exec_params=$2
    local batch_mode=$3
    local batch_params=$4
    local output_name=$5
    local exp_num=$6
    local total_missing=$7

    local output_dir="$BASE_OUTPUT/$output_name"

    echo ""
    echo "=========================================="
    echo "[$exp_num/$total_missing] 补跑实验: $output_name"
    echo "执行模式: $exec_mode"
    echo "批处理: $batch_mode"
    echo "=========================================="

    local start_time=$(date +%s)

    # 运行实验（无缓冲输出）
    python -u search.py \
        $exec_params \
        $batch_params \
        --dataset "$DATASET" \
        --local-data-path "$LOCAL_DATA_PATH" \
        --max-events $MAX_EVENTS \
        --trials $TRIALS \
        --epochs-per-trial $EPOCHS \
        --gpu-list "$GPU_LIST" \
        --seed $SEED \
        --space "$SPACE" \
        --output-dir "$output_dir"

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local elapsed_min=$((elapsed / 60))
    local elapsed_sec=$((elapsed % 60))

    echo ""
    echo "[$exp_num/$total_missing] 完成: $output_name"
    echo "耗时: ${elapsed_min}分${elapsed_sec}秒 (${elapsed}秒)"

    # 提取结果
    if [ -f "$output_dir/best_arch.json" ]; then
        local test_mrr=$(python -c "import json; print(json.load(open('$output_dir/best_arch.json'))['test_mrr'])")
        local test_recall=$(python -c "import json; print(json.load(open('$output_dir/best_arch.json'))['test_recall_at_k'])")
        RESULTS+=("$output_name|$test_mrr|$test_recall|$elapsed")
        echo "结果: Test MRR=$test_mrr, Recall@10=$test_recall"
    else
        RESULTS+=("$output_name|ERROR|ERROR|$elapsed")
        echo "警告: 未找到结果文件"
    fi
    echo ""
}

# 定义所有16组实验（按顺序）
declare -a ALL_EXPERIMENTS=(
    "serial|--execution-mode serial|serial|--batch-mode serial|serial_serial"
    "serial|--execution-mode serial|tbatch|--batch-mode tbatch --train-batch-size 32|serial_tbatch"
    "serial|--execution-mode serial|tgn-last|--batch-mode tgn --tgn-loss-mode last --tgn-window-size 10.0|serial_tgn_last"
    "serial|--execution-mode serial|tgn-all|--batch-mode tgn --tgn-loss-mode all --tgn-window-size 10.0|serial_tgn_all"
    "data_parallel|--execution-mode data_parallel --data-parallel-workers 3|serial|--batch-mode serial|data_parallel_serial"
    "data_parallel|--execution-mode data_parallel --data-parallel-workers 3|tbatch|--batch-mode tbatch --train-batch-size 32|data_parallel_tbatch"
    "data_parallel|--execution-mode data_parallel --data-parallel-workers 3|tgn-last|--batch-mode tgn --tgn-loss-mode last --tgn-window-size 10.0|data_parallel_tgn_last"
    "data_parallel|--execution-mode data_parallel --data-parallel-workers 3|tgn-all|--batch-mode tgn --tgn-loss-mode all --tgn-window-size 10.0|data_parallel_tgn_all"
    "pipeline_naive|--execution-mode ray_pipeline --pipeline-mode naive --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1|serial|--batch-mode serial|pipeline_naive_serial"
    "pipeline_naive|--execution-mode ray_pipeline --pipeline-mode naive --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1|tbatch|--batch-mode tbatch --train-batch-size 32|pipeline_naive_tbatch"
    "pipeline_naive|--execution-mode ray_pipeline --pipeline-mode naive --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1|tgn-last|--batch-mode tgn --tgn-loss-mode last --tgn-window-size 10.0|pipeline_naive_tgn_last"
    "pipeline_naive|--execution-mode ray_pipeline --pipeline-mode naive --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1|tgn-all|--batch-mode tgn --tgn-loss-mode all --tgn-window-size 10.0|pipeline_naive_tgn_all"
    "pipeline_smart|--execution-mode ray_pipeline --pipeline-mode smart --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1|serial|--batch-mode serial|pipeline_smart_serial"
    "pipeline_smart|--execution-mode ray_pipeline --pipeline-mode smart --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1|tbatch|--batch-mode tbatch --train-batch-size 32|pipeline_smart_tbatch"
    "pipeline_smart|--execution-mode ray_pipeline --pipeline-mode smart --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1|tgn-last|--batch-mode tgn --tgn-loss-mode last --tgn-window-size 10.0|pipeline_smart_tgn_last"
    "pipeline_smart|--execution-mode ray_pipeline --pipeline-mode smart --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1|tgn-all|--batch-mode tgn --tgn-loss-mode all --tgn-window-size 10.0|pipeline_smart_tgn_all"
)

# 第一步：扫描并识别缺失的实验
echo "扫描实验完成状态..."
echo ""

for exp_def in "${ALL_EXPERIMENTS[@]}"; do
    IFS='|' read -r exec_mode exec_params batch_mode batch_params output_name <<< "$exp_def"

    if is_completed "$output_name"; then
        echo "✓ $output_name (已完成)"
        # 提取已有结果
        test_mrr=$(python -c "import json; print(json.load(open('$BASE_OUTPUT/$output_name/best_arch.json'))['test_mrr'])" 2>/dev/null || echo "N/A")
        test_recall=$(python -c "import json; print(json.load(open('$BASE_OUTPUT/$output_name/best_arch.json'))['test_recall_at_k'])" 2>/dev/null || echo "N/A")
        elapsed_time=$(extract_elapsed_time "$output_name")
        RESULTS+=("$output_name|$test_mrr|$test_recall|$elapsed_time")
    else
        echo "❌ $output_name (缺失)"
        MISSING_EXPERIMENTS+=("$exp_def")
    fi
done

echo ""
echo "=========================================="
echo "扫描完成"
echo "已完成: $((16 - ${#MISSING_EXPERIMENTS[@]}))/16"
echo "缺失: ${#MISSING_EXPERIMENTS[@]}/16"
echo "=========================================="
echo ""

# 第二步：补跑缺失的实验
if [ ${#MISSING_EXPERIMENTS[@]} -eq 0 ]; then
    echo "所有实验已完成，无需补跑"
else
    echo "开始补跑 ${#MISSING_EXPERIMENTS[@]} 个缺失实验..."
    echo ""

    exp_counter=1
    for exp_def in "${MISSING_EXPERIMENTS[@]}"; do
        IFS='|' read -r exec_mode exec_params batch_mode batch_params output_name <<< "$exp_def"
        run_experiment "$exec_mode" "$exec_params" "$batch_mode" "$batch_params" "$output_name" "$exp_counter" "${#MISSING_EXPERIMENTS[@]}"
        exp_counter=$((exp_counter + 1))
    done
fi

# 第三步：生成汇总报告
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINS=$(((TOTAL_ELAPSED % 3600) / 60))
TOTAL_SECS=$((TOTAL_ELAPSED % 60))

SUMMARY_FILE="$BASE_OUTPUT/summary_report.txt"

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "总耗时: ${TOTAL_HOURS}小时${TOTAL_MINS}分${TOTAL_SECS}秒"
echo "=========================================="
echo ""

# 生成汇总表
{
    echo "=========================================="
    echo "16组全交叉实验汇总报告"
    echo "生成时间: $(date)"
    echo "总耗时: ${TOTAL_HOURS}小时${TOTAL_MINS}分${TOTAL_SECS}秒"
    echo "=========================================="
    echo ""
    printf "%-30s | %-18s | %-18s | %-15s\n" "实验名称" "Test MRR" "Test Recall@10" "耗时(秒)"
    echo "--------------------------------------------------------------------------------------------"

    for result in "${RESULTS[@]}"; do
        IFS='|' read -r name mrr recall time <<< "$result"
        printf "%-30s | %-18s | %-18s | %-15s\n" "$name" "$mrr" "$recall" "$time"
    done

    echo "--------------------------------------------------------------------------------------------"
    echo ""
    echo "详细结果保存在: $BASE_OUTPUT/"
    echo "各实验的完整输出: $BASE_OUTPUT/{实验名称}/"
} | tee "$SUMMARY_FILE"

echo ""
echo "汇总报告已保存到: $SUMMARY_FILE"
echo ""
