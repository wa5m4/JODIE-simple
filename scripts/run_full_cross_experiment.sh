#!/bin/bash
# 16组全交叉实验：4种执行架构 × 4种批处理策略
# 执行架构: serial, data_parallel, pipeline_naive, pipeline_smart
# 批处理策略: serial, tbatch, tgn-last, tgn-all

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
echo "16组全交叉实验开始"
echo "数据集: $DATASET ($LOCAL_DATA_PATH)"
echo "事件数: $MAX_EVENTS, Trials: $TRIALS, Epochs: $EPOCHS"
echo "GPU: $GPU_LIST, Seed: $SEED, Space: $SPACE"
echo "=========================================="
echo ""

# 实验计数器（从3开始，因为前2个已完成）
EXPERIMENT_NUM=2
TOTAL_EXPERIMENTS=16

# 存储结果的数组
declare -a RESULTS

# 函数：运行单个实验
run_experiment() {
    local exec_mode=$1
    local exec_params=$2
    local batch_mode=$3
    local batch_params=$4
    local output_name=$5

    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))

    local output_dir="$BASE_OUTPUT/$output_name"

    # 检查是否已完成
    if [ -f "$output_dir/best_arch.json" ]; then
        echo ""
        echo "=========================================="
        echo "[$EXPERIMENT_NUM/$TOTAL_EXPERIMENTS] 跳过: $output_name (已完成)"
        echo "=========================================="
        # 提取已有结果
        local test_mrr=$(python -c "import json; print(json.load(open('$output_dir/best_arch.json'))['test_mrr'])" 2>/dev/null || echo "N/A")
        local test_recall=$(python -c "import json; print(json.load(open('$output_dir/best_arch.json'))['test_recall_at_k'])" 2>/dev/null || echo "N/A")
        RESULTS+=("$output_name|$test_mrr|$test_recall|0")
        return
    fi

    echo ""
    echo "=========================================="
    echo "[$EXPERIMENT_NUM/$TOTAL_EXPERIMENTS] 实验: $output_name"
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
    echo "[$EXPERIMENT_NUM/$TOTAL_EXPERIMENTS] 完成: $output_name"
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

# ========== Serial 执行模式 (4组) ==========
EXEC_MODE="serial"
EXEC_PARAMS="--execution-mode serial"

# 1. serial + serial (已完成，跳过)
# run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "serial" "--batch-mode serial" "serial_serial"

# 2. serial + tbatch (已完成，跳过)
# run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "tbatch" "--batch-mode tbatch --train-batch-size 32" "serial_tbatch"

# 3. serial + tgn-last (从这里重新开始)
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "tgn-last" "--batch-mode tgn --tgn-loss-mode last --tgn-window-size 10.0" "serial_tgn_last"

# 4. serial + tgn-all
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "tgn-all" "--batch-mode tgn --tgn-loss-mode all --tgn-window-size 10.0" "serial_tgn_all"

# ========== Data Parallel 执行模式 (4组) ==========
EXEC_MODE="data_parallel"
EXEC_PARAMS="--execution-mode data_parallel --data-parallel-workers 3"

# 5. data_parallel + serial
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "serial" "--batch-mode serial" "data_parallel_serial"

# 6. data_parallel + tbatch
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "tbatch" "--batch-mode tbatch --train-batch-size 32" "data_parallel_tbatch"

# 7. data_parallel + tgn-last
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "tgn-last" "--batch-mode tgn --tgn-loss-mode last --tgn-window-size 10.0" "data_parallel_tgn_last"

# 8. data_parallel + tgn-all
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "tgn-all" "--batch-mode tgn --tgn-loss-mode all --tgn-window-size 10.0" "data_parallel_tgn_all"

# ========== Pipeline Naive 执行模式 (4组) ==========
EXEC_MODE="pipeline_naive"
EXEC_PARAMS="--execution-mode ray_pipeline --pipeline-mode naive --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1"

# 9. pipeline_naive + serial
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "serial" "--batch-mode serial" "pipeline_naive_serial"

# 10. pipeline_naive + tbatch
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "tbatch" "--batch-mode tbatch --train-batch-size 32" "pipeline_naive_tbatch"

# 11. pipeline_naive + tgn-last
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "tgn-last" "--batch-mode tgn --tgn-loss-mode last --tgn-window-size 10.0" "pipeline_naive_tgn_last"

# 12. pipeline_naive + tgn-all
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "tgn-all" "--batch-mode tgn --tgn-loss-mode all --tgn-window-size 10.0" "pipeline_naive_tgn_all"

# ========== Pipeline Smart 执行模式 (4组) ==========
EXEC_MODE="pipeline_smart"
EXEC_PARAMS="--execution-mode ray_pipeline --pipeline-mode smart --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1"

# 13. pipeline_smart + serial
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "serial" "--batch-mode serial" "pipeline_smart_serial"

# 14. pipeline_smart + tbatch
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "tbatch" "--batch-mode tbatch --train-batch-size 32" "pipeline_smart_tbatch"

# 15. pipeline_smart + tgn-last
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "tgn-last" "--batch-mode tgn --tgn-loss-mode last --tgn-window-size 10.0" "pipeline_smart_tgn_last"

# 16. pipeline_smart + tgn-all
run_experiment "$EXEC_MODE" "$EXEC_PARAMS" "tgn-all" "--batch-mode tgn --tgn-loss-mode all --tgn-window-size 10.0" "pipeline_smart_tgn_all"

# ========== 生成汇总报告 ==========
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINS=$(((TOTAL_ELAPSED % 3600) / 60))
TOTAL_SECS=$((TOTAL_ELAPSED % 60))

SUMMARY_FILE="$BASE_OUTPUT/summary_report.txt"

echo ""
echo "=========================================="
echo "16组实验全部完成！"
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
    printf "%-30s | %-12s | %-12s | %-15s\n" "实验名称" "Test MRR" "Test Recall@10" "耗时(秒)"
    echo "--------------------------------------------------------------------------------"

    for result in "${RESULTS[@]}"; do
        IFS='|' read -r name mrr recall time <<< "$result"
        printf "%-30s | %-12s | %-12s | %-15s\n" "$name" "$mrr" "$recall" "$time"
    done

    echo "--------------------------------------------------------------------------------"
    echo ""
    echo "详细结果保存在: $BASE_OUTPUT/"
    echo "各实验的完整输出: $BASE_OUTPUT/{实验名称}/"
} | tee "$SUMMARY_FILE"

echo ""
echo "汇总报告已保存到: $SUMMARY_FILE"
echo ""
