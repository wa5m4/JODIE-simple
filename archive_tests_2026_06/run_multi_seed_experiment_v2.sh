#!/bin/bash
# 多种子实验：3个种子 × 4种执行模式
# 种子: 20042, 12345, 67890
# 执行模式: serial, data_parallel, pipeline_naive, pipeline_smart

set -e

# 共用参数
DATASET="public_csv"
LOCAL_DATA_PATH="data/public/mooc.csv"
MAX_EVENTS=20000
TRIALS=27
EPOCHS=3
GPU_LIST="0,1,2"
SPACE="mixed"
EVAL_FROZEN="false"
BASE_OUTPUT="outputs/multi_seed_experiment"

# 种子列表
SEEDS=(20042 12345 67890)

# 创建输出目录
mkdir -p "$BASE_OUTPUT"

# 记录总开始时间
TOTAL_START=$(date +%s)

echo "=========================================="
echo "多种子实验开始"
echo "数据集: $DATASET ($LOCAL_DATA_PATH)"
echo "事件数: $MAX_EVENTS, Trials: $TRIALS, Epochs: $EPOCHS"
echo "GPU: $GPU_LIST, 种子: ${SEEDS[*]}"
echo "评估模式: 在线 (frozen=$EVAL_FROZEN)"
echo "=========================================="
echo ""

# 实验计数器
EXPERIMENT_NUM=0
TOTAL_EXPERIMENTS=12  # 3 seeds × 4 modes

# 存储结果的数组
declare -a RESULTS

# 函数：运行单个实验（NAS搜索 + 重训）
run_experiment() {
    local seed=$1
    local exec_mode=$2
    local exec_params=$3
    local output_name=$4

    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))

    local output_dir="$BASE_OUTPUT/seed_${seed}/$exec_mode"
    local retrain_dir="$output_dir/retrain"

    echo ""
    echo "=========================================="
    echo "[$EXPERIMENT_NUM/$TOTAL_EXPERIMENTS] 实验: seed=$seed, mode=$exec_mode"
    echo "=========================================="

    # 检查是否已完成
    local nas_done=false
    local retrain_done=false

    if [ -f "$output_dir/best_arch.json" ]; then
        nas_done=true
        echo "✓ NAS搜索已完成"
    fi

    if [ -f "$retrain_dir/result.json" ]; then
        retrain_done=true
        echo "✓ 重训已完成"
    fi

    # 如果全部完成，跳过
    if [ "$nas_done" = true ] && [ "$retrain_done" = true ]; then
        echo "跳过: 实验已完成"
        local nas_mrr=$(python -c "import json; print(json.load(open('$output_dir/best_arch.json'))['test_mrr'])")
        local retrain_mrr=$(python -c "import json; print(json.load(open('$retrain_dir/result.json'))['test_mrr'])")
        RESULTS+=("$output_name|$nas_mrr|$retrain_mrr|0|0|0")
        echo "NAS MRR: $nas_mrr, 重训 MRR: $retrain_mrr"
        echo ""
        return
    fi

    # ===== 步骤1: NAS搜索 =====
    local nas_elapsed=0
    if [ "$nas_done" = false ]; then
        echo "步骤1: NAS搜索..."
        local nas_start=$(date +%s)

    python -u search.py \
        $exec_params \
        --dataset "$DATASET" \
        --local-data-path "$LOCAL_DATA_PATH" \
        --max-events $MAX_EVENTS \
        --trials $TRIALS \
        --epochs-per-trial $EPOCHS \
        --gpu-list "$GPU_LIST" \
        --seed $seed \
        --space "$SPACE" \
        --eval-frozen "$EVAL_FROZEN" \
        --output-dir "$output_dir"

        local nas_end=$(date +%s)
        nas_elapsed=$((nas_end - nas_start))
        echo "NAS搜索完成，耗时: ${nas_elapsed}秒"
    else
        echo "步骤1: 跳过NAS搜索 (已完成)"
    fi

    # ===== 步骤2: 提取最佳架构 =====
    echo "步骤2: 提取最佳架构..."

    if [ ! -f "$output_dir/best_arch.json" ]; then
        echo "❌ 错误: 未找到best_arch.json"
        RESULTS+=("$output_name|ERROR|ERROR|$nas_elapsed|0")
        return
    fi

    local nas_mrr=$(python -c "import json; print(json.load(open('$output_dir/best_arch.json'))['test_mrr'])")
    echo "NAS最佳MRR: $nas_mrr"

    # 提取架构参数
    local model=$(python -c "import json; print(json.load(open('$output_dir/best_arch.json'))['config']['model'])")
    local emb_dim=$(python -c "import json; print(json.load(open('$output_dir/best_arch.json'))['config']['embedding_dim'])")
    local mem_cell=$(python -c "import json; print(json.load(open('$output_dir/best_arch.json'))['config']['memory_cell'])")
    local time_proj=$(python -c "import json; print(json.load(open('$output_dir/best_arch.json'))['config']['time_proj'])")

    echo "架构: model=$model, emb_dim=$emb_dim, mem_cell=$mem_cell, time_proj=$time_proj"

    # ===== 步骤3: 重训 =====
    local retrain_elapsed=0
    if [ "$retrain_done" = false ]; then
        echo "步骤3: 重训最佳架构..."
        mkdir -p "$retrain_dir"

        local retrain_start=$(date +%s)

    python -u train_single_arch.py \
        --dataset "$DATASET" \
        --local-data-path "$LOCAL_DATA_PATH" \
        --max-events $MAX_EVENTS \
        --epochs $EPOCHS \
        --seed $seed \
        --eval-frozen "$EVAL_FROZEN" \
        --output-dir "$retrain_dir" \
        --model "$model" \
        --embedding-dim $emb_dim \
        --memory-cell "$mem_cell" \
        --time-proj "$time_proj"

        local retrain_end=$(date +%s)
        retrain_elapsed=$((retrain_end - retrain_start))
        echo "重训完成，耗时: ${retrain_elapsed}秒"
    else
        echo "步骤3: 跳过重训 (已完成)"
    fi

    # 提取重训结果
    if [ -f "$retrain_dir/result.json" ]; then
        local retrain_mrr=$(python -c "import json; print(json.load(open('$retrain_dir/result.json'))['test_mrr'])")
        echo "重训MRR: $retrain_mrr"

        local total_elapsed=$((nas_elapsed + retrain_elapsed))
        RESULTS+=("$output_name|$nas_mrr|$retrain_mrr|$nas_elapsed|$retrain_elapsed|$total_elapsed")

        echo ""
        echo "[$EXPERIMENT_NUM/$TOTAL_EXPERIMENTS] 完成: $output_name"
        echo "NAS MRR: $nas_mrr, 重训 MRR: $retrain_mrr"
        echo "总耗时: ${total_elapsed}秒 (NAS: ${nas_elapsed}s, 重训: ${retrain_elapsed}s)"
    else
        echo "❌ 错误: 未找到重训结果"
        RESULTS+=("$output_name|$nas_mrr|ERROR|$nas_elapsed|$retrain_elapsed|0")
    fi
    echo ""
}

# ========== 主循环：遍历所有种子和模式 ==========

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "################################################################################"
    echo "# SEED: $seed"
    echo "################################################################################"
    echo ""

    # Serial
    run_experiment $seed "serial" "--execution-mode serial" "seed_${seed}_serial"

    # Data Parallel
    run_experiment $seed "data_parallel" "--execution-mode data_parallel --data-parallel-workers 3" "seed_${seed}_data_parallel"

    # Pipeline Naive
    run_experiment $seed "pipeline_naive" "--execution-mode ray_pipeline --pipeline-mode naive --num-pipeline-stages 2" "seed_${seed}_pipeline_naive"

    # Pipeline Smart
    run_experiment $seed "pipeline_smart" "--execution-mode ray_pipeline --pipeline-mode smart --num-pipeline-stages 2" "seed_${seed}_pipeline_smart"
done

# ========== 生成汇总报告 ==========
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINS=$(((TOTAL_ELAPSED % 3600) / 60))
TOTAL_SECS=$((TOTAL_ELAPSED % 60))

SUMMARY_FILE="$BASE_OUTPUT/SUMMARY_REPORT.txt"

echo ""
echo "=========================================="
echo "多种子实验全部完成！"
echo "总耗时: ${TOTAL_HOURS}小时${TOTAL_MINS}分${TOTAL_SECS}秒"
echo "=========================================="
echo ""

# 生成汇总表
{
    echo "=========================================="
    echo "多种子实验汇总报告"
    echo "生成时间: $(date)"
    echo "总耗时: ${TOTAL_HOURS}小时${TOTAL_MINS}分${TOTAL_SECS}秒"
    echo "=========================================="
    echo ""
    printf "%-25s | %-12s | %-12s | %-10s | %-10s | %-10s\n" "实验名称" "NAS MRR" "重训 MRR" "NAS耗时(s)" "重训耗时(s)" "总耗时(s)"
    echo "--------------------------------------------------------------------------------------------------------"

    for result in "${RESULTS[@]}"; do
        IFS='|' read -r name nas_mrr retrain_mrr nas_time retrain_time total_time <<< "$result"
        printf "%-25s | %-12s | %-12s | %-10s | %-10s | %-10s\n" "$name" "$nas_mrr" "$retrain_mrr" "$nas_time" "$retrain_time" "$total_time"
    done

    echo "--------------------------------------------------------------------------------------------------------"
    echo ""
    echo "详细结果保存在: $BASE_OUTPUT/"
} | tee "$SUMMARY_FILE"

echo ""
echo "汇总报告已保存到: $SUMMARY_FILE"
echo ""
