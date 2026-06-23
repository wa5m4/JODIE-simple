#!/bin/bash
# 完整对比实验：批处理 vs 非批处理
# 参数：20000数据，27trials，3epochs，3个随机种子，GPU 0-1-2

DATASET="public_csv"
DATA_PATH="data/public/mooc.csv"
MAX_EVENTS=20000
TRIALS=27
EPOCHS=3
BATCH_SIZE=32
SEEDS=(42 123 456)
OUTPUT_BASE="outputs/tgn_compare"

mkdir -p "$OUTPUT_BASE"

# 记录开始时间
echo "=== 批处理对比实验开始 ===" | tee "$OUTPUT_BASE/comparison.log"
echo "开始时间: $(date)" | tee -a "$OUTPUT_BASE/comparison.log"
echo "" | tee -a "$OUTPUT_BASE/comparison.log"

# 函数：运行单个实验
run_experiment() {
    local mode=$1
    local batch_flag=$2
    local seed=$3
    local gpu=$4
    local output_suffix=$5

    local output_dir="$OUTPUT_BASE/seed_${seed}/${output_suffix}"

    echo ">>> 运行: mode=$mode, batch=$batch_flag, seed=$seed, gpu=$gpu" | tee -a "$OUTPUT_BASE/comparison.log"

    local cmd="CUDA_VISIBLE_DEVICES=$gpu python -u search.py \
        --dataset $DATASET \
        --local-data-path $DATA_PATH \
        --max-events $MAX_EVENTS \
        --trials $TRIALS \
        --epochs $EPOCHS \
        --output-dir $output_dir \
        --seed $seed"

    if [ "$mode" = "data_parallel" ]; then
        cmd="$cmd --execution-mode data_parallel --data-parallel-workers 3 --data-parallel-visible-gpus \"$gpu\""
    elif [ "$mode" != "single" ]; then
        cmd="$cmd --pipeline-mode $mode"
    fi

    if [ "$batch_flag" = "true" ]; then
        cmd="$cmd --batch-training --train-batch-size $BATCH_SIZE"
    fi

    echo "命令: $cmd" | tee -a "$OUTPUT_BASE/comparison.log"

    mkdir -p "$output_dir"
    local start_time=$(date +%s)
    eval $cmd 2>&1 | tee "$output_dir/run.log"
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        echo "✅ 完成 (耗时: ${duration}s)" | tee -a "$OUTPUT_BASE/comparison.log"

        # 提取最佳结果
        if [ -f "$output_dir/best_arch.json" ]; then
            local best_score=$(grep -o '"val_score": [0-9.]*' "$output_dir/best_arch.json" | head -1 | awk '{print $2}')
            local test_score=$(grep -o '"test_score": [0-9.]*' "$output_dir/best_arch.json" | head -1 | awk '{print $2}')
            echo "  最佳验证分数: $best_score, 测试分数: $test_score" | tee -a "$OUTPUT_BASE/comparison.log"
        fi
    else
        echo "❌ 失败 (退出码: $exit_code)" | tee -a "$OUTPUT_BASE/comparison.log"
    fi

    # 清理Ray会话（如果使用了pipeline或data_parallel模式）
    if [ "$mode" != "single" ]; then
        echo "  清理Ray会话..." | tee -a "$OUTPUT_BASE/comparison.log"
        ray stop --force > /dev/null 2>&1 || true
        sleep 2
    fi

    echo "" | tee -a "$OUTPUT_BASE/comparison.log"
}

# 实验矩阵
# 每个seed运行8个实验：4种模式 × 2种批处理设置

for seed in "${SEEDS[@]}"; do
    echo "======================================" | tee -a "$OUTPUT_BASE/comparison.log"
    echo "随机种子: $seed" | tee -a "$OUTPUT_BASE/comparison.log"
    echo "======================================" | tee -a "$OUTPUT_BASE/comparison.log"
    echo "" | tee -a "$OUTPUT_BASE/comparison.log"

    # 1. 单GPU - 非批处理
    run_experiment "single" "false" $seed "0" "jodie_single_nobatch"

    # 2. 单GPU - 批处理
    run_experiment "single" "true" $seed "0" "jodie_single_batch"

    # 3. Native pipeline - 非批处理
    run_experiment "naive" "false" $seed "0,1,2" "jodie_naive_nobatch"

    # 4. Native pipeline - 批处理
    run_experiment "naive" "true" $seed "0,1,2" "jodie_naive_batch"

    # 5. Smart pipeline - 非批处理
    run_experiment "smart" "false" $seed "0,1,2" "jodie_smart_nobatch"

    # 6. Smart pipeline - 批处理
    run_experiment "smart" "true" $seed "0,1,2" "jodie_smart_batch"

    # 7. 数据并行 - 非批处理
    run_experiment "data_parallel" "false" $seed "0,1,2" "jodie_dataparallel_nobatch"

    # 8. 数据并行 - 批处理
    run_experiment "data_parallel" "true" $seed "0,1,2" "jodie_dataparallel_batch"
done

echo "======================================" | tee -a "$OUTPUT_BASE/comparison.log"
echo "所有实验完成" | tee -a "$OUTPUT_BASE/comparison.log"
echo "结束时间: $(date)" | tee -a "$OUTPUT_BASE/comparison.log"
echo "======================================" | tee -a "$OUTPUT_BASE/comparison.log"

# 生成汇总报告
echo "" | tee -a "$OUTPUT_BASE/comparison.log"
echo "=== 结果汇总 ===" | tee -a "$OUTPUT_BASE/comparison.log"
echo "" | tee -a "$OUTPUT_BASE/comparison.log"

for seed in "${SEEDS[@]}"; do
    echo "种子 $seed:" | tee -a "$OUTPUT_BASE/comparison.log"
    for exp_dir in "$OUTPUT_BASE/seed_${seed}"/*; do
        if [ -d "$exp_dir" ] && [ -f "$exp_dir/best_arch.json" ]; then
            exp_name=$(basename "$exp_dir")
            val_score=$(grep -o '"val_score": [0-9.]*' "$exp_dir/best_arch.json" | head -1 | awk '{print $2}')
            test_score=$(grep -o '"test_score": [0-9.]*' "$exp_dir/best_arch.json" | head -1 | awk '{print $2}')
            echo "  $exp_name: val=$val_score, test=$test_score" | tee -a "$OUTPUT_BASE/comparison.log"
        fi
    done
    echo "" | tee -a "$OUTPUT_BASE/comparison.log"
done

echo "详细日志保存在: $OUTPUT_BASE/comparison.log"
