#!/bin/bash
# 从中断处继续对比实验

DATASET="public_csv"
DATA_PATH="data/public/mooc.csv"
MAX_EVENTS=20000
TRIALS=27
EPOCHS=3
BATCH_SIZE=32
SEEDS=(42 123 456)
OUTPUT_BASE="outputs/tgn_compare"

mkdir -p "$OUTPUT_BASE"

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

        if [ -f "$output_dir/best_arch.json" ]; then
            local best_score=$(grep -o '"val_score": [0-9.]*' "$output_dir/best_arch.json" | head -1 | awk '{print $2}')
            local test_score=$(grep -o '"test_score": [0-9.]*' "$output_dir/best_arch.json" | head -1 | awk '{print $2}')
            echo "  最佳验证分数: $best_score, 测试分数: $test_score" | tee -a "$OUTPUT_BASE/comparison.log"
        fi
    else
        echo "❌ 失败 (退出码: $exit_code)" | tee -a "$OUTPUT_BASE/comparison.log"
    fi

    if [ "$mode" != "single" ]; then
        echo "  清理Ray会话..." | tee -a "$OUTPUT_BASE/comparison.log"
        ray stop --force > /dev/null 2>&1 || true
        sleep 2
    fi

    echo "" | tee -a "$OUTPUT_BASE/comparison.log"
}

echo "=== 从中断处继续实验 ===" | tee -a "$OUTPUT_BASE/comparison.log"
echo "继续时间: $(date)" | tee -a "$OUTPUT_BASE/comparison.log"
echo "" | tee -a "$OUTPUT_BASE/comparison.log"

# Seed 42 - 剩余实验
echo "======================================" | tee -a "$OUTPUT_BASE/comparison.log"
echo "随机种子: 42 (继续)" | tee -a "$OUTPUT_BASE/comparison.log"
echo "======================================" | tee -a "$OUTPUT_BASE/comparison.log"
echo "" | tee -a "$OUTPUT_BASE/comparison.log"

run_experiment "data_parallel" "false" 42 "0,1,2" "jodie_dataparallel_nobatch"
run_experiment "data_parallel" "true" 42 "0,1,2" "jodie_dataparallel_batch"

# Seed 123 - 全部实验
echo "======================================" | tee -a "$OUTPUT_BASE/comparison.log"
echo "随机种子: 123" | tee -a "$OUTPUT_BASE/comparison.log"
echo "======================================" | tee -a "$OUTPUT_BASE/comparison.log"
echo "" | tee -a "$OUTPUT_BASE/comparison.log"

run_experiment "single" "false" 123 "0" "jodie_single_nobatch"
run_experiment "single" "true" 123 "0" "jodie_single_batch"
run_experiment "naive" "false" 123 "0,1,2" "jodie_naive_nobatch"
run_experiment "naive" "true" 123 "0,1,2" "jodie_naive_batch"
run_experiment "smart" "false" 123 "0,1,2" "jodie_smart_nobatch"
run_experiment "smart" "true" 123 "0,1,2" "jodie_smart_batch"
run_experiment "data_parallel" "false" 123 "0,1,2" "jodie_dataparallel_nobatch"
run_experiment "data_parallel" "true" 123 "0,1,2" "jodie_dataparallel_batch"

# Seed 456 - 全部实验
echo "======================================" | tee -a "$OUTPUT_BASE/comparison.log"
echo "随机种子: 456" | tee -a "$OUTPUT_BASE/comparison.log"
echo "======================================" | tee -a "$OUTPUT_BASE/comparison.log"
echo "" | tee -a "$OUTPUT_BASE/comparison.log"

run_experiment "single" "false" 456 "0" "jodie_single_nobatch"
run_experiment "single" "true" 456 "0" "jodie_single_batch"
run_experiment "naive" "false" 456 "0,1,2" "jodie_naive_nobatch"
run_experiment "naive" "true" 456 "0,1,2" "jodie_naive_batch"
run_experiment "smart" "false" 456 "0,1,2" "jodie_smart_nobatch"
run_experiment "smart" "true" 456 "0,1,2" "jodie_smart_batch"
run_experiment "data_parallel" "false" 456 "0,1,2" "jodie_dataparallel_nobatch"
run_experiment "data_parallel" "true" 456 "0,1,2" "jodie_dataparallel_batch"

echo "======================================" | tee -a "$OUTPUT_BASE/comparison.log"
echo "所有剩余实验完成" | tee -a "$OUTPUT_BASE/comparison.log"
echo "结束时间: $(date)" | tee -a "$OUTPUT_BASE/comparison.log"
echo "======================================" | tee -a "$OUTPUT_BASE/comparison.log"
