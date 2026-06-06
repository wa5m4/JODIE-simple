#!/bin/bash
# 离线评估测试：验证Pipeline是否会选择更好的架构
# 配置: 20000数据, 27trials, 3epochs, GPUs 0,1,2, frozen=true

set -e

DATASET="public_csv"
LOCAL_DATA_PATH="data/public/mooc.csv"
MAX_EVENTS=20000
TRIALS=27
EPOCHS=3
GPU_LIST="0,1,2"
SEED=20042
SPACE="mixed"
EVAL_FROZEN="true"  # 离线评估
BASE_OUTPUT="outputs/offline_eval_test"

mkdir -p "$BASE_OUTPUT"

echo "=========================================="
echo "离线评估测试"
echo "数据: $MAX_EVENTS, Trials: $TRIALS, Epochs: $EPOCHS"
echo "GPU: $GPU_LIST, Seed: $SEED"
echo "评估模式: 离线 (frozen=$EVAL_FROZEN)"
echo "=========================================="
echo ""

# 测试4种模式
MODES=("serial" "data_parallel" "pipeline_naive" "pipeline_smart")

for mode in "${MODES[@]}"; do
    echo ""
    echo "=========================================="
    echo "测试模式: $mode"
    echo "=========================================="

    output_dir="$BASE_OUTPUT/$mode"

    # 设置执行参数
    if [ "$mode" = "serial" ]; then
        exec_params="--execution-mode serial"
    elif [ "$mode" = "data_parallel" ]; then
        exec_params="--execution-mode data_parallel --data-parallel-workers 3"
    elif [ "$mode" = "pipeline_naive" ]; then
        exec_params="--execution-mode ray_pipeline --pipeline-mode naive --num-pipeline-stages 2"
    elif [ "$mode" = "pipeline_smart" ]; then
        exec_params="--execution-mode ray_pipeline --pipeline-mode smart --num-pipeline-stages 2"
    fi

    # NAS搜索
    echo "步骤1: NAS搜索..."
    python -u search.py \
        $exec_params \
        --dataset "$DATASET" \
        --local-data-path "$LOCAL_DATA_PATH" \
        --max-events $MAX_EVENTS \
        --trials $TRIALS \
        --epochs-per-trial $EPOCHS \
        --gpu-list "$GPU_LIST" \
        --seed $SEED \
        --space "$SPACE" \
        --eval-frozen "$EVAL_FROZEN" \
        --output-dir "$output_dir"

    # 提取最佳架构
    if [ -f "$output_dir/best_arch.json" ]; then
        nas_mrr=$(python -c "import json; print(json.load(open('$output_dir/best_arch.json'))['test_mrr'])")
        time_proj=$(python -c "import json; print(json.load(open('$output_dir/best_arch.json'))['config']['time_proj'])")
        echo "NAS完成: MRR=$nas_mrr, time_proj=$time_proj"
    else
        echo "❌ NAS失败"
    fi
    echo ""
done

echo "=========================================="
echo "测试完成！查看结果:"
echo "  $BASE_OUTPUT/"
echo "=========================================="
