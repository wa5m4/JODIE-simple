#!/bin/bash
# 重跑 full_cross_experiment 中的 4 个 tbatch 实验
# 修复: 添加 --partition-size 1000

set -e

DATASET="public_csv"
LOCAL_DATA_PATH="data/public/mooc.csv"
MAX_EVENTS=20000
TRIALS=27
EPOCHS=3
GPU_LIST="0,1,2"
SEED=42
SPACE="mixed"
PARTITION_SIZE=1000  # 修复关键参数
BASE_OUTPUT="outputs/full_cross_experiment_fixed"

mkdir -p "$BASE_OUTPUT"

TOTAL_START=$(date +%s)

echo "=========================================="
echo "重跑 4 个 tbatch 实验 (partition_size=1000)"
echo "数据集: $DATASET"
echo "Partition Size: $PARTITION_SIZE"
echo "=========================================="
echo ""

# 1. serial_tbatch
echo "[1/4] Running: serial_tbatch"
python -u search.py \
    --execution-mode serial \
    --batch-mode tbatch \
    --train-batch-size 32 \
    --partition-size $PARTITION_SIZE \
    --dataset "$DATASET" \
    --local-data-path "$LOCAL_DATA_PATH" \
    --max-events $MAX_EVENTS \
    --trials $TRIALS \
    --epochs-per-trial $EPOCHS \
    --gpu-list "$GPU_LIST" \
    --seed $SEED \
    --space "$SPACE" \
    --output-dir "$BASE_OUTPUT/serial_tbatch"

echo ""
echo "[1/4] Completed: serial_tbatch"
echo ""

# 2. data_parallel_tbatch
echo "[2/4] Running: data_parallel_tbatch"
python -u search.py \
    --execution-mode data_parallel \
    --data-parallel-workers 3 \
    --batch-mode tbatch \
    --train-batch-size 32 \
    --partition-size $PARTITION_SIZE \
    --dataset "$DATASET" \
    --local-data-path "$LOCAL_DATA_PATH" \
    --max-events $MAX_EVENTS \
    --trials $TRIALS \
    --epochs-per-trial $EPOCHS \
    --gpu-list "$GPU_LIST" \
    --seed $SEED \
    --space "$SPACE" \
    --output-dir "$BASE_OUTPUT/data_parallel_tbatch"

echo ""
echo "[2/4] Completed: data_parallel_tbatch"
echo ""

# 3. pipeline_naive_tbatch (3 stages, 1 worker each)
echo "[3/4] Running: pipeline_naive_tbatch"
python -u search.py \
    --execution-mode ray_pipeline \
    --pipeline-mode naive \
    --num-pipeline-stages 3 \
    --pipeline-stage-train-workers 1,1,1 \
    --batch-mode tbatch \
    --train-batch-size 32 \
    --partition-size $PARTITION_SIZE \
    --dataset "$DATASET" \
    --local-data-path "$LOCAL_DATA_PATH" \
    --max-events $MAX_EVENTS \
    --trials $TRIALS \
    --epochs-per-trial $EPOCHS \
    --gpu-list "$GPU_LIST" \
    --seed $SEED \
    --space "$SPACE" \
    --output-dir "$BASE_OUTPUT/pipeline_naive_tbatch"

echo ""
echo "[3/4] Completed: pipeline_naive_tbatch"
echo ""

# 4. pipeline_smart_tbatch (1 stage, 3 workers)
echo "[4/4] Running: pipeline_smart_tbatch"
python -u search.py \
    --execution-mode ray_pipeline \
    --pipeline-mode smart \
    --num-pipeline-stages 1 \
    --pipeline-stage-train-workers 3 \
    --batch-mode tbatch \
    --train-batch-size 32 \
    --partition-size $PARTITION_SIZE \
    --dataset "$DATASET" \
    --local-data-path "$LOCAL_DATA_PATH" \
    --max-events $MAX_EVENTS \
    --trials $TRIALS \
    --epochs-per-trial $EPOCHS \
    --gpu-list "$GPU_LIST" \
    --seed $SEED \
    --space "$SPACE" \
    --output-dir "$BASE_OUTPUT/pipeline_smart_tbatch"

echo ""
echo "[4/4] Completed: pipeline_smart_tbatch"
echo ""

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINS=$(((TOTAL_ELAPSED % 3600) / 60))

echo "=========================================="
echo "所有 4 个 tbatch 实验完成！"
echo "总耗时: ${TOTAL_HOURS}小时${TOTAL_MINS}分"
echo "结果保存在: $BASE_OUTPUT/"
echo "=========================================="
