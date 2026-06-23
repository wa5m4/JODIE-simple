#!/bin/bash

# Pipeline Smart计时验证实验
# 配置与之前完全相同

SEED=999
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=27
COARSE_EPOCHS=3
OUTPUT_DIR="outputs/timing_verification/pipeline_smart_seed999"

echo "=========================================="
echo "Pipeline Smart 计时验证实验"
echo "=========================================="
echo "配置:"
echo "  种子: $SEED"
echo "  GPU: $GPU_LIST"
echo "  数据: $MAX_EVENTS events"
echo "  Trials: $COARSE_TRIALS"
echo "  Epochs: $COARSE_EPOCHS"
echo "  Pipeline: 1 stage, 3 workers"
echo "  Partition size: 5000"
echo "=========================================="
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

START_TIME=$(date +%s)

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode smart \
    --num-pipeline-stages 1 \
    --pipeline-stage-train-workers 3 \
    --pipeline-worker-gpus 1.0 \
    --partition-size 5000 \
    --gpu-list "$GPU_LIST" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events "$MAX_EVENTS" \
    --seed "$SEED" \
    --coarse-trials "$COARSE_TRIALS" \
    --coarse-epochs "$COARSE_EPOCHS" \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "总耗时: ${MINUTES}分${SECONDS}秒 (${ELAPSED}秒)"
echo "=========================================="
