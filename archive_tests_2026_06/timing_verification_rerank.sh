#!/bin/bash

# Pipeline Smart with Rerank验证实验

SEED=1000
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=27
COARSE_EPOCHS=3
RERANK_TOP_K=10
RERANK_EPOCHS=10
OUTPUT_DIR="outputs/timing_verification/pipeline_smart_rerank_seed1000"

echo "=========================================="
echo "Pipeline Smart + Rerank 验证实验"
echo "=========================================="
echo "配置:"
echo "  Coarse: $COARSE_TRIALS trials × $COARSE_EPOCHS epochs"
echo "  Rerank: top $RERANK_TOP_K × $RERANK_EPOCHS epochs"
echo "  Pipeline: 1 stage, 3 workers"
echo "=========================================="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

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
    --rerank-top-k "$RERANK_TOP_K" \
    --rerank-epochs "$RERANK_EPOCHS" \
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
echo "总耗时: ${MINUTES}分${SECONDS}秒"
echo "=========================================="
