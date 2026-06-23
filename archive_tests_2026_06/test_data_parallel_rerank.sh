#!/bin/bash

# Data Parallel + Rerank 最小测试
# 验证rerank功能是否正常工作

SEED=1000
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=3
COARSE_EPOCHS=3
RERANK_TOP_K=2
RERANK_EPOCHS=5
OUTPUT_DIR="outputs/test_dp_rerank"

echo "=========================================="
echo "Data Parallel + Rerank 最小测试"
echo "=========================================="
echo "配置: ${COARSE_TRIALS} trials × ${COARSE_EPOCHS} epochs + rerank top${RERANK_TOP_K} × ${RERANK_EPOCHS} epochs"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode data_parallel \
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

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
