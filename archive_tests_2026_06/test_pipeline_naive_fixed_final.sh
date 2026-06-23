#!/bin/bash

# Pipeline Naive测试 - 验证修改后Final Test使用Serial训练
# 预期: test_score应该接近Serial (~0.86)

SEED=1000
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=50
COARSE_EPOCHS=3
RERANK_TOP_K=15
RERANK_EPOCHS=10
OUTPUT_DIR="outputs/pipeline_naive_fixed_final/seed_${SEED}"

echo "=========================================="
echo "Pipeline Naive - Fixed Final Test"
echo "=========================================="
echo "修改: Final Test使用Serial训练（不分区）"
echo "对比: outputs/rerank_experiment_50trials/seed_1000/pipeline_naive"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode naive \
    --num-pipeline-stages 3 \
    --pipeline-stage-train-workers 1 \
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

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo "对比结果："
echo "  原实验 test_score: 0.7231"
echo "  新实验 test_score: (查看best_arch.json)"
echo "  预期: ~0.86 (接近Serial)"
echo "=========================================="
