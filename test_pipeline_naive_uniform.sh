#!/bin/bash

# Pipeline Naive Uniform Partition测试
# 测试partition均匀分配（6667）对准确率的影响

SEED=1000
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=50
COARSE_EPOCHS=3
RERANK_TOP_K=15
RERANK_EPOCHS=10
OUTPUT_DIR="outputs/pipeline_naive_uniform_partition/seed_${SEED}"

echo "=========================================="
echo "Pipeline Naive Uniform Partition测试"
echo "=========================================="
echo "partition_size: 6667 (均匀分配到3个stage)"
echo "对比实验: outputs/rerank_experiment_50trials/seed_1000/pipeline_naive"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode naive \
    --num-pipeline-stages 3 \
    --pipeline-stage-train-workers 1 \
    --pipeline-worker-gpus 1.0 \
    --partition-size 6667 \
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
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "对比查看结果："
echo "  新实验: ${OUTPUT_DIR}/best_arch.json"
echo "  原实验: outputs/rerank_experiment_50trials/seed_1000/pipeline_naive/best_arch.json"
echo "=========================================="
