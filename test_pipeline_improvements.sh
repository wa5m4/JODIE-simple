#!/bin/bash

# Pipeline改进验证实验
# 方案1：Epoch间状态持久化（默认启用）
# 方案2：Partition重叠预热（20%重叠）

SEED=1000
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=50
COARSE_EPOCHS=3
OUTPUT_DIR="outputs/pipeline_improved/seed_${SEED}"

echo "=========================================="
echo "Pipeline改进验证实验"
echo "=========================================="
echo "改进1: Epoch间状态持久化（已默认启用）"
echo "改进2: Partition重叠预热（20%重叠）"
echo "预期: 选出正确架构(time=off)，test≈0.86"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

echo ""
echo "运行Pipeline Smart（改进版）..."
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
    --output-dir "${OUTPUT_DIR}/pipeline_smart_improved" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo ""
echo "对比结果："
echo "  原版Pipeline Smart:"
echo "    - 选出架构: time=linear"
echo "    - Test MRR: 0.6093"
echo ""
echo "  改进版Pipeline Smart:"
echo "    - 查看: ${OUTPUT_DIR}/pipeline_smart_improved/best_arch.json"
echo "    - 预期选出: time=off"
echo "    - 预期Test MRR: ~0.86"
echo "=========================================="
