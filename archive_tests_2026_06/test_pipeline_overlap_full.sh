#!/bin/bash

# Pipeline方案2：Partition重叠预热（20%重叠）

SEED=1000
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=50
COARSE_EPOCHS=3
OUTPUT_DIR="outputs/pipeline_overlap_20pct/seed_${SEED}"

echo "=========================================="
echo "Pipeline方案2：Partition重叠预热"
echo "=========================================="
echo "改进: 20%重叠 + 原始reset机制"
echo "预期: 缓解Cold Start，选出time=off"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

echo ""
echo "运行Pipeline Smart（20%重叠）..."

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode smart \
    --num-pipeline-stages 1 \
    --pipeline-stage-train-workers 3 \
    --pipeline-worker-gpus 1.0 \
    --partition-size 5000 \
    --partition-overlap-ratio 0.2 \
    --gpu-list "$GPU_LIST" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events "$MAX_EVENTS" \
    --seed "$SEED" \
    --coarse-trials "$COARSE_TRIALS" \
    --coarse-epochs "$COARSE_EPOCHS" \
    --output-dir "${OUTPUT_DIR}/pipeline_smart_overlap" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo ""
echo "对比结果："
echo "  原版（无重叠）: time=linear, Test=0.6631"
echo "  20%重叠版:     查看 ${OUTPUT_DIR}/pipeline_smart_overlap/best_arch.json"
echo "=========================================="
