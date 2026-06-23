#!/bin/bash
# 重跑 Pipeline Naive 和 Pipeline Smart（修正GPU配置）

set -e

BASE_DIR="outputs/nas_search_fixed"

COMMON_ARGS="--dataset public_csv \
--local-data-path data/public/mooc.csv \
--max-events 20000 \
--train-ratio 0.7 \
--val-ratio 0.1 \
--feature-dim 8 \
--lr 0.001 \
--k 10 \
--seed 42 \
--coarse-trials 27 \
--coarse-epochs 3 \
--batch-mode tbatch \
--train-batch-size 32"

echo "========================================"
echo "重跑 Pipeline Naive 和 Smart"
echo "========================================"
echo "配置:"
echo "  - Pipeline Naive: 3 stages, 1 worker/stage"
echo "  - Pipeline Smart: 1 stage, 3 workers"
echo ""

# Pipeline Naive
echo "========================================"
echo "1/2: Pipeline Naive (3 stages, 1 worker/stage)"
echo "========================================"
python -u search.py \
    $COMMON_ARGS \
    --execution-mode ray_pipeline \
    --pipeline-mode naive \
    --num-pipeline-stages 3 \
    --pipeline-stage-train-workers "1,1,1" \
    --gpu-list 0,1,2 \
    --output-dir "$BASE_DIR/pipeline_naive_tbatch" \
    2>&1 | tee "$BASE_DIR/pipeline_naive_tbatch.log"

echo ""
echo "Pipeline Naive 完成！"
echo ""

# Pipeline Smart
echo "========================================"
echo "2/2: Pipeline Smart (1 stage, 3 workers)"
echo "========================================"
python -u search.py \
    $COMMON_ARGS \
    --execution-mode ray_pipeline \
    --pipeline-mode smart \
    --num-pipeline-stages 1 \
    --pipeline-stage-train-workers "3" \
    --gpu-list 0,1,2 \
    --output-dir "$BASE_DIR/pipeline_smart_tbatch" \
    2>&1 | tee "$BASE_DIR/pipeline_smart_tbatch.log"

echo ""
echo "Pipeline Smart 完成！"
echo ""

echo "========================================"
echo "Pipeline 模式重跑完成！"
echo "========================================"
