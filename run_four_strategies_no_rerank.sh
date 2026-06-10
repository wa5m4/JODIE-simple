#!/bin/bash

# 四种NAS策略对比实验（无Rerank）
# Seed 1000, 20000 events, 50 trials, 3 epochs
# 取消Rerank，直接用coarse最佳架构做final test

SEED=1000
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=50
COARSE_EPOCHS=3
OUTPUT_BASE="outputs/no_rerank_experiment/seed_${SEED}"

echo "=========================================="
echo "四种策略对比实验（无Rerank）"
echo "=========================================="
echo "配置: ${COARSE_TRIALS} trials × ${COARSE_EPOCHS} epochs (无rerank)"
echo "Final Test: Serial训练（train+val → test）"
echo "=========================================="

mkdir -p "$OUTPUT_BASE"

# 1. Serial
echo ""
echo "[1/4] Serial (无Rerank)..."
python search.py \
    --search-mode rl \
    --execution-mode serial \
    --gpu-list "$GPU_LIST" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events "$MAX_EVENTS" \
    --seed "$SEED" \
    --coarse-trials "$COARSE_TRIALS" \
    --coarse-epochs "$COARSE_EPOCHS" \
    --output-dir "${OUTPUT_BASE}/serial" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false

echo "✓ Serial完成"

# 2. Data Parallel
echo ""
echo "[2/4] Data Parallel (无Rerank)..."
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
    --output-dir "${OUTPUT_BASE}/data_parallel" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false

echo "✓ Data Parallel完成"

# 3. Pipeline Naive
echo ""
echo "[3/4] Pipeline Naive (无Rerank)..."
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
    --output-dir "${OUTPUT_BASE}/pipeline_naive" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false

echo "✓ Pipeline Naive完成"

# 4. Pipeline Smart
echo ""
echo "[4/4] Pipeline Smart (无Rerank)..."
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
    --output-dir "${OUTPUT_BASE}/pipeline_smart" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false

echo "✓ Pipeline Smart完成"

echo ""
echo "=========================================="
echo "所有四种策略完成！"
echo "输出目录: ${OUTPUT_BASE}"
echo "=========================================="
