#!/bin/bash

# 运行Seed 42和1000的完整实验

SEEDS=(42 1000)
GPU_LIST="0,1,2"
MAX_EVENTS=20000
TRIALS=50
EPOCHS=3
PARTITION_SIZE=5000

BASE_OUTPUT="outputs/comprehensive_comparison"

echo "========================================================================"
echo "运行Seed 42和1000实验"
echo "========================================================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "种子 $SEED 开始"
    echo "========================================================================"

    SEED_DIR="$BASE_OUTPUT/seed_$SEED"
    mkdir -p "$SEED_DIR"

    # 1. Serial
    echo "[1/6] Serial"
    python search.py \
        --search-mode rl \
        --execution-mode serial \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$TRIALS" \
        --coarse-epochs "$EPOCHS" \
        --output-dir "${SEED_DIR}/serial" \
        --space rnn_only \
        --batch-mode tbatch \
        2>&1 | tee "${SEED_DIR}/serial.log"

    # 2. 数据并行
    echo "[2/6] 数据并行"
    python search.py \
        --search-mode rl \
        --execution-mode data_parallel \
        --data-parallel-workers 3 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$TRIALS" \
        --coarse-epochs "$EPOCHS" \
        --output-dir "${SEED_DIR}/data_parallel" \
        --space rnn_only \
        --batch-mode tbatch \
        2>&1 | tee "${SEED_DIR}/data_parallel.log"

    # 3. Smart+20%
    echo "[3/6] Smart+20%"
    python search.py \
        --search-mode rl \
        --execution-mode ray_pipeline \
        --pipeline-mode smart \
        --num-pipeline-stages 1 \
        --pipeline-stage-train-workers 3 \
        --pipeline-worker-gpus 1.0 \
        --partition-size "$PARTITION_SIZE" \
        --partition-overlap-ratio 0.2 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$TRIALS" \
        --coarse-epochs "$EPOCHS" \
        --output-dir "${SEED_DIR}/smart_overlap20" \
        --space rnn_only \
        --batch-mode tbatch \
        2>&1 | tee "${SEED_DIR}/smart_overlap20.log"

    # 4. Smart+0%
    echo "[4/6] Smart+0%"
    python search.py \
        --search-mode rl \
        --execution-mode ray_pipeline \
        --pipeline-mode smart \
        --num-pipeline-stages 1 \
        --pipeline-stage-train-workers 3 \
        --pipeline-worker-gpus 1.0 \
        --partition-size "$PARTITION_SIZE" \
        --partition-overlap-ratio 0.0 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$TRIALS" \
        --coarse-epochs "$EPOCHS" \
        --output-dir "${SEED_DIR}/smart_no_overlap" \
        --space rnn_only \
        --batch-mode tbatch \
        2>&1 | tee "${SEED_DIR}/smart_no_overlap.log"

    # 5. Naive+20%
    echo "[5/6] Naive+20%"
    python search.py \
        --search-mode rl \
        --execution-mode ray_pipeline \
        --pipeline-mode naive \
        --num-pipeline-stages 1 \
        --pipeline-stage-train-workers 3 \
        --pipeline-worker-gpus 1.0 \
        --partition-size "$PARTITION_SIZE" \
        --partition-overlap-ratio 0.2 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$TRIALS" \
        --coarse-epochs "$EPOCHS" \
        --output-dir "${SEED_DIR}/naive_overlap20" \
        --space rnn_only \
        --batch-mode tbatch \
        2>&1 | tee "${SEED_DIR}/naive_overlap20.log"

    # 6. Naive+0%
    echo "[6/6] Naive+0%"
    python search.py \
        --search-mode rl \
        --execution-mode ray_pipeline \
        --pipeline-mode naive \
        --num-pipeline-stages 1 \
        --pipeline-stage-train-workers 3 \
        --pipeline-worker-gpus 1.0 \
        --partition-size "$PARTITION_SIZE" \
        --partition-overlap-ratio 0.0 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$TRIALS" \
        --coarse-epochs "$EPOCHS" \
        --output-dir "${SEED_DIR}/naive_no_overlap" \
        --space rnn_only \
        --batch-mode tbatch \
        2>&1 | tee "${SEED_DIR}/naive_no_overlap.log"

    echo "✓ 种子 $SEED 完成"
done

echo ""
echo "========================================================================"
echo "所有实验完成！"
echo "========================================================================"
