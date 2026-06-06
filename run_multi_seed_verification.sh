#!/bin/bash

# 多种子验证实验
# GPU: 0,1,2
# 数据: 20000
# Trials: 27
# Epochs: 3

SEEDS=(100 200 300)
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=27
COARSE_EPOCHS=3
OUTPUT_BASE="outputs/multi_seed_verification"

echo "=========================================="
echo "多种子验证实验"
echo "Seeds: ${SEEDS[@]}"
echo "GPU: $GPU_LIST"
echo "数据: $MAX_EVENTS events"
echo "Trials: $COARSE_TRIALS, Epochs: $COARSE_EPOCHS"
echo "=========================================="
echo

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "启动 Seed=$SEED 的验证"
    echo "=========================================="

    SEED_OUTPUT="${OUTPUT_BASE}/seed_${SEED}"
    mkdir -p "$SEED_OUTPUT"

    # Serial模式
    echo "1. Serial模式..."
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
        --output-dir "${SEED_OUTPUT}/serial" \
        --space rnn_only \
        --batch-mode tbatch \
        --eval-frozen false

    if [ $? -eq 0 ]; then
        echo "✓ Serial NAS完成"
        # 重训练
        BEST_ARCH="${SEED_OUTPUT}/serial/best_arch.json"
        MODEL=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['model'])")
        EMB_DIM=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['embedding_dim'])")
        MEMORY_CELL=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['memory_cell'])")
        TIME_PROJ=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['time_proj'])")
        NORMALIZE_STATE=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('normalize_state', 'off'))")
        USE_STATIC_EMB=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('use_static_embeddings', 'off'))")
        RETRAIN_SEED=$(python -c "import json; print(json.load(open('$BEST_ARCH')).get('seed', ${SEED}))")

        python train_single_arch.py \
            --model "$MODEL" \
            --embedding-dim "$EMB_DIM" \
            --memory-cell "$MEMORY_CELL" \
            --time-proj "$TIME_PROJ" \
            --normalize-state "$NORMALIZE_STATE" \
            --use-static-embeddings "$USE_STATIC_EMB" \
            --batch-mode tbatch \
            --partition-size 0 \
            --dataset public_csv \
            --local-data-path data/public/mooc.csv \
            --max-events "$MAX_EVENTS" \
            --epochs "$COARSE_EPOCHS" \
            --seed "$RETRAIN_SEED" \
            --output-dir "${SEED_OUTPUT}/serial/retrain" \
            --eval-frozen false

        echo "✓ Serial重训练完成"
    else
        echo "✗ Serial失败"
    fi

    echo
done

    # Data Parallel模式
    echo "2. Data Parallel模式..."
    python search.py \
        --search-mode rl \
        --execution-mode data_parallel \
        --num-workers 3 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$COARSE_TRIALS" \
        --coarse-epochs "$COARSE_EPOCHS" \
        --output-dir "${SEED_OUTPUT}/data_parallel" \
        --space rnn_only \
        --batch-mode tbatch \
        --eval-frozen false

    if [ $? -eq 0 ]; then
        echo "✓ Data Parallel NAS完成"
        # 重训练
        BEST_ARCH="${SEED_OUTPUT}/data_parallel/best_arch.json"
        MODEL=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['model'])")
        EMB_DIM=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['embedding_dim'])")
        MEMORY_CELL=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['memory_cell'])")
        TIME_PROJ=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['time_proj'])")
        NORMALIZE_STATE=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('normalize_state', 'off'))")
        USE_STATIC_EMB=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('use_static_embeddings', 'off'))")
        RETRAIN_SEED=$(python -c "import json; print(json.load(open('$BEST_ARCH')).get('seed', ${SEED}))")

        python train_single_arch.py \
            --model "$MODEL" \
            --embedding-dim "$EMB_DIM" \
            --memory-cell "$MEMORY_CELL" \
            --time-proj "$TIME_PROJ" \
            --normalize-state "$NORMALIZE_STATE" \
            --use-static-embeddings "$USE_STATIC_EMB" \
            --batch-mode tbatch \
            --partition-size 0 \
            --dataset public_csv \
            --local-data-path data/public/mooc.csv \
            --max-events "$MAX_EVENTS" \
            --epochs "$COARSE_EPOCHS" \
            --seed "$RETRAIN_SEED" \
            --output-dir "${SEED_OUTPUT}/data_parallel/retrain" \
            --eval-frozen false

        echo "✓ Data Parallel重训练完成"
    else
        echo "✗ Data Parallel失败"
    fi

    echo
done

echo "=========================================="
echo "所有种子验证完成！"
echo "=========================================="
