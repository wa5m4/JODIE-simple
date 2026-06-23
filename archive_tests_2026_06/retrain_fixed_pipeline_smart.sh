#!/bin/bash

# 为修复后的Pipeline Smart运行重训练
SEEDS=(100 200 300)
BASE_DIR="outputs/multi_seed_verification_fixed"

echo "=========================================="
echo "重训练修复后的Pipeline Smart最佳架构"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    NAS_DIR="${BASE_DIR}/seed_${SEED}/pipeline_smart"
    BEST_ARCH="${NAS_DIR}/best_arch.json"

    if [ ! -f "$BEST_ARCH" ]; then
        echo "✗ Seed $SEED: best_arch.json不存在"
        continue
    fi

    echo ""
    echo "Seed $SEED: 提取架构参数..."

    MODEL=$(python3 -c "import json; print(json.load(open('$BEST_ARCH'))['config']['model'])")
    EMB_DIM=$(python3 -c "import json; print(json.load(open('$BEST_ARCH'))['config']['embedding_dim'])")
    MEMORY_CELL=$(python3 -c "import json; print(json.load(open('$BEST_ARCH'))['config']['memory_cell'])")
    TIME_PROJ=$(python3 -c "import json; print(json.load(open('$BEST_ARCH'))['config']['time_proj'])")
    NORMALIZE_STATE=$(python3 -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('normalize_state', 'off'))")
    USE_STATIC_EMB=$(python3 -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('use_static_embeddings', 'off'))")
    RETRAIN_SEED=$(python3 -c "import json; print(json.load(open('$BEST_ARCH')).get('seed', ${SEED}))")

    echo "  架构: model=$MODEL, emb=$EMB_DIM, cell=$MEMORY_CELL, seed=$RETRAIN_SEED"

    RETRAIN_DIR="${NAS_DIR}/retrain"
    mkdir -p "$RETRAIN_DIR"

    python train_single_arch.py \
        --model "$MODEL" \
        --embedding-dim "$EMB_DIM" \
        --memory-cell "$MEMORY_CELL" \
        --time-proj "$TIME_PROJ" \
        --normalize-state "$NORMALIZE_STATE" \
        --use-static-embeddings "$USE_STATIC_EMB" \
        --batch-mode tbatch \
        --partition-size 5000 \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events 20000 \
        --epochs 3 \
        --seed "$RETRAIN_SEED" \
        --output-dir "$RETRAIN_DIR" \
        --eval-frozen false

    if [ $? -eq 0 ]; then
        echo "✓ Seed $SEED: 重训练完成"
    else
        echo "✗ Seed $SEED: 重训练失败"
    fi
done

echo ""
echo "=========================================="
echo "所有重训练完成！"
echo "=========================================="
