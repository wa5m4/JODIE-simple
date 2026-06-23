#!/bin/bash

# 重训脚本 - Bug修复验证的4个架构

SEED=42
OUTPUT_BASE="outputs/bug_fix_verification/seed_${SEED}"

echo "=========================================="
echo "Retraining Bug Fix Verification Architectures"
echo "=========================================="

MODES=("serial" "data_parallel" "pipeline_naive" "pipeline_smart")

for MODE in "${MODES[@]}"; do
    BEST_ARCH="${OUTPUT_BASE}/${MODE}/best_arch.json"
    RETRAIN_OUTPUT="${OUTPUT_BASE}/${MODE}/retrain"

    if [ ! -f "$BEST_ARCH" ]; then
        echo "✗ ${MODE}: best_arch.json not found"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Retraining: ${MODE}"
    echo "=========================================="

    # 从best_arch.json提取配置参数
    MODEL=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['model'])")
    EMB_DIM=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['embedding_dim'])")
    MEMORY_CELL=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['memory_cell'])")
    TIME_PROJ=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['time_proj'])")
    BATCH_MODE=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('batch_mode', 'tbatch'))")
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
        --batch-mode "$BATCH_MODE" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events 20000 \
        --epochs 3 \
        --seed "$RETRAIN_SEED" \
        --output-dir "$RETRAIN_OUTPUT" \
        --eval-frozen false

    echo "  Used seed: $RETRAIN_SEED (from NAS Final Test)"

    if [ $? -eq 0 ]; then
        echo "✓ ${MODE} retrain completed"
    else
        echo "✗ ${MODE} retrain failed"
    fi
done

echo ""
echo "=========================================="
echo "Retrain Results Summary"
echo "=========================================="
echo ""
