#!/bin/bash

# 为四种策略+Rerank运行重训练
SEED=1000
BASE_DIR="outputs/rerank_experiment/seed_${SEED}"

echo "=========================================="
echo "四种策略 + Rerank 重训练"
echo "=========================================="

STRATEGIES=("serial" "data_parallel" "pipeline_naive" "pipeline_smart")

for STRATEGY in "${STRATEGIES[@]}"; do
    NAS_DIR="${BASE_DIR}/${STRATEGY}"
    BEST_ARCH="${NAS_DIR}/best_arch.json"

    if [ ! -f "$BEST_ARCH" ]; then
        echo "✗ ${STRATEGY}: best_arch.json不存在"
        continue
    fi

    echo ""
    echo "[${STRATEGY}] 提取架构参数..."

    MODEL=$(python3 -c "import json; print(json.load(open('$BEST_ARCH'))['config']['model'])")
    EMB_DIM=$(python3 -c "import json; print(json.load(open('$BEST_ARCH'))['config']['embedding_dim'])")
    MEMORY_CELL=$(python3 -c "import json; print(json.load(open('$BEST_ARCH'))['config']['memory_cell'])")
    TIME_PROJ=$(python3 -c "import json; print(json.load(open('$BEST_ARCH'))['config']['time_proj'])")
    NORMALIZE_STATE=$(python3 -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('normalize_state', 'off'))")
    USE_STATIC_EMB=$(python3 -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('use_static_embeddings', 'off'))")
    RETRAIN_SEED=$SEED  # 使用与NAS相同的seed

    echo "  架构: model=$MODEL, emb=$EMB_DIM, cell=$MEMORY_CELL"

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
        --epochs 10 \
        --seed "$RETRAIN_SEED" \
        --output-dir "$RETRAIN_DIR" \
        --eval-frozen false

    if [ $? -eq 0 ]; then
        echo "✓ ${STRATEGY}: 重训练完成"
    else
        echo "✗ ${STRATEGY}: 重训练失败"
    fi
done

echo ""
echo "=========================================="
echo "所有重训练完成！"
echo "=========================================="
