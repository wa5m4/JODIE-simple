#!/bin/bash

# 为Seed 42和1000补充Serial重训练
# 重训练各策略的best架构（无分区完整训练）

SEEDS=(42 1000)
MAX_EVENTS=20000
EPOCHS=3

echo "========================================================================"
echo "补充Serial重训练 (Seeds: 42, 1000)"
echo "========================================================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "种子 $SEED 开始重训练"
    echo "========================================================================"

    SEED_DIR="outputs/comprehensive_comparison/seed_$SEED"

    STRATEGIES=(
        "serial"
        "data_parallel"
        "smart_overlap20"
        "smart_no_overlap"
        "naive_overlap20"
        "naive_no_overlap"
    )

    for STRATEGY in "${STRATEGIES[@]}"; do
        BEST_FILE="${SEED_DIR}/${STRATEGY}/best_arch.json"

        if [ ! -f "$BEST_FILE" ]; then
            echo "  跳过 $STRATEGY: best_arch.json 不存在"
            continue
        fi

        echo ""
        echo "  重训练: $STRATEGY"

        # 提取架构参数
        MODEL=$(python3 -c "import json; print(json.load(open('$BEST_FILE'))['config']['model'])")
        EMB_DIM=$(python3 -c "import json; print(json.load(open('$BEST_FILE'))['config']['embedding_dim'])")
        MEMORY_CELL=$(python3 -c "import json; print(json.load(open('$BEST_FILE'))['config']['memory_cell'])")
        TIME_PROJ=$(python3 -c "import json; print(json.load(open('$BEST_FILE'))['config']['time_proj'])")
        USE_STATIC=$(python3 -c "import json; print(json.load(open('$BEST_FILE'))['config']['use_static_embeddings'])")

        RETRAIN_DIR="${SEED_DIR}/${STRATEGY}/retrain"
        mkdir -p "$RETRAIN_DIR"

        python train_single_arch.py \
            --model "$MODEL" \
            --embedding-dim "$EMB_DIM" \
            --memory-cell "$MEMORY_CELL" \
            --time-proj "$TIME_PROJ" \
            --use-static-embeddings "$USE_STATIC" \
            --dataset public_csv \
            --local-data-path data/public/mooc.csv \
            --max-events "$MAX_EVENTS" \
            --epochs "$EPOCHS" \
            --seed "$SEED" \
            --batch-mode tbatch \
            --eval-frozen false \
            --output-dir "$RETRAIN_DIR" \
            2>&1 | tee "${RETRAIN_DIR}.log"
    done

    echo ""
    echo "========================================================================"
    echo "种子 $SEED 重训练完成"
    echo "========================================================================"
done

echo ""
echo "========================================================================"
echo "所有重训练完成！生成对比报告..."
echo "========================================================================"
