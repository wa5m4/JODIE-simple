#!/bin/bash

# 从断点继续comprehensive_comparison实验
# 已完成: serial, data_parallel, smart_overlap20, smart_no_overlap
# 待完成: naive_overlap20, naive_no_overlap, retrain

SEED=100
GPU_LIST="0,1,2"
MAX_EVENTS=20000
TRIALS=50
EPOCHS=3
PARTITION_SIZE=5000

SEED_DIR="outputs/comprehensive_comparison/seed_${SEED}"

echo "========================================================================"
echo "继续实验 (Seed $SEED)"
echo "========================================================================"

# ============================================================
# 5. Pipeline Naive + 20%预热
# ============================================================
echo ""
echo "[5/7] Pipeline Naive (同步 + 20%预热)"
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

# ============================================================
# 6. Pipeline Naive + 无预热
# ============================================================
echo ""
echo "[6/7] Pipeline Naive (同步 + 无预热)"
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

echo ""
echo "✓ NAS搜索完成"

# ============================================================
# 7. Serial重训练所有策略的best架构
# ============================================================
echo ""
echo "[7/7] Serial重训练各策略的best架构"

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
echo "实验完成！"
echo "========================================================================"
