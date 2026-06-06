#!/bin/bash

# 单个模式的验证脚本（NAS + Retrain）
# 用法: ./run_single_mode_verification.sh <seed> <mode> <output_dir>

SEED=$1
MODE=$2
OUTPUT_DIR=$3
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=27
COARSE_EPOCHS=3

echo "[$MODE] Seed=$SEED 开始..."

case $MODE in
    serial)
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
            --output-dir "$OUTPUT_DIR" \
            --space rnn_only \
            --batch-mode tbatch \
            --eval-frozen false
        ;;

    data_parallel)
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
            --output-dir "$OUTPUT_DIR" \
            --space rnn_only \
            --batch-mode tbatch \
            --eval-frozen false
        ;;

    pipeline_naive)
        python search.py \
            --search-mode rl \
            --execution-mode ray_pipeline \
            --pipeline-mode naive \
            --num-pipeline-stages 3 \
            --pipeline-stage-train-workers 1 \
            --pipeline-worker-gpus 1.0 \
            --partition-size 5000 \
            --gpu-list "$GPU_LIST" \
            --dataset public_csv \
            --local-data-path data/public/mooc.csv \
            --max-events "$MAX_EVENTS" \
            --seed "$SEED" \
            --coarse-trials "$COARSE_TRIALS" \
            --coarse-epochs "$COARSE_EPOCHS" \
            --output-dir "$OUTPUT_DIR" \
            --space rnn_only \
            --batch-mode tbatch \
            --eval-frozen false
        ;;

    pipeline_smart)
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
            --output-dir "$OUTPUT_DIR" \
            --space rnn_only \
            --batch-mode tbatch \
            --eval-frozen false
        ;;
esac

if [ $? -ne 0 ]; then
    echo "✗ [$MODE] NAS失败"
    exit 1
fi

echo "✓ [$MODE] NAS完成"

# 重训练
BEST_ARCH="$OUTPUT_DIR/best_arch.json"
MODEL=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['model'])")
EMB_DIM=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['embedding_dim'])")
MEMORY_CELL=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['memory_cell'])")
TIME_PROJ=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['time_proj'])")
NORMALIZE_STATE=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('normalize_state', 'off'))")
USE_STATIC_EMB=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('use_static_embeddings', 'off'))")
RETRAIN_SEED=$(python -c "import json; print(json.load(open('$BEST_ARCH')).get('seed', ${SEED}))")

# Pipeline使用partition_size=5000，其他模式不使用partition
if [[ $MODE == pipeline* ]]; then
    PARTITION_SIZE=5000
else
    PARTITION_SIZE=0
fi

python train_single_arch.py \
    --model "$MODEL" \
    --embedding-dim "$EMB_DIM" \
    --memory-cell "$MEMORY_CELL" \
    --time-proj "$TIME_PROJ" \
    --normalize-state "$NORMALIZE_STATE" \
    --use-static-embeddings "$USE_STATIC_EMB" \
    --batch-mode tbatch \
    --partition-size "$PARTITION_SIZE" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events "$MAX_EVENTS" \
    --epochs "$COARSE_EPOCHS" \
    --seed "$RETRAIN_SEED" \
    --output-dir "$OUTPUT_DIR/retrain" \
    --eval-frozen false

if [ $? -eq 0 ]; then
    echo "✓ [$MODE] 重训练完成"
else
    echo "✗ [$MODE] 重训练失败"
    exit 1
fi
