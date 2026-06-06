#!/bin/bash

# 重跑Pipeline Smart以验证修复后的时间统计
# GPU: 0,1,2
# 数据: 20000
# Trials: 27
# Epochs: 3

SEEDS=(100 200 300)
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=27
COARSE_EPOCHS=3
OUTPUT_BASE="outputs/multi_seed_verification_fixed"

echo "=========================================="
echo "重跑Pipeline Smart (修复时间统计后)"
echo "Seeds: ${SEEDS[@]}"
echo "GPU: $GPU_LIST"
echo "=========================================="

mkdir -p "$OUTPUT_BASE"

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${OUTPUT_BASE}/seed_${SEED}"
    OUTPUT_DIR="${SEED_DIR}/pipeline_smart"
    LOG_FILE="${SEED_DIR}/pipeline_smart.log"

    mkdir -p "$SEED_DIR"

    echo ""
    echo "启动 Seed=$SEED Pipeline Smart..."

    nohup python search.py \
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
        --eval-frozen false > "$LOG_FILE" 2>&1 &

    echo $! >> "${OUTPUT_BASE}/pids.txt"

    echo "  进程ID: $!"
    echo "  日志: $LOG_FILE"

    sleep 5
done

echo ""
echo "所有任务已启动！"
echo "进程ID: ${OUTPUT_BASE}/pids.txt"
echo ""
echo "查看进度: tail -f ${OUTPUT_BASE}/seed_100/pipeline_smart.log"
echo "查看进程: cat ${OUTPUT_BASE}/pids.txt | xargs ps -p"
