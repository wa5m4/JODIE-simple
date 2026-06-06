#!/bin/bash

# 主脚本：启动多种子验证实验
# 3个种子 × 4个模式 = 12个任务

SEEDS=(100 200 300)
MODES=(serial data_parallel pipeline_naive pipeline_smart)
OUTPUT_BASE="outputs/multi_seed_verification"

echo "=========================================="
echo "多种子验证实验启动"
echo "Seeds: ${SEEDS[@]}"
echo "Modes: ${MODES[@]}"
echo "输出目录: $OUTPUT_BASE"
echo "=========================================="

# 赋予执行权限
chmod +x run_single_mode_verification.sh

# 启动所有任务
for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${OUTPUT_BASE}/seed_${SEED}"
    mkdir -p "$SEED_DIR"

    for MODE in "${MODES[@]}"; do
        OUTPUT_DIR="${SEED_DIR}/${MODE}"
        LOG_FILE="${SEED_DIR}/${MODE}.log"

        echo "启动: Seed=$SEED, Mode=$MODE"
        nohup bash run_single_mode_verification.sh "$SEED" "$MODE" "$OUTPUT_DIR" > "$LOG_FILE" 2>&1 &

        # 记录进程ID
        echo $! >> "${OUTPUT_BASE}/pids.txt"

        # 避免同时启动太多进程，稍微延迟
        sleep 2
    done
done

echo
echo "所有任务已启动！"
echo "进程ID保存在: ${OUTPUT_BASE}/pids.txt"
echo "日志文件: ${OUTPUT_BASE}/seed_*/mode.log"
echo
echo "查看进度: tail -f ${OUTPUT_BASE}/seed_100/serial.log"
echo "查看所有进程: cat ${OUTPUT_BASE}/pids.txt | xargs ps -p"
