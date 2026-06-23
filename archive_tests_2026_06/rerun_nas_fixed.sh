#!/bin/bash
# 使用修复后的评估逻辑重新运行四种执行模式的NAS搜索
# 参数: t-batch, 20000 events, GPU 0,1,2, 27 trials, 3 epochs

set -e

BASE_DIR="outputs/nas_search_fixed"
mkdir -p "$BASE_DIR"

COMMON_ARGS="--dataset public_csv \
--local-data-path data/public/mooc.csv \
--max-events 20000 \
--train-ratio 0.7 \
--val-ratio 0.1 \
--feature-dim 8 \
--lr 0.001 \
--k 10 \
--seed 42 \
--coarse-trials 27 \
--coarse-epochs 3 \
--batch-mode tbatch \
--train-batch-size 32"

echo "========================================"
echo "使用修复后的评估逻辑重新运行NAS搜索"
echo "========================================"
echo "参数:"
echo "  - 数据集: MOOC (20000 events)"
echo "  - GPU: 0,1,2"
echo "  - Trials: 27"
echo "  - Epochs: 3"
echo "  - Batch mode: tbatch"
echo ""

# 1. Serial
echo "========================================"
echo "1/4: Serial + T-Batch"
echo "========================================"
python -u search.py \
    $COMMON_ARGS \
    --execution-mode serial \
    --gpu-list 0,1,2 \
    --output-dir "$BASE_DIR/serial_tbatch" \
    2>&1 | tee "$BASE_DIR/serial_tbatch.log"

echo ""
echo "Serial 完成！"
echo ""

# 2. Data Parallel
echo "========================================"
echo "2/4: Data Parallel + T-Batch"
echo "========================================"
python -u search.py \
    $COMMON_ARGS \
    --execution-mode data_parallel \
    --data-parallel-workers 3 \
    --data-parallel-visible-gpus 0,1,2 \
    --gpu-list 0,1,2 \
    --output-dir "$BASE_DIR/data_parallel_tbatch" \
    2>&1 | tee "$BASE_DIR/data_parallel_tbatch.log"

echo ""
echo "Data Parallel 完成！"
echo ""

# 3. Pipeline Naive
echo "========================================"
echo "3/4: Pipeline Naive + T-Batch"
echo "========================================"
python -u search.py \
    $COMMON_ARGS \
    --execution-mode ray_pipeline \
    --pipeline-mode naive \
    --num-pipeline-stages 3 \
    --pipeline-stage-train-workers "1,1,1" \
    --gpu-list 0,1,2 \
    --output-dir "$BASE_DIR/pipeline_naive_tbatch" \
    2>&1 | tee "$BASE_DIR/pipeline_naive_tbatch.log"

echo ""
echo "Pipeline Naive 完成！"
echo ""

# 4. Pipeline Smart
echo "========================================"
echo "4/4: Pipeline Smart + T-Batch"
echo "========================================"
python -u search.py \
    $COMMON_ARGS \
    --execution-mode ray_pipeline \
    --pipeline-mode smart \
    --num-pipeline-stages 1 \
    --pipeline-stage-train-workers "3" \
    --gpu-list 0,1,2 \
    --output-dir "$BASE_DIR/pipeline_smart_tbatch" \
    2>&1 | tee "$BASE_DIR/pipeline_smart_tbatch.log"

echo ""
echo "Pipeline Smart 完成！"
echo ""

echo "========================================"
echo "所有搜索完成！"
echo "========================================"
echo "结果保存在: $BASE_DIR"
echo ""
echo "查看各模式的最优架构:"
echo "  - Serial: $BASE_DIR/serial_tbatch/best_arch.json"
echo "  - Data Parallel: $BASE_DIR/data_parallel_tbatch/best_arch.json"
echo "  - Pipeline Naive: $BASE_DIR/pipeline_naive_tbatch/best_arch.json"
echo "  - Pipeline Smart: $BASE_DIR/pipeline_smart_tbatch/best_arch.json"
