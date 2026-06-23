#!/bin/bash

# Bug修复验证脚本 - 单种子版本
# 验证Final Test是否正确使用16000条训练数据

SEED=42
MAX_EVENTS=20000
TRIALS=27
EPOCHS=3
OUTPUT_BASE="outputs/bug_fix_verification/seed_${SEED}"

echo "=========================================="
echo "Bug Fix Verification Experiment"
echo "=========================================="
echo "Seed: ${SEED}"
echo "Max Events: ${MAX_EVENTS}"
echo "Trials: ${TRIALS}"
echo "Epochs: ${EPOCHS}"
echo "Output: ${OUTPUT_BASE}"
echo "=========================================="
echo ""

MODES=("serial" "data_parallel" "pipeline_naive" "pipeline_smart")

for MODE in "${MODES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: ${MODE}"
    echo "=========================================="

    if [ "$MODE" = "serial" ]; then
        OUTPUT_DIR="${OUTPUT_BASE}/serial"
        echo "Starting Serial mode..."
        python search.py \
            --execution-mode serial \
            --max-events ${MAX_EVENTS} \
            --trials ${TRIALS} \
            --epochs-per-trial ${EPOCHS} \
            --seed ${SEED} \
            --output-dir "${OUTPUT_DIR}" \
            --dataset public_csv \
            --local-data-path data/public/mooc.csv \
            --batch-mode tbatch \
            --eval-frozen false

    elif [ "$MODE" = "data_parallel" ]; then
        OUTPUT_DIR="${OUTPUT_BASE}/data_parallel"
        echo "Starting Data Parallel mode..."
        python search.py \
            --execution-mode data_parallel \
            --data-parallel-workers 3 \
            --data-parallel-worker-gpus 1.0 \
            --data-parallel-visible-gpus "0,1,2" \
            --max-events ${MAX_EVENTS} \
            --trials ${TRIALS} \
            --epochs-per-trial ${EPOCHS} \
            --seed ${SEED} \
            --output-dir "${OUTPUT_DIR}" \
            --dataset public_csv \
            --local-data-path data/public/mooc.csv \
            --batch-mode tbatch \
            --eval-frozen false

    elif [ "$MODE" = "pipeline_naive" ]; then
        OUTPUT_DIR="${OUTPUT_BASE}/pipeline_naive"
        echo "Starting Pipeline Naive mode..."
        python search.py \
            --execution-mode ray_pipeline \
            --pipeline-mode naive \
            --gpu-list "0,1,2" \
            --enable-auto-pipeline-config \
            --max-events ${MAX_EVENTS} \
            --trials ${TRIALS} \
            --epochs-per-trial ${EPOCHS} \
            --architectures-per-step 2 \
            --seed ${SEED} \
            --output-dir "${OUTPUT_DIR}" \
            --dataset public_csv \
            --local-data-path data/public/mooc.csv \
            --batch-mode tbatch \
            --eval-frozen false

    elif [ "$MODE" = "pipeline_smart" ]; then
        OUTPUT_DIR="${OUTPUT_BASE}/pipeline_smart"
        echo "Starting Pipeline Smart mode..."
        python search.py \
            --execution-mode ray_pipeline \
            --pipeline-mode smart \
            --gpu-list "0,1,2" \
            --enable-auto-pipeline-config \
            --max-events ${MAX_EVENTS} \
            --trials ${TRIALS} \
            --epochs-per-trial ${EPOCHS} \
            --architectures-per-step 2 \
            --seed ${SEED} \
            --output-dir "${OUTPUT_DIR}" \
            --dataset public_csv \
            --local-data-path data/public/mooc.csv \
            --batch-mode tbatch \
            --eval-frozen false
    fi

    if [ $? -eq 0 ]; then
        echo "✓ ${MODE} completed successfully"
    else
        echo "✗ ${MODE} failed"
    fi
done

echo ""
echo "=========================================="
echo "Extracting results..."
echo "=========================================="

# 提取重训结果
echo ""
echo "NAS Test Results (from best_arch.json):"
echo "Mode              | time_proj | val_mrr  | test_mrr | diff"
echo "------------------|-----------|----------|----------|------"

for MODE in "${MODES[@]}"; do
    BEST_ARCH="${OUTPUT_BASE}/${MODE}/best_arch.json"
    if [ -f "$BEST_ARCH" ]; then
        TIME_PROJ=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('time_proj', 'N/A'))")
        VAL_MRR=$(python -c "import json; data=json.load(open('$BEST_ARCH')); print(f\"{data.get('val_mrr', 0):.4f}\")")
        TEST_MRR=$(python -c "import json; data=json.load(open('$BEST_ARCH')); print(f\"{data.get('test_mrr', 0):.4f}\")")
        DIFF=$(python -c "import json; data=json.load(open('$BEST_ARCH')); val=data.get('val_mrr',0); test=data.get('test_mrr',0); print(f\"{val-test:+.4f}\")")

        printf "%-17s | %-9s | %-8s | %-8s | %s\n" "${MODE}" "${TIME_PROJ}" "${VAL_MRR}" "${TEST_MRR}" "${DIFF}"
    else
        printf "%-17s | N/A\n" "${MODE}"
    fi
done

echo ""
echo "验证标准:"
echo "1. 检查日志中Final Test应显示'16000 interactions'（修复后）而非'14000'（修复前）"
echo "2. val_mrr和test_mrr的差异应该较小（说明评估一致）"
echo ""
