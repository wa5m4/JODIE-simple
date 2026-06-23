#!/bin/bash

SEED=100
OUTPUT_BASE="outputs/bug_fix_verification_v2/seed_${SEED}"

echo "=========================================="
echo "完整验证：NAS + Retrain (Seed=$SEED)"
echo "=========================================="

MODES=("serial" "data_parallel" "pipeline_naive" "pipeline_smart")

for MODE in "${MODES[@]}"; do
    echo ""
    echo "=========================================="
    echo "模式: ${MODE}"
    echo "=========================================="

    OUTPUT_DIR="${OUTPUT_BASE}/${MODE}"

    # 第1步：NAS搜索
    echo "[1/2] 运行NAS搜索..."

    # 根据模式设置正确的参数
    if [ "$MODE" = "serial" ]; then
        EXEC_MODE="serial"
        EXTRA_ARGS=""
    elif [ "$MODE" = "data_parallel" ]; then
        EXEC_MODE="data_parallel"
        EXTRA_ARGS=""
    elif [ "$MODE" = "pipeline_naive" ]; then
        EXEC_MODE="ray_pipeline"
        EXTRA_ARGS="--pipeline-mode naive"
    elif [ "$MODE" = "pipeline_smart" ]; then
        EXEC_MODE="ray_pipeline"
        EXTRA_ARGS="--pipeline-mode smart"
    fi

    python search.py \
        --search-mode rl \
        --execution-mode "$EXEC_MODE" \
        $EXTRA_ARGS \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events 20000 \
        --seed "$SEED" \
        --coarse-trials 10 \
        --coarse-epochs 3 \
        --output-dir "$OUTPUT_DIR" \
        --space rnn_only

    if [ $? -ne 0 ]; then
        echo "✗ ${MODE} NAS搜索失败"
        continue
    fi

    echo "✓ ${MODE} NAS搜索完成"

    # 第2步：重训练
    echo "[2/2] 重训练best_arch..."

    BEST_ARCH="${OUTPUT_DIR}/best_arch.json"
    RETRAIN_OUTPUT="${OUTPUT_DIR}/retrain"

    if [ ! -f "$BEST_ARCH" ]; then
        echo "✗ ${MODE}: best_arch.json not found"
        continue
    fi

    # 提取参数
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

    if [ $? -eq 0 ]; then
        echo "✓ ${MODE} 重训练完成"
    else
        echo "✗ ${MODE} 重训练失败"
    fi
done

echo ""
echo "=========================================="
echo "生成对比报告"
echo "=========================================="

python << 'EOF'
import json
import os

SEED = 100
modes = ['serial', 'data_parallel', 'pipeline_naive', 'pipeline_smart']

print(f"\n{'='*70}")
print(f"NAS vs Retrain 对比报告 (Seed={SEED})")
print(f"{'='*70}\n")

print(f"{'模式':<15} {'NAS MRR':<10} {'Retrain MRR':<12} {'差异':<10} {'种子':<10}")
print("-" * 70)

for mode in modes:
    base_dir = f"outputs/bug_fix_verification_v2/seed_{SEED}/{mode}"
    best_arch_path = f"{base_dir}/best_arch.json"
    retrain_result_path = f"{base_dir}/retrain/result.json"

    if not os.path.exists(best_arch_path):
        print(f"{mode:<15} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
        continue

    with open(best_arch_path) as f:
        best_arch = json.load(f)

    nas_mrr = best_arch.get('test_mrr', best_arch.get('mrr', 0))
    nas_seed = best_arch.get('seed', 'NOT_SET')

    if os.path.exists(retrain_result_path):
        with open(retrain_result_path) as f:
            retrain_result = json.load(f)
        retrain_mrr = retrain_result.get('test_mrr', 0)
    else:
        retrain_mrr = 0

    if nas_mrr > 0 and retrain_mrr > 0:
        diff = abs(nas_mrr - retrain_mrr)
        diff_pct = f"{(diff/nas_mrr*100):.2f}%"
    else:
        diff_pct = "N/A"

    print(f"{mode:<15} {nas_mrr:<10.4f} {retrain_mrr:<12.4f} {diff_pct:<10} {str(nas_seed):<10}")

print("\n" + "="*70)
EOF

echo ""
echo "完成！结果保存在: ${OUTPUT_BASE}"
