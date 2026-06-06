#!/bin/bash

SEED=100
OUTPUT_BASE="outputs/bug_fix_verification_v2/seed_${SEED}"
GPU_LIST="0,1,2"

echo "=========================================="
echo "Pipeline模式重跑 (正确配置+修复后的代码)"
echo "Seed=$SEED, GPU:$GPU_LIST, 27trials"
echo "=========================================="

# Pipeline Naive: 3个stage，每个stage 1个worker
echo ""
echo "=========================================="
echo "Pipeline Naive: 3 stages, 1 worker/stage"
echo "=========================================="

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
    --max-events 20000 \
    --seed "$SEED" \
    --coarse-trials 27 \
    --coarse-epochs 3 \
    --output-dir "${OUTPUT_BASE}/pipeline_naive" \
    --space rnn_only

if [ $? -eq 0 ]; then
    echo "✓ Pipeline Naive NAS完成"

    # 重训练
    echo "开始重训练..."
    BEST_ARCH="${OUTPUT_BASE}/pipeline_naive/best_arch.json"

    MODEL=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['model'])")
    EMB_DIM=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['embedding_dim'])")
    MEMORY_CELL=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['memory_cell'])")
    TIME_PROJ=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['time_proj'])")
    NORMALIZE_STATE=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('normalize_state', 'off'))")
    USE_STATIC_EMB=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('use_static_embeddings', 'off'))")
    RETRAIN_SEED=$(python -c "import json; print(json.load(open('$BEST_ARCH')).get('seed', ${SEED}))")
    # Pipeline使用partition_size=5000（与NAS配置一致）
    PARTITION_SIZE=5000

    echo "提取的seed: $RETRAIN_SEED (应该是20100)"
    echo "使用partition_size: $PARTITION_SIZE (与NAS一致)"

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
        --max-events 20000 \
        --epochs 3 \
        --seed "$RETRAIN_SEED" \
        --output-dir "${OUTPUT_BASE}/pipeline_naive/retrain" \
        --eval-frozen false

    echo "✓ Pipeline Naive重训练完成"
else
    echo "✗ Pipeline Naive NAS失败"
fi

# Pipeline Smart: 1个stage，3个worker
echo ""
echo "=========================================="
echo "Pipeline Smart: 1 stage, 3 workers"
echo "=========================================="

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
    --max-events 20000 \
    --seed "$SEED" \
    --coarse-trials 27 \
    --coarse-epochs 3 \
    --output-dir "${OUTPUT_BASE}/pipeline_smart" \
    --space rnn_only

if [ $? -eq 0 ]; then
    echo "✓ Pipeline Smart NAS完成"

    # 重训练
    echo "开始重训练..."
    BEST_ARCH="${OUTPUT_BASE}/pipeline_smart/best_arch.json"

    MODEL=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['model'])")
    EMB_DIM=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['embedding_dim'])")
    MEMORY_CELL=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['memory_cell'])")
    TIME_PROJ=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config']['time_proj'])")
    NORMALIZE_STATE=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('normalize_state', 'off'))")
    USE_STATIC_EMB=$(python -c "import json; print(json.load(open('$BEST_ARCH'))['config'].get('use_static_embeddings', 'off'))")
    RETRAIN_SEED=$(python -c "import json; print(json.load(open('$BEST_ARCH')).get('seed', ${SEED}))")
    PARTITION_SIZE=$(python -c "import json; print(json.load(open('$BEST_ARCH')).get('distribution_metadata', {}).get('partition_size', 0))")

    echo "提取的seed: $RETRAIN_SEED (应该是20100)"

    python train_single_arch.py \
        --model "$MODEL" \
        --embedding-dim "$EMB_DIM" \
        --memory-cell "$MEMORY_CELL" \
        --time-proj "$TIME_PROJ" \
        --normalize-state "$NORMALIZE_STATE" \
        --use-static-embeddings "$USE_STATIC_EMB" \
        --batch-mode tbatch \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events 20000 \
        --epochs 3 \
        --seed "$RETRAIN_SEED" \
        --output-dir "${OUTPUT_BASE}/pipeline_smart/retrain" \
        --eval-frozen false

    echo "✓ Pipeline Smart重训练完成"
else
    echo "✗ Pipeline Smart NAS失败"
fi

echo ""
echo "=========================================="
echo "生成对比报告"
echo "=========================================="

python << 'EOF'
import json, os

SEED = 100
base = f"outputs/bug_fix_verification_v3/seed_{SEED}"

print(f"\n{'='*80}")
print(f"Pipeline重跑验证结果 (正确配置+修复后seed)")
print(f"{'='*80}\n")

for mode in ['pipeline_naive', 'pipeline_smart']:
    best_path = f"{base}/{mode}/best_arch.json"
    retrain_path = f"{base}/{mode}/retrain/result.json"

    if os.path.exists(best_path) and os.path.exists(retrain_path):
        with open(best_path) as f:
            best = json.load(f)
        with open(retrain_path) as f:
            retrain = json.load(f)

        nas_mrr = best.get('test_mrr', 0)
        retrain_mrr = retrain.get('test_mrr', 0)
        seed = best.get('seed', 'N/A')

        diff = abs(nas_mrr - retrain_mrr) / nas_mrr * 100 if nas_mrr > 0 else 0

        print(f"{mode}:")
        print(f"  Seed: {seed} (应该是20100)")
        print(f"  NAS MRR: {nas_mrr:.4f}")
        print(f"  Retrain MRR: {retrain_mrr:.4f}")
        print(f"  差异: {diff:.2f}%")
        print(f"  状态: {'✓ 完美' if diff < 1 else '⚠ 需检查'}")
        print()

print("="*80)
EOF

echo ""
echo "完成！结果保存在: ${OUTPUT_BASE}"
