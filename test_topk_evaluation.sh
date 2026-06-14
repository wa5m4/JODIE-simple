#!/bin/bash

# 验证Top-K评估策略

SEED=100
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=50
COARSE_EPOCHS=3
RERANK_TOP_K=3
RERANK_EPOCHS=5

OUTPUT_DIR="outputs/topk_validation/seed_${SEED}"

echo "========================================================================"
echo "Top-K评估策略验证"
echo "========================================================================"
echo "配置:"
echo "  • Coarse阶段: 50 trials × 3 epochs"
echo "  • Rerank阶段: Top-3候选 × 5 epochs"
echo "  • 无数据重叠（稳定的Naive模式）"
echo "========================================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode smart \
    --num-pipeline-stages 1 \
    --pipeline-stage-train-workers 3 \
    --pipeline-worker-gpus 1.0 \
    --partition-size 5000 \
    --partition-overlap-ratio 0.0 \
    --gpu-list "$GPU_LIST" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events "$MAX_EVENTS" \
    --seed "$SEED" \
    --coarse-trials "$COARSE_TRIALS" \
    --coarse-epochs "$COARSE_EPOCHS" \
    --rerank-top-k "$RERANK_TOP_K" \
    --rerank-epochs "$RERANK_EPOCHS" \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false \
    2>&1 | tee "${OUTPUT_DIR}.log"

echo ""
echo "========================================================================"
echo "分析结果"
echo "========================================================================"

python3 << 'PYEOF'
import json
from pathlib import Path

output_dir = Path("outputs/topk_validation/seed_100")
best_path = output_dir / "best_arch.json"

if best_path.exists():
    with open(best_path) as f:
        data = json.load(f)

    print()
    print("【Top-K评估结果】")
    print(f"  选中架构: {data['config'].get('time_proj')}")
    print(f"  Val MRR:  {data.get('val_mrr', 0):.4f}")
    print(f"  Test MRR: {data.get('test_mrr', 0):.4f}")
    print()

    # 对比之前的结果
    print("【对比】")
    print("  之前Naive（无Top-K）: time=off, Test=0.8356")
    print(f"  现在Naive（Top-3）:  time={data['config'].get('time_proj')}, Test={data.get('test_mrr', 0):.4f}")
    print()

    if data['config'].get('time_proj') == 'off' and data.get('test_mrr', 0) >= 0.83:
        print("✅ Top-K评估成功！选出正确架构")
    else:
        print("⚠️  仍需进一步优化")

PYEOF

echo ""
echo "完成！"
