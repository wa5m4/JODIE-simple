#!/bin/bash

# 对数据并行找到的最佳架构进行无分区Serial retrain验证

SEED=42
OUTPUT_DIR="outputs/comprehensive_comparison/seed_42/data_parallel_no_split/retrain"

echo "========================================================================"
echo "数据并行最佳架构 Retrain (无分区Serial)"
echo "========================================================================"

mkdir -p "$OUTPUT_DIR"

# 从best_arch.json提取配置
BEST_CONFIG="outputs/comprehensive_comparison/seed_42/data_parallel_no_split/best_arch.json"

python search.py \
    --search-mode rl \
    --execution-mode serial \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events 20000 \
    --seed 42 \
    --coarse-trials 50 \
    --coarse-epochs 3 \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    2>&1 | tee "${OUTPUT_DIR}.log"

echo ""
echo "========================================================================"
echo "Retrain结果对比"
echo "========================================================================"

python3 << 'PYEOF'
import json
from pathlib import Path

print("\n" + "=" * 70)
print("数据并行最佳架构的无分区Serial Retrain验证")
print("=" * 70)

# 数据并行结果
dp_path = Path("outputs/comprehensive_comparison/seed_42/data_parallel_no_split/best_arch.json")
if dp_path.exists():
    with open(dp_path) as f:
        dp_data = json.load(f)
    print(f"\n数据并行 (五分区):     Test MRR = {dp_data['test_mrr']:.4f}")
    print(f"  架构: {dp_data['config']['time_proj']}/{dp_data['config']['use_static_embeddings']}")

# Retrain结果
retrain_path = Path("outputs/comprehensive_comparison/seed_42/data_parallel_no_split/retrain/best_arch.json")
if retrain_path.exists():
    with open(retrain_path) as f:
        retrain_data = json.load(f)
    print(f"\n无分区Serial Retrain: Test MRR = {retrain_data['test_mrr']:.4f}")
    print(f"  架构: {retrain_data['config']['time_proj']}/{retrain_data['config']['use_static_embeddings']}")

    # 对比
    if abs(dp_data['test_mrr'] - retrain_data['test_mrr']) < 0.001:
        print("\n✅ 验证通过：数据并行结果与无分区Serial一致")
    else:
        print(f"\n⚠️  差异: {abs(dp_data['test_mrr'] - retrain_data['test_mrr']):.4f}")
else:
    print("\n❌ Retrain结果未生成")
PYEOF
