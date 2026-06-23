#!/bin/bash

# 修正后的Pipeline测试：Smart(1 stage) vs Naive(3 stages)

set -e

SEED=42
MAX_EVENTS=50000
COARSE_TRIALS=50
COARSE_EPOCHS=3
BASE_DIR="outputs/50k_comparison/seed_42"

echo "========================================================================"
echo "修正后的Pipeline测试"
echo "========================================================================"
echo "Smart: 1 stage (架构并行 + 异步生成)"
echo "Naive: 3 stages (GPU数量)"
echo ""

# ============================================================================
# Smart Pipeline: 1 stage + 异步架构生成
# ============================================================================
echo ""
echo "========================================================================"
echo "[1/2] Smart Pipeline (1 stage)"
echo "========================================================================"

OUTPUT_DIR="${BASE_DIR}/smart_1stage"
mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode smart \
    --num-pipeline-stages 1 \
    --pipeline-stage-train-workers 3 \
    --gpu-list "0,1,2" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events $MAX_EVENTS \
    --seed $SEED \
    --coarse-trials $COARSE_TRIALS \
    --coarse-epochs $COARSE_EPOCHS \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    --partition-size 12500 \
    --partition-strategy count \
    --partition-overlap-ratio 0.2 \
    --stage-balance-strategy cost \
    2>&1 | tee "${OUTPUT_DIR}.log"

echo "[1/2] Smart Pipeline完成"

# ============================================================================
# Naive Pipeline: 3 stages
# ============================================================================
echo ""
echo "========================================================================"
echo "[2/2] Naive Pipeline (3 stages)"
echo "========================================================================"

OUTPUT_DIR="${BASE_DIR}/naive_3stages"
mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode naive \
    --num-pipeline-stages 3 \
    --pipeline-stage-train-workers 3 \
    --gpu-list "0,1,2" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events $MAX_EVENTS \
    --seed $SEED \
    --coarse-trials $COARSE_TRIALS \
    --coarse-epochs $COARSE_EPOCHS \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    --partition-size 12500 \
    --partition-strategy count \
    --partition-overlap-ratio 0.0 \
    2>&1 | tee "${OUTPUT_DIR}.log"

echo "[2/2] Naive Pipeline完成"

# ============================================================================
# 生成对比报告
# ============================================================================
echo ""
echo "========================================================================"
echo "Pipeline测试结果对比"
echo "========================================================================"

python3 << 'EOF'
import json
from pathlib import Path

configs = [
    ("Serial", "serial"),
    ("数据并行", "data_parallel_improved"),
    ("Smart(1stage)", "smart_1stage"),
    ("Naive(3stages)", "naive_3stages"),
]

print(f"\n{'策略':<20} {'架构':<12} {'Test MRR':<12} {'时间(s)':<10}")
print("-" * 75)

for name, dir_name in configs:
    path = Path(f"outputs/50k_comparison/seed_42/{dir_name}/best_arch.json")
    if path.exists():
        with open(path) as f:
            data = json.load(f)

        config = data['config']
        arch = f"{config['time_proj']}/{config['use_static_embeddings']}"
        test_mrr = data.get('test_mrr', 0)
        time_sec = data.get('time_sec', 0)

        is_correct = config['time_proj'] == 'off' and config['use_static_embeddings'] == 'off'
        mark = "✅" if is_correct else "❌"

        print(f"{mark} {name:<18} {arch:<12} {test_mrr:.4f}      {time_sec:<10.1f}")
    else:
        print(f"⏳ {name:<18} 未完成")

print()
EOF

echo ""
echo "========================================================================"
echo "✅ Pipeline测试完成"
echo "========================================================================"
