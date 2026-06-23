#!/bin/bash

# 50K Events 完整测试脚本
# 按顺序运行4个策略：Serial → 数据并行 → Smart Pipeline → Naive Pipeline

set -e  # 遇到错误停止

SEED=42
MAX_EVENTS=50000
COARSE_TRIALS=50
COARSE_EPOCHS=3
BASE_DIR="outputs/50k_comparison/seed_42"

echo "========================================================================"
echo "50K Events 完整测试开始"
echo "========================================================================"
echo "配置: $MAX_EVENTS events, $COARSE_TRIALS trials, $COARSE_EPOCHS epochs"
echo "GPU: 0,1,2"
echo "Seed: $SEED"
echo ""

# ============================================================================
# 测试 1/4: Serial基准
# ============================================================================
echo ""
echo "========================================================================"
echo "[1/4] Serial基准测试"
echo "========================================================================"

OUTPUT_DIR="${BASE_DIR}/serial"
mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode serial \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events $MAX_EVENTS \
    --seed $SEED \
    --coarse-trials $COARSE_TRIALS \
    --coarse-epochs $COARSE_EPOCHS \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false \
    2>&1 | tee "${OUTPUT_DIR}.log"

echo "[1/4] Serial完成"

# ============================================================================
# 测试 2/4: 数据并行改进
# ============================================================================
echo ""
echo "========================================================================"
echo "[2/4] 数据并行改进测试"
echo "========================================================================"

OUTPUT_DIR="${BASE_DIR}/data_parallel_improved"
mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode data_parallel \
    --data-parallel-workers 3 \
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
    --partition-size 7500 \
    --partition-strategy count \
    --data-parallel-micro-batch-size 32 \
    2>&1 | tee "${OUTPUT_DIR}.log"

echo "[2/4] 数据并行改进完成"

# ============================================================================
# 测试 3/4: Smart Pipeline + 20%预热
# ============================================================================
echo ""
echo "========================================================================"
echo "[3/4] Smart Pipeline + 20%预热测试"
echo "========================================================================"

OUTPUT_DIR="${BASE_DIR}/smart_overlap20"
mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode smart \
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

echo "[3/4] Smart Pipeline完成"

# ============================================================================
# 测试 4/4: Naive Pipeline
# ============================================================================
echo ""
echo "========================================================================"
echo "[4/4] Naive Pipeline测试"
echo "========================================================================"

OUTPUT_DIR="${BASE_DIR}/naive_no_overlap"
mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode naive \
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

echo "[4/4] Naive Pipeline完成"

# ============================================================================
# 测试完成，生成对比报告
# ============================================================================
echo ""
echo "========================================================================"
echo "所有测试完成！生成对比报告..."
echo "========================================================================"

python3 << 'EOF'
import json
from pathlib import Path

print("\n" + "=" * 75)
print("50K Events 测试结果对比")
print("=" * 75)

configs = [
    ("Serial", "serial"),
    ("数据并行改进", "data_parallel_improved"),
    ("Smart+20%", "smart_overlap20"),
    ("Naive", "naive_no_overlap"),
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

print("\n测试结果已保存到: outputs/50k_comparison/seed_42/")
EOF

echo ""
echo "========================================================================"
echo "✅ 完整测试流程结束"
echo "========================================================================"
