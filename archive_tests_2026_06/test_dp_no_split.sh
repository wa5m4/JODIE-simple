#!/bin/bash

# 测试修复后的数据并行（完整时序，不split chunks）

SEED=42
OUTPUT_DIR="outputs/comprehensive_comparison/seed_42/data_parallel_no_split"

echo "========================================================================"
echo "数据并行修复测试 (Seed 42)"
echo "关键修复：不split chunks，保持完整时序"
echo "========================================================================"

mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode data_parallel \
    --data-parallel-workers 3 \
    --gpu-list "0,1,2" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events 20000 \
    --seed 42 \
    --coarse-trials 50 \
    --coarse-epochs 3 \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    --partition-size 3000 \
    --partition-strategy count \
    2>&1 | tee "${OUTPUT_DIR}.log"

echo ""
echo "========================================================================"
echo "结果对比"
echo "========================================================================"

python3 << 'EOF'
import json
from pathlib import Path

print("\n" + "=" * 75)
print("Seed 42 数据并行修复对比")
print("=" * 75)
print(f"{'配置':<30} {'架构':<12} {'Test MRR':<12} {'时间(s)':<10}")
print("-" * 75)

configs = [
    ("serial", "Serial (基准)"),
    ("data_parallel", "数据并行 (原始split)"),
    ("data_parallel_no_split", "数据并行 (完整时序) ✨"),
]

for config_dir, name in configs:
    path = Path(f"outputs/comprehensive_comparison/seed_42/{config_dir}/best_arch.json")
    if path.exists():
        with open(path) as f:
            data = json.load(f)

        config = data['config']
        arch = f"{config['time_proj']}/{config['use_static_embeddings']}"
        test_mrr = data.get('test_mrr', 0)
        time_sec = data.get('time_sec', 0)

        is_correct = config['time_proj'] == 'off' and config['use_static_embeddings'] == 'off'
        mark = "✅" if is_correct else "❌"

        print(f"{mark} {name:<28} {arch:<12} {test_mrr:.4f}      {time_sec:<10.1f}")
    else:
        print(f"⏳ {name:<28} 未完成")

print("\n修复说明：")
print("  • 不再split chunks（避免破坏时序依赖）")
print("  • 每个partition完整训练（保持准确性）")
print("  • 预期：架构选择正确(off/off)，Test MRR~0.85")
EOF
