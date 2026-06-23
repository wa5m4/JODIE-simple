#!/bin/bash

# 测试修复后的数据并行（串行应用worker updates保持时序依赖）

SEED=42
OUTPUT_DIR="outputs/comprehensive_comparison/seed_42/data_parallel_fixed"

echo "========================================================================"
echo "数据并行修复测试 (Seed 42)"
echo "关键修复：串行应用worker updates，保持时序依赖关系"
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
    2>&1 | tee "${OUTPUT_DIR}.log"

echo ""
echo "========================================================================"
echo "结果对比"
echo "========================================================================"

python3 << 'EOF'
import json
from pathlib import Path

print("\n" + "=" * 75)
print("Seed 42 数据并行修复效果对比")
print("=" * 75)
print(f"{'配置':<30} {'架构':<12} {'Test MRR':<12} {'时间(s)':<10}")
print("-" * 75)

configs = [
    ("serial", "Serial (基准)"),
    ("data_parallel", "数据并行 (原始)"),
    ("data_parallel_fixed", "数据并行 (修复) ✨"),
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

print("\n" + "=" * 75)
print("修复说明")
print("=" * 75)
print("原始问题：所有workers基于相同initial state并行训练 → 丢失时序依赖")
print("修复方案：Workers串行执行，每个worker基于前一个worker的state")
print("          Worker0→state1 → Worker1基于state1→state2 → Worker2基于state2→state3")
print("\n预期效果：")
print("  • 架构选择：off/on → off/off ✅")
print("  • Test MRR：0.61 → 0.85")
print("  • 保持时序依赖完整性")
EOF
