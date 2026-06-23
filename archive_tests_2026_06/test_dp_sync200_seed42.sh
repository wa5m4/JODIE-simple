#!/bin/bash

# 测试优化后的数据并行（同步粒度=200，适合时序模型）

SEED=42
GPU_LIST="0,1,2"
MAX_EVENTS=20000
TRIALS=50
EPOCHS=3
OUTPUT_DIR="outputs/comprehensive_comparison/seed_42/data_parallel_sync200"

echo "========================================================================"
echo "数据并行优化测试 (Seed 42, micro-batch-size=200)"
echo "========================================================================"
echo "配置:"
echo "  种子: $SEED"
echo "  同步粒度: micro-batch-size=200（适合时序模型）"
echo "  预期同步次数: ~70次"
echo "========================================================================"

mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode data_parallel \
    --data-parallel-workers 3 \
    --data-parallel-sync-level micro_batch \
    --data-parallel-micro-batch-size 200 \
    --gpu-list "$GPU_LIST" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events "$MAX_EVENTS" \
    --seed "$SEED" \
    --coarse-trials "$TRIALS" \
    --coarse-epochs "$EPOCHS" \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    2>&1 | tee "${OUTPUT_DIR}.log"

echo ""
echo "========================================================================"
echo "对比结果"
echo "========================================================================"

python3 << 'EOF'
import json
from pathlib import Path

print("\n" + "=" * 75)
print("Seed 42 数据并行对比")
print("=" * 75)
print(f"{'配置':<30} {'时间(s)':<12} {'Test MRR':<12} {'架构':<10}")
print("-" * 75)

configs = [
    ("serial", "Serial (基准)"),
    ("data_parallel", "数据并行 (默认sync=32)"),
    ("data_parallel_sync200", "数据并行 (sync=200)"),
    ("data_parallel_sync4000", "数据并行 (sync=4000)"),
]

for config_dir, name in configs:
    path = Path(f"outputs/comprehensive_comparison/seed_42/{config_dir}/best_arch.json")
    if path.exists():
        with open(path) as f:
            data = json.load(f)

        time_sec = data.get('time_sec', 0)
        test_mrr = data.get('test_mrr', 0)
        config = data['config']
        arch = f"{config['time_proj']}/{config['use_static_embeddings'][:2]}"

        is_correct = config['time_proj'] == 'off' and config['use_static_embeddings'] == 'off'
        mark = "✅" if is_correct else "❌"

        print(f"{mark} {name:<28} {time_sec:<12.1f} {test_mrr:.4f}      {arch:<10}")
    else:
        print(f"⏳ {name:<28} 未完成")

print("\n说明:")
print("  • sync=32:   同步438次（过于频繁，慢且过拟合）")
print("  • sync=200:  同步70次（平衡速度和准确性）✅")
print("  • sync=4000: 同步3-4次（时间跨度过大，有风险）")

EOF

echo ""
echo "测试完成！"
