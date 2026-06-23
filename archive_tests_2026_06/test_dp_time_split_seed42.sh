#!/bin/bash

# 测试优化后的数据并行（按时间分chunk，减少overlap）

SEED=42
GPU_LIST="0,1,2"
MAX_EVENTS=20000
TRIALS=50
EPOCHS=3
OUTPUT_DIR="outputs/comprehensive_comparison/seed_42/data_parallel_time_split"

echo "========================================================================"
echo "数据并行优化测试 (Seed 42, 按时间分chunk)"
echo "========================================================================"
echo "配置:"
echo "  种子: $SEED"
echo "  优化: 按时间范围分chunk（减少user/item重叠）"
echo "  GPU: $GPU_LIST"
echo "========================================================================"

mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode data_parallel \
    --data-parallel-workers 3 \
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

print("\n" + "=" * 80)
print("Seed 42 数据并行优化对比")
print("=" * 80)
print(f"{'配置':<35} {'架构':<15} {'Test MRR':<12} {'时间(s)':<10} {'状态'}")
print("-" * 80)

configs = [
    ("serial", "Serial (基准)"),
    ("data_parallel", "数据并行 (原始index分chunk)"),
    ("data_parallel_time_split", "数据并行 (时间分chunk) ✨"),
]

for config_dir, name in configs:
    path = Path(f"outputs/comprehensive_comparison/seed_42/{config_dir}/best_arch.json")
    if path.exists():
        with open(path) as f:
            data = json.load(f)

        config = data['config']
        time_proj = config.get('time_proj', 'N/A')
        use_static = config.get('use_static_embeddings', 'N/A')
        arch = f"{time_proj}/{use_static}"

        test_mrr = data.get('test_mrr', 0)
        time_sec = data.get('time_sec', 0)

        is_correct = time_proj == 'off' and use_static == 'off'
        mark = "✅" if is_correct else "❌"

        print(f"{mark} {name:<33} {arch:<15} {test_mrr:.4f}      {time_sec:<10.1f} {'正确' if is_correct else '错误'}")
    else:
        print(f"⏳ {name:<33} 未完成")

print("\n" + "=" * 80)
print("优化说明")
print("=" * 80)
print("原始方案：按index均分 → 同一user在不同chunks → merge丢失信息")
print("优化方案：按时间分chunk → 每个worker处理不重叠时间段 → 减少merge冲突")
print("\n预期改进：")
print("  • 架构选择：错误 → 正确（off/off）")
print("  • Test MRR：0.61 → 0.85")
print("  • 时间：124s → ~40s（接近3倍加速）")

EOF

echo ""
echo "测试完成！"
