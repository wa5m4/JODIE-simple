#!/bin/bash

# 快速验证：去掉Reset是否能解决Pipeline问题

SEED=100
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=50
COARSE_EPOCHS=3

echo "========================================================================"
echo "关键实验：验证Reset是Pipeline问题的根源"
echo "========================================================================"
echo ""
echo "假设：如果去掉Reset，Pipeline能达到Serial水平（~0.86）"
echo ""
echo "对比："
echo "  1. Pipeline Naive + Reset:    Test=0.84 (已知)"
echo "  2. Pipeline Naive + No Reset: Test=? (验证中)"
echo "  3. Serial:                    Test=0.86 (基准)"
echo ""
echo "========================================================================"
echo ""

# 临时修改代码：注释掉reset
echo "临时禁用Reset..."
sed -i.bak '193s/^/        # EXPERIMENT: /' /home/wanghaoyu/JODIE-simple/models/training.py
sed -i '258s/^/        # EXPERIMENT: /' /home/wanghaoyu/JODIE-simple/models/training.py

echo "✓ Reset已禁用（临时）"
echo ""

OUTPUT_DIR="outputs/no_reset_experiment/seed_${SEED}"
mkdir -p "$OUTPUT_DIR"

echo "运行Pipeline Naive（无Reset）..."

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
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false \
    2>&1 | tee "${OUTPUT_DIR}.log"

echo ""
echo "========================================================================"
echo "恢复代码..."
echo "========================================================================"

# 恢复原代码
mv /home/wanghaoyu/JODIE-simple/models/training.py.bak /home/wanghaoyu/JODIE-simple/models/training.py

echo "✓ 代码已恢复"
echo ""

echo "========================================================================"
echo "分析结果"
echo "========================================================================"

python3 << 'EOF'
import json
from pathlib import Path

# 读取实验结果
no_reset_path = Path("outputs/no_reset_experiment/seed_100/best_arch.json")
naive_path = Path("outputs/final_comparison/seed_100/pipeline_naive_no_overlap/best_arch.json")
serial_path = Path("outputs/final_comparison/seed_100/serial/best_arch.json")

print()
print("【结果对比】")
print("-" * 80)

if no_reset_path.exists():
    with open(no_reset_path) as f:
        no_reset = json.load(f)
    print(f"Pipeline Naive + No Reset: arch={no_reset['config'].get('time_proj'):7s} Test={no_reset.get('test_mrr', 0):.4f}")
else:
    print("Pipeline Naive + No Reset: 未完成")

if naive_path.exists():
    with open(naive_path) as f:
        naive = json.load(f)
    print(f"Pipeline Naive + Reset:    arch={naive['config'].get('time_proj'):7s} Test={naive.get('test_mrr', 0):.4f}")

if serial_path.exists():
    with open(serial_path) as f:
        serial = json.load(f)
    print(f"Serial (基准):             arch={serial['config'].get('time_proj'):7s} Test={serial.get('test_mrr', 0):.4f}")

print()
print("=" * 80)
print("结论")
print("=" * 80)
print()

if no_reset_path.exists() and naive_path.exists():
    no_reset_score = no_reset.get('test_mrr', 0)
    naive_score = naive.get('test_mrr', 0)
    serial_score = serial.get('test_mrr', 0) if serial_path.exists() else 0.86

    improvement = no_reset_score - naive_score

    if no_reset_score >= 0.85:
        print("✅ 成功！去掉Reset让Pipeline达到Serial水平")
        print(f"   提升: {improvement:+.4f} ({improvement/naive_score*100:+.1f}%)")
        print()
        print("🎯 结论: Reset确实是Pipeline问题的根源！")
        print()
        print("推荐方案:")
        print("  • Pipeline模式下默认禁用Reset")
        print("  • 或添加--disable-epoch-reset参数")
    elif improvement > 0.01:
        print("✓ 有改进！去掉Reset提升了性能")
        print(f"   提升: {improvement:+.4f} ({improvement/naive_score*100:+.1f}%)")
        print()
        print("但仍未达到Serial水平，可能需要进一步优化")
    else:
        print("⚠️  去掉Reset没有明显改进")
        print("   可能不是主要问题，需要探索其他方向")

EOF

echo ""
echo "实验完成！"
