#!/bin/bash

# 用Serial模式重训练方案B选出的最佳架构
# Serial模式：不分区，完整数据训练

SEED=100
MAX_EVENTS=20000
COARSE_TRIALS=50
COARSE_EPOCHS=3

echo "========================================================================"
echo "Serial重训练实验：验证Two-Stage策略"
echo "========================================================================"
echo ""
echo "策略验证："
echo "  Stage 1: Pipeline NAS (p3000,o5%) → 选出最佳架构"
echo "  Stage 2: Serial重训练该架构 → 获得真实性能"
echo ""
echo "预期："
echo "  • Serial重训练后 Test ≥ 0.83 → Two-Stage策略成功！"
echo "  • Serial重训练后 Test ≈ 0.77 → 需调整Stage 1配置"
echo ""
echo "========================================================================"
echo ""

OUTPUT_DIR="outputs/serial_retrain_best/seed_${SEED}"
mkdir -p "$OUTPUT_DIR"

echo "运行Serial重训练（无分区，完整数据）..."
echo ""

python search.py \
    --search-mode rl \
    --execution-mode serial \
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
echo "分析结果"
echo "========================================================================"

python3 << 'EOF'
import json
from pathlib import Path

print()
print("=" * 80)
print("Serial重训练结果对比")
print("=" * 80)
print()

# 读取结果
serial_retrain_path = Path("outputs/serial_retrain_best/seed_100/best_arch.json")
pipeline_b_path = Path("outputs/partition_improvements/seed_100/p3000_o5pct/best_arch.json")
baseline_path = Path("outputs/final_comparison/seed_100/pipeline_naive_no_overlap/best_arch.json")

pipeline_b_test = None
serial_test = None
baseline_test = None

if pipeline_b_path.exists():
    with open(pipeline_b_path) as f:
        data = json.load(f)
    print(f"方案B (Pipeline p3000,o5%): arch={data['config'].get('time_proj'):7s} Val={data.get('val_mrr', 0):.4f} Test={data.get('test_mrr', 0):.4f}")
    pipeline_b_test = data.get('test_mrr', 0)
    pipeline_b_arch = data['config'].get('time_proj')

if serial_retrain_path.exists():
    with open(serial_retrain_path) as f:
        data = json.load(f)
    print(f"Serial重训练 (无分区):      arch={data['config'].get('time_proj'):7s} Val={data.get('val_mrr', 0):.4f} Test={data.get('test_mrr', 0):.4f}")
    serial_test = data.get('test_mrr', 0)
    serial_arch = data['config'].get('time_proj')
else:
    print("Serial重训练: 未完成")

if baseline_path.exists():
    with open(baseline_path) as f:
        data = json.load(f)
    print(f"基准 (Pipeline p5000,o0%):  arch={data['config'].get('time_proj'):7s} Val={data.get('val_mrr', 0):.4f} Test={data.get('test_mrr', 0):.4f}")
    baseline_test = data.get('test_mrr', 0)

print()
print("=" * 80)
print("诊断结论")
print("=" * 80)
print()

if serial_test is not None:
    # 检查架构是否一致
    if serial_arch == pipeline_b_arch:
        print(f"✓ 架构选择一致: {serial_arch}")
        print()
    else:
        print(f"⚠️  架构选择不一致: Pipeline选{pipeline_b_arch}, Serial选{serial_arch}")
        print()

    if serial_test >= 0.83:
        improvement = serial_test - pipeline_b_test
        gap_to_baseline = abs(serial_test - baseline_test)

        print(f"✅ Two-Stage策略成功！")
        print()
        print(f"性能对比:")
        print(f"  Pipeline NAS (p3000,o5%): {pipeline_b_test:.4f}")
        print(f"  Serial重训练:            {serial_test:.4f} (+{improvement:+.4f})")
        print(f"  基准 (Pipeline p5000):    {baseline_test:.4f}")
        print()

        if gap_to_baseline < 0.01:
            print(f"🎯 完美！Serial重训练达到基准水平")
        else:
            print(f"✓ Serial重训练显著改进，接近基准（差距{gap_to_baseline:.4f}）")

        print()
        print("结论:")
        print("  • Pipeline (p3000,o5%) 成功选出正确架构")
        print("  • Serial重训练恢复了完整性能")
        print("  • Two-Stage方法有效：Pipeline快速搜索 + Serial充分训练")

    elif serial_test > pipeline_b_test + 0.03:
        improvement = serial_test - pipeline_b_test
        print(f"✓ Serial重训练有改进: {pipeline_b_test:.4f} → {serial_test:.4f} (+{improvement:+.4f})")
        print()
        print("结论:")
        print("  • Serial训练确实比Pipeline训练性能更好")
        print("  • 但未达到基准水平，可能需要调整Stage 1配置")
        print()
        print("建议:")
        print("  • 尝试 partition=5000 + overlap=5% (更大partition)")
        print("  • 或增加训练epochs")

    else:
        print(f"⚠️  Serial重训练未见显著改进: {pipeline_b_test:.4f} → {serial_test:.4f}")
        print()
        print("结论:")
        print("  • 问题可能不在训练方式")
        print("  • 可能是partition=3000导致的数据分布问题")
        print()
        print("建议:")
        print("  • 尝试 partition=5000 + overlap=5% (Stage 1改进)")

EOF

echo ""
echo "实验完成！"
