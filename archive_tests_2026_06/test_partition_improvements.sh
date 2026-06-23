#!/bin/bash

# 验证：更小Partition + 轻量预热方案

SEED=100
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=50
COARSE_EPOCHS=3

echo "========================================================================"
echo "实验：更小Partition + 轻量预热"
echo "========================================================================"
echo ""
echo "核心假设："
echo "  • 更小partition(3000) → 降低每个partition的新entity比例"
echo "  • 轻量重叠(5%) → 预热关键entity，但避免过拟合"
echo ""
echo "对比方案："
echo "  1. 当前基准: partition=5000, overlap=0%   → Test=0.84"
echo "  2. 失败方案: partition=5000, overlap=20%  → Test=0.63-0.87 (不稳定)"
echo "  3. 新方案B:  partition=3000, overlap=5%   → Test=? (验证中)"
echo "  4. 新方案A:  partition=3000, overlap=10%  → Test=? (可选)"
echo ""
echo "========================================================================"
echo ""

BASE_OUTPUT="outputs/partition_improvements/seed_${SEED}"
mkdir -p "$BASE_OUTPUT"

# ============================================================
# 方案1: partition=3000, overlap=5%（推荐）
# ============================================================
echo "────────────────────────────────────────"
echo "[1/2] 新方案B: partition=3000, overlap=5%"
echo "────────────────────────────────────────"

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode smart \
    --num-pipeline-stages 1 \
    --pipeline-stage-train-workers 3 \
    --pipeline-worker-gpus 1.0 \
    --partition-size 3000 \
    --partition-overlap-ratio 0.05 \
    --gpu-list "$GPU_LIST" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events "$MAX_EVENTS" \
    --seed "$SEED" \
    --coarse-trials "$COARSE_TRIALS" \
    --coarse-epochs "$COARSE_EPOCHS" \
    --output-dir "${BASE_OUTPUT}/p3000_o5pct" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false \
    2>&1 | tee "${BASE_OUTPUT}/p3000_o5pct.log"

echo "✓ 方案B完成"
echo ""

# ============================================================
# 方案2: partition=3000, overlap=10%（可选）
# ============================================================
echo "────────────────────────────────────────"
echo "[2/2] 新方案A: partition=3000, overlap=10%"
echo "────────────────────────────────────────"

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode smart \
    --num-pipeline-stages 1 \
    --pipeline-stage-train-workers 3 \
    --pipeline-worker-gpus 1.0 \
    --partition-size 3000 \
    --partition-overlap-ratio 0.1 \
    --gpu-list "$GPU_LIST" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events "$MAX_EVENTS" \
    --seed "$SEED" \
    --coarse-trials "$COARSE_TRIALS" \
    --coarse-epochs "$COARSE_EPOCHS" \
    --output-dir "${BASE_OUTPUT}/p3000_o10pct" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false \
    2>&1 | tee "${BASE_OUTPUT}/p3000_o10pct.log"

echo "✓ 方案A完成"
echo ""

echo "========================================================================"
echo "所有实验完成！生成对比报告..."
echo "========================================================================"
echo ""

# 生成对比报告
python3 << 'EOF'
import json
from pathlib import Path

seed = 100
base_dir = Path(f"outputs/partition_improvements/seed_{seed}")

print("=" * 80)
print("实验结果对比")
print("=" * 80)
print()

configs = [
    ("当前基准 (p5000,o0%)", "outputs/final_comparison/seed_100/pipeline_naive_no_overlap"),
    ("失败方案 (p5000,o20%)", "outputs/final_comparison/seed_100/pipeline_smart_overlap"),
    ("新方案B (p3000,o5%)", f"outputs/partition_improvements/seed_{seed}/p3000_o5pct"),
    ("新方案A (p3000,o10%)", f"outputs/partition_improvements/seed_{seed}/p3000_o10pct"),
]

results = []

for name, path in configs:
    best_path = Path(path) / "best_arch.json"
    if best_path.exists():
        with open(best_path) as f:
            data = json.load(f)

        arch = data['config'].get('time_proj')
        val_mrr = data.get('val_mrr', data.get('mrr', 0))
        test_mrr = data.get('test_mrr', data.get('score', 0))

        results.append({
            'name': name,
            'arch': arch,
            'val': val_mrr,
            'test': test_mrr
        })

        status = "✓" if arch == "off" and test_mrr >= 0.80 else "✗"
        print(f"{status} {name:25s}: arch={arch:7s} Val={val_mrr:.4f} Test={test_mrr:.4f}")
    else:
        print(f"⏳ {name:25s}: 未完成")

print()
print("=" * 80)
print("分析")
print("=" * 80)
print()

# 找到最佳方案
if len(results) >= 3:
    baseline = next((r for r in results if "基准" in r['name']), None)
    new_b = next((r for r in results if "方案B" in r['name']), None)
    new_a = next((r for r in results if "方案A" in r['name']), None)

    if baseline and new_b:
        improvement_b = new_b['test'] - baseline['test']
        print(f"新方案B vs 基准:")
        print(f"  Test MRR: {baseline['test']:.4f} → {new_b['test']:.4f} ({improvement_b:+.4f})")

        if improvement_b > 0.01:
            print(f"  ✅ 显著改进！更小partition + 轻量预热有效")
        elif improvement_b > 0:
            print(f"  ✓ 轻微改进，可进一步调优")
        else:
            print(f"  ⚠️  未见改进")
        print()

    if baseline and new_a:
        improvement_a = new_a['test'] - baseline['test']
        print(f"新方案A vs 基准:")
        print(f"  Test MRR: {baseline['test']:.4f} → {new_a['test']:.4f} ({improvement_a:+.4f})")

        if improvement_a > 0.01:
            print(f"  ✅ 显著改进！")
        print()

    if new_b and new_a:
        print(f"方案B(5%) vs 方案A(10%):")
        diff = new_b['test'] - new_a['test']
        print(f"  Test MRR: {new_a['test']:.4f} vs {new_b['test']:.4f}")

        if abs(diff) < 0.01:
            print(f"  → 相近，推荐5%（更轻量）")
        elif diff > 0:
            print(f"  → 5%更好（轻量预热避免过拟合）")
        else:
            print(f"  → 10%更好（需要更多预热）")

print()
print("=" * 80)
print("结论")
print("=" * 80)
print()

best = max((r for r in results if "方案" in r['name']),
           key=lambda x: x['test'], default=None)

if best and best['test'] >= 0.85:
    print(f"✅ 成功！{best['name']} 达到目标")
    print(f"   Test MRR: {best['test']:.4f} (接近Serial的0.86)")
    print()
    print("推荐配置:")
    if "方案B" in best['name']:
        print("  --partition-size 3000")
        print("  --partition-overlap-ratio 0.05")
    else:
        print("  --partition-size 3000")
        print("  --partition-overlap-ratio 0.1")
elif best and best['test'] > 0.84:
    print(f"✓ 改进！{best['name']} 有提升")
    print(f"   Test MRR: {best['test']:.4f}")
    print("   可作为Pipeline的改进方案")
else:
    print("⚠️  未达预期，需要探索其他方向")

EOF

echo ""
echo "实验完成！"
