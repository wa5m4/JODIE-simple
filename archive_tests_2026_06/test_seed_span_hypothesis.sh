#!/bin/bash

# 验证种子跨度假设：测试靠近1000的种子

SEEDS=(950 1050 1100)  # 靠近1000的种子
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=50
COARSE_EPOCHS=3
PARTITION_SIZE=5000

BASE_OUTPUT="outputs/seed_span_test"

echo "========================================================================"
echo "种子跨度假设验证"
echo "========================================================================"
echo "测试种子: ${SEEDS[@]} (靠近1000)"
echo "对照组: Seed 1000 (成功), Seeds 100/200/300 (失败)"
echo "========================================================================"
echo ""

mkdir -p "$BASE_OUTPUT"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "测试种子 $SEED"
    echo "========================================================================"

    SEED_OUTPUT="$BASE_OUTPUT/seed_$SEED"
    mkdir -p "$SEED_OUTPUT"

    # 只测试Smart模式（带预热）
    echo ""
    echo "运行 Pipeline Smart (20%预热)..."

    python search.py \
        --search-mode rl \
        --execution-mode ray_pipeline \
        --pipeline-mode smart \
        --num-pipeline-stages 1 \
        --pipeline-stage-train-workers 3 \
        --pipeline-worker-gpus 1.0 \
        --partition-size "$PARTITION_SIZE" \
        --partition-overlap-ratio 0.2 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$COARSE_TRIALS" \
        --coarse-epochs "$COARSE_EPOCHS" \
        --output-dir "${SEED_OUTPUT}/pipeline_smart_overlap" \
        --space rnn_only \
        --batch-mode tbatch \
        --eval-frozen false \
        2>&1 | tee "${SEED_OUTPUT}/pipeline_smart.log"

    echo "✓ 种子 $SEED 完成"
done

echo ""
echo "========================================================================"
echo "所有测试完成！生成分析报告..."
echo "========================================================================"
echo ""

# 分析结果
python3 << 'EOF'
import json
from pathlib import Path

base_dir = Path("outputs/seed_span_test")
test_seeds = [950, 1050, 1100]
ref_seeds = {
    1000: "outputs/pipeline_overlap_20pct/seed_1000/pipeline_smart_overlap",
    100: "outputs/final_comparison/seed_100/pipeline_smart_overlap",
}

print("=" * 80)
print("种子跨度假设验证结果")
print("=" * 80)
print()

print("【参考组】")
print("-" * 80)
for seed, path in ref_seeds.items():
    best_path = Path(path) / "best_arch.json"
    if best_path.exists():
        with open(best_path) as f:
            data = json.load(f)
        arch = data["config"].get("time_proj")
        val_mrr = data.get("val_mrr", data.get("mrr", 0))
        test_mrr = data.get("test_mrr", 0)
        print(f"Seed {seed}: arch={arch:10s} Val={val_mrr:.4f} Test={test_mrr:.4f}")
print()

print("【测试组 - 靠近1000的种子】")
print("-" * 80)
results = []
for seed in test_seeds:
    best_path = base_dir / f"seed_{seed}" / "pipeline_smart_overlap" / "best_arch.json"
    if best_path.exists():
        with open(best_path) as f:
            data = json.load(f)
        arch = data["config"].get("time_proj")
        val_mrr = data.get("val_mrr", data.get("mrr", 0))
        test_mrr = data.get("test_mrr", 0)
        print(f"Seed {seed}: arch={arch:10s} Val={val_mrr:.4f} Test={test_mrr:.4f}")
        results.append({"seed": seed, "arch": arch, "val": val_mrr, "test": test_mrr})
    else:
        print(f"Seed {seed}: 未完成")

print()
print("=" * 80)
print("结论")
print("=" * 80)
print()

if results:
    success_count = sum(1 for r in results if r["arch"] == "off")
    fail_count = len(results) - success_count

    print(f"成功选出time=off: {success_count}/{len(results)}")
    print(f"错误选出time=linear: {fail_count}/{len(results)}")
    print()

    if success_count > fail_count:
        print("✓ 假设支持：靠近1000的种子表现更好")
        print("  → 种子跨度可能是影响因素")
    else:
        print("✗ 假设不支持：靠近1000的种子仍然失败")
        print("  → 不是种子跨度的问题，是预热方案本身的问题")

EOF

echo ""
echo "完整报告已生成！"
