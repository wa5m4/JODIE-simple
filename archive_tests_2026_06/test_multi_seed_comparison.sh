#!/bin/bash

# 多种子全面对比实验
# 对比：Pipeline Smart (预热) vs Pipeline Naive vs Serial

SEEDS=(100 200 300)
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=50
COARSE_EPOCHS=3
OVERLAP_RATIO=0.2
PARTITION_SIZE=5000

BASE_OUTPUT="outputs/final_comparison"

echo "========================================================================"
echo "多种子全面对比实验"
echo "========================================================================"
echo "配置:"
echo "  • 种子: ${SEEDS[@]}"
echo "  • GPU: $GPU_LIST"
echo "  • 数据量: $MAX_EVENTS"
echo "  • Trials: $COARSE_TRIALS"
echo "  • Epochs: $COARSE_EPOCHS"
echo "========================================================================"
echo ""

mkdir -p "$BASE_OUTPUT"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "种子 $SEED 开始"
    echo "========================================================================"

    SEED_OUTPUT="$BASE_OUTPUT/seed_$SEED"
    mkdir -p "$SEED_OUTPUT"

    # ============================================================
    # 1. Pipeline Smart（带20%预热）
    # ============================================================
    echo ""
    echo "────────────────────────────────────────"
    echo "[1/3] Pipeline Smart (20%预热)"
    echo "────────────────────────────────────────"

    python search.py \
        --search-mode rl \
        --execution-mode ray_pipeline \
        --pipeline-mode smart \
        --num-pipeline-stages 1 \
        --pipeline-stage-train-workers 3 \
        --pipeline-worker-gpus 1.0 \
        --partition-size "$PARTITION_SIZE" \
        --partition-overlap-ratio "$OVERLAP_RATIO" \
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
        2>&1 | tee "${SEED_OUTPUT}/pipeline_smart_overlap.log"

    echo "✓ Pipeline Smart完成"

    # ============================================================
    # 2. Pipeline Naive（无预热）
    # ============================================================
    echo ""
    echo "────────────────────────────────────────"
    echo "[2/3] Pipeline Naive (无预热)"
    echo "────────────────────────────────────────"

    python search.py \
        --search-mode rl \
        --execution-mode ray_pipeline \
        --pipeline-mode naive \
        --num-pipeline-stages 1 \
        --pipeline-stage-train-workers 3 \
        --pipeline-worker-gpus 1.0 \
        --partition-size "$PARTITION_SIZE" \
        --partition-overlap-ratio 0.0 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$COARSE_TRIALS" \
        --coarse-epochs "$COARSE_EPOCHS" \
        --output-dir "${SEED_OUTPUT}/pipeline_naive_no_overlap" \
        --space rnn_only \
        --batch-mode tbatch \
        --eval-frozen false \
        2>&1 | tee "${SEED_OUTPUT}/pipeline_naive_no_overlap.log"

    echo "✓ Pipeline Naive完成"

    # ============================================================
    # 3. Serial（基准）
    # ============================================================
    echo ""
    echo "────────────────────────────────────────"
    echo "[3/3] Serial (基准)"
    echo "────────────────────────────────────────"

    python search.py \
        --search-mode rl \
        --execution-mode serial \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$COARSE_TRIALS" \
        --coarse-epochs "$COARSE_EPOCHS" \
        --output-dir "${SEED_OUTPUT}/serial" \
        --space rnn_only \
        --batch-mode tbatch \
        --eval-frozen false \
        2>&1 | tee "${SEED_OUTPUT}/serial.log"

    echo "✓ Serial完成"

    echo ""
    echo "========================================================================"
    echo "种子 $SEED 完成"
    echo "========================================================================"
done

echo ""
echo "========================================================================"
echo "所有实验完成！"
echo "========================================================================"
echo ""
echo "生成汇总报告..."

# 生成汇总报告
python3 << 'EOF'
import json
import os
from pathlib import Path

base_dir = "outputs/final_comparison"
seeds = [100, 200, 300]
modes = [
    ("pipeline_smart_overlap", "Pipeline Smart (预热)"),
    ("pipeline_naive_no_overlap", "Pipeline Naive (无预热)"),
    ("serial", "Serial (基准)")
]

print("=" * 80)
print("实验结果汇总")
print("=" * 80)
print()

results = []

for seed in seeds:
    print(f"种子 {seed}:")
    print("-" * 80)

    for mode_dir, mode_name in modes:
        best_path = Path(base_dir) / f"seed_{seed}" / mode_dir / "best_arch.json"

        if best_path.exists():
            with open(best_path) as f:
                data = json.load(f)

            arch = data["config"].get("time_proj", "unknown")
            val_mrr = data.get("val_mrr", data.get("mrr", 0))
            test_mrr = data.get("test_mrr", data.get("score", 0))

            print(f"  {mode_name:30s}: arch={arch:10s} Val={val_mrr:.4f} Test={test_mrr:.4f}")

            results.append({
                "seed": seed,
                "mode": mode_name,
                "arch": arch,
                "val_mrr": val_mrr,
                "test_mrr": test_mrr
            })
        else:
            print(f"  {mode_name:30s}: ✗ 未找到结果")

    print()

# 计算平均值
if results:
    print("=" * 80)
    print("平均值统计")
    print("=" * 80)
    print()

    for mode_dir, mode_name in modes:
        mode_results = [r for r in results if r["mode"] == mode_name]
        if mode_results:
            avg_val = sum(r["val_mrr"] for r in mode_results) / len(mode_results)
            avg_test = sum(r["test_mrr"] for r in mode_results) / len(mode_results)
            archs = [r["arch"] for r in mode_results]

            print(f"{mode_name:30s}:")
            print(f"  架构选择: {archs}")
            print(f"  平均Val MRR:  {avg_val:.4f}")
            print(f"  平均Test MRR: {avg_test:.4f}")
            print()

print("=" * 80)
print("详细结果保存在: outputs/final_comparison/")
print("=" * 80)

EOF

echo ""
echo "实验完成！查看汇总报告上方。"
