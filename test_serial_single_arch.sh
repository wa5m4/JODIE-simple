#!/bin/bash

# 用Serial模式训练方案B选出的最佳架构（单次训练，非搜索）

SEED=100
MAX_EVENTS=20000
EPOCHS=3

echo "========================================================================"
echo "Serial单架构训练：验证Two-Stage策略"
echo "========================================================================"
echo ""
echo "目标："
echo "  • 读取方案B选出的最佳架构配置"
echo "  • 用Serial模式训练该架构（无分区）"
echo "  • 对比性能：Pipeline训练 vs Serial训练"
echo ""
echo "========================================================================"
echo ""

# 读取方案B的最佳架构配置
BEST_ARCH_FILE="outputs/partition_improvements/seed_100/p3000_o5pct/best_arch.json"

if [ ! -f "$BEST_ARCH_FILE" ]; then
    echo "错误：找不到方案B的best_arch.json"
    exit 1
fi

echo "读取方案B的最佳架构配置..."
echo ""

# 提取架构参数
MODEL=$(python3 -c "import json; print(json.load(open('$BEST_ARCH_FILE'))['config']['model'])")
EMB_DIM=$(python3 -c "import json; print(json.load(open('$BEST_ARCH_FILE'))['config']['embedding_dim'])")
MEMORY_CELL=$(python3 -c "import json; print(json.load(open('$BEST_ARCH_FILE'))['config']['memory_cell'])")
TIME_PROJ=$(python3 -c "import json; print(json.load(open('$BEST_ARCH_FILE'))['config']['time_proj'])")
USE_STATIC=$(python3 -c "import json; print(json.load(open('$BEST_ARCH_FILE'))['config']['use_static_embeddings'])")

echo "架构配置："
echo "  model:            $MODEL"
echo "  embedding_dim:    $EMB_DIM"
echo "  memory_cell:      $MEMORY_CELL"
echo "  time_proj:        $TIME_PROJ"
echo "  use_static_emb:   $USE_STATIC"
echo ""

OUTPUT_DIR="outputs/serial_single_arch_retrain/seed_${SEED}"
mkdir -p "$OUTPUT_DIR"

echo "开始Serial训练（无分区，完整数据）..."
echo ""

python train_single_arch.py \
    --model "$MODEL" \
    --embedding-dim "$EMB_DIM" \
    --memory-cell "$MEMORY_CELL" \
    --time-proj "$TIME_PROJ" \
    --use-static-embeddings "$USE_STATIC" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events "$MAX_EVENTS" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --batch-mode tbatch \
    --eval-frozen false \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "${OUTPUT_DIR}.log"

echo ""
echo "========================================================================"
echo "结果对比"
echo "========================================================================"

python3 << 'EOF'
import json
from pathlib import Path

print()
print("=" * 80)
print("Serial单架构训练结果")
print("=" * 80)
print()

# 读取结果
serial_result_path = Path("outputs/serial_single_arch_retrain/seed_100/result.json")
pipeline_b_path = Path("outputs/partition_improvements/seed_100/p3000_o5pct/best_arch.json")
baseline_path = Path("outputs/final_comparison/seed_100/pipeline_naive_no_overlap/best_arch.json")

results = []

if pipeline_b_path.exists():
    with open(pipeline_b_path) as f:
        data = json.load(f)
    arch = data['config'].get('time_proj')
    val = data.get('val_mrr', 0)
    test = data.get('test_mrr', 0)
    print(f"方案B (Pipeline p3000,o5%): arch={arch:7s} Val={val:.4f} Test={test:.4f}")
    results.append(("Pipeline训练", test))

if serial_result_path.exists():
    with open(serial_result_path) as f:
        data = json.load(f)
    val = data.get('val_mrr', 0)
    test = data.get('test_mrr', 0)
    print(f"Serial单架构训练:          arch={data['config'].get('time_proj'):7s} Val={val:.4f} Test={test:.4f}")
    results.append(("Serial训练", test))
else:
    print("Serial单架构训练: 未完成或结果文件不存在")

if baseline_path.exists():
    with open(baseline_path) as f:
        data = json.load(f)
    arch = data['config'].get('time_proj')
    val = data.get('val_mrr', 0)
    test = data.get('test_mrr', 0)
    print(f"基准 (Pipeline p5000,o0%):  arch={arch:7s} Val={val:.4f} Test={test:.4f}")
    results.append(("基准", test))

print()
print("=" * 80)
print("Two-Stage策略验证")
print("=" * 80)
print()

if len(results) >= 2:
    pipeline_test = results[0][1]
    serial_test = results[1][1]
    improvement = serial_test - pipeline_test

    print(f"Pipeline训练 (p3000,o5%): Test={pipeline_test:.4f}")
    print(f"Serial训练   (同架构):    Test={serial_test:.4f} ({improvement:+.4f})")
    print()

    if serial_test >= 0.83:
        print("✅ Two-Stage策略成功！")
        print()
        print("结论:")
        print("  • Stage 1: Pipeline NAS (p3000,o5%) 成功选出正确架构")
        print("  • Stage 2: Serial训练恢复完整性能")
        print(f"  • 最终Test MRR: {serial_test:.4f} (达到基准水平)")
        print()
        print("推荐工作流程:")
        print("  1. 用Pipeline NAS快速搜索架构 (更快，分数可能偏低)")
        print("  2. 用Serial训练最佳架构 (充分训练，获得真实性能)")
    elif improvement > 0.03:
        print("✓ Serial训练有显著改进")
        print()
        print(f"改进: {improvement:+.4f} ({improvement/pipeline_test*100:+.1f}%)")
        print()
        print("但未达到基准水平 (~0.83)，可能需要:")
        print("  • Stage 1改进: partition=5000 + overlap=5%")
        print("  • 或增加训练epochs")
    else:
        print("⚠️  Serial训练未见显著改进")
        print()
        print("可能原因:")
        print("  • partition=3000太小，导致数据分布问题")
        print("  • 训练方式不是主要瓶颈")
        print()
        print("建议: Stage 1改用 partition=5000 + overlap=5%")

EOF

echo ""
echo "实验完成！"
