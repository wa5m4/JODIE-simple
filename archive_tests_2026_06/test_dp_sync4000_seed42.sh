#!/bin/bash

# 测试优化后的数据并行（同步粒度=4000）
# 使用Seed 42，其他参数与Serial保持一致

SEED=42
GPU_LIST="0,1,2"
MAX_EVENTS=20000
TRIALS=50
EPOCHS=3
OUTPUT_DIR="outputs/comprehensive_comparison/seed_42/data_parallel_sync4000"

echo "========================================================================"
echo "数据并行优化测试 (Seed 42, micro-batch-size=4000)"
echo "========================================================================"
echo "配置:"
echo "  种子: $SEED"
echo "  GPU: $GPU_LIST"
echo "  数据: $MAX_EVENTS events"
echo "  Trials: $TRIALS"
echo "  Epochs: $EPOCHS"
echo "  同步粒度: micro-batch-size=4000"
echo "========================================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode data_parallel \
    --data-parallel-workers 3 \
    --data-parallel-sync-level micro_batch \
    --data-parallel-micro-batch-size 4000 \
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
print(f"{'配置':<25} {'时间(s)':<12} {'架构':<12} {'Test MRR':<12}")
print("-" * 75)

configs = [
    ("serial", "Serial (基准)"),
    ("data_parallel", "数据并行 (默认sync)"),
    ("data_parallel_sync4000", "数据并行 (sync=4000)"),
]

results = []
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

        print(f"{mark} {name:<23} {time_sec:<12.1f} {arch:<12} {test_mrr:.4f}")
        results.append((name, time_sec, test_mrr, is_correct))
    else:
        print(f"⏳ {name:<23} 未完成")

if len(results) >= 2:
    print("\n" + "=" * 75)
    print("改进分析")
    print("=" * 75)

    serial_time = results[0][1]
    serial_test = results[0][2]

    if len(results) >= 3:
        new_time = results[2][1]
        new_test = results[2][2]
        new_correct = results[2][3]

        time_improve = (serial_time - new_time) / serial_time * 100
        test_diff = new_test - serial_test

        print(f"速度改进: {time_improve:+.1f}% ({new_time:.1f}s vs {serial_time:.1f}s)")
        print(f"Test差异: {test_diff:+.4f} ({new_test:.4f} vs {serial_test:.4f})")
        print(f"架构选择: {'✅ 正确' if new_correct else '❌ 错误'}")

        if new_correct and abs(test_diff) < 0.01:
            print("\n✅ 优化成功！数据并行达到Serial水平且速度更快")
        elif new_correct:
            print("\n✅ 架构选择正确，但性能仍有差距")
        else:
            print("\n⚠️ 仍需进一步优化同步粒度")

EOF

echo ""
echo "测试完成！"
