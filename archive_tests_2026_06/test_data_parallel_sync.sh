#!/bin/bash

# 测试不同同步粒度的数据并行性能

SEED=100
MAX_EVENTS=20000
TRIALS=10  # 减少trials快速测试
EPOCHS=3
GPU_LIST="0,1,2"
OUTPUT_BASE="outputs/data_parallel_sync_test"

echo "========================================================================"
echo "数据并行同步粒度测试"
echo "========================================================================"

# 1. Partition粒度（最粗，最快）
echo ""
echo "[1/3] Partition粒度同步"
python search.py \
    --search-mode rl \
    --execution-mode data_parallel \
    --data-parallel-workers 3 \
    --data-parallel-sync-level partition \
    --gpu-list "$GPU_LIST" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events "$MAX_EVENTS" \
    --seed "$SEED" \
    --coarse-trials "$TRIALS" \
    --coarse-epochs "$EPOCHS" \
    --output-dir "${OUTPUT_BASE}/partition_sync" \
    --space rnn_only \
    --batch-mode tbatch \
    2>&1 | tee "${OUTPUT_BASE}/partition_sync.log"

# 2. 大micro-batch（4000）
echo ""
echo "[2/3] Micro-batch=4000"
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
    --output-dir "${OUTPUT_BASE}/micro_batch_4000" \
    --space rnn_only \
    --batch-mode tbatch \
    2>&1 | tee "${OUTPUT_BASE}/micro_batch_4000.log"

# 3. 当前默认（作为对比）
echo ""
echo "[3/3] 默认micro-batch（对比基准）"
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
    --output-dir "${OUTPUT_BASE}/default_sync" \
    --space rnn_only \
    --batch-mode tbatch \
    2>&1 | tee "${OUTPUT_BASE}/default_sync.log"

echo ""
echo "========================================================================"
echo "测试完成！生成对比报告..."
echo "========================================================================"

python3 << 'EOF'
import json
from pathlib import Path

configs = [
    ("partition_sync", "Partition粒度"),
    ("micro_batch_4000", "Micro-batch=4000"),
    ("default_sync", "默认(auto)"),
]

print("\n" + "=" * 70)
print("数据并行同步粒度对比")
print("=" * 70)
print(f"{'配置':<20} {'时间(s)':<12} {'Test MRR':<12} {'架构':<12}")
print("-" * 70)

for config_dir, name in configs:
    path = Path(f"outputs/data_parallel_sync_test/{config_dir}/best_arch.json")
    if path.exists():
        with open(path) as f:
            data = json.load(f)

        time_sec = data.get('time_sec', 0)
        test_mrr = data.get('test_mrr', 0)
        config = data['config']
        arch = f"{config['time_proj']}/{config['use_static_embeddings'][:2]}"

        print(f"{name:<20} {time_sec:<12.1f} {test_mrr:<12.4f} {arch:<12}")
    else:
        print(f"{name:<20} 未完成")

print("\n预期结果:")
print("  • Partition粒度: 最快（同步1次）")
print("  • Micro-batch=4000: 较快（同步3-4次）")
print("  • 默认: 最慢（同步100次）")

EOF

echo ""
echo "测试报告完成！"
