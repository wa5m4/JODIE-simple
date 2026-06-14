#!/bin/bash

# Pipeline方案2测试：Partition重叠预热（20%重叠）
# 已恢复原来的reset机制，启用Partition重叠

SEED=1000
GPU_LIST="0,1,2"
MAX_EVENTS=20000
COARSE_TRIALS=50
COARSE_EPOCHS=3
OUTPUT_DIR="outputs/pipeline_overlap/seed_${SEED}"

echo "=========================================="
echo "Pipeline方案2测试：Partition重叠预热"
echo "=========================================="
echo "配置: 20%重叠 + 原始reset机制"
echo "预期: 缓解Cold Start，选出正确架构"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

# 注意：需要修改search.py来支持overlap_ratio参数
# 暂时先手动测试overlap功能

echo ""
echo "Pipeline Smart（20%重叠）..."

# 创建测试脚本
python3 << 'EOF'
"""
测试Partition重叠功能
"""
import sys
sys.path.insert(0, '/home/wanghaoyu/JODIE-simple')

from data.public_dataset import load_public_dataset
from data.temporal_partition import build_partition_plan

# 加载数据
interactions, num_users, num_items = load_public_dataset(
    dataset_name='public_csv',
    dataset_dir='data/public',
    feature_dim=8,
    max_events=20000,
    local_data_path='data/public/mooc.csv',
)

train_split = int(len(interactions) * 0.7)
train_data = interactions[:train_split]

print("="*70)
print("测试Partition重叠功能")
print("="*70)
print()

# 测试无重叠
plan_no_overlap = build_partition_plan(
    train_interactions=train_data,
    val_interactions=[],
    test_interactions=[],
    partition_size=5000,
    overlap_ratio=0.0,
)

# 测试20%重叠
plan_overlap = build_partition_plan(
    train_interactions=train_data,
    val_interactions=[],
    test_interactions=[],
    partition_size=5000,
    overlap_ratio=0.2,
)

train_parts_no = plan_no_overlap.get_split_partitions('train')
train_parts_yes = plan_overlap.get_split_partitions('train')

print(f"训练数据: {len(train_data)}个事件")
print()
print(f"无重叠: {len(train_parts_no)}个partition")
for i, p in enumerate(train_parts_no):
    print(f"  P{i}: [{p.start_ts:.0f}, {p.end_ts:.0f}), {len(p.interactions)}个事件")

print()
print(f"20%重叠: {len(train_parts_yes)}个partition")
for i, p in enumerate(train_parts_yes):
    print(f"  P{i}: [{p.start_ts:.0f}, {p.end_ts:.0f}), {len(p.interactions)}个事件")
    if i > 0:
        prev = train_parts_yes[i-1]
        overlap_count = sum(1 for inter in p.interactions
                          if inter.timestamp >= prev.start_ts and inter.timestamp <= prev.end_ts)
        print(f"       与P{i-1}重叠: {overlap_count}个事件 ({overlap_count/len(p.interactions)*100:.1f}%)")

print()
print("="*70)
print("结论")
print("="*70)
print()
if len(train_parts_yes) > len(train_parts_no):
    print(f"✓ 重叠功能生效")
    print(f"  • 无重叠: {len(train_parts_no)}个partition")
    print(f"  • 20%重叠: {len(train_parts_yes)}个partition (增加了{len(train_parts_yes)-len(train_parts_no)}个)")
    print(f"  • 每个partition与前一个有约20%的数据重叠")
    print()
    print("效果:")
    print("  • 新entity在前一个partition中已经出现，有'预热'")
    print("  • 缓解Cold Start问题")
else:
    print("✗ 重叠功能可能未生效")

EOF

echo ""
echo "=========================================="
echo "Partition重叠功能验证完成"
echo "=========================================="
