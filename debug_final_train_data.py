"""调试脚本：验证final_train_data的实际大小"""
import sys
sys.path.insert(0, '/home/wanghaoyu/JODIE-simple')

from data.public_dataset import load_public_dataset
from data.temporal_partition import build_partition_plan

# 模拟trainer.py中的数据加载
max_events = 20000
train_ratio = 0.7
val_ratio = 0.1

interactions, num_users, num_items = load_public_dataset(
    dataset_name="wikipedia",
    dataset_dir="data/public",
    feature_dim=8,
    max_events=max_events,
    local_data_path="",
)

interactions = sorted(interactions, key=lambda x: x.timestamp)
total_events = len(interactions)
train_end = int(total_events * train_ratio)
val_end = int(total_events * (train_ratio + val_ratio))

train_data = interactions[:train_end]
val_data = interactions[train_end:val_end]
test_data = interactions[val_end:]

print(f"原始划分:")
print(f"  train_data: {len(train_data)} interactions")
print(f"  val_data: {len(val_data)} interactions")
print(f"  test_data: {len(test_data)} interactions")

# 模拟final test的连接操作
final_train_data = train_data + val_data
print(f"\nfinal_train_data = train_data + val_data:")
print(f"  final_train_data: {len(final_train_data)} interactions")

# 创建final_partition_plan
final_partition_plan = build_partition_plan(
    train_interactions=final_train_data,
    val_interactions=[],
    test_interactions=test_data,
    partition_size=None,
    strategy="count",
)

# 检查partition中的实际交互数
train_partitions = final_partition_plan.get_split_partitions("train")
total_train_interactions = sum(len(p.interactions) for p in train_partitions)

print(f"\nfinal_partition_plan中的train分区:")
print(f"  分区数: {len(train_partitions)}")
print(f"  总交互数: {total_train_interactions} interactions")

if total_train_interactions != len(final_train_data):
    print(f"\n❌ BUG发现！")
    print(f"  期望: {len(final_train_data)} interactions")
    print(f"  实际: {total_train_interactions} interactions")
    print(f"  差异: {len(final_train_data) - total_train_interactions} interactions丢失")
else:
    print(f"\n✅ 数据一致，无bug")
