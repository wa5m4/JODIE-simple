"""
诊断脚本：精确复现Serial NAS Final Test
验证为什么NAS得到0.85而retrain得到0.34
"""
import sys
import json
import torch
import numpy as np

sys.path.insert(0, '/home/wanghaoyu/JODIE-simple')

from data.public_dataset import load_public_dataset
from models.factory import build_model
from models.training import train_model_ce, evaluate_ranking_metrics
from data.temporal_partition import build_temporal_partitions

# 加载best_arch配置
with open('outputs/bug_fix_verification/seed_42/serial/best_arch.json') as f:
    best_arch = json.load(f)

config = best_arch['config']
seed = best_arch['seed']

print("=" * 70)
print("精确复现Serial NAS Final Test")
print("=" * 70)
print(f"使用配置: time_proj={config['time_proj']}, seed={seed}")
print()

# 设置种子
torch.manual_seed(seed)
np.random.seed(seed)

# 加载数据
print("加载数据...")
interactions, num_users, num_items = load_public_dataset(
    dataset_name="wikipedia",
    dataset_dir="data/public",
    feature_dim=8,
    max_events=20000,
    local_data_path="data/public/mooc.csv",
)

# 划分数据
interactions.sort(key=lambda x: x.timestamp)
n_train = int(len(interactions) * 0.7)
n_val = int(len(interactions) * 0.1)

train_data = interactions[:n_train]
val_data = interactions[n_train:n_train+n_val]
test_data = interactions[n_train+n_val:]

final_train_data = train_data + val_data

print(f"数据划分: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
print(f"Final训练数据: {len(final_train_data)}")
print()

# 更新config
config['num_users'] = num_users
config['num_items'] = num_items
config['feature_dim'] = 8
config['seed'] = seed

# 构建模型
print("构建模型...")
model = build_model(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 创建partitions
partitions = build_temporal_partitions(
    final_train_data, split="train", partition_size=1000
)

# 训练 (使用train_model_ce，与NAS一致)
print(f"训练 (3 epochs, train_model_ce)...")
train_model_ce(
    model=model,
    interactions=final_train_data,
    num_epochs=3,
    lr=config['lr'],
    graph_ctx=None,
    seed=seed,
    partitions=partitions,
    batch_training=False,
    batch_size=config.get('train_batch_size', 32),
    batch_mode=config.get('batch_mode', 'tbatch'),
    tgn_loss_mode=config.get('tgn_loss_mode', 'all'),
    tgn_window_size=config.get('tgn_window_size', 10.0),
)

# 评估 (frozen=False，与NAS一致)
print("评估...")
metrics = evaluate_ranking_metrics(
    model,
    test_data,
    k=10,
    graph_ctx=None,
    partitions=None,
    frozen=False,  # 在线评估
)

print()
print("=" * 70)
print("结果:")
print("=" * 70)
print(f"Test MRR: {metrics['mrr']:.4f}")
print(f"NAS记录的test_mrr: {best_arch['test_mrr']:.4f}")
print(f"差异: {abs(metrics['mrr'] - best_arch['test_mrr']):.4f}")
print()

if abs(metrics['mrr'] - best_arch['test_mrr']) < 0.01:
    print("✓ 成功复现NAS结果！")
else:
    print("✗ 无法复现NAS结果，说明存在差异")
