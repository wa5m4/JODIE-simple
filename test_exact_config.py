"""
测试：使用best_arch.json的完整配置进行训练
"""
import json
import torch
import numpy as np
from data.public_dataset import load_public_dataset
from models.factory import build_model
from models.training import train_model_ce, evaluate_ranking_metrics
from data.temporal_partition import build_temporal_partitions

# 加载best_arch配置
with open('outputs/bug_fix_verification/seed_42/serial/best_arch.json') as f:
    best_arch = json.load(f)

config = best_arch['config'].copy()
seed = best_arch['seed']

print(f"使用seed: {seed}")
print(f"使用配置: time_proj={config['time_proj']}, model={config['model']}, memory_cell={config['memory_cell']}")

# 加载数据
all_interactions, num_users, num_items = load_public_dataset(
    dataset_name=config['dataset'],
    dataset_dir=config['dataset_dir'],
    feature_dim=config['feature_dim'],
    max_events=config['max_events'],
    local_data_path=config['local_data_path'],
)

# 划分数据
all_interactions.sort(key=lambda x: x.timestamp)
n_train = int(len(all_interactions) * config['train_ratio'])
n_val = int(len(all_interactions) * config['val_ratio'])

train_data = all_interactions[:n_train]
val_data = all_interactions[n_train:n_train+n_val]
test_data = all_interactions[n_train+n_val:]
final_train_data = train_data + val_data

# 更新config
config['num_users'] = num_users
config['num_items'] = num_items
config['seed'] = seed

# 设置种子
print("\n设置种子...")
torch.manual_seed(seed)
np.random.seed(seed)

# 构建模型
print("构建模型...")
model = build_model(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 创建partitions
partitions = build_temporal_partitions(
    final_train_data, split="train", partition_size=1000
)

# 训练
print(f"\n训练 (3 epochs)...")
train_model_ce(
    model=model,
    interactions=final_train_data,
    num_epochs=3,
    lr=config['lr'],
    graph_ctx=None,
    seed=seed,
    partitions=partitions,
    batch_training=config.get('batch_training', False),
    batch_size=config.get('train_batch_size', 32),
    batch_mode=config.get('batch_mode', 'tbatch'),
    tgn_loss_mode=config.get('tgn_loss_mode', 'all'),
    tgn_window_size=config.get('tgn_window_size', 10.0),
)

# 评估
print("\n评估...")
metrics = evaluate_ranking_metrics(
    model, test_data, k=10, graph_ctx=None, partitions=None,
    frozen=config.get('eval_frozen', False)
)

print(f"\n结果: Test MRR = {metrics['mrr']:.4f}")
print(f"期望: {best_arch['test_mrr']:.4f}")
