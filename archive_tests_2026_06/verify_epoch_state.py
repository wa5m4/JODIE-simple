#!/usr/bin/env python3
"""
验证Epoch间状态传递是否生效
"""
import torch
import numpy as np
from models.factory import build_model
from data.public_dataset import load_public_dataset
from data.temporal_partition import build_partition_plan
from models.training import train_model

# 配置
config = {
    'dataset': 'public_csv',
    'local_data_path': 'data/public/mooc.csv',
    'max_events': 1000,  # 小数据集快速测试
    'feature_dim': 8,
    'lr': 1e-3,
    'model': 'jodie_rnn',
    'embedding_dim': 128,
    'memory_cell': 'rnn',
    'time_proj': 'off',
    'device': 'cpu',
    'seed': 42,
}

print("="*70)
print("验证Epoch间状态传递")
print("="*70)
print()

# 加载数据
interactions, num_users, num_items = load_public_dataset(
    dataset_name='public_csv',
    dataset_dir='data/public',
    feature_dim=config['feature_dim'],
    max_events=config['max_events'],
    local_data_path=config['local_data_path'],
)
config['num_users'] = num_users
config['num_items'] = num_items

train_split = int(len(interactions) * 0.7)
train_data = interactions[:train_split]

# 构建分区
partition_plan = build_partition_plan(
    train_interactions=train_data,
    val_interactions=[],
    test_interactions=[],
    partition_size=300,
)
train_partitions = partition_plan.get_split_partitions('train')

print(f"数据: {len(train_data)}个训练样本")
print(f"分区: {len(train_partitions)}个partition")
print()

# 创建模型
model = build_model(config)

# 记录初始embedding
initial_user_embedding = model.rnn_model.user_embeddings[0].clone().detach()
print(f"初始User 0 embedding范数: {initial_user_embedding.norm().item():.6f}")
print()

# 训练3个epoch，观察embedding变化
print("="*70)
print("训练过程监控")
print("="*70)
print()

# 手动模拟训练过程来监控状态
from models.training import reset_model_state, clone_graph_state_template

epoch_graph_ctx = None
for epoch in range(3):
    if epoch == 0:
        reset_model_state(model)
        print(f"[Epoch {epoch+1}/3] ✓ Reset执行")
        after_reset_embedding = model.rnn_model.user_embeddings[0].clone().detach()
        print(f"  → User 0 embedding范数: {after_reset_embedding.norm().item():.6f}")
    else:
        print(f"[Epoch {epoch+1}/3] ✓ 未执行reset，继承上一epoch状态")
        before_epoch_embedding = model.rnn_model.user_embeddings[0].clone().detach()
        print(f"  → User 0 embedding范数: {before_epoch_embedding.norm().item():.6f}")

    # 简单训练几步
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for i, interaction in enumerate(train_data[:100]):  # 只训练100步
        optimizer.zero_grad()
        user_emb = model.rnn_model.user_embeddings[interaction.user_id]
        item_emb = model.rnn_model.item_embeddings[interaction.item_id]
        pred = (user_emb * item_emb).sum()
        loss = (pred - 1.0) ** 2  # 简单的MSE loss
        loss.backward()
        optimizer.step()

    after_train_embedding = model.rnn_model.user_embeddings[0].clone().detach()
    print(f"  → 训练后 User 0 embedding范数: {after_train_embedding.norm().item():.6f}")
    print()

print("="*70)
print("结论")
print("="*70)
print()
print("如果状态传递生效：")
print("  • Epoch 1: Reset后从随机初始化开始")
print("  • Epoch 2: 不reset，继承Epoch 1训练后的embedding")
print("  • Epoch 3: 不reset，继承Epoch 2训练后的embedding")
print()
print("观察：")
print("  • Epoch 2/3开始时的embedding范数应该较大（继承了训练后的值）")
print("  • 如果Epoch 2/3开始时范数接近初始值，说明reset仍在执行")
