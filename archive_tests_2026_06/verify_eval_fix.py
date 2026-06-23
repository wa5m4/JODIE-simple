"""
验证评估bug修复：用Pipeline Naive架构测试
"""
import json
import time
from pathlib import Path

from data.public_dataset import load_public_dataset
from data.temporal_partition import build_temporal_partitions
from models.factory import build_model
from models.training import train_model, evaluate_ranking_metrics, reset_model_state

# Pipeline Naive的架构配置
config = {
    "model": "jodie_rnn",
    "embedding_dim": 32,
    "memory_cell": "rnn",
    "time_proj": "linear",
    "use_static_embeddings": "off",
    "normalize_state": "on",
    "event_agg": "none",
    "max_neighbors": 0,
    "batch_mode": "tbatch",
    "train_batch_size": 32,
    "dataset": "public_csv",
    "local_data_path": "data/public/mooc.csv",
    "max_events": 20000,
    "seed": 42,
    "lr": 0.001,
    "neg_sample_size": 5,
    "k": 10,
    "device": "cuda",
}

print("加载数据...")
all_interactions, num_users, num_items = load_public_dataset(
    dataset_name="public_csv",
    dataset_dir="data/public",
    feature_dim=8,
    max_events=20000,
    local_data_path="data/public/mooc.csv",
)

all_interactions.sort(key=lambda x: x.timestamp)
n_train = int(len(all_interactions) * 0.7)
n_val = int(len(all_interactions) * 0.1)

train_data = all_interactions[:n_train]
val_data = all_interactions[n_train:n_train + n_val]
test_data = all_interactions[n_train + n_val:]

config["num_users"] = num_users
config["num_items"] = num_items
config["feature_dim"] = 8

print(f"数据集: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

# 构建模型
model = build_model(config)
print(f"模型参数: {sum(p.numel() for p in model.parameters())}")

# 训练
print("\n训练 (3 epochs)...")
partition_plan = build_temporal_partitions(train_data, split="train", partition_size=1000)
train_model(
    model=model,
    interactions=train_data,
    num_epochs=3,
    lr=config["lr"],
    neg_sample_size=config["neg_sample_size"],
    seed=config["seed"],
    partitions=partition_plan,
    batch_size=config["train_batch_size"],
    batch_mode=config["batch_mode"],
)

# 测试1: 不重置状态（模拟原始bug）
print("\n=== 测试1: 不重置状态（原始bug） ===")
model.eval()
metrics_no_reset = evaluate_ranking_metrics(model, test_data, k=10, graph_ctx=None)
print(f"Test MRR: {metrics_no_reset['mrr']:.4f}")
print(f"Test Recall@10: {metrics_no_reset['recall_at_k']:.4f}")

# 测试2: 重置状态（修复后）
print("\n=== 测试2: 重置状态（修复后） ===")
model.eval()
reset_model_state(model)
metrics_with_reset = evaluate_ranking_metrics(model, test_data, k=10, graph_ctx=None)
print(f"Test MRR: {metrics_with_reset['mrr']:.4f}")
print(f"Test Recall@10: {metrics_with_reset['recall_at_k']:.4f}")

# 对比统一重训练的结果
print("\n=== 对比统一重训练结果 ===")
print(f"统一重训练 MRR: 0.6123")
print(f"修复后评估 MRR: {metrics_with_reset['mrr']:.4f}")
print(f"差异: {abs(metrics_with_reset['mrr'] - 0.6123):.4f}")

if abs(metrics_with_reset['mrr'] - 0.6123) < 0.05:
    print("\n✓ 修复验证成功！MRR与统一重训练结果一致")
else:
    print("\n✗ 修复验证失败！MRR与统一重训练结果不一致")
