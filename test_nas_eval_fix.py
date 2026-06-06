"""
测试修复后的评估逻辑：直接调用 _train_and_eval 模拟 NAS 搜索的评估
"""
import sys
sys.path.insert(0, '/home/wanghaoyu/JODIE-simple')

from nas.trainer import GraphNASTrainer
from data.public_dataset import load_public_dataset

# Pipeline Naive的架构配置
arch_config = {
    "model": "jodie_rnn",
    "embedding_dim": 32,
    "memory_cell": "rnn",
    "time_proj": "linear",
    "use_static_embeddings": "off",
    "normalize_state": "on",
    "event_agg": "none",
    "max_neighbors": 0,
}

base_config = {
    "dataset": "public_csv",
    "dataset_dir": "data/public",
    "local_data_path": "data/public/mooc.csv",
    "max_events": 20000,
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "feature_dim": 8,
    "lr": 0.001,
    "neg_sample_size": 5,
    "k": 10,
    "seed": 42,
    "batch_mode": "tbatch",
    "train_batch_size": 32,
    "device": "cuda",
    "num_users": 1435,
    "num_items": 21,
}

print("初始化 NAS Trainer...")
trainer = GraphNASTrainer(base_config)

print("准备数据...")
train_data, val_data, test_data, user_type_prefs, item_type, graph_template, partition_plan = trainer._prepare_data()

print(f"数据集: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

# 测试：用 _train_and_eval 在 test 数据上评估
print("\n=== 测试修复后的评估（模拟 NAS 搜索的 test 评估） ===")
config = dict(base_config)
config.update(arch_config)

metrics = trainer._train_and_eval(
    config=config,
    train_data=train_data + val_data,  # 最终评估用 train+val 训练
    eval_data=test_data,
    user_type_prefs=user_type_prefs,
    item_type=item_type,
    graph_template=graph_template,
    epochs=3,
    trial_seed=42 + 20000,
)

print(f"Test MRR: {metrics['mrr']:.4f}")
print(f"Test Recall@10: {metrics['recall_at_k']:.4f}")

print("\n=== 对比统一重训练结果 ===")
print(f"统一重训练 MRR: 0.6123")
print(f"修复后评估 MRR: {metrics['mrr']:.4f}")
print(f"差异: {abs(metrics['mrr'] - 0.6123):.4f}")

if abs(metrics['mrr'] - 0.6123) < 0.05:
    print("\n✓ 修复验证成功！MRR与统一重训练结果一致")
else:
    print("\n✗ MRR差异较大，可能还有其他问题")
