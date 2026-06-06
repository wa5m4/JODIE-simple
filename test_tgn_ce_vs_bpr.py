#!/usr/bin/env python
"""测试TGN在CE loss和BPR loss下的表现差异"""
import torch
from data.public_dataset import load_public_dataset
from data.temporal_partition import TemporalPartition
from data.synthetic import init_dynamic_graph_state
from models.jodie_rnn import JODIERNN
from models.training import train_model, train_model_ce, evaluate_ranking_metrics

# 加载2000条mooc数据
interactions, num_users, num_items = load_public_dataset(
    dataset_name="public_csv",
    dataset_dir="data/public",
    feature_dim=8,
    max_events=2000,
    local_data_path="data/public/mooc.csv",
)

# 划分训练/测试集（70%/30%）
train_size = int(len(interactions) * 0.7)
train_data = interactions[:train_size]
test_data = interactions[train_size:]

print(f"数据集: {len(interactions)} 交互, {num_users} 用户, {num_items} 物品")
print(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")
print()

# 测试1: BPR loss + TGN (loss_mode=last)
print("="*60)
print("[Test 1/2] BPR loss + TGN (loss_mode=last)")
print("="*60)
model_bpr = JODIERNN(num_users, num_items, embedding_dim=64, feature_dim=8, cell_type="gru")
train_model(
    model_bpr, train_data, num_epochs=3, lr=1e-3,
    batch_mode="tgn", tgn_loss_mode="last", tgn_window_size=10.0,
    seed=42,
)
metrics_bpr = evaluate_ranking_metrics(model_bpr, test_data, k=10)
print(f"\n✓ BPR + TGN(last): MRR={metrics_bpr['mrr']:.4f}, Recall@10={metrics_bpr['recall_at_k']:.4f}")

# 测试2: CE loss + TGN (loss_mode=last)
print("\n" + "="*60)
print("[Test 2/2] CE loss + TGN (loss_mode=last)")
print("="*60)
model_ce = JODIERNN(num_users, num_items, embedding_dim=64, feature_dim=8, cell_type="gru")
train_model_ce(
    model_ce, train_data, num_epochs=3, lr=1e-3,
    batch_mode="tgn", tgn_loss_mode="last", tgn_window_size=10.0,
    seed=42,
)
metrics_ce = evaluate_ranking_metrics(model_ce, test_data, k=10)
print(f"\n✓ CE + TGN(last): MRR={metrics_ce['mrr']:.4f}, Recall@10={metrics_ce['recall_at_k']:.4f}")

# 比较结果
print("\n" + "="*60)
print("结果对比")
print("="*60)
print(f"BPR + TGN(last): MRR={metrics_bpr['mrr']:.4f}, Recall@10={metrics_bpr['recall_at_k']:.4f}")
print(f"CE  + TGN(last): MRR={metrics_ce['mrr']:.4f}, Recall@10={metrics_ce['recall_at_k']:.4f}")
print(f"MRR差异: {abs(metrics_bpr['mrr'] - metrics_ce['mrr']):.4f} ({abs(metrics_bpr['mrr'] - metrics_ce['mrr'])/metrics_bpr['mrr']*100:.1f}%)")

if metrics_ce['mrr'] < metrics_bpr['mrr'] * 0.5:
    print("\n⚠️  警告: CE loss的MRR显著低于BPR loss (低于50%)，存在问题！")
else:
    print("\n✓ CE loss的MRR在合理范围内")
