#!/usr/bin/env python
"""测试统一后的TGN接口"""
import torch
from data.public_dataset import load_public_dataset
from data.temporal_partition import TemporalPartition
from data.synthetic import init_dynamic_graph_state
from models.jodie_rnn import JODIERNN
from models.hybrid_jodie import TemporalEventGNNJODIE
from models.training import train_partition_bpr_tgn, BPRLoss

# 加载少量mooc数据
interactions, num_users, num_items = load_public_dataset(
    dataset_name="public_csv",
    dataset_dir="data/public",
    feature_dim=8,
    max_events=500,
    local_data_path="data/public/mooc.csv",
)

partition = TemporalPartition(
    partition_id=0,
    split="train",
    start_ts=float(interactions[0].timestamp),
    end_ts=float(interactions[-1].timestamp),
    interactions=interactions,
)

# 测试JODIERNN
print("[Test 1/2] Testing JODIERNN with TGN...")
model_rnn = JODIERNN(
    num_users=num_users,
    num_items=num_items,
    embedding_dim=64,
    feature_dim=8,
    cell_type="gru",
)
optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=1e-3)
criterion = BPRLoss()

loss_rnn = train_partition_bpr_tgn(
    model=model_rnn,
    partition=partition,
    optimizer=optimizer_rnn,
    criterion=criterion,
    time_window_size=10.0,
    aggregator="mean",
    loss_mode="last",
    neg_sample_size=3,
    seed=42,
    graph_ctx=None,
)
print(f"✓ JODIERNN TGN OK (loss={loss_rnn:.4f})")

# 测试TemporalEventGNNJODIE
print("\n[Test 2/2] Testing TemporalEventGNNJODIE with TGN...")
graph_ctx = init_dynamic_graph_state(num_users, num_items, max_neighbors=20)
model_hybrid = TemporalEventGNNJODIE(
    num_users=num_users,
    num_items=num_items,
    embedding_dim=64,
    feature_dim=8,
    event_agg="mean",
    memory_cell="gru",
)
optimizer_hybrid = torch.optim.Adam(model_hybrid.parameters(), lr=1e-3)

loss_hybrid = train_partition_bpr_tgn(
    model=model_hybrid,
    partition=partition,
    optimizer=optimizer_hybrid,
    criterion=criterion,
    time_window_size=10.0,
    aggregator="mean",
    loss_mode="last",
    neg_sample_size=3,
    seed=42,
    graph_ctx=graph_ctx,
)
print(f"✓ Hybrid TGN OK (loss={loss_hybrid:.4f})")

print("\n" + "="*50)
print("✓ 两个模型的TGN接口统一验证通过！")
print("="*50)
