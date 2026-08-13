#!/usr/bin/env python3
"""Test: Serial vs Pipeline 使用 L2 (CE) Loss 是否一致。"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from jodie.data.public_dataset import load_public_dataset
from jodie.data.temporal_partition import build_partition_plan, TemporalPartition
from jodie.models.factory import build_model
from jodie.training.loops import reset_model_state, train_partition_ce

BASE_CONFIG = {
    "dataset": "public_csv", "dataset_dir": "data/public",
    "local_data_path": "data/public/mooc.csv",
    "max_events": 14000, "train_ratio": 0.7, "val_ratio": 0.1,
    "feature_dim": 4, "seed": 42, "partition_size": 2000,
    "partition_strategy": "count", "partition_overlap_ratio": 0.0,
    "device": "cuda:0",
}

ARCH = {
    "model": "jodie_rnn", "embedding_dim": 128, "memory_cell": "rnn",
    "time_proj": "off", "use_static_embeddings": "off", "normalize_state": "off",
    "lr": 0.001,
}

print("[1] Loading data...")
interactions, num_users, num_items = load_public_dataset(
    dataset_name=BASE_CONFIG["dataset"],
    dataset_dir=BASE_CONFIG["dataset_dir"],
    feature_dim=BASE_CONFIG["feature_dim"],
    max_events=BASE_CONFIG["max_events"],
    local_data_path=BASE_CONFIG["local_data_path"],
)
interactions = sorted(interactions, key=lambda x: x.timestamp)
total_events = len(interactions)
train_end = int(total_events * BASE_CONFIG["train_ratio"])
train_data = interactions[:train_end]

partition_plan = build_partition_plan(
    train_data, train_data[:10], train_data[:10],
    partition_size=BASE_CONFIG["partition_size"],
    strategy=BASE_CONFIG["partition_strategy"],
    overlap_ratio=BASE_CONFIG["partition_overlap_ratio"],
)
train_partitions = partition_plan.get_split_partitions("train")
print(f"  Partitions: {len(train_partitions)} × ~{BASE_CONFIG['partition_size']}")

# Build model, save init state
torch.manual_seed(BASE_CONFIG["seed"])
init_config = {**BASE_CONFIG, **ARCH, "num_users": num_users, "num_items": num_items}
init_model = build_model(init_config).to("cuda")
init_state = {k: v.cpu().clone() for k, v in init_model.state_dict().items()}
del init_model

# ── Serial training (one big partition) ──
print("\n[2] Serial L2 training (1 partition, all data)...")
single_part = TemporalPartition(
    partition_id=0, split="train",
    start_ts=train_data[0].timestamp, end_ts=train_data[-1].timestamp,
    interactions=list(train_data),
)

model_s = build_model(init_config).to("cuda")
model_s.load_state_dict(init_state)
opt_s = torch.optim.Adam(model_s.parameters(), lr=ARCH["lr"])

for epoch in range(3):
    reset_model_state(model_s)
    model_s.train()
    loss = train_partition_ce(model_s, single_part, opt_s)
    print(f"  Serial epoch {epoch}: loss={loss:.4f}")

s_emb = model_s.user_embeddings.data.float().mean().item()

# ── Pipeline training (multiple partitions) ──
print("\n[3] Pipeline L2 training (same data, partitioned)...")
model_p = build_model(init_config).to("cuda")
model_p.load_state_dict(init_state)
opt_p = torch.optim.Adam(model_p.parameters(), lr=ARCH["lr"])

for epoch in range(3):
    reset_model_state(model_p)
    model_p.train()
    for pid, partition in enumerate(train_partitions):
        loss = train_partition_ce(model_p, partition, opt_p)
    print(f"  Pipeline epoch {epoch}: loss={loss:.4f}")

p_emb = model_p.user_embeddings.data.float().mean().item()

# ── Compare ──
print(f"\n[4] Result:")
print(f"  Serial user_emb mean:   {s_emb:.10f}")
print(f"  Pipeline user_emb mean: {p_emb:.10f}")
diff = abs(s_emb - p_emb)
if diff < 1e-14:
    print(f"  Diff: {diff:.2e}  ✅ IDENTICAL")
else:
    print(f"  Diff: {diff:.2e}  ❌ DIFFERENT — L2 loss Pipeline diverges too!")
    # 检查第一个不一致的位置
    for name, (sp, pp) in zip(
        [n for n, _ in model_s.named_parameters()],
        zip(model_s.parameters(), model_p.parameters())
    ):
        d = (sp.data - pp.data).abs().max().item()
        if d > 1e-7:
            print(f"    First diff in '{name}': max_abs={d:.6e}")
            break
