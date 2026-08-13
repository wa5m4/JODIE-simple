#!/usr/bin/env python3
"""验证 Pipeline Ray 训练是否与 Serial 产生相同结果。"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from jodie.data.public_dataset import load_public_dataset
from jodie.data.temporal_partition import build_partition_plan, TemporalPartition
from jodie.models.factory import build_model
from jodie.training.loops import train_model_ce, train_partition_ce
from jodie.training.metrics import evaluate_ranking_metrics
from jodie.nas.ray_pipeline import RayPipelineExecutor

BASE_CONFIG = {
    "dataset": "public_csv", "dataset_dir": "data/public",
    "local_data_path": "data/public/mooc.csv",
    "max_events": 14000, "train_ratio": 0.7, "val_ratio": 0.1,
    "feature_dim": 4, "seed": 42, "partition_size": 2000,
    "partition_strategy": "count", "partition_overlap_ratio": 0.0,
    "num_pipeline_stages": 3, "device": "cuda:0",
    "pipeline_worker_gpus": 1.0,
    "pipeline_stage_train_workers": "1,1,1",
    "gpu_list": "0",
    "ray_address": "",
    "output_dir": "/tmp/pipeline_test",
    "batch_mode": "serial",
    "eval_frozen": False,
    "data_parallel_workers": 1,
}

ARCH = {
    "model": "jodie_rnn", "embedding_dim": 128, "memory_cell": "rnn",
    "time_proj": "off", "use_static_embeddings": "off", "normalize_state": "off",
    "lr": 0.001, "neg_sample_size": 5, "k": 10,
    "selection_metric": "mrr",
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
val_end = int(total_events * (BASE_CONFIG["train_ratio"] + BASE_CONFIG["val_ratio"]))
train_data = interactions[:train_end]
val_data = interactions[train_end:val_end]

partition_plan = build_partition_plan(
    train_data, val_data, interactions[val_end:],
    partition_size=BASE_CONFIG["partition_size"],
    strategy=BASE_CONFIG["partition_strategy"],
    overlap_ratio=BASE_CONFIG["partition_overlap_ratio"],
)
train_partitions = partition_plan.get_split_partitions("train")
print(f"  Partitions: {len(train_partitions)} × ~{BASE_CONFIG['partition_size']}")

# Build full config
full_config = {**BASE_CONFIG, **ARCH, "num_users": num_users, "num_items": num_items}

# Save init state for reproducible comparison
torch.manual_seed(BASE_CONFIG["seed"])
init_model = build_model(full_config).to("cuda")
init_state = {k: v.cpu().clone() for k, v in init_model.state_dict().items()}
del init_model

# ── Serial training ──
print("\n[2] Serial training (2 epochs)...")
model_s = build_model(full_config).to("cuda")
model_s.load_state_dict(init_state)

train_model_ce(
    model_s, train_data, num_epochs=2, lr=ARCH["lr"],
    seed=BASE_CONFIG["seed"], partitions=train_partitions,
    batch_mode="serial",
)

metrics_s = evaluate_ranking_metrics(model_s, val_data, k=10)
print(f"  Serial val MRR: {metrics_s['mrr']:.6f}, recall@10: {metrics_s['recall_at_k']:.6f}")

s_emb = model_s.user_embeddings.data.float().mean().item()
print(f"  Serial user_emb mean: {s_emb:.10f}")

# ── Pipeline training (via Ray executor) ──
print("\n[3] Pipeline training (via Ray executor, 2 epochs)...")

# Initialize Ray
try:
    import ray
    if not ray.is_initialized():
        ray.init(num_gpus=1, ignore_reinit_error=True)
except Exception:
    pass

executor = RayPipelineExecutor(full_config, partition_plan)
arch_configs = [dict(ARCH)]

trained = executor.run_train_only(arch_configs, num_train_epochs=2)
payload = trained[0]

# Load trained model
model_p = build_model(full_config).to("cuda")
model_p.load_state_dict(payload.model_state_dict)

metrics_p = evaluate_ranking_metrics(model_p, val_data, k=10)
print(f"  Pipeline val MRR: {metrics_p['mrr']:.6f}, recall@10: {metrics_p['recall_at_k']:.6f}")

p_emb = model_p.user_embeddings.data.float().mean().item()
print(f"  Pipeline user_emb mean: {p_emb:.10f}")

# ── Compare ──
print(f"\n[4] Comparison:")
print(f"  Serial MRR:   {metrics_s['mrr']:.10f}")
print(f"  Pipeline MRR: {metrics_p['mrr']:.10f}")
print(f"  Serial emb:   {s_emb:.10f}")
print(f"  Pipeline emb: {p_emb:.10f}")

emb_diff = abs(s_emb - p_emb)
mrr_diff = abs(metrics_s['mrr'] - metrics_p['mrr'])

if emb_diff < 1e-7 and mrr_diff < 1e-6:
    print(f"  ✅ Serial == Pipeline (Ray) — 结果完全一致!")
else:
    print(f"  ❌ 差异! emb_diff={emb_diff:.2e}, mrr_diff={mrr_diff:.2e}")

# Cleanup
try:
    ray.shutdown()
except Exception:
    pass
