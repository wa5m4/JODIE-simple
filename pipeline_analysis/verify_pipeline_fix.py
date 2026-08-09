"""
=============================================================================
 用与 run_all 相同的训练参数验证排名 (tbatch + 3 stages + real data)
=============================================================================
"""

import os, sys, time, json, csv
from typing import Dict
import torch
import numpy as np

from jodie.data.synthetic import generate_synthetic_data
from jodie.data.temporal_partition import build_partition_plan
from jodie.models.factory import build_model
from jodie.training.loops import train_model
from jodie.training.metrics import evaluate_ranking_metrics
from jodie.nas.ray_pipeline import RayPipelineExecutor, create_ray_worker, _safe_ray_init
from jodie.nas.search_space import sanitize_config
from jodie.nas.controller import RandomGraphNASController
from jodie.nas.trainer import GraphNASTrainer

NUM_USERS = 200
NUM_ITEMS = 500
NUM_INTERACTIONS = 1000
FEATURE_DIM = 8
EPOCHS = 2
PARTITION_SIZE = 100
NUM_STAGES = 3  # 与 run_all 一致
BATCH_MODE = "tbatch"  # 与 run_all 一致
SEED = 42

# 测试三个不同架构
ARCHS = [
    ("A:proj=off,static=off", {
        "model": "jodie_rnn", "embedding_dim": 128,
        "memory_cell": "rnn", "time_proj": "off",
        "use_static_embeddings": "off", "normalize_state": "off",
        "event_agg": "none", "agg_activation": "none",
        "attn_type": "dot", "time_decay": "none",
        "hidden_dim": 0, "memory_gate": "off",
        "enable_event_agg": "off", "enable_graph_update": "off",
        "message_mode": "peer", "msg_linear": "off",
    }),
    ("B:proj=linear,static=on", {
        "model": "jodie_rnn", "embedding_dim": 128,
        "memory_cell": "rnn", "time_proj": "linear",
        "use_static_embeddings": "on", "normalize_state": "on",
        "event_agg": "none", "agg_activation": "none",
        "attn_type": "dot", "time_decay": "none",
        "hidden_dim": 0, "memory_gate": "off",
        "enable_event_agg": "off", "enable_graph_update": "off",
        "message_mode": "peer", "msg_linear": "off",
    }),
    ("C:proj=off,static=on", {
        "model": "jodie_rnn", "embedding_dim": 128,
        "memory_cell": "rnn", "time_proj": "off",
        "use_static_embeddings": "on", "normalize_state": "off",
        "event_agg": "none", "agg_activation": "none",
        "attn_type": "dot", "time_decay": "none",
        "hidden_dim": 0, "memory_gate": "off",
        "enable_event_agg": "off", "enable_graph_update": "off",
        "message_mode": "peer", "msg_linear": "off",
    }),
]


def make_config(device="cuda"):
    return {
        "dataset": "synthetic", "num_users": NUM_USERS,
        "num_items": NUM_ITEMS, "num_interactions": NUM_INTERACTIONS,
        "feature_dim": FEATURE_DIM, "lr": 1e-3,
        "neg_sample_size": 5, "k": 10,
        "selection_metric": "mrr", "device": device,
        "seed": SEED, "partition_size": PARTITION_SIZE,
        "partition_strategy": "count",
        "partition_overlap_ratio": 0.0,
        "num_pipeline_stages": NUM_STAGES,
        "pipeline_worker_gpus": 1.0, "pipeline_worker_cpus": 1.0,
        "pipeline_stage_train_workers": "1,1,1",
        "pipeline_stage_eval_workers": "1,1,1",
        "stage_balance_strategy": "cost",
        "pipeline_mode": "naive",
        "pipeline_trace": False, "pipeline_trace_log_path": "",
        "ray_address": "", "gpu_list": "0",
        "batch_mode": BATCH_MODE, "train_batch_size": 32,
        "batch_training": False,
        "tgn_loss_mode": "all", "tgn_window_size": 10.0,
        "eval_frozen": False, "max_neighbors": 0,
        "data_parallel_workers": 1, "data_parallel_worker_gpus": 1.0,
        "data_parallel_visible_gpus": "0",
        "enable_auto_pipeline_config": False,
    }


def train_serial(config, interactions, train_parts, val_parts):
    device = torch.device(config["device"])
    model = build_model(config).to(device)
    train_model(model, interactions, num_epochs=EPOCHS, lr=config["lr"],
                neg_sample_size=config["neg_sample_size"],
                graph_ctx=None, seed=SEED,
                partitions=train_parts, batch_mode=BATCH_MODE)
    val_m = evaluate_ranking_metrics(model, interactions, k=10, graph_ctx=None, partitions=val_parts)
    return float(val_m["mrr"])


def train_pipeline(config, plan, interactions, val_parts):
    device = torch.device(config["device"])
    executor = RayPipelineExecutor(dict(config), plan)
    payload = executor._make_payload(config, trial_id=0, seed=SEED)

    train_groups = executor._group_partitions("train", NUM_STAGES)
    train_workers = [[create_ray_worker(g, config)] for g in train_groups]
    trained = executor._run_train_pipeline([payload], train_groups, train_workers,
                                           use_bpr=True, num_train_epochs=EPOCHS)
    executor._shutdown_worker_pool(train_workers)

    final_payload = trained[0]
    model = build_model(config).to(device)
    model.load_state_dict({k: v.to(device) for k, v in final_payload.model_state_dict.items()})

    val_m = evaluate_ranking_metrics(model, interactions, k=10, graph_ctx=None, partitions=val_parts)
    executor.shutdown()
    return float(val_m["mrr"])


def main():
    print("=" * 60)
    print(f"  排名验证: tbatch + {NUM_STAGES} stages + 3 架构")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    interactions, _, _ = generate_synthetic_data(NUM_USERS, NUM_ITEMS, NUM_INTERACTIONS, FEATURE_DIM, SEED)
    n = len(interactions)
    train_ints = interactions[:int(n * 0.7)]
    val_ints = interactions[int(n * 0.7):int(n * 0.8)]
    test_ints = interactions[int(n * 0.8):]
    plan = build_partition_plan(train_ints, val_ints, test_ints,
                                partition_size=PARTITION_SIZE, strategy="count")
    train_parts = plan.get_split_partitions("train")
    val_parts = plan.get_split_partitions("val")

    import ray
    if ray.is_initialized():
        ray.shutdown()
    _safe_ray_init(ignore_reinit_error=True)

    results = {}
    for name, arch in ARCHS:
        config = make_config(device)
        config.update(arch)
        config = sanitize_config(config)

        sv = train_serial(config, interactions, train_parts, val_parts)
        pv = train_pipeline(config, plan, interactions, val_parts)
        results[name] = (sv, pv)
        print(f"  {name}: Serial={sv:.4f}  Pipeline={pv:.4f}")

    ray.shutdown()

    # 排名
    s_rank = sorted(ARCHS, key=lambda a: results[a[0]][0], reverse=True)
    p_rank = sorted(ARCHS, key=lambda a: results[a[0]][1], reverse=True)

    print(f"\n  Serial 排名:    {' > '.join([n for n,_ in s_rank])}")
    print(f"  Pipeline 排名:  {' > '.join([n for n,_ in p_rank])}")
    print(f"  {'✅ 一致!' if s_rank == p_rank else '❌ 不一致'}")

    return 0 if s_rank == p_rank else 1


if __name__ == "__main__":
    sys.exit(main())
