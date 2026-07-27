"""
=============================================================================
 逐层隔离测试 v2：训练同一个架构，对比各层模型权重是否与 Serial 一致
=============================================================================

 固定种子 → 训练同一个架构 → 比较权重。权重一致的层级无影响，
 权重最先偏离的层级即为根因。
"""

import os, sys, time, copy
import torch
import numpy as np

from jodie.data.synthetic import generate_synthetic_data
from jodie.data.temporal_partition import build_partition_plan
from jodie.models.factory import build_model
from jodie.training.loops import train_model, BPRLoss, train_partition_bpr_batch, reset_model_state
from jodie.training.metrics import evaluate_ranking_metrics
from jodie.nas.ray_pipeline import RayPipelineExecutor, create_ray_worker, _safe_ray_init, PipelineModelPayload
from jodie.nas.search_space import sanitize_config

NUM_USERS = 300
NUM_ITEMS = 500
NUM_INTERACTIONS = 3000
FEATURE_DIM = 8
EPOCHS = 2
PARTITION_SIZE = 300
NUM_STAGES = 3
SEED = 42

# 用 Serial 最优的那个架构
ARCH = {
    "model": "jodie_rnn", "embedding_dim": 64,
    "memory_cell": "rnn", "time_proj": "off",
    "use_static_embeddings": "off", "normalize_state": "off",
    "event_agg": "none", "agg_activation": "none",
    "attn_type": "dot", "time_decay": "none",
    "hidden_dim": 0, "memory_gate": "off",
    "enable_event_agg": "off", "enable_graph_update": "off",
    "message_mode": "peer", "msg_linear": "off",
}


def make_config(device="cpu"):
    c = {
        "dataset": "synthetic", "num_users": NUM_USERS, "num_items": NUM_ITEMS,
        "num_interactions": NUM_INTERACTIONS, "feature_dim": FEATURE_DIM,
        "lr": 1e-3, "neg_sample_size": 5, "k": 10,
        "selection_metric": "mrr", "device": device, "seed": SEED,
        "partition_size": PARTITION_SIZE, "partition_strategy": "count",
        "partition_overlap_ratio": 0.0,
        "num_pipeline_stages": NUM_STAGES,
        "pipeline_worker_gpus": 0.0, "pipeline_worker_cpus": 1.0,
        "pipeline_stage_train_workers": "1,1,1",
        "pipeline_stage_eval_workers": "1,1,1",
        "stage_balance_strategy": "count",
        "pipeline_mode": "naive",
        "pipeline_trace": False, "pipeline_trace_log_path": "",
        "ray_address": "", "gpu_list": "",
        "batch_mode": "serial", "train_batch_size": 32,
        "batch_training": False,
        "tgn_loss_mode": "all", "tgn_window_size": 10.0,
        "eval_frozen": False, "max_neighbors": 0,
        "data_parallel_workers": 1, "data_parallel_worker_gpus": 0.0,
        "data_parallel_visible_gpus": "",
        "enable_auto_pipeline_config": False,
    }
    c.update(ARCH)
    return sanitize_config(c)


def weights_fingerprint(model):
    """提取模型所有权重的展平向量 (前 200 维 + 哈希) 用于精确比较。"""
    vecs = []
    for name, p in model.named_parameters():
        vecs.append(p.data.float().flatten())
    full = torch.cat(vecs)
    total_sum = full.sum().item()
    total_norm = full.norm().item()
    head = full[:20].tolist()
    return {"sum": total_sum, "norm": total_norm, "head20": head, "n_params": full.numel()}


def print_diff(label, base_fp, test_fp):
    """打印与基线的差异。"""
    s_diff = abs(test_fp["sum"] - base_fp["sum"])
    n_diff = abs(test_fp["norm"] - base_fp["norm"])
    match = "✅ 相同" if (s_diff < 1e-3 and n_diff < 1e-3) else "❌ 偏离"
    print(f"  {label:<30}  sumΔ={s_diff:.2e}  normΔ={n_diff:.2e}  {match}")
    return s_diff > 1e-3 or n_diff > 1e-3


def l0_baseline(config, interactions, train_parts, val_parts):
    """L0: 纯 Serial。返回最终模型权重指纹。"""
    device = torch.device(config["device"])
    torch.manual_seed(SEED)
    model = build_model(config).to(device)
    train_model(model, interactions, num_epochs=EPOCHS, lr=config["lr"],
                neg_sample_size=config["neg_sample_size"],
                graph_ctx=None, seed=SEED,
                partitions=train_parts, batch_mode=config["batch_mode"])
    return weights_fingerprint(model), model


def l1_rebuild_only(config, interactions, train_parts, val_parts):
    """L1: 同进程，3 stage 间 rebuild model（不重建 optimizer）。"""
    device = torch.device(config["device"])
    torch.manual_seed(SEED)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    groups = _split_groups(train_parts, NUM_STAGES)

    for epoch in range(EPOCHS):
        if epoch > 0:
            reset_model_state(model)
        for si, group in enumerate(groups):
            if si > 0:
                sd = {k: v.cpu() for k, v in model.state_dict().items()}
                model = build_model(config).to(device)
                model.load_state_dict({k: v.to(device) for k, v in sd.items()})
            for p in group:
                train_partition_bpr_batch(
                    model=model, partition=p, optimizer=optimizer,
                    neg_sample_size=config["neg_sample_size"],
                    batch_size=config["train_batch_size"],
                    graph_ctx=None, seed=SEED + epoch * 100000 + p.partition_id)
    return weights_fingerprint(model)


def l2_opt_rebuild(config, interactions, train_parts, val_parts):
    """L2: L1 + optimizer rebuild。"""
    device = torch.device(config["device"])
    torch.manual_seed(SEED)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    groups = _split_groups(train_parts, NUM_STAGES)

    for epoch in range(EPOCHS):
        if epoch > 0:
            reset_model_state(model)
        for si, group in enumerate(groups):
            if si > 0:
                sd = {k: v.cpu() for k, v in model.state_dict().items()}
                os_ = {k: v for k, v in optimizer.state_dict().items()}
                model = build_model(config).to(device)
                model.load_state_dict({k: v.to(device) for k, v in sd.items()})
                optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
                optimizer.load_state_dict(os_)
            for p in group:
                train_partition_bpr_batch(
                    model=model, partition=p, optimizer=optimizer,
                    neg_sample_size=config["neg_sample_size"],
                    batch_size=config["train_batch_size"],
                    graph_ctx=None, seed=SEED + epoch * 100000 + p.partition_id)
    return weights_fingerprint(model)


def l3_add_epoch_reset(config, interactions, train_parts, val_parts):
    """L3: L2 + epoch 边界 reset_state()（修复 runtime_state=None）。"""
    device = torch.device(config["device"])
    torch.manual_seed(SEED)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    groups = _split_groups(train_parts, NUM_STAGES)

    for epoch in range(EPOCHS):
        for si, group in enumerate(groups):
            if si > 0 or epoch > 0:
                sd = {k: v.cpu() for k, v in model.state_dict().items()}
                os_ = {k: v for k, v in optimizer.state_dict().items()}
                model = build_model(config).to(device)
                model.load_state_dict({k: v.to(device) for k, v in sd.items()})
                optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
                optimizer.load_state_dict(os_)
                if si == 0 and epoch > 0:
                    model.reset_state()  # ← 关键修复
            for p in group:
                train_partition_bpr_batch(
                    model=model, partition=p, optimizer=optimizer,
                    neg_sample_size=config["neg_sample_size"],
                    batch_size=config["train_batch_size"],
                    graph_ctx=None, seed=SEED + epoch * 100000 + p.partition_id)
    return weights_fingerprint(model)


def l4_full_simulation(config, interactions, train_parts, val_parts):
    """L4: 完整 Pipeline 模拟（同进程，等价于 Ray Pipeline 但无 RPC）。"""
    device = torch.device(config["device"])
    torch.manual_seed(SEED)

    # 初始 payload
    m0 = build_model(config).to(device)
    runtime = m0.export_runtime_state()
    payload = PipelineModelPayload(
        trial_id=0, arch_config=config,
        model_state_dict={k: v.cpu() for k, v in m0.state_dict().items()},
        runtime_state={k: v.cpu() for k, v in runtime.items()},
        graph_state=None, optimizer_state=None, seed=SEED)

    groups = _split_groups(train_parts, NUM_STAGES)
    stage_ids = [[p.partition_id for p in g] for g in groups]
    id_to_part = {p.partition_id: p for p in train_parts}

    for epoch in range(EPOCHS):
        if epoch > 0:
            payload = PipelineModelPayload(
                trial_id=0, arch_config=config,
                model_state_dict=payload.model_state_dict,
                runtime_state=None, graph_state=None,
                optimizer_state=payload.optimizer_state, seed=SEED)

        for si in range(NUM_STAGES):
            model = build_model(config).to(device)
            model.load_state_dict({k: v.to(device) for k, v in payload.model_state_dict.items()})
            if payload.runtime_state is not None:
                model.import_runtime_state({k: v.to(device) for k, v in payload.runtime_state.items()})
            else:
                model.reset_state()

            opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
            if payload.optimizer_state is not None:
                opt.load_state_dict({k: v for k, v in payload.optimizer_state.items()})

            for pid in stage_ids[si]:
                p = id_to_part[pid]
                train_partition_bpr_batch(
                    model=model, partition=p, optimizer=opt,
                    neg_sample_size=config["neg_sample_size"],
                    batch_size=config["train_batch_size"],
                    graph_ctx=None, seed=SEED + epoch * 100000 + pid)

            runtime = model.export_runtime_state()
            payload = PipelineModelPayload(
                trial_id=0, arch_config=config,
                model_state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                runtime_state={k: v.cpu() for k, v in runtime.items()},
                graph_state=None,
                optimizer_state={k: v for k, v in opt.state_dict().items()},
                seed=SEED)

    return weights_fingerprint(model)


def l5_ray_pipeline(config, plan, train_parts, val_parts):
    """L5: 真正的 Ray Pipeline。"""
    executor = RayPipelineExecutor(dict(config), plan)
    payload = executor._make_payload(config, trial_id=0, seed=SEED)
    train_groups = executor._group_partitions("train", NUM_STAGES)
    tw = [[create_ray_worker(g, config)] for g in train_groups]
    trained = executor._run_train_pipeline([payload], train_groups, tw,
                                           use_bpr=True, num_train_epochs=EPOCHS)
    executor._shutdown_worker_pool(tw)

    device = torch.device(config["device"])
    model = build_model(config).to(device)
    fp = trained[0]
    model.load_state_dict({k: v.to(device) for k, v in fp.model_state_dict.items()})
    if fp.runtime_state:
        model.import_runtime_state({k: v.to(device) for k, v in fp.runtime_state.items()})
    executor.shutdown()
    return weights_fingerprint(model)


def _split_groups(parts, n):
    b = len(parts) // n
    r = len(parts) % n
    gs, s = [], 0
    for i in range(n):
        sz = b + (1 if i < r else 0)
        gs.append(parts[s:s + sz])
        s += sz
    return gs


def main():
    print("=" * 60)
    print("  逐层隔离测试 v2：权重指纹对比")
    print("=" * 60)
    print(f"  架构: proj=off,static=off  | 数据:{NUM_USERS}×{NUM_ITEMS}×{NUM_INTERACTIONS}")
    print(f"  epochs={EPOCHS}, stages={NUM_STAGES}, partition_size={PARTITION_SIZE}")
    print()

    device = "cpu"
    config = make_config(device)

    interactions, _, _ = generate_synthetic_data(
        NUM_USERS, NUM_ITEMS, NUM_INTERACTIONS, FEATURE_DIM, SEED)
    n = len(interactions)
    train_ints = interactions[:int(n * 0.7)]
    val_ints = interactions[int(n * 0.7):int(n * 0.8)]
    test_ints = interactions[int(n * 0.8):]
    plan = build_partition_plan(train_ints, val_ints, test_ints,
                                partition_size=PARTITION_SIZE, strategy="count")
    train_parts = plan.get_split_partitions("train")
    val_parts = plan.get_split_partitions("val")

    print(f"  训练分区: {len(train_parts)}, 每组: {[len(g) for g in _split_groups(train_parts, NUM_STAGES)]}")
    print()

    # 初始化 Ray
    import ray
    if ray.is_initialized():
        ray.shutdown()
    _safe_ray_init(ignore_reinit_error=True)

    # L0 baseline
    print("  训练中...")
    base_fp, base_model = l0_baseline(config, interactions, train_parts, val_parts)
    print(f"\n  {'='*55}")
    print(f"  基线 Serial 权重指纹: sum={base_fp['sum']:.4f}, norm={base_fp['norm']:.4f}, params={base_fp['n_params']}")
    print(f"  {'='*55}\n")

    tests = [
        ("L1 model rebuild", lambda: l1_rebuild_only(config, interactions, train_parts, val_parts)),
        ("L2 +opt rebuild", lambda: l2_opt_rebuild(config, interactions, train_parts, val_parts)),
        ("L3 +epoch reset_state", lambda: l3_add_epoch_reset(config, interactions, train_parts, val_parts)),
        ("L4 Pipeline模拟(同进程)", lambda: l4_full_simulation(config, interactions, train_parts, val_parts)),
        ("L5 Ray Pipeline(真)", lambda: l5_ray_pipeline(config, plan, train_parts, val_parts)),
    ]

    first_diverged = None
    for name, fn in tests:
        fp = fn()
        diverged = print_diff(name, base_fp, fp)
        if diverged and first_diverged is None:
            first_diverged = name

    ray.shutdown()

    print(f"\n  {'='*55}")
    if first_diverged:
        print(f"  根因定位: 权重在 [{first_diverged}] 首次偏离基线")
        print(f"  该层级引入的差异即为 Pipeline 评分偏差的来源")
    else:
        print(f"  所有层级权重与基线一致 — 差异可能来自评估方式或数据特征")
    print(f"  {'='*55}")


if __name__ == "__main__":
    main()
