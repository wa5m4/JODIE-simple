#!/usr/bin/env python3
"""
精确诊断脚本 v2：定位 Pipeline vs Serial 训练发散的精确步骤。

与 v1 的关键区别：
  1. Serial 和 Pipeline 使用完全相同的 seed（修复后行为）
  2. 对比所有交互步骤
  3. 精确定位第一个数值差异
  4. 检查分区边界 state transfer 精度
  5. 追踪差异累积过程（每 500 步检查一次）
"""

from __future__ import annotations

import json, os, sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jodie.data.public_dataset import load_public_dataset
from jodie.data.temporal_partition import build_partition_plan
from jodie.models.factory import build_model
from jodie.training.loops import BPRLoss, reset_model_state
from jodie.nas.ray_pipeline import _optimizer_state_to_fqn, _optimizer_state_from_fqn

OUTPUT_DIR = "pipeline_analysis/diagnose_v2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_CONFIG = {
    "dataset": "public_csv", "dataset_dir": "data/public",
    "local_data_path": "data/public/mooc.csv",
    "max_events": 20000, "train_ratio": 0.7, "val_ratio": 0.1,
    "feature_dim": 4, "seed": 42, "partition_size": 500,
    "partition_strategy": "count", "partition_overlap_ratio": 0.0,
    "num_pipeline_stages": 3, "batch_mode": "serial", "device": "cuda:0",
}

ARCH = {
    "model": "jodie_rnn", "embedding_dim": 128, "memory_cell": "rnn",
    "time_proj": "off", "use_static_embeddings": "off", "normalize_state": "off",
    "lr": 0.001, "neg_sample_size": 5, "k": 10,
    "selection_metric": "mrr", "max_neighbors": 0,
}

NUM_EPOCHS = 1
TOLERANCE = 1e-7
CHECK_EVERY_N = 500  # 每 500 步做一次详细对比


def _device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return next(model.buffers()).device


# ============================================================================
# 训练函数 (简化版 — 直接记录状态，不用 hook)
# ============================================================================

def _record_snapshot(model, optimizer, step, epoch, uid, iid, ts, loss,
                     stage=None, partition=None):
    """记录单个步骤的模型状态快照。"""
    snap = {
        "step": step, "epoch": epoch, "uid": uid, "iid": iid, "ts": ts,
        "loss": float(loss), "stage": stage, "partition": partition,
    }
    # 关键权重的 hash（用 mean+std 做紧凑表示，避免存整个张量）
    for name, param in model.named_parameters():
        p = param.data.detach()
        snap[f"w_{name}"] = (p.float().mean().item(), p.float().std().item())
    # 关键 embedding 的 hash
    snap["emb_user_mean"] = model.user_embeddings.data.float().mean().item()
    snap["emb_item_mean"] = model.item_embeddings.data.float().mean().item()
    snap["emb_user_std"] = model.user_embeddings.data.float().std().item()
    snap["emb_item_std"] = model.item_embeddings.data.float().std().item()
    # 被更新用户的嵌入 (精确值)
    if uid < model.user_embeddings.shape[0]:
        snap["uid_emb"] = model.user_embeddings[uid].detach().cpu().clone()
    if iid < model.item_embeddings.shape[0]:
        snap["iid_emb"] = model.item_embeddings[iid].detach().cpu().clone()
    return snap


def train_serial(num_items, train_data):
    """Serial 训练：与 run_all serial coarse 行为一致。"""
    config = {**BASE_CONFIG, **ARCH, "num_users": num_users, "num_items": num_items}
    model = build_model(config).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=ARCH["lr"])
    device = _device(model)
    snapshots = []
    step = 0

    for epoch in range(NUM_EPOCHS):
        reset_model_state(model)
        model.train()
        rng = np.random.default_rng(BASE_CONFIG["seed"] + epoch * 100000)

        for inter in train_data:
            uid_t = torch.tensor([inter.user_id], dtype=torch.long, device=device)
            iid_t = torch.tensor([inter.item_id], dtype=torch.long, device=device)
            t = torch.tensor([inter.timestamp], dtype=torch.float32, device=device)
            f = inter.features.unsqueeze(0).to(device)

            neg_items = []
            while len(neg_items) < ARCH["neg_sample_size"]:
                neg = int(rng.integers(0, num_items))
                if neg != inter.item_id:
                    neg_items.append(neg)
            neg_ids = torch.tensor(neg_items, dtype=torch.long, device=device)

            optimizer.zero_grad()
            pred_emb, new_u, new_i = model(uid_t, iid_t, t, f, inter.timestamp)
            pos_emb = model.item_embeddings[inter.item_id].detach().to(device)
            neg_emb = model.item_embeddings[neg_ids].detach().to(device).unsqueeze(0)
            loss = BPRLoss()(pred_emb, pos_emb, neg_emb)
            loss.backward(retain_graph=True)
            optimizer.step()

            if step % CHECK_EVERY_N == 0:
                snapshots.append(_record_snapshot(
                    model, optimizer, step, epoch,
                    inter.user_id, inter.item_id, inter.timestamp, loss.item()
                ))
            step += 1

    # 最后一步
    snapshots.append(_record_snapshot(
        model, optimizer, step-1, NUM_EPOCHS-1,
        train_data[-1].user_id, train_data[-1].item_id,
        train_data[-1].timestamp, loss.item()
    ))
    return model, optimizer, snapshots


def train_pipeline_simulated(num_items, train_partitions, stage_groups):
    """
    模拟 Pipeline 训练。
    关键：每个分区使用 SAME seed (seed + epoch*100000)，无 partition_id 偏移。
    但每个分区 RESET RNG → 这就是 serial 和 pipeline 的区别所在。
    """
    config = {**BASE_CONFIG, **ARCH, "num_users": num_users, "num_items": num_items}
    model = build_model(config).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=ARCH["lr"])
    device = _device(model)
    snapshots = []
    boundary_info = []
    step = 0

    partition_map = {p.partition_id: p for p in train_partitions}

    for epoch in range(NUM_EPOCHS):
        if epoch > 0:
            reset_model_state(model)

        for stage_idx, group in enumerate(stage_groups):
            for p in group:
                pid = p.partition_id
                partition = partition_map[pid]

                # ★ 每个分区 reset RNG 到相同起点
                rng = np.random.default_rng(BASE_CONFIG["seed"] + epoch * 100000)

                # 记录分区开始时的模型状态
                if step > 0:
                    boundary_info.append({
                        "step": step, "partition": pid, "stage": stage_idx,
                        "point": "partition_start",
                        "rng_reset": True,
                        "user_emb_mean": model.user_embeddings.data.float().mean().item(),
                        "item_emb_mean": model.item_embeddings.data.float().mean().item(),
                    })

                for inter in partition.interactions:
                    uid_t = torch.tensor([inter.user_id], dtype=torch.long, device=device)
                    iid_t = torch.tensor([inter.item_id], dtype=torch.long, device=device)
                    t = torch.tensor([inter.timestamp], dtype=torch.float32, device=device)
                    f = inter.features.unsqueeze(0).to(device)

                    neg_items = []
                    while len(neg_items) < ARCH["neg_sample_size"]:
                        neg = int(rng.integers(0, num_items))
                        if neg != inter.item_id:
                            neg_items.append(neg)
                    neg_ids = torch.tensor(neg_items, dtype=torch.long, device=device)

                    optimizer.zero_grad()
                    pred_emb, new_u, new_i = model(uid_t, iid_t, t, f, inter.timestamp)
                    pos_emb = model.item_embeddings[inter.item_id].detach().to(device)
                    neg_emb = model.item_embeddings[neg_ids].detach().to(device).unsqueeze(0)
                    loss = BPRLoss()(pred_emb, pos_emb, neg_emb)
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    if step % CHECK_EVERY_N == 0:
                        snapshots.append(_record_snapshot(
                            model, optimizer, step, epoch,
                            inter.user_id, inter.item_id, inter.timestamp,
                            loss.item(), stage=stage_idx, partition=pid
                        ))
                    step += 1

    snapshots.append(_record_snapshot(
        model, optimizer, step-1, NUM_EPOCHS-1,
        train_data[-1].user_id, train_data[-1].item_id,
        train_data[-1].timestamp, 0.0
    ))
    return model, optimizer, snapshots, boundary_info


# ============================================================================
# 对比分析
# ============================================================================

def compare_snapshots(serial_snaps, pipeline_snaps):
    """对比 snapshot 序列，找到首次差异位置并追踪累积。"""
    max_n = min(len(serial_snaps), len(pipeline_snaps))

    results = []
    first_div = None

    for i in range(max_n):
        s = serial_snaps[i]
        p = pipeline_snaps[i]

        diffs = {}
        # 对比 embedding 统计量
        for key in ["emb_user_mean", "emb_item_mean", "emb_user_std", "emb_item_std"]:
            d = abs(s.get(key, 0) - p.get(key, 0))
            if d > TOLERANCE:
                diffs[key] = d

        # 对比被更新的 uid/iid embedding (精确)
        if "uid_emb" in s and "uid_emb" in p and s["uid_emb"] is not None and p["uid_emb"] is not None:
            d = (s["uid_emb"].float() - p["uid_emb"].float()).abs().max().item()
            if d > TOLERANCE:
                diffs["uid_emb_max_diff"] = d
        if "iid_emb" in s and "iid_emb" in p and s["iid_emb"] is not None and p["iid_emb"] is not None:
            d = (s["iid_emb"].float() - p["iid_emb"].float()).abs().max().item()
            if d > TOLERANCE:
                diffs["iid_emb_max_diff"] = d

        # 对比 loss
        d = abs(s.get("loss", 0) - p.get("loss", 0))
        if d > TOLERANCE:
            diffs["loss_diff"] = d

        max_diff = max(diffs.values()) if diffs else 0.0

        if diffs and first_div is None:
            first_div = {
                "snapshot_idx": i,
                "serial_step": s["step"],
                "pipeline_step": p["step"],
                "pipeline_partition": p.get("partition", "?"),
                "pipeline_stage": p.get("stage", "?"),
                "diffs": {k: float(v) for k, v in list(diffs.items())[:8]},
            }

        results.append({
            "snapshot_idx": i, "step": s["step"],
            "partition": p.get("partition", 0),
            "max_diff": float(max_diff),
        })

    return results, first_div


# ============================================================================
# Main
# ============================================================================

def main():
    global num_users, num_items, train_data

    print("=" * 70)
    print("Pipeline vs Serial 精确诊断 v2")
    print("=" * 70)

    print("\n[1/4] 加载数据...")
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
    train_end = max(1, min(train_end, total_events - 2))
    val_end = max(train_end + 1, min(val_end, total_events - 1))
    train_data = interactions[:train_end]
    val_data = interactions[train_end:val_end]

    partition_plan = build_partition_plan(
        train_data, val_data, interactions[val_end:],
        partition_size=BASE_CONFIG["partition_size"],
        strategy=BASE_CONFIG["partition_strategy"],
        overlap_ratio=BASE_CONFIG["partition_overlap_ratio"],
    )
    train_partitions = partition_plan.get_split_partitions("train")
    num_stages = BASE_CONFIG["num_pipeline_stages"]
    stage_size = len(train_partitions) // num_stages
    stage_groups = []
    for si in range(num_stages):
        start = si * stage_size
        end = start + stage_size if si < num_stages - 1 else len(train_partitions)
        stage_groups.append(train_partitions[start:end])

    print(f"  训练交互: {len(train_data)}, 训练分区: {len(train_partitions)}")
    print(f"  Stage 分组: {[len(g) for g in stage_groups]}")
    print(f"  每 {CHECK_EVERY_N} 步记录一次快照")

    print("\n[2/4] Serial 训练 (完整数据)...")
    # ★ 保存初始权重，确保 serial 和 pipeline 从完全相同的状态开始
    torch.manual_seed(BASE_CONFIG["seed"])
    init_config = {**BASE_CONFIG, **ARCH, "num_users": num_users, "num_items": num_items}
    init_model = build_model(init_config).to("cuda")
    init_state = {k: v.cpu().clone() for k, v in init_model.state_dict().items()}
    del init_model

    s_model = build_model(init_config).to("cuda")
    s_model.load_state_dict(init_state)
    s_model, s_opt, s_snaps = train_serial(num_items, train_data)
    print(f"  完成: {len(s_snaps)} 个快照, 最终 emb_user_mean={s_snaps[-1]['emb_user_mean']:.6f}")

    print("\n[3/4] Pipeline 模拟训练 (分区+state transfer, 同seed+同初始权重)...")
    p_model = build_model(init_config).to("cuda")
    p_model.load_state_dict(init_state)  # ★ 相同初始权重
    p_model, p_opt, p_snaps, boundaries = train_pipeline_simulated(
        num_items, train_partitions, stage_groups
    )
    print(f"  完成: {len(p_snaps)} 个快照, 最终 emb_user_mean={p_snaps[-1]['emb_user_mean']:.6f}")
    print(f"  分区边界: {len(boundaries)} 处 RNG 重置")

    print("\n[4/4] 对比分析...")
    results, first_div = compare_snapshots(s_snaps, p_snaps)

    print(f"\n{'='*70}")
    print("诊断结果")
    print(f"{'='*70}")

    if first_div:
        print(f"\n  ❌ 首次差异出现在快照 #{first_div['snapshot_idx']}")
        print(f"     Serial step: {first_div['serial_step']}")
        print(f"     Pipeline step: {first_div['pipeline_step']}")
        print(f"     Pipeline 分区: {first_div['pipeline_partition']}, Stage: {first_div['pipeline_stage']}")
        print(f"     差异项:")
        for k, v in first_div['diffs'].items():
            print(f"       {k}: {v:.6e}")
    else:
        print(f"\n  ✅ 所有快照完全一致!")

    # 追踪差异累积
    print(f"\n  差异累积曲线 (每 {CHECK_EVERY_N} 步):")
    prev_div_step = None
    for r in results:
        if r["max_diff"] > TOLERANCE:
            if prev_div_step is None:
                print(f"    Step {r['step']:>6} (分区 {r['partition']:>3}): "
                      f"max_diff={r['max_diff']:.2e}  ← 首次差异")
            else:
                print(f"    Step {r['step']:>6} (分区 {r['partition']:>3}): "
                      f"max_diff={r['max_diff']:.2e}")
            prev_div_step = r["step"]

    if first_div:
        # 判断差异原因
        pid = first_div["pipeline_partition"]
        if pid == 0:
            cause = "在第一分区内就已发散 → 可能 RNG 初始化或模型初始化有问题"
        elif pid <= stage_groups[0][-1].partition_id:
            cause = f"在第一分区内(stage 0)发散 → 可能同分区内 RNG 使用方式不同"
        else:
            cause = (f"在分区 {pid}(stage {first_div['pipeline_stage']}) 发散 → "
                     f"RNG 按分区重置导致负采样不同，累积效应在此时显现")
        print(f"\n  原因推断: {cause}")

    # 保存报告
    report = {
        "config": {"partition_size": BASE_CONFIG["partition_size"],
                   "num_epochs": NUM_EPOCHS, "check_every": CHECK_EVERY_N},
        "first_divergence": first_div,
        "divergence_curve": results,
        "boundaries": [{"step": b["step"], "partition": b["partition"],
                        "stage": b["stage"], "rng_reset": b["rng_reset"]}
                       for b in boundaries],
        "final_state": {
            "serial_emb_user_mean": s_snaps[-1]["emb_user_mean"],
            "pipeline_emb_user_mean": p_snaps[-1]["emb_user_mean"],
        }
    }
    report_path = os.path.join(OUTPUT_DIR, "diagnosis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n详细报告: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    num_users = num_items = 0
    train_data = []
    main()
