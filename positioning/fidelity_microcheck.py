#!/usr/bin/env python
"""保真度微对比(Fix C 验证):同架构同种子,serial vs pipeline 训练+评估是否位级等价。

用法(仓库根,必须用 env 限制可见卡,ray_pipeline 不读 GPU_LIST):
  env CUDA_VISIBLE_DEVICES=0,1,2 python positioning/fidelity_microcheck.py --cell D --seed 42
  env CUDA_VISIBLE_DEVICES=0,1   python positioning/fidelity_microcheck.py --cell B --seed 42

对比项:
  1. 分数:serial 路径(真实 _train_and_eval) vs pipeline 路径(真实 evaluate_arch_pipeline
     方案C)的 val score,逐位比较;
  2. 终态:两条路径训练后的 model.state_dict() 逐张量 max|Δ|(脚本内复刻训练步骤保留模型,
     并先与真实路径分数互校,证明复刻忠实);
  3. 载荷种子:确认 run_train_only 的 payload.seed == 42 + trial_ids[i](Fix B 生效)。

架构默认用 D serial 的 8896(comparison.json 中的 best_config,rnn_only 空间)。
"""
import argparse
import copy
import json
import random
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch

from jodie.models.factory import build_model
from jodie.nas.ray_pipeline import RayPipelineExecutor
from jodie.nas.trainer import GraphNASTrainer
from jodie.training.loops import train_model_ce
from jodie.training.metrics import evaluate_ranking_metrics

# D serial 8896 的架构(rnn_only 空间,取自 experiments/results/cellD_serial_20260909/comparison.json)
ARCH_8896 = {
    "model": "jodie_rnn", "embedding_dim": 32, "memory_cell": "rnn",
    "time_proj": "off", "use_static_embeddings": "off", "normalize_state": "off",
    "event_agg": "none", "agg_activation": "none", "attn_type": "dot",
    "time_decay": "none", "hidden_dim": 0, "memory_gate": "off",
    "enable_event_agg": "off", "enable_graph_update": "off",
    "message_mode": "peer", "msg_linear": "off", "max_neighbors": 0,
}

CELL_CFG = {
    # max_events, num_pipeline_stages, stage_workers, 与 positioning/configs.py 的 CELLS 一致
    "B": {"max_events": 20000, "stages": 2, "workers": "1,1"},
    "D": {"max_events": 100000, "stages": 3, "workers": "1,1,1"},
}


def build_base_config(cell: str) -> dict:
    cc = CELL_CFG[cell]
    return {
        "dataset": "public_csv",
        "dataset_dir": "data/public",
        "local_data_path": "data/public/mooc.csv",
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "max_events": cc["max_events"],
        "num_users": 500,
        "num_items": 1000,
        "num_interactions": 3000,
        "feature_dim": 4,
        "lr": 1e-3,
        "neg_sample_size": 5,
        "k": 10,
        "selection_metric": "mrr",
        "device": "cuda",
        "seed": 42,
        "partition_size": 2000,
        "partition_strategy": "count",
        "partition_overlap_ratio": 0.0,
        "num_pipeline_stages": cc["stages"],
        "pipeline_worker_gpus": 1.0,
        "pipeline_worker_cpus": 1.0,
        "pipeline_stage_train_workers": cc["workers"],
        "pipeline_stage_eval_workers": cc["workers"],
        "stage_balance_strategy": "cost",
        "stage_balance_user_weight": 0.25,
        "stage_balance_item_weight": 0.25,
        "stage_balance_span_weight": 0.0,
        "pipeline_mode": "naive",
        "pipeline_trace": False,
        "pipeline_trace_log_path": "",
        "ray_address": "",
        "output_dir": "outputs/microcheck",
        "enable_efficiency_monitor": False,
        "efficiency_monitor_interval": 10,
        "data_parallel_workers": 3,
        "data_parallel_worker_gpus": 1.0,
        "gpu_list": "",
        "enable_auto_pipeline_config": False,
        "batch_training": False,
        "train_batch_size": 32,
        "batch_mode": "serial",
        "tgn_loss_mode": "all",
        "tgn_window_size": 10.0,
        "eval_frozen": False,
        "max_neighbors": 20,
    }


def state_fingerprint(state_dict) -> str:
    import hashlib
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def state_max_abs_diff(sd_a, sd_b):
    keys = sorted(set(sd_a.keys()) | set(sd_b.keys()))
    diffs = {}
    for k in keys:
        a = sd_a.get(k)
        b = sd_b.get(k)
        if a is None or b is None:
            diffs[k] = "missing"
        else:
            diffs[k] = float((a.detach().cpu() - b.detach().cpu()).abs().max())
    return diffs


def serial_half(trainer, train_data, val_data, graph_template, arch, trial_seed, epochs):
    """复刻 trainer._train_and_eval 的 serial 训练路径,保留终态模型。"""
    config = dict(trainer.base_config)
    config.update(arch)
    trainer._set_seed(trial_seed)
    model = build_model(config)
    device = torch.device(trainer.base_config.get("device", "cpu"))
    model = model.to(device)
    model_name = config.get("model", "temporal_event_gnn_jodie")
    graph_ctx = None if model_name == "jodie_rnn" else graph_template
    train_model_ce(
        model, train_data, num_epochs=epochs, lr=config.get("lr", 1e-3),
        graph_ctx=graph_ctx, seed=trial_seed, partitions=None,
        batch_training=False, batch_size=32, batch_mode="serial",
        tgn_loss_mode="all", tgn_window_size=10.0,
    )
    metrics = evaluate_ranking_metrics(
        model, val_data, k=config.get("k", 10), graph_ctx=graph_ctx,
        partitions=None, frozen=trainer.base_config.get("eval_frozen", False),
    )
    return float(metrics["mrr"]), model.state_dict()


def pipeline_half(trainer, partition_plan, val_data, arch, epochs, trial_ids):
    """复刻 evaluate_arch_pipeline 方案C:pipeline 训练 + 全数据评估,保留终态。"""
    executor = RayPipelineExecutor(trainer.base_config, partition_plan)
    payloads = executor.run_train_only([arch], num_train_epochs=epochs, trial_ids=trial_ids)
    payload = payloads[0]
    config = dict(trainer.base_config)
    config.update(payload.arch_config)
    model = build_model(config)
    model = model.to(torch.device(trainer.base_config.get("device", "cpu")))
    # payload 里的张量已在 worker 侧 .cpu(),load_state_dict 会自动拷到参数所在设备
    model.load_state_dict(payload.model_state_dict)
    if payload.runtime_state is not None and hasattr(model, "import_runtime_state"):
        model.import_runtime_state(payload.runtime_state)
    metrics = evaluate_ranking_metrics(
        model, val_data, k=config.get("k", 10),
        partitions=None, frozen=trainer.base_config.get("eval_frozen", False),
    )
    executor.shutdown()
    return float(metrics["mrr"]), model.state_dict(), payload.seed


def pool_half(trainer, partition_plan, arch, epochs, user_type_prefs, item_type, k):
    """复刻 trainer._search_pipeline_async 的持久池路径(Fix D 验证):
    池内训练(全局 epoch-major)+ 池内评估,与 serial 分数位级对比。"""
    executor = RayPipelineExecutor(trainer.base_config, partition_plan)
    eval_kwargs = {
        "eval_split": "val",
        "item_type": item_type,
        "user_type_prefs": user_type_prefs,
        "k": k,
    }
    executor.start_persistent_pool(eval_kwargs, num_train_epochs=epochs)
    tid = executor.submit_arch(arch)
    print(f"[Microcheck] pool submitted trial_id={tid} "
          f"(期望 seed={trainer.base_config.get('seed', 42)} + {tid})")
    results = []
    deadline = time.time() + 3600
    try:
        while time.time() < deadline and not results:
            results.extend(executor.poll_completed(timeout=1.0))
            time.sleep(0.5)
    finally:
        executor.shutdown_persistent_pool()
    if not results:
        raise RuntimeError("pool 1 小时内未完成 trial")
    r = results[0]
    return float(r["score"]), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["B", "D"], default="D")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--expected-serial-score", type=float, default=None,
                    help="已知的 serial 真值(D seed42 = 0.4760190275145158),用于互校复刻忠实性")
    ap.add_argument("--pool", action="store_true",
                    help="验证异步持久池路径(Fix D)与 serial 位级一致,替代同步 pipeline 路径")
    args = ap.parse_args()

    base_config = build_base_config(args.cell)
    trainer = GraphNASTrainer(base_config)
    train_data, val_data, test_data, user_type_prefs, item_type, graph_template, partition_plan = trainer._prepare_data()
    print(f"[Microcheck] cell={args.cell} max_events={base_config['max_events']} "
          f"train={len(train_data)} val={len(val_data)} test={len(test_data)} epochs={args.epochs}")

    arch = dict(ARCH_8896)

    # ── serial 复刻训练(一次,保留终态)──
    s_score, s_state = serial_half(trainer, train_data, val_data, graph_template, arch, args.seed, args.epochs)
    print(f"[Microcheck] serial score(mrr) = {s_score!r}")
    if args.expected_serial_score is not None:
        print(f"  与已知真值互校: {s_score == args.expected_serial_score} "
              f"(expected {args.expected_serial_score!r})")

    # ── pipeline 复刻训练(一次,保留终态;run_train_only 含 Fix B 全局种子)──
    if args.pool:
        p_score, pool_result = pool_half(
            trainer, partition_plan, arch, args.epochs,
            user_type_prefs, item_type, int(base_config["k"]),
        )

        print(f"\n===== 结果(异步池路径,Fix D 验证)=====")
        print(f"serial score = {s_score!r}")
        print(f"pool   score = {p_score!r}")
        print(f"pool trial_id = {pool_result['trial_id']} (期望 0 → seed=42 全局计数)")
        print(f"pool mrr={pool_result['mrr']!r} recall={pool_result['recall_at_k']!r}")
        print(f"\nscore 位级等价: {s_score == p_score}")
        if s_score != p_score:
            print(f"  Δ = {abs(s_score - p_score):.9e}")
    else:
        p_score, p_state, p_seed = pipeline_half(trainer, partition_plan, val_data, arch, args.epochs, [0])

        print(f"\n===== 结果 =====")
        print(f"serial   score = {s_score!r}")
        print(f"pipeline score = {p_score!r}")
        print(f"payload.seed = {p_seed} (期望 {args.seed}, Fix B 生效: {p_seed == args.seed})")
        print(f"\nscore 位级等价: {s_score == p_score}")
        if s_score != p_score:
            print(f"  Δ = {abs(s_score - p_score):.9e}")
        fp_s, fp_p = state_fingerprint(s_state), state_fingerprint(p_state)
        print(f"state_dict 指纹: serial={fp_s} pipeline={fp_p} 相同: {fp_s == fp_p}")
        if fp_s != fp_p:
            diffs = state_max_abs_diff(s_state, p_state)
            print("逐张量 max|Δ|:")
            for k, v in diffs.items():
                print(f"  {k}: {v}")
        else:
            print("state_dict 全部张量逐位一致 ✓")


if __name__ == "__main__":
    main()
