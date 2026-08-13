#!/usr/bin/env python3
"""最小 4 策略搜索测试：Serial + DataParallel + PipelineNaive + PipelineSmart"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from jodie.data.public_dataset import load_public_dataset
from jodie.nas.controller import RLGraphNASController
from jodie.nas.search_space import get_search_space
from jodie.nas.trainer import GraphNASTrainer

BASE_CONFIG = {
    "dataset": "public_csv", "dataset_dir": "data/public",
    "local_data_path": "data/public/mooc.csv",
    "max_events": 20000, "train_ratio": 0.7, "val_ratio": 0.1,
    "feature_dim": 4, "seed": 42, "partition_size": 2000,
    "partition_strategy": "count", "partition_overlap_ratio": 0.0,
    "num_pipeline_stages": 3, "device": "cuda",
    "pipeline_worker_gpus": 1.0,
    "pipeline_stage_train_workers": "1,1,1",
    "pipeline_stage_eval_workers": "1,1,1",
    "gpu_list": "0,1,2",
    "ray_address": "",
    "output_dir": "/tmp/test_minimal_search",
    "batch_mode": "serial",
    "eval_frozen": False,
    "data_parallel_workers": 3,
    "lr": 0.001, "neg_sample_size": 5, "k": 10,
    "selection_metric": "mrr",
    "pipeline_mode": "naive",
    "stage_balance_strategy": "cost",
    "stage_balance_user_weight": 0.25,
    "stage_balance_item_weight": 0.25,
    "stage_balance_span_weight": 0.0,
    "pipeline_worker_cpus": 1.0,
    "enable_auto_pipeline_config": False,
    "batch_training": False,
    "train_batch_size": 32,
    "enable_efficiency_monitor": False,
    "efficiency_monitor_interval": 10,
    "data_parallel_worker_gpus": 1.0,
    "data_parallel_visible_gpus": "0",
    "data_parallel_sync_level": "micro_batch",
    "data_parallel_micro_batch_size": 200,
    "pipeline_trace": False,
    "pipeline_trace_log_path": "",
    "tgn_loss_mode": "all",
    "tgn_window_size": 10.0,
    "max_neighbors": 0,
}

SEARCH_SPACE_CONFIG = "rnn_only"
SEARCH_MODE = "rl"
SEED = 42
CONTROLLER_LR = 1e-2
ARCHITECTURES_PER_STEP = 2  # 每次采样 2 个（触发多次 controller 更新）

COARSE_TRIALS = 4
COARSE_EPOCHS = 1

print("=" * 60)
print("最小 4 策略搜索测试")
print(f"  各策略 {COARSE_TRIALS} trials × {COARSE_EPOCHS} epoch, 无 rerank")
print("=" * 60)

# 准备数据（所有策略共享 num_users/num_items）
interactions, num_users, num_items = load_public_dataset(
    dataset_name=BASE_CONFIG["dataset"],
    dataset_dir=BASE_CONFIG.get("dataset_dir", "data/public"),
    feature_dim=BASE_CONFIG["feature_dim"],
    max_events=BASE_CONFIG.get("max_events", 0),
    local_data_path=BASE_CONFIG.get("local_data_path", ""),
)
full_config = dict(BASE_CONFIG)
full_config["num_users"] = num_users
full_config["num_items"] = num_items

results_summary = {}

# ── 1. Serial ──
print("\n[1/4] Serial 搜索...")
trainer_s = GraphNASTrainer(base_config=dict(full_config))
controller_s = RLGraphNASController(get_search_space(SEARCH_SPACE_CONFIG), seed=SEED, lr=CONTROLLER_LR)
t0 = time.time()
best_s, results_s = trainer_s.search(
    controller=controller_s, coarse_trials=COARSE_TRIALS, coarse_epochs=COARSE_EPOCHS,
    rerank_top_k=0, rerank_epochs=0, time_budget_sec=0,
)
elapsed_s = time.time() - t0
results_summary["serial"] = {"best": best_s["score"], "trials": len(results_s), "time": elapsed_s}
print(f"  ✅ Serial: {elapsed_s:.0f}s, best={best_s['score']:.4f}, trials={len(results_s)}")

# ── 2. Data Parallel ──
print("\n[2/4] Data Parallel 搜索...")
dp_config = dict(full_config)
dp_config["data_parallel_workers"] = 3  # 3 workers, 3 GPUs
dp_config["data_parallel_visible_gpus"] = "0,1,2"
dp_config["output_dir"] = "/tmp/test_minimal_search/dp"
trainer_dp = GraphNASTrainer(base_config=dp_config)
controller_dp = RLGraphNASController(get_search_space(SEARCH_SPACE_CONFIG), seed=SEED, lr=CONTROLLER_LR)
t0 = time.time()
try:
    best_dp, results_dp = trainer_dp.search_data_parallel(
        controller=controller_dp, coarse_trials=COARSE_TRIALS, coarse_epochs=COARSE_EPOCHS,
        num_workers=3, rerank_top_k=0, rerank_epochs=0, time_budget_sec=0,
    )
    elapsed_dp = time.time() - t0
    results_summary["data_parallel"] = {"best": best_dp["score"], "trials": len(results_dp), "time": elapsed_dp}
    print(f"  ✅ Data Parallel: {elapsed_dp:.0f}s, best={best_dp['score']:.4f}, trials={len(results_dp)}")
except Exception as e:
    elapsed_dp = time.time() - t0
    import traceback
    print(f"  ❌ Data Parallel 失败 ({elapsed_dp:.0f}s): {e}")
    traceback.print_exc()
    results_summary["data_parallel"] = {"best": None, "trials": 0, "time": elapsed_dp, "error": str(e)}

# ── 3. Pipeline Naive ──
print("\n[3/4] Pipeline Naive 搜索 (3 stages × 1,1,1)...")
naive_config = dict(full_config)
naive_config["num_pipeline_stages"] = 3
naive_config["pipeline_stage_train_workers"] = "1,1,1"
naive_config["pipeline_stage_eval_workers"] = "1,1,1"
naive_config["pipeline_mode"] = "naive"
naive_config["output_dir"] = "/tmp/test_minimal_search/pnaive"
trainer_pn = GraphNASTrainer(base_config=naive_config)
controller_pn = RLGraphNASController(get_search_space(SEARCH_SPACE_CONFIG), seed=SEED, lr=CONTROLLER_LR)
t0 = time.time()
try:
    best_pn, results_pn = trainer_pn.search_pipeline(
        controller=controller_pn, coarse_trials=COARSE_TRIALS,
        architectures_per_step=ARCHITECTURES_PER_STEP, coarse_epochs=COARSE_EPOCHS,
        rerank_top_k=0, rerank_epochs=0, time_budget_sec=0,
    )
    elapsed_pn = time.time() - t0
    results_summary["pipeline_naive"] = {"best": best_pn["score"], "trials": len(results_pn), "time": elapsed_pn}
    print(f"  ✅ Pipeline Naive: {elapsed_pn:.0f}s, best={best_pn['score']:.4f}, trials={len(results_pn)}")
except Exception as e:
    elapsed_pn = time.time() - t0
    import traceback
    print(f"  ❌ Pipeline Naive 失败 ({elapsed_pn:.0f}s): {e}")
    traceback.print_exc()
    results_summary["pipeline_naive"] = {"best": None, "trials": 0, "time": elapsed_pn, "error": str(e)}

# ── 4. Pipeline Smart (1 stage × 3 workers) ──
print("\n[4/4] Pipeline Smart 搜索 (1 stage × 3 workers)...")
smart_config = dict(full_config)
smart_config["num_pipeline_stages"] = 1
smart_config["pipeline_stage_train_workers"] = "3"
smart_config["pipeline_stage_eval_workers"] = "3"
smart_config["pipeline_mode"] = "smart"
smart_config["output_dir"] = "/tmp/test_minimal_search/psmart"
trainer_ps = GraphNASTrainer(base_config=smart_config)
controller_ps = RLGraphNASController(get_search_space(SEARCH_SPACE_CONFIG), seed=SEED, lr=CONTROLLER_LR)
t0 = time.time()
try:
    best_ps, results_ps = trainer_ps.search_pipeline(
        controller=controller_ps, coarse_trials=COARSE_TRIALS,
        architectures_per_step=ARCHITECTURES_PER_STEP, coarse_epochs=COARSE_EPOCHS,
        rerank_top_k=0, rerank_epochs=0, time_budget_sec=0,
    )
    elapsed_ps = time.time() - t0
    results_summary["pipeline_smart"] = {"best": best_ps["score"], "trials": len(results_ps), "time": elapsed_ps}
    print(f"  ✅ Pipeline Smart: {elapsed_ps:.0f}s, best={best_ps['score']:.4f}, trials={len(results_ps)}")
except Exception as e:
    elapsed_ps = time.time() - t0
    import traceback
    print(f"  ❌ Pipeline Smart 失败 ({elapsed_ps:.0f}s): {e}")
    traceback.print_exc()
    results_summary["pipeline_smart"] = {"best": None, "trials": 0, "time": elapsed_ps, "error": str(e)}

# ── Summary ──
print("\n" + "=" * 60)
print("最小测试结果汇总")
print("=" * 60)
for name, info in results_summary.items():
    status = "❌ FAIL" if info.get("error") else f"✅ best={info['best']:.4f}"
    print(f"  {name:20s}: {status:30s}  {info['time']:.0f}s, {info['trials']} trials")
print("\n全部策略最小测试完成" if all(not v.get("error") for v in results_summary.values()) else "\n有策略失败!")
