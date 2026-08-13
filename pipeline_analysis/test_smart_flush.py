#!/usr/bin/env python3
"""Pipeline Smart flush 路径最小测试：验证 inplace version 修复。

5 trials × 2 arch/step → 2 次批量更新 + 最后 flush 1 个剩余 buffer，
恰好复现全量运行失败时（12×4+2 最后 flush 2 个）的代码路径。
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    "num_pipeline_stages": 1, "device": "cuda",
    "pipeline_worker_gpus": 1.0,
    "pipeline_stage_train_workers": "3",
    "pipeline_stage_eval_workers": "3",
    "gpu_list": "0,1,2",
    "ray_address": "",
    "output_dir": "/tmp/test_smart_flush",
    "batch_mode": "serial",
    "eval_frozen": False,
    "data_parallel_workers": 3,
    "lr": 0.001, "neg_sample_size": 5, "k": 10,
    "selection_metric": "mrr",
    "pipeline_mode": "smart",
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

COARSE_TRIALS = 5          # 5 % 2 = 1 → 最后 flush 1 个剩余 buffer
COARSE_EPOCHS = 1
ARCHITECTURES_PER_STEP = 2

print("=" * 60)
print("Pipeline Smart flush 路径最小测试")
print(f"  {COARSE_TRIALS} trials × {COARSE_EPOCHS} epoch, "
      f"{ARCHITECTURES_PER_STEP} arch/step → 2 批更新 + flush 1 个")
print("=" * 60)

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

trainer = GraphNASTrainer(base_config=dict(full_config))
controller = RLGraphNASController(get_search_space("rnn_only"), seed=42, lr=1e-2)
t0 = time.time()
try:
    best, results = trainer.search_pipeline(
        controller=controller, coarse_trials=COARSE_TRIALS,
        architectures_per_step=ARCHITECTURES_PER_STEP, coarse_epochs=COARSE_EPOCHS,
        rerank_top_k=0, rerank_epochs=0, time_budget_sec=0,
    )
    elapsed = time.time() - t0
    print(f"✅ Smart flush 测试通过: {elapsed:.0f}s, best={best['score']:.4f}, "
          f"trials={len(results)}")
except Exception as e:
    elapsed = time.time() - t0
    import traceback
    print(f"❌ Smart flush 测试失败 ({elapsed:.0f}s): {e}")
    traceback.print_exc()
    sys.exit(1)
