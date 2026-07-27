"""
=============================================================================
 JODIE GraphNAS 快速冒烟测试 —— 验证四种策略均可正常运行
=============================================================================

 使用合成数据 + 极小配置，确保每个策略的 Ray 初始化、worker 创建、
 训练和评估流程都能完整走通。
"""

import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from jodie.nas.controller import RLGraphNASController, RandomGraphNASController
from jodie.nas.search_space import get_search_space
from jodie.nas.trainer import GraphNASTrainer


# =============================================================================
#  极简配置 —— 只验证流程不追求精度
# =============================================================================

SEARCH_SPACE = "rnn_only"
SEARCH_MODE = "random"              # random 比 RL 快，不需要控制器训练
COARSE_TRIALS = 4                   # 最少 trial 数
COARSE_EPOCHS = 1                   # 1 轮即可
RERANK_TOP_K = 0                    # 跳过重排序
RERANK_EPOCHS = 1
CONTROLLER_LR = 1e-2
TIME_BUDGET_SEC = 0.0

# 合成数据 —— 极小规模
DATASET = "synthetic"
LOCAL_DATA_PATH = ""
MAX_EVENTS = 500
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1

NUM_USERS = 100
NUM_ITEMS = 200
NUM_INTERACTIONS = 500

# 训练
FEATURE_DIM = 8
LR = 1e-3
NEG_SAMPLE_SIZE = 5
K = 10
SELECTION_METRIC = "mrr"
BATCH_MODE = "serial"               # 最简单模式
TRAIN_BATCH_SIZE = 32
TGN_WINDOW_SIZE = 10.0
TGN_LOSS_MODE = "all"
EVAL_FROZEN = False

# 分区 —— 小分区
PARTITION_SIZE = 100
PARTITION_OVERLAP = 0.0

# Pipeline 配置 —— 单阶段
NUM_PIPELINE_STAGES = 1
ARCHITECTURES_PER_STEP = 2
PIPELINE_STAGE_TRAIN_WORKERS = "1"
PIPELINE_STAGE_EVAL_WORKERS = "1"
SMART_ENABLE_AUTO_PIPELINE_CONFIG = False   # 关掉自动配置，用单阶段
SMART_PIPELINE_STAGE_TRAIN_WORKERS = "1"
STAGE_BALANCE_STRATEGY = "cost"

# GPU —— 单卡
GPU_LIST = "0"
PIPELINE_WORKER_GPUS = 1.0
DATA_PARALLEL_WORKERS = 1
DATA_PARALLEL_WORKER_GPUS = 1.0

# 其他
DEVICE = "auto"
SEED = 42
RESULTS_DIR = "results"
EVAL_SEEDS = ""
FAMILY_BALANCED_RERANK = False
FAMILY_BALANCE_PER_MODEL = 1

ENABLE_STRATEGIES = [
    "serial",
    "data_parallel",
    "pipeline_naive",
    "pipeline_smart",
]

# =============================================================================
#  以下代码与 run_all.py 保持一致
# =============================================================================


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_header(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"  {_timestamp()}")
    print(f"{'=' * width}\n", flush=True)


def build_base_config(strategy: str, output_dir: str, pipeline_mode: Optional[str] = None) -> Dict:
    import torch as _torch
    device = DEVICE if DEVICE != "auto" else ("cuda" if _torch.cuda.is_available() else "cpu")

    pipeline_trace_log_path = ""
    if strategy in ("pipeline_naive", "pipeline_smart"):
        pipeline_trace_log_path = os.path.join(output_dir, "pipeline_trace.log")

    if strategy == "pipeline_smart":
        enable_auto = SMART_ENABLE_AUTO_PIPELINE_CONFIG
        num_stages = NUM_PIPELINE_STAGES
        train_workers = SMART_PIPELINE_STAGE_TRAIN_WORKERS
        eval_workers = ""
        pipeline_md = pipeline_mode if pipeline_mode else "smart"
    elif strategy == "pipeline_naive":
        enable_auto = False
        num_stages = NUM_PIPELINE_STAGES
        train_workers = PIPELINE_STAGE_TRAIN_WORKERS
        eval_workers = PIPELINE_STAGE_EVAL_WORKERS
        pipeline_md = "naive"
    else:
        enable_auto = False
        num_stages = NUM_PIPELINE_STAGES
        train_workers = ""
        eval_workers = ""
        pipeline_md = "naive"

    config = {
        "dataset": DATASET,
        "dataset_dir": "data/public",
        "local_data_path": LOCAL_DATA_PATH,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "max_events": MAX_EVENTS,
        "num_users": NUM_USERS,
        "num_items": NUM_ITEMS,
        "num_interactions": NUM_INTERACTIONS,
        "feature_dim": FEATURE_DIM,
        "lr": LR,
        "neg_sample_size": NEG_SAMPLE_SIZE,
        "k": K,
        "selection_metric": SELECTION_METRIC,
        "device": device,
        "seed": SEED,
        "partition_size": PARTITION_SIZE,
        "partition_strategy": "count",
        "partition_overlap_ratio": PARTITION_OVERLAP,
        "num_pipeline_stages": num_stages,
        "pipeline_worker_gpus": PIPELINE_WORKER_GPUS,
        "pipeline_worker_cpus": 1.0,
        "pipeline_stage_train_workers": train_workers,
        "pipeline_stage_eval_workers": eval_workers,
        "stage_balance_strategy": STAGE_BALANCE_STRATEGY,
        "stage_balance_user_weight": 0.25,
        "stage_balance_item_weight": 0.25,
        "stage_balance_span_weight": 0.0,
        "pipeline_mode": pipeline_md,
        "pipeline_trace": False,
        "pipeline_trace_log_path": pipeline_trace_log_path,
        "ray_address": "",
        "output_dir": output_dir,
        "enable_efficiency_monitor": False,
        "efficiency_monitor_interval": 10,
        "data_parallel_workers": DATA_PARALLEL_WORKERS,
        "data_parallel_worker_gpus": DATA_PARALLEL_WORKER_GPUS,
        "data_parallel_visible_gpus": GPU_LIST,
        "data_parallel_sync_level": "micro_batch",
        "data_parallel_micro_batch_size": max(1, MAX_EVENTS // 100) if MAX_EVENTS > 0 else 32,
        "gpu_list": GPU_LIST,
        "enable_auto_pipeline_config": enable_auto,
        "batch_training": False,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "batch_mode": BATCH_MODE,
        "tgn_loss_mode": TGN_LOSS_MODE,
        "tgn_window_size": TGN_WINDOW_SIZE,
        "eval_frozen": EVAL_FROZEN,
        "max_neighbors": 20,
    }
    return config


def run_serial(output_dir: str) -> Tuple[Dict, List[Dict], float]:
    print_header("策略 1/4: Serial (串行搜索)")
    base_config = build_base_config("serial", output_dir)
    trainer = GraphNASTrainer(base_config)
    search_space = get_search_space(SEARCH_SPACE)
    controller = RandomGraphNASController(search_space, seed=SEED)

    t0 = time.time()
    best, results = trainer.search(
        controller=controller,
        coarse_trials=COARSE_TRIALS,
        coarse_epochs=COARSE_EPOCHS,
        rerank_top_k=RERANK_TOP_K,
        rerank_epochs=RERANK_EPOCHS,
        eval_seeds=None,
        family_balanced_rerank=FAMILY_BALANCED_RERANK,
        family_balance_per_model=FAMILY_BALANCE_PER_MODEL,
        time_budget_sec=TIME_BUDGET_SEC,
    )
    elapsed = time.time() - t0
    save_strategy_results(output_dir, "serial", best, results)
    return best, results, elapsed


def run_data_parallel(output_dir: str) -> Tuple[Dict, List[Dict], float]:
    print_header("策略 2/4: Data Parallel (数据并行搜索)")
    base_config = build_base_config("data_parallel", output_dir)
    trainer = GraphNASTrainer(base_config)
    search_space = get_search_space(SEARCH_SPACE)
    controller = RandomGraphNASController(search_space, seed=SEED)

    t0 = time.time()
    best, results = trainer.search_data_parallel(
        controller=controller,
        coarse_trials=COARSE_TRIALS,
        coarse_epochs=COARSE_EPOCHS,
        num_workers=DATA_PARALLEL_WORKERS,
        rerank_top_k=RERANK_TOP_K,
        rerank_epochs=RERANK_EPOCHS,
        time_budget_sec=TIME_BUDGET_SEC,
    )
    elapsed = time.time() - t0
    save_strategy_results(output_dir, "data_parallel", best, results)
    return best, results, elapsed


def run_pipeline_naive(output_dir: str) -> Tuple[Dict, List[Dict], float]:
    print_header("策略 3/4: Pipeline Naive (流水线批次同步)")
    base_config = build_base_config("pipeline_naive", output_dir, pipeline_mode="naive")
    trainer = GraphNASTrainer(base_config)
    search_space = get_search_space(SEARCH_SPACE)
    controller = RandomGraphNASController(search_space, seed=SEED)

    t0 = time.time()
    best, results = trainer.search_pipeline(
        controller=controller,
        coarse_trials=COARSE_TRIALS,
        architectures_per_step=ARCHITECTURES_PER_STEP,
        coarse_epochs=COARSE_EPOCHS,
        rerank_top_k=RERANK_TOP_K,
        rerank_epochs=RERANK_EPOCHS,
        family_balanced_rerank=FAMILY_BALANCED_RERANK,
        family_balance_per_model=FAMILY_BALANCE_PER_MODEL,
        time_budget_sec=TIME_BUDGET_SEC,
    )
    elapsed = time.time() - t0
    save_strategy_results(output_dir, "pipeline_naive", best, results)
    return best, results, elapsed


def run_pipeline_smart(output_dir: str) -> Tuple[Dict, List[Dict], float]:
    print_header("策略 4/4: Pipeline Smart (流水线异步持久化池)")
    base_config = build_base_config("pipeline_smart", output_dir, pipeline_mode="smart")
    trainer = GraphNASTrainer(base_config)
    search_space = get_search_space(SEARCH_SPACE)
    controller = RandomGraphNASController(search_space, seed=SEED)

    t0 = time.time()
    best, results = trainer.search_pipeline(
        controller=controller,
        coarse_trials=COARSE_TRIALS,
        architectures_per_step=ARCHITECTURES_PER_STEP,
        coarse_epochs=COARSE_EPOCHS,
        rerank_top_k=RERANK_TOP_K,
        rerank_epochs=RERANK_EPOCHS,
        family_balanced_rerank=FAMILY_BALANCED_RERANK,
        family_balance_per_model=FAMILY_BALANCE_PER_MODEL,
        time_budget_sec=TIME_BUDGET_SEC,
    )
    elapsed = time.time() - t0
    save_strategy_results(output_dir, "pipeline_smart", best, results)
    return best, results, elapsed


def save_strategy_results(output_dir: str, strategy: str, best: Dict, results: List[Dict]) -> None:
    strat_dir = os.path.join(output_dir, strategy)
    os.makedirs(strat_dir, exist_ok=True)

    best_path = os.path.join(strat_dir, "best_arch.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(_serialize(best), f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(strat_dir, "leaderboard.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "phase", "eval_split", "score", "val_score", "test_score",
            "mrr", "recall_at_k", "params", "time_sec", "model", "config_json"
        ])
        sorted_results = sorted(
            results,
            key=lambda x: (x.get("score", 0), -x.get("params", 0), -x.get("time_sec", 0)),
            reverse=True,
        )
        for idx, row in enumerate(sorted_results, start=1):
            writer.writerow([
                idx,
                row.get("phase", "na"),
                row.get("eval_split", "na"),
                row.get("score"),
                row.get("val_score"),
                row.get("test_score"),
                row.get("mrr"),
                row.get("recall_at_k"),
                row.get("params"),
                row.get("time_sec"),
                row.get("config", {}).get("model", "unknown"),
                json.dumps(row.get("config", {}), ensure_ascii=False),
            ])

    summary_path = os.path.join(strat_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"策略: {strategy}\n")
        f.write(f"搜索空间: {SEARCH_SPACE}\n")
        f.write(f"搜索模式: {SEARCH_MODE}\n")
        f.write(f"粗搜索数: {COARSE_TRIALS}\n")
        f.write(f"粗搜索epochs: {COARSE_EPOCHS}\n")
        f.write(f"\n--- 最佳架构 ---\n")
        f.write(f"val_score: {best.get('selected_val_score', best.get('score', 'N/A'))}\n")
        f.write(f"test_score: {best.get('test_score', 'N/A')}\n")
        f.write(f"模型: {best.get('config', {}).get('model', 'N/A')}\n")
        f.write(f"\n架构配置:\n{json.dumps(best.get('config', {}), ensure_ascii=False, indent=2)}\n")

    print(f"  [{strategy}] 结果已保存到: {strat_dir}", flush=True)


def _serialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


def ensure_ray_clean():
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            print("[Init] Ray shutdown from previous run", flush=True)
            time.sleep(2)
    except Exception:
        pass


def print_final_summary(strategy_results: Dict[str, Tuple[Dict, float]]) -> None:
    print_header("最终结果摘要")

    labels = {
        "serial": "Serial    ",
        "data_parallel": "DataPara  ",
        "pipeline_naive": "PipeNaive ",
        "pipeline_smart": "PipeSmart ",
    }

    print(f"  {'策略':<12} {'耗时':>8} {'val_score':>10} {'test_score':>10} {'test_mrr':>10} {'模型':>15}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*15}")

    for strategy in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]:
        if strategy not in strategy_results:
            continue
        best, elapsed = strategy_results[strategy]
        label = labels.get(strategy, strategy)
        test_score = best.get('test_score')
        test_mrr = best.get('test_mrr')
        print(f"  {label:<12} {elapsed:>7.1f}s {best.get('selected_val_score', best.get('score', 0)):>10.4f} "
              f"{test_score if test_score is not None else 'N/A':>10} "
              f"{test_mrr if test_mrr is not None else 'N/A':>10} "
              f"{str(best.get('config', {}).get('model', '?')):>15}")
    print()


def main():
    print_header("JODIE GraphNAS 快速冒烟测试")

    print("  [预检] 正在验证运行环境...")
    checks_ok = True

    try:
        import torch as _t
        cuda_avail = _t.cuda.is_available()
        cuda_count = _t.cuda.device_count() if cuda_avail else 0
        print(f"  [预检] ✓ CUDA 可用: {cuda_avail}, 可见GPU数: {cuda_count}")
        if cuda_count < 1:
            print(f"  [预检] ⚠ 无 GPU，使用 CPU")
    except Exception:
        print(f"  [预检] ⚠ 无法检测 GPU, 将使用 CPU")
        cuda_count = 0

    try:
        import ray
        print(f"  [预检] ✓ Ray 已安装 (version={ray.__version__})")
    except ImportError:
        print(f"  [预检] ✗ Ray 未安装!")
        checks_ok = False

    if not checks_ok:
        print("\n  [预检] 存在致命错误, 请修复后重新运行。")
        sys.exit(1)
    print()

    print("  当前配置:")
    print(f"    搜索空间: {SEARCH_SPACE}  搜索模式: {SEARCH_MODE}")
    print(f"    数据集: {DATASET} ({NUM_USERS}用户 × {NUM_ITEMS}物品 × {NUM_INTERACTIONS}交互)")
    print(f"    粗搜索: {COARSE_TRIALS} trials × {COARSE_EPOCHS} epochs  重排序: top {RERANK_TOP_K}")
    print(f"    GPU: {GPU_LIST}  Pipeline stages: {NUM_PIPELINE_STAGES}  DP workers: {DATA_PARALLEL_WORKERS}")
    print(f"    启用策略: {ENABLE_STRATEGIES}")
    print()

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"quick_test_{run_tag}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"  结果目录: {run_dir}\n")

    config_path = os.path.join(run_dir, "config.json")
    config_snapshot = {}
    for k, v in globals().items():
        if not k.isupper() or k.startswith("_"):
            continue
        if isinstance(v, (str, int, float, bool, list, tuple, type(None))):
            config_snapshot[k] = v
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(_serialize(config_snapshot), f, ensure_ascii=False, indent=2)

    strategy_runners = {
        "serial": run_serial,
        "data_parallel": run_data_parallel,
        "pipeline_naive": run_pipeline_naive,
        "pipeline_smart": run_pipeline_smart,
    }

    strategy_results: Dict[str, Tuple[Dict, float]] = {}
    failed_strategies: List[str] = []

    total_start = time.time()

    for i, strategy in enumerate(ENABLE_STRATEGIES, 1):
        if strategy not in strategy_runners:
            print(f"[警告] 未知策略 '{strategy}'，跳过", flush=True)
            continue

        strategy_output_dir = os.path.join(run_dir, strategy)
        os.makedirs(strategy_output_dir, exist_ok=True)

        try:
            ensure_ray_clean()
            best, results, elapsed = strategy_runners[strategy](strategy_output_dir)
            strategy_results[strategy] = (best, elapsed)
            print(f"\n  [{strategy}] ✓ 完成 — 耗时 {elapsed:.1f}s, "
                  f"val_score={best.get('selected_val_score', best.get('score', 0)):.4f}\n", flush=True)
        except Exception as e:
            print(f"\n  [{strategy}] ✗ 失败: {e}", flush=True)
            traceback.print_exc()
            failed_strategies.append(strategy)
            continue

        if strategy in ("pipeline_naive", "pipeline_smart") and i < len(ENABLE_STRATEGIES):
            print(f"  [Info] 等待 Ray 资源释放...", flush=True)
            time.sleep(3)

    total_elapsed = time.time() - total_start
    ensure_ray_clean()

    if strategy_results:
        print_final_summary(strategy_results)

    print(f"\n  总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  成功策略: {list(strategy_results.keys())}")
    if failed_strategies:
        print(f"  失败策略: {failed_strategies}")
    print(f"  结果目录: {run_dir}")
    print(f"\n{'=' * 70}")
    print(f"  全部完成 — {_timestamp()}")
    print(f"{'=' * 70}\n")

    # 退出码: 全部成功=0, 部分失败=1
    sys.exit(0 if not failed_strategies else 1)


if __name__ == "__main__":
    main()
