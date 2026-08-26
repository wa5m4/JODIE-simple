"""
=============================================================================
 JODIE GraphNAS 全策略对比执行入口
=============================================================================

 本文件一次性运行四种 NAS 搜索策略并生成对比报告：
   - serial         : 单机串行搜索（基线）
   - data_parallel  : Ray 数据并行搜索（架构内并行）
   - pipeline_naive : Ray 流水线并行（批次同步模式）
   - pipeline_smart : Ray 流水线并行（异步持久化池 + 离线 RL）

 每种策略独立完成：粗搜索 → 重排序 → final test（统一使用 serial 逻辑）。
 最终以 final_test_score 作为架构搜索结果的最终分数。

=============================================================================
 可修改参数清单
=============================================================================

---- 搜索控制 ----
SEARCH_SPACE        : 搜索空间预设
    选项: "small" | "paper_compare" | "rnn_only" | "mixed"
    说明: small=完整18维GNN搜索, paper_compare=受限JODIE对比,
          rnn_only=仅RNN超参(6维), mixed=混合模型族搜索

SEARCH_MODE         : 控制器策略
    选项: "random" | "rl"
    说明: random=均匀随机采样, rl=REINFORCE策略梯度引导搜索

COARSE_TRIALS       : 粗搜索阶段评估的架构数量
    选项: 整数 (如 6, 12, 32)
    说明: 每个架构用 COARSE_EPOCHS 轮训练后评估

COARSE_EPOCHS       : 粗搜索阶段每个架构的训练轮数
    选项: 整数 (如 1, 2, 4)
    说明: 轮数越多评估越准确但越慢

RERANK_TOP_K        : 粗搜索结束后重排序的前K个架构 (0=跳过)
    选项: 整数 (如 0, 4, 8)

RERANK_EPOCHS       : 重排序阶段的训练轮数 (通常 > COARSE_EPOCHS)
    选项: 整数 (如 2, 4, 8)

CONTROLLER_LR       : RL控制器的学习率 (仅 SEARCH_MODE="rl" 时有效)
    选项: 浮点数 (如 1e-2, 5e-3)

TIME_BUDGET_SEC     : 搜索总时间预算秒数 (0=无限制)
    选项: 浮点数 (如 0.0, 3600.0)

---- 数据 ----
DATASET             : 数据集名称
    选项: "synthetic" | "wikipedia" | "reddit" | "public_csv"
    说明: synthetic=合成数据(快速验证), public_csv=本地CSV文件

LOCAL_DATA_PATH     : 本地CSV文件路径 (仅 DATASET="public_csv" 时有效)
    选项: 文件路径字符串 (如 "data/public/mooc.csv")

MAX_EVENTS          : 最大使用事件数 (0=全部)
    选项: 整数 (如 0, 10000, 50000)

TRAIN_RATIO         : 训练集占比
    选项: 浮点数 (如 0.7)

VAL_RATIO           : 验证集占比
    选项: 浮点数 (如 0.1)

---- 合成数据参数 (仅 DATASET="synthetic" 时有效) ----
NUM_USERS           : 合成用户数
    选项: 整数 (如 500)

NUM_ITEMS           : 合成物品数
    选项: 整数 (如 1000)

NUM_INTERACTIONS    : 合成交互数
    选项: 整数 (如 3000)

---- 训练 ----
FEATURE_DIM         : 输入特征维度
    选项: 整数 (如 8, 32, 64)

LR                  : 模型训练学习率
    选项: 浮点数 (如 1e-3)

NEG_SAMPLE_SIZE     : BPR负采样数量 (仅合成数据)
    选项: 整数 (如 5)

K                   : Recall@K 中的K值
    选项: 整数 (如 10)

SELECTION_METRIC    : 架构选择指标
    选项: "mrr" | "recall_at_k"
    说明: 决定搜索结果按哪个指标排序

BATCH_MODE          : 训练批量模式
    选项: "serial" | "tbatch" | "tgn" | "stale_batch"
    说明: serial=逐个交互, tbatch=贪心唯一节点批处理, tgn=时间窗口批处理,
          stale_batch=朴素分批(连续切块、批内读批前状态,破坏 RAW)

TRAIN_BATCH_SIZE    : 批量训练大小 (仅 tbatch/tgn/stale_batch)
    选项: 整数 (如 32)

TGN_WINDOW_SIZE     : TGN时间窗口时长 (仅 BATCH_MODE="tgn")
    选项: 浮点数 (如 10.0)

TGN_LOSS_MODE       : TGN损失计算模式
    选项: "all" | "last"

EVAL_FROZEN         : 是否冻结评估 (True=离线, False=在线)
    选项: True | False

---- 分区 ----
PARTITION_SIZE      : 分区大小 (0=自动)
    选项: 整数 (如 0, 100, 500)

PARTITION_OVERLAP   : 分区重叠比例 [0, 1)
    选项: 浮点数 (如 0.0, 0.1)

---- 流水线 (用于 pipeline_naive 和 pipeline_smart) ----
NUM_PIPELINE_STAGES     : 流水线阶段数
    选项: 整数 (如 2, 4)

ARCHITECTURES_PER_STEP  : 每个流水线批次的架构数
    选项: 整数 (如 2, 4)

PIPELINE_STAGE_TRAIN_WORKERS : 各阶段训练worker数 (逗号分隔, 空=自动)
    选项: 字符串 (如 "" 或 "2,1" 或 "2,2,1,1")

PIPELINE_STAGE_EVAL_WORKERS  : 各阶段评估worker数
    选项: 字符串 (如 "" 或 "1,1")

STAGE_BALANCE_STRATEGY : 分区到阶段的均衡策略
    选项: "cost" | "count"
    说明: cost=DP最小化阶段成本方差, count=均匀按数量分

ENABLE_AUTO_PIPELINE_CONFIG : 启用自动流水线配置
    选项: True | False

---- GPU ----
GPU_LIST            : 可用GPU ID列表
    选项: 逗号分隔字符串 (如 "0" 或 "0,1" 或 "0,1,2")

PIPELINE_WORKER_GPUS    : 每个流水线worker的GPU数 (0=自动)
    选项: 浮点数 (如 0.0, 0.5, 1.0)

DATA_PARALLEL_WORKERS   : 数据并行worker数
    选项: 整数 (如 2, 3, 4)

DATA_PARALLEL_WORKER_GPUS : 每个DP worker的GPU数
    选项: 浮点数 (如 0.5, 1.0)

---- 其他 ----
DEVICE              : 计算设备
    选项: "auto" | "cpu" | "cuda"

SEED                : 全局随机种子
    选项: 整数 (如 42)

RESULTS_DIR         : 结果根目录 (每次运行在此下创建时间戳子目录)
    选项: 路径字符串 (如 "results")

EVAL_SEEDS          : 多种子评估列表 (空=单种子)
    选项: 逗号分隔整数 (如 "" 或 "42,123,456")

FAMILY_BALANCED_RERANK : 重排序时确保模型族多样性
    选项: True | False

FAMILY_BALANCE_PER_MODEL : 重排序每个模型族最少候选数
    选项: 整数 (如 1)

ENABLE_STRATEGIES   : 启用的策略列表 (可注释掉不需要的)
    说明: 至少保留一个策略

=============================================================================
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
#  可修改参数 —— 在此处修改配置
# =============================================================================

# ---- 搜索控制 ----
SEARCH_SPACE = "rnn_only"       # "small" | "paper_compare" | "rnn_only" | "mixed"
SEARCH_MODE = "rl"              # "random" | "rl"
COARSE_TRIALS = 50              # 粗搜索架构数 (20000事件/3GPU 适合50)
COARSE_EPOCHS = 2               # 粗搜索每架构训练轮数
RERANK_TOP_K = 8                # 重排序前K个 (50 trials 取前8)
RERANK_EPOCHS = 5               # 重排序训练轮数 (比粗搜索多)
CONTROLLER_LR = 1e-2            # RL 控制器学习率
TIME_BUDGET_SEC = 0.0           # 搜索时间预算 (0=不限制, 全量运行)

# ---- 数据 ----
DATASET = "public_csv"           # "synthetic" | "wikipedia" | "reddit" | "public_csv"
LOCAL_DATA_PATH = "data/public/mooc.csv"  # 仅 public_csv 时有效
MAX_EVENTS = 20000               # 使用前 20000 条交互事件
TRAIN_RATIO = 0.7                # 训练集占比
VAL_RATIO = 0.1                  # 验证集占比 (test=0.2)

# ---- 合成数据 ----
NUM_USERS = 500
NUM_ITEMS = 1000
NUM_INTERACTIONS = 3000

# ---- 训练 ----
FEATURE_DIM = 4                  # MOOC 数据集实际特征维度为4 (已确认CSV header)
LR = 1e-3
NEG_SAMPLE_SIZE = 5
K = 10
SELECTION_METRIC = "mrr"        # "mrr" | "recall_at_k"
BATCH_MODE = "serial"           # "serial" | "tbatch" | "tgn" | "stale_batch"  ← smart-async 重跑:保持与基线一致的保真训练协议,唯一变量=搜索策略
TRAIN_BATCH_SIZE = 32
TGN_WINDOW_SIZE = 10.0
TGN_LOSS_MODE = "all"           # "all" | "last"
EVAL_FROZEN = False

# ---- 分区 (20000事件: train≈14000, 每个分区约2000个交互≈7个分区) ----
PARTITION_SIZE = 2000             # ⬆ 方案A: 500→2000，减少训练偏差 (0=自动/不分)
PARTITION_OVERLAP = 0.0          # 分区重叠比例 [0, 1)

# ---- 流水线 ----
# Naive:  固定多 stage, 验证"多 stage 流水线能否加速"
# Smart:  自动选择最优分配(通常1 stage)+异步池+离线RL, 验证"智能调度优势"
NUM_PIPELINE_STAGES = 3                              # Naive 用的阶段数 (3 GPU → 3 stages)
ARCHITECTURES_PER_STEP = 4                            # 每批次架构数
PIPELINE_STAGE_TRAIN_WORKERS = "1,1,1"                # Naive: 每 stage 1 worker
PIPELINE_STAGE_EVAL_WORKERS = "1,1,1"                 # Naive: 每 stage 1 eval worker
SMART_ENABLE_AUTO_PIPELINE_CONFIG = False             # Smart: 关闭自动，手动指定 1stage×3worker
SMART_PIPELINE_STAGE_TRAIN_WORKERS = "3"              # Smart: 1 stage × 3 workers
SMART_NUM_PIPELINE_STAGES = 1                         # Smart: 1 stage
STAGE_BALANCE_STRATEGY = "cost"                       # DP 最小化阶段间成本方差

# ---- GPU (3卡: 0,1,2) ----
GPU_LIST = "0,1,2"                                    # 3 GPU 全部使用
PIPELINE_WORKER_GPUS = 1.0                            # 每个流水线 worker 独占 1 GPU
DATA_PARALLEL_WORKERS = 3                             # 每 GPU 一个 DP worker
DATA_PARALLEL_WORKER_GPUS = 1.0                       # 每个 DP worker 独占 1 GPU

# ---- 其他 ----
DEVICE = "auto"
SEED = 42
RESULTS_DIR = "results"
EVAL_SEEDS = ""                     # 多种子评估, 如 "42,123,456"
FAMILY_BALANCED_RERANK = False
FAMILY_BALANCE_PER_MODEL = 1

# ---- 启用的策略 (可注释不需要的) ----
ENABLE_STRATEGIES = [
    "pipeline_smart",      # smart-async 重跑:异步生成+训练机制在修复全开协议下的复证(唯一变量=搜索策略)
]

# =============================================================================
#  执行逻辑 —— 一般不需要修改以下代码
# =============================================================================


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _short_ts() -> str:
    return datetime.now().strftime("%H%M%S")


def print_header(title: str) -> None:
    """打印醒目标题。"""
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"  {_timestamp()}")
    print(f"{'=' * width}\n", flush=True)


def print_step(step_name: str) -> None:
    """打印步骤进度。"""
    print(f"[{_timestamp()}] >>> {step_name}", flush=True)


def build_base_config(strategy: str, output_dir: str, pipeline_mode: Optional[str] = None) -> Dict:
    """构建 trainer 所需的 base_config 字典。

    Naive 和 Smart 使用不同的 pipeline 配置:
      - Naive:  固定多 stage (NUM_PIPELINE_STAGES), 手动 workers, auto_config=OFF
      - Smart:  自动选择最优 (auto_config=ON, workers=空由系统决定)
    """
    import torch as _torch
    device = DEVICE if DEVICE != "auto" else ("cuda" if _torch.cuda.is_available() else "cpu")

    # 流水线追踪日志
    pipeline_trace_log_path = ""
    if strategy in ("pipeline_naive", "pipeline_smart"):
        pipeline_trace_log_path = os.path.join(output_dir, "pipeline_trace.log")

    # 根据策略选择 pipeline 配置
    if strategy == "pipeline_smart":
        # Smart: 1 stage × 3 workers (手动指定)
        enable_auto = SMART_ENABLE_AUTO_PIPELINE_CONFIG
        num_stages = SMART_NUM_PIPELINE_STAGES
        train_workers = SMART_PIPELINE_STAGE_TRAIN_WORKERS
        eval_workers = ""
        pipeline_md = pipeline_mode if pipeline_mode else "smart"
    elif strategy == "pipeline_naive":
        # Naive: 固定多 stage 流水线 (强制 pipeline)
        enable_auto = False
        num_stages = NUM_PIPELINE_STAGES
        train_workers = PIPELINE_STAGE_TRAIN_WORKERS
        eval_workers = PIPELINE_STAGE_EVAL_WORKERS
        pipeline_md = "naive"
    else:
        # Serial / Data Parallel: pipeline 参数无关
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
    """运行 Serial 搜索策略。"""
    print_header("策略 1/4: Serial (串行搜索)")

    base_config = build_base_config("serial", output_dir)
    trainer = GraphNASTrainer(base_config)

    search_space = get_search_space(SEARCH_SPACE)
    if SEARCH_MODE == "rl":
        controller = RLGraphNASController(search_space, seed=SEED, lr=CONTROLLER_LR)
    else:
        controller = RandomGraphNASController(search_space, seed=SEED)

    coarse_trials = COARSE_TRIALS
    coarse_epochs = COARSE_EPOCHS
    rerank_epochs = RERANK_EPOCHS if RERANK_EPOCHS > 0 else coarse_epochs
    eval_seeds = (
        [int(x.strip()) for x in EVAL_SEEDS.split(",") if x.strip()]
        if EVAL_SEEDS else None
    )

    t0 = time.time()
    best, results = trainer.search(
        controller=controller,
        coarse_trials=coarse_trials,
        coarse_epochs=coarse_epochs,
        rerank_top_k=RERANK_TOP_K,
        rerank_epochs=rerank_epochs,
        eval_seeds=eval_seeds,
        family_balanced_rerank=FAMILY_BALANCED_RERANK,
        family_balance_per_model=FAMILY_BALANCE_PER_MODEL,
        time_budget_sec=TIME_BUDGET_SEC,
    )
    elapsed = time.time() - t0

    # 保存结果
    save_strategy_results(output_dir, "serial", best, results)
    return best, results, elapsed


def run_data_parallel(output_dir: str) -> Tuple[Dict, List[Dict], float]:
    """运行 Data Parallel 搜索策略。"""
    print_header("策略 2/4: Data Parallel (数据并行搜索)")

    base_config = build_base_config("data_parallel", output_dir)
    trainer = GraphNASTrainer(base_config)

    search_space = get_search_space(SEARCH_SPACE)
    if SEARCH_MODE == "rl":
        controller = RLGraphNASController(search_space, seed=SEED, lr=CONTROLLER_LR)
    else:
        controller = RandomGraphNASController(search_space, seed=SEED)

    coarse_trials = COARSE_TRIALS
    coarse_epochs = COARSE_EPOCHS
    rerank_epochs = RERANK_EPOCHS if RERANK_EPOCHS > 0 else coarse_epochs

    t0 = time.time()
    best, results = trainer.search_data_parallel(
        controller=controller,
        coarse_trials=coarse_trials,
        coarse_epochs=coarse_epochs,
        num_workers=DATA_PARALLEL_WORKERS,
        rerank_top_k=RERANK_TOP_K,
        rerank_epochs=rerank_epochs,
        time_budget_sec=TIME_BUDGET_SEC,
    )
    elapsed = time.time() - t0

    save_strategy_results(output_dir, "data_parallel", best, results)
    return best, results, elapsed


def run_pipeline_naive(output_dir: str) -> Tuple[Dict, List[Dict], float]:
    """运行 Pipeline Naive (批次同步) 搜索策略。"""
    print_header("策略 3/4: Pipeline Naive (流水线批次同步)")

    base_config = build_base_config("pipeline_naive", output_dir, pipeline_mode="naive")
    trainer = GraphNASTrainer(base_config)

    search_space = get_search_space(SEARCH_SPACE)
    if SEARCH_MODE == "rl":
        controller = RLGraphNASController(search_space, seed=SEED, lr=CONTROLLER_LR)
    else:
        controller = RandomGraphNASController(search_space, seed=SEED)

    coarse_trials = COARSE_TRIALS
    coarse_epochs = COARSE_EPOCHS
    rerank_epochs = RERANK_EPOCHS if RERANK_EPOCHS > 0 else coarse_epochs

    t0 = time.time()
    best, results = trainer.search_pipeline(
        controller=controller,
        coarse_trials=coarse_trials,
        architectures_per_step=ARCHITECTURES_PER_STEP,
        coarse_epochs=coarse_epochs,
        rerank_top_k=RERANK_TOP_K,
        rerank_epochs=rerank_epochs,
        family_balanced_rerank=FAMILY_BALANCED_RERANK,
        family_balance_per_model=FAMILY_BALANCE_PER_MODEL,
        time_budget_sec=TIME_BUDGET_SEC,
    )
    elapsed = time.time() - t0

    save_strategy_results(output_dir, "pipeline_naive", best, results)
    return best, results, elapsed


def run_pipeline_smart(output_dir: str) -> Tuple[Dict, List[Dict], float]:
    """运行 Pipeline Smart (异步持久化池) 搜索策略。"""
    print_header("策略 4/4: Pipeline Smart (流水线异步持久化池)")

    base_config = build_base_config("pipeline_smart", output_dir, pipeline_mode="smart")
    trainer = GraphNASTrainer(base_config)

    search_space = get_search_space(SEARCH_SPACE)
    if SEARCH_MODE == "rl":
        controller = RLGraphNASController(search_space, seed=SEED, lr=CONTROLLER_LR)
    else:
        controller = RandomGraphNASController(search_space, seed=SEED)

    coarse_trials = COARSE_TRIALS
    coarse_epochs = COARSE_EPOCHS
    rerank_epochs = RERANK_EPOCHS if RERANK_EPOCHS > 0 else coarse_epochs

    t0 = time.time()
    best, results = trainer.search_pipeline(
        controller=controller,
        coarse_trials=coarse_trials,
        architectures_per_step=ARCHITECTURES_PER_STEP,
        coarse_epochs=coarse_epochs,
        rerank_top_k=RERANK_TOP_K,
        rerank_epochs=rerank_epochs,
        family_balanced_rerank=FAMILY_BALANCED_RERANK,
        family_balance_per_model=FAMILY_BALANCE_PER_MODEL,
        time_budget_sec=TIME_BUDGET_SEC,
    )
    elapsed = time.time() - t0

    save_strategy_results(output_dir, "pipeline_smart", best, results)
    return best, results, elapsed


def save_strategy_results(output_dir: str, strategy: str, best: Dict, results: List[Dict]) -> None:
    """保存单个策略的结果到子目录。"""
    strat_dir = os.path.join(output_dir, strategy)
    os.makedirs(strat_dir, exist_ok=True)

    # 最佳架构 JSON
    best_path = os.path.join(strat_dir, "best_arch.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(_serialize(best), f, ensure_ascii=False, indent=2)

    # Leaderboard CSV
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

    # 可读摘要文本
    summary_path = os.path.join(strat_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"策略: {strategy}\n")
        f.write(f"搜索空间: {SEARCH_SPACE}\n")
        f.write(f"搜索模式: {SEARCH_MODE}\n")
        f.write(f"粗搜索数: {COARSE_TRIALS}\n")
        f.write(f"粗搜索epochs: {COARSE_EPOCHS}\n")
        f.write(f"重排序K: {RERANK_TOP_K}\n")
        f.write(f"重排序epochs: {RERANK_EPOCHS}\n")
        f.write(f"\n--- 最佳架构 ---\n")
        f.write(f"val_score: {best.get('selected_val_score', best.get('score', 'N/A'))}\n")
        f.write(f"test_score: {best.get('test_score', 'N/A')}\n")
        f.write(f"test_mrr: {best.get('test_mrr', 'N/A')}\n")
        f.write(f"test_recall_at_k: {best.get('test_recall_at_k', 'N/A')}\n")
        f.write(f"参数量: {best.get('params', 'N/A')}\n")
        f.write(f"模型: {best.get('config', {}).get('model', 'N/A')}\n")
        f.write(f"\n架构配置:\n{json.dumps(best.get('config', {}), ensure_ascii=False, indent=2)}\n")

    print(f"  [{strategy}] 结果已保存到: {strat_dir}", flush=True)


def _serialize(obj: Any) -> Any:
    """将不可JSON序列化的对象转换为可序列化形式。"""
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


def generate_comparison(run_dir: str, strategy_results: Dict[str, Tuple[Dict, float]]) -> str:
    """生成最终对比报告。"""
    print_header("生成最终对比报告")

    md_path = os.path.join(run_dir, "comparison.md")
    json_path = os.path.join(run_dir, "comparison.json")

    headers = ["策略", "执行时间(s)", "val_score", "test_score", "test_mrr",
               "test_recall@k", "参数量", "模型族", "搜索架构数"]

    rows = []
    json_data = {}

    strategy_labels = {
        "serial": "Serial (串行基线)",
        "data_parallel": "Data Parallel (数据并行)",
        "pipeline_naive": "Pipeline Naive (批次同步)",
        "pipeline_smart": "Pipeline Smart (异步池)",
    }

    for strategy, (best, elapsed) in strategy_results.items():
        row = [
            strategy_labels.get(strategy, strategy),
            f"{elapsed:.1f}",
            f"{best.get('selected_val_score', best.get('score', 0)):.4f}",
            f"{best.get('test_score', 0):.4f}" if best.get('test_score') is not None else "N/A",
            f"{best.get('test_mrr', 0):.4f}" if best.get('test_mrr') is not None else "N/A",
            f"{best.get('test_recall_at_k', 0):.4f}" if best.get('test_recall_at_k') is not None else "N/A",
            str(best.get('params', 'N/A')),
            best.get('config', {}).get('model', 'unknown'),
            str(best.get('coarse_trials', COARSE_TRIALS)),
        ]
        rows.append(row)

        json_data[strategy] = {
            "label": strategy_labels.get(strategy, strategy),
            "elapsed_sec": round(elapsed, 1),
            "val_score": best.get('selected_val_score', best.get('score')),
            "test_score": best.get('test_score'),
            "test_mrr": best.get('test_mrr'),
            "test_recall_at_k": best.get('test_recall_at_k'),
            "params": best.get('params'),
            "model": best.get('config', {}).get('model'),
            "best_config": best.get('config', {}),
        }

    # 写 Markdown 对比报告
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# JODIE GraphNAS 全策略对比报告\n\n")
        f.write(f"**生成时间**: {_timestamp()}\n\n")
        f.write(f"## 实验配置\n\n")
        f.write(f"| 参数 | 值 |\n")
        f.write(f"|------|----|\n")
        f.write(f"| 搜索空间 | {SEARCH_SPACE} |\n")
        f.write(f"| 搜索模式 | {SEARCH_MODE} |\n")
        f.write(f"| 数据集 | {DATASET} |\n")
        f.write(f"| 粗搜索trial数 | {COARSE_TRIALS} |\n")
        f.write(f"| 粗搜索epochs | {COARSE_EPOCHS} |\n")
        f.write(f"| 重排序K | {RERANK_TOP_K} |\n")
        f.write(f"| 重排序epochs | {RERANK_EPOCHS} |\n")
        f.write(f"| 选择指标 | {SELECTION_METRIC} |\n")
        f.write(f"| 训练模式 | {BATCH_MODE} |\n")
        f.write(f"| 随机种子 | {SEED} |\n")
        if DATASET == "synthetic":
            f.write(f"| 用户数 | {NUM_USERS} |\n")
            f.write(f"| 物品数 | {NUM_ITEMS} |\n")
            f.write(f"| 交互数 | {NUM_INTERACTIONS} |\n")
        f.write(f"| GPU列表 | {GPU_LIST} |\n")
        f.write(f"| 分区大小 | {PARTITION_SIZE} |\n")
        f.write(f"| 流水线Naive | {NUM_PIPELINE_STAGES} stages × {PIPELINE_STAGE_TRAIN_WORKERS} workers |\n")
        f.write(f"| 流水线Smart | {'自动' if SMART_ENABLE_AUTO_PIPELINE_CONFIG else '手动'} |\n\n")

        f.write(f"## 核心指标对比\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["------"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")

        f.write(f"\n## 速度对比\n\n")
        f.write("| 策略 | 执行时间 | 相对Serial加速比 | 平均每架构时间 |\n")
        f.write("|------|----------|-----------------|---------------|\n")
        has_serial = "serial" in strategy_results
        serial_time = strategy_results["serial"][1] if has_serial else 1.0
        if serial_time <= 0.1:
            serial_time = 1.0

        for strategy, (best, elapsed) in strategy_results.items():
            if has_serial and strategy != "serial":
                speedup = f"{serial_time / max(elapsed, 0.1):.1f}x"
            elif strategy == "serial":
                speedup = "1.0x (基线)"
            else:
                speedup = "N/A (Serial失败)"
            per_arch = elapsed / max(COARSE_TRIALS, 1)
            f.write(f"| {strategy_labels.get(strategy, strategy)} | {elapsed:.1f}s | {speedup} | {per_arch:.1f}s |\n")

        f.write(f"\n## 最佳架构详情\n\n")
        for strategy, (best, elapsed) in strategy_results.items():
            f.write(f"### {strategy_labels.get(strategy, strategy)}\n\n")
            config = best.get('config', {})
            f.write(f"```json\n{json.dumps(config, ensure_ascii=False, indent=2)}\n```\n\n")

        f.write(f"\n## 说明\n\n")
        f.write(f"- **val_score**: 验证集上选择出的最佳架构分数\n")
        f.write(f"- **test_score**: 在 train+val 上重新训练后在 test 集上的最终分数\n")
        f.write(f"- 最终架构排名应**以 test_score 为准**\n")
        f.write(f"- Pipeline 策略的速度优势在 trial 数和 GPU 数增大时更加明显\n")

    # 写 JSON 对比数据
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "search_space": SEARCH_SPACE,
                "search_mode": SEARCH_MODE,
                "dataset": DATASET,
                "coarse_trials": COARSE_TRIALS,
                "coarse_epochs": COARSE_EPOCHS,
                "rerank_top_k": RERANK_TOP_K,
                "rerank_epochs": RERANK_EPOCHS,
                "selection_metric": SELECTION_METRIC,
                "batch_mode": BATCH_MODE,
                "seed": SEED,
            },
            "strategies": json_data,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  对比报告 (Markdown): {md_path}")
    print(f"  对比数据 (JSON):    {json_path}")
    return md_path


def print_final_summary(strategy_results: Dict[str, Tuple[Dict, float]]) -> None:
    """在终端打印最终摘要表格。"""
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


def ensure_ray_clean():
    """确保 Ray 在干净状态。"""
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            print("[Init] Ray shutdown from previous run", flush=True)
            time.sleep(2)
    except Exception:
        pass


def main():
    """主入口：依次运行所有启用的策略并生成对比报告。"""
    print_header("JODIE GraphNAS 全策略对比")

    # ── 服务器启动前检查 ──
    print("  [预检] 正在验证运行环境...")
    checks_ok = True

    # 检查数据文件
    if DATASET == "public_csv":
        data_path = LOCAL_DATA_PATH
        if not os.path.exists(data_path):
            print(f"  [预检] ✗ 数据文件不存在: {data_path}")
            checks_ok = False
        else:
            fsize_mb = os.path.getsize(data_path) / (1024*1024)
            print(f"  [预检] ✓ 数据文件: {data_path} ({fsize_mb:.1f} MB)")

    # 检查 GPU
    gpu_ids = [x.strip() for x in GPU_LIST.split(",") if x.strip()]
    try:
        import torch as _t
        cuda_avail = _t.cuda.is_available()
        cuda_count = _t.cuda.device_count() if cuda_avail else 0
        print(f"  [预检] ✓ CUDA 可用: {cuda_avail}, 可见GPU数: {cuda_count}")
        if cuda_count < len(gpu_ids):
            print(f"  [预检] ⚠ 配置请求 {len(gpu_ids)} 个GPU, 但只有 {cuda_count} 个可见")
    except Exception:
        print(f"  [预检] ⚠ 无法检测 GPU, 将使用 CPU")
        cuda_count = 0

    # 检查 Ray
    try:
        import ray
        print(f"  [预检] ✓ Ray 已安装 (version={ray.__version__})")
    except ImportError:
        print(f"  [预检] ✗ Ray 未安装! pipeline 和 data_parallel 策略将失败")
        checks_ok = False

    # 检查 Pipeline 配置 (Naive 和 Smart 使用不同策略)
    worker_str = PIPELINE_STAGE_TRAIN_WORKERS
    workers_list = [int(x) for x in worker_str.split(",") if x.strip()]
    print(f"  [预检] ✓ Pipeline Naive: {NUM_PIPELINE_STAGES} stages × {workers_list} workers (固定)")
    if sum(workers_list) > len(gpu_ids):
        print(f"  [预检] ⚠ Naive 总 worker {sum(workers_list)} > GPU {len(gpu_ids)}, Ray 将做时分复用")
    if SMART_ENABLE_AUTO_PIPELINE_CONFIG:
        print(f"  [预检] ✓ Pipeline Smart: 自动配置 (将根据数据量选择最优 stage/worker)")
    else:
        print(f"  [预检] ✓ Pipeline Smart: 手动配置")

    if not checks_ok:
        print("\n  [预检] 存在致命错误, 请修复后重新运行。")
        sys.exit(1)
    print()

    # 打印配置
    print("  当前配置:")
    print(f"    搜索空间: {SEARCH_SPACE}  搜索模式: {SEARCH_MODE}")
    print(f"    数据集: {DATASET}  选择指标: {SELECTION_METRIC}")
    print(f"    粗搜索: {COARSE_TRIALS} trials × {COARSE_EPOCHS} epochs")
    print(f"    重排序: top {RERANK_TOP_K} × {RERANK_EPOCHS} epochs")
    print(f"    GPU: {GPU_LIST}  流水线Naive: {NUM_PIPELINE_STAGES}stages×{PIPELINE_STAGE_TRAIN_WORKERS}workers")
    print(f"    流水线Smart: {'自动' if SMART_ENABLE_AUTO_PIPELINE_CONFIG else '手动'}  DP workers: {DATA_PARALLEL_WORKERS}")
    print(f"    时间预算: 无限制(全量运行)  启用策略: {ENABLE_STRATEGIES}")
    print()

    # 创建运行时目录
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"  结果目录: {run_dir}\n")

    # 保存运行时配置（仅保存原始类型的参数，排除模块引用）
    config_path = os.path.join(run_dir, "config.json")
    config_snapshot = {}
    for k, v in globals().items():
        if not k.isupper() or k.startswith("_"):
            continue
        if isinstance(v, (str, int, float, bool, list, tuple, type(None))):
            config_snapshot[k] = v
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(_serialize(config_snapshot), f, ensure_ascii=False, indent=2)
    print(f"  配置快照已保存: {config_path}\n")

    # 策略执行映射
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
            print(f"\n  [{strategy}] ✓ 完成 — 耗时 {elapsed:.1f}s, val_score={best.get('selected_val_score', best.get('score', 0)):.4f}\n", flush=True)
        except Exception as e:
            print(f"\n  [{strategy}] ✗ 失败: {e}", flush=True)
            traceback.print_exc()
            failed_strategies.append(strategy)
            # 失败时继续执行下一个策略
            continue

        # 策略之间短暂休息，确保 Ray 资源释放
        if strategy in ("pipeline_naive", "pipeline_smart") and i < len(ENABLE_STRATEGIES):
            print(f"  [Info] 等待 Ray 资源释放...", flush=True)
            time.sleep(3)

    total_elapsed = time.time() - total_start

    # 最终清理
    ensure_ray_clean()

    # 生成对比报告
    if strategy_results:
        md_path = generate_comparison(run_dir, strategy_results)
        print_final_summary(strategy_results)

    print(f"\n  总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  成功策略: {list(strategy_results.keys())}")
    if failed_strategies:
        print(f"  失败策略: {failed_strategies}")
    print(f"  结果目录: {run_dir}")
    print(f"\n{'=' * 70}")
    print(f"  全部完成 — {_timestamp()}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
