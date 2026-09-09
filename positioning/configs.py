"""定位实验的 7 个 (cell, strategy) 运行配置表 —— 与 POSITIONING_GUIDE.md §2 一一对应。

每个 run 的 config 会被 apply_config.py 写入 run_all.py 顶部配置区。
固定参数(指南 §1 全局规定,apply 时不改动、由 precheck 校验):
  SEARCH_MODE=rl, SEED=42, PARTITION_SIZE=2000, BATCH_MODE="serial",
  FEATURE_DIM=4, DATASET=public_csv, SEARCH_SPACE="rnn_only",
  SMART_ENABLE_AUTO_PIPELINE_CONFIG=False
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 启动顺序 = 列表顺序。2026-09-07 调整:0/1/2 被 chengzitong 长期占用,
# 2 卡 cell(B)先跑在空闲的 5,6 上;3 卡 cell(D)的 GPU_LIST="auto" 由链条
# 动态选卡(任意 3 张空闲卡,不再绑定 0,1,2)。serial 仍放最后(最慢,过夜)。
RUNS = [
    {"cell": "B", "strategy": "smart",  "log": "run_pos_cellB_smart.log"},
    {"cell": "B", "strategy": "naive",  "log": "run_pos_cellB_naive.log"},
    {"cell": "B", "strategy": "dp",     "log": "run_pos_cellB_dp.log"},
    {"cell": "D", "strategy": "smart",  "log": "run_pos_cellD_smart.log"},
    {"cell": "D", "strategy": "naive",  "log": "run_pos_cellD_naive.log"},
    {"cell": "D", "strategy": "dp",     "log": "run_pos_cellD_dp.log"},
    {"cell": "D", "strategy": "serial", "log": "run_pos_cellD_serial.log"},
]

# 策略简名 → run_all.py ENABLE_STRATEGIES 里的内部名
STRATEGY_KEY = {
    "smart": "pipeline_smart",
    "naive": "pipeline_naive",
    "dp": "data_parallel",
    "serial": "serial",
}

# 每个 cell 的基础参数(指南 §2 表格)
# 2026-09-07 调整:
#   - B 固定用 5,6(0,1 被占;4090 同型号,卡号不影响实验设计)
#   - D 用 "auto":由 chain_all.sh 在启动时动态选任意 3 张空闲卡
#     (apply_config/precheck 需配合 --gpu-list 传入实际卡号)
CELL_D = {
    "MAX_EVENTS": 100000,
    "COARSE_TRIALS": 12,
    "RERANK_TOP_K": 0,
    "GPU_LIST": "auto",
    "DATA_PARALLEL_WORKERS": 3,
}
CELL_B = {
    "MAX_EVENTS": 20000,
    "COARSE_TRIALS": 50,
    "RERANK_TOP_K": 8,
    "GPU_LIST": "5,6",
    "DATA_PARALLEL_WORKERS": 2,
}

# 每个 (cell, strategy) 的策略专属参数(指南 §2 表格)
_STRATEGY_CONFIGS = {
    "D_smart":  {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "3"},
    "D_naive":  {"NUM_PIPELINE_STAGES": 3,
                 "PIPELINE_STAGE_TRAIN_WORKERS": "1,1,1",
                 "PIPELINE_STAGE_EVAL_WORKERS": "1,1,1"},
    "D_dp":     {},
    "D_serial": {},
    "B_smart":  {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "2"},
    "B_naive":  {"NUM_PIPELINE_STAGES": 2,
                 "PIPELINE_STAGE_TRAIN_WORKERS": "1,1",
                 "PIPELINE_STAGE_EVAL_WORKERS": "1,1"},
    "B_dp":     {},
    "B_serial": {},
}


def get_config(cell: str, strategy: str) -> dict:
    """返回该 run 需要写入 run_all.py 的参数覆盖(含 ENABLE_STRATEGIES 单策略)。"""
    base = CELL_D if cell == "D" else CELL_B
    cfg = dict(base)
    cfg.update(_STRATEGY_CONFIGS[f"{cell}_{strategy}"])
    cfg["ENABLE_STRATEGIES"] = [STRATEGY_KEY[strategy]]
    return cfg


# 预检固定值(与结果目录 config.json 快照逐一比对)
FIXED = {
    "SEARCH_SPACE": "rnn_only",
    "SEARCH_MODE": "rl",
    "BATCH_MODE": "serial",
    "SEED": 42,
    "PARTITION_SIZE": 2000,
    "FEATURE_DIM": 4,
    "SMART_ENABLE_AUTO_PIPELINE_CONFIG": False,
}


def expected_snapshot(cell: str, strategy: str) -> dict:
    """precheck 用:该 run 的完整预期快照值。"""
    exp = get_config(cell, strategy)
    exp.update(FIXED)
    return exp
