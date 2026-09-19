"""定位实验的 (cell, strategy) 运行配置表 —— 与 POSITIONING_GUIDE.md §2 一一对应。

每个 run 的 config 会被 apply_config.py 写入 run_all.py 顶部配置区。
固定参数(指南 §1 全局规定,apply 时不改动、由 precheck 校验):
  SEARCH_MODE=rl, SEED=42, PARTITION_SIZE=2000, BATCH_MODE="serial",
  FEATURE_DIM=4, DATASET=public_csv,
  SMART_ENABLE_AUTO_PIPELINE_CONFIG=False
SEARCH_SPACE 从 2026-09-09 起按 cell 配置(rnn_only 或 C cell 的 mixed)。

2026-09-09 五臂合并(Phase 3):用户决定终止 Phase 2 链条 → 修复 DP 种子 bug →
把「DP 复测 + 五臂新臂 + Phase 2 剩余」合并成一条链(RUNS_P3)。
五臂 = smart / smart_sync(去异步的架构并行,SMART_PIPELINE_MODE="naive" 纯配置切换)/
       naive_alloc(2,1 前重后轻分配)/ naive_static(1,1,1)/ dp(修复后);serial 为绝对加速比参考。
沿用(已跑、干净,不受 DP bug 影响):B-smart/B-naive、D-naive/D-serial、E-smart/E-naive。
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

# Phase 2(2026-09-09 启动,由用户终止于 E-dp(DP 种子 bug);E-smart/E-naive 有效沿用)
RUNS_P2 = [
    {"cell": "E", "strategy": "smart",  "log": "run_pos_cellE_smart.log"},
    {"cell": "E", "strategy": "naive",  "log": "run_pos_cellE_naive.log"},
    {"cell": "E", "strategy": "dp",     "log": "run_pos_cellE_dp.log"},
    {"cell": "F", "strategy": "smart",  "log": "run_pos_cellF_smart.log"},
    {"cell": "F", "strategy": "naive",  "log": "run_pos_cellF_naive.log"},
    {"cell": "F", "strategy": "dp",     "log": "run_pos_cellF_dp.log"},
    {"cell": "F", "strategy": "serial", "log": "run_pos_cellF_serial.log"},
]

# Phase 3(2026-09-09 五臂合并链,顺序 = 运行顺序):
#   ① B-dp 修复复测放最前:DP 修复的保真门槛(应回到 133K 家族),失败则止损重想
#   ② B-smart_sync:C3 归因决定性对照(去异步的架构并行 @ 20K)
#   ③ E 补齐(dp 复测 / smart_sync / serial 参考)——E-smart/E-naive 沿用
#   ④ D 补齐(smart 重跑[可疑争用]/ dp 复测 / smart_sync / naive_alloc)——D-naive/D-serial 沿用
#   ⑤ F 全臂(200K×3)+ ⑥ C 全臂(mixed 大模型 20K×3,唯一 pipeline 赢面 regime)
#   ⑦ B-serial 垫底(20K 绝对加速比参考,过夜)
RUNS_P3 = [
    {"cell": "B", "strategy": "dp",          "log": "experiments/logs/cellB_dp_fixed_20260909.log"},
    {"cell": "B", "strategy": "smart_sync",  "log": "experiments/logs/cellB_smart_sync_20260909.log"},
    {"cell": "E", "strategy": "dp",          "log": "experiments/logs/cellE_dp_fixed_20260909.log"},
    {"cell": "E", "strategy": "smart_sync",  "log": "experiments/logs/cellE_smart_sync_20260909.log"},
    {"cell": "E", "strategy": "serial",      "log": "experiments/logs/cellE_serial_20260910.log"},
    {"cell": "D", "strategy": "smart",       "log": "experiments/logs/cellD_smart_sync_rep_contended_20260910.log"},
    {"cell": "D", "strategy": "dp",          "log": "experiments/logs/cellD_dp_fixed_20260911.log"},
    {"cell": "D", "strategy": "smart_sync",  "log": "experiments/logs/cellD_smart_sync_20260911.log"},
    {"cell": "D", "strategy": "naive_alloc", "log": "experiments/logs/cellD_naive_alloc_20260911.log"},
    {"cell": "F", "strategy": "smart",       "log": "experiments/logs/cellF_smart_sync_rep_20260911.log"},
    {"cell": "F", "strategy": "naive",       "log": "experiments/logs/cellF_naive_20260911.log"},
    {"cell": "F", "strategy": "naive_alloc", "log": "experiments/logs/cellF_naive_alloc_20260911.log"},
    {"cell": "F", "strategy": "smart_sync",  "log": "experiments/logs/cellF_smart_sync_20260911.log"},
    {"cell": "F", "strategy": "dp",          "log": "experiments/logs/cellF_dp_20260911.log"},
    {"cell": "F", "strategy": "serial",      "log": "experiments/logs/cellF_serial_20260911.log"},
    {"cell": "C", "strategy": "smart",       "log": "experiments/logs/cellC_smart_sync_rep_20260912.log"},
    {"cell": "C", "strategy": "naive",       "log": "experiments/logs/cellC_naive_20260912.log"},
    {"cell": "C", "strategy": "naive_alloc", "log": "experiments/logs/cellC_naive_alloc_20260912.log"},
    {"cell": "C", "strategy": "smart_sync",  "log": "experiments/logs/cellC_smart_sync_20260912.log"},
    {"cell": "C", "strategy": "dp",          "log": "experiments/logs/cellC_dp_20260911.log"},
    {"cell": "C", "strategy": "serial",      "log": "experiments/logs/cellC_serial_20260911.log"},
    {"cell": "B", "strategy": "serial",      "log": "experiments/logs/cellB_serial_20260912.log"},
    # 真异步补臂(2026-09-12):D/F/C smart 原槽位跑成了同步重复(SMART_PIPELINE_MODE 漏写),
    # 这里补跑异步池版;同步重复版(_rep)留作与 smart_sync 的重复校验对。
    {"cell": "D", "strategy": "smart",       "log": "experiments/logs/cellD_smart_async_20260912.log"},
    {"cell": "F", "strategy": "smart",       "log": "experiments/logs/cellF_smart_async_20260912.log"},
    {"cell": "C", "strategy": "smart",       "log": "experiments/logs/cellC_smart_async_20260912.log"},
    # G cell(2026-09-12 用户拍板):100K mixed「模型重 × 数据大」组合格,
    # 检验方差放大后 pipeline 反超架构并行(批同步 straggler)假设。四臂 ~12-16h。
    # 顺序:naive → smart_sync → smart(异步)→ serial(最慢,过夜)。
    {"cell": "G", "strategy": "naive",       "log": "experiments/logs/cellG_naive_20260912.log"},
    {"cell": "G", "strategy": "smart_sync",  "log": "experiments/logs/cellG_smart_sync_20260912.log"},
    {"cell": "G", "strategy": "smart",       "log": "experiments/logs/cellG_smart_async_20260912.log"},
    {"cell": "G", "strategy": "serial",      "log": "experiments/logs/cellG_serial_20260912.log"},
    # naive_async 四臂(2026-09-12 用户拍板):多阶段流水线 + 异步池驱动,与同 cell naive 只差驱动变量。
    # 顺序:B 最便宜(约 2.2h,金丝雀验证从未跑过的「流水线×异步池」路径)→ D → F → G(假设格,过夜机器最静)。
    {"cell": "B", "strategy": "naive_async", "log": "experiments/logs/cellB_naive_async_20260913.log"},
    {"cell": "D", "strategy": "naive_async", "log": "experiments/logs/cellD_naive_async_20260913.log"},
    {"cell": "F", "strategy": "naive_async", "log": "experiments/logs/cellF_naive_async_20260913.log"},
    {"cell": "G", "strategy": "naive_async", "log": "experiments/logs/cellG_naive_async_20260913.log"},
    # E-naive_async(2026-09-13 用户拍板):分解 B 大胜的混杂(2 卡 vs 小数据)。~1.5h。
    {"cell": "E", "strategy": "naive_async", "log": "experiments/logs/cellE_naive_async_20260913.log"},
    # H cell(2026-09-13 用户拍板):200K×3GPU mixed,同步/异步 × pipeline/架构并行 全交叉。
    # 顺序按「同驱动相邻」:naive→smart_sync(同步战线相邻对),naive_async→smart_async(异步战线相邻对)。
    {"cell": "H", "strategy": "naive",       "log": "experiments/logs/cellH_naive_20260913.log"},
    {"cell": "H", "strategy": "smart_sync",  "log": "experiments/logs/cellH_smart_sync_20260913.log"},
    {"cell": "H", "strategy": "naive_async", "log": "experiments/logs/cellH_naive_async_20260913.log"},
    {"cell": "H", "strategy": "smart",       "log": "experiments/logs/cellH_smart_async_20260913.log"},
]

# Phase 4/5(2026-09-15 晚,保真修复 Fix A-D 落地后全量重跑):
#   Fix A 换专用 Generator 使采样流与旧全局流不同 → 旧选择数据全部作废;
#   修复后需按实验设计表 4/6 重跑五臂矩阵并验证位级一致。
#   phase4 = B 三臂金丝雀(20K 最便宜):serial(真值)/ naive(同步批协议,诚实可比)/
#     naive_async(应 =serial 位级)——验证完整搜索回路,通过后放行 phase5。
#   phase5 = 其余五臂矩阵(serial/dp/naive/smart_sync/naive_async,
#     C 无 naive_async、G/H 无 dp,与实验设计表 4 一致),按单 cell 成本升序。
RUNS_P4 = [
    {"cell": "B", "strategy": "serial",      "log": "experiments/logs/cellB_serial_20260916.log"},
    {"cell": "B", "strategy": "naive",       "log": "experiments/logs/cellB_naive_20260916.log"},
    {"cell": "B", "strategy": "naive_async", "log": "experiments/logs/cellB_naive_async_20260916.log"},
]
RUNS_P5 = [
    {"cell": "B", "strategy": "dp",          "log": "experiments/logs/cellB_dp_20260916.log"},
    {"cell": "B", "strategy": "smart_sync",  "log": "experiments/logs/cellB_smart_sync_20260916.log"},
    {"cell": "C", "strategy": "serial",      "log": "experiments/logs/cellC_serial_20260916.log"},
    {"cell": "C", "strategy": "naive",       "log": "experiments/logs/cellC_naive_20260916.log"},
    {"cell": "C", "strategy": "dp",          "log": "experiments/logs/cellC_dp_20260916.log"},
    {"cell": "C", "strategy": "smart_sync",  "log": "experiments/logs/cellC_smart_sync_20260916.log"},
    {"cell": "G", "strategy": "serial",      "log": "experiments/logs/cellG_serial_20260916.log"},
    {"cell": "G", "strategy": "naive",       "log": "experiments/logs/cellG_naive_20260916.log"},
    {"cell": "G", "strategy": "naive_async", "log": "experiments/logs/cellG_naive_async_20260916.log"},
    {"cell": "G", "strategy": "smart_sync",  "log": "experiments/logs/cellG_smart_sync_20260916.log"},
    {"cell": "D", "strategy": "serial",      "log": "experiments/logs/cellD_serial_20260916.log"},
    {"cell": "D", "strategy": "naive",       "log": "experiments/logs/cellD_naive_20260916.log"},
    {"cell": "D", "strategy": "naive_async", "log": "experiments/logs/cellD_naive_async_20260916.log"},
    {"cell": "D", "strategy": "dp",          "log": "experiments/logs/cellD_dp_20260916.log"},
    {"cell": "D", "strategy": "smart_sync",  "log": "experiments/logs/cellD_smart_sync_20260916.log"},
    {"cell": "E", "strategy": "serial",      "log": "experiments/logs/cellE_serial_20260916.log"},
    {"cell": "E", "strategy": "naive",       "log": "experiments/logs/cellE_naive_20260916.log"},
    {"cell": "E", "strategy": "naive_async", "log": "experiments/logs/cellE_naive_async_20260916.log"},
    {"cell": "E", "strategy": "dp",          "log": "experiments/logs/cellE_dp_20260916.log"},
    {"cell": "E", "strategy": "smart_sync",  "log": "experiments/logs/cellE_smart_sync_20260916.log"},
    {"cell": "F", "strategy": "serial",      "log": "experiments/logs/cellF_serial_20260916.log"},
    {"cell": "F", "strategy": "naive",       "log": "experiments/logs/cellF_naive_20260916.log"},
    {"cell": "F", "strategy": "naive_async", "log": "experiments/logs/cellF_naive_async_20260916.log"},
    {"cell": "F", "strategy": "dp",          "log": "experiments/logs/cellF_dp_20260916.log"},
    {"cell": "F", "strategy": "smart_sync",  "log": "experiments/logs/cellF_smart_sync_20260916.log"},
    {"cell": "H", "strategy": "serial",      "log": "experiments/logs/cellH_serial_20260916.log"},
    {"cell": "H", "strategy": "naive",       "log": "experiments/logs/cellH_naive_20260916.log"},
    {"cell": "H", "strategy": "naive_async", "log": "experiments/logs/cellH_naive_async_20260916.log"},
    {"cell": "H", "strategy": "smart_sync",  "log": "experiments/logs/cellH_smart_sync_20260916.log"},
]

# RUNS_P6(2026-09-19 用户拍板):终检补跑队列——B/C 行带 ★/⚠ 的非 serial 臂,
# 干净条件下重跑拿可信计时。注意:保真结论不会因重跑改变(dp 分叉是结构性微批平均,
# 只修计时);C-serial 的 ★ 偏差 2.3% 已过判据,不重跑。新 log 名不覆盖旧带标注日志。
# 顺序与 RUNS_P5 一致:B 先 C 后;待 phase5 链结束后以 chain_all.sh phase6 启动。
RUNS_P6 = [
    {"cell": "B", "strategy": "dp",          "log": "experiments/logs/cellB_dp_rerun_20260920.log"},
    {"cell": "B", "strategy": "smart_sync",  "log": "experiments/logs/cellB_smart_sync_rerun_20260920.log"},
    {"cell": "C", "strategy": "naive",       "log": "experiments/logs/cellC_naive_rerun_20260920.log"},
    {"cell": "C", "strategy": "dp",          "log": "experiments/logs/cellC_dp_rerun_20260920.log"},
    {"cell": "C", "strategy": "smart_sync",  "log": "experiments/logs/cellC_smart_sync_rerun_20260920.log"},
]

# 策略简名 → run_all.py ENABLE_STRATEGIES 里的内部名
# 注意 smart_sync / naive_alloc 与 smart / naive 共用内部策略名,
# 差别只靠配置区分:smart_sync = SMART_PIPELINE_MODE="naive";
# naive_alloc = stages/workers 配比(2,1)。
STRATEGY_KEY = {
    "smart": "pipeline_smart",
    "smart_sync": "pipeline_smart",
    "naive": "pipeline_naive",
    "naive_alloc": "pipeline_naive",
    "naive_async": "pipeline_naive",
    "dp": "data_parallel",
    "serial": "serial",
}

# 每个 cell 的基础参数(指南 §2 表格)
# 2026-09-07 调整:
#   - B 固定用 5,6(0,1 被占;4090 同型号,卡号不影响实验设计)
#   - D 用 "auto":由 chain_all.sh 在启动时动态选任意 3 张空闲卡
# 2026-09-09 调整:
#   - B 改回 "auto"(5,6 现被他人占用,0/1/2 已空;动态选卡更稳)
#   - 新增 CELL_C(mixed 大模型 20K×3,指南 §2)、CELL_S(修复冒烟测试,不入链条)
CELL_D = {
    "MAX_EVENTS": 100000,
    "SEARCH_SPACE": "rnn_only",
    "COARSE_TRIALS": 12,
    "RERANK_TOP_K": 0,
    "GPU_LIST": "auto",
    "DATA_PARALLEL_WORKERS": 3,
}
CELL_B = {
    "MAX_EVENTS": 20000,
    "SEARCH_SPACE": "rnn_only",
    "COARSE_TRIALS": 50,
    "RERANK_TOP_K": 8,
    "GPU_LIST": "auto",
    "DATA_PARALLEL_WORKERS": 2,
}
# Phase 2(2026-09-09 用户拍板跑):
#   E = 100K×2GPU(隔离数据/GPU 轴,判 naive-vs-smart 交叉点由谁驱动)
#   F = 200K×3GPU(找 naive-vs-DP 交叉点;MAX_EVENTS/COARSE_TRIALS/RERANK 同 handoff §9.1)
CELL_E = {
    "MAX_EVENTS": 100000,
    "SEARCH_SPACE": "rnn_only",
    "COARSE_TRIALS": 12,
    "RERANK_TOP_K": 0,
    "GPU_LIST": "auto",
    "DATA_PARALLEL_WORKERS": 2,
}
CELL_F = {
    "MAX_EVENTS": 200000,
    "SEARCH_SPACE": "rnn_only",
    "COARSE_TRIALS": 12,
    "RERANK_TOP_K": 0,
    "GPU_LIST": "auto",
    "DATA_PARALLEL_WORKERS": 3,
}
# C = 20K×3GPU mixed 空间(432 候选、含 hybrid 大模型):模型变大(阶段计算变重)
# 是否帮 pipeline —— 唯一的「pipeline 赢面 regime」probe(指南 §2,50/8 与基准一致)
CELL_C = {
    "MAX_EVENTS": 20000,
    "SEARCH_SPACE": "mixed",
    "COARSE_TRIALS": 50,
    "RERANK_TOP_K": 8,
    "GPU_LIST": "auto",
    "DATA_PARALLEL_WORKERS": 3,
}
# G = 100K×3GPU mixed 空间(2026-09-12 用户拍板加跑):「模型重 × 数据大」组合格。
# 假设:大数据放大分区成本倾斜(100K new-users 倾斜 4.1×,20K 仅 1.8×)+ mixed 候选
# 成本方差 → 架构并行(批同步)straggler 效应,存在 pipeline 反超窗口。
# 12t/rerank0 与 D/F 同协议(50t 在 100K mixed 下 ~25h 不可行)。
CELL_G = {
    "MAX_EVENTS": 100000,
    "SEARCH_SPACE": "mixed",
    "COARSE_TRIALS": 12,
    "RERANK_TOP_K": 0,
    "GPU_LIST": "auto",
    "DATA_PARALLEL_WORKERS": 3,
}
# H = 200K×3GPU mixed 空间(2026-09-13 用户拍板加跑):把 G 的「高方差」推到最大数据规模,
# 四臂 naive / naive_async / smart_sync / smart_async——同步/异步两条战线各比一次
# pipeline vs 架构并行,回答「同步/异步 pipeline 能否反超架构并行」的终审问题。
# 12t/rerank0 与 F/G 同协议。
CELL_H = {
    "MAX_EVENTS": 200000,
    "SEARCH_SPACE": "mixed",
    "COARSE_TRIALS": 12,
    "RERANK_TOP_K": 0,
    "GPU_LIST": "auto",
    "DATA_PARALLEL_WORKERS": 3,
}
# S = 冒烟测试 cell(DP 修复验证专用,不进链条):5K×2GPU×12 trials,~15 分钟
CELL_S = {
    "MAX_EVENTS": 5000,
    "SEARCH_SPACE": "rnn_only",
    "COARSE_TRIALS": 12,
    "RERANK_TOP_K": 0,
    "GPU_LIST": "auto",
    "DATA_PARALLEL_WORKERS": 2,
}

CELLS = {"B": CELL_B, "D": CELL_D, "E": CELL_E, "F": CELL_F, "C": CELL_C, "S": CELL_S, "G": CELL_G, "H": CELL_H}

# GPU_LIST="auto" 时链条要选的空闲卡数(不进入 run_all.py,只给 chain_all.sh 用)
GPU_COUNT_BY_CELL = {"B": 2, "D": 3, "E": 2, "F": 3, "C": 3, "S": 2, "G": 3, "H": 3}

# 每个 (cell, strategy) 的策略专属参数(指南 §2 表格)
_STRATEGY_CONFIGS = {
    "D_smart":  {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "3",
                 "SMART_PIPELINE_MODE": "smart"},  # 2026-09-12:显式真异步(此前漏写→继承naive默认→跑成同步重复)
    "D_naive":  {"NUM_PIPELINE_STAGES": 3,
                 "PIPELINE_STAGE_TRAIN_WORKERS": "1,1,1",
                 "PIPELINE_STAGE_EVAL_WORKERS": "1,1,1",
                 "NAIVE_PIPELINE_MODE": "naive"},
    "D_dp":     {},
    "D_serial": {},
    "B_smart":  {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "2"},
    "B_naive":  {"NUM_PIPELINE_STAGES": 2,
                 "PIPELINE_STAGE_TRAIN_WORKERS": "1,1",
                 "PIPELINE_STAGE_EVAL_WORKERS": "1,1",
                 "NAIVE_PIPELINE_MODE": "naive"},
    "B_dp":     {},
    "B_serial": {},
    # Phase 2:E/F 的 naive 阶段数 = GPU 数(与 B/D 同构),smart = 1 阶段 × 全 worker
    "E_smart":  {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "2"},
    "E_naive":  {"NUM_PIPELINE_STAGES": 2,
                 "PIPELINE_STAGE_TRAIN_WORKERS": "1,1",
                 "PIPELINE_STAGE_EVAL_WORKERS": "1,1",
                 "NAIVE_PIPELINE_MODE": "naive"},
    "E_dp":     {},
    "E_serial": {},
    "F_smart":  {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "3",
                 "SMART_PIPELINE_MODE": "smart"},  # 2026-09-12:显式真异步
    "F_naive":  {"NUM_PIPELINE_STAGES": 3,
                 "PIPELINE_STAGE_TRAIN_WORKERS": "1,1,1",
                 "PIPELINE_STAGE_EVAL_WORKERS": "1,1,1",
                 "NAIVE_PIPELINE_MODE": "naive"},
    "F_dp":     {},
    "F_serial": {},
    # C cell(mixed 大模型):naive=3 stage×1 / smart=1×3 / DP=3(与指南 §2 一致)
    "C_smart":  {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "3",
                 "SMART_PIPELINE_MODE": "smart"},  # 2026-09-12:显式真异步
    "C_naive":  {"NUM_PIPELINE_STAGES": 3,
                 "PIPELINE_STAGE_TRAIN_WORKERS": "1,1,1",
                 "PIPELINE_STAGE_EVAL_WORKERS": "1,1,1",
                 "NAIVE_PIPELINE_MODE": "naive"},
    "C_dp":     {},
    "C_serial": {},
    "S_dp":     {},
    # 五臂新臂(2026-09-09):
    #   smart_sync = 去异步的架构并行:结构同 smart(1 阶段×N),驱动改批同步
    #   naive_alloc = 前重后轻分配:2 stages,stage1 拿 2 workers(new users 倾斜在前)
    "B_smart_sync": {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "2",
                     "SMART_PIPELINE_MODE": "naive"},
    "E_smart_sync": {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "2",
                     "SMART_PIPELINE_MODE": "naive"},
    "D_smart_sync": {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "3",
                     "SMART_PIPELINE_MODE": "naive"},
    "F_smart_sync": {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "3",
                     "SMART_PIPELINE_MODE": "naive"},
    "C_smart_sync": {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "3",
                     "SMART_PIPELINE_MODE": "naive"},
    "D_naive_alloc": {"NUM_PIPELINE_STAGES": 2,
                      "PIPELINE_STAGE_TRAIN_WORKERS": "2,1",
                      "PIPELINE_STAGE_EVAL_WORKERS": "2,1",
                      "NAIVE_PIPELINE_MODE": "naive"},
    "F_naive_alloc": {"NUM_PIPELINE_STAGES": 2,
                      "PIPELINE_STAGE_TRAIN_WORKERS": "2,1",
                      "PIPELINE_STAGE_EVAL_WORKERS": "2,1",
                      "NAIVE_PIPELINE_MODE": "naive"},
    "C_naive_alloc": {"NUM_PIPELINE_STAGES": 2,
                      "PIPELINE_STAGE_TRAIN_WORKERS": "2,1",
                      "PIPELINE_STAGE_EVAL_WORKERS": "2,1",
                      "NAIVE_PIPELINE_MODE": "naive"},
    # G cell(100K mixed, 2026-09-12):naive=3 stage×1 / smart(异步池)=1×3 / smart_sync 同结构批同步
    "G_naive":  {"NUM_PIPELINE_STAGES": 3,
                 "PIPELINE_STAGE_TRAIN_WORKERS": "1,1,1",
                 "PIPELINE_STAGE_EVAL_WORKERS": "1,1,1",
                 "NAIVE_PIPELINE_MODE": "naive"},
    "G_smart":  {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "3",
                 "SMART_PIPELINE_MODE": "smart"},
    "G_smart_sync": {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "3",
                 "SMART_PIPELINE_MODE": "naive"},
    "G_serial": {},
    # naive_async 四臂(2026-09-12 用户拍板):多阶段流水线结构不变,只换驱动
    # (NAIVE_PIPELINE_MODE=smart → 异步池 + off-policy 批量 RL),与同 cell naive 只差驱动一个变量,
    # 直接量「异步引擎对流水线」的加速/拖累。阶段数与同 cell naive 完全一致。
    "B_naive_async": {"NUM_PIPELINE_STAGES": 2,
                      "PIPELINE_STAGE_TRAIN_WORKERS": "1,1",
                      "PIPELINE_STAGE_EVAL_WORKERS": "1,1",
                      "NAIVE_PIPELINE_MODE": "smart"},
    "D_naive_async": {"NUM_PIPELINE_STAGES": 3,
                      "PIPELINE_STAGE_TRAIN_WORKERS": "1,1,1",
                      "PIPELINE_STAGE_EVAL_WORKERS": "1,1,1",
                      "NAIVE_PIPELINE_MODE": "smart"},
    "F_naive_async": {"NUM_PIPELINE_STAGES": 3,
                      "PIPELINE_STAGE_TRAIN_WORKERS": "1,1,1",
                      "PIPELINE_STAGE_EVAL_WORKERS": "1,1,1",
                      "NAIVE_PIPELINE_MODE": "smart"},
    "G_naive_async": {"NUM_PIPELINE_STAGES": 3,
                      "PIPELINE_STAGE_TRAIN_WORKERS": "1,1,1",
                      "PIPELINE_STAGE_EVAL_WORKERS": "1,1,1",
                      "NAIVE_PIPELINE_MODE": "smart"},
    # E-naive_async(2026-09-13 用户拍板):2 卡×100K——与 B-naive_async(2 卡×20K)对照,
    # 分解 B 的 1.69× 大胜是「2 卡」效应还是「小数据」效应。
    "E_naive_async": {"NUM_PIPELINE_STAGES": 2,
                      "PIPELINE_STAGE_TRAIN_WORKERS": "1,1",
                      "PIPELINE_STAGE_EVAL_WORKERS": "1,1",
                      "NAIVE_PIPELINE_MODE": "smart"},
    # H cell(200K mixed, 2026-09-13 用户拍板):同步/异步 × pipeline/架构并行 全交叉四臂。
    "H_naive":  {"NUM_PIPELINE_STAGES": 3,
                 "PIPELINE_STAGE_TRAIN_WORKERS": "1,1,1",
                 "PIPELINE_STAGE_EVAL_WORKERS": "1,1,1",
                 "NAIVE_PIPELINE_MODE": "naive"},
    "H_naive_async": {"NUM_PIPELINE_STAGES": 3,
                      "PIPELINE_STAGE_TRAIN_WORKERS": "1,1,1",
                      "PIPELINE_STAGE_EVAL_WORKERS": "1,1,1",
                      "NAIVE_PIPELINE_MODE": "smart"},
    "H_smart":  {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "3",
                 "SMART_PIPELINE_MODE": "smart"},
    "H_smart_sync": {"SMART_NUM_PIPELINE_STAGES": 1, "SMART_PIPELINE_STAGE_TRAIN_WORKERS": "3",
                 "SMART_PIPELINE_MODE": "naive"},
}


def get_config(cell: str, strategy: str) -> dict:
    """返回该 run 需要写入 run_all.py 的参数覆盖(含 ENABLE_STRATEGIES 单策略)。"""
    base = CELLS[cell]
    cfg = dict(base)
    cfg.update(_STRATEGY_CONFIGS[f"{cell}_{strategy}"])
    cfg["ENABLE_STRATEGIES"] = [STRATEGY_KEY[strategy]]
    return cfg


def gpu_count_needed(cell: str) -> int:
    """GPU_LIST="auto" 时链条要选的空闲卡数。"""
    return GPU_COUNT_BY_CELL[cell]


# 预检固定值(与结果目录 config.json 快照逐一比对)
FIXED = {
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
