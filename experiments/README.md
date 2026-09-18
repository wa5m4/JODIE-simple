# 五臂定位实验 · 数据归档目录(2026-09-12 整理)

**整理原则**:数据错误的删除(buggy 版本 / 争用污染 / 崩溃残骸 / 调试冒烟);正确数据移入本目录并重命名为准确名字(cell + 数据规模 + 空间 + 策略 + 真实身份 + 日期)。
**整理状态(09-12 17:11 链条 22/22 全部结束)**:全部归位完成——results 28 份、logs 27 个、archive 11 个,仓库根零残留;configs.py RUNS_P3 日志路径已同步到新位置(裸重启链仍能正确跳过)。唯一未做:补跑真异步 D/F/C-smart 三臂(待用户决定,见 §4)。

## 1. 正确结果(results/,24 份,全部有 comparison.json)

| 新名 | 原目录 | 策略 | 耗时 | 架构/test |
|---|---|---|---|---|
| cellB_smart_async_20260907 | 20260907_184345 | smart(真异步) | 2.15h | 133888/0.8561 |
| cellB_naive_20260907 | 20260907_205330 | naive | 2.18h | 133888/0.8561 |
| cellB_dp_fixed_20260909 | 20260909_154706 | dp(修复后) | 4.16h | 133888/0.8561 |
| cellB_smart_sync_20260909 | 20260909_195657 | smart_sync | 1.46h | 133888/0.8561 |
| cellB_serial_20260912 | 20260912_001051 | serial | 1.97h | 133888/0.8561 |
| cellD_naive_20260908 | 20260908_233815 | naive | 1.49h | 338240/0.5652 |
| cellD_serial_20260909 | 20260909_023255 | serial | 1.78h | 8896/0.3670 |
| cellD_smart_sync_rep_contended_20260910 | 20260910_191746 | smart 槽位=smart_sync 重复+机器争用 | 3.21h | 338240/0.5652 |
| cellD_dp_fixed_20260911 | 20260911_002014 | dp(修复后) | 1.60h | 782976/0.6402 |
| cellD_smart_sync_20260911 | 20260911_015659 | smart_sync | 1.00h | 338240/0.5652 |
| cellD_naive_alloc_20260911 | 20260911_025743 | naive_alloc(2,1) | 1.16h | 338240/0.5652 |
| cellE_smart_async_20260909 | 20260909_112947 | smart(真异步) | 1.34h | 1114752/0.2838 |
| cellE_naive_20260909 | 20260909_125055 | naive | 1.59h | 338240/0.5652 |
| cellE_dp_fixed_20260909 | 20260909_212532 | dp(修复后) | 1.74h | 782976/0.6402 |
| cellE_smart_sync_20260909 | 20260909_231107 | smart_sync | 1.34h | 338240/0.5652 |
| cellE_serial_20260910 | 20260910_154802 | serial(attempt2 采用) | 2.28h | 8896/0.3670 |
| cellF_smart_sync_rep_20260911 | 20260911_040828 | smart 槽位=smart_sync 重复 | 1.91h | 133888/0.5933 |
| cellF_naive_20260911 | 20260911_060414 | naive | 1.99h | 133888/0.5933 |
| cellF_naive_alloc_20260911 | 20260911_080500 | naive_alloc(2,1) | 2.25h | 133888/0.5933 |
| cellF_smart_sync_20260911 | 20260911_102046 | smart_sync | 1.93h | 133888/0.5933 |
| cellF_dp_20260911 | 20260911_121732 | dp(干净;分叉是结果) | 4.53h | 334592/0.3213 |
| cellF_serial_20260911 | 20260911_165029 | serial | 3.55h | 133888/0.5933 |
| cellC_dp_20260911 | 20260911_204417 | dp(干净;分叉是结果) | 1.42h | 34176/0.6045 |
| cellC_serial_20260911 | 20260911_221103 | serial | 1.99h | 133888/0.8561 |
| cellC_smart_sync_rep_20260912 | 20260912_123037 | smart 槽位=smart_sync 重复(与 smart_sync 配对校验) | 1.02h | 133888/0.8561 |
| cellC_naive_20260912 | 20260912_133223 | naive | 1.22h | 133888/0.8561 |
| cellC_naive_alloc_20260912 | 20260912_144609 | naive_alloc(2,1) | 1.42h | 133888/0.8561 |
| cellC_smart_sync_20260912 | 20260912_161255 | smart_sync | 0.97h | 133888/0.8561 |

**命名要点(准确身份)**:
- `smart_async` = 真异步 smart(B/E,翻转默认值前)。`smart_sync_rep` = 名义 smart 槽位、实际跑成 smart_sync 重复(D/F;`SMART_PIPELINE_MODE` 默认翻转 bug)。`_contended` = 带机器争用(D-smart 3.21h 与同配置 smart_sync 1.00h 的 3.2× 差=争用,不是策略差异)。
- `dp_fixed` = DP 种子纪律修复后;`cellF_dp`/`cellC_dp` 无后缀但同属修复后(干净 run,架构分叉是真实结果,不是错误)。

## 2. 证据日志(archive/,11 个留档)

| 新名 | 原文件 | 说明 |
|---|---|---|
| cellB_smart_killed_attempt_20260908.log | run_pos_cellB_smart_killed_2154.log | 09-08 21:54 重启动 50s 被杀;B-smart 真数据=cellB_smart_async_20260907 |
| cellD_smart_polluted_phase1_20260908.log | run_pos_cellD_smart_phase1.log | Phase1 争用污染(34176/0.5074) |
| cellD_dp_buggy_phase1_20260909.log | run_pos_cellD_dp_phase1.log | buggy 种子版 |
| cellE_dp_buggy_killed_20260909.log | run_pos_cellE_dp_buggy_killed.log | Phase 2 终止点 |
| cellD_smart_sync_rep_attempt1_killed_20260910.log | run_pos_cellD_smart_rerun_polluted_1917.log | 重跑 attempt1 19:17 被杀 |
| cellD_dp_fixed_attempt1_polluted_20260911.log | run_pos_cellD_dp_fixed_polluted_0020.log | attempt1 00:20 污染 |
| cellE_serial_attempt1_polluted_20260910.log | run_pos_cellE_serial_polluted_1547.log | attempt1 15:47 污染 |
| cellC_smart_crashed_graphctx_20260911.log | run_pos_cellC_smart_crashed_2028.log | graph_ctx 崩溃(已修复) |
| cellC_naive_crashed_graphctx_20260911.log | run_pos_cellC_naive_crashed_2033.log | 同上 |
| cellC_naive_alloc_crashed_graphctx_20260911.log | run_pos_cellC_naive_alloc_crashed_2039.log | 同上 |
| cellC_smart_sync_crashed_graphctx_20260911.log | run_pos_cellC_smart_sync_crashed_2044.log | 同上 |

## 3. 已删除(数据错误,14 目录 + 2 日志,签名核对后删除)

| 文件 | 错误原因 |
|---|---|
| results/20260907_230515 | B-dp buggy 种子版(test 0.6687,已由 cellB_dp_fixed 替换) |
| results/20260908_215548 | D-smart Phase1 争用污染 |
| results/20260909_010858 | D-dp Phase1 buggy 种子版 |
| results/20260909_142651 | E-dp buggy 被杀残骸(无结果) |
| results/20260909_151927 / 153313 | CELL_S 冒烟 ×2(调试烟测,已由 B-dp 保真门槛取代) |
| results/20260910_124744 | B-dp 12:47 被杀启动残骸(timing_log 仅表头) |
| results/20260910_125236 | E-serial attempt1 争用污染 |
| results/20260910_180619 | D-smart 重跑 attempt1 被杀残骸(无结果) |
| results/20260910_223221 | D-dp attempt1 争用污染 |
| results/20260911_202422 / 202906 / 203349 / 203933 | C 四策略 graph_ctx 崩溃残骸(无结果) |
| run_pos_smoke1.log / run_pos_smoke2.log | CELL_S 调试冒烟日志 |

## 4. 进行中的两批补跑(2026-09-12 用户拍板)

1. **真异步三臂 D/F/C-smart**:configs.py `_STRATEGY_CONFIGS` 已加 `SMART_PIPELINE_MODE="smart"`,RUNS_P3 追加 3 run。链#1(chain_all_p3c.log)19:14 启动、自动跳过 22 个,顺序 D→F→C,预计 09-13 凌晨 1 点前出齐。日志名 `cellX_smart_async_20260912.log`。
2. **G cell = 100K×3GPU mixed 四臂**(naive / smart_sync / smart_async / serial,12t+rerank0,~12-16h):用户假设「数据大×模型重 → 分区成本方差(100K new-users 倾斜 4.1×)+ 候选成本方差 → 架构并行批同步 straggler,存在 pipeline 反超窗口」。链#2(chain_all_p3d.log)在链#1 退出后自动启动(PID 1590942 守护),顺序 naive→smart_sync→smart→serial,预计 09-13 中午前后出齐。日志名 `cellG_*_20260912.log`。
3. **naive_async 四臂 B/D/F/G**(2026-09-12 用户拍板):多阶段流水线结构不变(B=2×1,D/F/G=3×1),只换驱动 `NAIVE_PIPELINE_MODE="smart"`(异步池 + off-policy 批量 RL)——与同 cell naive 只差驱动一个变量,直接量「异步引擎对流水线」的加速/拖累。紧随 G cell 之后排队:顺序 B(2.2h,金丝雀验证从未跑过的「流水线×异步池」路径)→ D → F → G(假设格,过夜),预计 09-14 凌晨出齐。日志名 `cellX_naive_async_20260913.log`。实现:run_all.py 新增 NAIVE_PIPELINE_MODE 开关(第 461 行参数化),trainer 异步路径本就支持手动多阶段分配。**已完成 09-13 07:21**:四臂全胜 naive(B +98%/D +77%/F +21%/G +26%),B 臂 62.7 t/h 全系列最快。
4. **E-naive_async + H cell 四臂**(2026-09-13 用户拍板,链#3 chain_all_p3e.log 19:54 启动):(a) E-naive_async(100K×2卡)分解 B 大胜的混杂——2 卡效应还是小数据效应;(b) H = 200K×3GPU mixed 全交叉四臂 naive/naive_async/smart_sync/smart_async,同步/异步两条战线各比一次 pipeline vs 架构并行,回答「同步/异步 pipeline 能否反超架构并行」的终审。顺序按同驱动相邻:naive→smart_sync、naive_async→smart_async。预计 09-14 凌晨 5 点前后出齐。**注意**:所有同步 naive 臂已显式钉死 `NAIVE_PIPELINE_MODE="naive"`(否则继承上一个异步 run 留下的 "smart" 会静默跑成异步——SMART_PIPELINE_MODE 翻转 bug 的同款)。
5. 全部出数后:结果目录归入 results/(cellG_*/cellH_* 命名),logs 归位,补终裁分析。

## 5. 判读口径不变

指南 §0 三结局表 + coarse 阶段 trials/h;D/F/C 的 `smart_sync_rep` 只作「与 smart_sync 的重复校验对」,不得当 smart 臂进五臂终裁。详见 positioning/RESULTS_20260912.md §6。
