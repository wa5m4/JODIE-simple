# Pipeline 定位实验 · 运行说明(C2 生死线)

**实验名称**:流水线策略(pipeline_naive)相对 trial 并行(DP / pipeline_smart)的交叉点扫描
**目的**:C2 的定位生死线。2026-08-13 数据(FINAL_REPORT,修复全开)显示:在 NAS 主工作负载(50 trial × 2 epoch、133K 小模型、MOOC 20K 事件)上,流水线 **输给** trial 并行——Naive 1.5× vs DP 1.6× vs Smart 2.3×(test 全部 0.8561);2026-08-26 异步复证后 Smart 更达 **3.0×**。诊断:NAS 的天然可并行单元是 trial 本身(trial 数 ≫ worker 数),而流水线要填满需要"1 个 trial 的阶段数 ≥ worker 数";Naive 的流水深度收益被 stage 状态迁移 + batch 同步 barrier 吃掉。
**本实验系统扫描 GPU × 数据 × 搜索空间,寻找"流水线 ≥ trial 并行"的交叉点**:

- **找到交叉点** → C2 成立,故事 = "按 workload 选执行策略"(多执行策略 NAS 框架)
- **找不到** → C2 降级为系统组件(保真度故事里的对照实验),贡献重排

**日期**:2026-09-03(Phase 1 先行;Phase 2 视结果而定)

---

## 0. 判读标准(先定,再跑)

每个 cell 判读 **粗搜索阶段吞吐**(trials/hour,只看 coarse 阶段耗时,不看 total;total 含 final test 固定成本):

| 结局 | 判据 | C2 命运 |
|---|---|---|
| **反超** | 任一 cell 中 naive 吞吐 > smart(且 > DP) | C2 成立:流水线在该 workload 是正确选择,故事 = 按 workload 选策略 |
| **趋势** | 无一反超,但 naive/smart 比值随**数据变大**或**GPU 变少**单调上升、逼近 1 | 部分成立:C2 改写为"重负载下流水线逼近 trial 并行",需 Phase 2 加跑确认 |
| **恒输** | 所有 cell naive 都明显落后,比值无趋势 | C2 降级为系统组件,贡献重排(保真度 + 异步为主) |

**保真检查(每个 cell 都要做,防止策略在大负载下破坏保真)**:同一 cell 内三策略选出的架构必须一致(20K cell 应回到 133K 家族附近;100K cell 三策略互相一致即可,不要求 = 133K——数据变了,最优架构可能变)。

## 1. 设计:两阶段网格

所有 cell 固定:`BATCH_MODE=serial`(保真协议,唯一变量=搜索策略)、`SEARCH_MODE=rl`、seed 42、partition 2000、MOOC public_csv、FEATURE_DIM=4。

### Phase 1(诊断性,先跑这两个 cell)

| Cell | 数据 | 空间 | GPU | 策略配置 | COARSE_TRIALS / RERANK | 预计时长 |
|---|---|---|---|---|---|---|
| B | 20K | rnn_only | 2 (0,1) | naive=2 stage×1 / smart=1×2 / DP=2 | 50 / 8(与基准一致) | naive ~5h + smart ~4.5h + DP ~6.5h ≈ **16h** |
| D | **100K** | rnn_only | 3 (0,1,2) | naive=3 stage×1 / smart=1×3 / DP=3 + serial 参考 | **12 / 0**(减负) | naive ~2h + smart ~1.5h + DP ~2h + serial ~6.5h ≈ **12h** |

- **Cell B 回答**:GPU 变少(流水线阶段变少)时,naive 是输得更惨还是追平?(stage 迁移次数随阶段数减少)
- **Cell D 回答**:数据变大(单分区计算变重、迁移成本被摊薄)时,pipeline 是否追平?——**最有希望的方向,若时间紧只跑 D**
- serial 参考(100K、12 trial)只跑一次,用于报告各策略绝对加速比

### Phase 2(条件:Phase 1 出现"趋势"或"反超"后加跑)

| Cell | 数据 | 空间 | GPU | 目的 |
|---|---|---|---|---|
| C | 20K | **mixed**(432 候选、含 hybrid 大模型) | 3 | 模型变大(阶段计算变重)是否帮 pipeline |
| E | 100K | rnn_only | 2 | 交叉确认(D 的 GPU 轴) |
| F(可选) | 400K(全量) | rnn_only | 3 | 极端数据规模(COARSE_TRIALS=4、RERANK=0) |

## 2. 配置怎么改(run_all.py 顶部配置区,每 cell 一次)

| 参数 | Cell B | Cell D | Cell C | Cell E |
|---|---|---|---|---|
| `MAX_EVENTS` | 20000 | 100000 | 20000 | 100000 |
| `SEARCH_SPACE` | rnn_only | rnn_only | mixed | rnn_only |
| `COARSE_TRIALS` | 50 | 12 | 50 | 12 |
| `RERANK_TOP_K` | 8 | **0** | 8 | **0** |
| `GPU_LIST` | "0,1" | "0,1,2" | "0,1,2" | "0,1" |
| naive:`NUM_PIPELINE_STAGES` | 2 | 3 | 3 | 2 |
| naive:`PIPELINE_STAGE_TRAIN_WORKERS` | "1,1" | "1,1,1" | "1,1,1" | "1,1" |
| naive:`PIPELINE_STAGE_EVAL_WORKERS` | "1,1" | "1,1,1" | "1,1,1" | "1,1" |
| smart:`SMART_NUM_PIPELINE_STAGES` | 1 | 1 | 1 | 1 |
| smart:`SMART_PIPELINE_STAGE_TRAIN_WORKERS` | "2" | "3" | "3" | "2" |
| `DATA_PARALLEL_WORKERS` | 2 | 3 | 3 | 2 |
| `ENABLE_STRATEGIES` | 每次只留一个策略(见下) | 同左 | 同左 | 同左 |

- `RERANK_TOP_K=0` 安全:trainer.py 有 `if rerank_top_k > 0` 守卫,final test 自动降为 2 epochs(便宜)。
- smart 保持手动分配(`SMART_ENABLE_AUTO_PIPELINE_CONFIG=False`,智慧分配已证伪)。
- `BATCH_MODE` 保持 `"serial"`;串行参考跑法:`ENABLE_STRATEGIES=["serial"]`。

## 3. 服务器步骤

```bash
cd <项目目录>
git fetch origin
git checkout ablation/positioning   # 定位实验分支(基于 smart-async,含全部保真修复)
git pull

# 每个 cell:改好 run_all.py 顶部配置 → 预检(1 分钟内看 log)→ 全量
nohup PYTHONIOENCODING=utf-8 python run_all.py > run_pos_cellX_<策略>.log 2>&1 &
tail -f run_pos_cellX_<策略>.log   # 预检:策略名、MAX_EVENTS、SEARCH_SPACE、GPU_LIST、BATCH_MODE=serial、seed 42
```

- 每个策略单独一次运行(一次运行 = 一个 log = 一个结果目录)。**不要并行跑两个 cell**——计时实验并行会互相抢 GPU,吞吐数字作废。
- Cell 顺序建议:D(smart → naive → DP → serial)然后 B(smart → naive → DP)。serial 参考放最后(最慢,可过夜)。

## 4. 跑完带回来的东西(每个策略一次)

1. `results/<时间戳>/<策略>/<策略>/summary.txt`(耗时、test_score、架构)
2. `results/<时间戳>/<策略>/<策略>/best_arch.json`
3. `results/<时间戳>/<策略>/<策略>/leaderboard.csv`
4. `results/<时间戳>/comparison.json` ← **关键**:各阶段耗时明细(粗搜索 / 重排序 / 最终测试)
5. `timing_log.csv`(smart 有;naive/DP 若有也带)
6. log 尾部 + 一句话总结:**cell / 策略 / 总耗时 / coarse 耗时(如有) / test_score / 参数量 / 目录时间戳**

## 5. 判读(Claude 拿到数据后计算)

- **第一层·吞吐**:每个 cell 内 naive vs smart vs DP 的 coarse 阶段 trials/h 比值;serial 参考 → 各策略绝对加速比
- **第二层·趋势**:naive/smart 比值跨 cell 的方向(B vs D;Phase 2 再验证)
- **第三层·保真**:同 cell 内三策略最终架构是否一致;20K cell 是否 133K 家族、test ≈ 0.8561 ± 噪声地板(mean|d|=0.0069 / max|d|=0.0887)
- 按 §0 三结局表判 C2 命运,结论同步进论文草稿 + 记忆

## 6. 顺带采集(部分已本地完成)

### 6.1 Challenge III 的 X 倍数字(✅ 已本地算出,partition_stats.py)

| 规模 | 分区数(train) | 事件数 max/min | unique users max/min | **new users max/min** |
|---|---|---|---|---|
| 20K | 7 | 1.00(均匀) | 1.30 | **1.83** |
| 100K | 35 | 1.00 | 1.60 | **4.09** |
| 411K 全量 | 145 | 8.93(尾区 224 条) | 6.46 | 114.5 |

**发现**:count 策略下事件数严格均匀(1.00),倾斜全在 **new users**(每分区首次出现的用户数)。引言 Challenge III 原来写"interaction counts ... vary widely"不成立,已改为 new users 表述、数字用 **up to 4×**(锚定 100K 事件;20K 是 1.8×)。若论文主配置改为 20K 锚定,数字换成 1.8×。重算:`python partition_stats.py <max_events>`。

### 6.2 时间分解数据(导师要求的时间分解图数据源)

comparison.json 里每个策略的 coarse / rerank / final 阶段耗时 → 画"各策略时间去向"柱状图。四个 cell 的同一策略可再拼"加速比 vs 数据规模"曲线(可扩展性实验)。

## 8. 五臂合并实验(Phase 3,2026-09-09 启动)

**背景**:Phase 2 链条启动后发现 DP 后端种子纪律缺失(见 `positioning/DP_FIDELITY_ANALYSIS.md`),E-dp 跑到一半终止;用户拍板:先修 bug、再写自动化、把全部实验合并成一条链。

- **DP 修复**:`trainer.search_data_parallel` 复刻 serial 的「采样 → 设种子 → 训练/评估 → RL 更新」,每 trial `_set_seed(42+trial_idx)`。
- **五臂**:smart / **smart_sync**(去异步的架构并行,纯配置切换 `SMART_PIPELINE_MODE="naive"`)/ **naive_alloc**(2,1 前重后轻分配)/ naive_static(1,1,1)/ dp(修复后);serial 为绝对加速比参考。
- **run 清单**:`positioning/configs.py` RUNS_P3(22 run;顺序 B-dp 复测[保真门槛]→ B-smart_sync[C3 决定性]→ E → D → F → C → B-serial)。cell C = 20K×3GPU **mixed 空间**(432 候选、含 hybrid 大模型),是唯一的「pipeline 赢面 regime」probe。
- **沿用干净数据**(不受 DP bug 影响):B-smart/B-naive、D-naive/D-serial、E-smart/E-naive;需要复测:B-dp/D-dp/E-dp。
- **判据不变**:§0 三结局表 + §5 三层判读 + 分配探针预注册规则(搜索空间限 {1,1,1/2,1/1,2},赢的配置必须 ≥2 stages,不利结果照报)。
- **新纪律**:每次 run 启动/结束把 `nvidia-smi` 负载快照写进 run log;链条全程监控他人作业(2026-09-08 晚 D-smart 疑似被争用污染,机器为多人共享)。**污染判定(2026-09-10 用户裁定放宽)**:0% 利用率只占显存的作业不影响计时——不等待、不警报;他人进程 sm 利用率 >15%(pmon 2s 采样)才算污染,警报 + 保留污染 log + 自动重跑一次;已完成干净 run 自动跳过。
- **2026-09-12 事故与修复**:C cell 四个 pipeline 策略(smart/naive/naive_alloc/smart_sync)在 mixed 空间崩溃——`evaluate_arch_pipeline` 方案 C 分支没给 hybrid 模型传 `graph_ctx`(serial 路径有:jodie_rnn 之外传 graph_template)。rnn_only 的 B/E/D/F 从没触发。已修:trainer.py 补传 graph_template;chain_all.sh 完成判定改为「全部完成 且 无『失败策略:』行」,遇失败策略行中止链条(此前 run_all.py 的「全部完成」横幅在策略失败时也打印,把崩溃当完成)。崩溃 log 留档 `run_pos_cellC_*_crashed_HHMM.log`,四臂 09-12 12:30 起重跑。
- **2026-09-12 链收官 + 补跑 + G cell**:链 22/22 17:11 全部正常收工(含 C 四臂重跑)。当晚补跑真异步三臂 D/F/C-smart(`SMART_PIPELINE_MODE` 显式化,此前默认翻转导致 smart 槽位跑成同步重复);用户拍板新增 **G cell = 100K×3GPU mixed 四臂**(naive/smart_sync/smart_async/serial,12t+rerank0)——检验「数据大×模型重 → 成本方差放大 → pipeline 反超架构并行(批同步 straggler)」假设,09-13 中午前后出齐。同日用户再拍板 **naive_async 四臂**(B/D/F/G):多阶段流水线结构不变、只换异步池驱动(`NAIVE_PIPELINE_MODE="smart"`,run_all.py 第 461 行参数化)——与同 cell naive 只差驱动变量,直接量异步引擎对流水线的加速/拖累;排队在 G cell 之后,顺序 B→D→F→G。**09-13 07:21 全部出齐**:naive_async 四臂对 naive 全胜(+98%/+77%/+21%/+26%),B 62.7 t/h 全系列最快;G cell 无异步 naive/smart_sync=0.99(预注册反超判据未成立)、异步 smart_async +49%。09-13 用户再拍板:**E-naive_async**(分解 B 大胜是 2 卡效应还是小数据效应)+ **H cell = 200K×3GPU mixed 全交叉四臂**(naive/naive_async/smart_sync/smart_async,同步/异步两战线各比一次 pipeline vs 架构并行)——链#3(chain_all_p3e.log)19:54 启动,预计 09-14 凌晨 5 点出齐。所有同步 naive 臂配置显式钉死 `NAIVE_PIPELINE_MODE="naive"`(防继承上一个异步 run 的 "smart" 静默翻转,同款 SMART 翻转 bug 的教训)。全部数据已归位 `experiments/`(results 28+ / logs 27 / archive 11),总账见 `experiments/README.md`,分析见 `positioning/ANALYSIS_20260912.md`。

## 7. 时间线备忘

- 08-13:FINAL_REPORT 四策略一致(PipeSmart 11,112s / 2.3×),但流水线本身(naive)输给 DP
- 08-24~28:五档消融 + 双臂消融 + 补测,全部以 pipeline_naive 为载体(3 stage × 1)完成;段 4 数字全部实锤
- 08-26~27:smart-async 复证(3.0×、位级一致)→ C3 定稿
- 09-03:本指南建立;Phase 1 未跑
- 09-08/09:Phase 1 完成(7/7 run,链条 04:20 结束;GPU 争抢与 CUDA_VISIBLE_DEVICES 事故均已解决)。结果与判读见 `positioning/RESULTS_20260909.md`;论文同步见 `论文/draft_v2_实验结果_C2.md`。**C2 判定:部分成立** —— naive/smart 0.85→1.13 跨 1(反超);naive/DP 2.73→0.93(未反超);绝对加速比 100K 塌缩至 1.17-1.42×。Phase 2(变量隔离)待定
- 09-09:Phase 2 终止于 E-dp(DP 种子 bug 发现);DP 修复 + 五臂合并链启动(§8);E-smart(10.4 t/h)/E-naive(8.9 t/h)有效沿用;D-smart(8.4 t/h)疑似争用污染列入重跑
