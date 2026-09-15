# PipeTGL 论文:实验结果章骨架(C2 定位 —— 2026-09-09 Phase 1 数据)

> **数据来源**:`positioning/RESULTS_20260909.md`(7 run 全完成,2026-09-08/09;协议:BATCH_MODE=serial、SEED=42、partition 2000、rnn_only)
> **用途**:Section 6 的「执行策略选择」小节素材;最终数字以实验章定稿为准。
> **关键局限**:Phase 1 只跑 B(20K/2GPU)与 D(100K/3GPU)两个 cell;B→D 数据与 GPU 同时变化,归因需 Phase 2 隔离。

---

## C2 主线:pipeline_naive 何时追平 trial 并行?

**测量口径**:coarse 阶段 trials/hour(timing_log 跨度,不含 rerank/final test)。

| 规模 | smart | naive | DP | serial | naive/smart | naive/DP |
|---|---|---|---|---|---|---|
| 20K (2 GPU) | 37.2 | 31.7 | 11.6 | — | 0.85 | **2.73×** |
| 100K (3 GPU) | 8.4 | **9.5** | **10.2** | 7.2 | **1.13** | 0.93 |

**结论(供 Section 6 撰写)**:

1. **naive 反超 smart**:naive/smart 随数据 0.85 → 1.13 单调上升并跨过 1 —— 20K 下 2.3-3.0× 的 smart 异步引擎在 100K 被朴素流水线反超 13%。「重负载下流水线逼近 trial 并行」成立了一半。
2. **naive 未反超 DP**:DP 的 micro_batch 同步开销随数据摊薄,2.73× → 0.93×(100K 下 DP 快 7%)。naive-vs-DP 交叉点未找到,需 Phase 2(更大数据 / 3GPU+20K 隔离 cell)。
3. **绝对加速比塌缩**:100K 下 3 卡相对单卡 serial 只有 smart 1.17× / naive 1.32× / DP 1.42×,远低于 20K 主配置的 2.3-3.0× —— 「按 workload 选执行策略」故事的最有力论据:没有哪种策略在所有规模下都是赢家。
4. **保真度分层**:20K 下 smart ≡ naive 位级一致、test 0.8561 命中 133K 家族锚点;DP 在 20K/100K 均偏离家族架构且 val 质量最差(100K:val 0.36 vs naive 0.56;val/test 相关性破裂)。DP 的吞吐优势伴随搜索质量损失 —— 报告并行加速比时必须与保真度并列。

## C2 命运(按 POSITIONING_GUIDE §0 三结局表)

- ✗ 不满足「反超」(需 naive > smart 且 > DP;DP 条件未过)
- ✗ 不满足「恒输」(naive 已反超 smart)
- ✓ **部分成立**:「重负载下流水线逼近 trial 并行」;naive-vs-smart 交叉点已定位(20K~100K 之间);naive-vs-DP 交叉点待 Phase 2
- 贡献定位:C2 降级为「按 workload 选策略」的证据链一环(保真度 + 异步为主线的补充),与指南「找不到完整交叉点 → C2 降级」的分支一致。

## Phase 2 建议(若做)

- 隔离变量:cell E = 100K × 2 GPU(数据不变、GPU 减少)判断 naive-vs-smart 交叉点由数据还是 GPU 驱动;cell F = 200K × 3 GPU 找 naive-vs-DP 交叉点。
- DP 保真度专项:DP 在两种规模都选中 time_linear+normalize 变体 —— 单独排查 DP 的搜索路径(同 seed 下 trial 调度顺序不同 → RL 探索路径分叉?)。

## 待补(实验章最终定稿时)

- [ref] 引用、误差棒(若复跑)、与 FINAL_REPORT 08-13 数字的协议差异说明(DP 1.6× 是 BATCH_MODE≠serial 旧协议,不能与新数字并列)。
