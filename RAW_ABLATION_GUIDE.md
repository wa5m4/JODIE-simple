# stale_batch 消融实验运行说明(服务器版)

**实验名称**:RAW 破坏 · 单因素对照(朴素分批 `stale_batch`)
**目的**:为引言段 4 v4 提供干净归因——"朴素分批(不做冲突消解)破坏写后读(RAW)依赖 → 搜索选错架构"。此前段 4 的失败数据(402K/0.7000)干净归因是随机状态 bug(用户定位:bug 不进 intro),所以需要一组**唯一变量 = 批处理模式**的干净对照。
**日期**:2026-08-26

---

## 0. 核心原则(继承 ABLATION_FACTOR_GUIDE 第 0 节)

三层观察,不要只看最终选出的架构:第一层·终点(最终架构 + test)、第二层·轨迹(50 个 coarse 采样序列 / top-8 集合 / leaderboard 分布)、第三层·评分(同一架构的 val 分数差 vs 噪声天花板)。

## 1. 分支与唯一变量

- **分支**:`ablation/stale-batch`(基于 `refactored`,三修复全部开启 ✅)
- **唯一变量**:`BATCH_MODE = "stale_batch"` —— 朴素分批:连续交互直接切块(不查冲突),批内所有交互对**批前状态**计算(deferred 前向),批末统一写回(同节点最后写入生效)。同批重复节点 → 后出现的交互读到旧嵌入 → 破坏 RAW。
- **其余与五档消融完全一致**:rnn_only、50×2 + 8×5、MOOC(public_csv / 20000 事件)、`pipeline_naive`、seed 42、partition 2000、GPU 0,1,2。
- **对照基准**:修复全开 + BATCH_MODE=serial 的 pipeline_naive 运行(133K/0.8561,结果目录 `results/20260811_204240` 与 `results/20260825_133519`)。

## 2. 服务器部署与运行

```bash
cd <项目目录>
git fetch origin
git checkout ablation/stale-batch
nohup PYTHONIOENCODING=utf-8 python run_all.py > run_stale_batch_ablation.log 2>&1 &
```

- 预检(启动后 1 分钟内看 `run_stale_batch_ablation.log`):应显示 8 GPU、`pipeline_naive`、`BATCH_MODE=stale_batch`、rnn_only、seed 42 等。
- 时长:与 f1/f2 单关消融同量级(一个晚上)。

## 3. 跑完带回来的东西

1. `results/<时间戳>/pipeline_naive/pipeline_naive/summary.txt`
2. `results/<时间戳>/pipeline_naive/pipeline_naive/best_arch.json`
3. `results/<时间戳>/pipeline_naive/pipeline_naive/leaderboard.csv`
4. `run_stale_batch_ablation.log`(完整)
5. 一句总结:**最终参数量 / test_score / 结果目录时间戳**

## 4. 判读(Claude 拿到数据后计算)

- 第一层·终点:与基准 133K/0.8561 对比 → 是否选错架构(错成什么)
- 第二层·轨迹:50 个 coarse 采样序列、top-8 集合、leaderboard 分布 vs 基准目录(20260811_204240 / 20260825_133519)
- 第三层·评分:同架构 val 分数差 vs 噪声天花板(mean 0.0069 / max 0.0887,来自两次修复全开运行)
- 判读结论 → 填段 4 v4 的三个占位数字(错误架构参数量/特征、test MRR、差值)

## 5. 跑完恢复

```bash
git checkout refactored
```

注意:`refactored` 上没有 stale_batch 代码(该模式只在消融分支上),跑其他实验用原分支即可;消融分支原封不动保留在 GitHub 上。

## 6. 时间线备忘

- 已跑五档:全关 402K/0.69999、f1 402K/0.69999、f2 402K/0.69999、f3 133K/0.8561(负对照)、修复全开 ×2(bit-identical 133K/0.8561)
- 本次(第六档):stale_batch → RAW 单独效应
