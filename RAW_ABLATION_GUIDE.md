# RAW 破坏 · 双臂消融实验运行说明(服务器版)

**实验名称**:RAW 破坏 · 单因素对照 · 双臂(stale_batch 臂 + t-Batch 桥梁臂)
**目的**:为引言段 4 v4 提供干净归因——"朴素分批(不做冲突消解)破坏写后读(RAW)依赖 → 搜索选错架构"。此前段 4 的失败数据(402K/0.7000)干净归因是随机状态 bug(用户定位:bug 不进 intro),所以需要**唯一变量 = 批处理模式**的干净对照。
**日期**:2026-08-26(2026-08-26 晚更新为双臂)

---

## 0. 核心原则(继承 ABLATION_FACTOR_GUIDE 第 0 节)

三层观察,不要只看最终选出的架构:第一层·终点(最终架构 + test)、第二层·轨迹(50 个 coarse 采样序列 / top-8 集合 / leaderboard 分布)、第三层·评分(同一架构的 val 分数差 vs 噪声天花板)。

**双臂设计(trio 归因)**:serial 基准(已有)→ t-Batch 桥梁臂 → stale_batch 臂。
t-Batch 与 stale_batch 批粒度、每批一更新的优化器语义**完全相同**,唯一区别是批内是否做冲突消解(能否读到旧账)。若 t-Batch ≈ serial 而 stale_batch 大跌,归因钉死在 stale read 上——"分批本身无害,不懂依赖的分批才有害"。这正是 C1 系统保真度协议(评估保真度 2026-08-27 并入系统后)最硬的一组证据——同时充当段 4 case study 的归因数字。

## 1. 分支与唯一变量

| 臂 | 分支 | BATCH_MODE | 预期 |
|---|---|---|---|
| 基准(已有) | refactored | serial | 133K / 0.8561 |
| t-Batch 桥梁臂 | `ablation/tbatch` | `tbatch` | ≈ 133K / ≈ 0.8561 |
| stale_batch 臂 | `ablation/stale-batch` | `stale_batch` | 选错架构 / 掉分(已实锤,见 §7) |

- **stale_batch**:朴素分批——连续交互直接切块(不查冲突),批内所有交互对**批前状态**计算(deferred 前向),批末统一写回(同节点最后写入生效)。同批重复节点 → 后出现的交互读到旧嵌入 → 破坏 RAW。
- **t-Batch**:贪心冲突消解分批——批内节点 ID 唯一,批内永远读不到旧账,前向语义与串行一致。
- **三臂其余配置完全一致**:rnn_only、50×2 + 8×5、MOOC(public_csv / 20000 事件)、`pipeline_naive`、seed 42、partition 2000、GPU 0,1,2。
- **负样本三臂同源**:stale_batch 与 t-Batch 均已改为优先使用数据加载时预分配的 `neg_samples_by_epoch`(与 serial 路径完全一致),负样本来源不再是变量。(stale 修复 commit e729b4c;t-Batch 修复 commit 64518e2)
- **对照基准**:修复全开 + BATCH_MODE=serial 的 pipeline_naive 运行(133K/0.8561,结果目录 `results/20260811_204240` 与 `results/20260825_133519`)。

## 2. 服务器部署与运行(两臂先后跑,各约 4.5 小时,一夜正好)

```bash
cd <项目目录>
git fetch origin

# 第一臂:stale_batch
git checkout ablation/stale-batch
git pull
nohup PYTHONIOENCODING=utf-8 python run_all.py > run_stale_batch_ablation.log 2>&1 &

# 等第一臂跑完后(可选:tail run_stale_batch_ablation.log 确认结束),再跑第二臂
git checkout ablation/tbatch
git pull
nohup PYTHONIOENCODING=utf-8 python run_all.py > run_tbatch_bridge_ablation.log 2>&1 &
```

- 预检(启动后 1 分钟内看对应 log):应显示 8 GPU、`pipeline_naive`、对应 `BATCH_MODE`、rnn_only、seed 42 等。
- 时长:每臂与 f1/f2 单关消融同量级(一个晚上)。

## 3. 跑完带回来的东西(每臂各一份)

1. `results/<时间戳>/pipeline_naive/pipeline_naive/summary.txt`
2. `results/<时间戳>/pipeline_naive/pipeline_naive/best_arch.json`
3. `results/<时间戳>/pipeline_naive/pipeline_naive/leaderboard.csv`
4. `run_stale_batch_ablation.log` 与 `run_tbatch_bridge_ablation.log`(完整)
5. 一句总结:**最终参数量 / test_score / 结果目录时间戳**(两臂各一句)

## 4. 判读(Claude 拿到数据后计算)

- 第一层·终点:两臂与基准 133K/0.8561 对比 → t-Batch 是否守住、stale 是否选错(错成什么)
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
- 本次(第六档,双臂):stale_batch → RAW 单独效应;t-Batch → 桥梁(证明分批本身无害)

## 7. 结果与补测(2026-08-28,全部实锤)

三臂终点:

| 臂 | 选中架构 | 保真 val | 各自报告的 test | 真实串行 test |
|---|---|---|---|---|
| serial 基准 | 133,888(emb128/static-off) | 0.8267 | 0.8561(串行) | 0.856121275963994 |
| t-Batch 桥梁 | 133,888(同家族) | 0.8646 | 0.8793(tbatch 训练) | — |
| stale_batch | 147,840(emb64/static-on) | 0.9649 | 0.9335(stale 训练) | **0.6014** |

- **评分污染(第三层)**:同一 147K 架构,保真 val 0.6194(tbatch)/0.6284(serial)vs stale 0.9649 → **+0.345**;stale 自报 test 0.9335 vs 真实串行 0.6014 → **+0.332**。val/test 污染幅度一致 = 同一机制(stale read 压低动态嵌入 → static=on 被系统性高估),贯穿搜索评分与最终测试。
- **选择洗牌 + 掉分(第一层)**:stale 选的 147K 架构保真 val 仅 0.62(排不进前 8),真实串行 test 0.6014 vs 基线 0.8561 = **-0.2547**。原预言"选错架构 / 掉分"成立,且多一层"坏选择带完美成绩单"(val 0.9649 + 自报 test 0.9335)。
- **补测分支**:`ablation/reeval-fixed-arch`(基于 origin/refactored + `reeval_fixed_arch.py`)。133K 探针复现 0.856121275963994(bit-identical)→ 补测路径 = 基线协议确认。
- **陷阱教训**:两臂日志写 "Serial training",但 `_train_and_eval` 的 batch_mode 取自 `base_config`(各臂自己的模式)——最终测试实为 stale/tbatch 训练。判读必须落到代码,不能只看日志文案。
