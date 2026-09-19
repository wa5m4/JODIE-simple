# PipeTGL 论文实验表格(待填数据版)


> run 一完成就立刻填入;确凿的直接填,不确凿的照填 + 标注(★/⚠/✗/📝),不因不确定留空。填表口径见文末「填表说明」。

## 表格 1:数据集统计与大规模论证

| 数据集 | #交互 | #用户 | #物品 | 时间跨度 | 特征维 | 角色 |
|---|---|---|---|---|---|---|
| MOOC | 411,749 | 7,047 | 97 | 30 天 | — | 主数据集(终点) |
| Wikipedia | 157,474 | 8,227 | 1,000 | 30 天 | 172 | 辅助 |
| Reddit | 672,447 | 10,000 | 984 | 30 天 | 172 | 辅助 |
| LastFM | 1,293,103 | 980 | 1,000 | 5 年 | — | 辅助(最大真实规模) |

**大规模论证**(正文一段,待写):
- 计算量口径:1 次 NAS trial = 完整训练 + 验证 + 测试,MOOC 50 trials ≈ 2000 万事件级前向;
- 负载倾斜:每分区新用户极差 1.83×(20K)→ 4.09×(100K)→ 114.5×(411K),事件数严格均匀;
- 与静态 GNN 基准/PageRank 尺子的对比论述。

## 表格 2:总体性能对比(主表)

> 对应 DUET 表 1。时间与吞吐均为端到端;Speedup = 同 cell 下 serial 为分母。

| Cell(数据集·规模·卡数·空间) | Method | Test MRR | Recall@10 | E2E Time (s) | E2E Throughput (trials/h) | Speedup |
|---|---|---|---|---|---|---|
| MOOC-B · 20K · 2卡 · rnn 50t | JODIE 串行(serial) | 0.7000 | 0.9150 | 10864.3 | 16.57 | 1.0× |
| | DDP 并行(dp) | 0.66866 | 0.91377 | 14003.6 ★ | 12.85 ★ | 0.78× ★ |
| | 并行调度(smart_sync) | 0.7000 | 0.9150 | 13211.8 ★ | 13.62 ★ | 0.82× ★ |
| | PipeTGL-P(naive, ours) | 0.7000 | 0.9150 | 9928.8 | 18.13 | 1.09× |
| | PipeTGL(naive_async, ours) | 0.7000 | 0.9150 | 11266.6 | 15.98 | 0.96× |
| MOOC-C · 20K · 3卡 · mixed 50t | JODIE 串行(serial) | 0.7000 | 0.9150 | 10610.6 ★ | 16.96 ★ | 1.0× |
| | DDP 并行(dp) | 0.71938 | 0.96601 | 23213.7 ★⚠ | 7.75 ★⚠ | 0.46× ★⚠ |
| | 并行调度(smart_sync) | 0.7000 | 0.9150 | 5203.6 ⚠ | 34.59 ⚠ | 2.04× ⚠ |
| | PipeTGL-P(naive, ours) | 0.7000 | 0.9150 | 18464.1 ★⚠ | 9.75 ★⚠ | 0.57× ★⚠ |
| MOOC-D · 100K · 3卡 · rnn 12t | JODIE 串行(serial) | 0.5535 | 0.7423 | 7188.0 | 6.01 | 1.0× |
| | DDP 并行(dp) | 0.2240 | 0.4899 | 5919.5 | 7.30 | 1.21× |
| | 并行调度(smart_sync) | 0.5535 | 0.7423 | 4247.2 | 10.17 | 1.69× |
| | PipeTGL-P(naive, ours) | 0.5535 | 0.7423 | 4652.8 | 9.28 | 1.54× |
| | PipeTGL(naive_async, ours) | 0.5535 | 0.7423 | 3660.9 | 11.80 | 1.96× |
| MOOC-E · 100K · 2卡 · rnn 12t | JODIE 串行(serial) | 0.5535 | 0.7423 | 6945.9 | 6.22 | 1.0× |
| | DDP 并行(dp) | 0.2240 | 0.4899 | 5668.5 | 7.62 | 1.23× |
| | 并行调度(smart_sync) | 0.5535 | 0.7423 | 5071.9 | 8.52 | 1.37× |
| | PipeTGL-P(naive, ours) | 0.5535 | 0.7423 | 5718.8 | 7.55 | 1.21× |
| | PipeTGL(naive_async, ours) | 0.5535 | 0.7423 | 5012.1 | 8.62 | 1.39× |
| MOOC-F · 200K · 3卡 · rnn 12t | JODIE 串行(serial) | | | | | 1.0× |
| | DDP 并行(dp) | | | | | |
| | 并行调度(smart_sync) | | | | | |
| | PipeTGL-P(naive, ours) | | | | | |
| | PipeTGL(naive_async, ours) | | | | | |
| MOOC-G · 100K · 3卡 · mixed 12t | JODIE 串行(serial) | 0.5074 | 0.8122 | 6577.0 | 6.57 | 1.0× |
| | 并行调度(smart_sync) | 0.5074 | 0.8122 | 4180.6 | 10.33 | 1.57× |
| | PipeTGL-P(naive, ours) | 0.5074 | 0.8122 | 4388.9 | 9.84 | 1.50× |
| | PipeTGL(naive_async, ours) | 0.5074 | 0.8122 | 3230.9 | 13.37 | 2.04× |
| MOOC-H · 200K · 3卡 · mixed 12t | JODIE 串行(serial) | | | | | 1.0× |
| | 并行调度(smart_sync) | | | | | |
| | PipeTGL-P(naive, ours) | | | | | |
| | PipeTGL(naive_async, ours) | | | | | |
| MOOC-full · 411K · 3卡 · rnn 12t | JODIE 串行(serial) | | | | | 1.0× |
| | DDP 并行(dp) | | | | | |
| | 并行调度(smart_sync) | | | | | |
| | PipeTGL-P(naive, ours) | | | | | |
| | PipeTGL(naive_async, ours) | | | | | |
| Wiki-W1 · 100K · 3卡 · rnn 12t | JODIE 串行(serial) | | | | | 1.0× |
| | DDP 并行(dp) | | | | | |
| | 并行调度(smart_sync) | | | | | |
| | PipeTGL-P(naive, ours) | | | | | |
| | PipeTGL(naive_async, ours) | | | | | |
| Wiki-W2 · 100K · 3卡 · mixed 12t | JODIE 串行(serial) | | | | | 1.0× |
| | DDP 并行(dp) | | | | | |
| | 并行调度(smart_sync) | | | | | |
| | PipeTGL-P(naive, ours) | | | | | |
| | PipeTGL(naive_async, ours) | | | | | |
| Reddit-R1 · 100K · 3卡 · rnn 12t | JODIE 串行(serial) | | | | | 1.0× |
| | DDP 并行(dp) | | | | | |
| | 并行调度(smart_sync) | | | | | |
| | PipeTGL-P(naive, ours) | | | | | |
| | PipeTGL(naive_async, ours) | | | | | |
| LastFM-L1 · 100K · 3卡 · rnn 12t | JODIE 串行(serial) | | | | | 1.0× |
| | DDP 并行(dp) | | | | | |
| | 并行调度(smart_sync) | | | | | |
| | PipeTGL-P(naive, ours) | | | | | |
| | PipeTGL(naive_async, ours) | | | | | |

**头条结论(部分,2026-09-19)**:已完成 B/C/D/E/G 中 PipeTGL(naive_async)对 JODIE 串行(serial)最大加速比 **2.04×(MOOC-G)**;对并行调度(smart_sync)反超 **1.01–1.29×**(B/D/E/G 四格,随卡数增长:2 卡 1.01×、3 卡 1.16–1.29×;待 F/H 收口)。

**已填验证说明(2026-09-17)**:MOOC-B 三臂来自保真金丝雀 run(serial 20260915_211715 + 20260916_002427 两次复跑、naive 20260916_034440、naive_async 20260916_063131),三臂 val/test **位级一致**(0.7655034809596591 / 0.6999869339656702),计时为洪水前干净负载(完成于 09-16 00:18–09:39,节奏正常)。
⚠️ 口径与旧表 4 不同:本表 = 50 coarse trials ÷ 总耗时(含 rerank 8×5 epochs),旧表 4 为旧配置(1 epoch)数字,两者**不可混用**;phase5 链条将以新口径重跑全部 cell。
⚠️ B 上 naive_async(15.98)低于 naive(18.13),但差异在重复方差内(serial 两次复跑差 10%),异步增益在 B 上不显著——待干净补跑确认,如照实填入则 B 行异步头条不成立,主头条落 D/F。

## 表格 3:保真矩阵(「快,但答案不变」核心表)

> 对应 DUET 表 2(蒸馏效果)的同位替换:我们展示的「效果」是各臂选出架构与 serial 位级一致。
> 格内填:✓(与 JODIE 串行(serial)位级一致)/ ✗(不一致)+ 选出的参数规模;/ 表示该臂不适用此 cell。

| Cell | JODIE 串行(serial) 选出架构(参数) | test | PipeTGL-P(naive, ours) | 并行调度(smart_sync) | DDP 并行(dp) | PipeTGL(naive_async, ours) |
|---|---|---|---|---|---|---|
| MOOC-B | 402,176(linear/on/on) | 0.7000 | ✓(402,176,位级一致) | ✓(402,176,位级一致) | ✗(133,888) | ✓(402,176,位级一致) |
| MOOC-C | 402,176(linear/on/on) | 0.7000 | ✓(402,176,位级一致) | ✓(402,176,位级一致) | ✓(402,176,位级一致) | / |
| MOOC-D | 34,176(linear/off/on) | 0.5535 | ✓(34,176,位级一致) | ✓(34,176,位级一致) | ✗(464,192) | ✓(34,176,位级一致) |
| MOOC-E | 34,176(linear/off/on) | 0.5535 | ✓(34,176,位级一致) | ✓(34,176,位级一致) | ✗(464,192) | ✓(34,176,位级一致) |
| MOOC-F | | | | | | |
| MOOC-G | 34,176(off/off/off) | 0.5074 | ✓(34,176,位级一致) | ✓(34,176,位级一致) | / | ✓(34,176,位级一致) |
| MOOC-H | | | | | / | |
| MOOC-full | | | | | | |
| Wiki-W1 | | | | | | |
| Wiki-W2 | | | | | | |
| Reddit-R1 | | | | | | |
| LastFM-L1 | | | | | | |

**保真结论(部分,2026-09-19)**:已收口 cell 中 PipeTGL(naive_async)与 JODIE 串行(serial) **4/4 位级一致**(B/D/E/G;C 未跑 naive_async);DDP 并行(dp)在 **rnn_only 空间 3/3 分叉**(B 133,888、D/E 464,192——D/E 分叉架构完全同构,可复现签名)、在 mixed 空间 1/1 一致(C 402,176)——保真不是并行化的默认结果,是协议保证的。
**已验证(2026-09-17/18)**:MOOC-B 行收口——serial 选出 402,176,**naive/smart_sync/naive_async 均位级一致**(架构与 val/test 逐位相同,val 0.7655034809596591 / test 0.6999869339656702);**dp 判决 ✗**:修复后复测(20260917_101853)选出 133,888(val 0.76633/test 0.66866),与修复前几乎一致 → 种子纪律+负采样两层修复后仍分叉,归因微批梯度平均的训练数学差异,dp 按「近似后端」叙事。**结论:B 上 4/4 协议化策略位级保真,唯一分叉=无协议的数据并行。**
**C-serial(20260918_022954)**:选 402,176(test 0.7000 同 B)。污染史:首跑(23:19)pmon 判污染改名;重跑(02:29)**亦被 pmon 判污染**(链不再三跑,留人工判读)。但计时 10610.6s vs B-serial 干净 10864.3s **偏差 2.3% ≤ 15% 判据 → 实证通过,保留入表**(污染判罚为保守护栏,偏差判据为入表口径);列入夜间补跑候选池,填表终稿时定夺。
**C-naive(完成,★⚠)**:首跑 05:27 污染改名;重跑 09:41→14:49 完成,期间 10:06 又报活跃污染(PID 559760 sm>15%),按 09-18 政策不再三跑。**保真 ✓**:选出 402,176,test 0.6999869339656702 / recall@10 0.9150212446888278 与 C-serial **位级一致**;计时 18464.1s(9.75 trials/h)受污染+洪水双重影响,★⚠ 入表,列入终检。
**C-dp(完成,★⚠)**:21:37 完成,E2E 23213.7s(7.75 trials/h),全程洪水(load 120-136),链在 run 结束写入污染标注。**保真 ✓**:选出 402,176(val 0.72725 / test 0.71938 / recall@10 0.96601),与 C-serial 参数规模位级一致(同架构不同权重,符合判据「不要求分数相等」)——与 B-dp 分叉(133,888)形成对照,dp 分叉非必然。计时较 B-dp(14003.6s)慢 66%(mixed 空间 + 洪水),★⚠ 入表,列入终检。
**C-smart_sync(完成,⚠)**:23:05 完成,E2E 5203.6s(34.59 trials/h,2.04×),C 行最快臂。**保真 ✓**:402,176,test 0.6999869339656702 与 C-serial 位级一致。计时:21:38 启动即带污染警报(PID 3409138),21:50 起 dongyu 暂停、load 33-39,约 85% 时长在干净区,⚠ 入表,列入终检。**C 行收口:serial/naive/dp/smart_sync 四臂全选 402,176,4/4 位级一致。**
**G 行收口(2026-09-19,四臂全干净)**:G/serial 6577.0s → G/naive 4388.9s(1.50×)→ G/naive_async 3230.9s(**2.04×**,G 行最快)→ G/smart_sync 4180.6s(1.57×),全程无污染警报(夜间负载,他人作业仅占显存 0% util)。四臂全选 **34,176(off/off/off)**,test 0.5074352485944837 / recall@10 0.8122 **4/4 位级一致**。naive_async 对 smart_sync 反超 1.29×。
**D 行收口(2026-09-19,五臂全干净)**:D/serial 7188.0s → D/naive 4652.8s(1.54×)→ D/naive_async 3660.9s(1.96×)→ D/smart_sync 4247.2s(1.69×)→ D/dp 5919.5s(1.21×),全程无污染警报。serial/naive/naive_async/smart_sync 四臂全选 **34,176(linear/off/on)** 位级一致(test 0.5534986661839446 / recall@10 0.74235);**dp ✗ 分叉**:464,192(test 0.2240)——与 B-dp(133,888)同构,rnn_only 空间微批梯度平均再次破坏逐事件更新(C-mixed 一致的反例仍成立)。naive_async 对 smart_sync 反超 1.16×。
**E 行收口(2026-09-19,五臂全干净)**:E/serial 6945.9s → E/naive 5718.8s(1.21×)→ E/naive_async 5012.1s(1.39×)→ E/smart_sync 5071.9s(1.37×)→ E/dp 5668.5s(1.23×),全程无污染警报。serial/naive/naive_async/smart_sync 四臂全选 **34,176(linear/off/on)** 位级一致;**dp ✗ 分叉**:464,192——与 D-dp **完全同架构**(rnn_only 100K 同空间同种子),dp 分叉呈可复现签名,非随机噪声。2 卡敏感:naive 1.21×(D 3 卡 1.54×),naive_async 对 smart_sync 仅 1.01×(3 卡 1.16-1.29×)——异步反超随卡数增长。
**终检补跑队列(2026-09-19 用户拍板)**:B/C 行带 ★/⚠ 的非 serial 臂共 5 个——B-dp ★、B-smart_sync ★、C-naive ★⚠、C-dp ★⚠、C-smart_sync ⚠——已排入 RUNS_P6,phase5 链结束后以 chain_all.sh phase6 重跑;完成后用干净计时替换表中 ★/⚠ 数字并标注替换来源。C-serial ★ 不重跑(偏差 2.3% 已过判据);B-serial 干净无需重跑。保真结论不受重跑影响(dp 分叉是结构性微批平均,重跑只修计时)。

## 表格 4:消融实验(技术级 + RAW 依赖)

> 对应 DUET 表 3。两层:技术级 = 两个贡献(流水线结构、异步引擎)逐项加减;
> RAW 依赖 = 朴素替代实现的对照(证明「协议是设计出来的」而非修 bug)。
> 协议组件二阶拆解(f1/f2/f3)已裁(2026-09-17):dp 臂在表 3 的分叉即「无协议」对照,二阶拆解与叙事重复。

### 4a · 技术级消融(主链式:JODIE 串行 → +流水线(PipeTGL-P) → +异步(PipeTGL))

> 标准消融顺序 base → base+A → base+A+B:每行都是在前一行基础上累加一个组件,加速比逐行递增即消融成立。

| Cell | Variant | Test MRR | E2E Throughput (trials/h) | Speedup | 备注 |
|---|---|---|---|---|---|
| MOOC-D | base(JODIE 串行, serial) | 0.5535 | 6.01 | 1.0× | |
| | +流水线(PipeTGL-P, naive) | 0.5535 | 9.28 | 1.54× | |
| | +异步(PipeTGL, naive_async) | 0.5535 | 11.80 | 1.96× | |
| Wiki-W1 | base(JODIE 串行, serial) | | | | 1.0× |
| | +流水线(PipeTGL-P, naive) | | | | |
| | +异步(PipeTGL, naive_async) | | | | |

### 4b · RAW 依赖(tbatch / stale_batch)

| 处理方式 | 选出架构 | 自报 test | 中立复检 | 备注 |
|---|---|---|---|---|
| 串行(正确基准) | | | | |
| t-Batch(分批+冲突消解) | | | | |
| stale_batch(朴素分批) | | | | |

## 表格 5:时间分解

> 对应 DUET 表 4(生命周期)的改造:生命周期不适用于 NAS 搜索系统,我们的对应物 = 搜索循环的阶段耗时分解。

| Cell | Method | 数据准备 (s) | 训练 (s) | 评估 (s) | 调度与同步 (s) | 序列化 (s) | 合计 (s) | 训练占比 |
|---|---|---|---|---|---|---|---|---|
| MOOC-B | JODIE 串行(serial) | | | | | | | |
| | DDP 并行(dp) | | | | | | | |
| | PipeTGL-P(naive, ours) | | | | | | | |
| | PipeTGL(naive_async, ours) | | | | | | | |
| MOOC-D | JODIE 串行(serial) | | | | | | | |
| | DDP 并行(dp) | | | | | | | |
| | PipeTGL-P(naive, ours) | | | | | | | |
| | PipeTGL(naive_async, ours) | | | | | | | |

**配图(待画)**:两 cell 的阶段占比堆叠条形图。

## 表格 6:超参敏感性

| 旋钮 | 取值 | E2E Throughput (trials/h) | 选出架构 | 备注 |
|---|---|---|---|---|
| 流水线段数(naive) | 2 ★(默认) | | | |
| | 3 | | | |
| | 4 | | | |
| 每段训练 worker 数(naive) | 1 | | | |
| | 2 ★(默认) | | | |
| 微批大小(naive) | 16 | | | |
| | 32 ★(默认) | | | |
| | 64 | | | |
| DP worker 数(dp) | 2 | | | |
| | 3 ★(默认) | | | |
| | 4 | | | |
| 分区大小 | 0.5× | | | |
| | 1× ★(默认) | | | |
| | 2× | | | |

## 表格 7:可扩展性

| 维度 | 配置 | Method | E2E Throughput (trials/h) | Speedup | 备注 |
|---|---|---|---|---|---|
| 数据规模 | 20K(MC B) | PipeTGL-P(naive, ours) | 18.13 | 1.09× | |
| | 100K(MC D) | PipeTGL-P(naive, ours) | 9.28 | 1.54× | |
| | 200K(MC F) | PipeTGL-P(naive, ours) | | | |
| | 411K(MC full) | PipeTGL-P(naive, ours) | | | |
| 资源数 | 2 卡(MC E) | PipeTGL-P(naive, ours) | 7.55 | 1.21× | |
| | 3 卡(MC D) | PipeTGL-P(naive, ours) | 9.28 | 1.54× | |
| trials 数 | 12t(MC D) | PipeTGL-P(naive, ours) | 9.28 | 1.54× | |
| | 50t(MC B) | PipeTGL-P(naive, ours) | 18.13 | 1.09× | |

**扩展性结论(待填)**:规模增大____×时,naive 加速比从____变化到____。

## 表格 8:环境完备性(两环境交叉验证)

| 环境 | 机器配置 | Cell | Method | E2E Throughput (trials/h) | Speedup | 备注 |
|---|---|---|---|---|---|---|
| A | sduu-SYS-420GP-TNR, 8×RTX 4090, 48 核 | MOOC-B | JODIE 串行(serial) | 16.57 | 1.0× | 主环境 |
| | | | PipeTGL-P(naive, ours) | 18.13 | 1.09× | |
| | | | PipeTGL(naive_async, ours) | 15.98 | 0.96× | |
| | | MOOC-D | JODIE 串行(serial) | 6.01 | 1.0× | |
| | | | PipeTGL-P(naive, ours) | 9.28 | 1.54× | |
| | | | PipeTGL(naive_async, ours) | 11.80 | 1.96× | |
| B | ____(待定:实验室其他机器) | MOOC-B | JODIE 串行(serial) | | 1.0× | |
| | | | PipeTGL-P(naive, ours) | | | |
| | | | PipeTGL(naive_async, ours) | | | |
| | | MOOC-D | JODIE 串行(serial) | | 1.0× | |
| | | | PipeTGL-P(naive, ours) | | | |
| | | | PipeTGL(naive_async, ours) | | | |

---

## 填表说明

0. **填表节奏(2026-09-17 用户裁定)**:实验以本表为唯一填表目标,链条按本表空白格顺序跑;每个 run 一完成就立刻填数——确凿的直接填,**不确凿的照填 + 标注,不留空等确认**。标注体系:★ 洪水期计时(校验中/待补跑)、⚠ 污染(已标注,不重跑——2026-09-18 裁定取消污染自动重跑,终检统一补跑)、✗ 保真分叉(判据见第 5 条)、📝 其他存疑。
1. **数据来源**:每个 run 完成后,`results/<时间戳>/comparison.json` 提供分数与选出架构;`outputs/*/timing_log.csv` 提供每 trial 耗时;throughput = coarse trials / 总耗时(trials/h 口径)。
2. **Speedup**:分母 = 同 cell 的 serial 数字,不跨 cell 比较。
3. **洪水期标注**:负载 > 96 期间的 run,分数/架构照填;计时照记并加 ★。填表前按偏差判据校验(见 positioning/FLOOD_TIMING_NOTE_20260917.md):有干净同构对照的(20K)偏差 ≤ 15% 直接入表;无直接对照的(100K/200K)按三方校验(per-trial 平稳性 / 跨臂自洽 / 成本外推),异常才夜间补跑。入表后 ★ 保留,脚注注明校验方式与偏差。
4. **G/H 无 dp 臂、C 无 naive_async 臂**(矩阵设计),对应格子填 /。
5. **保真判定**:✓ = 与 serial 选出架构位级一致(同一签名、同一参数规模),不要求分数相等。
6. **Baseline 引用映射(2026-09-18 用户裁定,同日修正)**:Method 列 = 论文名;对应已发表系统的挂引用,我们自己的系统变体标 ours;括号内为内部配置名(run 数据/日志仍以内部名识别)。**Baseline**:JODIE 串行 = JODIE(KDD'19);DDP 并行 = PyTorch Distributed(VLDB'20);并行调度 = Ray Tune(arXiv'18)/BOHB(ICML'18)的 master-worker 并行评估范式(smart 智慧分配证伪后退化的架构并行)。**本系统(ours)**:PipeTGL-P = 流水线结构(同步变体);PipeTGL = 流水线+异步引擎(Full)。表格 4a 的 base / +流水线 / +异步 依次对应 JODIE 串行 / PipeTGL-P / PipeTGL。TGL(VLDB'22)/DistTGL(SC'23)/Hogwild!(NeurIPS'11)在正文相关工作段引用,不进表格 Method 列。
