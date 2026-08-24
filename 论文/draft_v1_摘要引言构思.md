# DepTGL 论文:摘要与引言构思 (v1)

> 依据:导师《科研过程-第三阶段》+ DynaHB 论文结构
> 状态:构思草稿,数字为占位符,待实验补全

---

## 一、模板拆解:DynaHB 是怎么写的

### 摘要 = 4 句公式

| # | 角色 | DynaHB 原句大意 | 回答导师的哪个问题 |
|---|------|----------------|------------------|
| 1 | **背景+动机** | DGNN 效果好,但训练开销大到大图根本训不动 | 动机(现有方法不行) |
| 2 | **现有方案的问题** | 现有分布式框架刚起步,有通信瓶颈/负载不均/显存溢出三个挑战 | 挑战 |
| 3 | **我们的方法** | 我们提出 DynaHB:顶点缓存→少通信;负载感知划分→负载均衡;hybrid batch→省时省显存;RL 调 batch + 流水线生成器→降成本 | 贡献(技术) |
| 4 | **实验结果** | 实验表明最高 93×、平均 8.06× 加速 | 贡献(实验) |

一句话 = 一个作用,绝不混用。

### 引言 = 8 段公式

| # | 段落 | DynaHB 实际内容 |
|---|------|----------------|
| 1 | 背景 | 静态 GNN 成功 → 动态图出现 → DGNN 诞生 |
| 2 | 为什么 DGNN 训练比 GNN 难(3 条具体理由) | 多个快照、RNN 跨快照通信、同步更复杂 |
| 3 | 相关工作扫描(单机 → 分布式,指出各自不行在哪) | PyGT/CacheG/PiPAD 受单机限制;ESDG/DGC/BLAD/DynaGraph 起步阶段 |
| 4 | **三个挑战**(Challenge I/II/III,每条带具体机理) | 通信开销 / 大规模低效 / 负载倾斜 |
| 5 | 我们的方案(逐条对应挑战) | "To address the above challenges, we propose..." |
| 6 | 贡献列表(3 技术 + 1 实验) | bullet 列表 |
| 7 | 论文结构 | Section 2 相关工作... |
| 8 | (导师额外要求)**Case study 带具体数值** | 例:划分方案通信 6→3 |

**导师硬性要求对照表**:
- 动机/挑战/贡献三要素 → 对应摘要公式的 1-2 句 + 3-4 句
- 三个技术贡献 + 一个实验贡献 → 贡献列表必须 4 条
- 引言必须有 case study + 具体数值
- 用词往数据管理靠:temporal data dependency / data caching / read-after-write (RAW) 这类词
- 把评审当大同行:RNN 只当作"维护时序状态的模块",不讲内部结构

---

## 二、DepTGL 的故事映射

### 动机(为什么做)

1. 时序交互数据无处不在(推荐、链接预测),JODIE 类时序 GNN 是 SOTA。
2. 但架构设计空间大(聚合函数、时间衰减、记忆单元、时间投影……),手工调参费时费力 → 需要 NAS。
3. 现有 NAS 框架都是给静态 GNN 的(GraphNAS 等),**静态图训练假设样本独立同分布,时序训练不是**:
   - 每个交互都会改写用户和物品的嵌入 → 交互流上存在**时序数据依赖**(写后读,read-after-write (RAW))
   - 现有 NAS 无法把时序架构评估并行化——并行就会打破依赖 → 评分失真 → 选错架构

### 挑战(难在哪)——3 个

| 挑战 | 内容 | 数据管理话术 |
|------|------|-------------|
| **Challenge I:时序数据依赖** | 嵌入是递归更新的,交互流天然串行。朴素并行(乱序/冲突批)会破坏写后读依赖,数学结果与串行不同 | temporal data dependency / read-after-write (RAW) conflicts |
| **Challenge II:评分保真度** | NAS 要求"同一架构在哪跑都得同一个分数"。跨进程/跨阶段迁移模型与优化器状态时,种子、RNG、控制器更新节奏的微小偏差会累积成**系统性偏差**(修前:Pipeline 系统性高估 static=on 架构),最终选错架构 | evaluation fidelity / state migration |
| **Challenge III:负载倾斜** | 时序分区的成本极不均匀(事件数、新用户/物品数差异大),流水线各阶段负载失衡,worker 空转 | skewed workloads / cost model / load balancing |

### 贡献(我们怎么解决)——3 技术 + 1 实验

| 贡献 | 类型 | 内容 |
|------|------|------|
| C1 DepTGL 框架 | 系统 | 面向时序 GNN 的 NAS 框架:JODIE 风格搜索空间 + REINFORCE 控制器 + 三种并行执行策略(数据并行 / 流水线 / 异步架构并行)+ 代价模型自动配置 |
| C2 评估保真度(旗舰) | 算法 | 定位并消除并行执行中的评分偏差:per-trial 种子纪律、跨阶段 RNG 保存/恢复、离策略(off-policy)logprob 重算 → 各执行策略选出与串行一致的最优架构 |
| C3 代价模型配置(候选) | 算法 | CostModel(事件数+新节点数估计分区成本)+ DP 分区分组 + Lagrange 最优 worker 分配;把 stage 边界状态迁移开销编码进代价模型,使配置器正确倾向少 stage。备选:MemShare 热点感知状态合并 |
| C4 实验 | 实验 | 在 X 个公开数据集 + 1 个合成数据集上,对比 Y 个模型家族:异步架构并行最高 2.3× 加速,搜索质量与串行一致(待扩展) |

> ⚠️ **原创性雷区(2026-08-14)**:t-Batch 出自 JODIE 论文,TGN 窗口批处理出自 TGN 论文——不能写"we propose t-Batch",只能写"集成/适配到 NAS 评估循环"。注意力聚合 + 时间衰减是 TGAT 风格。DepTGL 真正原创的:NAS 框架本身、保真度机制、代价模型、MemShare。

**为什么 C3 是旗舰贡献**(来自项目学习日志的结论):排障链(ATTEMPTS_LOG / FINAL_REPORT)揭示的根因——per-trial 种子缺失、冗余 build_model 污染 RNG、批量 vs 逐 trial 控制器更新——是"并行 NAS 评分保真度"这个真问题的完整答案,不是工程事故。修前 Pipeline Naive test_score 0.7000、选 402K 错误架构;修后 0.8561、与 Serial 选的 133K 架构完全一致。

**备选贡献**(导师说 C2/C3 必须有算法创新,可替换讨论):ConfigOptimizer 的代价模型 + DP 分组 + Lagrange 分配(更 DB 味);MemShare 热点感知状态合并。

**两个候选的现状核查(2026-08-14)**:① 代价模型——代码已接线(`--enable-auto-pipeline-config`),但赢家实验 Smart 是手动关掉自动配置跑的(run_all.py: `SMART_ENABLE_AUTO_PIPELINE_CONFIG = False`,手动 1 stage × 3 workers),且 1 stage 时 DP 分组/Lagrange 分配全部退化;当 C3 必须先补"配置器 vs 手动最优"对比实验。② MemShare——只有类定义,无任何入口调用(死代码),ISSUES.md 5.3 甚至提议弃用,还缺 tbatch/tgn 模式支持;当 C3 需先补功能再补实验,暂出局。

---

## 三、摘要初稿 v1(英文,仿 DynaHB 4 句公式)

> Temporal Graph Neural Networks (TGNNs) such as JODIE have achieved
> state-of-the-art performance on temporal interaction prediction tasks.
> However, designing TGNN architectures involves labor-intensive manual
> tuning over a large hyperparameter space, and existing neural
> architecture search (NAS) frameworks target static GNNs, whose training
> assumes independent and identically distributed samples and thus ignores
> the temporal data dependencies inherent in interaction streams.
> Naively parallelizing TGNN training breaks the read-after-write (RAW)
> dependencies among consecutive interactions, producing biased
> architecture scores that lead NAS to select inferior architectures.
> We introduce \texttt{DepTGL}, an NAS framework for TGNNs that supports
> serial, data-parallel, and pipeline execution backends. DepTGL features
> conflict-free t-Batch batching that enables lossless parallel training,
> preserves evaluation fidelity across backends by means of per-trial seed
> discipline, RNG-preserving state migration, and off-policy controller
> updates, and balances pipeline stages using a cost-model-based
> configurator. Extensive experiments on X datasets with Y model families
> show that DepTGL achieves up to 2.3x speedup over serial NAS
> without degrading search accuracy.

**句号核对**:句1=背景+动机,句2=现有方案问题,句3=挑战机理,句4=方法(三项技术),句5=实验。比 DynaHB 多一句,因为导师要求把"现有方法不好"讲透;正式写时可合并 2、3。

---

### 摘要 v2(用户改写 + 批改后,2026-08-14)

> Temporal Graph Neural Networks (TGNNs) such as JODIE have achieved
> state-of-the-art performance on temporal interaction prediction tasks.
> However, designing TGNN architectures requires labor-intensive manual
> tuning over a large hyperparameter space, and existing neural
> architecture search (NAS) frameworks target static GNNs, whose training
> assumes i.i.d. samples and thus cannot handle the temporal data
> dependencies inherent in interaction streams. Searching over hundreds
> of candidate architectures demands parallel execution, yet naively
> parallelizing TGNN training breaks the read-after-write (RAW) dependencies
> among consecutive interactions, yielding biased architecture scores
> that cause NAS to select inferior architectures. We introduce
> \texttt{DepTGL}, an NAS framework for TGNNs that supports serial,
> data-parallel, and pipeline execution. DepTGL features conflict-free
> batching (t-Batch), which eliminates read-after-write (RAW) conflicts on the
> temporal memory and thereby enables lossless parallel training;
> preserves evaluation fidelity across backends via per-trial seed
> discipline, RNG-preserving state migration, and off-policy controller
> updates; and balances pipeline stages using a cost-model-based
> configurator. Extensive experiments on X datasets with Y model
> families show that DepTGL achieves up to 2.3× speedup over serial NAS
> without degrading search accuracy.

**v2 相对用户版改了 5 处**:
1. 挑战具体化:temporal **data** dependencies(数据管理话术);"ignore" → "cannot handle"(更强的 gap)
2. 补因果桥梁:"Searching over hundreds of candidate architectures demands parallel execution, yet ..."(先说必须并行,再说并行会坏)
3. read-after-write (RAW) 点名机理,与方法的"eliminates read-after-write (RAW) conflicts"前后呼应
4. "statistically equivalent accuracy" → "identical ... with the same accuracy"(数据是四后端完全相等 0.8561,不是统计等价;不要假谦虚)
5. "conflict-free t-Batch batching" 去冗余,补"eliminates read-after-write (RAW) conflicts on the temporal memory"(方法讲清为什么 lossless)
6. **术语修正(用户指出,2026-08-14)**:依赖类型是 **RAW(read-after-write,真依赖)**——后面的交互要读前面交互写入的嵌入,不是 WAR(反依赖)。中文"写后读"= 读在写后 = RAW,但英文不能硬翻成 write-after-read。系统/体系结构背景的审稿人对 RAW/WAR/WAW 极其敏感,用错必被质疑。附带事实:t-Batch 的"每批每个节点至多出现一次"同时消除了批次内的 RAW(无跨交互读依赖)与 WAW(无同节点双写);邻居读的精确语义留到方法章节展开。
7. **"identical ... with the same accuracy" 降级为 "without degrading search accuracy"(用户质疑,2026-08-14)**:修 bug 后的数据里,四策略 val_score 本身并不相等(Serial 0.8648 / DP 0.7521 / Naive 0.8267 / Smart 0.8267),只是最终都选了同一架构、test 精度相同。"identical" 只能作为"本次 MOOC 实验的事实"写进引言 case study,不能作为普适断言写进摘要——换数据集或换种子,分数接近的两个架构可能互换排名,是 NAS 正常现象;声称 identical 会被审稿人用逐数据集表格攻击。另外"same accuracy"在选出同一架构时是循环论证,信息量为零;真正要声明的是"加速 2.3× 且搜索质量不降"。DynaHB 摘要也只报加速比,准确率一致性放在引言和实验章节。多数据集跑完后:若逐数据集都一致,可升级为 "the same architecture on all tested datasets";否则用 "test accuracy within ε of serial NAS"。

**仍待讨论**:3 个挑战目前只出现 2 个(时序依赖、评分失真),负载倾斜只在方法句里。若想严格对齐 DynaHB"挑战点名→方法对应",可在第 2/3 句补"and temporal workloads are highly skewed"。另外 "off-policy" 对不懂 RL 的评审是黑话,定稿时考虑换成 "policy updates robust to asynchronous execution" 或保留(评审可跳过不深究)。

---

## 四、引言提纲 v1(英文,仿 DynaHB 8 段公式)

**段 1 — 背景**:时序交互数据无处不在(推荐、社交、知识图谱);JODIE、TGN 等时序 GNN 在交互预测上取得 SOTA。

**段 2 — 为什么难(NAS 视角)**:架构设计空间大,手工调参费时;NAS 自动化了搜索,但现有 NAS 面向静态 GNN:静态训练中样本是 i.i.d. 的,可以随意并行;而时序训练中每个交互都会改写用户与物品的嵌入,**后续交互依赖前面的写入**——训练天然是串行的。

**段 3 — 相关工作扫描**(三段式,和 DynaHB 一样先扫单机再扫分布式,指出各自缺陷):
- TGNN 训练框架:PyGT / CacheG / PiPAD(单机,资源受限)→ DynaHB / ESDG(分布式,但目标是"训练一个模型"而非"搜索架构")
- 面向静态 GNN 的 NAS:GraphNAS 等(假设 i.i.d.,不处理时序依赖)
- → 空白:没有既并行、又保真的时序 GNN NAS 框架

**样板对照**(DynaHB 引言段 3 原文 + 结构拆解):
- 组式结构:"提名若干系统 → 肯定它们做了什么 → However/but 转折说不够";转折句勾住下一段("face three major challenges" → 下一段就是 Challenge I/II/III)
- DepTGL 版三组:① TGNN 训练框架(单机+分布式;转折:它们是"训练给定模型",NAS 是"评估成百上千候选架构",问题不同)→ ② 静态 GNN 的 NAS(GraphNAS 等;转折:假设 i.i.d.,忽略 RAW 时序依赖)→ ③ 空白声明句:"没有面向时序 GNN 的 NAS 框架"——落点为段 4 case study 铺路

### 引言段 3 初稿(2026-08-14,代写,待用户消化)

> With the growing interest in temporal graph learning, several frameworks
> have been developed for TGNN training. Single-machine frameworks such as
> PyGT [ref], CacheG [ref], and PiPAD [ref] optimize caching and reuse of
> intermediate embeddings, pipelined parallelism, and the like, but they are
> constrained by the limited resources of a single machine. Distributed
> frameworks such as ESDG [ref] and DynaHB [ref] scale TGNN training to
> multiple machines by means of partitioning and communication-avoidance
> techniques. These systems, however, are designed to train and evaluate a
> given architecture, not to compare candidate architectures. NAS, in
> contrast, must evaluate hundreds or thousands of candidates, and none of
> these systems offers either a search strategy or an evaluation mechanism
> for this purpose. To the best of our knowledge, no NAS framework exists
> for temporal GNNs. The natural approach, then, is to parallelize the
> evaluation of candidate architectures on top of existing training
> frameworks. As the following example shows, however, doing so naively
> violates the read-after-write (RAW) dependencies of temporal training.
> The resulting scores are so unreliable that NAS selects the wrong
> architecture.

**写作观察(v4,三轮评审修订后)**:
1. 静态 NAS 组已删除(v2 评审)——与段 2 重复(GraphNAS 提名 + i.i.d./时序依赖机制段 2 已讲);DynaHB 引言段 3 也只有训练框架一组,intro 相关工作必须压缩
2. 划界句(v4 软化):v3 的 "neither a search strategy nor an evaluation mechanism" 被批太绝对(训练框架当然有评估机制)——改为 "designed to train and evaluate a given architecture, not to compare candidate architectures",承认有训练+评估能力,缺口限定为**比较候选架构**;"either a search strategy or an evaluation mechanism **for this purpose**" 保留双项(v2 评审要的)同时加 scoping(v4 评审要的)
3. 桥句(v4):"A natural alternative" → "**The natural approach, then**"(评审:alternative 指代模糊);"then" 勾住前句空白声明(没有框架 → 那怎么办 → 自然的路子);"on top of existing training frameworks" 保留——把前半段的训练框架变成"诱饵"(读者以为有现成的路,下一句被打回),前后素材全部用上
4. 末两句(v4 拆分):"breaks"→"**violates**"(系统文献标准话术,DB 大同行熟悉);长句拆成 "doing so naively violates the RAW dependencies..." + "**The resulting scores are so biased that NAS selects the wrong architecture.**" 短句收尾;"naively" 与 "natural approach" 对比——自然想到的路,朴素做就会坏;RAW 重现(第三次,呼应段 2 thesis);末句**预告段 4** 内容
5. 评审建议段末加 "we introduce DepTGL" 预告句——**不采纳**:按 DynaHB 模板方法在段 6 出场(段 3 → 段 4 case study → 段 5 挑战 → 段 6 提出方法),提前预告预支段 6、抢段 4-5 的悬念;段 3 的预告任务已由末句完成
6. snapshot vs 事件流(JODIE 是事件流)的区别没提——放 Section 2 展开,但评审会追问,Section 2 要主动交代

**段 4 — Case study(导师硬性要求,带具体数值)初稿 v1(2026-08-23)**:

> This failure is not hypothetical. On the MOOC dataset [ref], we ran the
> same NAS search twice, once evaluating candidate architectures with
> serial training and once with a naive parallelization that spreads
> training across three workers by migrating model state. The serial
> search converges to a compact architecture with 133K parameters, whose
> test MRR is 0.8561. The parallel search instead selects an architecture
> three times larger — 402K parameters, with static embeddings and
> projection layers enabled — whose test MRR is only 0.7000, a drop of
> 0.156. The cause is not a single large error: the parallel search
> manages random state differently from serial search — trials draw from
> shared random streams instead of independent seeds, redundant model
> rebuilds perturb the streams consumed by later trials, and the
> controller is updated in batches rather than after each trial. Each
> deviation is negligible in isolation, but they compound across the
> search: the evaluation scores no longer faithfully reflect the quality
> of the candidate architectures, and the search converges to the wrong
> architecture. Naively batching training is hazardous too: with
> consecutive interactions (u1,i1), (u1,i2), (u2,i1), a naive batch
> processes the first two against the same pre-batch embedding, so
> (u1,i2) misses the update that serial training applies first — the
> read-after-write dependency along the interaction stream is violated
> [ref].

**写作观察(段 4 v2,按用户质疑修正机制句)**:
1. 开句 "This failure is not hypothetical" 直接兑现段 3 末句的预告(as the following example shows...)——case study 是"法庭举证段",举证对象是段 2 的 thesis、段 3 的断言
2. 数值清单(导师硬性要求):133K vs 402K(参数量)、0.8561 vs 0.7000(test MRR)、0.156(差距)、three workers——全部来自我们自己的实验(FINAL_REPORT),不是文献里的二手数字
3. 机制句修正(v2):**只保留 FINAL_REPORT 盖章的三处搜索态偏差**(trial 共享随机流/冗余 build 污染 RNG/controller 批量更新);**删掉 "optimizer 重建"**——那是 ATTEMPTS_LOG tbatch 时代的证据,最终 serial 配置下单架构训练 Serial≡Pipeline diff=0,该因素在最终配置下无效;**删掉 "favoring static-on"**——偏袒方向同样是 tbatch 时代结论,未在最终配置复验
4. 措辞对齐:"biased"→"**unreliable**"、开句 bias→failure——已验证的机制是评分被污染(不可信),不是有方向的偏袒;段 3 末句同步改 "so biased"→"so unreliable"
5. t-Batch 微例从"可选"变**必需**(引 JODIE,新颖性地雷):段 3 末句承诺了 "violates RAW",段 4 必须兑现——两个例子分工:微例兑现 RAW 承诺,0.8561/0.7000 兑现"选错架构"承诺
6. 遗留风险(已查证 git):BATCH_MODE 在 fb591ff 从 tbatch 改为 serial;首次对比(阶段 1,08-05~06)早于该切换(ATTEMPTS_LOG 尝试 10)→ **0.7000 那次是 tbatch 时代的跑法**,混杂了 tbatch 放大的训练态偏差(④⑤:tbatch 每 epoch 仅 ~14 步,epoch 边界动量断裂被放大,3 数据集系统性偏袒 static-on)+ 搜索态 bug(①②③)。最终 serial 配置下 ④⑤ 已无效(单架构训练 diff=0)。→ **干净消融**(serial 配置下仅关闭三修复、重跑 Pipeline Naive)是段 4 案例能否站住的**判据**:0.7000 复现 → 机制归因干净;不复现 → 案例段必须改写成 tbatch 故事或复合故事。列入实验表最高优先级
7. 待确认:MOOC 的引用、test MRR 是否就是论文要报的指标(与实验表统一)

**段 5 — 三个挑战**(Challenge I/II/III 正式陈述,用数据管理话术):
- Challenge I:temporal data dependency——递归状态更新造成交互流上的写后读依赖,朴素并行破坏语义
- Challenge II:evaluation fidelity——同一架构在不同后端必须得到相同分数;跨进程状态迁移的微小偏差累积成系统性评分偏差
- Challenge III:skewed workloads——时序分区的成本极不均匀,流水线阶段负载失衡

**段 6 — 我们的方案**:"To address the above challenges, we propose DepTGL..."(逐条对应:无冲突 t-Batch → Challenge I;种子纪律 + RNG 保持 + 离策略更新 → Challenge II;代价模型 + DP 分组 + Lagrange 分配 → Challenge III)

**段 7 — 贡献列表**(4 条,C1 系统 / C2 t-Batch / C3 保真度 / C4 实验,英文表述见"二")

**段 8 — 论文结构**:Section 2 相关工作;Section 3 预备知识(时序图、JODIE 式训练、NAS);Section 4 DepTGL 框架概述;Section 5 无冲突批处理与保真度设计;Section 6 实验;Section 7 结论。

### 引言段 1-2 初稿(2026-08-14,代写,待用户消化)

> Temporal interaction data is ubiquitous in real-world services, including
> recommendation systems, social networks, and question-answering platforms,
> where each record captures a (user, item, timestamp) event. Temporal Graph
> Neural Networks (TGNNs), such as JODIE [ref] and TGN [ref], have achieved
> state-of-the-art performance at predicting future interactions: as each
> interaction arrives, they recursively update the embeddings of the
> participating user and item, so that a node's latest embedding summarizes
> its interaction history.
>
> Choosing a TGNN architecture is, however, far from trivial. A TGNN
> involves a large hyperparameter space — aggregation functions, temporal
> decay functions, memory cells, and time projection, among others — and
> manual tuning over this space is labor intensive and error prone. Neural
> Architecture Search (NAS) automates architecture design; NAS frameworks
> for static GNNs (e.g., GraphNAS [ref]) find architectures that rival or
> outperform hand-designed ones. Yet these frameworks cannot be applied to
> temporal graphs directly. Static GNN training assumes that samples are
> independent and identically distributed, so mini-batches can be processed
> in arbitrary order and in parallel. TGNN training is different: every
> interaction rewrites the embeddings of its user and item, and subsequent
> interactions must read the values just written, creating read-after-write
> (RAW) dependencies along the interaction stream. TGNN training is
> therefore inherently sequential. Since NAS must evaluate hundreds or
> thousands of candidate architectures, training each serially is
> prohibitively slow, making parallel execution essential. Enabling
> parallel TGNN training without violating these RAW dependencies — and
> thus without biasing architecture evaluations — is the central
> challenge this paper addresses.

**写作技巧对照**(学习用):
- "ubiquitous... including 三例子":数据普遍性一句带过
- "such as [引用] + as each interaction arrives":点名 SOTA,同时用大白话交代 TGNN 工作方式(评审不懂 RNN 也能懂)
- "is, however, far from trivial":转折钩
- 列举 + "among others":展示空间大,不解释每个超参
- "Yet... directly":制造 gap(DynaHB 的 "fall short" 同款)
- RAW 只点名字不解释——解释留给挑战段(段 5)
- "Since NAS must evaluate hundreds...":压力句,先让读者明白"必须并行",再让"并行会坏"成立(评审建议,2026-08-14)
- 末句 = thesis 句,点名两个后果(不破坏依赖 = 无损并行 → t-Batch;不偏置评分 = 评估保真度 → C2),第一段就埋伏方法主线(评审建议,2026-08-14)
- hyperparameter 不带连字符,与多数论文一致(评审建议)

---

## 五、写作要点清单(导师《第三阶段》逐条对照)

- [ ] 简洁:去掉不影响表达的冗余词("如果一句话去掉几个单词完全不影响表达,说明这些单词冗余")
- [ ] 数据管理话术:data caching / temporal data dependency / state migration(避免纯 AI 行话)
- [ ] 评审当大同行:RNN 只说成"维护时序状态的模块",不进内部结构
- [ ] 所有 DepTGL 加 `\texttt{DepTGL}`(图表里除外)
- [ ] 算法篇幅:双栏最多半页多,核心流程伪代码,显而易见过程用函数代替
- [ ] 图:字号介于正文与标题之间、内容紧凑、莫兰迪配色

## 六、待补数据(下一步实验设计时确定)

| 项 | 导师要求 | 现状 |
|----|---------|------|
| 数据集 | ≥4 个,最好 6 个 | MOOC ✅(其余待跑:Wikipedia、Reddit、合成) |
| 模型 | ≥2 个算法 | JODIERNN ✅ + Hybrid(TemporalEventGNNJODIE)待补 |
| 常驻 baseline | 3 个(所有实验都要有) | 待定(候选:官方 JODIE 手工配置、串行随机搜索、静态 NAS 适配、PiPAD 式单机框架) |
| 实验环境 | ≥2 个 | 待定 |
| 加速比数字 | — | 现有:DP 1.6× / Naive 1.5× / Smart 2.3×(MOOC 单数据集,需扩) |

## 七、待确认问题

1. DepTGL 全称是什么?(猜测:Dependency-aware Temporal Graph Learning,需确认)
2. C2/C3 算法贡献是否就用 t-Batch + 评估保真度?还是换 ConfigOptimizer(代价模型)做 C3?
3. 论文目标会议?(DynaHB 是 PVLDB,语气参照的是 VLDB 风格)

## 八、定位危机:流水线 vs 架构并行(2026-08-14,用户提出)

**事实**:Serial 1.0× / DP 1.6× / Naive 1.5× / Smart 2.3×;Smart 最优配置恒为 1 stage × N workers = 零流水线、纯异步架构并行;Naive ≈ DP(流水深度收益被 stage 状态迁移 + batch 同步 barrier 吃掉)。Smart 相对 DP 的真实增量 ≈ 1.4×,全部来自异步无 barrier + 持久 worker 不重建状态 + 连续控制器更新,与流水线无关。

**诊断**(非 bug,是 NAS 工作负载的结构性事实):trial 多且小(133K 参数、少 epoch),天然可并行单元是 trial 本身;流水线要填满需要"1 个 trial 的阶段数 ≥ worker 数",NAS 恰好相反(trial 数 ≫ worker 数)。

**出路**:
- A(推荐):重新定位为"多执行策略 NAS 框架",卖点 = 保真度 + 代价模型自动选策略;"1 stage 最优"写成工作负载分析发现;Smart vs DP 的 1.4× 作为干净加速声明
- B:主打评估保真度,执行策略退居其次
- C:硬改结构让 pipeline 赢——不建议,与 NAS 工作负载特性对着干
- 若走 A:把 stage 边界状态迁移开销编码进代价模型(保真度排障的副产品),让配置器"正确地"输出少 stage——保真度与代价模型统一成一条主线

**定位实验(先做,便宜)**:GPU 数 1/2/3/4 × 数据集大小(小/中/大)× 模型大小(小/大)扫四种策略找交叉点。找到 pipeline 反超点 → 故事完整("configurator 按 workload 选策略");永远找不到 → pipeline 降级为保真度故事里的"对照实验",论文重心移至保真度 + 架构并行。该实验同时充当导师要求的可扩展性实验与时间分解实验的数据源。
