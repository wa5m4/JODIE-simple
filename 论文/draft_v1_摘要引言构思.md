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
| **Challenge II:评分保真度** | NAS 要求"同一架构在哪跑都得同一个分数"。跨进程/跨阶段迁移模型与优化器状态时,种子、RNG、控制器更新节奏的微小偏差会累积成**系统性偏差**,最终选错架构 | evaluation fidelity / state migration |
| **Challenge III:负载倾斜** | 时序分区的成本极不均匀(事件数、新用户/物品数差异大),流水线各阶段负载失衡,worker 空转 | skewed workloads / cost-based load balancing |

### 贡献(我们怎么解决)——三贡献定稿(2026-08-27,用户拍板)

> **定稿(2026-08-27)**:三个贡献 = ①图神经网络搜索系统(DepTGL)②Pipeline 流水线策略 ③异步架构生成+训练。
> - **stale_batch / t-Batch 降级**:不作为重点和贡献,仅保留为段 4 案例燃料(双臂实验照跑拿数字,不进贡献列表)。
> - **评估保真度机制(旧 C2 旗舰)整体并入 ① 作为系统设计属性**——五档消融梯证据不浪费,成为"系统为什么可信"的证明("既快又真")。
> - **智慧分配(ConfigOptimizer 自动配置)已证伪、MemShare 死代码:永久出局,不再作为备选。**

| 贡献 | 类型 | 内容 |
|------|------|------|
| C1 DepTGL 搜索系统(旗舰) | 系统 | 面向时序 GNN 的端到端 NAS 框架:JODIE 风格搜索空间 + REINFORCE 控制器 + 多执行策略后端(串行 / 数据并行 / 流水线 / 异步);**保真执行内置为设计属性**:并行训练尊重 RAW 依赖 + 随机状态协议(per-trial 种子纪律、跨阶段 RNG 保存/恢复、离策略 logprob 重算、负样本预分配)→ 任何后端选出的架构与串行一致 |
| C2 Pipeline 流水线策略 | 技术 | 时序交互流 count 分区 → stage 划分 → cost 负载均衡的流水线执行后端,在 RAW 依赖下保持评分一致 |
| C3 异步架构生成+训练 | 技术 | 持久化 worker 池 + 架构生成与训练时间重叠(预填充 2×arch_per_step、headroom 补货、端 flush):3.0× 加速(5,486s vs naive 16,290s)、质量位级一致(0.856121275963994)、RL 路径零崩溃 |
| 实验章(支撑,非贡献) | 实验 | 五档消融梯 + 噪声地板判据(保真度机制必要性);stale_batch/t-Batch 双臂(段 4 数字);定位实验(pipeline 反超点,待跑) |

> ⚠️ **原创性雷区(2026-08-14,仍有效)**:t-Batch 出自 JODIE 论文,TGN 窗口批处理出自 TGN 论文——不能写"we propose t-Batch",只能写"集成/适配到 NAS 评估循环"(且已降级为非贡献)。注意力聚合 + 时间衰减是 TGAT 风格。DepTGL 真正原创的三块:C1 框架 + 保真度机制、C2 流水线策略、C3 异步池。

**为什么 C1 是旗舰**:它把"并行 NAS 评分保真度"这个真问题(排障链揭示的完整答案:per-trial 种子缺失、冗余 build_model 污染 RNG、批量 vs 逐 trial 控制器更新)做成框架内置协议,而不是一次性的工程修补。修前 Pipeline Naive 选 402K 错误架构(test 0.69999);修后任何后端都选 133K(test 0.8561),两次全开运行位级一致——"任何执行后端、同一架构、同一分数"。

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

> 📌 **待按三贡献定稿重写(2026-08-27)**:v1/v2 摘要仍含旧框架要素——t-Batch 作为 headline feature(已降级)、cost-model configurator(已出局)、2.3×(已更新为 3.0×)。重写时:方法句 = 保真度内置的多后端系统 + pipeline 流水线 + 异步架构生成;加速数字用 3.0×;t-Batch 可留一句作为"无损并行训练"的技术实现(集成自 JODIE),但不作为 headline。

**仍待讨论**:3 个挑战目前只出现 2 个(时序依赖、评分失真),负载倾斜只在方法句里。若想严格对齐 DynaHB"挑战点名→方法对应",可在第 2/3 句补"and temporal workloads are highly skewed"。另外 "off-policy" 对不懂 RL 的评审是黑话,定稿时考虑换成 "policy updates robust to asynchronous execution" 或保留(评审可跳过不深究)。

---

### 摘要 v3(按三贡献定稿重写,2026-08-27,代写候选,待用户批改)

> Temporal Graph Neural Networks (TGNNs) such as JODIE have achieved
> state-of-the-art performance on temporal interaction prediction tasks.
> However, designing TGNN architectures requires labor-intensive manual
> tuning over a large hyperparameter space, and existing neural
> architecture search (NAS) frameworks target static GNNs, whose training
> assumes i.i.d. samples and thus cannot handle the temporal data
> dependencies inherent in interaction streams. Searching over hundreds
> of candidate architectures demands parallel execution, yet naively
> parallelizing TGNN training breaks the read-after-write (RAW)
> dependencies among consecutive interactions, yielding unreliable
> architecture scores that cause NAS to select inferior architectures.
> We introduce \texttt{DepTGL}, an NAS system for TGNNs built on three
> techniques: (i) faithful execution, where parallel training respects
> the stream's RAW dependencies and a random-state protocol — per-trial
> seed discipline, RNG-preserving state migration, and off-policy
> controller updates — makes every backend reproduce the serial
> search's scores and selection; (ii) a pipeline strategy that
> partitions the interaction stream into cost-balanced stages while
> preserving RAW semantics at stage boundaries; and (iii) asynchronous
> architecture generation that overlaps candidate generation with
> training in a persistent worker pool. Extensive experiments on X datasets with Y
> model families show that DepTGL achieves up to 3.0× speedup over its
> synchronous counterpart without degrading search accuracy.

**v3 相对 v2 的改动(逐条交代)**:
1. t-Batch **整体移出摘要**(v2 曾为 headline,一度降为 (i) 内子句,2026-08-27 用户再质疑后连子句也不保留)——RAW 的账由 (i) 的结果级承诺"任何后端复现串行结果"来关,实现机制(无冲突批处理,集成自 JODIE)下沉到 Section 5
2. cost-model configurator 句删除(已出局)——(ii) 只保留 cost-balanced stages(手工成本均衡,事实成立)
3. 加速数字 2.3× → **3.0×(vs 同步 naive,5,486s vs 16,290s)**;"over serial NAS" 改为 "over its synchronous counterpart"——3.0× 的分母是 naive 同步批不是串行;串行倍率待多数据集实验表补,暂不虚报
4. 方法句改为三项并列 (i)(ii)(iii),与段 7 贡献列表一一对应
5. 保留 "off-policy controller updates"(见上"仍待讨论");(i) 里 "reproduces the serial search's scores and selection" 就是"位级一致 0.856121275963994"的抽象说法,不预支方法细节

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
> The resulting scores are so biased that NAS selects the wrong
> architecture.

**写作观察(v4,三轮评审修订后)**:
1. 静态 NAS 组已删除(v2 评审)——与段 2 重复(GraphNAS 提名 + i.i.d./时序依赖机制段 2 已讲);DynaHB 引言段 3 也只有训练框架一组,intro 相关工作必须压缩
2. 划界句(v4 软化):v3 的 "neither a search strategy nor an evaluation mechanism" 被批太绝对(训练框架当然有评估机制)——改为 "designed to train and evaluate a given architecture, not to compare candidate architectures",承认有训练+评估能力,缺口限定为**比较候选架构**;"either a search strategy or an evaluation mechanism **for this purpose**" 保留双项(v2 评审要的)同时加 scoping(v4 评审要的)
3. 桥句(v4):"A natural alternative" → "**The natural approach, then**"(评审:alternative 指代模糊);"then" 勾住前句空白声明(没有框架 → 那怎么办 → 自然的路子);"on top of existing training frameworks" 保留——把前半段的训练框架变成"诱饵"(读者以为有现成的路,下一句被打回),前后素材全部用上
4. 末两句(v4 拆分):"breaks"→"**violates**"(系统文献标准话术,DB 大同行熟悉);长句拆成 "doing so naively violates the RAW dependencies..." + "**The resulting scores are so biased that NAS selects the wrong architecture.**" 短句收尾;"naively" 与 "natural approach" 对比——自然想到的路,朴素做就会坏;RAW 重现(第三次,呼应段 2 thesis);末句**预告段 4** 内容(2026-08-28:正文曾改 "unreliable";双臂单因素验证方向性后恢复 "biased",见段 4 note 12.4)
5. 评审建议段末加 "we introduce DepTGL" 预告句——**不采纳**:按 DynaHB 模板方法在段 6 出场(段 3 → 段 4 case study → 段 5 挑战 → 段 6 提出方法),提前预告预支段 6、抢段 4-5 的悬念;段 3 的预告任务已由末句完成
6. snapshot vs 事件流(JODIE 是事件流)的区别没提——放 Section 2 展开,但评审会追问,Section 2 要主动交代

**段 4 — Case study(导师硬性要求,带具体数值)v5(2026-08-28,数字已填齐:双臂 + 补测)**:

**数字速览(2026-08-28,唯一变量 = 批处理模式)**:

| 执行方式 | 选中架构 | 搜索 val | 自报 test | 真实串行 test |
|---|---|---|---|---|
| serial(基线) | 133,888(emb=128/static=off) | 0.8267 | 0.8561 | 0.8561(bit-identical) |
| 冲突消解分批(桥梁) | 133,888(同家族) | 0.8646 | 0.8793 | — |
| 朴素分批(naive) | 147,840(emb=64/static=on) | 0.9649 | 0.9335 | **0.6014** |

同一 147K 架构:保真 val 0.62 → naive val 0.96(**+0.345**);自报 test 0.9335 → 真实 0.6014(**+0.332 双重污染**,val/test 同机制)。桥梁/naive 的"自报 test"为各臂批模式训练所得(日志 "Serial training" 是写死字符串,实验章交代协议细节)。

> This failure is not hypothetical. On the MOOC dataset [ref], we
> parallelized the evaluation of candidate architectures across three
> workers and ran the same NAS search three times, changing only how the
> workers form training batches: serial processing; conflict-free
> batching, which keeps the nodes within each batch unique; and naive
> batching, which chunks consecutive interactions without resolving
> conflicts. The serial search converges to a compact architecture with
> 133K parameters and a test MRR of 0.8561; the conflict-free search
> selects the same architecture, confirming that batching per se is
> harmless. The naive search instead selects a 147K architecture that
> leans on static embeddings, whose serial re-training achieves a test
> MRR of only 0.6014 — a drop of 0.2547. Consider a single batch of
> consecutive interactions (u1,i1), (u1,i2), (u2,i1): the naive batch
> processes the first two against the same pre-batch embedding, so
> (u1,i2) misses the update that serial training applies first; the
> read-after-write dependency along the interaction stream is violated
> [ref]. Worse, the failure is self-concealing: under naive evaluation
> the same architecture scores 0.96 on validation — against 0.62 under
> faithful evaluation — and its reported test MRR of 0.9335 even exceeds
> the serial search's 0.8561. Predictions computed from stale states
> systematically overrate architectures that rely on static features,
> distort the leaderboard, and drive the search to the wrong
> architecture.

**写作观察(段 4 v2,按用户质疑修正机制句)**:
1. 开句 "This failure is not hypothetical" 直接兑现段 3 末句的预告(as the following example shows...)——case study 是"法庭举证段",举证对象是段 2 的 thesis、段 3 的断言
2. 数值清单(导师硬性要求):133K vs 402K(参数量)、0.8561 vs 0.7000(test MRR)、0.156(差距)、three workers——全部来自我们自己的实验(FINAL_REPORT),不是文献里的二手数字
3. 机制句修正(v2):**只保留 FINAL_REPORT 盖章的三处搜索态偏差**(trial 共享随机流/冗余 build 污染 RNG/controller 批量更新);**删掉 "optimizer 重建"**——那是 ATTEMPTS_LOG tbatch 时代的证据,最终 serial 配置下单架构训练 Serial≡Pipeline diff=0,该因素在最终配置下无效;**删掉 "favoring static-on"**——偏袒方向同样是 tbatch 时代结论,未在最终配置复验
4. 措辞对齐:"biased"→"**unreliable**"、开句 bias→failure——已验证的机制是评分被污染(不可信),不是有方向的偏袒;段 3 末句同步改 "so biased"→"so unreliable"
5. t-Batch 微例从"可选"变**必需**(引 JODIE,新颖性地雷):段 3 末句承诺了 "violates RAW",段 4 必须兑现——两个例子分工:微例兑现 RAW 承诺,0.8561/0.7000 兑现"选错架构"承诺
6. 遗留风险(已查证 git):BATCH_MODE 在 fb591ff 从 tbatch 改为 serial;首次对比(阶段 1,08-05~06)早于该切换(ATTEMPTS_LOG 尝试 10)→ **0.7000 那次是 tbatch 时代的跑法**,混杂了 tbatch 放大的训练态偏差(④⑤:tbatch 每 epoch 仅 ~14 步,epoch 边界动量断裂被放大,3 数据集系统性偏袒 static-on)+ 搜索态 bug(①②③)。最终 serial 配置下 ④⑤ 已无效(单架构训练 diff=0)。→ **干净消融**(serial 配置下仅关闭三修复、重跑 Pipeline Naive)是段 4 案例能否站住的**判据**:0.7000 复现 → 机制归因干净;不复现 → 案例段必须改写成 tbatch 故事或复合故事。列入实验表最高优先级

   **→ 消融结果(2026-08-24,判据命中第 1 行)**:服务器(8 GPU,16299.7s ≈ 4.5h)以 serial 配置重跑 Pipeline Naive,**唯一变量 = 关闭三修复** → 再次选出 402K 全开架构(emb=128 / rnn / linear / static=on / norm=on),test = 0.69999。**三修复效应独立成立,段 4 机制句定稿,该消融进实验表 C2。** 附带证据:0.69999 与修复前 0.7000 精确重合(最终测试走同一串行评估路径、同架构、同 seed=42)→ 评估本身确定,分数差异全部来自搜索选择;证据文件在服务器提交 07fe26e(ablation/fidelity-off 分支):summary.txt / best_arch.json / leaderboard.csv / run_ablation_fidelity_off.log
7. 待确认:MOOC 的引用、test MRR 是否就是论文要报的指标(与实验表统一)
8. 判据升级(2026-08-24,用户提出):"最终选出的架构相同"≠"搜索未被干扰"——终点之外还要看搜索轨迹(采样序列/leaderboard 分布)与评分一致性(同一架构在两次运行中的分数差)。单因素消融 f1/f2/f3(各关一个修复)按三层观察判读;"Each deviation is negligible in isolation" 需要轨迹证据支撑,不能只看终点。运行说明见 ABLATION_FACTOR_GUIDE.md
9. 定位风险(评审必问,2026-08-26,用户提出):"①② 是你们自己代码的 bug,修了 bug 论文还有什么贡献?"——回答框架:①② 是 bug,但不是**机械性 bug**(笔误/off-by-one),而是**结构性陷阱**:沉默(单 trial 一切正常、不报错,只在搜索级显现;单架构训练 Serial≡Pipeline diff=0 为证)、同源(①②③ 是同一个结构空洞的三实例——搜索没有随机状态协议)、单独致命(协议零冗余,五档消融为证)、普遍(任何朴素并行实现都会踩,不是本代码特有事故)。**论证"多数人会这么设计"(评审判据:naive 设计→贡献成立;自家缺陷→不算)的三条证据**:① 朴素写法=教科书写法——build-then-load 是通用模式、批量 REINFORCE 是 Williams 1992 原始算法(ENAS 等都用)、共享随机流是 Ray 默认行为,三处无一处独创怪写法;② 失败沉默(不崩不报错、单 trial 正常)→ 朴素设计必然带雷上线;③ "仔细点设个种子就行"的反驳不成立——f2 消融中种子纪律仍开启、只关 ② 照样翻车(402K/0.7000),证明协议≠设种子一件事,必须框架显式执行整份协议。注意:此论证在论文中要**显式写出来**,不能只断言。论文写法:段 4 的 bug 是"**病例**"(量化损失 0.156/选错架构,证明坑值得防),C1 的保真度协议是"**疫苗**"(种子纪律/状态迁移保 RNG/离策略更新做成框架内置,让这类错误**写不出来**,而不是"这次写对了");消融证明每个机制必要(单独关闭即翻车 → 不是过度工程)。代码注释里的"★ 修复"仅内部用语,论文一律称 mechanism/fidelity mechanism,不称 fix/bugfix。
10. **段 4 v3(2026-08-26,按单因素消融证据 + 用户定位决定改写)**:
    1. **定位决定(用户,2026-08-26)**:①② 在论文里"先当作 bug"描述——不把"build-then-load 是行业习惯"当作贡献论证写进正文;note 9 的病例/疫苗框架与三条证据保留为评审防御材料,不预支。措辞区分:"bug/flaw" 用于描述朴素并行实现的行为;DepTGL 自身的机制仍称 mechanism(与 note 9 一致)。
    2. **删除 "The cause is not a single large error" + "Each deviation is negligible in isolation, but they compound"**——被单因素消融证伪:f1-off 与 f2-off 各自单独翻车(402K/0.69999),"单独即致命",不是"单看可忽略、合起来才坏"。
    3. **③(批量 controller 更新)从因果句移除**——f3-off 终点正确(133K/0.8561)且评分波动在噪声包络内(max 0.0845 ≤ 0.0887 天花板)→ 它是负对照,进实验表,不进案例段因果句。摘要里"off-policy controller updates" 作为机制保留。
    4. **新因果句**:共享随机流耦合(① 种子纪律 + ② build 扰动流),"Either flaw alone suffices to derail the search";新增第三层观察数字:同一架构分数跨运行差最高 0.166(噪声天花板 0.089)→ 证明分数本身被污染,呼应"不要只看终点"。
    5. 正文 0.7000 = 干净消融 0.69999 的四舍五入;实验表报原始值 0.69999。
    6. **未写进段 4**:修复后搜索 bit-identical 可复现(两次全开运行轨迹逐位一致)——按"段 6 前不预告方法"原则(段 3 写作观察第 5 条),留给段 6/实验表。
11. **段 4 v4(2026-08-26,用户定位决定:bug 不进概述)**:
    1. ①② 随机状态 bug 从段 4 因果句**整体移除**——bug 是工程事故,撑不起 intro 的论点(评审会说"修了 bug 不就行了,要框架干嘛");intro 案例必须归因于固有机制(RAW)。
    2. 归因换成:朴素分批(stale_batch,不做冲突消解)破坏写后读依赖。微例从段末 "too" 位置**提升为因果说明**(紧跟失败数字,解释机理),末句 "Predictions computed from such stale states..." 收尾。
    3. **数字改为待补**:原 402K/0.7000 的干净归因是 bug(serial 配置下只关三修复复现 0.69999),不能用于 RAW 归因。需要"唯一变量=批处理模式"的干净对照:BATCH_MODE=stale_batch、修复全开、pipeline_naive,其余与五档消融一致。运行说明见 RAW_ABLATION_GUIDE.md。
    4. bug 故事(五档消融梯:全关/f1/f2/f3 + 噪声天花板 0.089 vs 污染 0.166)整体移出 intro,归实验章(C1 保真度机制的消融证据)。
    5. **ripple effect(写段 5 时处理)**:段 5 Challenge II(evaluation fidelity)的 intro 级动机同样不能讲 bug 故事,需抽象陈述(并行评估必须跨后端保持同一分数;跨进程状态迁移的偏差会累积)——具体证据在实验章。
12. **段 4 v5(2026-08-28,双臂 + 补测数字填齐)**:
    1. **三占位数字落定**:错误架构 = 147,840 参数(emb=64/static=on);test MRR = 0.6014;差值 = 0.2547。数据来自双臂消融(唯一变量=批处理模式)+ 补测分支 ablation/reeval-fixed-arch(串行复评;133K 探针 bit-identical 复现 0.856121275963994 → 补测路径 = 基线协议确认)。
    2. **"掉分"三幕反转**:初判读按日志 "Serial training" 判"掉分被证伪"(0.9335 > 0.8793)→ 查代码发现 batch_mode 取自 base_config、日志字符串写死(两臂最终测试实为各臂批模式训练)→ 补测 0.6014 掉分成立。教训:判读落到代码,不落到日志文案。
    3. **双重污染(新证据)**:同一 147K 架构 val 0.62 → 0.96(+0.345)、test 0.6014 → 0.9335(+0.332)——搜索评分与最终测试同机制、同幅度被污染。
    4. **方向性恢复**:v2 note 4 删 "favoring static-on"(tbatch 时代证据、最终配置未复验)——现在双臂单因素下复验成立(static-on 家族 +0.345,static-off 家族基本不动),机制句恢复方向表述 "systematically overrate architectures that rely on static features";段 3 末句同步 "unreliable" → "biased"。
    5. **桥梁句(处置原则内)**:新增 "conflict-free batching... selects the same architecture"——单变量归因必需(分批本身无害,坏的是不懂依赖的分批);不点名 t-Batch、不称贡献,符合"t-Batch 只在 Section 5 以集成自 JODIE 出现一次"。
    6. **伪装幕(题眼)**:自报 test 0.9335 > 基线 0.8561 > 真实 0.6014——坏选择带完美成绩单,段 4 从"举失败例子"升级为"失败且伪装成成功"。
    7. **数字口径**:0.6014 来自补测串行复评(最终测试协议 seed=20042);实验章交代双臂最终测试的协议细节;段 4 只报三个数字,不展开协议。

**段 5 — 三个挑战**(Challenge I/II/III 正式陈述,用数据管理话术):

写作铁律:只陈述问题类别,**不出现方法成分**(种子/logprob 等药方成分留到段 6);每个挑战 = 名(一行黑体)+ 机理(2-3 句,为何在时序 GNN 的 NAS 里必然发生)+ 为什么现有方法不够(1 句,指回段 3)。段 5 是"诊断书":把段 4 的一次发作升华为三类一般性问题。

- Challenge I:temporal data dependency——递归状态更新造成交互流上的写后读(RAW)依赖,朴素并行破坏语义;静态 NAS 假设 i.i.d. 所以不面对它;JODIE 的 t-Batch 为训练吞吐设计,未解决"评分也要保真"的问题
- Challenge II:evaluation fidelity——同一架构在不同后端必须得到相同分数;跨进程状态迁移的微小偏差累积成系统性评分偏差
- Challenge III:skewed workloads——时序分区的成本极不均匀,流水线阶段负载失衡

**挑战 → 贡献对应**(段 7 贡献列表与之一一对上,评审必查):CI+II → C1 系统(**保真执行** = 结果级承诺"任何后端复现串行搜索结果",实现机制进 Section 5);CIII → C2 pipeline;"必须快"(段 2 压力句的伏笔)→ C3 异步。

**三病审计(2026-08-27,用户要求)**:判断"病"是否值得写的三条标准 = ①被某个贡献治 ②在我们实验里真实发作(最好带数字)③能说明现有方法不够。逐条审:CI(RAW)= 必须留,时序 NAS 问题的定义本身,双臂实验正在验证; CII(保真度)= 必须留,证据最硬(五档消融/位级一致/噪声地板),C1 旗舰的实心; CIII(负载倾斜)= **存疑有理**:赢家配置 1 stage 零流水线,负载均衡尚未在任何实验展示价值,数字全靠定位实验——定位实验 = CIII 的"发作记录",找不到则 CIII 与 C2 一起降级。三病的来历不是文献模板,是项目排障史三个坑的升华(朴素分批坑→CI,pipeline 排障链→CII,定位危机→CIII)。

**用户质疑(2026-08-27):"保真度排障已定性为 bug 不算贡献"**——澄清,与段 4 note 9/10.1 一致:bug 是**病案**(三个事故,不进 intro/贡献,只进实验章当"坑值得防"的证据);随机状态协议是**疫苗**(系统设计属性,进 C1)。两者不是一回事。措辞纪律:"bug/flaw" 只描述朴素并行实现的行为;DepTGL 自身的机制一律称 mechanism,不称 fix/bugfix。删除协议 = C1 变空壳,五档消融证据无家可归,"修 bug 不算贡献"的攻击反而成立。

**候选初稿(英文,2026-08-27,待用户消化)**:

> **Challenge I: temporal data dependency.** TGNN training recursively
> rewrites node embeddings as interactions arrive, so consecutive
> interactions touching the same node form read-after-write (RAW)
> dependencies along the stream. Static NAS frameworks assume i.i.d.
> samples and process batches in arbitrary order; applied to TGNNs,
> out-of-order or interleaved batches read stale states and yield scores
> that do not reflect the candidate's true quality. Existing batching
> schemes eliminate read-after-write conflicts to speed up training, but
> they do not keep the scores themselves faithful — an open problem in
> NAS evaluation.
>
> **Challenge II: evaluation fidelity.** NAS compares architectures by
> score, so the same candidate must receive the same score no matter
> which execution backend evaluated it. Parallel backends evaluate
> candidates in different processes and pipeline stages, migrating model
> and optimizer state across process boundaries; tiny deviations in
> random state and update cadence accumulate into systematic score bias.
> The leaderboard is distorted, and search converges to an architecture
> favored by the evaluation artifact rather than by the data.
>
> **Challenge III: skewed workloads.** Temporal partitions are
> structurally uneven — interaction counts and the numbers of new users
> and items vary widely across partitions — so static worker allocation
> leaves some workers idle while others become stragglers. An evaluator
> must balance these stages cheaply, without an expensive profiling pass
> per candidate.

**Challenge III 数字已补(2026-09-03,partition_stats.py,已同步 draft_v2_引言 + main.tex)**:new users 每分区 max/min = 1.83×(20K 事件)/ 4.09×(100K)/ 114.5×(全量);unique users 1.30/1.60/6.46;事件数严格均匀(1.00,count 策略)→ 原占位"事件数 X 倍"证伪,表述改为 new users、句内数字 "up to 4×"(锚 100K;主配置锚 20K 则 1.8×)。重算命令 `python partition_stats.py <max_events>`。

**裁决(2026-08-28,用户拍板 B)**:Challenge I 末句不点名 JODIE 无冲突分批,改用泛化写法("现有批处理方案消冲突提速,但不保评分")——"唯一一次点名出场"完整留给 Section 5,处置原则零违例。段 4 的 "conflict-free batching" 不受影响(对照组描述,无名字无引用)。draft_v2_引言.md 已同步。

**段 6 — 我们的方案**:"To address the above challenges, we present DepTGL..."(逐条对应:**保真执行** = t-Batch 集成(引 JODIE,→ CI)+ 随机状态协议(→ CII),合归 C1;pipeline 流水线 + cost 负载均衡 → Challenge III;异步持久化池 → 加速/C3)

写作要点:每个技术句三段式(技术名 / 干什么 / 解决哪个挑战);不预支机制细节;t-Batch 首次正式登场必须带 [JODIE] + "integrated";与摘要 v3 的 (i)(ii)(iii) 同构。

**候选初稿(英文,2026-08-27,含"保真执行"修正)**:

> To address these challenges, we present \texttt{DepTGL}, an NAS
> system for temporal GNNs. DepTGL searches a JODIE-family architecture
> space under a REINFORCE controller and evaluates each candidate
> through interchangeable execution backends — serial, data-parallel,
> pipeline, and asynchronous — that share a common evaluation
> interface. Three techniques make these backends both fast and
> faithful. First, **faithful execution** addresses Challenges I and
> II: parallel training respects the stream's RAW dependencies, while a
> random-state protocol — per-trial seed discipline, RNG-preserving
> state migration, off-policy controller updates, and preallocated
> negative samples — enforces evaluation fidelity by construction, so
> every backend reproduces the serial search's scores and selection.
> Second, a **pipeline strategy** partitions the
> interaction stream into stages and balances them by estimated
> partition cost, keeping workers saturated under skewed workloads
> without violating RAW semantics at stage boundaries, addressing
> Challenge III. Third, an **asynchronous generation engine**
> maintains a persistent pool of workers in which candidate generation
> overlaps candidate training, so the GPUs rarely idle and the search
> completes in a fraction of the serial wall-clock time.

**修正记录**(两轮,2026-08-27):①用户质疑"CI 的解法被塞进协议"→ 区分 CI(RAW)与 CII(随机状态一致性)各自的解法;②用户再质疑"为什么一直提 t-Batch,它只是批处理功能"→ **定处置原则:降级机制不占 intro 结构位**。CI 的解法在 intro 只写结果级承诺("并行训练尊重 RAW 依赖"),实现机制(无冲突批处理,集成自 JODIE [ref])在 Section 5 出现一次。命名逻辑:保真执行 = 数学保真(保 RAW)+ 评分保真(保一致性),与"既快又真"的"真"对齐;**C1 的贡献是框架本身,承诺在结果层,不在具体批处理机制层**。

**段 7 — 贡献列表**(3 条定稿:C1 系统(含保真度)/ C2 Pipeline 流水线策略 / C3 异步架构生成+训练;stale_batch/t-Batch 不进贡献列表)

**段 8 — 论文结构**:Section 2 相关工作;Section 3 预备知识(时序图、JODIE 式训练、NAS);Section 4 DepTGL 框架概述;Section 5 保真度保持的并行执行(流水线 + 异步架构生成);Section 6 实验;Section 7 结论。

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
- 末句 = thesis 句,点名两个后果(不破坏依赖 = 无损并行 → t-Batch;不偏置评分 = 评估保真度 → 现归 C1 系统内置协议),第一段就埋伏方法主线(评审建议,2026-08-14;标签 2026-08-27 更新)
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
| 加速比数字 | — | 现有:DP 1.6× / Naive 1.5× / 异步 3.0× vs naive(5,486s vs 16,290s,位级一致;MOOC 单数据集,需扩) |

## 七、待确认问题

1. DepTGL 全称是什么?(猜测:Dependency-aware Temporal Graph Learning,需确认)
2. ~~C2/C3 算法贡献是否就用 t-Batch + 评估保真度?还是换 ConfigOptimizer(代价模型)做 C3?~~ **已解决(2026-08-27)**:三贡献定稿 = ①搜索系统(含保真度,旗舰)/ ②pipeline 流水线 / ③异步架构生成+训练;智慧分配与 MemShare 出局;stale_batch/t-Batch 降级为案例材料。遗留:pipeline 反超点待定位实验。
3. 论文目标会议?(DynaHB 是 PVLDB,语气参照的是 VLDB 风格)

## 八、定位危机:流水线 vs 架构并行(2026-08-14,用户提出)

**事实**:Serial 1.0× / DP 1.6× / Naive 1.5× / Smart 2.3×;Smart 最优配置恒为 1 stage × N workers = 零流水线、纯异步架构并行;Naive ≈ DP(流水深度收益被 stage 状态迁移 + batch 同步 barrier 吃掉)。Smart 相对 DP 的真实增量 ≈ 1.4×,全部来自异步无 barrier + 持久 worker 不重建状态 + 连续控制器更新,与流水线无关。

**诊断**(非 bug,是 NAS 工作负载的结构性事实):trial 多且小(133K 参数、少 epoch),天然可并行单元是 trial 本身;流水线要填满需要"1 个 trial 的阶段数 ≥ worker 数",NAS 恰好相反(trial 数 ≫ worker 数)。

**出路**:
- ~~A(推荐):重新定位为"多执行策略 NAS 框架",卖点 = 保真度 + 代价模型自动选策略~~ 代价模型自动配置已证伪出局
- ~~B:主打评估保真度,执行策略退居其次~~
- ~~C:硬改结构让 pipeline 赢——不建议,与 NAS 工作负载特性对着干~~
- **定案(2026-08-27)**:用户拍板三贡献含 C2 pipeline → 定位实验从"可选出路"升级为 **C2 生死线**:找到 pipeline 反超点 → C2 成立,故事 = "按 workload 选执行策略";找不到 → C2 降级为系统组件,贡献重排。定位实验是 C2 立项前的第一个实验,优先于其余扩展实验。

**定位实验(先做,便宜)**:GPU 数 1/2/3/4 × 数据集大小(小/中/大)× 模型大小(小/大)扫四种策略找交叉点。找到 pipeline 反超点 → 故事完整("configurator 按 workload 选策略");永远找不到 → pipeline 降级为保真度故事里的"对照实验",论文重心移至保真度 + 架构并行。该实验同时充当导师要求的可扩展性实验与时间分解实验的数据源。

## 九、逐句批改(2026-08-28,外部修改意见 → 已应用进 draft_v2_引言.md)

**批量采纳**(句子拆分/data are、however 句首、labor-intensive 连字符、括号替代破折号、"for future-interaction prediction"、相关工作段压缩"Several frameworks support... / train and evaluate a fixed architecture"、段 4 三个实验条件收紧为一句、段 6 pipeline 句重排为 "addresses Challenge III by..."、faithful execution 拆句 "As a result, every backend reproduces..."、贡献 C1 条目去掉协议四项列表更短、结构段 including / introduces preliminaries / reports experimental results)。

**3 处不采纳(有既有决策背书)**:
1. 相关工作末两句**不合并**、保留短句收尾 "The resulting scores are so biased that NAS selects the wrong architecture."(v4 观察:短句收尾的冲击力,合并则失去)
2. 保留 "The natural approach, **then**"(v4 观察 note 3:then 勾住上一句的空白声明)
3. 段 4 伪装幕保留 "**even** exceeds"(even 是三层反差的自报 > 基线 > 真实的修辞重点)

**其余小保留**:段 4 保留 "serial processing"(serial 不是批策略,不能归进"批处理策略"括号定义);段 6 保留 "fast and faithful"(与 faithful execution 术语呼应,风格建议第 6 条术语一致性)。

**Challenge I 末句与裁决 B 的关系**:外部建议 "Existing conflict-free batching speeds up training but does not guarantee faithful scores — an open problem in NAS evaluation." 为泛化写法、不点名、不带引用,与 2026-08-28 裁决 B 完全一致 → 采用其更简版本,替换此前的裁决 B 句。

**外部风格规则(留作后续自查清单)**:①单句 ≤3 行;②主动语态优先;③however 放句首(不嵌中);④[ref] 放句末不嵌中;⑤Challenge III 的 X 倍数字必须补;⑥术语一致(faithful/RAW 全篇统一)。

**本批改未覆盖**:摘要(draft_v2_摘要.md v4)未被本次意见覆盖,待用户批改;段 4 全部数字未动(0.8561 / 0.6014 / 0.2547 / 0.96 / 0.62 / 0.9335 / 133K / 147K)。

## 十、逐句批改(2026-08-28 第二轮,外部意见 7 点 → 已应用进 draft_v2_引言.md)

**采纳(7 点中 6 点 + 半句保留)**:

1. **段 4 实验设置句改写(消歧)**——事实核查结论:三臂同一 pipeline_naive 框架、3 stage × 1 worker(GPU 0,1,2);单个候选按 stage 顺序流动(stream 顺序保持),3 GPU 并行在**候选之间**(3 个 trial 同时在跑);serial = 逐交互按流顺序处理(loops.py 落空分支)。改写:"All three runs used the same three-worker parallel search framework; the only variable was how each candidate's training forms its batches: serial processing (one interaction at a time, in stream order), conflict-free batching (...), or naive batching (...)"。**未采纳审稿人"串行=单 worker"改法**——那是事实错误,会破坏"唯一变量=批处理模式"。
2. **段 2 结尾**"without violating RAW dependencies and thus without biasing" → "that respects RAW dependencies and does not otherwise bias architecture evaluation"(thus 因果过强:Challenge II 正是"RAW 之外"的偏差来源)。
3. **Challenge I** "Consecutive interactions touching the same node" → "Interactions that touch the same node"(RAW 不要求相邻;段 4 微例的 consecutive 保留——那里事件确实相邻)。
4. **Challenge II** "a candidate must receive the same score regardless of the backend" → "the relative ranking of candidates must not depend on which backend evaluates them";连带段 6/段 7 的 "reproduces the serial search's scores and selection" → "reproduces the serial search's selection"。**承诺软化记录:C1 结果级承诺从"分数+选择"降为"选择"级**——候选级分数跨后端从未验证过精确相等(同配置重跑噪声地板 mean 0.0069);位级一致证据(0.856121275963994)是最终 test 分数,留在实验章当细节。⚠️ 摘要 v4 "reproduces the serial search's scores and selection" 需同步,待用户批改摘要时一并定。
5. **段 6** "Three techniques make these backends fast and faithful" → "Three techniques address these challenges while improving efficiency."(第三技术纯加速,不担保真;faithful 由技术名自己承担)。
6. **段 3** "As the following example shows" → "As we demonstrate next"(承上启下句已有,只补显式指向;未加整句,避免与现有两句重复)。
7a. **段 4** "its reported test MRR of 0.9335" → "the test MRR it reports under its own naive evaluation, 0.9335"(数字来源:0.9335 = naive 自评;0.6014 = 串行重训,原句已写明)。
7c. **段 6 协议四项列表删除**,引言只留 "a random-state protocol"(密度控制;四项细节归 Section 5;摘要仍保留三项列表)。

**未采纳/挂账**:7b Challenge III 的 X 倍数字——保持占位(来自定位实验分区统计,现无数据);摘要同步点(第 4 点)已挂账。
