# PipeTGL 论文:摘要(assembled v7)

> **组装日期**:2026-09-15
> **来源**:v6(2026-09-15 pipeline 系统框架)+ 用户决定(2026-09-15):**贡献结构调整**——
> 保真执行**不作独立贡献**(并入系统贡献,作为系统实现的设计属性);技术贡献1 = 流水线策略;
> 技术贡献2 = 异步生成引擎;实验贡献改用经典模板句式(大量实验 → 验证有效性 → 对比 → XX×
> 加速 → 不影响准确率)。摘要正文不变,main.tex 贡献列表已同步。
> **改动链(v5→v7)**:
> ① (v6)系统框架重定位:pipeline 系统 + 三 baseline(serial / 数据并行 / 架构并行同步),
> 架构并行**不给**异步。
> ② (v6)smart_async 移出系统结构,降级为探索性边界探针(数据保留在
> positioning/SUMMARY_20260915.md,建议 Section 6 一行 ablation)。
> ③ (v6)头条数字 2.7× → **2.3×(naive_async vs serial)**,保真更强:naive_async 在**全部
> 4 个已测 cell**(B/D/F/G)位级保持 serial 选择;D 2.32× / F 2.31× 为锚点。
> ④ (v7)贡献列表 4 条:系统(pipeline 系统,**保真执行为设计属性**,含 Challenges I+II)/
> 技术1 流水线(Challenge III)/ 技术2 异步引擎(pipeline 独有)/ 实验(模板句式)。
> ⑤ (v7)实验贡献模板句式:「在 MOOC 数据集七配置上大量实验,验证流水线与异步引擎的有效性;
> 与三种自然并行化 baseline 对比,至 2.3× 加速(sync 流水线恒胜 DP 1.06–2.48×、
> async 流水线反超架构并行 1.15–1.38×),同时不影响准确率(4/4 cell 位级一致)」。
> ⑥ 挂账:「现有最先进系统」槽位暂无数据支撑(未跑 PyGT/CacheG/ESDG 等外部系统),当前填
> 「三种自然并行化 baseline」;「XX 个数据集」暂填 MOOC 单数据集,可选补 reddit/wikipedia
> 快照对照。
> ⑦ 保留挂账点:「among interactions」措辞、"selection" 级承诺;终裁(H/E 补齐)后复核。

---

> Temporal Graph Neural Networks (TGNNs) such as JODIE have achieved
> state-of-the-art performance on temporal interaction prediction tasks.
> However, designing TGNN architectures requires labor-intensive manual
> tuning over a large hyperparameter space, and existing neural
> architecture search (NAS) frameworks target static GNNs, whose training
> assumes i.i.d. samples and thus cannot handle the temporal data
> dependencies inherent in interaction streams. Searching over hundreds
> of candidate architectures demands parallel execution, yet naively
> parallelizing TGNN training breaks the read-after-write (RAW)
> dependencies among interactions, yielding biased
> architecture scores that cause NAS to select inferior architectures.
> We introduce \texttt{PipeTGL}, a pipeline-based NAS system for TGNNs.
> PipeTGL evaluates candidates through a pipeline executor, and drives
> three baselines --- serial training, data-parallel training, and
> architecture-parallel evaluation --- through the same search harness
> (identical controller, search space, and trial counts), so the
> execution strategy is the only variable across comparisons. PipeTGL is
> built on three techniques: (i) faithful execution, where parallel
> training respects the stream's RAW dependencies and a random-state
> protocol --- per-trial seed discipline, RNG-preserving state
> migration, and off-policy controller updates --- makes every backend
> reproduce the serial search's selection; (ii) a pipeline strategy that
> partitions the interaction stream into cost-balanced stages while
> preserving RAW semantics at stage boundaries; and (iii) an
> asynchronous generation engine, unique to the pipeline structure,
> that overlaps candidate generation with training in a persistent
> worker pool. A seven-cell experimental study spanning data sizes
> (20K--200K interactions), GPU counts (2--3), and search spaces
> (rnn-only and mixed) shows that the synchronous pipeline always beats
> the data-parallel baseline (1.06--2.48$\times$) and approaches the
> architecture-parallel baseline as load grows (throughput ratio
> 0.65--0.99), while the asynchronous pipeline crosses the
> architecture-parallel baseline by 1.15--1.38$\times$ and beats the
> serial baseline in every tested cell (up to 2.3$\times$); synchronous
> strategies stay bit-identical with the serial reference at 20K and
> mutually consistent at every scale, the asynchronous pipeline
> preserves the serial search's selection in all four tested cells, and
> the data-parallel baseline diverges from the reference in four of
> five cells.
