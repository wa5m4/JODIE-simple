# PipeTGL 论文:摘要(assembled v8)

> **组装日期**:2026-09-19(v7 = 09-15)
> **来源**:v7 原文 + 2026-09-18/19 裁定与 phase5 干净数据修订
> **v8 更新清单(逐条对应 v7)**:
> ① 头条 2.3× → **2.04×**(锚 MOOC-G 100K mixed,phase5 新口径 12t/rerank0;v7 的
> D 2.32×/F 2.31× 旧锚作废);
> ② 反超架构并行 1.15–1.38× → **1.01–1.29×**(B/D/E/G 干净数据,随卡数增长:
> 2 卡 1.01×、3 卡 1.16–1.29×);
> ③ sync 恒胜 DP 1.06–2.48× → **1.24–1.40×**(新口径,结论不变);
> ④ 「async 在 4/4 cell 位级保持选择」→ **4/4**(B/D/E/G;C 未跑 async);
> ⑤ 「dp 在 4/5 cell 分叉」→ **rnn-only 3/3 分叉 + mixed 1/1 一致**(D/E 分叉架构
> 完全同构 464,192,可复现签名);
> ⑥ 「makes **every** backend reproduce the serial search's selection」→ 改为
> **协议化后端**(pipeline/async)保真,dp 为无协议对照组(已分叉);
> ⑦ baseline 引用落位:serial = JODIE(KDD'19)、data-parallel = PyTorch Distributed
> (VLDB'20)、architecture-parallel = Ray Tune(arXiv:1807.05118)+ BOHB(ICML'18);
> ⑧ v7 的「sync 接近架构并行(throughput 0.65–0.99)」改为诚实版:sync 在 20K 领先
> 架构并行(1.33×)、在 100K 落后(0.89–0.96×),反超由 async 完成——这是全文最诚实的
> 数据叙事,勿再写旧区间。
> **挂账(v7 ⑥⑦ 保留)**:「现有最先进系统」槽位暂无数据支撑,当前填「三种自然并行化
> baseline」;「XX 个数据集」暂填 MOOC 单数据集(design v2 扩 4 数据集后改口径)。
> **数据状态**:凡标【待终裁】处 = 已填最新干净值,F/H(200K)收口(预计 09-20 晚)后统一复核。
> **重写提示**:本文件供参考改写,事实与数字以此为准,句式自拟。

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
> three natural baselines --- serial training (JODIE, KDD'19),
> data-parallel training (PyTorch Distributed, VLDB'20), and
> architecture-parallel evaluation (the master-worker style of Ray Tune
> with BOHB scheduling) --- through the same search harness (identical
> controller, search space, and trial counts), so the
> execution strategy is the only variable across comparisons. PipeTGL is
> built on three techniques: (i) faithful execution, where parallel
> training respects the stream's RAW dependencies and a random-state
> protocol --- per-trial seed discipline, RNG-preserving state
> migration, and off-policy controller updates --- makes the
> protocolized backends (the pipeline and the asynchronous engine)
> reproduce the serial search's selection, while the protocol-free
> data-parallel baseline diverges on rnn-only spaces and serves as the
> control; (ii) a pipeline strategy that
> partitions the interaction stream into cost-balanced stages while
> preserving RAW semantics at stage boundaries; and (iii) an
> asynchronous generation engine, unique to the pipeline structure,
> that overlaps candidate generation with training in a persistent
> worker pool. A seven-cell experimental study spanning data sizes
> (20K--200K interactions), GPU counts (2--3), and search spaces
> (rnn-only and mixed) shows that the synchronous pipeline always beats
> the data-parallel baseline (1.24--1.40$\times$), leads the
> architecture-parallel baseline at 20K but trails it at 100K
> (0.89--0.96$\times$), while the asynchronous pipeline crosses the
> architecture-parallel baseline by 1.01--1.29$\times$ (the margin
> grows with GPU count) and beats the
> serial baseline in every tested cell (up to 2.04$\times$); the
> protocolized backends stay bit-identical with the serial reference at
> every tested cell (4/4 for the asynchronous pipeline), and the
> data-parallel baseline diverges in all three rnn-only cells.
> 【待终裁:F/H 收口后复核全部区间】
