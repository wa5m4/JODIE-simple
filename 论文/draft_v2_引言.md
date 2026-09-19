# PipeTGL 论文:引言(assembled v3)

> **组装日期**:2026-09-19(v2 = 08-28 组装)
> **来源**:v2 原文 + 2026-09-18/19 裁定与干净数据修订
> **本文件只含引言正文**,不含构思笔记。
> **v3 更新清单(2026-09-19,逐条对应 09-18 裁定与 phase5 干净数据)**:
> ① baseline 命名/引用按 09-18 裁定落位:serial = JODIE(KDD'19)、data-parallel = PyTorch
> Distributed(VLDB'20)、architecture-parallel = Ray Tune(arXiv:1807.05118)+ BOHB(ICML'18);
> ② 段 6「every backend reproduces serial selection」措辞修正:dp 已在 B/D 两 cell 分叉,
> 保真只承诺协议化后端(pipeline/async);dp 作为无协议对照组写;
> ③ 贡献 3 的「3.0×」作废(旧 C3 归因混淆了结构+异步两个变量),换干净数据的
> 1.15–1.36×(MOOC-D/E/G @100K,异步 vs 同步流水线;B@20K 为 0.88× 在方差内);
> ④ 段 3 相关工作补 TGL(VLDB'22)/ DistTGL(SC'23)/ Hogwild!(NeurIPS'11);
> ⑤ 全部加速比数字按 phase5 新口径(12t/rerank0,干净 run)重写,待 F/H 收口终裁;
> ⑥ 段 4 case study 数字保留但标「待 4b 表复核」;
> ⑦ 贡献列表对齐摘要 v7 的四条结构(系统/流水线/异步引擎/实验)。
> **数据状态**:凡标【待终裁】处 = 已填最新干净值,链条 F/H 收口(预计 09-20 晚)后统一复核。
> **重写提示**:本文件供参考改写,事实与数字以此为准,句式自拟。

---

## 段 1-2:背景与难点

> Temporal interaction data are ubiquitous in services such as
> recommendation systems, social networks, and question-answering
> platforms. Each record is a (user, item, timestamp) event. Temporal
> Graph Neural Networks (TGNNs) such as JODIE [ref] and TGN [ref]
> achieve state-of-the-art performance for future-interaction
> prediction. As each interaction arrives, they recursively update the
> embeddings of the participating user and item, so a node's latest
> embedding summarizes its interaction history.
>
> However, choosing a TGNN architecture is far from trivial. A TGNN
> exposes a large hyperparameter space (aggregation functions, temporal
> decay functions, memory cells, time projection), and manual tuning is
> labor-intensive and error-prone. Neural Architecture Search (NAS)
> automates architecture design. For static GNNs, NAS frameworks such as
> GraphNAS [ref] find architectures that rival or outperform
> hand-designed ones. Yet these frameworks cannot be directly applied to
> temporal graphs. Static GNN training assumes i.i.d. samples, so
> mini-batches can be processed in arbitrary order and in parallel. TGNN
> training is different: each interaction rewrites the embeddings of its
> user and item, and subsequent interactions must read the values just
> written, creating read-after-write (RAW) dependencies along the
> stream. TGNN training is thus inherently sequential. Since NAS must
> evaluate hundreds or thousands of candidates, serial training is
> prohibitively slow and parallel execution is essential. The central
> challenge is to enable parallel TGNN training that respects RAW
> dependencies and does not otherwise bias architecture evaluation.

## 段 3:相关工作与空白

> Several systems support TGNN training. Single-machine frameworks such
> as PyGT [ref], CacheG [ref], and PiPAD [ref] optimize caching,
> embedding reuse, and pipelined parallelism, but are limited by the
> resources of a single machine; TGL [ref] accelerates single-machine
> TGNN training with a cache built on random chunking of temporal
> dependencies. Distributed frameworks such as ESDG [ref] and DynaHB
> [ref] scale TGNN training across multiple machines through
> partitioning and communication avoidance, and DistTGL [ref] overlaps
> training with prefetching across epoch partitions. Parallel training
> at its origin is also relevant: Hogwild! [ref] first showed that
> asynchronous SGD without locks can converge in practice. These
> systems, however, train and evaluate a fixed architecture; they do not
> compare candidates. NAS, in contrast, must evaluate hundreds or
> thousands of candidates, and none of these systems provides a search
> strategy or evaluation mechanism for this purpose. To the best of our
> knowledge, no NAS framework exists for temporal GNNs — and no
> AutoML system at all targets temporal event-stream GNNs (the closest
> line of work searches spatiotemporal traffic models, a different
> task). The natural approach, then, is to parallelize candidate
> evaluation on top of existing training frameworks. As we demonstrate
> next, doing so naively violates the read-after-write (RAW)
> dependencies of temporal training. The resulting scores are so biased
> that NAS selects the wrong architecture.

## 段 4:Case study

> This failure is not hypothetical. On the MOOC dataset [ref], we ran
> the same NAS search three times. All three runs used the same
> three-worker parallel search framework; the only variable was how each
> candidate's training forms its batches: serial processing (one
> interaction at a time, in stream order), conflict-free batching (each
> batch contains unique nodes), or naive batching (consecutive
> interactions are chunked without conflict resolution). The serial
> search converges to a compact architecture
> with 133K parameters and a test MRR of 0.8561. The conflict-free
> search selects the same architecture, confirming that batching itself
> is harmless. The naive search instead selects a 147K architecture that
> relies on static embeddings; its serial re-training achieves a test
> MRR of only 0.6014, a drop of 0.2547. Consider a batch of consecutive
> interactions (u1,i1), (u1,i2), (u2,i1). Naive batching processes the
> first two against the same pre-batch embedding, so (u1,i2) misses the
> update that serial training would have applied first. The RAW
> dependency along the stream is violated [ref]. Worse, the failure is
> self-concealing: under naive evaluation the same architecture scores
> 0.96 on validation, whereas faithful evaluation yields 0.62, and the
> test MRR it reports under its own naive evaluation, 0.9335, even
> exceeds the serial search's 0.8561.
> Predictions computed from stale states systematically overrate
> architectures that rely on static features, distort the leaderboard,
> and drive NAS toward the wrong architecture.
>
> 【待复核】本段全部数字(133K/0.8561、147K/0.6014、0.96/0.62/0.9335)来自早期 RAW
> case study,与表格 4b 同源——4b 表填完时核对一次口径(协议/epoch 数)。

## 段 5:三个挑战

> **Challenge I: temporal data dependency.** Interactions that touch
> the same node form read-after-write (RAW) dependencies because
> training rewrites node embeddings as interactions arrive.
> Static NAS frameworks assume i.i.d. samples and process batches in
> arbitrary order; applied to TGNNs, out-of-order or interleaved batches
> read stale states and produce scores that do not reflect a candidate's
> true quality. Existing conflict-free batching speeds up training but
> does not guarantee faithful scores — an open problem in NAS
> evaluation.
>
> **Challenge II: evaluation fidelity.** NAS compares architectures by
> score, so the relative ranking of candidates must not depend on which
> backend evaluates them. Parallel backends evaluate candidates in
> different processes
> and pipeline stages and migrate model and optimizer state across
> process boundaries. Small deviations in random state and update
> cadence accumulate into systematic score bias, distorting the
> leaderboard and causing search to converge to an architecture favored
> by the evaluation artifact rather than by the data.
>
> **Challenge III: skewed workloads.** Temporal partitions are
> structurally uneven: the numbers of new users per partition, and
> hence the cost of processing each partition, vary widely across
> partitions (by up to 4× on MOOC), causing static worker allocation
> to leave some workers idle while others become stragglers. An
> evaluator must balance stages cheaply, without an expensive profiling
> pass per candidate.

【数字已补(2026-09-03,partition_stats.py):new users 每分区 max/min = 1.83×(20K 事件)/ 4.09×(100K)/ 114.5×(全量);事件数严格均匀(1.00,count 策略)。句内数字锚定 100K 的 "up to 4×";若论文主配置锚 20K 则改 1.8×。】

## 段 6:我们的方案

> To address these challenges, we present \texttt{PipeTGL}, an NAS
> system for temporal GNNs. PipeTGL searches a JODIE-family architecture
> space using a REINFORCE controller. All comparisons are driven through
> one search harness — identical controller, search space, and trial
> counts — so the execution strategy is the only variable across arms.
> The harness runs three natural baselines: serial training (the JODIE
> system, KDD'19 [ref]), data-parallel training (PyTorch Distributed,
> VLDB'20 [ref]), and architecture-parallel evaluation in the
> master-worker style of Ray Tune [ref] with BOHB scheduling [ref].
> Against these, PipeTGL contributes a synchronous pipeline variant
> (PipeTGL-P) and the full system (PipeTGL), which adds an asynchronous
> generation engine.
>
> Three techniques address the challenges above while improving
> efficiency. First, **faithful execution** addresses Challenges I and
> II. Parallel training respects the stream's RAW dependencies, and a
> random-state protocol (per-trial seed discipline, RNG-preserving state
> migration, off-policy controller updates) enforces evaluation fidelity
> by construction. As a result, the protocolized backends — the
> synchronous pipeline and the asynchronous engine — reproduce the
> serial search's selection in every cell tested; the data-parallel
> baseline, which has no such protocol, diverges on rnn-only search
> spaces and serves as the control. Second, a **pipeline strategy**
> addresses Challenge III by partitioning the interaction stream into
> stages and balancing them according to estimated partition cost. This
> keeps workers saturated under skewed workloads without violating RAW
> semantics at stage boundaries. Third, an **asynchronous generation
> engine** maintains a persistent worker pool in which candidate
> generation overlaps candidate training, so GPUs rarely idle and the
> search completes in a fraction of the serial wall-clock time.

## 段 7:贡献列表

> In summary, this paper makes the following contributions:
>
> - **A NAS system for temporal GNNs.** \texttt{PipeTGL} is an
>   end-to-end framework that searches a JODIE-family architecture space
>   using a REINFORCE controller and drives serial, data-parallel, and
>   architecture-parallel baselines through the same harness as its own
>   pipeline and asynchronous backends. Faithful execution is a design
>   property: parallel training respects RAW dependencies, and a
>   random-state protocol ensures that every protocolized backend
>   reproduces the serial search's selection (Challenges I and II).
> - **A pipeline strategy** that partitions the interaction stream into
>   cost-balanced stages and preserves RAW semantics at stage
>   boundaries, keeping workers saturated under skewed workloads
>   (Challenge III).
> - **An asynchronous generation engine** that overlaps candidate
>   generation with training in a persistent worker pool. On the clean
>   phase-5 measurements, it speeds up the synchronous pipeline by
>   1.15–1.36× (MOOC-D/E/G at 100K interactions) while preserving the
>   serial search's selection bit-for-bit; at 20K the gain is within
>   run-to-run variance (B: 0.88×), so the engine's benefit grows with
>   per-candidate cost.【待终裁:F/H(200K)收口后复核区间】
> - **An experimental study** over seven MOOC configurations spanning
>   data sizes (20K–200K interactions), GPU counts (2–3), and search
>   spaces (rnn-only and mixed). The full system beats the serial
>   baseline in every tested cell (up to 2.04× at 100K) and crosses the
>   architecture-parallel baseline by 1.01–1.29× (margin growing with
>   GPU count); the synchronous
>   pipeline beats data-parallel training by 1.24–1.40× in every clean
>   cell; selection is preserved bit-for-bit by the protocolized
>   backends (4/4 cells), while the data-parallel baseline diverges in
>   all three rnn-only cells.【待终裁:F/H 收口后所有区间复核】

## 段 8:论文结构

> The remainder of this paper is organized as follows. Section 2 reviews
> related work. Section 3 introduces preliminaries on temporal graphs,
> JODIE-style training, and NAS. Section 4 presents an overview of
> \texttt{PipeTGL}. Section 5 describes faithful parallel execution,
> including the pipeline strategy and asynchronous generation engine.
> Section 6 reports experimental results, and Section 7 concludes.

---

## 待办(重写前不必处理,仅备忘)

1. **F/H 收口终裁**(预计 09-20 晚):段 6/7 所有加速比区间以终稿数据复核。
2. **段 4 case study 数字**与表格 4b 核对口径后替换。
3. **引用待补**:PyGT/CacheG/PiPAD/ESDG/DynaHB/TGN/GraphNAS 具体 [ref];baseline 三条引用
   已落位(JODIE KDD'19、PyTorch Distributed VLDB'20、Ray Tune+BOHB)。
4. **数据集口径**:design v2 扩到 4 数据集后,「seven MOOC configurations」改口径。
5. **smart 身份**:实验段写架构并行时,不得再提「智慧分配」(已证伪),其定位 =
   master-worker 并行评估范式(与 Ray Tune 同构)。
