# DepTGL 论文:引言(assembled v2)

> **组装日期**:2026-08-28;当日逐句批改(外部修改意见,采纳记录见 v1 文件"九、逐句批改")
> **来源**:`draft_v1_摘要引言构思.md` 各段最新候选稿(构思、批改、观察记录保留在 v1 文件)
> **本文件只含引言正文**,不含构思笔记。
> **占位符清单**:`[ref]` 引用待补;Challenge III 数字已补(up to 4×,锚 100K;2026-09-03);加速比 3.0× 为 MOOC 单数据集,扩数据集后复核。
> **段状态**:段 1-3 = 08-14 初稿 + 08-28 两轮逐句批改;段 4 = 08-28 v5(数字已实锤)+ 两轮逐句批改(第二轮:实验设置句改写消歧);段 5 = 08-27 候选稿 + 08-28 Challenge I 末句裁决(不点名 JODIE)+ 两轮逐句批改 + 09-03 Challenge III 数字补入;段 6 = 08-27 候选稿 + 两轮逐句批改(承诺软化:selection 替代 scores);段 7-8 = 08-28 首稿 + 两轮逐句批改。

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

> Several frameworks support TGNN training. Single-machine frameworks
> such as PyGT [ref], CacheG [ref], and PiPAD [ref] optimize caching,
> embedding reuse, and pipelined parallelism, but are limited by the
> resources of a single machine. Distributed frameworks such as ESDG
> [ref] and DynaHB [ref] scale TGNN training across multiple machines
> through partitioning and communication avoidance. These systems,
> however, train and evaluate a fixed architecture; they do not compare
> candidates. NAS, in contrast, must evaluate hundreds or thousands of
> candidates, and none of these systems provides a search strategy or
> evaluation mechanism for this purpose. To the best of our knowledge,
> no NAS framework exists for temporal GNNs. The natural approach, then,
> is to parallelize candidate evaluation on top of existing training
> frameworks. As we demonstrate next, doing so naively violates the
> read-after-write (RAW) dependencies of temporal training. The
> resulting scores are so biased that NAS selects the wrong
> architecture.

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

【数字已补(2026-09-03,partition_stats.py):new users 每分区 max/min = 1.83×(20K 事件)/ 4.09×(100K)/ 114.5×(全量);事件数严格均匀(1.00,count 策略)。句内数字锚定 100K 的 "up to 4×";若论文主配置锚 20K 则改 1.8×。原 "interaction counts vary widely" 与 count 策略矛盾,已删。】

## 段 6:我们的方案

> To address these challenges, we present \texttt{DepTGL}, an NAS
> system for temporal GNNs. DepTGL searches a JODIE-family architecture
> space using a REINFORCE controller and evaluates candidates through
> interchangeable execution backends (serial, data-parallel, pipeline,
> asynchronous) that share a common evaluation interface. Three
> techniques address these challenges while improving efficiency. First,
> **faithful execution** addresses Challenges I and II. Parallel
> training respects the stream's RAW dependencies, and a random-state
> protocol enforces evaluation fidelity by construction. As a result,
> every backend reproduces the serial search's selection. Second, a
> **pipeline strategy**
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
> - **A NAS system for temporal GNNs.** \texttt{DepTGL} is an end-to-end
>   framework that searches a JODIE-family architecture space using a
>   REINFORCE controller and evaluates candidates through
>   interchangeable execution backends (serial, data-parallel, pipeline,
>   asynchronous). Faithful execution is a design property: parallel
>   training respects RAW dependencies, and a random-state protocol
>   ensures that every backend reproduces the serial search's selection
>   (Challenges I and II).
> - **A pipeline strategy** that partitions the interaction stream into
>   cost-balanced stages and preserves RAW semantics at stage
>   boundaries, keeping workers saturated under skewed workloads
>   (Challenge III).
> - **An asynchronous generation engine** that overlaps candidate
>   generation with training in a persistent worker pool, achieving a
>   3.0× speedup over the synchronous pipeline without degrading search
>   accuracy.

## 段 8:论文结构

> The remainder of this paper is organized as follows. Section 2 reviews
> related work. Section 3 introduces preliminaries on temporal graphs,
> JODIE-style training, and NAS. Section 4 presents an overview of
> \texttt{DepTGL}. Section 5 describes faithful parallel execution,
> including the pipeline strategy and asynchronous generation engine.
> Section 6 reports experimental results, and Section 7 concludes.
