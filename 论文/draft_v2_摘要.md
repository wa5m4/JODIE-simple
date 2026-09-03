# DepTGL 论文:摘要(assembled v4)

> **组装日期**:2026-08-28
> **来源**:`draft_v1_摘要引言构思.md` 摘要 v3 + 2026-08-28 方向性证据(双臂 + 补测)后的措辞统一
> **占位符**:"X datasets / Y model families" 待完整实验表;加速比 3.0× 为 MOOC 单数据集,扩数据集后复核。
> **相对 v3 的改动**:① "unreliable" → "biased"(方向性已单因素验证,与段 3/4 一致);② 其余未动(t-Batch 不出现、configurator 不出现、3.0×、结果级承诺)。

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
> dependencies among consecutive interactions, yielding biased
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
> training in a persistent worker pool. Extensive experiments on X
> datasets with Y model families show that DepTGL achieves up to 3.0×
> speedup over its synchronous counterpart without degrading search
> accuracy.
