# Pipeline / Serial / Data Parallel / run_all 对比总结

## 结论先说（2026-08-08 更新 — seed 修复后重新实验）

**seed 修复（去掉 `+ partition_id`）没有改变 Pipeline Naive 的搜索结果。真正的根因是 Pipeline 分区训练（500 interactions/partition）系统性地偏向超参数化架构。**

| 策略 | test_score | 与 Serial 差距 | 选出架构 | 参数量 |
|------|-----------|---------------|---------|--------|
| Serial | **0.8561** | — | all features off | 133K |
| Data Parallel | **0.8561** | 0 | 与 Serial 相同 | 133K |
| Pipeline Naive (修复后) | **0.7000** | **-0.156** | all features on | 402K |
| Pipeline Smart | 待完成 | — | — | — |

核心发现：**Serial 最优架构（133K, all off）在 Pipeline 评分中仅排第 28/59 名（score=0.496）。** 同一个架构，Serial 评第 1、Pipeline 评第 28。这是系统性的排名逆转，不是随机噪声。

详见 [PIPELINE_DIVERGENCE_ANALYSIS.md](PIPELINE_DIVERGENCE_ANALYSIS.md)。

## 实验配置

这份总结按 `run_all.py` 的配置家族整理：
- 搜索空间：`rnn_only`
- 搜索模式：`rl`
- 数据集：`public_csv` / `data/public/mooc.csv`
- 粗搜索：`50 trials × 2 epochs`
- 重排序：`top 8 × 5 epochs`
- 选择指标：`mrr`
- 训练模式：`tbatch`
- 随机种子：`42`
- 分区大小：`500`
- GPU 列表：`0,1,2`
- Pipeline Naive：`3 stages × 1,1,1 workers`
- Pipeline Smart：`auto`

## 实验结果历史

### 20260807_221920（seed 修复后，2026-08-08）

| 策略 | 执行时间(s) | val_score | test_score | 参数量 | 选出架构特征 |
|------|------------|-----------|------------|--------|-------------|
| Serial | ~17,900 | 0.8648 | 0.8561 | 133,888 | time_proj=off, static_emb=off, norm=off |
| Data Parallel | ~9,500 | 0.7535 | 0.8561 | 133,888 | 与 Serial 相同 |
| Pipeline Naive | ~10,400 | 0.7802 | 0.7000 | 402,176 | time_proj=linear, static_emb=on, norm=on |
| Pipeline Smart | 待完成 | — | — | — | — |

### 20260806_161217（seed 修复前，同一配置）

| 策略 | test_score | 参数量 | 选出架构特征 |
|------|-----------|--------|-------------|
| Serial | 0.8561 | 133K | time_proj=off, static_emb=off, norm=off |
| Data Parallel | 0.8561 | 133K | 与 Serial 相同 |
| Pipeline Naive | 0.7000 | 402K | time_proj=linear, static_emb=on, norm=on |
| Pipeline Smart | 0.6687 | 133K | time_proj=linear, norm=on |

### Pipeline Naive 修复前后对比：完全相同

| 指标 | 修复前 (20260806) | 修复后 (20260807) |
|------|------------------|-------------------|
| test_score | 0.7000 | **0.7000** |
| 架构 | all features on, 402K | **完全相同** |
| val_score | 0.7802 | **0.7802** |

**seed 修复没有改变任何结果。** per-partition seed 不是根因。

## 根因分析（2026-08-08 修正）

### 执行层修复（已完成 ✅）

`jodie/nas/ray_pipeline.py` 中的改动：
- Optimizer state FQN 传输
- Epoch / stage 的 seed 传递补齐
- Multi-epoch 调度边界修补
- Payload 拷贝
- graph_state 处理

这些修复已在 `diagnose_pipeline_divergence.py` 中验证通过。

### 之前的错误结论（已证伪 ❌）

> ~~"Pipeline 与 Serial 的差异 100% 来自 per-partition seed"~~

这个结论被 seed 修复实验证伪。去掉 `+ partition_id` 后 Pipeline Naive 结果与修复前完全相同。

### 真正的根因：分区规模偏差 ✅

**Pipeline 搜索和 Serial 最终评估回答的是不同的问题：**

| 评估阶段 | 问题 | 数据量 |
|----------|------|--------|
| Pipeline (搜索) | "哪种架构在 500 个交互上训练 2 epoch 后验证分数最高？" | 500/partition |
| Serial (最终) | "哪种架构在 14,000 个交互上训练 5 epoch 后测试分数最高？" | 14,000 total |

**小数据（500 interactions）→ 超参数化模型占优（402K）**
- static_embeddings 提供直接的用户/物品记忆
- time_proj=linear 增强局部时序拟合
- 更多参数 → 更快拟合小数据

**大数据（14,000 interactions）→ 极简模型占优（133K）**
- RNN 动态嵌入足够捕捉时序模式
- 不需要额外的 static embeddings
- 更少参数 → 更好泛化

**关键证据**：Serial 最优架构 `{time_proj=off, static_emb=off, norm=off}` 在 Pipeline 评分中仅排 28/59（score=0.496），而在 Serial 中排第 1（score=0.865）。

### Data Parallel 为什么没有这个问题

Data Parallel 每个 worker 看到完整数据的随机子集，梯度同步保持全局优化方向，且 val 评估基于全数据。因此 DP 的架构排名与 Serial 一致。

## 修复建议

1. **增加 partition_size**：从 500 → 2000-5000，减小每个分区与全数据的分布差距
2. **两阶段搜索**：Pipeline 粗筛（100 trials × 1 epoch）→ Serial 精筛（top 20 × 2 epochs）
3. **全数据评估**：Pipeline 分区训练（快）+ 全数据 val 评估（准）
4. **短期替代方案**：Data Parallel 搜索速度已是 Serial 的 2x，且没有排名偏差
