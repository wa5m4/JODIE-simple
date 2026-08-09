# Pipeline 偏差问题：实验记录与报告索引

按时间顺序排列的诊断实验和报告。

---

## 实验时间线

### 阶段 1: 问题发现（2026-08-05~06）

| 序号 | 文件 | 类型 | 说明 |
|------|------|------|------|
| 1 | `run_all.log` | 日志 | 首次全策略对比运行，发现 Pipeline test_score (0.70) 远低于 Serial (0.86) |
| 2 | `PIPELINE_BIAS_ANALYSIS.md` | 报告 | 初步偏差分析，提出 seed 差异假说 |

### 阶段 2: 诊断与种子修复（2026-08-06~07）

| 序号 | 文件 | 类型 | 说明 |
|------|------|------|------|
| 3 | `verify_pipeline_fix.py` | 诊断脚本 | 验证 Pipeline 各算子正确性 |
| 4 | `verify_optimizer_state_roundtrip.py` | 诊断脚本 | 验证 optimizer state transfer 无精度损失 |
| 5 | `diagnose_pipeline_divergence.py` | 诊断脚本 | 综合逐个算子对比 Serial vs Pipeline（仅前30步，有缺陷） |
| 6 | `run_all_seedfix.log` | 日志 | Seed 修复实验（去掉 `+ partition_id`），结果与前次完全相同 —— 证伪 seed 假说 |

### 阶段 3: 根因分析与方案 C（2026-08-08）

| 序号 | 文件 | 类型 | 说明 |
|------|------|------|------|
| 7 | `PIPELINE_DIVERGENCE_ANALYSIS.md` | 报告 | 根因分析：提出"分区规模偏差"理论 |
| 8 | `PIPELINE_RUNALL_SUMMARY.md` | 报告 | 实验结果汇总，含历史对比 |
| 9 | `test_plan_c.py` | 验证脚本 | 方案 C 快速验证（1阶段×2架构），显示排名恢复正确 |

### 阶段 4: 方案 C 全量实验（2026-08-08~09）

| 序号 | 文件 | 类型 | 说明 |
|------|------|------|------|
| 10 | `run_all_planc.log` | 日志 | 方案 C 全量运行（Pipeline分区训练+全数据评估），结果与前次完全相同 |
| 11 | `FINAL_REPORT.md` | **最终报告** | 整合所有发现的最终版本 |

### 阶段 5: 增大 partition_size + 精确诊断 v2（2026-08-09）

| 序号 | 文件 | 类型 | 说明 |
|------|------|------|------|
| 12 | `run_all_partition2000.log` | 日志 | partition_size=2000 实验，结果与 psize=500 逐项相同 |
| 13 | **`diagnose_v2_precise.py`** | ⭐ **诊断脚本** | **精确诊断 v2**：相同初始权重、所有14K步对比、验证负采样序列 → 定位 RNG 重置为根因 |

---

## 核心结论

**根因已精确到代码行：`loops.py:141` — `rng = np.random.default_rng(seed)`。**

每个 Pipeline 分区重新创建 RNG，导致负采样序列在每个分区边界从相同的起点重新开始。Serial 训练持续消费同一 RNG → 不同的负采样 → 不同的梯度 → 模型发散。这是 Ray 多进程架构的结构性限制，在保持 Pipeline 并行的前提下无法修复。

### 排除的假说

| 假说 | 实验 | 结论 |
|------|------|------|
| RNN 算子有 bug | 算子对比 + forward 验证 | ❌ max_diff=0.00 |
| State transfer 精度损失 | Roundtrip 验证 | ❌ 零损失 |
| Per-partition seed 差异 | Seed 修复 | ❌ 结果不变 |
| 分区太小 | psize 500→2000 | ❌ 逐项相同 |
| 分区评估偏差 | 全数据评估 Plan C | ❌ 结果不变 |
| **Per-partition RNG 重置** | **v2 精确诊断** | ✅ **确认根因** |

### 推荐方案

- ✅ **Data Parallel**：2x 加速 + 零排名偏差 + 选出与 Serial 完全相同的架构（test_score=0.8561）
- ⚠️ 两阶段搜索（Pipeline 粗筛 + Serial 精筛）：保留 Pipeline 速度但仍有漏筛风险

---

## 关键文件速览

| 想了解... | 看这个 |
|-----------|--------|
| 最终结论和推荐方案 | `FINAL_REPORT.md` |
| 精确根因的代码级证明 | `diagnose_v2_precise.py` |
| 详细根因链分析 | `PIPELINE_DIVERGENCE_ANALYSIS.md` |
| 实验数据和对比 | `PIPELINE_RUNALL_SUMMARY.md` |
| 初始偏差分析 | `PIPELINE_BIAS_ANALYSIS.md` |
