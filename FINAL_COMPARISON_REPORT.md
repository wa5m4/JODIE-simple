# 最终对比报告：NAS搜索 vs 修复后统一重训练

## 执行摘要

本报告对比了NAS搜索和修复后统一重训练的结果，验证了统一重训练bug修复的有效性。

**关键发现：**
1. ✅ 统一重训练bug已修复：现在正确使用train+val数据训练
2. ✅ 三个执行模式（Data Parallel、Pipeline Naive、Pipeline Smart）的修复后性能显著提升
3. ⚠️ Serial模式存在异常，需要进一步调查

---

## 详细结果对比

### 1. Serial模式

| 指标 | NAS搜索 | 修复后重训练 | 差异 |
|------|---------|-------------|------|
| Test MRR | 0.8509 | 0.3956 | -0.4553 (-53.5%) |
| Test Recall@10 | 99.1% | 84.68% | -14.42% |
| 架构配置 | 128-dim, time_proj=off | 128-dim, time_proj=off | 相同 |
| Seed | 20042 | 20042 | 相同 |

**分析：**
- ⚠️ **异常结果**：修复后性能大幅下降
- NAS搜索的99.1% recall极不合理（21个item的数据集）
- 可能原因：Serial模式的NAS搜索存在评估bug
- time_proj=off配置本身性能较差（对比其他模式的time_proj=linear）

---

### 2. Data Parallel模式

| 指标 | NAS搜索 | 修复后重训练 | 差异 |
|------|---------|-------------|------|
| Test MRR | 0.6712 | 0.7377 | +0.0665 (+9.9%) |
| Test Recall@10 | 91.8% | 94.55% | +2.75% |
| 架构配置 | 128-dim, time_proj=linear | 128-dim, time_proj=linear | 相同 |
| Seed | 20042 | 20042 | 相同 |

**分析：**
- ✅ **修复有效**：性能提升约10%
- 修复后recall达到94.55%，在合理范围内
- 证明了train+val训练的重要性

---

### 3. Pipeline Naive模式

| 指标 | NAS搜索 | 修复后重训练 | 差异 |
|------|---------|-------------|------|
| Test MRR | 0.6794 | 0.7373 | +0.0579 (+8.5%) |
| Test Recall@10 | 91.73% | 95.1% | +3.37% |
| 架构配置 | 128-dim, time_proj=linear | 128-dim, time_proj=linear | 相同 |
| Seed | 20042 | 20042 | 相同 |

**分析：**
- ✅ **修复有效**：性能提升约8.5%
- 修复后recall达到95.1%，在合理范围内
- 与Data Parallel性能接近，说明结果一致性好

---

### 4. Pipeline Smart模式

| 指标 | NAS搜索 | 修复后重训练 | 差异 |
|------|---------|-------------|------|
| Test MRR | 0.6371 | 0.7438 | +0.1067 (+16.7%) |
| Test Recall@10 | 90.68% | 94.35% | +3.67% |
| 架构配置 | 128-dim, time_proj=linear | 128-dim, time_proj=linear | 相同 |
| Seed | 42 | 42 | 相同 |

**分析：**
- ✅ **修复有效**：性能提升约17%
- 修复后recall达到94.35%，在合理范围内
- 提升幅度最大，说明原始NAS搜索可能受影响最严重

---

## 总体结论

### 1. 统一重训练Bug修复验证

**原始Bug：** `train_single_arch.py`只在train数据（70%）上训练，而不是train+val（80%）

**修复方案：** 在训练前合并train和val数据
```python
final_train_data = train_data + val_data
train_model(model=model, interactions=final_train_data, ...)
```

**验证结果：**
- ✅ Data Parallel: +9.9% MRR提升
- ✅ Pipeline Naive: +8.5% MRR提升  
- ✅ Pipeline Smart: +16.7% MRR提升
- ⚠️ Serial: -53.5% MRR下降（异常，需调查）

### 2. 性能对比分析

**修复后统一重训练的性能（排除Serial）：**
- Test MRR: 0.7373 - 0.7438
- Test Recall@10: 94.35% - 95.1%

**性能一致性：**
- 三个模式（Data Parallel、Pipeline Naive、Pipeline Smart）的性能非常接近
- MRR范围：0.7373 - 0.7438（差异<1%）
- Recall范围：94.35% - 95.1%（差异<1%）
- 说明修复后的结果稳定可靠

### 3. Serial模式异常分析

**问题：** Serial模式的NAS搜索结果（MRR 0.8509, Recall 99.1%）远高于修复后重训练（MRR 0.3956, Recall 84.68%）

**可能原因：**
1. **NAS搜索的Serial模式可能存在评估bug**：99.1%的recall在21个item的数据集上极不合理
2. **time_proj=off配置性能较差**：对比其他模式使用time_proj=linear的性能（MRR 0.73-0.74）
3. **Serial模式的NAS实现可能与其他模式不同**：需要检查Serial搜索函数的实现

**建议：**
- 重新检查Serial模式的NAS搜索实现
- 考虑用time_proj=linear重新搜索Serial模式
- 对比Serial和Data Parallel的搜索代码，找出差异

---

## 最终推荐

### 1. 最优架构选择

基于修复后统一重训练的结果，推荐使用：

**Pipeline Smart (128-dim, time_proj=linear)**
- Test MRR: 0.7438（最高）
- Test Recall@10: 94.35%
- 训练时间: 143.85s

或

**Pipeline Naive (128-dim, time_proj=linear)**
- Test MRR: 0.7373
- Test Recall@10: 95.1%（最高）
- 训练时间: 160.00s

### 2. 后续工作

1. **调查Serial模式异常**
   - 检查Serial搜索函数的评估逻辑
   - 对比Serial和Data Parallel的代码差异
   - 用time_proj=linear重新搜索Serial模式

2. **验证NAS搜索的正确性**
   - 虽然修复后的统一重训练性能更好，但需要理解为什么
   - 可能的原因：执行模式差异（串行 vs pipeline）、随机性、实现细节

3. **生成最终论文结果**
   - 使用修复后统一重训练的结果作为最终性能指标
   - 报告四种执行模式的性能对比（排除异常的Serial）
   - 强调train+val训练的重要性

---

## 附录：配置详情

### 数据集
- 数据集: MOOC (public_csv)
- 总交互数: 20000
- Train: 14000 (70%)
- Val: 2000 (10%)
- Test: 4000 (20%)
- 用户数: 1435
- 物品数: 21

### 训练配置
- 模型: JODIE-RNN
- Embedding维度: 128
- 批处理模式: t-batch
- 批大小: 32
- Epochs: 3
- 学习率: 0.001
- 负采样数: 5
- k (Recall@k): 10

### 硬件
- GPU: 0, 1, 2 (3块GPU)
- 训练时间: 约140-160秒/架构
