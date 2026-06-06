# 评估泄露修复总结

## 问题描述

**发现的bug**: 测试评估时模型会更新embeddings，导致数据泄露

**影响**: 测试性能被高估（Test Recall 91-99%）

---

## 根本原因

### 代码层面

**位置**: `models/jodie_rnn.py`, `process_interaction()` 方法

```python
def process_interaction(..., deferred: bool = False):
    # ... 计算新embeddings ...
    
    if not deferred:  # 默认False
        self.user_embeddings[user_ids] = new_user_emb.detach()  # 更新！
        self.item_embeddings[item_ids] = new_item_emb.detach()  # 更新！
```

**问题**: 
- 评估时调用`model()`会触发embedding更新
- 不受`model.eval()`模式影响
- 后续测试样本使用更新后的embeddings

### 泄露机制

```
测试交互1 → 更新user_1, item_A的embeddings
测试交互2 → 使用更新后的embeddings（如果涉及user_1或item_A）
测试交互3 → 使用累积更新的embeddings
...
```

**结果**: 测试性能被人为提升

---

## 修复方案

### 实现的修复

**文件**: `models/training.py`

**修改1**: 添加`frozen`参数到评估函数

```python
def evaluate_ranking_metrics(..., frozen: bool = True):  # 默认True
    ...
    for partition in ordered_partitions:
        metrics = evaluate_partition_ranking(
            model, partition, k=k, frozen=frozen
        )
```

**修改2**: 在`evaluate_partition_ranking`中实现冻结逻辑

```python
def evaluate_partition_ranking(..., frozen: bool = False):
    # 1. 保存原始embeddings
    if frozen:
        original_user_emb = model.user_embeddings.data.clone()
        original_item_emb = model.item_embeddings.data.clone()
        ...
    
    # 2. 评估时使用deferred=True
    for interaction in partition.interactions:
        if frozen:
            pred_emb, _, _ = model(..., deferred=True)  # 不更新
        else:
            pred_emb, _, _ = model(...)  # 更新（旧行为）
    
    # 3. 恢复原始embeddings
    if frozen:
        model.user_embeddings.data = original_user_emb
        model.item_embeddings.data = original_item_emb
        ...
```

**修改3**: 将`frozen=True`设为默认值

这确保所有评估默认使用正确的协议。

---

## 验证

**测试脚本**: `test_frozen_eval.py`

**结果**:
```
✅ 成功: 冻结模式正确阻止了embedding更新
```

---

## 对现有结果的影响

### 之前的结果（有泄露）

| 模式 | Test MRR | Test Recall@10 |
|------|----------|----------------|
| Serial | 0.8509 | 99.1% |
| Data Parallel | 0.6712 | 91.8% |
| Pipeline Naive | 0.6794 | 91.73% |
| Pipeline Smart | 0.6371 | 90.68% |

**问题**: 这些结果包含测试时泄露，不可信

### 预期影响

修复后，性能预计会下降：
- **Test Recall**: 从91-99%降至合理范围（可能80-90%）
- **Test MRR**: 也会相应下降

具体下降幅度取决于：
- 测试集大小
- 物品数量（21个物品，每个被更新~190次）
- 模型对动态更新的依赖程度

---

## 后续行动

### 必须做的

1. **重新运行所有实验**
   - 使用修复后的评估（frozen=True）
   - 获得正确的性能指标

2. **更新所有报告**
   - 替换旧的（有泄露的）结果
   - 使用新的（正确的）结果

### 建议做的

1. **对比分析**
   - 运行一次frozen=False（记录泄露影响）
   - 运行一次frozen=True（正确结果）
   - 量化泄露的影响程度

2. **文档说明**
   - 在论文中明确说明使用的评估协议
   - 说明为什么frozen=True是正确的

---

## 技术细节

### 为什么frozen=True是正确的？

**标准机器学习评估协议**:
1. 在训练集上训练模型
2. **冻结模型**（不再更新任何参数或状态）
3. 在测试集上评估性能

**JODIE的特殊性**:
- JODIE设计用于动态/流式场景
- 但在**离线评估**时，仍应遵循标准协议
- 只有在**在线部署**时才应该持续更新

### 什么时候用frozen=False？

**仅在以下场景**:
- 模拟在线/流式部署
- 评估模型的在线适应能力
- 明确说明使用的是在线评估协议

**不应该用于**:
- 标准论文对比
- 与其他方法的公平对比
- 报告模型的离线性能

---

## 总结

### 修复前
❌ 测试时更新embeddings（数据泄露）
❌ 性能被高估
❌ 不符合标准评估协议

### 修复后
✅ 测试时冻结embeddings（无泄露）
✅ 性能真实可信
✅ 符合标准评估协议

### 关键教训

**"严谨的测试不可能修改数据"** - 这是机器学习评估的基本原则。

任何在测试时修改模型状态的行为都是数据泄露，会导致性能被高估。
