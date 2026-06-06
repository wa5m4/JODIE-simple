# 统一重训练Bug分析

## 问题描述

NAS搜索的Test Recall (91-99%) 比统一重训练 (86%) 高5-13%，最初认为是NAS搜索的评估bug导致指标虚高。但深入分析后发现：**统一重训练的实现有bug**。

## 根本原因

### NAS搜索的正确实现

**文件**: `nas/trainer.py`, lines 1122-1135

```python
final_train_data = train_data + val_data  # ← 合并train和val数据
final_result = self._evaluate_arch_multi_seed(
    arch_config=selected["config"],
    train_data=final_train_data,  # ← 在train+val上训练
    eval_data=test_data,           # ← 在test上评估
    ...
)
```

**正确行为**: 在最终test评估前，在train+val数据上训练模型

### 统一重训练的错误实现

**文件**: `train_single_arch.py`, lines 83-109, 124-126

```python
# Lines 83-85: 数据分割
train_data = all_interactions[:n_train]
val_data = all_interactions[n_train:n_train + n_val]
test_data = all_interactions[n_train + n_val:]

# Lines 107-117: 训练
train_model(
    model=model,
    interactions=train_data,  # ← 只在train上训练，没有使用val数据！
    ...
)

# Lines 124-126: 评估
metrics = evaluate_ranking_metrics(
    model, test_data, ...  # ← 在test上评估
)
```

**错误行为**: 只在train数据上训练，完全忽略了val数据

## 标准机器学习实践

在NAS/AutoML中，标准流程是：

1. **搜索阶段**: 
   - 在train上训练候选架构
   - 在val上评估和选择最优架构
   
2. **最终评估阶段**:
   - 在train+val上重新训练选中的架构
   - 在test上评估最终性能

**原因**: Val数据已经通过架构选择过程"泄露"了信息，所以最终训练应该使用所有可用的非test数据（train+val）。

## 性能差异解释

| 模式 | NAS Test Recall | 统一重训练 Recall | 差异 |
|------|----------------|------------------|------|
| Serial | 99.1% | - | - |
| Data Parallel | 91.8% | - | - |
| Pipeline Naive | 91.7% | 86.5% | +5.2% |
| Pipeline Smart | 90.7% | - | - |

**差异原因**: 
- NAS搜索: 在train+val (80%数据) 上训练
- 统一重训练: 只在train (70%数据) 上训练
- 少了10%的训练数据导致性能下降5%

## 结论

1. **NAS搜索的评估逻辑是正确的** - Test Recall 91-99%是真实的性能
2. **统一重训练的实现有bug** - 应该在train+val上训练，而不是只在train上
3. **之前的"评估bug修复"是误判** - 实际上没有评估bug，只是统一重训练实现错误

## 修复方案

修改 `train_single_arch.py`:

```python
# 修改前
train_model(
    model=model,
    interactions=train_data,  # ← 错误：只用train
    ...
)

# 修改后
final_train_data = train_data + val_data  # ← 合并train和val
train_model(
    model=model,
    interactions=final_train_data,  # ← 正确：用train+val
    ...
)
```
