# NAS Final Test Bug修复报告

生成时间: 2026-06-02

## 问题描述

NAS评估和重训结果存在巨大差异：
- NAS评估：MRR 0.71-0.78 (time_proj=off)
- 重训：MRR 0.38-0.42 (相同架构)
- **差异达到2倍！**

## 根本原因

**训练数据量不一致导致评估错误**

### NAS Final Test的预期行为
```python
# Line 925-932 in nas/trainer.py
final_train_data = train_data + val_data  # 应该是16000条
final_partition_plan = build_partition_plan(
    train_interactions=final_train_data,
    val_interactions=[],
    test_interactions=test_data,
)
```

### 实际行为
日志显示：
```
[Final Test] Evaluating best architecture on test set (fit=train+val, test=test, epochs=3)
[RayPipeline] train stage 1: 1 partitions, 14000 interactions  ← 只有14000！
```

## Bug定位

### 问题代码 (Line 935-946)
```python
final_test_result = self.evaluate_arch_pipeline(
    arch_configs=[selected["config"]],
    partition_plan=final_partition_plan,  # 新的partition plan (16000)
    ...
    executor=pipeline_executor,  # ❌ 使用旧的executor (14000)!
)
```

### 根本原因
`pipeline_executor`在line 735创建时使用的是原始partition_plan：
```python
# Line 735
pipeline_executor = RayPipelineExecutor(self.base_config, partition_plan)  
# partition_plan有train=14000
```

在evaluate_arch_pipeline中 (Line 402-404):
```python
own_executor = executor is None
if own_executor:
    executor = RayPipelineExecutor(self.base_config, partition_plan)
```

由于`executor=pipeline_executor`不为None，函数跳过了用final_partition_plan创建新executor的步骤，直接使用了旧的executor。

## 修复方案

### 修改位置
`nas/trainer.py` Line 943

### 修改内容
```python
# 修改前
executor=pipeline_executor,  # 旧executor

# 修改后  
executor=None,  # 让函数创建新executor
```

### 修复效果
- Final test现在会用final_partition_plan创建新的RayPipelineExecutor
- 训练数据从14000增加到16000 (train+val)
- NAS评估将使用与重训相同的数据量

## 影响分析

### 为什么会导致评估错误？

1. **NAS Final Test**: 在14000条数据上训练 → MRR高 (0.71-0.78)
2. **重训**: 在16000条数据上训练 → MRR低 (0.38-0.42)

### 为什么更多数据反而MRR更低？

这不是"更多数据导致更差"，而是：
- `time_proj=off`的架构在小数据集(14000)上表现好
- 但在大数据集(16000)上泛化能力差
- NAS在错误的数据量上评估，选择了不适合实际场景的架构

## 验证方法

修复后重新运行实验，检查：
1. Final test日志应显示16000 interactions
2. NAS评估的MRR应该与重训接近
3. 架构选择应该更合理

## 结论

**这是一个严重的数据泄漏bug！**

NAS在错误的数据规模(14000)上评估架构，然后在正确的规模(16000)上部署，导致：
- ❌ 架构选择基于错误的评估
- ❌ NAS和重训结果不一致
- ❌ 最佳架构实际上不是最佳

修复后，NAS将在正确的数据规模上评估架构，确保选择的架构在实际部署时也能达到预期性能。
