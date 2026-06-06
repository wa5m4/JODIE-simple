# 评估隔离性分析报告

生成时间: 2026-05-30

## 问题背景

用户担心：在线评估模式（frozen=False）下，一个模式的评估可能会更新embeddings，导致后续模式使用被污染的数据。

## 隔离性验证

### 1. 不同执行模式之间的隔离 ✅

**问题**: Serial评估后更新embeddings → Data Parallel使用更新后的embeddings？

**验证结果**: ❌ **不会发生**

**原因**:
```python
# run_full_nas_pipeline.py
for mode in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]:
    # 每个模式调用独立的subprocess
    subprocess.run(["python", "search.py", ...])
```

- 每个模式运行在独立的Python进程中
- 进程之间不共享内存
- 每个进程加载独立的数据和模型

**结论**: ✅ **完全隔离，无污染风险**

---

### 3. frozen=False的真正含义

**frozen=False并不是指跨模型/跨trial的污染，而是指单个模型评估过程中的行为**

#### 在单个模型的评估过程中：

**frozen=True (离线评估)**:
```python
# 评估前保存embeddings
original_embeddings = model.embeddings.clone()

for interaction in test_data:
    pred = model(interaction, deferred=True)  # 不更新embeddings
    compute_metric(pred, interaction.item_id)

# 评估后恢复embeddings
model.embeddings = original_embeddings
```

**frozen=False (在线评估)**:
```python
for interaction in test_data:
    pred = model(interaction, deferred=False)  # 更新embeddings
    compute_metric(pred, interaction.item_id)
    # embeddings已更新，下一个interaction使用新的embeddings
```

**关键区别**:
- frozen=False: 测试集的第1000个样本使用的embeddings，已经被前999个样本更新过
- frozen=True: 测试集的所有样本都使用训练结束时的embeddings

---

## 最终结论

### ✅ 评估完全隔离，无污染

1. **不同执行模式之间**: 独立进程，完全隔离
2. **同一模式不同trial之间**: 每个trial创建全新模型，完全隔离
3. **frozen=False的影响**: 仅限于单个模型的单次评估过程内部

### frozen=False的正确理解

**不是**: Serial的评估影响Data Parallel的评估  
**而是**: 在单个模型的评估过程中，后面的测试样本使用前面样本更新后的embeddings

### 为什么这是合理的？

**在线评估模式(frozen=False)的目的**:
- 模拟真实在线部署场景
- 评估模型的在线适应能力
- 允许模型在测试时持续学习

**离线评估模式(frozen=True)的目的**:
- 符合标准ML评估协议
- 评估模型的泛化能力
- 不依赖测试数据

---

**报告生成时间**: 2026-05-30  
**结论**: 当前实现的评估隔离性是正确的，无需修改


**问题**: Trial 1评估后更新embeddings → Trial 2使用更新后的embeddings？

**验证结果**: ❌ **不会发生**

**关键代码** (nas/trainer.py:161):
```python
def _train_and_eval(self, config, train_data, eval_data, ...):
    self._set_seed(trial_seed)
    model = build_model(config)  # ✅ 每个trial创建全新模型
    
    # 训练
    train_model(model, train_data, ...)
    
    # 评估
    metrics = evaluate_ranking_metrics(model, eval_data, frozen=...)
    
    return metrics
    # model被销毁，不会影响下一个trial
```

**隔离机制**:
1. 每个trial调用`build_model()`创建全新模型
2. 模型包含全新的、随机初始化的embeddings
3. Trial结束后，模型对象被销毁
4. 下一个trial从零开始

**结论**: ✅ **完全隔离，无污染风险**

