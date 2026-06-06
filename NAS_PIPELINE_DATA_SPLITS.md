# NAS Pipeline数据划分详解

## 数据集初始划分

max_events=20000时的划分：
```
总数据: 20000条交互
├── Train: 14000条 (70%)
├── Val:   2000条  (10%)  
└── Test:  4000条  (20%)
```

## NAS Pipeline各阶段

### 阶段1: Coarse Search (粗搜索)
**时间**: 搜索开始
**目的**: 快速评估多个候选架构
**数据使用**:
```
训练: Train (14000条)
评估: Val (2000条)
测试: 未使用
```

**流程**:
1. Controller采样N个架构（如27个）
2. 每个架构在Train上训练K个epoch（如3个）
3. 在Val上评估，记录validation MRR
4. 按照val_score排序选出最佳架构

**日志示例**:
```
[Coarse Phase] Evaluating architectures 1-2/27
[RayPipeline] train stage 1: 1 partitions, 14000 interactions  ← Train only
```

---

### 阶段2: Rerank (可选)
**时间**: Coarse search完成后
**目的**: 对top-k候选架构进行更充分的训练和评估
**数据使用**:
```
训练: Train (14000条)
评估: Val (2000条)
测试: 未使用
```

**流程**:
1. 选出coarse phase的top-k架构（如top-5）
2. 每个架构训练更多epoch（如10个epoch）
3. 在Val上重新评估
4. 选出最佳架构

---

### 阶段3: Final Test ⭐
**时间**: 架构选择完成后（NAS pipeline最后阶段）
**目的**: 
- 使用所有可用训练数据重新训练最佳架构
- 在test集上获得无偏的性能估计
- 报告最终的test性能

**数据使用**:
```
训练: Train + Val (16000条)  ← 合并train和val
评估: 未使用
测试: Test (4000条)
```

**流程**:
1. 创建final_partition_plan: `train_interactions = train_data + val_data`
2. 在16000条数据上重新训练最佳架构
3. 在Test (4000条)上评估，得到test_mrr
4. 报告最终性能

**日志示例**:
```
[Final Test] Evaluating best architecture on test set (fit=train+val, test=test, epochs=3)
[RayPipeline] train stage 1: 1 partitions, 16000 interactions  ← Train+Val (修复后)
```

**为什么要Final Test?**
- **标准ML实践**: Val用于模型选择，Test用于最终评估
- **避免过拟合Val**: 如果直接报告val性能，可能过于乐观
- **充分利用数据**: 合并train+val可以训练更好的最终模型
- **无偏估计**: Test数据在整个搜索过程中未被使用

---

## Bug前后对比

### Bug存在时（修复前）

```
阶段1 Coarse Search:
  训练数据: 14000 (Train only)
  评估数据: 2000 (Val)
  
阶段3 Final Test:
  训练数据: 14000 (Train only) ❌ BUG！
  测试数据: 4000 (Test)
```

**问题**: Final Test声称使用train+val，但实际只用了train
**后果**: NAS在14000条数据上评估选出的架构，与重训(16000条)环境不一致

### 修复后（正确行为）

```
阶段1 Coarse Search:
  训练数据: 14000 (Train only)
  评估数据: 2000 (Val)
  
阶段3 Final Test:
  训练数据: 16000 (Train+Val) ✅ 正确！
  测试数据: 4000 (Test)
```

**效果**: Final Test与重训使用相同的训练数据量
**结果**: NAS评估分数应该与重训接近

---

## 重训(Retrain)的数据划分

重训脚本(train_single_arch.py)的数据使用：
```
训练: Train + Val (16000条)
测试: Test (4000条)
```

**这与Final Test应该一致！** 修复后两者都使用16000条训练数据。

---

## 总结

| 阶段 | 训练数据 | 评估/测试数据 | 目的 |
|------|---------|-------------|------|
| Coarse Search | 14000 (Train) | 2000 (Val) | 快速筛选架构 |
| Rerank | 14000 (Train) | 2000 (Val) | 精细化评估top-k |
| Final Test | 16000 (Train+Val) | 4000 (Test) | 最终性能报告 |
| Retrain | 16000 (Train+Val) | 4000 (Test) | 独立验证 |

**关键点**: Final Test和Retrain必须使用相同的数据划分，否则NAS选出的架构在实际部署时性能会不符合预期。
