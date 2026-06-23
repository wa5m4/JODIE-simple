# 为什么多Stage + Overlap正确，1 Stage + Overlap错误？

## 问题现象

| 配置 | Stages | Overlap | 选出架构 | Test MRR | 状态 |
|------|--------|---------|----------|----------|------|
| Serial | - | 0% | off/off | 0.8012 | ✅ |
| 数据并行 | - | 0% | off/off | 0.8012 | ✅ |
| Smart (2 stages) | 2 | 20% | off/off | 0.8012 | ✅ |
| **Smart (1 stage)** | **1** | **20%** | **linear/off** | **0.6715** | **❌ 错误** |

---

## 技术分析

### 1. 数据划分现状

**50K数据的时间顺序划分：**
```
Train: events 0-34999    (35000 events, 70%)
Val:   events 35000-39999 (5000 events, 10%)
Test:  events 40000-49999 (10000 events, 20%)
```

**Partition划分 (size=12500)：**
```
Partition 0: events 0-12499      [全Train]
Partition 1: events 12500-24999  [全Train]
Partition 2: events 25000-37499  [Train + Val，边界在35000]
Partition 3: events 37500-49999  [Val + Test，边界在40000]
```

### 2. 20% Overlap的具体影响

**Overlap机制：**
- overlap_ratio = 0.2
- step = partition_size × (1 - overlap_ratio) = 12500 × 0.8 = 10000
- 每个partition向前滑动10000，而不是12500

**实际划分（带overlap）：**
```
Partition 0: events 0-12499      (12500 events)
Partition 1: events 10000-22499  (12500 events) ← 前2500与P0重叠
Partition 2: events 20000-32499  (12500 events) ← 前2500与P1重叠
Partition 3: events 30000-42499  (12500 events) ← 前2500与P2重叠
Partition 4: events 40000-49999  (10000 events) ← 前2500与P3重叠
```

**等等！我之前理解错了！**

让我重新计算带overlap的partition划分：

```python
step = 12500 × 0.8 = 10000
Partition 0: start=0,     end=12500   → events 0-12499
Partition 1: start=10000, end=22500   → events 10000-22499
Partition 2: start=20000, end=32500   → events 20000-32499
Partition 3: start=30000, end=42500   → events 30000-42499
Partition 4: start=40000, end=50000   → events 40000-49999
```

### 3. 关键发现：Val数据泄露

**Partition 3: events 30000-42499**
- Train结束于event 34999
- Val: 35000-39999
- Partition 3包含：
  - Train数据：30000-34999 (5000 events)
  - **Val数据：35000-39999 (5000 events)** ⚠️
  - Test数据：40000-42499 (2500 events)

**这意味着：在训练Partition 3时，Val数据被当作训练数据使用了！**

### 4. 为什么2 Stages没问题，1 Stage有问题？

#### 假设：2 Stages的划分方式

可能的stage划分：
```
Stage 0: Partitions 0, 1    (全Train数据)
Stage 1: Partitions 2, 3, 4 (包含Val和Test)
```

或者：
```
Stage 0: Partitions 0, 1, 2 (Train + 一部分Val)
Stage 1: Partitions 3, 4    (Val + Test)
```

**关键假设：Val评估可能只在特定partitions上进行**

如果Val评估的逻辑是：
1. 只评估Val set的events（35000-39999）
2. 在处理完包含Val数据的partitions后评估

那么：

**2 Stages时：**
- Partition 3在Stage 1中
- 如果Stage 1的某些机制阻止了Val数据参与训练
- 或者Val评估在更早的阶段进行（在Partition 3之前）
- Val评估可能是准确的

**1 Stage时：**
- 所有partitions串行处理：P0 → P1 → P2 → P3 → P4
- Partition 3明确包含了Val数据用于训练
- 当Val评估时，模型已经见过Val数据
- **导致Val评估不准确**

### 5. 数据泄露的影响

**Val数据泄露的后果：**

1. **过拟合Val set**
   - 模型在训练Partition 3时见过Val数据
   - Val MRR被人为提高
   - 不能真实反映泛化能力

2. **架构选择偏差**
   - 更复杂的模型（linear/off）能更好地"记住"Val数据
   - 在被污染的Val评估中得分更高
   - 但在真实的Test set上表现差

3. **off/off为什么被低估？**
   ```
   1 Stage配置下：
   - off/off在被污染的Val上：Val MRR=0.3863 (Rank 19)
   - linear/off在被污染的Val上：Val MRR=0.6440 (Rank 1) ✅ 选中
   
   但在真实Test上：
   - off/off: Test MRR=0.8012 (实际最优)
   - linear/off: Test MRR=0.6715 (实际较差)
   ```

### 6. 为什么2 Stages能避免这个问题？

**可能的机制（需要验证）：**

#### 机制1：Stage-level的数据隔离
- 每个stage只处理特定的partitions
- Val评估可能在stage边界进行
- Stage 0完成后的Val评估不受Stage 1的影响

#### 机制2：Val评估的时机不同
- 2 stages时，Val评估可能在更早的checkpoint进行
- 避免了使用Partition 3训练后的模型状态

#### 机制3：Partition分组避免了跨边界泄露
- 2 stages的partition分组恰好避免了Val数据泄露
- 或者overlap在stage边界处的处理方式不同

### 7. 验证方法

需要检查的关键点：

1. **Val评估的具体实现**
   ```python
   # 在哪个阶段评估？
   # 使用哪些partitions的数据？
   # 使用哪个时刻的模型状态？
   ```

2. **Stage划分对Val评估的影响**
   ```python
   # 2 stages如何划分partitions？
   # Val评估是在stage内还是stage间？
   ```

3. **Overlap在不同stage配置下的行为**
   ```python
   # 1 stage: 所有partitions的overlap如何处理？
   # 2 stages: stage边界的overlap如何处理？
   ```

---

## 结论

### 根本原因

**1 Stage + 20% Overlap导致Val数据泄露：**

1. ✅ Overlap机制使得Partition 3包含了Val数据（35000-42499）
2. ✅ 1 Stage配置下，Partition 3的数据被用于训练
3. ✅ Val评估使用了被污染的数据，导致选错架构
4. ❌ 选出linear/off（在污染Val上表现好）
5. ❌ 忽略off/off（在污染Val上表现差，但在Test上最优）

**2 Stages + 20% Overlap避免了问题：**

1. ✅ Stage划分可能隔离了Val数据的影响
2. ✅ 或者Val评估的时机避免了使用Partition 3后的状态
3. ✅ 正确选出off/off

### 修复建议

1. **短期：使用2 Stages配置**
   - 已验证：2 stages + 20% overlap正确
   - 避免使用1 stage + overlap的组合

2. **中期：修复1 Stage + Overlap的数据泄露**
   - 确保Val数据不被用于训练
   - 或者在Val评估时使用未见过Val的checkpoint

3. **长期：重新设计Partition和Val评估机制**
   - Partition边界应该对齐Train/Val/Test边界
   - 或者Val评估应该独立于partition训练

---

**文档创建时间**: 2026-06-22  
**核心发现**: 1 Stage + Overlap导致Val数据泄露，破坏架构选择准确性
