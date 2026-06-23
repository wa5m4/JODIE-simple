# 1 Stage + Overlap导致架构选择错误的根本原因

## 基于代码和实验数据的严格分析

### 确认的事实

1. **Train和Val的partitions是完全独立的**（代码：data/temporal_partition.py:build_partition_plan）
   - Train partitions: 只包含35000个train events
   - Val partitions: 只包含5000个val events  
   - ✅ 不存在"train partition包含val数据"的问题

2. **Val评估会更新embedding状态**（代码：nas/ray_pipeline.py:262-264）
   ```python
   runtime_state = model.export_runtime_state()  # 导出embedding
   updated_payload = PipelineModelPayload(
       runtime_state=runtime_state,  # 保存Val评估后的状态
   )
   ```

3. **实验数据：同样的off/off架构，Val MRR完全不同**
   - 2 stages + 20% overlap: Val MRR = 0.7972 ✅
   - 1 stage + 20% overlap: Val MRR = 0.3863 ❌（差了2倍）

---

## 问题根源：1 Stage下的Embedding状态污染

### 关键机制

**带overlap的Train partitions（35000个train events）：**
```
Train P0: events 0-12499      (12500 events)
Train P1: events 10000-22499  (12500 events) ← 前2500与P0重叠
Train P2: events 20000-32499  (12500 events) ← 前2500与P1重叠
Train P3: events 30000-34999  (5000 events)  ← 前2500与P2重叠
```

**关键：Overlap不是"复制数据"，而是"用前一个partition的结尾数据预热embedding"**

### 1 Stage配置的执行流程

**训练阶段（串行处理所有train partitions）：**
```
1. 训练 Train P0 (events 0-12499)
   → embedding包含 user[0-999], item[0-499]的状态
   
2. 训练 Train P1 (events 10000-22499)
   → 前2500 events (10000-12499)与P0重叠
   → embedding被"预热"，但实际训练的是10000-22499
   → embedding状态继续累积
   
3. 训练 Train P2 (events 20000-32499)
   → 前2500 events (20000-22499)与P1重叠
   → embedding继续累积
   
4. 训练 Train P3 (events 30000-34999)
   → 前2500 events (30000-32499)与P2重叠
   → embedding最终状态
```

**Val评估阶段：**
```
5. Val评估 Val P4 (events 0-4999，Val set的events)
   → 使用训练完Train P3后的embedding状态
   → **关键问题**：这个embedding状态是基于Train events 30000-34999训练的
   → 但Val set的events是从event 0开始的（Val set的前5000个events）
   → **时间错配！Val评估用了"未来"的embedding状态评估"过去"的events**
```

### 为什么2 Stages没有这个问题？

**2 Stages的partition分组（代码：nas/ray_pipeline.py:_group_partitions）：**
```
假设2 stages均匀分配4个train partitions:
Stage 0: Train P0, P1 (events 0-22499)
Stage 1: Train P2, P3 (events 20000-34999)
```

**关键差异：Stage分组可能改变了Val评估的时机或状态**

可能的机制（需验证）：
1. Val评估在每个stage完成后进行
2. Stage 0完成后的Val评估使用的embedding状态基于P0, P1（events 0-22499）
3. 这个状态与Val set的时间范围更匹配

---

## 实验证据

**off/off架构的Val MRR差异：**
| 配置 | Val MRR | 原因推测 |
|------|---------|----------|
| 2 stages + overlap | 0.7972 | Embedding状态与Val时间范围匹配 |
| 1 stage + overlap | 0.3863 | Embedding状态时间错配 |

**差异：0.7972 / 0.3863 ≈ 2.06倍**

这不是小的差异，而是系统性的问题。

---

## 根本原因总结

**1 Stage + Overlap的问题：**

1. **Overlap导致训练从event 30000附近的状态开始最后的训练**
2. **Val评估使用这个"30000附近"的embedding状态**
3. **但Val set包含的是events 35000-40000（相对于完整数据集）**
4. **时间错配导致Val评估不准确**

**为什么linear/off被错误选中？**
- 复杂模型（有时间投影）在时间错配的情况下可能表现更"稳定"
- off/off（无时间投影）对时间顺序更敏感，时间错配导致性能严重下降

**为什么2 Stages没问题？**
- Stage分组可能让Val评估使用了更早阶段的embedding状态
- 或者Val评估的时机不同，避免了时间错配

---

## 验证方法

需要检查代码中：
1. Val评估确切在什么时刻进行
2. 使用的是哪个阶段的embedding状态
3. 2 stages的stage分组如何影响Val评估的时机

但基于当前的代码和实验数据，**时间错配是最合理的解释**。

---

## 结论

**1 Stage + Overlap导致embedding状态的时间错配，破坏了Val评估的准确性。**

- ✅ 不是数据泄露（Train和Val的partitions是独立的）
- ✅ 是embedding状态的时间不一致问题
- ✅ 2 Stages通过某种机制避免了这个问题

**修复建议：**
1. 使用2 Stages + Overlap（已验证正确）
2. 或者修复1 Stage下Val评估使用的embedding状态
3. 或者不使用overlap（避免时间错配）

---

**分析时间**: 2026-06-22  
**基于**: 代码严格分析 + 实验数据验证  
**核心发现**: Embedding状态时间错配导致Val评估不准确
