# Pipeline流水线训练改进方案

## 问题诊断

### 核心问题及影响权重
1. **Epoch间失忆** (40%) - 每轮重置导致随机性放大、排序翻转
2. **Entity Cold Start** (35%) - 新实体训练不足、embedding质量差
3. **训练轨迹破碎** (20%) - 实体无法持续演化、后期预测质量下降
4. **评估方差大** (5%) - 单次评估不稳定

### 当前表现
- 速度：20x加速 ✓
- 准确性：Test MRR差距29% ✗
- 根本原因：**评估不准确导致架构选择错误**

---

## 改进方案（保证流水线并行）

### 方案1：Epoch间状态持久化（核心改进）

**目标**：消除"失忆重学"，稳定embedding质量

#### 实现方式
```python
# models/training.py: train_model()

for epoch in range(num_epochs):
    # 【改进】只在第一个epoch reset
    if epoch == 0:
        reset_model_state(model)
        epoch_graph_ctx = clone_graph_state_template(graph_ctx)
    # else: 继承上一epoch的状态，embedding持续累积
    
    for partition in ordered_partitions:
        train_partition(model, partition, graph_ctx=epoch_graph_ctx)
```

#### 预期效果
- ✅ Embedding质量：从"每轮随机"变为"持续累积"
- ✅ 排序稳定性：消除epoch数不同导致的排序翻转
- ✅ 速度影响：无（0%性能损失）
- ✅ 实现成本：极低（5行代码）

**预期改善：40-50%的准确性问题**

---

### 方案2：Partition重叠预热（缓解Cold Start）

**目标**：让新实体有足够的训练机会

#### 实现方式
```python
# data/temporal_partition.py

def build_partition_plan_with_overlap(
    train_interactions,
    partition_size: int,
    overlap_ratio: float = 0.2,  # 20%重叠
):
    """
    P1: [0, 5000)
    P2: [4000, 9000)  ← 与P1重叠1000个事件
    P3: [8000, 13000) ← 与P2重叠1000个事件
    """
    partitions = []
    step = int(partition_size * (1 - overlap_ratio))
    
    for i, start in enumerate(range(0, len(train_interactions), step)):
        end = min(start + partition_size, len(train_interactions))
        partition = TemporalPartition(
            partition_id=i,
            start_ts=train_interactions[start].timestamp,
            end_ts=train_interactions[end-1].timestamp,
            interactions=train_interactions[start:end],
        )
        partitions.append(partition)
        if end >= len(train_interactions):
            break
    
    return partitions
```

#### 重叠策略说明
```
无重叠（当前）:
  P1: User 1-100, Item 1-10  (5000 events)
  P2: User 101-200, Item 11-20 (5000 events) ← 新user完全cold start
  P3: User 201-300, Item 21-30 (5000 events)

20%重叠（改进）:
  P1: User 1-100, Item 1-10 (5000 events)
  P2: User 80-180, Item 8-18 (5000 events) ← 80-100是"热启动"
  P3: User 160-260, Item 16-26 (5000 events) ← 160-180是"热启动"
```

#### 预期效果
- ✅ Cold Start缓解：20%的实体从"cold"变为"warm"
- ✅ Embedding质量：新实体有更多训练机会
- ⚠️ 速度影响：轻微增加（约15%，因为partition数增加）
- ✅ 实现成本：中等（新增partition策略）

**预期改善：15-20%的准确性问题**

---

### 方案3：渐进式Micro-Epoch（改善训练轨迹）

**目标**：让实体在后续partition也能更新

#### 实现方式
```python
# models/training.py

def train_with_micro_epochs(
    model, 
    ordered_partitions,
    num_macro_epochs: int = 3,
    micro_update_ratio: float = 0.1,  # 10%的实体做微更新
):
    """
    Macro-epoch内，后续partition对前面的实体做轻量级更新
    """
    for macro_epoch in range(num_macro_epochs):
        if macro_epoch == 0:
            reset_model_state(model)
        
        for pid, partition in enumerate(ordered_partitions):
            # 主训练：当前partition的数据
            train_partition(model, partition, lr=base_lr)
            
            # 微更新：前面partition的高频实体
            if pid > 0 and micro_update_ratio > 0:
                prev_entities = sample_frequent_entities(
                    ordered_partitions[:pid], 
                    ratio=micro_update_ratio
                )
                micro_update_embeddings(
                    model, 
                    prev_entities, 
                    current_partition=partition,
                    lr=base_lr * 0.1  # 小学习率
                )
```

#### 策略说明
```
当前（无更新）:
  P1训练: User 1更新 ✓
  P2训练: User 1不更新 ✗
  P3训练: User 1不更新 ✗

改进（微更新）:
  P1训练: User 1更新 ✓
  P2训练: User 1微更新 ✓ (如果User 1在P2也有交互)
  P3训练: User 1微更新 ✓
```

#### 预期效果
- ✅ 训练轨迹：实体能持续演化
- ✅ 后期预测：质量提升
- ⚠️ 速度影响：中等增加（约20-30%）
- ⚠️ 实现成本：高（需要careful设计）

**预期改善：10-15%的准确性问题**

---

### 方案4：多重采样验证（降低评估方差）

**目标**：用ensemble降低单次评估的随机性

#### 实现方式
```python
# nas/trainer.py

def evaluate_arch_pipeline_ensemble(
    arch_config,
    partition_plan,
    num_samples: int = 3,  # 3次独立训练
):
    """
    同一架构训练多次（不同随机种子），取平均
    """
    scores = []
    for sample_id in range(num_samples):
        seed = base_seed + sample_id * 10000
        result = evaluate_arch_pipeline(
            arch_config,
            partition_plan,
            seed=seed,
        )
        scores.append(result['score'])
    
    return {
        'score': np.mean(scores),
        'score_std': np.std(scores),
        'scores': scores,
    }
```

#### 预期效果
- ✅ 评估稳定性：方差降低√3倍
- ✅ 排序可靠性：减少随机翻转
- ⚠️ 速度影响：大幅增加（3倍训练量）
- ✅ 实现成本：低（wrapper即可）

**预期改善：5-10%的准确性问题**

---

## 综合方案推荐

### 阶段1：立即实施（P0优先级）
**方案1: Epoch间状态持久化**
- 预期改善：40-50%
- 速度损失：0%
- 实现成本：极低
- **立即收益最大**

### 阶段2：短期优化（P1优先级）
**方案2: Partition重叠预热**
- 预期改善：15-20%
- 速度损失：15%
- 实现成本：中等
- **性价比高**

### 阶段3：中期增强（P2优先级）
**方案4: 多重采样验证（仅用于Rerank阶段）**
- 预期改善：5-10%
- 速度损失：对Rerank的top-15候选×3，可接受
- 实现成本：低
- **提升最终可靠性**

### 阶段4：长期研究（P3优先级）
**方案3: 渐进式Micro-Epoch**
- 预期改善：10-15%
- 速度损失：20-30%
- 实现成本：高
- **需要更多研究验证**

---

## 预期总体改善

### 当前性能
- Serial: Test MRR = 0.8626
- Pipeline Smart: Test MRR = 0.6093
- **差距：-29.4% (0.2533)**

### 改进后预期（阶段1+2）
- Serial: Test MRR = 0.8626
- Pipeline Smart (改进): Test MRR ≈ 0.82
- **预期差距：-5% (0.04)**

### 性能权衡
- 速度：20x → 约15x（仍然很快）
- 准确性：-29% → -5%（可接受）
- **实现成本：中等（主要是方案2）**

---

## 实现路线图

### Week 1: 快速验证
```bash
1. 实现方案1（Epoch间状态持久化）
2. 重跑Pipeline Smart实验（seed 1000）
3. 验证能否选出正确架构（time=off）
4. 如果验证成功 → 立即合并到主线
```

### Week 2-3: Partition重叠
```bash
1. 实现方案2（20%重叠）
2. 设计实验：对比无重叠vs有重叠
3. 调优重叠比例（10%, 20%, 30%）
4. 找到最优配置点（速度vs准确性）
```

### Week 4: 整合测试
```bash
1. 方案1+2组合测试
2. 多种子验证（seed 100, 200, 300, 1000）
3. 对比改进前后的完整指标
4. 撰写技术报告
```

---

## 技术风险与缓解

### 风险1: Epoch间不reset可能影响收敛
**缓解**:
- 可配置开关：`--no-epoch-reset`
- 对比实验验证收敛性
- 如果发现问题，可以退回传统方式

### 风险2: Partition重叠可能导致数据泄露
**缓解**:
- 重叠区域只用于训练，不用于评估
- 评估时使用完全独立的val/test split
- 严格的时间顺序保证

### 风险3: 速度损失超出预期
**缓解**:
- 分阶段实施，每阶段评估速度影响
- 方案1完全无损，可以先上
- 方案2可以根据数据集大小调整重叠比例

---

## 总结

### 核心策略
**"持久化 + 重叠 + 验证"三步走**
1. **持久化**：消除epoch间失忆（最重要）
2. **重叠**：缓解cold start（性价比高）
3. **验证**：降低评估方差（最后防线）

### 预期结果
- ✅ 保持流水线并行特性
- ✅ 速度仍有15x优势
- ✅ 准确性接近Serial（差距<5%）
- ✅ Pipeline成为真正可用的高效NAS方法

### 下一步行动
1. **立即实施方案1**（1天内完成）
2. 验证效果后决定是否继续方案2
3. 渐进式优化，避免一次性大改动
