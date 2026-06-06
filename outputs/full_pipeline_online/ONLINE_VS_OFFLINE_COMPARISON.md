# 在线评估 vs 离线评估对比报告

生成时间: 2026-05-30

## 评估模式说明

### 离线评估 (frozen=True)
- **定义**: 测试时冻结embeddings，不允许更新
- **适用场景**: 学术论文、公平对比、评估泛化能力
- **符合**: 标准机器学习评估协议
- **结果目录**: `outputs/full_pipeline`

### 在线评估 (frozen=False)
- **定义**: 测试时允许embeddings更新，模拟在线学习
- **适用场景**: 真实部署模拟、评估在线适应能力
- **注意**: 存在测试数据泄露，不符合标准ML协议
- **结果目录**: `outputs/full_pipeline_online`

---

## 结果对比汇总

| 执行模式 | 阶段 | 离线MRR | 离线Recall | 在线MRR | 在线Recall | MRR提升 |
|---------|------|---------|-----------|---------|-----------|---------|
| **Serial** | NAS搜索 | 0.2263 | 64.08% | 0.8509 | 99.10% | +276% |
| **Serial** | 重训练 | 0.1780 | 51.85% | 0.3896 | 83.25% | +119% |
| **Data Parallel** | NAS搜索 | 0.2674 | 65.38% | 0.6712 | 91.83% | +151% |
| **Data Parallel** | 重训练 | 0.2132 | 63.32% | 0.7381 | 95.25% | +246% |

---

## 详细分析

### 1. 性能提升幅度

**在线评估相比离线评估的MRR提升**：
- Serial NAS: +276% (0.23 → 0.85)
- Serial 重训: +119% (0.18 → 0.39)
- Data Parallel NAS: +151% (0.27 → 0.67)
- Data Parallel 重训: +246% (0.21 → 0.74)

**关键发现**：
- 在线评估的MRR普遍提升100-300%
- Recall提升到90-99%的高水平
- 说明JODIE在允许动态更新时性能显著提升

### 2. NAS vs 重训性能差异

#### 离线评估（frozen=True）
- Serial: NAS 0.23 → 重训 0.18 (下降21%)
- Data Parallel: NAS 0.27 → 重训 0.21 (下降20%)
- **结论**: 重训性能略低于NAS，差异在合理范围

#### 在线评估（frozen=False）
- Serial: NAS 0.85 → 重训 0.39 (下降54%)
- Data Parallel: NAS 0.67 → 重训 0.74 (上升10%)
- **异常**: Serial模式出现大幅下降，需要进一步调查

### 3. 架构选择差异

#### Serial模式
- **离线NAS**: 选择了 embedding_dim=128, time_proj=linear, memory_cell=gru
- **在线NAS**: 选择了 embedding_dim=128, time_proj=off, memory_cell=rnn
- **重训**: 使用 embedding_dim=128, time_proj=off, memory_cell=rnn

#### Data Parallel模式
- **离线NAS**: 选择了 embedding_dim=64, time_proj=off, memory_cell=gru
- **在线NAS**: 选择了 embedding_dim=128, time_proj=linear, memory_cell=rnn
- **重训**: 使用 embedding_dim=128, time_proj=linear, memory_cell=rnn

**观察**: 不同评估模式下NAS选择的架构有所不同

---

## 关键发现

### ✅ 验证了评估模式的影响

1. **在线评估性能显著更高**
   - MRR从0.18-0.27提升到0.39-0.85
   - Recall从52-65%提升到83-99%

2. **离线评估更严格**
   - MRR接近随机猜测（0.167）
   - 说明JODIE在不允许动态更新时泛化能力有限

3. **评估模式影响架构选择**
   - 不同评估模式下NAS找到的最优架构不同
   - 说明评估协议会影响架构搜索结果

### ⚠️ 发现的问题

1. **Serial在线重训性能异常低**
   - NAS: MRR 0.85 → 重训: MRR 0.39 (下降54%)
   - 可能原因：随机种子差异、训练不稳定、或架构不匹配

2. **实验未完成**
   - Pipeline Naive和Pipeline Smart模式的在线评估结果缺失
   - 需要补充完整实验

---

## 结论与建议

### 1. 评估模式选择

**学术研究/论文发表**：
- ✅ 使用离线评估（frozen=True）
- 理由：符合标准ML协议，可与其他方法公平对比
- 接受MRR较低（0.18-0.27）是真实性能

**真实部署/在线系统**：
- ✅ 使用在线评估（frozen=False）
- 理由：反映真实部署场景，评估在线适应能力
- 预期MRR较高（0.39-0.85）

### 2. 后续工作建议

**立即行动**：
1. 调查Serial在线重训性能异常（MRR 0.39 vs NAS 0.85）
2. 补充Pipeline模式的在线评估实验
3. 使用相同随机种子重新运行Serial重训，验证稳定性

**进一步研究**：
1. 分析不同评估模式下架构选择的差异
2. 研究如何在离线评估下提升JODIE性能
3. 考虑尝试其他更适合离线评估的模型（GRU4Rec、SASRec等）

### 3. 报告使用指南

**如何引用结果**：
- 论文中报告离线评估结果（MRR 0.18-0.27）
- 明确说明使用的是"离线评估协议（frozen embeddings）"
- 在线评估结果可作为补充材料，说明模型的在线适应能力

**如何解释性能差异**：
- 离线评估：评估模型的泛化能力（不依赖测试数据）
- 在线评估：评估模型的在线适应能力（持续学习）
- 两者都是有效的评估方式，取决于应用场景

---

## 附录：实验配置

### 数据集
- 名称: MOOC
- 事件数: 20,000
- 用户数: 1,435
- 物品数: 21
- 划分: 70% train, 10% val, 20% test

### 训练配置
- Epochs: 3
- Batch mode: t-batch
- Batch size: 32
- Learning rate: 0.001
- Negative samples: 5

### 评估配置
- Metric: MRR, Recall@10
- 离线模式: frozen=True（冻结embeddings）
- 在线模式: frozen=False（允许更新embeddings）

---

**报告生成时间**: 2026-05-30  
**实验目录**: 
- 离线评估: `outputs/full_pipeline`
- 在线评估: `outputs/full_pipeline_online`
