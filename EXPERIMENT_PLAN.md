# Pipeline NAS架构选择准确性实验方案

## 实验目标

系统性地验证影响Pipeline NAS架构选择准确性的关键因素：
1. **Stage划分** (1-4 stages)
2. **Overlap比例** (0%, 10%, 20%)
3. **搜索模式** (Random vs RL)
4. **异步执行** (同步采样 vs 异步采样)

---

## 实验设计

### 控制变量

**固定参数**：
- 数据集: MOOC 50K events
- Random seed: 42 (用于数据划分和初始化)
- NAS trials: 50个架构
- 搜索空间: rnn_only
- Partition size: 12500
- 其他超参数保持一致

**评估指标**：
- 选出的最佳架构
- 最佳架构的Test MRR
- off/off架构的Val MRR (关键诊断指标)
- RL搜索路径 (如果适用)

### 已有数据复用

**可以直接使用的现有实验** (`outputs/50k_comparison/seed_42/`)：
- ✅ Serial baseline → `serial/`
- ✅ Data parallel → `data_parallel/` (或相关目录)
- ✅ 1-stage + 20% overlap → `smart_1stage/`
- ✅ 2-stages + 20% overlap → `smart_overlap20/`
- ✅ 3-stages + 0% overlap → `naive_3stages/`

**需要补充的实验**：
- 4-stages配置
- 不同overlap比例 (0%, 10%)
- 交叉验证组合
- Random搜索对比

---

## 实验组设计

### 第一部分：基准组 (2个实验)

验证Serial和数据并行作为ground truth。

| ID | 名称 | 执行模式 | Stage | Overlap | 搜索模式 | 预期结果 |
|----|------|----------|-------|---------|----------|----------|
| **B1** | Serial baseline | Serial | N/A | N/A | RL | off/off (Test MRR≈0.80) |
| **B2** | Data parallel | DataParallel | N/A | N/A | RL | off/off (Test MRR≈0.80) |

**目的**：确认正确答案，作为其他实验的对照。

---

### 第二部分：Stage数量实验 (4个实验)

**固定**：Overlap=20%, 搜索模式=RL

| ID | 名称 | Stage数量 | Overlap | 预期off/off Val MRR | 预期选出架构 |
|----|------|-----------|---------|---------------------|--------------|
| **S1** | Pipeline 1-stage | 1 | 20% | ~0.38 (低估) | linear/off (错误) |
| **S2** | Pipeline 2-stages | 2 | 20% | ~0.80 (正常) | off/off (正确) |
| **S3** | Pipeline 3-stages | 3 | 20% | ~0.60-0.70? | 待验证 |
| **S4** | Pipeline 4-stages | 4 | 20% | 待验证 | 待验证 |

**目的**：
- 验证stage数量对off/off评估准确性的影响
- 找出最优stage数量
- 验证"2 stages是最佳平衡"的假设

---

### 第三部分：Overlap比例实验 (3个实验)

**固定**：Stage=2, 搜索模式=RL

| ID | 名称 | Stage | Overlap比例 | 预期off/off Val MRR | 预期选出架构 |
|----|------|-------|-------------|---------------------|--------------|
| **O1** | 2-stages no-overlap | 2 | 0% | ~0.60-0.70 (中等) | 可能错误 |
| **O2** | 2-stages 10%-overlap | 2 | 10% | 待验证 | 待验证 |
| **O3** | 2-stages 20%-overlap | 2 | 20% | ~0.80 (正常) | off/off (正确) |

**目的**：
- 验证overlap对评估准确性的影响
- 找出最优overlap比例
- 验证"20% overlap提供最佳连续性"的假设

---

### 第四部分：Stage×Overlap交叉验证 (关键组合，4个实验)

验证不同stage和overlap组合的交互效应。

| ID | 名称 | Stage | Overlap | 预期结果 | 验证假设 |
|----|------|-------|---------|----------|----------|
| **C1** | 1-stage no-overlap | 1 | 0% | off/off低估 | overlap污染 vs 连续性 |
| **C2** | 3-stages no-overlap | 3 | 0% | off/off中等低估 | 已有数据：0.588 |
| **C3** | 3-stages 20%-overlap | 3 | 20% | 待验证 | reset过度 vs 连续性 |
| **C4** | 4-stages 10%-overlap | 4 | 10% | 待验证 | 多stage + 少overlap |

**目的**：
- 验证stage和overlap的交互效应
- 确认"2 stages + 20% overlap"是唯一最优组合还是有其他组合也可行

---

### 第五部分：搜索模式对比 (2个实验)

验证Random vs RL搜索的差异。

| ID | 名称 | Stage | Overlap | 搜索模式 | 预期结果 |
|----|------|-------|---------|----------|----------|
| **M1** | 2-stages Random | 2 | 20% | Random | 可能错误 (采样运气) |
| **M2** | 2-stages RL | 2 | 20% | RL | off/off (正确) |

**目的**：
- 验证搜索模式是否影响结果
- 确认RL被错误reward误导的假设
