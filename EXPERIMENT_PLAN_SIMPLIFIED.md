# Pipeline NAS Stage数量验证实验（简化版）

## 实验目标

验证Pipeline的Stage划分对架构选择准确性的影响。

**固定参数**：
- Overlap: 20% (固定，不测试不同比例)
- 搜索模式: RL (固定)
- 数据集: MOOC 50K events
- Seed: 42
- Trials: 50个架构

## 实验设计

### 第一部分：基准组 (2个实验)

作为ground truth对照。

| ID | 执行模式 | 预期结果 |
|----|----------|----------|
| **B1** | Serial | 选出off/off，Test MRR≈0.80 |
| **B2** | Data Parallel | 选出off/off，Test MRR≈0.80 |

### 第二部分：Stage数量实验 (4个实验)

固定：Overlap=20%, RL搜索

| ID | Stage数 | 预期off/off Val MRR | 预期选出架构 | 备注 |
|----|---------|---------------------|--------------|------|
| **S1** | 1 | ~0.38 (低估) | ❌ linear/off (错误) | 已有数据 |
| **S2** | 2 | ~0.80 (正常) | ✅ off/off (正确) | 已有数据 |
| **S3** | 3 | 待验证 | 待验证 | **需要补充** |
| **S4** | 4 | 待验证 | 待验证 | **需要补充** |

## 核心假设

- **H1**: Stage数量影响Val评估准确性
- **H2**: 2 stages是最佳配置（1 stage太少，>2 stages可能过度分割）
- **H3**: Overlap=20%足以提供连续性（无需测试其他比例）

## 数据复用

**可直接使用的现有实验**：
- ✅ S1: `outputs/50k_comparison/seed_42/smart_1stage/`
- ✅ S2: `outputs/50k_comparison/seed_42/smart_overlap20/`
- ⚠️ B1/B2: 如果存在且seed=42可复用，否则重跑

**需要补充的实验**：
- S3: 3 stages + 20% overlap
- S4: 4 stages + 20% overlap
- 可能需要: B1, B2 (如果现有数据不符合)

## 验证标准

**成功标准**：
- Serial和Data Parallel都选出off/off
- 找出哪些Stage配置能正确选出off/off
- 确认最佳Stage数量

**关键指标**：
- 选出的架构 (time_proj/use_static)
- off/off的Val MRR (诊断指标)
- Test MRR (最终性能)

## 资源需求

- 总实验数: 6个 (2个基准 + 4个Stage测试)
- 可复用: 2-4个 (取决于现有数据)
- 需新跑: 2-4个
- 预计时间: 1-2小时 (如果复用现有数据)
