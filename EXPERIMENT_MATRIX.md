# Pipeline NAS三因素验证实验

## 验证目标

系统验证影响Pipeline NAS架构选择准确性的三个关键因素：
1. **Stage划分** (1/2/3 stages)
2. **Overlap** (0% vs 20%)
3. **架构生成模式** (同步 vs 异步)

## 实验矩阵

### 基准组

| ID | 模式 | 现有数据 |
|----|------|----------|
| **B1** | Serial | `outputs/50k_comparison/seed_42/serial/` ✅ |
| **B2** | Data Parallel | `outputs/50k_comparison/seed_42/data_parallel_improved/` ✅ |

### 主实验组（3×2×2 = 12个实验）

| ID | Stages | Overlap | 架构生成 | Worker配置 | 现有数据 | 状态 |
|----|--------|---------|----------|------------|----------|------|
| **E1** | 1 | 20% | 异步 | [3] | `smart_1stage/` | ✅ |
| **E2** | 1 | 0% | 异步 | [3] | - | ⏳ |
| **E3** | 2 | 20% | 异步 | [3,3] | `smart_overlap20/` | ✅ |
| **E4** | 2 | 0% | 异步 | [3,3] | `naive_no_overlap/` | ✅ |
| **E5** | 3 | 20% | 异步 | [1,1,1] | - | ⏳ |
| **E6** | 3 | 0% | 异步 | [1,1,1] | `naive_3stages/` | ✅ |
| **E7** | 1 | 20% | 同步 | [3] | - | ⏳ |
| **E8** | 1 | 0% | 同步 | [3] | - | ⏳ |
| **E9** | 2 | 20% | 同步 | [3,3] | - | ⏳ |
| **E10** | 2 | 0% | 同步 | [3,3] | - | ⏳ |
| **E11** | 3 | 20% | 同步 | [1,1,1] | - | ⏳ |
| **E12** | 3 | 0% | 同步 | [1,1,1] | - | ⏳ |

## Worker分配策略

根据代码自动分配逻辑（3个GPU）：
- **1 stage**: [3] - 3个worker在stage 0
- **2 stages**: [3, 3] - 每stage 3个worker
- **3 stages**: [1, 1, 1] - 每stage 1个worker

## 架构生成模式说明

**异步模式（现有实验默认）**：
- RL agent边训边生成新架构
- Pipeline保持满载，最大化GPU利用率
- 日志显示：`submitted=50` (一次性提交所有架构)

**同步模式（需补充）**：
- 一批架构训完才生成下一批
- 更接近传统NAS的批次执行
- 需要修改search代码实现

## 复用数据统计

- ✅ **已有数据**: 6个实验 (B1, B2, E1, E3, E4, E6)
- ⏳ **需补充**: 8个实验 (E2, E5, E7-E12)

## 预期发现

### 假设1：Stage数量影响
- 1 stage: Val评估不准确（缺少预热）
- 2 stages: 最佳平衡
- 3 stages: 可能过度分割

### 假设2：Overlap影响
- 0% overlap: 时间不连续，embedding状态跳变
- 20% overlap: 提供连续性，缓解状态断裂

### 假设3：架构生成模式影响
- 异步: Pipeline满载，但RL可能基于过时信息决策
- 同步: RL基于最新结果，但GPU利用率降低

## 固定参数

- Dataset: MOOC (public_csv)
- Max events: 50,000
- Seed: 42
- Trials: 50
- Search space: rnn_only
- Search mode: RL
- Partition size: 12,500
- Coarse epochs: 1
