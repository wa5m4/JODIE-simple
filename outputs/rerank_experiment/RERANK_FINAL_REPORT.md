# 四种NAS策略 + Rerank 最终报告

**实验配置**: Seed 1000, 20000 events, 27 trials × 3 epochs (coarse) + top 10 × 10 epochs (rerank)

## 一、NAS阶段结果

| 策略 | Test MRR | Rerank执行 |
|------|----------|-----------|
| Serial | 0.6698 | ✓ |
| Data Parallel | 0.7226 | ✗ (代码不支持) |
| Pipeline Naive | 0.6557 | ✓ |
| Pipeline Smart | 0.6765 | ✓ |

**NAS准确率范围**: 0.6557 - 0.7226 (差距6.69个百分点)

## 二、重训练结果（使用正确seed=1000）

| 策略 | NAS MRR | Retrain MRR | 差异 |
|------|---------|-------------|------|
| Serial | 0.6698 | 0.6677 | -0.3% |
| Data Parallel | 0.7226 | 0.7578 | +4.9% |
| Pipeline Naive | 0.6557 | 0.6205 | -5.4% |
| Pipeline Smart | 0.6765 | 0.6677 | -1.3% |

**重训练准确率范围**: 0.6205 - 0.7578 (差距13.7个百分点)

## 三、关键发现

### 1. Rerank执行情况
- **Serial, Pipeline Naive, Pipeline Smart**: 成功执行coarse + rerank两阶段
- **Data Parallel**: 只执行了coarse阶段，代码不支持rerank功能

### 2. Seed问题
- `best_arch.json`中有两个seed值：
  - 顶层`seed=21000`: NAS为该trial生成的独立随机seed
  - `config.seed=1000`: 实验的全局seed（用于数据划分）
- 初始重训练错误地使用了21000，导致数据划分不同，结果不可比
- 修正后使用1000，结果更合理

### 3. 准确率持平问题
尽管使用了rerank功能，**四种策略的准确率并未达到持平**：
- 最高: Data Parallel 0.7578
- 最低: Pipeline Naive 0.6205
- 差距: 13.7个百分点

### 4. 异常观察
- **Serial和Pipeline Smart重训练结果完全相同** (0.6677)，但它们的NAS结果不同
- **Pipeline Naive重训练显著下降** (-5.4%)，可能的原因：
  - NAS阶段的验证集评估不够准确
  - 最佳架构在训练集过拟合
  - 重训练的10 epochs不足以收敛

## 四、为什么准确率未持平？

### 根本原因
1. **搜索空间探索不足**: 27个trial对于搜索空间来说太少
   - 不同策略的搜索效率差异大
   - Serial可能利用RL反馈更高效
   - Data Parallel/Pipeline是纯并行搜索，缺乏策略调整

2. **Rerank的局限性**: 
   - 只能从已有的27个trial中选top 10精炼
   - 无法扩大搜索范围
   - 如果coarse阶段没找到好架构，rerank也帮助有限

3. **策略本身的差异**:
   - 不同执行模式的搜索能力本质上不同
   - 即使用相同的trial数和rerank，最终质量仍有差异

## 五、改进建议

### 短期改进
1. **增加coarse trials**: 从27增加到50-100个
   - 给每个策略更多机会探索搜索空间
   - 提高找到优质架构的概率

2. **增加rerank数量**: 从top 10增加到top 15-20
   - 给更多候选架构充分训练的机会

3. **增加rerank epochs**: 从10增加到15-20
   - 确保架构充分收敛

### 长期改进
1. **修复Data Parallel的rerank支持**
   - 实现Data Parallel模式的rerank逻辑
   - 确保所有策略使用相同的评估流程

2. **改进搜索策略**:
   - 为效率较低的策略（如Data Parallel）调整搜索参数
   - 考虑使用更智能的搜索算法（如进化算法、贝叶斯优化）

3. **统一评估标准**:
   - 确保所有策略使用相同的验证集和测试集
   - 使用更稳定的评估指标（如多次运行取平均）

## 六、时间对比（NAS阶段）

| 策略 | NAS时间(min) | 加速比 |
|------|-------------|--------|
| Serial | ~207 | 1.00x |
| Data Parallel | ~133 | 1.56x |
| Pipeline Naive | ~144 | 1.44x |
| Pipeline Smart | ~10 | 20.70x |

**Pipeline Smart在保持合理准确率的同时提供了20倍加速**

## 七、结论

1. **Rerank功能正常工作**（除Data Parallel外），但未能显著缩小不同策略间的准确率差距
2. **27个trial太少**，无法充分探索搜索空间
3. **建议增加trial数量**（50-100个）以提高所有策略找到优质架构的概率
4. **Pipeline Smart在速度和准确率间取得了良好平衡**，是实用的选择
