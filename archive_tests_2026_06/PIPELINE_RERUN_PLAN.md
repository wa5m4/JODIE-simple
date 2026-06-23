# Pipeline模式配置分析与重跑计划

## 当前问题总结

### 1. Pipeline配置检查

**Pipeline Naive:**
- num_pipeline_stages: 2
- pipeline_mode: naive
- 配置正常保存 ✓

**Pipeline Smart:**
- pipeline相关字段缺失
- 可能原因：配置保存bug或搜索未正确运行

### 2. 理论vs实际分析

**理论预期:**
- Pipeline Smart应该比Naive快（更智能的stage分配）
- 所有模式应该找到相似的最优架构（相同搜索空间）

**实际观察:**
- Pipeline Smart选择了不同的架构参数
- 结果不如Serial/Data Parallel/Naive

**可能原因:**
1. **Seed bug影响**: Pipeline用错误seed(100)导致搜索结果不稳定
2. **配置未正确保存**: Smart的pipeline配置字段缺失
3. **搜索收敛问题**: 27trials可能不足以让所有模式收敛到全局最优

## 种子规划方案

### 当前种子使用情况

| 阶段 | Serial/Data Parallel | Pipeline (Bug) | Pipeline (修复后) |
|------|---------------------|----------------|------------------|
| Coarse (trial i) | 100 + i | 100 + i | 100 + i |
| Final Test | 100 + 20000 = 20100 | 100 + 0 = 100 ✗ | 100 + 20000 = 20100 ✓ |
| Retrain | 从best_arch提取 | 100 (错误) | 20100 (正确) |

### 修复后的种子方案

**Coarse Phase:**
```python
trial_seed = base_seed + trial_id
# trial 0: seed = 100
# trial 1: seed = 101
# ...
# trial 26: seed = 126
```

**Final Test:**
```python
final_seed = base_seed + 20000 = 20100  # 所有模式统一
```

**Retrain:**
```python
retrain_seed = best_arch["seed"]  # 从NAS Final Test提取，应该是20100
```

## 重跑计划

### 需要重跑的部分

由于发现了seed bug，需要重新运行：

**✅ 不需要重跑:**
- Serial: 已经使用正确seed (20100)
- Data Parallel: 已经使用正确seed (20100)

**⚠️ 需要重跑:**
- Pipeline Naive: NAS用了错误seed (100)
- Pipeline Smart: NAS用了错误seed (100)

### 重跑步骤

1. **使用修复后的代码运行Pipeline NAS**
   ```bash
   # Pipeline Naive
   python search.py \
       --search-mode rl \
       --execution-mode ray_pipeline \
       --pipeline-mode naive \
       --gpu-list 0,1,2 \
       --dataset public_csv \
       --local-data-path data/public/mooc.csv \
       --max-events 20000 \
       --seed 100 \
       --coarse-trials 27 \
       --coarse-epochs 3 \
       --output-dir outputs/bug_fix_verification_v3/seed_100/pipeline_naive
   
   # Pipeline Smart  
   python search.py \
       --search-mode rl \
       --execution-mode ray_pipeline \
       --pipeline-mode smart \
       --gpu-list 0,1,2 \
       --dataset public_csv \
       --local-data-path data/public/mooc.csv \
       --max-events 20000 \
       --seed 100 \
       --coarse-trials 27 \
       --coarse-epochs 3 \
       --output-dir outputs/bug_fix_verification_v3/seed_100/pipeline_smart
   ```

2. **提取正确的seed并重训练**
   - 验证best_arch.json中seed字段为20100
   - 使用seed=20100进行重训练
   - 对比NAS Final Test和Retrain结果

### 预期结果

修复后，所有4个模式应该：
- Final Test seed统一为20100
- NAS vs Retrain差异<1%
- 可能找到相似的最优架构（但不保证完全相同）

## 关于架构差异的说明

**为什么不同模式可能找到不同架构？**

1. **搜索随机性**: RL controller的随机采样可能导致不同探索路径
2. **搜索trials有限**: 27trials不足以完全探索搜索空间
3. **执行模式影响**: 不同并行化方式可能导致轻微的训练动态差异

**但是:**
- seed统一后，结果应该可复现
- 相同架构+相同seed应该得到相同MRR
- 最优架构的MRR应该相近（即使参数不完全相同）

## 总结

1. ✅ **代码已修复**: trainer.py:954现在使用正确的seed计算
2. ⚠️ **需要重跑**: Pipeline Naive和Smart的NAS需要用修复后的代码重新运行
3. ✅ **验证方法**: 检查best_arch.json的seed字段是否为20100
4. ✅ **预期结果**: 所有模式NAS vs Retrain差异<1%
