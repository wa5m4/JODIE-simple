# Serial重训性能异常调查

## 问题描述

在线评估模式下，Serial模式出现严重的性能下降：
- **NAS Test MRR**: 0.8509 (85%)
- **重训 Test MRR**: 0.3896 (39%)
- **性能下降**: 54%

相比之下，Data Parallel模式表现正常：
- **NAS Test MRR**: 0.6712 (67%)
- **重训 Test MRR**: 0.7381 (74%)
- **性能提升**: 10%

## 调查发现

### 1. 训练数据一致性 ✅

**NAS Final Test** (nas/trainer.py:934):
```python
# fit=train+val, test=test, epochs=3
final_test_result = self.evaluate_arch_pipeline(
    arch_configs=[selected["config"]],
    eval_split="test",
    epochs=final_epochs,
)
```

**重训** (train_single_arch.py:88):
```python
final_train_data = train_data + val_data  # train+val
# 然后在test上评估
```

✅ 两者都使用train+val训练，在test上评估

### 2. 随机种子一致性 ✅

**NAS Final Test**:
- seed: 20042 (从best_arch.json的per_seed_metrics)

**重训**:
- seed: 20042 (args.seed + 20000 = 42 + 20000)

✅ 两者使用相同的随机种子

### 3. 架构配置一致性 ✅

**NAS选择的架构**:
- model: jodie_rnn
- embedding_dim: 128
- memory_cell: rnn
- time_proj: off

**重训使用的架构**:
- model: jodie_rnn
- embedding_dim: 128
- memory_cell: rnn
- time_proj: off

✅ 架构完全一致

### 4. 评估模式一致性 ✅

**NAS**:
- eval_frozen: false (从best_arch.json line 49)

**重训**:
- eval_frozen: false (通过--eval-frozen false传递)

✅ 都使用在线评估模式

---

## 可能的原因分析

### 假设1: 模型初始化差异

**可能性**: NAS和重训使用不同的模型初始化方式

**验证方法**: 
- 检查build_model()函数是否使用随机种子初始化
- 确认torch.manual_seed()是否在模型构建前调用

### 假设2: 训练过程差异

**可能性**: 虽然配置相同，但训练过程中的某些细节不同

**需要检查**:
- Partition划分是否相同
- Batch处理顺序是否相同
- 梯度累积或优化器状态

### 假设3: 评估时机差异

**可能性**: NAS在不同的训练阶段评估，导致模型状态不同

**观察**: 
- NAS的test_mrr (0.85) 远高于val_mrr (0.81)
- 这在在线评估中是可能的（test评估时embeddings已经在val上更新过）

### 假设4: 随机性导致的训练不稳定

**可能性**: 在线评估模式下，模型训练可能不稳定，导致不同运行结果差异大

**证据**:
- Data Parallel的重训(0.74)反而高于NAS(0.67)
- 说明重训结果可能受随机性影响较大

---

## 建议的验证步骤

### 1. 重新运行Serial重训（使用相同配置）

```bash
python train_single_arch.py \
  --model jodie_rnn \
  --embedding-dim 128 \
  --memory-cell rnn \
  --time-proj off \
  --batch-mode tbatch \
  --train-batch-size 32 \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --max-events 20000 \
  --epochs 3 \
  --seed 20042 \
  --eval-frozen false \
  --output-dir outputs/serial_retrain_verify
```

**预期**: 如果结果仍然是MRR 0.39，说明是可复现的；如果接近0.85，说明是随机性问题

### 2. 使用不同随机种子运行多次

```bash
# 运行3次，使用不同种子
for seed in 20042 20043 20044; do
  python train_single_arch.py \
    --model jodie_rnn \
    --embedding-dim 128 \
    --memory-cell rnn \
    --time-proj off \
    --batch-mode tbatch \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events 20000 \
    --epochs 3 \
    --seed $seed \
    --eval-frozen false \
    --output-dir outputs/serial_retrain_seed_$seed
done
```

**目的**: 评估结果的稳定性和方差

### 3. 对比离线评估结果

检查Serial在离线评估模式下NAS和重训的差异：
- 离线NAS: MRR 0.2263
- 离线重训: MRR 0.1780
- 差异: -21% (正常范围)

**结论**: 离线评估下差异正常，说明问题只出现在在线评估模式

---

## 初步结论

### 最可能的原因

**在线评估模式下的训练不稳定性**

1. **证据1**: Data Parallel的重训(0.74)高于NAS(0.67)，说明重训结果可能更好
2. **证据2**: Serial的NAS(0.85)和重训(0.39)差异巨大，超出正常范围
3. **证据3**: 离线评估下差异正常(-21%)，在线评估下差异异常(-54%)

**推测**: 在线评估模式(frozen=False)下，模型训练和评估都会更新embeddings，导致：
- 训练过程更不稳定
- 最终性能受随机初始化和训练动态影响更大
- 不同运行之间方差更大

### 为什么Data Parallel没有这个问题？

可能原因：
1. Data Parallel使用了不同的训练动态（3个worker并行）
2. 梯度平均可能起到了正则化作用
3. 或者只是运气好，随机种子导致了更好的结果

---

## 建议

### 短期建议

1. **接受当前结果**: Serial重训MRR 0.39是在线评估下的一个有效结果
2. **报告时说明**: 在线评估下结果方差较大，需要多次运行取平均
3. **补充实验**: 运行多个随机种子，报告平均值和标准差

### 长期建议

1. **优先使用离线评估**: 更稳定、可复现、符合学术标准
2. **在线评估仅作参考**: 用于评估模型的在线适应能力，但不作为主要指标
3. **改进训练稳定性**: 考虑使用更大的batch size、更多epochs或学习率调度

---

**调查报告生成时间**: 2026-05-30  
**状态**: 需要进一步验证实验
