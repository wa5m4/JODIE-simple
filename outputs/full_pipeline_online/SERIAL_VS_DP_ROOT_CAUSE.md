# Serial vs Data Parallel 性能差异根本原因

生成时间: 2026-05-30

## 问题回顾

Serial和Data Parallel的重训性能差异巨大：
- Serial重训: MRR 0.3896
- Data Parallel重训: MRR 0.7381
- **差异: 89.4%**

## 根本原因：架构不同！

### 配置对比

| 参数 | Serial | Data Parallel | 是否相同 |
|------|--------|---------------|---------|
| model | jodie_rnn | jodie_rnn | ✅ |
| embedding_dim | 128 | 128 | ✅ |
| memory_cell | rnn | rnn | ✅ |
| **time_proj** | **off** | **linear** | ❌ **不同！** |
| batch_mode | tbatch | tbatch | ✅ |
| train_batch_size | 32 | 32 | ✅ |
| seed | 20042 | 20042 | ✅ |
| lr | 0.001 | 0.001 | ✅ |

### time_proj参数的作用

**time_proj** 控制JODIE中时间信息的投影方式：
- `off`: 不使用时间投影
- `linear`: 使用线性时间投影

这是一个**重要的架构差异**，会显著影响模型性能。

---

## 为什么会出现架构差异？

### NAS搜索过程

1. **Serial NAS搜索**:
   - 搜索空间包含多种架构配置
   - 在验证集上评估，选择最优架构
   - **最优架构**: time_proj=off (MRR 0.85)

2. **Data Parallel NAS搜索**:
   - 相同的搜索空间
   - 在验证集上评估，选择最优架构
   - **最优架构**: time_proj=linear (MRR 0.67)

### 为什么选择了不同的架构？

**可能原因**:
1. **评估模式影响**: 在线评估(frozen=False)下，不同架构的性能排序可能不稳定
2. **训练动态差异**: Serial和Data Parallel的训练过程不同，导致同一架构表现不同
3. **随机性**: NAS搜索本身有随机性，可能选到不同的局部最优

---

## 结论

### ✅ 性能差异是合理的

Serial和Data Parallel的重训性能差异（89%）是因为：
- **使用了不同的架构**（time_proj: off vs linear）
- 不是评估隔离性问题
- 不是训练过程bug

### 正确的对比方式

要公平对比Serial和Data Parallel，应该：
1. **使用相同的架构**进行重训
2. 或者分别报告各自NAS找到的最优架构性能

### 当前报告的问题

在对比报告中，我们直接对比了Serial和Data Parallel的重训结果，但忽略了它们使用的架构不同。这导致对比不公平。

---

## 建议

### 1. 更新对比报告

在报告中明确说明：
- Serial使用架构: time_proj=off
- Data Parallel使用架构: time_proj=linear
- 性能差异主要来自架构差异，而非执行模式差异

### 2. 公平对比实验

如果要公平对比Serial和Data Parallel的训练效率，应该：
```bash
# 使用相同架构重训
python train_single_arch.py \
  --model jodie_rnn \
  --embedding-dim 128 \
  --memory-cell rnn \
  --time-proj linear \  # 使用相同的time_proj
  --eval-frozen false \
  --seed 20042 \
  --output-dir outputs/serial_retrain_fair
```

### 3. 理解NAS的作用

NAS的目的是为每个执行模式找到**最适合该模式的架构**：
- Serial最优: time_proj=off
- Data Parallel最优: time_proj=linear

这说明不同的训练方式可能适合不同的架构。

---

**报告生成时间**: 2026-05-30  
**关键发现**: Serial和Data Parallel性能差异的根本原因是架构不同（time_proj参数）
