# 数据泄露分析报告

## 执行摘要

经过系统检查，发现了一个**测试时embedding更新**的问题，这可能导致测试性能被高估。

---

## 1. 数据分割检查 ✅

**检查位置**: `train_single_arch.py`, lines 74-85

**分割方式**:
```python
all_interactions.sort(key=lambda x: x.timestamp)  # 按时间排序
train_data = all_interactions[:n_train]           # 0-70%
val_data = all_interactions[n_train:n_train + n_val]  # 70-80%
test_data = all_interactions[n_train + n_val:]    # 80-100%
```

**结论**: ✅ **无泄露**
- 时间顺序正确（过去→未来）
- 无重叠
- 无随机打乱

---

## 2. 评估过程检查 ⚠️

**检查位置**: `models/training.py`, `evaluate_partition_ranking()`, lines 297-326

**评估流程**:
```python
for interaction in test_partition.interactions:
    # 调用model()进行预测
    pred_emb, _, _ = model(uid, item_id, timestamp, features, ...)
    
    # 获取所有item embeddings
    all_item_emb = _all_item_embeddings(model)
    
    # 计算距离并排序
    distances = torch.norm(all_item_emb - pred_emb, p=2, dim=-1)
    top_k = torch.argsort(distances, ...)
```

**问题**: 每次调用`model()`都会触发embedding更新

---

## 3. JODIE模型的Embedding更新机制 ⚠️

**检查位置**: `models/jodie_rnn.py`, `process_interaction()`, lines 281-285

**关键代码**:
```python
def process_interaction(..., deferred: bool = False):
    # ... 计算新的embeddings ...
    
    if not deferred:  # 默认为False
        self.user_embeddings[user_ids] = new_user_emb.detach()  # ← 更新！
        self.item_embeddings[item_ids] = new_item_emb.detach()  # ← 更新！
        self.user_last_time[user_ids] = timestamps
        self.item_last_time[item_ids] = timestamps
```

**发现**:
- `deferred=False`（默认）：立即更新内部embeddings
- `deferred=True`：不更新，由调用方决定
- **评估时没有传递`deferred`参数，使用默认值False**
- **不受`model.eval()`模式影响**

---

## 4. 测试时Embedding更新的影响

### 4.1 泄露机制

在测试评估时：

```
Test Interaction 1: 
  - 使用初始embeddings预测
  - 更新user_1和item_A的embeddings
  
Test Interaction 2:
  - 如果涉及user_1或item_A，使用更新后的embeddings
  - 更新user_2和item_B的embeddings
  
Test Interaction 3:
  - 如果涉及之前见过的user/item，使用累积更新的embeddings
  - ...
```

**结果**: 后面的测试样本受益于前面测试样本的信息

### 4.2 影响程度

**理论分析**:
- 测试集有4000个交互
- 用户数1435，物品数21
- 平均每个用户在测试集中出现: 4000/1435 ≈ 2.8次
- 平均每个物品在测试集中出现: 4000/21 ≈ 190次

**物品embeddings受影响最大**:
- 每个物品平均被更新190次
- 后期测试样本使用的是经过大量更新的embeddings
- 这可能显著提升性能

### 4.3 实验证据

从我们的实验结果：
- NAS搜索Test Recall: 91-99%
- 这在21个物品的数据集上偏高
- 可能部分原因是测试时embedding更新

---

## 5. 这是Bug还是Feature？

### 5.1 JODIE原始论文的评估协议

需要查阅JODIE原始论文确认标准评估协议。通常有两种：

**协议A: 在线/流式评估**
- 模型在测试时持续更新
- 模拟真实部署场景
- 测试时embedding更新是**正确的**

**协议B: 离线/批量评估**  
- 模型在训练后冻结
- 测试时不更新
- 测试时embedding更新是**泄露**

### 5.2 当前实现

当前实现采用**协议A（在线评估）**:
- 测试时更新embeddings
- 符合JODIE的动态特性
- 但可能不符合标准离线评估

---

## 6. 建议的修复方案

### 方案1: 冻结评估（推荐用于论文对比）

**修改**: `models/training.py`, `evaluate_partition_ranking()`

```python
def evaluate_partition_ranking(model, partition, k=10, graph_ctx=None, ...):
    # 保存原始embeddings
    original_user_emb = model.user_embeddings.clone()
    original_item_emb = model.item_embeddings.clone()
    original_user_time = model.user_last_time.clone()
    original_item_time = model.item_last_time.clone()
    
    try:
        for interaction in partition.interactions:
            # 使用deferred=True防止更新
            pred_emb, _, _ = model(
                ...,
                deferred=True  # ← 关键修改
            )
            # ... 评估逻辑 ...
    finally:
        # 恢复原始embeddings
        model.user_embeddings.data = original_user_emb
        model.item_embeddings.data = original_item_emb
        model.user_last_time.data = original_user_time
        model.item_last_time.data = original_item_time
```

**优点**:
- 符合标准离线评估协议
- 可与其他方法公平对比
- 消除测试时泄露

**缺点**:
- 不符合JODIE的在线特性
- 性能可能下降

### 方案2: 明确区分两种评估模式

添加参数控制评估模式：

```python
def evaluate_ranking_metrics(
    model, 
    test_interactions, 
    k=10, 
    online_mode=False,  # ← 新参数
    ...
):
    if online_mode:
        # 允许embedding更新（当前行为）
        deferred = False
    else:
        # 冻结embeddings
        deferred = True
        # 保存和恢复embeddings
```

**优点**:
- 灵活支持两种评估
- 可以报告两种性能
- 明确评估协议

---

## 7. 实验验证建议

### 7.1 对比实验

运行两种评估模式：
1. **当前模式**（在线，允许更新）
2. **冻结模式**（离线，不允许更新）

对比性能差异，量化测试时更新的影响。

### 7.2 预期结果

如果测试时更新有显著影响：
- 冻结模式的Recall应该明显低于当前模式
- 差异可能在5-15%

---

## 8. 结论

### 8.1 当前状态

✅ **数据分割**: 无泄露
⚠️ **评估过程**: 存在测试时embedding更新
❓ **是否为问题**: 取决于评估协议

### 8.2 建议行动

1. **短期**: 明确当前使用的是在线评估协议
2. **中期**: 实现冻结评估模式，对比两种结果
3. **长期**: 在论文中明确说明评估协议

### 8.3 对现有结果的影响

- 如果采用在线评估：当前结果有效
- 如果采用离线评估：需要重新评估，性能可能下降5-15%

---

## 附录：快速验证脚本

创建一个简单的测试来验证embedding是否在评估时更新：

```python
# test_eval_leakage.py
import torch
from models.jodie_rnn import JODIERNN

model = JODIERNN(num_users=10, num_items=5, embedding_dim=16, feature_dim=4)
model.eval()

# 保存初始embeddings
initial_user_emb = model.user_embeddings[0].clone()

# 模拟评估
uid = torch.tensor([0])
iid = torch.tensor([1])
ts = torch.tensor([1.0])
feat = torch.randn(1, 4)

pred_emb, _, _ = model(uid, iid, ts, feat, query_time=1.0)

# 检查embeddings是否改变
final_user_emb = model.user_embeddings[0]
changed = not torch.allclose(initial_user_emb, final_user_emb)

print(f"Embeddings changed during eval: {changed}")
# 预期输出: True（说明存在更新）
```
