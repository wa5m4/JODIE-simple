# JODIE项目批处理实现诊断报告

**生成时间：** 2026-05-18  
**诊断目标：** 系统性自检批处理实现，定位准确率异常问题

---

## 执行摘要

**核心问题：** 当前代码混淆了t-Batch和TGN两种批处理策略，导致三个严重问题：

1. 🔴 **信息丢失**：同一节点的多条交互被"last"策略丢弃
2. 🔴 **训练信号错误**：Loss计算使用了batch处理前的旧embedding
3. 🔴 **策略不一致**：Batch内允许节点重复但没有正确聚合

**推荐方案：** 修复为标准t-Batch（无损并行），优先恢复准确率

---

## 步骤1：代码库扫描结果

### 1.1 批处理方案分布表

| 批处理方案 | 文件位置 | 关键函数/类 | 说明 |
|-----------|---------|------------|------|
| **TGN风格窗口批处理** | `models/training.py` | `train_partition_bpr_batch` (402-527行)<br>`train_partition_ce_batch` (530-635行) | 实现了节点聚合批处理，使用`deferred=True`延迟更新 |
| **JODIE逐条处理** | `models/training.py` | `train_partition_bpr` (94-136行)<br>`train_partition_ce` (139-169行) | 标准逐交互训练，每条立即更新 |
| **RNN状态管理** | `models/jodie_rnn.py` | `process_interaction` (145-213行)<br>`forward` (228-241行) | 支持`deferred`参数控制更新时机 |
| **GNN+RNN混合** | `models/hybrid_jodie.py` | `process_interaction` (191-279行)<br>`forward` (290-389行) | 包含邻居聚合逻辑，支持`deferred` |
| **消息聚合** | `models/gnn_encoder.py` | `event_aggregate` (100-158行) | 实现mean/sum/attn三种聚合策略 |

### 1.2 关键发现

- ❌ **没有找到**任何显式的"t-Batch"实现或命名
- ✅ 存在两套并行的训练路径：`batch_training=False`（逐条）和`batch_training=True`（批处理）
- ⚠️ 批处理实现使用了TGN风格的节点聚合策略，而非t-Batch的无损并行
- ⚠️ 代码注释声称是"TGN风格"，但实际实现既不是标准TGN也不是t-Batch

---

## 步骤2：追踪数据流

### 2.1 问题A - Batch组装方式

**数据加载方式：** 直接遍历交互列表，无自定义DataLoader或collate_fn

**逐条训练模式** (`models/training.py:110-135`)：
```python
for idx, interaction in enumerate(partition.interactions, start=1):
    uid = torch.tensor([interaction.user_id], ...)
    iid = torch.tensor([interaction.item_id], ...)
    # 每次处理单条交互
```

**批处理模式** (`models/training.py:455-456`)：
```python
for batch_start in range(0, len(interactions), batch_size):
    batch = interactions[batch_start: batch_start + batch_size]
```

**结论：**
- ✅ **批处理模式允许同一节点出现多次**：使用简单的滑动窗口切分`[batch_start:batch_start+batch_size]`，没有去重逻辑
- ❌ **没有按时间戳分桶**：直接按顺序切分，batch_size是固定的交互数（默认32）
- ⚠️ **这不是t-Batch**：t-Batch要求batch内节点唯一，但当前实现允许重复

### 2.2 问题B - 消息处理方式

**发现：存在邻居消息聚合，但仅在hybrid_jodie模型中**

**位置：** `models/hybrid_jodie.py:339-357`

```python
user_msg = self.event_operator.event_aggregate(
    center_idx=uid,
    center_emb=proj_user.squeeze(0),
    memory=self.memory,
    neighbors=user_neighbors,  # 从图状态获取邻居
    edge_last_time=graph_state["edge_last_time"],
    current_time=ts,
).unsqueeze(0)
```

**聚合实现：** `models/gnn_encoder.py:100-158`

支持三种聚合策略：
- `mean`: 时间衰减加权平均 (第153-158行)
- `sum`: 时间衰减加权求和 (第137-142行)  
- `attn`: 注意力机制聚合 (第144-151行)

**结论：**
- ✅ **hybrid_jodie模型有消息聚合**：对同一节点的多个邻居消息做聚合
- ❌ **jodie_rnn模型没有消息聚合**：只使用对方节点的embedding
- ⚠️ **批处理时的聚合策略问题**：见下文详细分析

### 2.3 问题C - Embedding更新时机

**关键发现：批处理模式下的更新策略存在严重问题**

**批处理流程分析** (`models/training.py:455-527`)：

**阶段1（459-463行）：收集batch内每个节点的交互**
```python
for interaction in batch:
    user_interactions[interaction.user_id].append(interaction)
    item_interactions[interaction.item_id].append(interaction)
```

**阶段2（469-487行）：聚合并计算新状态**
```python
for uid, user_batch in user_interactions.items():
    user_batch_sorted = sorted(user_batch, key=lambda x: x.timestamp)
    last_interaction = user_batch_sorted[-1]  # ⚠️ 只使用最后一个交互！
    
    result = model.process_interaction(..., deferred=True, ...)  # 延迟写回
    node_updates[('user', uid)] = (new_user_emb, last_interaction.timestamp, ...)
```

**阶段3（490-510行）：计算loss**
```python
for interaction in batch:  # 对每条原始交互计算loss
    pred_emb, _ = model.predict(uid, interaction.timestamp)
    # ⚠️ 使用的是batch处理前的旧embedding！
```

**阶段4（514-525行）：写回更新**
```python
with torch.no_grad():
    for (node_type, node_id), (new_emb, ts, new_c) in node_updates.items():
        model.user_embeddings[node_id] = new_emb.detach()  # 统一写回
```

**结论：**
- ❌ **RNN state在batch内不是逐条更新**：等整个batch算完再统一更新（阶段4）
- ⚠️ **同一节点的多条交互只保留最后一条**：第471行的`last_interaction`策略丢弃了中间交互
- 🔴 **严重问题**：阶段3计算loss时使用的是`model.predict()`，读取的是**batch处理前的旧embedding**，而不是阶段2计算出的新embedding

### 2.4 问题D - 损失计算粒度

**位置：** `models/training.py:490-510`

```python
# 阶段3: 计算loss（对每个原始交互）
batch_losses = []
for interaction in batch:
    pred_emb, _ = model.predict(uid, interaction.timestamp)  # 从模型缓冲区读取
    pos_emb = _item_embeddings_for_loss(model, iid).detach()
    loss = criterion(pred_emb, pos_emb, neg_emb) / len(batch)
    batch_losses.append(loss)
```

**对比逐条训练** (`models/training.py:127-132`)：

```python
pred_emb, _, _ = model(uid, iid, t, f, interaction.timestamp, ...)  # forward立即更新
pos_emb = _item_embeddings_for_loss(model, iid).detach()
loss = criterion(pred_emb, pos_emb, neg_emb)
loss.backward(retain_graph=True)
optimizer.step()  # 立即更新参数
```

**结论：**
- ✅ **损失是在每条交互级别计算**：遍历batch中的每条交互
- 🔴 **预测用的是batch处理前的embedding**：`model.predict()`读取的是旧状态
- 🔴 **目标embedding也可能是旧的**：`_item_embeddings_for_loss()`读取的是模型缓冲区，而不是阶段2计算的`node_updates`

---

## 步骤3：冲突与不一致分析

基于以上发现，识别出以下**严重问题**：

### 🔴 问题1：批处理实现混淆了TGN和t-Batch概念

**位置：** `models/training.py:402-527`

**问题描述：**
- 代码注释声称是"TGN风格批处理"（第411行）
- 但实际上既不是标准TGN，也不是t-Batch
- **TGN**：允许batch内节点重复，使用消息聚合
- **t-Batch**：要求batch内节点唯一，无损并行
- **当前实现**：允许节点重复（类TGN），但使用"last"聚合策略丢弃中间交互（既不是TGN的消息聚合，也不是t-Batch的无损）

**为什么导致准确率下降：**

如果user_5在同一batch内有3条交互（时间戳t1 < t2 < t3），当前实现只使用t3的交互来更新embedding，t1和t2的信息被完全丢弃。这违反了时序图神经网络的核心假设：每条交互都应该更新状态。

**示例：**
```
Batch = [
    (user=5, item=10, t=1.0),  # 交互1
    (user=5, item=20, t=2.0),  # 交互2
    (user=5, item=30, t=3.0),  # 交互3
]

当前实现：
- 只用交互3更新user_5的embedding
- 交互1和交互2的信息完全丢失
- 但loss仍然对所有3条交互计算
```

### 🔴 问题2：Loss计算使用了错误的embedding

**位置：** `models/training.py:502`

**问题描述：**

在阶段2中，`process_interaction(deferred=True)`计算了新的embedding并存储在`node_updates`字典中，但**没有写回模型缓冲区**。阶段3计算loss时，`model.predict()`从模型缓冲区读取的仍然是**batch处理前的旧embedding**。

**代码流程：**
```python
# 阶段2: 计算新embedding
result = model.process_interaction(..., deferred=True, ...)
node_updates[('user', uid)] = (new_user_emb, ...)  # 存在字典中，未写回

# 阶段3: 计算loss
pred_emb, _ = model.predict(uid, interaction.timestamp)  # 从缓冲区读取旧embedding！

# 阶段4: 写回（但已经太晚了）
model.user_embeddings[node_id] = new_emb.detach()
```

**为什么导致准确率下降：**
- **梯度信号错误**：loss是基于旧embedding计算的，但反向传播更新的是新embedding的参数
- **训练目标不一致**：模型学习的是"用旧状态预测"，而不是"用更新后的状态预测"
- **因果关系破坏**：这相当于引入了一个时间步的延迟

### 🔴 问题3：同一节点的多条交互被"last"策略丢弃

**位置：** `models/training.py:471`

**问题描述：**
```python
user_batch_sorted = sorted(user_batch, key=lambda x: x.timestamp)
last_interaction = user_batch_sorted[-1]  # 只保留最后一条
```

**为什么导致准确率下降：**
- **信息丢失**：中间交互的特征和时序信息被忽略
- **训练不一致**：对交互1计算loss，但user_5的embedding从未见过交互1
- **RNN状态错误**：LSTM/GRU的隐状态应该逐步演化，而不是跳跃式更新

**正确的做法应该是：**
- 方案A（t-Batch）：确保batch内节点唯一，无需聚合
- 方案B（TGN）：对多条交互使用真正的消息聚合（mean/sum/attention），而不是简单取last

### ⚠️ 问题4：Embedding更新时机与批处理策略不匹配

**位置：** `models/jodie_rnn.py:194-209` 和 `models/hybrid_jodie.py:266-275`

**问题描述：**

`process_interaction`函数支持`deferred`参数：
- `deferred=False`（默认）：立即写回，适合逐条训练
- `deferred=True`：延迟写回，适合批处理

但在`hybrid_jodie.py`的`process_interaction`中，即使`deferred=True`，图状态的更新仍然被延迟：

```python
if not deferred:
    self.memory[user_nodes] = new_user.detach()
    ...
    if self.enable_graph_update:
        self._update_graph_state(graph_state, uid, iid, ts)
```

这意味着batch内的后续交互看不到前面交互对图结构的更新。

**为什么导致准确率下降：**
- **图结构不一致**：batch内的交互应该看到动态演化的图，但实际上都基于batch开始时的图状态
- **邻居信息过时**：消息聚合使用的邻居列表不包含batch内的新边

---

## 步骤4：核心代码段标注

### 代码段1：批处理的batch组装逻辑

**文件：** `models/training.py:455-463`

```python
for batch_start in range(0, len(interactions), batch_size):
    batch = interactions[batch_start: batch_start + batch_size]
    
    # 阶段1: 收集batch内每个节点的交互
    user_interactions = defaultdict(list)
    item_interactions = defaultdict(list)
    for interaction in batch:
        user_interactions[interaction.user_id].append(interaction)
        item_interactions[interaction.item_id].append(interaction)
```

**问题标注：**
- ⚠️ **中等严重度**：简单滑动窗口切分，允许同一节点在batch内出现多次
- **问题**：这不是t-Batch（要求节点唯一），也不是标准TGN（应该有时间窗口）
- **修复建议**：如果要实现t-Batch，需要添加节点去重逻辑；如果要实现TGN，需要按时间窗口分桶

### 代码段2：节点聚合策略（"last"策略）

**文件：** `models/training.py:469-487`

```python
for uid, user_batch in user_interactions.items():
    # 按时间排序，使用最后一个交互  ⚠️ 问题点1
    user_batch_sorted = sorted(user_batch, key=lambda x: x.timestamp)
    last_interaction = user_batch_sorted[-1]  # 🔴 只保留最后一条！
    
    uid_t = torch.tensor([uid], dtype=torch.long, device=device)
    iid_t = torch.tensor([last_interaction.item_id], dtype=torch.long, device=device)
    t = torch.tensor([last_interaction.timestamp], dtype=torch.float32, device=device)
    f = last_interaction.features.unsqueeze(0).to(device)
    
    if has_lstm:
        result = model.process_interaction(uid_t, iid_t, t, f, 
                                          deferred=True,  # 🔴 问题点2：延迟写回
                                          return_cell_state=True)
        new_user_emb, new_item_emb, new_user_c, new_item_c = result
        node_updates[('user', uid)] = (new_user_emb, last_interaction.timestamp, new_user_c)
        node_updates[('item', last_interaction.item_id)] = (new_item_emb, last_interaction.timestamp, new_item_c)
```

**问题标注：**
- 🔴 **高严重度**：`last_interaction`策略丢弃了batch内同一节点的所有中间交互
- 🔴 **高严重度**：`deferred=True`导致新embedding未写回模型缓冲区
- **问题**：如果user_5有3条交互[t1, t2, t3]，只有t3被用于更新，t1和t2完全丢失
- **修复建议**：应该逐条调用`process_interaction`，或者实现真正的消息聚合（mean/sum/attention）


### 代码段3：Loss计算使用旧embedding

**文件：** `models/training.py:490-510`

```python
# 阶段3: 计算loss（对每个原始交互）
batch_losses = []
for interaction in batch:
    uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
    iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)
    
    # 🔴 问题点：从模型缓冲区读取，得到的是batch处理前的旧embedding
    pred_emb, _ = model.predict(uid, interaction.timestamp)
    
    # 🔴 问题点：目标embedding也是旧的
    pos_emb = _item_embeddings_for_loss(model, iid).detach().to(device)
    neg_emb = _item_embeddings_for_loss(model, neg_ids).detach().to(device).unsqueeze(0)
    
    loss = criterion(pred_emb, pos_emb, neg_emb) / len(batch)
    batch_losses.append(loss)

total_batch_loss = sum(batch_losses)
total_batch_loss.backward()  # 🔴 问题点：梯度基于旧embedding计算
optimizer.step()
```

**问题标注：**
- 🔴 **高严重度**：`model.predict()`读取的是batch处理前的旧embedding，而不是阶段2计算的新embedding
- 🔴 **高严重度**：loss和梯度都基于错误的状态计算
- **问题**：训练信号完全错误，模型学习的是"用过时状态预测"
- **修复建议**：应该使用`node_updates`中的新embedding来计算loss，或者改为逐条更新

### 代码段4：RNN状态更新逻辑

**文件：** `models/jodie_rnn.py:189-213`

```python
if self.cell_type == "lstm":
    user_c = self.user_cell_state[user_ids].detach().clone()
    item_c = self.item_cell_state[item_ids].detach().clone()
    new_user_emb, new_user_c = self.user_cell(user_rnn_input, (user_emb, user_c))
    new_item_emb, new_item_c = self.item_cell(item_rnn_input, (item_emb, item_c))
    
    if not deferred:  # ⚠️ 控制写回时机
        self.user_cell_state[user_ids] = new_user_c.detach()
        self.item_cell_state[item_ids] = new_item_c.detach()
else:
    new_user_emb = self.user_cell(user_rnn_input, user_emb)
    new_item_emb = self.item_cell(item_rnn_input, item_emb)

new_user_emb = self._normalize(new_user_emb)
new_item_emb = self._normalize(new_item_emb)

if not deferred:  # ⚠️ 控制写回时机
    self.user_embeddings[user_ids] = new_user_emb.detach()
    self.item_embeddings[item_ids] = new_item_emb.detach()
    self.user_last_time[user_ids] = timestamps
    self.item_last_time[item_ids] = timestamps
```

**问题标注：**
- ✅ **设计正确**：`deferred`参数可以控制是否立即写回
- ⚠️ **中等严重度**：但批处理代码没有正确使用这个机制
- **问题**：`deferred=True`时返回的新embedding没有被用于后续的loss计算
- **修复建议**：要么改为`deferred=False`逐条更新，要么修改loss计算逻辑使用返回的新embedding

### 代码段5：消息聚合逻辑（仅hybrid_jodie模型）

**文件：** `models/hybrid_jodie.py:224-248`

```python
if self.message_mode == "peer":
    user_msg = proj_item  # 直接使用对方embedding
    item_msg = proj_user
else:
    user_neighbors = self._neighbors(graph_state, uid)  # 从图状态获取邻居
    item_neighbors = self._neighbors(graph_state, iid)
    
    # 🔴 问题点：聚合邻居消息
    user_msg = self.event_operator.event_aggregate(
        center_idx=uid,
        center_emb=proj_user.squeeze(0),
        memory=self.memory,  # 使用全局memory
        neighbors=user_neighbors,
        edge_last_time=graph_state["edge_last_time"],
        current_time=ts,
    ).unsqueeze(0)
```

**问题标注：**
- 🔴 **高严重度**：这是TGN风格的消息聚合，但与t-Batch的"无聚合"原则冲突
- ⚠️ **中等严重度**：批处理时，`graph_state`在batch内不更新，导致邻居信息过时
- **问题**：如果声称实现t-Batch，不应该有这个聚合逻辑；如果实现TGN，应该正确处理batch内的图更新
- **修复建议**：
  - 方案A（t-Batch）：移除消息聚合，使用`message_mode="peer"`
  - 方案B（TGN）：在batch内也更新图状态，或使用时间窗口批处理

### 代码段6：延迟写回后的统一更新

**文件：** `models/training.py:514-525`

```python
# 阶段4: 写回更新
with torch.no_grad():
    for (node_type, node_id), (new_emb, ts, new_c) in node_updates.items():
        if node_type == 'user':
            model.user_embeddings[node_id] = new_emb.detach()
            model.user_last_time[node_id] = ts
            if has_lstm and new_c is not None:
                model.user_cell_state[node_id] = new_c
        else:  # item
            model.item_embeddings[node_id] = new_emb.detach()
            model.item_last_time[node_id] = ts
            if has_lstm and new_c is not None:
                model.item_cell_state[node_id] = new_c
```

**问题标注：**
- ⚠️ **中等严重度**：写回发生在loss计算和反向传播之后
- **问题**：这个时序是错误的，应该在计算loss之前写回，或者使用返回的新embedding计算loss
- **修复建议**：将阶段4移到阶段3之前，或者重构loss计算逻辑

---

## 步骤5：快速修复方案

基于诊断结果，给出两个修复方案的对比和推荐。

### 方案A：标准t-Batch（无损并行，优先保证准确率）

**核心原则：**
- Batch内节点唯一（无重复user或item）
- 无消息聚合（或使用peer模式）
- Batch内交互可并行处理
- 立即更新embedding（`deferred=False`）

**需要修改的文件和函数：**

1. **`models/training.py:455-527`** - `train_partition_bpr_batch`
   - 修改batch组装：添加节点去重逻辑，确保batch内每个user/item只出现一次
   - 移除"last"聚合策略：改为并行处理所有交互
   - 移除`deferred=True`：改为`deferred=False`立即更新
   - 简化为类似逐条训练的逻辑，但batch内交互可并行

2. **`models/training.py:530-635`** - `train_partition_ce_batch`
   - 同上修改

3. **`models/hybrid_jodie.py`** - 模型配置
   - 设置`message_mode="peer"`禁用邻居聚合
   - 或者移除`enable_event_agg`相关逻辑

**优点：**
- ✅ 准确率与逐条训练一致（无损）
- ✅ 实现简单，逻辑清晰
- ✅ 适合JODIE原始设计

**缺点：**
- ❌ 加速有限（batch_size受节点唯一性约束）
- ❌ 需要动态batch组装，增加数据加载复杂度


### 方案B：TGN窗口式（有损并行，优先保证速度）

**核心原则：**
- 允许batch内节点重复
- 使用消息聚合（mean/sum/attn）
- 按时间窗口分batch
- 正确处理batch内的状态更新

**需要修改的文件和函数：**

1. **`models/training.py:455-527`** - `train_partition_bpr_batch`
   - 修改"last"策略为真正的消息聚合：对同一节点的多条交互使用mean/sum聚合
   - 修改loss计算：使用`node_updates`中的新embedding，而不是从模型缓冲区读取
   - 调整更新顺序：先写回embedding（阶段4移到阶段3之前），再计算loss

2. **`models/training.py:530-635`** - `train_partition_ce_batch`
   - 同上修改

3. **添加时间窗口分batch逻辑**（新增函数）
   - 按固定时间窗口（如1秒）或固定交互数分batch
   - 确保batch内交互时间跨度不会太大

**优点：**
- ✅ 加速明显（batch_size不受约束）
- ✅ 适合大规模数据集
- ✅ 符合TGN论文设计

**缺点：**
- ❌ 准确率会有损失（消息聚合是近似）
- ❌ 实现复杂度高
- ❌ 需要调优聚合策略和窗口大小

---

### 推荐方案：**方案A（标准t-Batch）**

**推荐理由：**

1. **当前问题的根源是实现错误，而非方案选择**
   - 现有代码试图实现TGN但实现错误（"last"策略丢失信息）
   - 修复为正确的t-Batch比修复为正确的TGN更简单

2. **准确率优先**
   - 你提到"准确率异常"，说明当前最紧迫的是恢复准确率
   - t-Batch是无损的，可以作为baseline验证其他优化

3. **实现成本低**
   - 只需修改batch组装逻辑和移除聚合
   - 不需要重新设计训练流程

4. **易于调试**
   - t-Batch的行为可预测，容易验证正确性
   - 可以通过对比逐条训练的结果来验证

**具体修改清单：**

| 文件 | 函数 | 修改内容 | 优先级 |
|------|------|---------|--------|
| `models/training.py` | `train_partition_bpr_batch` (402行) | 添加节点去重的batch组装逻辑 | 🔴 高 |
| `models/training.py` | `train_partition_bpr_batch` (469-487行) | 移除"last"策略，改为并行处理 | 🔴 高 |
| `models/training.py` | `train_partition_bpr_batch` (480行) | 将`deferred=True`改为`deferred=False` | 🔴 高 |
| `models/training.py` | `train_partition_bpr_batch` (502行) | 移除阶段3的loss计算（改为在forward中计算） | 🔴 高 |
| `models/training.py` | `train_partition_bpr_batch` (514-525行) | 移除阶段4（不再需要延迟写回） | 🔴 高 |
| `models/training.py` | `train_partition_ce_batch` (530行) | 同上修改 | 🔴 高 |

---

## 总结

### 核心问题

当前代码混淆了t-Batch和TGN两种批处理策略，导致：

1. 🔴 **同一节点的多条交互被"last"策略丢弃**（信息丢失）
2. 🔴 **Loss计算使用了batch处理前的旧embedding**（训练信号错误）
3. 🔴 **Batch内允许节点重复但没有正确聚合**（既不是t-Batch也不是TGN）

### 推荐修复路径

1. **立即修复为标准t-Batch（方案A）**，恢复准确率
2. **验证准确率恢复到逐条训练水平**
3. **如果需要更高加速，再考虑实现正确的TGN（方案B）**

### 运行时验证建议

如果你想在运行时确认问题，可以在以下位置添加调试print语句：

**验证点1：检查"last"策略丢弃的交互**

在 `models/training.py:471` 之后添加：
```python
print(f"[DEBUG] User {uid} has {len(user_batch)} interactions in batch, "
      f"using last at t={last_interaction.timestamp}")
if len(user_batch) > 1:
    print(f"[DEBUG] Discarded {len(user_batch)-1} interactions: "
          f"{[i.timestamp for i in user_batch_sorted[:-1]]}")
```

**验证点2：检查embedding不一致**

在 `models/training.py:502` 之后添加：
```python
old_emb = model.user_embeddings[uid][:3].cpu().numpy()
new_emb = node_updates[('user', uid)][0][:3].cpu().numpy()
print(f"[DEBUG] Interaction at t={interaction.timestamp}")
print(f"  Old embedding (from buffer): {old_emb}")
print(f"  New embedding (from updates): {new_emb}")
print(f"  Difference: {new_emb - old_emb}")
```

**验证点3：检查batch内节点重复情况**

在 `models/training.py:463` 之后添加：
```python
total_interactions = len(batch)
unique_users = len(user_interactions)
unique_items = len(item_interactions)
print(f"[DEBUG] Batch size: {total_interactions}, "
      f"unique users: {unique_users}, unique items: {unique_items}")
if total_interactions > unique_users + unique_items:
    print(f"[DEBUG] WARNING: Batch has duplicate nodes!")
```

这些调试语句可以直观地展示当前实现的问题。

---

## 附录：相关文件索引

- **训练逻辑**：`models/training.py`
- **JODIE RNN模型**：`models/jodie_rnn.py`
- **混合GNN+JODIE模型**：`models/hybrid_jodie.py`
- **消息聚合实现**：`models/gnn_encoder.py`
- **数据结构**：`data/synthetic.py`, `data/temporal_partition.py`

---

**报告生成完成。建议优先实施方案A（标准t-Batch），恢复准确率后再考虑性能优化。**
