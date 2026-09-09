# jodie_rnn.md — 对应源码 [jodie_rnn.py](../../jodie/models/jodie_rnn.py)

> **定位**:两个可搜模型之一——**经典 JODIE 互递归模型**(KDD'19 原版思路的实现)。你在"印象卡片"课学的四步(读卡 → 互递归改写 → 投影 → 打分)全部在这里落地。**这是整个项目最重要的单个文件**,建议配合第 3 课笔记反复读。

---

## 🔑 构造函数:模型的"零件清单"(源码 12-82 行)

### 第一组零件:动态嵌入——注意,是 buffer 不是 parameter!(源码 34-37 行)

```python
self.register_buffer("user_embeddings", torch.zeros(num_users, embedding_dim))
self.register_buffer("item_embeddings", torch.zeros(num_items, embedding_dim))
self.register_buffer("user_last_time", torch.zeros(num_users))
self.register_buffer("item_last_time", torch.zeros(num_items))
```

**这是整个项目最值得停下来想的三行。** PyTorch 里模型内部有两种东西:

- `nn.Parameter`:**可学习参数**,靠梯度更新(比如线性层的 W)
- `register_buffer`:**状态存储**,不参与梯度,但跟着模型一起 `.to(device)`、一起 save/load

"用户 5 的卡片"是哪种?**buffer**——因为卡片不是被梯度"学"出来的,而是被每次交互**显式改写**出来的(见下方 process_interaction)。这正是 JODIE 与普通 GNN 的根本区别:普通模型的节点嵌入是参数(训练完就冻结),**JODIE 的嵌入是活的,边跑边写**。

`user_last_time` 记录"每个用户最后一次交互的时间"——算 Δt 要用(第 3 课细节 1 的代码版)。

### 第二组零件:出生卡片 + 静态嵌入(源码 43-55 行)

```python
self.user_init = nn.Parameter(torch.zeros(embedding_dim))   # 所有用户的"出生卡片"

if self.use_static_embeddings:
    self.user_static = nn.Embedding(num_users, embedding_dim)   # 静态属性表
    update_static_dim = embedding_dim * 2
```

- `user_init`:一张**可学习的出生卡片**,训练开始时所有用户卡片都等于它(见 reset_state)。它自己是 parameter,靠梯度学
- `user_static`:你在 C2 问过的"用户静态属性"——普通 Embedding 查表,**不随交互改变**。ATTEMPTS_LOG 里的 `static=on/off` 就是它的开关。static=on 时 RNN 输入里多塞 2 个向量(自己的 + 对方的),所以 `update_static_dim = embedding_dim * 2`

### 第三组零件:RNN 单元 + 输入维度公式(源码 57-69 行)

```python
user_rnn_input = embedding_dim * 2 + update_static_dim + feature_dim + 2
```

对照 C4 你答对的"四样输入",这个公式正好是:

- 自己的投影卡片 + 对方的投影卡片 = `embedding_dim * 2`
- 静态属性(可选)= `update_static_dim`
- 交互特征 = `feature_dim`
- 两个时间差特征(自己的 Δt、对方的 Δt)= `+ 2`

三种"写卡机器"可选:`RNNCell`(最简单)/ `GRUCell`(带门控)/ `LSTMCell`(多一个 cell 状态)。这就是搜索空间里 `memory_cell` 维度。GRU 门控可以理解成:机器自己决定"旧卡片保留多少、新信息吸收多少"。

### 第四组零件:时间投影 + 预测头(源码 71-79 行)

```python
self.user_time_proj = nn.Linear(1, embedding_dim, bias=False)   # 投影:卡片×(1+W·Δt)
...
self.predict_layer = nn.Sequential(
    nn.Linear(predict_in_dim, embedding_dim),
    nn.Tanh(),
    nn.Linear(embedding_dim, embedding_dim),
)
```

第 3 课的投影,代码版就是 `卡片 × (1 + W·Δt)`,"移动方向" W 是学出来的。预测头输入投影后的用户卡片,输出一个**"预测的物品嵌入"**——注意输出的不是分数而是向量:评估时拿它和所有物品嵌入算 L2 距离排名(metrics.py 的做法)。

### reset_state(源码 94-101 行)

```python
self.user_embeddings.copy_(self.user_init.detach().unsqueeze(0).expand(self.num_users, -1))
self.user_last_time.zero_()
```

所有卡片重置为"出生卡片",last_time 清零。**每个架构训练前都调它**,保证不同架构从同一起跑线开始。执行层讲 epoch 边界时这个函数会被反复点名(ATTEMPTS_LOG 的主角之一)。

---

## 🔑 时间投影与时间特征(源码 133-146 行)

```python
def get_projected_embedding(self, node_embedding, delta_t, projection_layer):
    if not self.use_time_proj:
        return node_embedding                      # time_proj=off → 退化为恒等
    time_factor = projection_layer(delta_t)
    return node_embedding * (1 + time_factor)

def _delta_feature(self, delta_t):
    return torch.log1p(torch.clamp(delta_t, min=0.0))
```

两个细节:

1. `use_time_proj=False` 时投影变恒等——这就是搜索空间 `time_proj=off` 的含义:模型不做时间外推
2. `log1p` 把 Δt 压进对数尺度:真实数据的时间间隔可能是 1 秒和 100 万秒并存(长尾分布),取对数后差距从 10⁶ 变成约 13.8,数值稳定得多

---

## 🔑 compute_message:拼装 RNN 输入,互递归在这里(源码 148-186 行)

```python
user_emb_proj = self.get_projected_embedding(user_emb, delta_user, self.user_time_proj)
item_emb_proj = self.get_projected_embedding(item_emb, delta_item, self.item_time_proj)

user_inputs = [user_emb_proj, item_emb_proj]      # 更新用户:自己的投影 + 对方的投影
item_inputs = [item_emb_proj, user_emb_proj]      # 更新物品:正好反过来
...
user_rnn_input = torch.cat(user_inputs, dim=-1)
```

**互递归就是这两行**:更新用户时,输入 = 投影后的自己 + 投影后的对方;更新物品时反序。这正是 C4 你答对的"改写 u 时用了 i 的旧卡片"。

---

## 🔑 process_interaction:改卡片 + detach 的玄机(源码 224-292 行)

```python
new_user_emb = self.user_cell(user_rnn_input, user_emb)   # 写卡机器干活
...
if not deferred:
    self.user_embeddings[user_ids] = new_user_emb.detach()   # ← 注意 .detach()
    self.item_embeddings[item_ids] = new_item_emb.detach()
    self.user_last_time[user_ids] = timestamps
```

**为什么写回缓冲区必须 `.detach()`?** 想一个问题:如果不 detach,新卡片写进缓冲区,下一条交互又把"旧卡片"(=上一条的新卡片)读出来当输入——梯度就会沿着缓冲区里的历史一路回传到**所有之前的交互**。几十万条交互的计算图会瞬间炸掉显存。detach 的含义:**历史写入内存,但计算图到此为止**,梯度只从"当前这次交互"流出。这是时序模型训练的关键手法,论文里值得写一句。

`deferred=False`(默认):立刻写回,严格逐条处理(JODIE 原版方式);`deferred=True`:先不改,由调用方统一写回(TGN 批处理方式,训练层 batching.md 细讲)。

---

## 🔑 predict + forward(源码 294-320 行)

```python
def predict(self, user_ids, query_time):
    user_emb = self.user_embeddings[user_ids]
    delta_t = (query_time - self.user_last_time[user_ids]).unsqueeze(-1)
    projected = self.get_projected_embedding(user_emb, delta_t, self.user_time_proj)
    pred_item_emb = self.predict_layer(...)      # 预测头 → 预测的物品嵌入
    return pred_item_emb, projected

def forward(self, user_ids, item_ids, timestamps, features, query_time, ...):
    pred_item_emb, _ = self.predict(user_ids, query_time)
    new_user_emb, new_item_emb = self.process_interaction(...)
    return pred_item_emb, new_user_emb, new_item_emb
```

predict = 投影到 query_time → 预测头输出物品嵌入。forward = predict + process 的串联,训练和评估都走它。

---

## 📖 export_runtime_state / import_runtime_state(源码 103-126 行)

把"活的状态"(卡片、last_time)打包/灌入。**执行层的跨进程传递全靠它**——Pipeline 把一个 worker 训完的卡片序列化发给下一个 worker。ATTEMPTS_LOG 的排障战役就是围绕"这套传递是否无损"展开的,执行层回来细讲。

---

## ❓ 检查题

**E1.** `user_embeddings` 为什么用 `register_buffer` 而不是 `nn.Parameter`?如果改成 Parameter,在概念上错在哪?

**E2.** `process_interaction` 写回缓冲区时为什么 `.detach()`?如果不 detach,训练长序列交互时会发生什么?
