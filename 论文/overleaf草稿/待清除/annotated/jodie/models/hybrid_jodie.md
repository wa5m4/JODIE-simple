# hybrid_jodie.md — 对应源码 [hybrid_jodie.py](../../jodie/models/hybrid_jodie.py)

> **定位**:本项目的**旗舰模型**——事件级图神经网络 + JODIE 记忆更新的混合体。搜索空间的主战场,论文 Method 部分的核心描述对象。逻辑比 JODIERNN 多一层:**先做图聚合,再做记忆更新**。

---

## 🔑 一张统一的内存表(源码 59-60、112-130 行)

```python
self.register_buffer("memory", torch.zeros(self.num_nodes, embedding_dim))
...
@property
def user_embeddings(self):
    return self.memory[:self.num_users]        # 用户区:前 num_users 行
@property
def item_embeddings(self):
    return self.memory[self.num_users:...]     # 物品区:后面 num_items 行
```

**设计选择**:用户和物品的卡片放在**同一张表** `memory` 里,用户占前半、物品占后半。四个 property(`user_embeddings` / `item_embeddings` / `user_last_time` / `item_last_time`)是"视图切片"——**对外伪装成和 JODIERNN 一模一样的接口**。这就是为什么训练循环不用改一行代码就能在两个模型间切换(见 factory.md)。

---

## 🔑 节点 ID 偏移(源码 163-168 行)

```python
user_nodes = user_ids
item_nodes = item_ids + self.num_users    # 物品的"节点 ID"= 物品 ID + 用户总数
```

比如 1000 用户、2000 物品:用户 5 的节点 ID 是 5,物品 3 的节点 ID 是 1003。**为什么?** 因为邻接表是"用户-物品二部图"的统一编号空间——用户和物品必须用同一套 ID,否则"5 号节点"分不清是用户还是物品。

---

## 🔑 动态图状态维护(源码 180-190 行)

```python
graph_state["adj"].setdefault(user_node, []).append(item_node)   # 加边
...
self._trim_neighbors(graph_state, user_node)      # 邻居超长时裁到最近 max_neighbors 个
graph_state["edge_last_time"][(user_node, item_node)] = ts       # 边时间
graph_state["edge_weight"][(user_node, item_node)] = ... + 1.0   # 边权重累计
```

每次交互后:往邻接表加边、记录边时间、累计边权重。`_trim_neighbors` 是性能关键——真实数据里热门节点可能有上万邻居,聚合成本失控,所以只保留**最近的** `max_neighbors` 个(列表末尾 = 最近添加)。

**注意图状态不在模型内部**:由外部传入的 `graph_ctx` 持有,模型只读写它(源码 486-487 行注释)。这个设计让执行层可以把"图状态"和"模型状态"分开序列化、跨进程传递——Pipeline 实现的重要前提。

---

## 🔑 message_mode:agg vs peer(源码 407-430 行)

```python
if self.message_mode == "peer":
    user_msg = proj_item            # peer:消息 = 对方的投影卡片(纯 JODIE 式)
else:
    user_msg = self.event_operator.event_aggregate(...)   # agg:消息 = 邻居聚合(默认)
```

`agg`(默认)= 消息来自**邻居聚合**(gnn_encoder 的 mean/sum/attn);`peer` = 消息直接是对方的投影卡片,行为退化接近 JODIERNN。

📄 **论文角度**:message_mode 是一个天然的**消融(ablation)开关**——同一模型骨架下,关掉图聚合(peer)还剩多少性能,直接量化"图结构信息"的贡献。这是实验设计里很值钱的一组对照。

---

## 🔑 memory_gate:门控(源码 202-206 行)

```python
g = torch.sigmoid(self.gate_layer(torch.cat([old_state, new_state], dim=-1)))
return g * new_state + (1 - g) * old_state
```

新旧卡片的**混合器**:g 接近 1 → 全盘接受新卡片;g 接近 0 → 保持旧卡片不变。**动机**:不是每次交互都该同等程度地改写画像(偶尔的误点击不该彻底推翻用户画像),门控让模型自己学"这次该改多少"。

---

## 🔑 process_interaction 全景(源码 374-462 行)

和 JODIERNN 同款骨架,多了图聚合和门控两步:

```
读卡片 → 时间投影 → 图聚合出消息(agg)或取对方卡片(peer)
→ 拼 RNN 输入 → 细胞更新 → 门控混合 → 写回 memory(detach)
→ 更新 last_time → 更新图状态(加边)
```

---

## 📄 JODIERNN vs TemporalEventGNNJODIE 对比表(论文素材)

| 维度 | JODIERNN | Hybrid(agg 模式) |
|------|----------|------------------|
| 消息来源 | 对方投影卡片 | 邻居聚合(mean/sum/attn) |
| 图结构信息 | 不用 | 用 |
| 静态嵌入 | 有(可关) | 无 |
| 门控 | 无 | 有(memory_gate) |
| 参数量 | 少 | 多 |

ATTEMPTS_LOG 里的主角就是它俩:搜索空间 `model` 维度二选一,加上 static/proj/norm 等开关,构成完整架构谱系。

---

## ❓ 检查题

**E5.** 为什么物品的节点 ID 要加 `num_users` 偏移?如果用户和物品各用各的 ID 从 0 编号,邻接表会出什么问题?

**E6.** `_apply_gate` 公式 `g*new + (1-g)*old` 中,g=1 和 g=0 分别对应什么行为?设计门控的动机是什么?
