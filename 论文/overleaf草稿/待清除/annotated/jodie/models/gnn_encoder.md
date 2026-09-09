# gnn_encoder.md — 对应源码 [gnn_encoder.py](../../jodie/models/gnn_encoder.py)

> **定位**:Hybrid 模型的"图聚合"组件。**搜索空间三大维度(`event_agg` / `attn_type` / `time_decay`)的直接实现**,也是本项目相对原版 JODIE 的核心扩展点。

---

## 🔑 三种聚合方式(源码 100-162 行)

一次交互发生时,中心节点 u 要问它的邻居们:"你们有什么要告诉我?"三种回答方式:

### sum(加权求和)

```python
w = self._decay_weight(delta_t).unsqueeze(-1)
out = (msg * w).sum(dim=0)
```

邻居消息按时间衰减加权后**直接相加**。特点:邻居越多,输出数值越大。

### mean(加权平均)

```python
w = self._decay_weight(delta_t).unsqueeze(-1)
norm = w.sum(dim=0).clamp(min=1e-8)
out = (msg * w).sum(dim=0) / norm
```

除以权重总和做**归一化**。特点:邻居数量不影响输出量级。`clamp(min=1e-8)` 防止除以 0。

### attn(注意力)

```python
score = self._attention_score(center_expand, neigh_emb, delta_t)
alpha = F.softmax(score, dim=0).unsqueeze(-1)
out = (msg * alpha).sum(dim=0)
```

每个邻居先算一个"重要性分数",softmax 归一化成概率,再加权求和。**模型自己决定听谁的**——和中心节点更相关的邻居获得更大话语权。

📄 **论文角度**:sum / mean / attn 三者的本质是信息聚合光谱上的三个点——**等权加和 → 时间衰减加权 → 内容相关加权**。哪些场景需要"听内容"、哪些场景简单加权就够,正是 NAS 要替你回答的问题。

---

## 🔑 时间衰减(源码 62-71 行)

```python
if self.time_decay == "exp":      # 指数衰减(快)
    return torch.exp(-delta_t.clamp(min=0.0))
if self.time_decay == "inverse":  # 反比例衰减(慢)
    return 1.0 / (1.0 + delta_t.clamp(min=0.0))
return torch.ones_like(delta_t)   # none:不衰减
```

**直觉**:邻居多久前和我交互过?越久远,它的消息越该打折扣。`exp` 衰减快(Δt=10 时权重 e⁻¹⁰≈0.00005),`inverse` 衰减慢(1/11≈0.09)。`none` = "我不管时间,一视同仁"。

---

## 🔑 注意力分数:内容与时间的融合(源码 73-89 行)

```python
if self.attn_type == "dot":
    score = (center_emb * neigh_emb).sum(dim=-1)       # 点积:两张卡片"像不像"
else:
    feat = torch.cat([center_emb, neigh_emb, (center_emb - neigh_emb).abs()], dim=-1)
    score = self.attn_mlp(feat).squeeze(-1)            # MLP:学一个打分器
return score + torch.log(self._decay_weight(delta_t) + 1e-8)   # ← 关键一行
```

**最后一行是精髓**:最终分数 = 内容相似度 + log(时间衰减)。取 log 是因为"内容相关性 × 时间新鲜度"的乘法在 log 空间变成加法,和 softmax 前的分数域匹配;`1e-8` 防止 log(0)。于是注意力**天然兼顾"这个邻居和我像不像"与"这个邻居是不是陈年旧事"**。搜索空间的 `attn_type: dot/mlp` 决定内容部分怎么算。

---

## 📖 防御性细节(源码 117-121、47 行)

- 邻居索引 clamp:图状态由外部维护,可能出现越界索引,clamp 兜底防崩溃
- `msg_linear`:聚合前先对邻居卡片做线性变换,可开关(`msg_linear` 搜索维度)

---

## ❓ 检查题

**E3.** sum 聚合和 mean 聚合的结果在什么情况下几乎一样?(提示:想想"除不除以 norm"的差别什么时候消失)

**E4.** 时间衰减在 attn 聚合里是怎么结合的?在 sum/mean 聚合里呢?两处的结合方式一样吗?
