# factory.md — 对应源码 [factory.py](../../jodie/models/factory.py)

> **定位**:配置字典 → 模型实例的**转换器**。NAS 搜索出的每一个架构(config)都要经过它变成活生生的模型。文件很短,但它是"搜索空间"与"模型实现"之间的接口,是整个 NAS 框架能运转的结构性原因之一。

---

## 🔑 build_model(源码 11-47 行)

```python
def build_model(config: Dict):
    model_type = config.get("model", "temporal_event_gnn_jodie")

    if model_type in {"temporal_event_gnn_jodie", "hybrid"}:
        return TemporalEventGNNJODIE(
            num_users=config["num_users"],
            ...
            event_agg=config.get("event_agg", "mean"),
            ...
        )
    if model_type == "jodie_rnn":
        return JODIERNN(...)
    raise ValueError(f"Unsupported model type: {model_type}")
```

**读法**:从 config 字典取 key,取不到就用默认值。NAS 控制器生成的就是这个字典(比如 `{"model": "jodie_rnn", "memory_cell": "gru", "time_proj": "off", ...}`),工厂负责把它变成对象。**config 就是"架构"的完全描述**——所有搜索维度都在这个字典里。

---

## 🔑 字符串开关的解析(源码 42-44 行)

```python
use_time_proj = str(config.get("time_proj", "linear")).lower() not in {"off", "none"}
use_static_embeddings = str(config.get("use_static_embeddings", "on")).lower() not in {"off", "none", "false", "0"}
```

**为什么开关用字符串 "on"/"off" 而不是布尔 True/False?** 因为架构配置要**可序列化**:要写进 JSON 文件(best_arch.json)、要跨进程传给 Ray worker、要打印在 leaderboard CSV 里。字符串在这一切场景里都安全且可读。`.lower()` + 排除集合的写法是防御性编程:什么大小写、什么写法都容错。

---

## 📄 工厂模式对 NAS 的意义(论文可写的一段)

工厂把"**架构描述**"和"**模型实现**"解耦了:

1. 控制器/搜索算法只操作 config 字典——它完全不关心模型内部有几层
2. 训练器对所有架构用**同一套训练循环**——模型通过统一接口(forward 签名一致)插入
3. 搜索、评分、排序全部围绕"配置"这个干净的数据结构进行

这就是为什么整个框架能同时搜两种结构迥异的模型(JODIERNN 和 Hybrid)而训练代码一行不改:**统一接口 + 工厂解耦**。写论文时,这一小段可以扩展成 Method 里"Search Space and Model Instantiation"的一小节。

---

## ❓ 检查题

**E7.** 如果 config 里写了 `"time_proj": "OFF"`(大写),工厂会怎么处理?为什么要这样防御?

**E8.** 用自己的话解释:工厂模式让"NAS 控制器"和"模型实现"解耦有什么好处?如果控制器直接 new 模型对象会有什么麻烦?
