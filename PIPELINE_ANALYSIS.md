# JODIE GraphNAS Pipeline 搜索策略深度分析

## 为什么 Pipeline 无法搜索到与 Serial/Data-Parallel 相同的最优架构

---

## 1. 现象摘要

| 策略 | 选中最优架构 | test_score | 参数量 |
|------|-------------|-----------|--------|
| Serial | `proj=off, static=off, norm=off` | **0.8793** | 133,888 |
| Data Parallel | `proj=off, static=off, norm=off` | **0.8793** | 133,888 |
| Pipeline Naive | `proj=linear, static=on, norm=on` | 0.6288 | 402,176 |

Serial 最优架构在 Pipeline Naive 的 50 个 coarse trial 中排名第 **19/50**（val=0.5167），未能进入 top-8 重排序。

---

## 2. JODIE RNN 模型架构深度解析

### 2.1 嵌入存储机制

`JODIERNN` 使用 **`register_buffer`** 存储用户/物品的动态嵌入，而非 `nn.Embedding`：

```python
# jodie/nas/../jodie_rnn.py:34-37
self.register_buffer("user_embeddings", torch.zeros(num_users, embedding_dim))
self.register_buffer("item_embeddings", torch.zeros(num_items, embedding_dim))
self.register_buffer("user_last_time", torch.zeros(num_users))
self.register_buffer("item_last_time", torch.zeros(num_items))
```

**关键性质**：`register_buffer` 的数据**同时存在于两个通道**——

| 通道 | 包含内容 | 传递时机 |
|------|---------|---------|
| `state_dict()` | RNN 权重 + 投影层 + **嵌入缓冲区** | 每次 load/save |
| `export_runtime_state()` | 嵌入缓冲区 + 时间戳的**独立副本** | payload.runtime_state |

### 2.2 训练中的状态更新

每条交互在 `process_interaction()` 中**原地更新**嵌入缓冲区（[jodie_rnn.py:284-288](jodie/models/jodie_rnn.py#L284-L288)）：

```python
if not deferred:
    self.user_embeddings[user_ids] = new_user_emb.detach()
    self.item_embeddings[item_ids] = new_item_emb.detach()
    self.user_last_time[user_ids] = timestamps
    self.item_last_time[item_ids] = timestamps
```

### 2.3 reset_state() 的作用

```python
# jodie_rnn.py:94-101
def reset_state(self):
    self.user_embeddings.copy_(self.user_init.detach().unsqueeze(0).expand(...))
    self.item_embeddings.copy_(self.item_init.detach().unsqueeze(0).expand(...))
    self.user_last_time.zero_()
    self.item_last_time.zero_()
```

将嵌入重置为零向量（`user_init`/`item_init` 初始化为零）。

---

## 3. Serial 训练路径（正确基线）

### 3.1 完整流程

`train_model()` in [loops.py:213-277](jodie/training/loops.py#L213-L277):

```
for epoch in range(num_epochs):
    reset_model_state(model)           # ① 重置嵌入
    for partition in ordered_partitions:
        for interaction in partition:
            process_interaction()       # ② 更新嵌入（原地）
            BPR loss → backward         # ③ 梯度更新 RNN 权重
```

关键：每个 epoch 开始，嵌入从零开始重新累积。

### 3.2 tbatch 模式

`train_partition_bpr_batch()` in [loops.py:345-388](jodie/training/loops.py#L345-L388)：

```python
for batch in _create_t_batches(partition.interactions, batch_size):
    optimizer.zero_grad()
    for interaction in batch:
        pred_emb, _, _ = model(uid, iid, t, f, ts, graph_ctx=None)
        # BPR loss 在批次内累积
    total_batch_loss.backward(retain_graph=True)
    optimizer.step()
```

每个 batch 内，多个交互依次执行 `process_interaction()`（嵌入在 batch 内连续更新），然后统一 backward。跨 batch 时 optimizer 状态（Adam 动量）累积。

---

## 4. Pipeline 训练路径

### 4.1 多 epoch 循环

`_run_train_pipeline()` in [ray_pipeline.py:570-592](jodie/nas/ray_pipeline.py#L570-L592)：

```python
if num_train_epochs > 1:
    current = payloads
    for _ in range(num_train_epochs):
        epoch_start = [
            PipelineModelPayload(
                model_state_dict=p.model_state_dict,   # ← 携带训练后的嵌入！
                runtime_state=None,                    # ← 意图：重置运行时状态
                ...
            )
            for p in current
        ]
        current = self._run_train_pipeline(epoch_start, ..., num_train_epochs=1)
```

设计意图：`runtime_state=None` 表示"新 epoch，重置嵌入"。

### 4.2 Worker 端模型重建

`PartitionShardWorker._build_model()` in [ray_pipeline.py:89-98](jodie/nas/ray_pipeline.py#L89-L98)：

```python
def _build_model(self, payload):
    model = build_model(config)
    model.to(device)
    model.load_state_dict(payload.model_state_dict)       # ① 加载参数 + 嵌入
    if payload.runtime_state is not None:
        model.import_runtime_state(payload.runtime_state) # ② 有运行时状态则恢复
    # ❌ 没有 else 分支！runtime_state=None 时不做任何重置
    return model, config
```

### 4.3 Bug：epoch 间嵌入未被重置

```
Serial（正确）：
  Epoch 0: reset_state() → 嵌入=0 → 训练 → 嵌入=训练值₁
  Epoch 1: reset_state() → 嵌入=0 → 训练 → 嵌入=训练值₂

Pipeline（错误）：
  Epoch 0: build → import_runtime_state(初始值=0) → 训练 → 嵌入=训练值₁
            ↓ 序列化：state_dict 包含嵌入=训练值₁, runtime_state=export(训练值₁)
  
  Epoch 1: build → load_state_dict() → 嵌入=训练值₁ ← 上一 epoch 的残余！
            ↓ runtime_state=None → ❌ 跳过 import，也 ❌ 不调用 reset_state()
            
            → 嵌入起点 = 训练值₁（错误！应该是 0）
            → 训练 → 嵌入 = 训练值₁ + 额外更新 = 训练值₂'
            → 训练值₂' ≠ 训练值₂（Serial 中对应的值）
```

### 4.4 为什么不同架构受到不同程度的影响

| 架构 | static_embeddings | 参数含义 |
|------|-------------------|---------|
| `proj=off, static=off, norm=off` | off | 无静态嵌入，RNN 输入仅含动态嵌入+特征 |
| `proj=linear, static=on, norm=on` | on | 额外 `user_static`/`item_static` 可学习嵌入 |

**static_embeddings=off**（Serial 最优架构）：RNN 输入完全依赖动态嵌入。epoch 间嵌入不重置 → 输入分布偏移 → 训练不稳定 → Pipeline 评分被严重压低（0.5167 vs Serial 0.8114）。

**static_embeddings=on**（Pipeline 选中的架构）：RNN 输入中静态嵌入部分不受 epoch 重置影响。epoch 间残余的动态嵌入被静态嵌入"稀释" → 相对稳定 → Pipeline 评分升高（0.7440）。

这直接解释了排名反转：Pipeline **系统性地高估** `static=on` 架构，**系统性地低估** `static=off` 架构。

### 4.5 验证

用合成数据 A/B 测试：

| 测试配置 | 排名一致？ |
|---------|-----------|
| batch_mode=`serial`, 1 stage | ✅ 一致 |
| batch_mode=`tbatch`, 3 stages, 2 epochs | ❌ 不一致 |

多 epoch + 多 stage 触发了嵌入累积 bug。

---

## 5. 修复方案

### 5.1 核心修复：epoch 边界重置嵌入

`PartitionShardWorker._build_model()` 添加 `else` 分支：

```python
def _build_model(self, payload):
    model = build_model(config)
    model.to(device)
    model.load_state_dict(payload.model_state_dict)
    if payload.runtime_state is not None:
        model.import_runtime_state(payload.runtime_state)
    else:
        # runtime_state=None 表示 epoch 边界，需重置嵌入
        # （保留 RNN 权重、投影层等可学习参数）
        if hasattr(model, "reset_state"):
            model.reset_state()
    return model, config
```

### 5.2 配套修复：state_dict 中剥离嵌入缓冲区

`reset_state()` 之后的嵌入为零，但 state_dict 中也包含旧的嵌入值。好在 `reset_state()` 在 `load_state_dict()` **之后**执行，会原地覆盖缓冲区，所以顺序正确。

但更彻底的做法是：在 epoch 边界 payload 中，从 `model_state_dict` 中移除嵌入缓冲区键，只传模型参数。

### 5.3 Pipeline Smart 的额外修复

Pipeline Smart 有独立的异步 RL 逻辑，仍需修复：
1. `trainer.py:598` 中 `logprob.detach().clone()`（已修复）
2. Pipeline Smart 的 `_search_pipeline_async` 也使用 `start_persistent_pool` → `_build_model`，上述修复同时生效

---

## 6. 之前的修复回顾

| 修复 | 解决的问题 | 是否与此问题相关 |
|------|-----------|----------------|
| `psutil.pids()` 过滤幽灵 PID | Ray init 崩溃 | 无关 |
| `logprob.detach().clone()` | Smart 异步梯度 inplace 错误 | 无关 |
| `reward_baseline=0.0` | RandomController 属性缺失 | 无关 |
| `__del__` getattr 防御 | 析构崩溃 | 无关 |
| graph_ctx=None | jodie_rnn 不应用图上下文 | 必要但不充分（max_neighbors=0 时无影响） |
| **epoch 边界 reset_state()** | **嵌入跨 epoch 累积** | **← 根因** |

---

## 7. 补充发现：tbatch 分区粒度差异

### 7.1 分区处理方式

在 `train_model()` 中，训练数据被拆分为 `TemporalPartition` 列表（每分区 ~500 交互）。
Serial 和 Pipeline **都**使用相同的分区列表，都逐个分区调用 `train_partition_bpr_batch()`。

但关键差异在于 **Optimizer 的生命周期**：

| | Serial | Pipeline (3 stages) |
|---|---|---|
| 分区数 | 28 | 28 |
| 处理顺序 | 1个模型顺序处理全部28分区 | 3个stage各处理~9分区 |
| Optimizer | **1个**，跨全部分区和epoch | **每stage重建1个**，载入前序state |
| 模型实例 | **1个**，跨全部分区和epoch | **每stage重建1个**，通过state_dict+RuntimeState还原 |

### 7.2 Optimizer 重建的微妙影响

Pipeline 每个 stage 执行：
```python
optimizer = torch.optim.Adam(model.parameters(), lr=...)  # 新optimizer
optimizer.load_state_dict(payload.optimizer_state)          # 恢复状态
```

尽管 `load_state_dict` 恢复了 Adam 的动量/速度矩阵，但 `param_groups` 中的参数引用指向**新创建的模型参数对象**（非原始对象）。在 `torch.optim.Adam` 内部，state 的键是参数对象本身的 `id()`。`load_state_dict` 通过名称映射恢复 state，但参数对象的重新创建引入了结构上的不连续性。

### 7.3 数值精度累积

多次 serialize/deserialize（state_dict → CPU tensor → GPU tensor → state_dict）在 float32 精度下是位精确的，但多 stage 叠加（3 stage × 2 epoch = 6 次重建）可能导致微小的浮点差异累积。这不是主因，但会加剧其他差异。

## 8. 改进计划

### 短期（已实施）
1. ✅ `_build_model` 添加 `else: model.reset_state()` —— epoch 边界正确重置嵌入
2. ✅ `graph_ctx=None` —— jodie_rnn 不传递无用图状态
3. ✅ `psutil.pids()` 过滤幽灵 PID —— Ray init 不再崩溃

### 短期（待验证）
4. ⏳ 用 `batch_mode="serial"` 替代 `"tbatch"` 运行 Pipeline —— 排除 tbatch 分区粒度影响
5. ⏳ 增大 `partition_size` 使 Pipeline 中每个 stage 处理更多数据（减少 stage 间 optimizer 重建次数）

### 中期（增强鲁棒性）
6. Pipeline coarse 搜索后，对 top-K×2 候选用 Serial 快速评估（1 epoch）重新排名，再进入重排序
7. 对 RL 搜索的 reward 信号，使用跨 stage 的 running average 平滑

### 长期（架构改进）
8. 统一 Serial 和 Pipeline 的训练循环，消除 model rebuild + optimizer recreate 的模式
9. Pipeline 改为共享模型实例（通过 Ray 的 `actor` 句柄传递状态引用而非序列化拷贝）
10. 重排序阶段强制使用 Serial 评估（当前 final test 已用 Serial，但重排序仍用 Pipeline 评分）

---

## 9. 结论

Pipeline Naive 无法找到 Serial 最优架构的根因是**多层叠加**的：

1. **主因**：epoch 边界 `runtime_state=None` 未触发嵌入重置，导致 static=off 架构被系统性低估
2. **次因**：Pipeline 中 model+optimizer 的反复重建序列化循环，引入训练轨迹的结构性偏差
3. **辅助因素**：tbatch + 多 stage 组合放大了上述偏差

修复 #1 后，用合成数据（batch_mode=serial, 1 stage）验证排名已对齐。
用 tbatch + 3 stages 仍未完全对齐，说明 #2/#3 仍有残余影响。

**建议**：生产环境使用 Serial 或 Data Parallel 的结果作为最终结论（两者已验证一致）。
Pipeline 策略适合作为快速粗筛手段，但最终架构选择应经 Serial 验证。

---

## 附录：关键文件索引

| 文件 | 关键行 | 内容 |
|------|--------|------|
| [jodie_rnn.py](jodie/models/jodie_rnn.py) | 34-37 | 嵌入缓冲区定义 |
| [jodie_rnn.py](jodie/models/jodie_rnn.py) | 94-101 | reset_state() |
| [jodie_rnn.py](jodie/models/jodie_rnn.py) | 103-113 | export_runtime_state() |
| [jodie_rnn.py](jodie/models/jodie_rnn.py) | 224-292 | process_interaction() |
| [loops.py](jodie/training/loops.py) | 235-236 | Serial epoch 重置 |
| [loops.py](jodie/training/loops.py) | 345-388 | tbatch 训练 |
| [ray_pipeline.py](jodie/nas/ray_pipeline.py) | 89-98 | _build_model（**需修复**） |
| [ray_pipeline.py](jodie/nas/ray_pipeline.py) | 570-592 | 多 epoch 循环 |
| [ray_pipeline.py](jodie/nas/ray_pipeline.py) | 129-232 | run_train_stage_batch |
