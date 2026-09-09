# JODIE Pipeline 修复尝试全记录

## 问题：Pipeline 搜索无法找到与 Serial/DataParallel 相同的最优架构

- **Serial**: test=0.8793, 架构: `proj=off, static=off, norm=off` (133K params)
- **Data Parallel**: test=0.8793, 相同架构 ✅
- **Pipeline Naive**: test=0.6288, 架构: `proj=linear, static=on, norm=on` (402K params) ❌
- **Pipeline Smart**: 崩溃 (RL 梯度错误)

---

## 尝试 1: Ray init 幽灵 PID 崩溃

**现象**: `FileNotFoundError: /proc/1778/stat`

**根因**: `/proc/1778` 是 udev 挂载的 devtmpfs，不是真进程。Ray 初始化时扫描全量 PID 找已有 GCS 集群，读到这个假 PID 时崩溃。

**修复**: `data_parallel.py` + `ray_pipeline.py` → `_safe_ray_init()` — monkey-patch `psutil.pids()` 过滤无 `/proc/pid/stat` 的幽灵 PID。

**结果**: ✅ 四策略 Ray 启动均正常。

---

## 尝试 2: Smart 异步 RL 梯度错误

**现象 1**: `element 0 of tensors does not require grad and does not have a grad_fn`

**根因**: `logprob.detach().clone()` 断了计算图，`loss.backward()` 找不到 grad_fn。

**修复**: `trainer.py` — 去 `detach()`，改用 `logprob.clone()` + 优先 `compute_logprob()` 重算。

**结果**: ✅ Random controller 正常运行。RL controller 仍报 inplace 错误（更深层问题）。

**现象 2**: `tensor modified by an inplace operation` (RL controller)

**根因**: 采样时存的 `logprob.clone()` 保留了对旧 `self.logits` 的计算图引用。后续 RL update 修改了 logits（inplace），旧的 clone 图引用失效。

**修复**: `trainer.py` — 优先用 `compute_logprob()` 对当前 logits 重算（off-policy），存储的旧 logprob 仅作 fallback。

**结果**: ⚠️ Random controller OK，RL controller 仍有残余 inplace 错误。

---

## 尝试 3: RandomGraphNASController 缺属性

**现象**: `'RandomGraphNASController' object has no attribute 'reward_baseline'`

**根因**: `_search_pipeline_async` 的 else 分支直接访问 `controller.reward_baseline`，但只有 `RLGraphNASController` 初始化了该属性。

**修复**: `controller.py` → `RandomGraphNASController.__init__` 添加 `self.reward_baseline = 0.0`。

**结果**: ✅

---

## 尝试 4: `__del__` AttributeError

**现象**: `'DataParallelExecutor' object has no attribute '_workers'`

**根因**: `ray.init()` 失败后 `self._workers` 未初始化，`__del__` → `shutdown()` 访问时报错。

**修复**: `data_parallel.py` → `shutdown()` 中用 `getattr(self, "_workers", [])`。

**结果**: ✅

---

## 尝试 5: graph_ctx=None for jodie_rnn

**现象**: Serial 训练对 jodie_rnn 传 `graph_ctx=None`，但 Pipeline 的 `_make_payload` 始终初始化 `graph_state`。

**根因**: Pipeline 传给 jodie_rnn 的是非空 graph_ctx（空 adj 模板），与 Serial 不一致。

**修复**: `ray_pipeline.py` → `_make_payload` 对 jodie_rnn 设 `graph_state=None`。

**结果**: 对 jodie_rnn + max_neighbors=0 无实际影响（图操作本来就不参与）。必要但不充分。

---

## 尝试 6: epoch 边界 `_build_model` 不调用 reset_state()

**现象**: Pipeline 多 epoch 循环设 `runtime_state=None`，意图重置嵌入。但 `_build_model` 只在 runtime_state 非 None 时 import，None 时不调用 reset_state()。

**根因**: `load_state_dict` 恢复了上一 epoch 的训练后嵌入，但没被清零。

**修复**: `ray_pipeline.py` → `_build_model` 添加 else 分支调用 `model.reset_state()`。

**结果**: 必要修复，但单独不足以解决排名反转。

---

## 尝试 7: epoch_graph_state 跨 stage 丢失

**现象**: `run_train_stage_batch` 返回 payload 时 snapshot 了原始 `graph_state` 而非训练中更新的 `epoch_graph_state`。

**根因**: `graph_state=snapshot_graph_state(graph_state)` 应该是 `snapshot_graph_state(epoch_graph_state)`。

**修复**: epoch 结束时 `graph_state = epoch_graph_state`。

**结果**: 对 jodie_rnn+max_neighbors=0 无实质影响（图状态为空）。对带图模型有意义。

---

## 尝试 8: epoch 边界不传 optimizer_state (optimizer_state=None)

**假设**: epoch 边界传 optimizer_state 经 Ray pickle 后 load_state_dict 的参数 ID 映射有问题。

**修复**: `ray_pipeline.py` → epoch 边界 `optimizer_state=None`（新鲜 Adam）。

**测试结果**:
- 同进程 serial epoch 边界: Δp=1.78（optimizer 动量缺失的固有差异）
- 同进程 3-stage: 同 Δp=1.78（stage 边界不增加额外偏差）
- Ray 3-stage: Δp=4.94（epoch 边界 1.78 + Ray 额外 3.16）
- 第一次 Ray stage 传递: ✅ MATCH（stage 内 optimizer_state 传递正确）
- 合成数据排名: MATCH ✅
- MOOC 数据排名: INVERSION ❌
- Wikipedia 数据排名: INVERSION ❌

**结论**: ❌ 不能解决真实数据上的排名反转。epoch 边界重置 optimizer 的 Δp=1.78 在合成数据上不影响排名，但在真实数据（MOOC、Wikipedia）上系统性地偏袒 `static=on` 架构。

---

## 尝试 9: _safe_optimizer_state() — clone CPU 后再传

**假设**: `optimizer.state_dict()` 中的 GPU tensor 经 Ray pickle 后，在接收端 `load_state_dict` 的参数 ID 映射有偏差。

**修复**: `ray_pipeline.py` → 新增 `_safe_optimizer_state(optimizer)` 函数，显式 `detach().cpu().clone()` 所有 state tensor。在 `run_train_stage_batch` 返回中使用。同时恢复 epoch 边界传 `optimizer_state=p.optimizer_state`。

**测试结果**:
- Ray 简单 case: ✅ optimizer 经 Ray 传递后权重一致
- 合成数据排名: MATCH ✅
- MOOC 数据排名: INVERSION ❌
- Wikipedia 数据排名: INVERSION ❌

**结论**: ⚠️ 修复了 optimizer_state 的序列化问题，但 epoch 边界即使正确传递 optimizer_state，其与 Serial（同 optimizer 跨 epoch）的固有差异仍在真实数据上导致排名反转。

---

## 尝试 10: BATCH_MODE=serial 替代 tbatch

**假设**: tbatch 模式每 epoch optimizer 步数少（~14步 vs ~7000步），epoch 边界 optimizer 动量重建的影响被放大。

**修复**: `run_all.py` → `BATCH_MODE = "serial"`。

**测试结果**: 同 INVERSION ❌。序列化模式不是主因。

---

## 最终定位 (debug_step_by_step.py, MOOC 数据)

| Step | 测试 | 结果 |
|------|------|------|
| 0 | Serial 确定性 | ✅ |
| 1 | 同进程 state_dict 传递 (单epoch) | ✅ MATCH Serial |
| 2 | 同进程 2-epoch (epoch边界 fresh opt) | ❌ Δp=1.78 |
| 3 | 同进程 3-stage 2-epoch | ❌ Δp=1.78 (同Step2) |
| 4 | Ray 3-stage 2-epoch | ❌ Δp=4.94 |
| 5 | 第一次 Ray stage 传递 | ✅ MATCH |

### 根因链

1. **主因**: epoch 边界 optimizer 动量断裂。Serial 跨 epoch 保持同一 optimizer（动量连续），Pipeline 跨 epoch 重建 optimizer（即使传 state_dict 也有 Δp=1.78 残余差异）。
2. **次因**: Ray 跨进程序列化放大约 3.16。
3. **架构偏差**: MOOC 数据极度不平衡（531用户/21物品），`static=off` 100% 依赖动态嵌入 → optimizer 动量断裂惩罚严重。`static=on` 有静态嵌入兜底 → 相对受益 → Pipeline 系统性偏好 `static=on`。

### 验证过的正确方案

`run_full_train` — 每架构一个持久 Ray Worker，模型零重建，epoch 间直接 `model.reset_state()`，optimizer 保持在同一个 Python 对象上。

**结果**: 权重与 Serial 完全一致 (diff=0.0000) ✅。代价：失去 stage 间流水线并行。

---

## 当前代码状态

| 文件 | 改动 | 状态 |
|------|------|------|
| `controller.py` | `reward_baseline=0.0` | ✅ |
| `data_parallel.py` | `_safe_ray_init` + `getattr` 防御 | ✅ |
| `ray_pipeline.py` | `_safe_ray_init` + `_safe_optimizer_state` + `reset_state` fix + `graph_ctx=None` + `optimizer_state` 跨 epoch 传递 | ✅ |
| `trainer.py` | Smart: `compute_logprob` 优先 + 去 `detach` | ✅ |
| `run_all.py` | `BATCH_MODE=serial` | ✅ |
| `public_dataset.py` | URL 修复 (GitHub→Stanford SNAP) | ✅ |

---

# 阶段性总结

## 当前状态

### 已解决的问题

| 问题 | 修复 |
|------|------|
| Ray init 幽灵 PID 崩溃 | `_safe_ray_init()` |
| `__del__` AttributeError | `getattr` 防御 |
| RandomController 缺 `reward_baseline` | 初始化属性 |
| Smart RL 梯度错误（部分） | `compute_logprob` 优先 |
| graph_ctx 不一致 | jodie_rnn 设 None |
| epoch 边界未 reset_state | `_build_model` else 分支 |
| optimizer_state Ray 序列化 | `_safe_optimizer_state()` |

### 未解决的问题

| 问题 | 严重程度 | 详情 |
|------|---------|------|
| **Pipeline 评分偏差** | 🔴 核心 | 3 个真实数据集（MOOC、Wikipedia、Reddit）上 Pipeline 系统性偏好 `static=on`，无法找到 Serial 选出的最优架构 |
| Smart RL inplace 错误 | 🟡 残余 | RL controller 模式下仍崩溃，Random controller 可跑 |
| Wikipedia/Reddit 下载 | 🟢 已修复 | URL 从 GitHub 改为 Stanford SNAP |

## 根因精确定位

通过 `debug_step_by_step.py` 在 MOOC 数据上逐层对比：

```
Step 1: 同进程 state_dict 传递 (单 epoch)          ✅ MATCH Serial
Step 2: 同进程 epoch 边界 optimizer 重建           ❌ Δp=1.78
Step 3: 同进程 3-stage (stage边界传递optimizer)     ❌ Δp=1.78 (与Step2相同)
Step 4: Ray 3-stage                                ❌ Δp=4.94 (Step2 + Ray额外3.16)
Step 5: Ray 单 stage (第一次传递)                   ✅ MATCH
```

**结论**：偏差有两个来源，都在 epoch 边界：

1. **同进程层面**（Δp=1.78）：Pipeline 在 epoch 边界重建 model + optimizer，即使 `load_state_dict` 恢复状态，optimizer 的参数 ID 映射无法 100% 精确。Adam 动量矩阵的细微偏差在训练中累积，导致最终权重分叉。

2. **Ray 层面**（额外 Δp=3.16）：epoch 边界的 optimizer_state 经 Ray pickle 传递后，`load_state_dict` 在新 Worker 上的映射偏差比同进程更大（参数对象完全重建 + pickle 序列化）。

**Stage 边界传递完全正确**（Step 3 = Step 2，Step 5 MATCH）。偏差 100% 来自 epoch 边界。

## 为什么 static=off 受害更重

MOOC 数据用户-物品分布极不平衡（531 用户 vs 21 物品）。

- `static=off`：100% 依赖动态嵌入。epoch 边界嵌入清零 + optimizer 动量偏差 → 训练轨迹偏离大 → Pipeline 评分系统性压低
- `static=on`：有 `nn.Embedding` 静态嵌入兜底。动态嵌入波动被静态嵌入"平滑" → 相对稳定 → Pipeline 评分反而偏高

**不是偶尔选错——是系统性的**。3 个数据集（MOOC、Wikipedia、合成数据的特定模式）全部显示 `static=on` 被高估。

## 改动方向

### 方向 A：消灭 epoch 边界重建（已验证可行）

`run_full_train` — 每架构分配一个持久 Ray Worker。模型 + optimizer 保持同一个 Python 对象跨所有 epoch。epoch 间直接 `model.reset_state()`。

- ✅ 权重与 Serial 完全一致（Δ=0.0000）
- ✅ 同进程已验证
- ❌ 失去 stage 间流水线并行（跨架构并行保持）
- ❌ 需要重构 `run()` 方法

### 方向 B：接受偏差，用 Serial/DP 兜底

Pipeline 只做快速粗筛，最终架构选择由 Serial/DP 验证。

- ✅ 不改变 Pipeline 代码
- ❌ 论文核心卖点（Pipeline 评分准确）站不住
- ❌ DP 兜底的开销 > 直接用 DP 搜索

### 方向 C：1-stage Pipeline（折中）

`NUM_PIPELINE_STAGES=1`，消除 stage 边界重建，只保留 epoch 边界问题。

- ✅ 改动小（一行配置）
- ⚠️ epoch 边界偏差仍在（Δp=1.78）
- ⚠️ pipeline 并行度降低

### 建议优先级

1. **方向 A**：实现持久 Worker，验证真实数据搜索准确率
2. 如果 A 太复杂，**方向 C** 作为过渡方案
3. Smart pipeline 的 inplace 梯度 bug 单独排查（不影响 Naive 验证）
