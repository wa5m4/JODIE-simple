# Pipeline 评分准确率专项分析

## 问题定义

同一架构在 Serial（串行）和 Pipeline（流水线）训练下，验证集 MRR 评分不一致。
导致 Pipeline 粗搜索的架构排名与 Serial 真实排名不同，最终选错最优架构。

## 已确认的事实

### 正确的地方
- ✅ 同进程内 state_dict 传递：正确
- ✅ 同进程内 model rebuild + optimizer rebuild：正确
- ✅ Ray 同 epoch 内 stage 间传递：正确
- ✅ Ray Worker 复用（同对象不重建）：正确

### 出错的地方
- ❌ Ray 跨 epoch 传递 optimizer state：Δp 从 1.78 扩大到 4.94
- ❌ Epoch 边界重建 model + optimizer：即使同进程也有 Δp=1.78

### 根因
Adam optimizer 的 `state_dict()` 用 parameter ID 做 key。跨进程传递后 `load_state_dict` 的 ID 映射（依赖 param_groups 顺序）不完全精确。虽然同进程内正确（因为 CPU pickle 保留引用），但 Ray pickle 破坏了某些关联。

## 网上调研

### PyTorch Distributed 方案
PyTorch 1.13+ 提供了 `torch.distributed.checkpoint.state_dict` 模块，包含：
- `get_optimizer_state_dict()` — 将 parameter ID 替换为 FQN（fully qualified name）
- `set_optimizer_state_dict()` — 从 FQN 恢复 optimizer state

但需要 `torch.distributed` 初始化，不适用于 Ray actor 模式。

### Ray RLlib 方案
Ray RLlib PR #61937 修复了 optimizer state restore 时 scalar metadata 被 Tensor 化的 bug。
方案：deep-copy loaded state → 只将 per-parameter state subtree 转到目标设备 → 保持 param_groups 为 Python scalar。

### 简化方案（brute-force）
不跨进程传递 `optimizer.state_dict()`，而是：
1. 在发送端将 optimizer state 的每个 tensor 单独 `.cpu().clone()`
2. 在接收端根据参数名称（而非 ID）手动重建 optimizer state 映射

## 思路

### 思路 A：FQN 映射（中复杂度）

将 optimizer.state_dict() 中的 parameter ID → FQN：
```python
def _optimizer_state_to_fqn(optimizer, model):
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    osd = optimizer.state_dict()
    fqn_state = {}
    for pid, state in osd['state'].items():
        name = id_to_name.get(pid, str(pid))
        fqn_state[name] = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v 
                           for k, v in state.items()}
    return {'state': fqn_state, 'param_groups': osd['param_groups']}

def _optimizer_state_from_fqn(fqn_state, optimizer, model):
    name_to_id = {n: id(p) for n, p in model.named_parameters()}
    pg = fqn_state['param_groups']
    # 将 param_groups 中的 params 从 FQN 转回 ID
    for g in pg:
        g['params'] = [name_to_id.get(n, 0) for n in g['params']]
    optimizer.load_state_dict({'state': fqn_state['state'], 'param_groups': pg})
```

### 思路 B：torch.save/load（低复杂度, 已尝试）

```python
def _pack_optimizer(optimizer):
    buf = io.BytesIO()
    torch.save(optimizer.state_dict(), buf)
    return buf.getvalue()

def _unpack_optimizer(data, optimizer):
    osd = torch.load(io.BytesIO(data))
    optimizer.load_state_dict(osd)
```

已验证：单个步骤的正确性 MATCH。但完整 3-stage 2-epoch 流水线中仍有偏差。

### 思路 C：完全不传 optimizer state（零复杂度）

Epoch 边界 optimizer_state=None（当前代码）。Δp=1.78 的偏差是"新鲜 Adam 动量缺失"，对 serial 模式 7000 步/epoch 影响小，对 tbatch 模式 14 步/epoch 影响大。

配合 BATCH_MODE=serial 使用可接受。

### 思路 D：run_full_train（已验证正确）

每架构一个持久 Worker，模型+optimizer 保持同一 Python 对象跨所有 epoch。缺点：失去 stage 间流水线并行。

## 决定

思路 A（FQN 映射）是理论上最干净的方案。PyTorch 分布式训练就是用 FQN 替代 parameter ID 来保证跨进程兼容性。

下一步：实现思路 A，用 debug_step_by_step.py 的单架构测试验证，如果通过则跑 run_all.py 完整验证。

## 运行决策

先不清算准确率问题，run_all.py 用现有代码先跑——看看最新结果。同时在线索 A 的实现方向继续探索。
