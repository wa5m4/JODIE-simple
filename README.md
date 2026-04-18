# Temporal Event-GNN JODIE + GraphNAS（RL搜索）

本项目实现的是**事件驱动**的时序图神经网络 JODIE：

- 每条交互事件到来时，立即执行图操作（邻域聚合/注意力/时间衰减）
- 每条事件后，立即更新用户与物品记忆，以及动态图状态（邻接、边时间、边权）
- 基于该事件级时序GNN模型，用 GraphNAS 搜索图模块与时序模块
- 支持 **强化学习（REINFORCE）** 控制器进行架构搜索

---

## 1. 项目结构

```text
jodie-simple/
├── search.py                 # GraphNAS入口（支持 random / rl）
├── README.md
├── data/
│   └── synthetic.py          ## 交互数据 + 动态图状态初始化
├── models/
│   ├── factory.py            # 构建 TemporalEventGNNJODIE
│   ├── hybrid_jodie.py       # 事件级时序GNN-JODIE主模型
│   ├── gnn_encoder.py        # 事件级图操作器（mean/sum/attn）
│   ├── jodie_rnn.py          # 保留的JODIE子模块（兼容）
│   └── training.py           # 流式训练/评估 + BPRLoss
└── nas/
    ├── search_space.py       # 可搜索模块定义
    ├── controller.py         # Random + RL(REINFORCE) 控制器
    └── trainer.py            # 架构评估与训练调度
```

---

## 2. 核心算法：每事件图操作 + 时序记忆更新

主模型文件：[models/hybrid_jodie.py](models/hybrid_jodie.py)

对每条事件 `(u, i, t, f)`：

1. 读取事件前记忆 `memory[u], memory[i]`
2. 计算时间差并做时间投影（`off/linear/mlp`）
3. 用当前动态图邻居做**事件级聚合**：
   - `event_agg`: `mean/sum/attn`
   - `attn_type`: `dot/mlp`
   - `time_decay`: `none/exp/inverse`
4. 将图消息 + 事件特征输入记忆更新单元：
   - `memory_cell`: `rnn/gru/add`
5. 可选记忆门控：`memory_gate`=`on/off`
6. 写回记忆与最近时间
7. 更新动态图状态：新增边、更新边权、边最近时间、邻域裁剪

> 这里没有静态整图编码；图操作在事件循环中发生。

---

## 3. 可搜索模块（GraphNAS）

定义文件：[nas/search_space.py](nas/search_space.py)

小搜索空间包含：

- `embedding_dim`: `[32, 64]`
- `event_agg`: `[mean, sum, attn]`
- `attn_type`: `[dot, mlp]`
- `time_decay`: `[none, exp, inverse]`
- `max_neighbors`: `[10, 20, 40]`
- `memory_cell`: `[rnn, gru, add]`
- `time_proj`: `[off, linear, mlp]`
- `memory_gate`: `[on, off]`

模型名固定：`temporal_event_gnn_jodie`

---

## 4. 强化学习搜索（REINFORCE）

控制器文件：[nas/controller.py](nas/controller.py)

- 每个可搜索维度维护可学习 logits
- 每轮按策略分布采样一个架构
- 奖励使用验证 Recall@K
- 用基线减法（moving baseline）做 advantage，执行 REINFORCE 更新

可通过参数切换：
- `--search-mode rl`（默认）
- `--search-mode random`

---

## 5. 运行方式

### 5.1 强化学习搜索（推荐）

```bash
python search.py --search-mode rl --trials 6 --epochs-per-trial 1
```

### 5.2 随机搜索（对照）

```bash
python search.py --search-mode random --trials 6 --epochs-per-trial 1
```

常用参数：

- `--controller-lr`（RL控制器学习率）
- `--num-users`
- `--num-items`
- `--num-interactions`
- `--feature-dim`
- `--lr`
- `--k`
- `--seed`
- `--output-dir`

---

## 6. 输出文件

默认输出到 `outputs/`：

- `best_arch.json`：最佳架构与分数
- `leaderboard.csv`：所有候选排序

其中 `config_json` 会包含事件级可搜索字段（`event_agg/memory_cell/time_proj/...`）。

---

## 7. 一句话说明

这是一个“**每事件图操作 + 时序记忆更新**”的 Temporal Event-GNN JODIE，并在该模型上做了可切换 Random/RL 的 GraphNAS 搜索。