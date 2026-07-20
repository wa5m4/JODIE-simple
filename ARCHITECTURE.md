# JODIE-simple-refactored 架构文档

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构](#2-目录结构)
3. [逐目录详解](#3-逐目录详解)
4. [逐文件文档](#4-逐文件文档)
   - 4.1 `jodie/data/` -- 数据层
   - 4.2 `jodie/models/` -- 模型层
   - 4.3 `jodie/training/` -- 训练与评估
   - 4.4 `jodie/nas/` -- NAS 框架
   - 4.5 `jodie/baseline/` -- 基线适配器
   - 4.6 入口点
5. [数据流](#5-数据流)
6. [配置参考](#6-配置参考)

---

## 1. 项目概述

### 项目功能

JODIE-simple-refactored 是一个面向**时序图神经网络（Temporal Graph Neural Network）的神经架构搜索（NAS）框架**，专门针对 JODIE（联合动态用户-物品嵌入）模型家族。它支持三种模型类型：

- **TemporalEventGNNJODIE（混合型）**：事件级时序 GNN，在每次交互时通过动态计算图进行消息传递。
- **JODIERNN（纯 RNN）**：无图操作的互递归 RNN 基线，与原始 JODIE 论文一致。
- **官方 JODIE 适配器**：基于子进程的包装器，运行原始 JODIE 代码库以进行对比。

该框架支持三种 NAS 搜索**执行模式**：

| 模式 | 说明 | 并行策略 |
|---|---|---|
| `serial` | 一次评估一个架构，顺序执行 | 无（基线） |
| `data_parallel` | 基于 Ray 的数据并行：在每个架构评估内进行分区级微批次并行 | 架构内并行 |
| `ray_pipeline` | 基于 Ray 的流水线并行：多个架构同时流经分区化的训练/评估阶段 | 架构间并行 |

### 架构图

```
                          ┌─────────────────────────────────┐
                          │        search.py (CLI)           │
                          │   解析 SearchConfig，调度执行     │
                          └──────────┬──────────────────────┘
                                     │
                          ┌──────────▼──────────────────────┐
                          │     GraphNASTrainer (NAS)       │
                          │   编排搜索循环：                 │
                          │   控制器 → 评估 → 排序          │
                          └──┬──────────┬──────────┬───────┘
                             │          │          │
               ┌─────────────┘          │          └──────────────┐
               ▼                        ▼                        ▼
     ┌─────────────────┐   ┌─────────────────────┐   ┌────────────────────────┐
     │    serial       │   │  DataParallelExecutor│   │  RayPipelineExecutor   │
     │  (直接评估)     │   │  (Ray 工作池)        │   │  (多阶段流水线)        │
     └────────┬────────┘   └──────────┬──────────┘   └───────────┬────────────┘
              │                       │                          │
              └───────────┬───────────┴──────────────┬───────────┘
                          │                          │
              ┌───────────▼──────────┐   ┌───────────▼──────────┐
              │   训练循环           │   │ PartitionShardWorker │
              │  serial / tbatch     │   │  (Ray actor, 阶段)   │
              │  TGN 窗口            │   │  训练 + 评估阶段     │
              └───────────┬──────────┘   └───────────┬──────────┘
                          │                          │
              ┌───────────▼──────────────────────────▼──────────┐
              │               模型与数据                        │
              │  TemporalEventGNNJODIE / JODIERNN               │
              │  交互数据集, 时序分区                           │
              └─────────────────────────────────────────────────┘
```

### 执行模式详解

**串行模式** (`execution_mode=serial`)：
- 控制器采样 `coarse_trials` 个架构。
- 每个架构通过 `_evaluate_arch_multi_seed` 顺序训练和评估。
- 粗搜索后，对前 K 个候选架构用更多 epoch 重新排序。
- 最佳架构在 train+val 上重新训练，在 test 上评估。

**数据并行模式** (`execution_mode=data_parallel`)：
- 使用 Ray actor（`_DataParallelWorker`）在每个架构内并行化训练。
- 每个时序分区被拆分为微批次；worker 并发处理它们。
- 梯度在 worker 间取平均；运行时状态通过 `_merge_runtime_states` 合并。
- 架构迭代仍然是顺序的，但每个 trial 的训练被加速。

**Ray 流水线模式** (`execution_mode=ray_pipeline`)：
- 分区被分组为流水线阶段。
- 每个阶段获得一个 Ray 工作池（一个或多个 `PartitionShardWorker` actor）。
- 多个架构同时流经训练-评估流水线（架构间并行）。
- 支持两种流水线模式：`naive`（批次同步）和 `smart`（异步，带持久化池和 RL 离线策略更新）。

---

## 2. 目录结构

```
jodie-simple-refactored/
├── ARCHITECTURE.md                    # 本文件
├── search.py                          # NAS 搜索入口点
├── train.py                           # 单架构训练入口点
├── data/
│   └── public/                        # 数据集 CSV 文件
│       ├── mooc.csv
│       └── tiny_jodie.csv
└── jodie/                             # 核心 Python 包
    ├── __init__.py                    # 包文档字符串 + 重导出
    ├── data/                          # 数据加载与分区
    │   ├── __init__.py                # 空文件
    │   ├── synthetic.py               # 合成交互数据生成
    │   ├── public_dataset.py          # CSV 数据集加载（Wikipedia、Reddit、自定义）
    │   └── temporal_partition.py      # TemporalPartition 与分区构建
    ├── models/                        # 神经网络架构
    │   ├── __init__.py                # 空文件
    │   ├── hybrid_jodie.py            # TemporalEventGNNJODIE（事件级 GNN）
    │   ├── jodie_rnn.py               # JODIERNN（互递归基线）
    │   ├── gnn_encoder.py             # EventGraphOperator（聚合操作）
    │   └── factory.py                 # build_model 工厂函数
    ├── training/                      # 训练循环与评估指标
    │   ├── __init__.py                # 空文件
    │   ├── batching.py                # t-Batch 与 TGN 窗口构建器
    │   ├── loops.py                   # 所有训练循环（BPR/CE, serial/tbatch/TGN）
    │   └── metrics.py                 # MRR、Recall@K 评估
    ├── nas/                           # NAS 框架
    │   ├── __init__.py                # 空文件
    │   ├── search_space.py            # 搜索空间定义 + 清理
    │   ├── controller.py              # 随机与 RL（REINFORCE）控制器
    │   ├── trainer.py                 # GraphNASTrainer 编排
    │   ├── ray_pipeline.py            # 流水线并行（Ray）
    │   ├── data_parallel.py           # 数据并行（Ray）
    │   └── config_optimizer.py        # 自动配置优化器
    └── baseline/                      # 官方 JODIE 基线适配器
        ├── __init__.py                # 空文件
        └── official_jodie.py          # 基于子进程的 JODIE 运行器
```

---

## 3. 逐目录详解

### `jodie/data/` -- 数据层

**角色**：为系统其余部分提供交互数据。支持**合成数据生成**（用于受控实验）和**公共数据集加载**（Wikipedia、Reddit 或自定义 CSV）。

**数据流**：原始事件记录（来自生成或 CSV 解析）被转换为 `Interaction` 数据类实例，然后按时间排序并拆分为用于训练、验证和测试的时序分区。

**关键设计决策**：
- `Interaction` 是一个简单的数据类，包含 `timestamp`、`user_id`、`item_id` 和 `features` 张量——无继承、无多态。
- 分区策略是基于数量的（固定大小块），而不是基于时间的，可选重叠以实现更平滑的过渡。
- 图状态（`init_dynamic_graph_state`）作为可变字典在模型外部维护，支持快照/恢复，从而实现时序回滚和每个 epoch 的独立副本。
- 公共数据集加载包括从 GitHub 自动下载 Wikipedia/Reddit，将原始 ID 重新映射为连续 0 索引 ID，并将特征向量填充/截断到 `feature_dim`。
- 在合成数据生成过程中保存/恢复 RNG 状态，以避免对调用方 RNG 的副作用。

### `jodie/models/` -- 模型层

**角色**：定义可被搜索的神经网络架构。包含两个主要模型家族和图聚合操作符。

**数据流**：模型接收批处理的 `(user_ids, item_ids, timestamps, features, query_time, graph_ctx)` 并产生 `(predicted_item_embedding, new_user_embedding, new_item_embedding)`。图上下文是模型读取/写入的外部字典，用于邻居跟踪。

**关键设计决策**：
- `TemporalEventGNNJODIE` 使用统一的 `memory` 缓冲区（用户嵌入 + 物品嵌入拼接）和节点 ID 偏移，而 `JODIERNN` 使用分离的 `user_embeddings` / `item_embeddings` 缓冲区。统一方法简化了邻居查找。
- 两个模型都暴露 `export_runtime_state()` / `import_runtime_state()` 用于跨流水线阶段的状态序列化。
- `compute_message()` / `apply_aggregated_message()` 两阶段 API 支持 TGN 风格批量训练：首先为窗口内所有交互计算消息，然后按节点聚合并更新。
- `EventGraphOperator` 支持三种聚合模式（`mean`、`sum`、`attn`）和三种时间衰减函数（`none`、`exp`、`inverse`），构成 NAS 搜索空间的核心。

### `jodie/training/` -- 训练与评估

**角色**：包含训练循环（三种粒度：serial、t-Batch、TGN 窗口）和评估指标（MRR、Recall@K、基于类型的 recall）。

**数据流**：`TemporalPartition` 被送入六个训练函数之一（`train_partition_bpr`、`train_partition_bpr_batch`、`train_partition_bpr_tgn`，加上三个 CE 变体）。这些函数产生一个总 loss。评估函数消费模型和分区，产生 `{"hits", "mrr_sum", "total"}`。

**关键设计决策**：
- **t-Batch**：一种贪心批处理策略，保证每个批次中不会有重复的用户或物品 ID，消除模型内存缓冲区中的写后读冲突。这使得无损的并行前向传播成为可能，而不改变数学结果。
- **TGN**：基于时间窗口的批处理，窗口内所有交互首先用于计算每个节点的消息，然后消息被聚合（mean/sum/last），最后更新嵌入。loss 可在所有交互上计算（`loss_mode=all`）或仅在每个节点的最后一次交互上计算（`loss_mode=last`）。
- `BPRLoss` 是主要目标函数：`-log(sigmoid(pos_score - neg_score))`。CE/L2 loss 适用于无负采样的公共数据集。
- 辅助函数（`_item_embeddings_for_loss`、`_all_item_embeddings`、`_model_device`）抽象了 `TemporalEventGNNJODIE` 和 `JODIERNN` 之间的模型属性命名差异。

### `jodie/nas/` -- NAS 框架

**角色**：核心 NAS 引擎。定义搜索空间、控制器（采样策略）、训练编排器以及两个并行执行后端。

**数据流**：`GraphNASTrainer` 用 `base_config` 初始化。它调用 `controller.sample_arch()` 生成架构，评估它们（serial、data_parallel 或 pipeline），按分数排序结果，可选地重新排序前 K 个候选，并返回带有测试集指标的最佳架构。

**关键设计决策**：
- **搜索空间**：四个预设（`small`、`paper_compare`、`rnn_only`、`mixed`）控制哪些超参数是可搜索的。`sanitize_config()` 规范化配置，使得 `jodie_rnn` 模型忽略图特定键，反之亦然。`canonical_config_signature()` 提供基于 JSON 的去重。
- **控制器**：`RandomGraphNASController` 均匀采样；`RLGraphNASController` 使用 REINFORCE 算法，包含每个参数的 logits、指数移动平均奖励基线，以及用于异步流水线更新的离线策略 `compute_logprob()`。
- **流水线并行**：分区按数量或 DP 优化的成本分组为阶段。每个阶段有一个 Ray 工作池（`PartitionShardWorker`）。负载（`PipelineModelPayload`）携带模型状态、运行时状态、图状态和优化器状态流经流水线。`smart` 流水线模式使用持久化工作池，支持异步提交和连续控制器更新。
- **数据并行**：`DataParallelExecutor` 将每个分区拆分为微批次，分发到 Ray worker，平均梯度，并合并运行时状态（最大时间戳胜出）。`MemShareDPExecutor` 扩展了此功能，使用热点感知状态合并（频繁节点的加权平均，冷节点的最大时间戳）。
- **自动配置**：`ConfigOptimizer` 启发式地选择阶段数量、worker 分配和分区大小，基于 GPU 数量、事件数量和估算内存。`CostModel` 使用 DP 最小化阶段间成本方差。

### `jodie/baseline/` -- 基线适配器

**角色**：将官方 JODIE 代码库作为子进程运行，并规范化输出以与 NAS 结果进行比较。

**数据流**：一个协议 JSON 文件描述实验配置。适配器定位官方仓库，复制数据集 CSV，运行 `jodie.py` 进行训练和 `evaluate_interaction_prediction.py` 进行评估，并解析文本输出以提取 MRR 和 Recall@10。

**关键设计决策**：
- 支持三种入口点模式：自定义命令模板、内置 `jodie.py` 脚本或 `official_compare_adapter.py` 适配器脚本。
- 包含 `xrange` 兼容补丁以兼容原始 JODIE 代码的 Python 2。
- 处理 GPU 可选执行：如果 CUDA 不可用，则 monkey-patch `torch.nn.Module.cuda()` 和 `torch.Tensor.cuda()` 为空操作。

### 入口点：`search.py` 和 `train.py`

**角色**：两个主要工作流的 CLI 入口点。

- `search.py`：完整的 NAS 工作流。将约 50 个 CLI 参数解析为 `SearchConfig` 数据类，选择搜索空间和控制器，分发到适当的执行模式，并保存 `best_arch.json` 和 `leaderboard.csv`。
- `train.py`：单架构训练。接受固定模型配置，加载公共数据集，用 CE loss 训练，在测试集上评估，并保存 `result.json`。

---

## 4. 逐文件文档

### 4.1 `jodie/data/` -- 数据层

---

#### `jodie/data/synthetic.py`

**用途**：定义 `Interaction` 数据类并为受控实验生成带用户偏好类型的合成交互数据。

**依赖**（导入的模块）：`dataclasses`、`typing`、`numpy`、`torch`
**被依赖**（被导入）：`jodie/data/public_dataset.py`、`jodie/data/temporal_partition.py`、`jodie/training/loops.py`、`jodie/training/metrics.py`、`jodie/training/batching.py`、`jodie/nas/trainer.py`、`jodie/nas/data_parallel.py`、`jodie/nas/ray_pipeline.py`、`jodie/__init__.py`

**类**：

| 类 | 父类 | 描述 |
|---|---|---|
| `Interaction` | `dataclass` | 表示单个用户-物品交互事件 |

**`Interaction` 字段**：

| 字段 | 类型 | 描述 |
|---|---|---|
| `timestamp` | `float` | 交互事件时间戳 |
| `user_id` | `int` | 用户标识符 |
| `item_id` | `int` | 物品标识符 |
| `features` | `torch.Tensor` | 此交互的特征向量 |

**独立函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `generate_synthetic_data` | `num_users: int`, `num_items: int`, `num_interactions: int`, `feature_dim: int`, `seed: int = 42` | `Tuple[List[Interaction], Dict[int, Set[int]], np.ndarray]` | 生成带用户偏好类型的合成交互序列。80% 的交互是偏好驱动的（用户从首选物品类型中选择），20% 是随机的。返回交互列表、用户类型偏好字典和物品类型数组。RNG 状态被保存/恢复。 |
| `init_dynamic_graph_state` | `num_users: int`, `num_items: int`, `max_neighbors: int` | `Dict` | 创建空的图状态字典，包含 `adj`、`edge_last_time`、`edge_weight` 容器。 |
| `clone_graph_state_template` | `state_template: Dict` | `Dict` | 创建与模板具有相同形状参数的新空图状态。 |
| `snapshot_graph_state` | `graph_state: Dict` | `Dict` | 将当前图状态深拷贝为可序列化的快照（将列表从 numpy/torch 类型转换）。 |
| `restore_graph_state` | `snapshot: Dict` | `Dict` | 从 `snapshot_graph_state` 创建的快照恢复图状态。 |

---

#### `jodie/data/public_dataset.py`

**用途**：加载并规范化公共 JODIE 风格数据集（Wikipedia、Reddit 或自定义 CSV）为 `Interaction` 事件列表。

**依赖**（导入的模块）：`csv`、`math`、`os`、`urllib.request`、`typing`、`torch`、`.synthetic`
**被依赖**（被导入）：`jodie/nas/trainer.py`、`train.py`

**独立函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `_resolve_dataset_path` | `dataset_name: str`, `dataset_dir: str`, `local_data_path: str` | `str` | 解析数据集的 CSV 文件路径。支持三种情况：`public_csv`（需要 `local_data_path`）、命名数据集（`wikipedia`/`reddit`——如果未缓存则从 GitHub 下载）或预设的 `local_data_path`。 |
| `_to_float` | `value: str`, `path: str`, `line_no: int`, `field_name: str` | `float` | 从 CSV 单元格解析浮点数，带错误报告。验证有限性。 |
| `_to_int` | `value: str`, `path: str`, `line_no: int`, `field_name: str` | `int` | 从 CSV 单元格解析整数，带错误报告。使用 `int(float(value))` 以提高鲁棒性。 |
| `_load_public_dataset` | `dataset_name: str`, `dataset_dir: str`, `feature_dim: int`, `max_events: int = 0`, `local_data_path: str = ""` | `Tuple[List[Interaction], int, int]` | 主加载函数。读取 CSV 行（期望列：user_id, item_id, timestamp, label, features...），将原始 ID 重新映射为连续的 0 索引 ID，将特征填充/截断到 `feature_dim`，按 (timestamp, line_no) 排序，可选地限制事件数量。返回 (interactions, num_users, num_items)。 |

**常量**：
- `_JODIE_URLS`：将 `"wikipedia"` 和 `"reddit"` 映射到其 GitHub 原始 CSV URL 的字典。

---

#### `jodie/data/temporal_partition.py`

**用途**：定义 `TemporalPartition` 和 `TemporalPartitionPlan` 数据结构，并提供从交互列表构建时序分区的函数。

**依赖**（导入的模块）：`dataclasses`、`typing`、`.synthetic`
**被依赖**（被导入）：`jodie/training/loops.py`、`jodie/training/metrics.py`、`jodie/nas/trainer.py`、`jodie/nas/data_parallel.py`、`jodie/nas/ray_pipeline.py`、`train.py`

**类**：

| 类 | 父类 | 字段 | 描述 |
|---|---|---|---|
| `TemporalPartition` | `dataclass` | `partition_id: int`, `split: str`, `start_ts: float`, `end_ts: float`, `interactions: List[Interaction]` | 属于一个划分（train/val/test）的时间有界交互块。 |
| `TemporalPartitionPlan` | `dataclass` | `partitions: List[TemporalPartition]`, `split_to_partition_ids: Dict[str, List[int]]`, `split_sizes: Dict[str, int]` | 组织所有跨划分的分区，并提供划分名称到分区 ID 的映射。 |

**`TemporalPartitionPlan` 方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `get_split_partitions` | `split: str` | `List[TemporalPartition]` | 返回属于给定划分名称的所有分区。 |

**独立函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `split_partition_interactions` | `partition: TemporalPartition`, `num_workers: int` | `List[List[Interaction]]` | 将分区的交互按时间范围（而非数量）拆分为 `num_workers` 个时序有序块，减少块之间的用户/物品重叠。 |
| `sort_interactions_by_time` | `interactions: Sequence[Interaction]` | `List[Interaction]` | 返回按 `timestamp` 排序的交互。 |
| `_build_count_partitions` | `interactions: List[Interaction]`, `partition_size: int`, `overlap_ratio: float = 0.0` | `List[List[Interaction]]` | 将交互拆分为固定大小的块，可选重叠。验证 `overlap_ratio` 在 [0, 1) 范围内。 |
| `_build_num_partitions` | `interactions: List[Interaction]`, `num_partitions: int` | `List[List[Interaction]]` | 将交互拆分为固定数量的大致相等的块。 |
| `build_temporal_partitions` | `interactions: Sequence[Interaction]`, `split: str`, `partition_size: Optional[int]`, `num_partitions: Optional[int]`, `strategy: str = "count"`, `partition_id_offset: int = 0`, `overlap_ratio: float = 0.0` | `List[TemporalPartition]` | 排序交互，按给定策略分块，将每个块包装为带元数据的 `TemporalPartition`。目前仅支持 `"count"` 策略。 |
| `build_partition_plan` | `train_interactions: Sequence[Interaction]`, `val_interactions: Sequence[Interaction]`, `test_interactions: Sequence[Interaction]`, `partition_size: Optional[int]`, `num_partitions: Optional[int]`, `strategy: str = "count"`, `overlap_ratio: float = 0.0` | `TemporalPartitionPlan` | 构建跨所有三个划分的完整分区计划。按递增分区 ID 顺序构建分区。 |

---

### 4.2 `jodie/models/` -- 模型层

---

#### `jodie/models/hybrid_jodie.py`

**用途**：定义 `TemporalEventGNNJODIE`，主要的在每次交互时执行图消息传递和状态更新的事件级时序 GNN。

**依赖**（导入的模块）：`typing`、`torch`、`torch.nn`、`.gnn_encoder`
**被依赖**（被导入）：`jodie/models/factory.py`、`jodie/__init__.py`

**类**：

| 类 | 父类 | 描述 |
|---|---|---|
| `TemporalEventGNNJODIE` | `nn.Module` | 具有动态图状态和记忆更新的事件级时序 GNN。 |
| `HybridJODIE` | (别名) | `TemporalEventGNNJODIE` 的兼容性别名。 |

**`TemporalEventGNNJODIE.__init__` 参数**：

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `num_users` | `int` | -- | 用户数量 |
| `num_items` | `int` | -- | 物品数量 |
| `embedding_dim` | `int` | -- | 嵌入维度 |
| `feature_dim` | `int` | -- | 输入特征维度 |
| `event_agg` | `str` | `"mean"` | 事件聚合方法：`"mean"`、`"sum"`、`"attn"`、`"none"` |
| `agg_activation` | `str` | `"none"` | 聚合后的激活函数：`"none"`、`"relu"`、`"tanh"`、`"gelu"` |
| `attn_type` | `str` | `"dot"` | 注意力类型：`"dot"` 或 `"mlp"` |
| `time_decay` | `str` | `"none"` | 时间衰减：`"none"`、`"exp"`、`"inverse"` |
| `max_neighbors` | `int` | `20` | 每个节点的最大邻居数 |
| `hidden_dim` | `int` | `128` | MLP 注意力的隐藏维度 |
| `memory_cell` | `str` | `"gru"` | 记忆单元类型：`"rnn"`、`"gru"`、`"lstm"`、`"add"` |
| `time_proj` | `str` | `"linear"` | 时间投影：`"linear"`、`"mlp"`、`"off"`/`"none"` |
| `memory_gate` | `str` | `"on"` | 记忆门控：`"on"` 或 `"off"` |
| `enable_event_agg` | `bool` | `True` | 启用事件级邻居聚合 |
| `enable_graph_update` | `bool` | `True` | 启用动态图结构更新 |
| `message_mode` | `str` | `"agg"` | 消息来源：`"agg"`（聚合邻居）或 `"peer"`（直接使用对方嵌入） |
| `msg_linear` | `bool` | `True` | 是否对聚合的邻居消息应用线性层 |

**`TemporalEventGNNJODIE` 方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `user_embeddings`（属性） | -- | `torch.Tensor` | `memory` 前 `num_users` 行的视图 |
| `item_embeddings`（属性） | -- | `torch.Tensor` | `memory[num_users:num_users+num_items]` 的视图 |
| `user_last_time`（属性） | -- | `torch.Tensor` | `last_time` 前 `num_users` 条目的视图 |
| `item_last_time`（属性） | -- | `torch.Tensor` | `last_time[num_users:num_users+num_items]` 的视图 |
| `reset_state` | -- | `None` | 将 memory、last_time 和 LSTM cell 状态归零 |
| `export_runtime_state` | -- | `Dict[str, torch.Tensor]` | 返回 memory、last_time 和 LSTM cell 状态的分离克隆 |
| `import_runtime_state` | `state: Dict[str, torch.Tensor]` | `None` | 将给定状态张量复制到模型的缓冲区中 |
| `_project_time` | `emb: torch.Tensor`, `delta_t: torch.Tensor` | `torch.Tensor` | 应用时间投影：`emb * (1 + proj(delta_t))` |
| `_node_ids` | `user_ids: torch.Tensor`, `item_ids: torch.Tensor` | `Tuple[torch.Tensor, torch.Tensor]` | 将用户/物品 ID 转换为节点 ID（物品节点偏移 `num_users`） |
| `_neighbors` | `graph_state: Dict`, `node_id: int` | `List[int]` | 从图状态获取节点的邻居列表 |
| `_trim_neighbors` | `graph_state: Dict`, `node_id: int` | `None` | 将邻居列表裁剪到 `max_neighbors`（保留最近的） |
| `_update_graph_state` | `graph_state: Dict`, `user_node: int`, `item_node: int`, `ts: float` | `None` | 添加双向边，更新 last_time 和 weight |
| `_memory_update` | `cell_type: str`, `update_input: torch.Tensor`, `old_state: torch.Tensor` | `torch.Tensor` | 应用一步 RNN/GRU/LSTM/add cell |
| `_apply_gate` | `old_state: torch.Tensor`, `new_state: torch.Tensor` | `torch.Tensor` | 旧状态和新状态之间的 Sigmoid 门控插值 |
| `_predict_item_embedding` | `user_state: torch.Tensor` | `torch.Tensor` | 用户状态的线性投影到预测的物品嵌入 |
| `compute_message` | `user_ids`, `item_ids`, `timestamps`, `features`, `graph_ctx` | `Tuple[torch.Tensor, torch.Tensor]` | 计算 RNN 输入向量（user_msg, item_msg）但不更新状态。对于 "peer" 模式，直接使用对方的投影嵌入。对于 "agg" 模式，通过 `event_operator.event_aggregate` 聚合邻居消息。 |
| `_apply_aggregated_message_batch` | `user_ids`, `item_ids`, `timestamps`, `features`, `user_msg`, `item_msg`, `graph_ctx` | `None` | 将预计算的消息应用于更新 memory、last_time，以及可选地更新图状态。内部用于批量更新。 |
| `apply_aggregated_message` | `node_id: int`, `aggregated_message: torch.Tensor`, `node_type: str = "user"` | `Tuple[torch.Tensor, Optional[torch.Tensor]]` | TGN 风格聚合消息更新的公共 API。返回 `(new_embedding, new_cell_state)`。 |
| `process_interaction` | `user_ids`, `item_ids`, `timestamps`, `features`, `graph_ctx=None`, `deferred=False`, `return_cell_state=False` | `Tuple[torch.Tensor, torch.Tensor, ...]` | 处理一个交互：时间投影、聚合邻居（或对方）、更新记忆。支持 TGN 批量处理的延迟更新。 |
| `predict` | `user_ids: torch.Tensor`, `query_time: float` | `Tuple[torch.Tensor, torch.Tensor]` | 使用时间投影的用户状态预测用户在查询时刻的物品嵌入。 |
| `forward` | `user_ids`, `item_ids`, `timestamps`, `features`, `query_time`, `graph_ctx=None`, `deferred=False` | `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | 完整前向传播：预测物品嵌入，处理交互，返回 `(pred_item_emb, new_user_emb, new_item_emb)`。 |

---

#### `jodie/models/jodie_rnn.py`

**用途**：定义 `JODIERNN`，一个无图操作的互递归 RNN 基线（与原始 JODIE 论文的方法一致）。

**依赖**（导入的模块）：`typing`、`torch`、`torch.nn`
**被依赖**（被导入）：`jodie/models/factory.py`、`jodie/__init__.py`

**类**：

| 类 | 父类 | 描述 |
|---|---|---|
| `JODIERNN` | `nn.Module` | JODIE 风格模型，具有互用户/物品递归更新。 |

**`JODIERNN.__init__` 参数**：

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `num_users` | `int` | -- | 用户数量 |
| `num_items` | `int` | -- | 物品数量 |
| `embedding_dim` | `int` | -- | 嵌入维度 |
| `feature_dim` | `int` | -- | 输入特征维度 |
| `cell_type` | `str` | `"rnn"` | RNN 单元类型：`"rnn"`、`"gru"`、`"lstm"` |
| `use_time_proj` | `bool` | `True` | 启用时间投影 |
| `use_static_embeddings` | `bool` | `True` | 启用静态嵌入表 |
| `normalize_state` | `bool` | `True` | 每次更新后对嵌入进行 L2 归一化 |

**`JODIERNN` 方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `reset_state` | -- | `None` | 将所有嵌入重置为学习到的初始值，将时间戳和 cell 状态归零。 |
| `export_runtime_state` | -- | `dict[str, torch.Tensor]` | 返回所有运行时缓冲区（嵌入、时间戳、LSTM 状态）的分离克隆。 |
| `import_runtime_state` | `state: dict[str, torch.Tensor]` | `None` | 将给定状态复制到模型的缓冲区中。 |
| `_normalize` | `emb: torch.Tensor` | `torch.Tensor` | 如果 `normalize_state` 为 True，则对嵌入进行 L2 归一化。 |
| `get_projected_embedding` | `node_embedding`, `delta_t`, `projection_layer` | `torch.Tensor` | 应用时间投影：`emb * (1 + proj(delta_t))`。 |
| `_delta_feature` | `delta_t: torch.Tensor` | `torch.Tensor` | 返回 `log1p(clamp(delta_t, min=0))`——对数时间间隔特征。 |
| `compute_message` | `user_ids`, `item_ids`, `timestamps`, `features`, `graph_ctx=None` | `Tuple[torch.Tensor, torch.Tensor]` | 从用户/物品嵌入、时间投影、静态嵌入、特征和时间间隔特征计算 RNN 输入向量。不分离——梯度流过。 |
| `apply_aggregated_message` | `node_id: int`, `aggregated_message: torch.Tensor`, `node_type: str = "user"` | `Tuple[torch.Tensor, Optional[torch.Tensor]]` | 使用聚合消息更新单个节点的嵌入。返回 `(new_embedding, new_cell_state)`。 |
| `process_interaction` | `user_ids`, `item_ids`, `timestamps`, `features`, `deferred=False`, `return_cell_state=False` | `Tuple[torch.Tensor, torch.Tensor, ...]` | 交互的完整状态更新。支持延迟模式和可选的 LSTM cell 状态返回。 |
| `predict` | `user_ids: torch.Tensor`, `query_time: float` | `Tuple[torch.Tensor, torch.Tensor]` | 从时间投影的用户状态（带可选静态嵌入）预测物品嵌入。 |
| `forward` | `user_ids`, `item_ids`, `timestamps`, `features`, `query_time`, `graph_ctx=None`, `deferred=False` | `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | 预测、处理交互，返回 `(pred_item_emb, new_user_emb, new_item_emb)`。 |

---

#### `jodie/models/gnn_encoder.py`

**用途**：实现 `TemporalEventGNNJODIE` 在每个交互事件中使用的核心图聚合操作符。

**依赖**（导入的模块）：`typing`、`torch`、`torch.nn`、`torch.nn.functional`
**被依赖**（被导入）：`jodie/models/hybrid_jodie.py`

**类**：

| 类 | 父类 | 描述 |
|---|---|---|
| `EventGraphOperator` | `nn.Module` | 事件级图聚合模块，支持均值、求和和基于注意力的邻居消息传递。 |

**`EventGraphOperator.__init__` 参数**：

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `embedding_dim` | `int` | -- | 节点嵌入维度 |
| `event_agg` | `str` | `"mean"` | 聚合方法：`"mean"`、`"sum"`、`"attn"`、`"none"` |
| `agg_activation` | `str` | `"none"` | 聚合后的激活函数：`"none"`、`"relu"`、`"tanh"`、`"gelu"` |
| `hidden_dim` | `int` | `None`（=embedding_dim） | 注意力 MLP 的隐藏维度 |
| `attn_type` | `str` | `"dot"` | 注意力类型：`"dot"`（点积）或 `"mlp"`（拼接 MLP） |
| `time_decay` | `str` | `"none"` | 时间衰减函数：`"none"`（无衰减）、`"exp"`（指数）、`"inverse"`（1/(1+t)） |
| `msg_linear` | `bool` | `True` | 在聚合前对邻居消息应用 `nn.Linear` |

**`EventGraphOperator` 方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `_decay_weight` | `delta_t: torch.Tensor` | `torch.Tensor` | 计算时间衰减权重：exp 用 `exp(-t)`，inverse 用 `1/(1+t)`，none 用 `ones`。 |
| `_attention_score` | `center_emb`, `neigh_emb`, `delta_t` | `torch.Tensor` | 计算注意力分数：点积或基于 MLP，结合时间衰减权重的对数。 |
| `_apply_agg_activation` | `x: torch.Tensor` | `torch.Tensor` | 应用配置的激活函数（relu/tanh/gelu/identity）。 |
| `event_aggregate` | `center_idx: int`, `center_emb: torch.Tensor`, `memory: torch.Tensor`, `neighbors: List[int]`, `edge_last_time: Dict`, `current_time: float` | `torch.Tensor` | 主聚合入口点。获取邻居嵌入，计算时间差，应用消息线性层，然后按配置的方法（mean/sum/attn）聚合，应用激活，返回聚合向量。 |

**`event_aggregate` 中的聚合方法**：
- **`sum`**：在所有邻居上 `sum(msg * decay_weight)`
- **`attn`**：`sum(msg * softmax(attention_score))`——结合学习到的注意力和时间衰减
- **默认（mean）**：`sum(msg * decay_weight) / sum(decay_weight)`——按时间加权平均

---

#### `jodie/models/factory.py`

**用途**：从配置字典创建模型实例的工厂函数。

**依赖**（导入的模块）：`typing`、`.hybrid_jodie`、`.jodie_rnn`
**被依赖**（被导入）：`jodie/nas/trainer.py`、`jodie/nas/data_parallel.py`、`jodie/nas/ray_pipeline.py`、`jodie/__init__.py`、`train.py`

**独立函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `build_model` | `config: Dict` | `nn.Module` | 基于 `config["model"]` 创建模型实例。支持 `"temporal_event_gnn_jodie"`/`"hybrid"`（创建 `TemporalEventGNNJODIE`）和 `"jodie_rnn"`（创建 `JODIERNN`）。传递 `num_users`、`num_items`、`embedding_dim`、`feature_dim` 以及模型特定的键。对未知模型类型抛出 `ValueError`。 |

---

### 4.3 `jodie/training/` -- 训练与评估

---

#### `jodie/training/batching.py`

**用途**：为 t-Batch（贪心唯一节点 ID 批处理）和 TGN 时间窗口批处理提供批次构建工具。

**依赖**（导入的模块）：`typing`、`jodie.data.synthetic`
**被依赖**（被导入）：`jodie/training/loops.py`

**独立函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `_create_t_batches` | `interactions: List`, `batch_size: int` | `List[List]` | 贪心 t-Batch 构建：按顺序迭代交互，在没有重复用户或物品 ID 且批次大小在 `batch_size` 以内时填充当前批次。当发生冲突时，关闭批次并开始新批次。保证每个批次中每个节点最多出现一次。 |
| `_create_time_windows` | `interactions: List[Interaction]`, `window_size: float` | `List[List[Interaction]]` | 将排序后的交互拆分为固定持续时间的时间窗口。当自窗口开始以来的累计时间超过 `window_size` 时开始新窗口。 |

---

#### `jodie/training/loops.py`

**用途**：为时序图模型提供所有三种粒度的训练循环：serial（逐个）、t-Batch（贪心批处理）和 TGN（时间窗口批处理）。支持 BPR 和 CE/L2 两种损失函数。

**依赖**（导入的模块）：`typing`、`numpy`、`torch`、`torch.nn`、`torch.nn.functional`、`jodie.data.synthetic`、`jodie.data.temporal_partition`、`.batching`
**被依赖**（被导入）：`jodie/training/metrics.py`、`jodie/nas/trainer.py`、`jodie/nas/data_parallel.py`、`jodie/nas/ray_pipeline.py`

**类**：

| 类 | 父类 | 描述 |
|---|---|---|
| `BPRLoss` | `nn.Module` | Bayesian Personalized Ranking 损失：`-mean(log(sigmoid(pos_score - neg_score)))`。 |

**`BPRLoss.forward`**：
- `pred_emb: torch.Tensor`——预测的物品嵌入
- `pos_emb: torch.Tensor`——真实物品嵌入
- `neg_emb: torch.Tensor`——负样本嵌入（batch x neg_samples x dim）
- 返回：`torch.Tensor`（标量损失）

**辅助函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `_model_device` | `model` | `torch.device` | 返回模型第一个参数或缓冲区的设备。 |
| `_item_embeddings_for_loss` | `model`, `item_ids: torch.Tensor` | `torch.Tensor` | 查找物品嵌入，尝试 `item_embeddings`、`item_base`、`rnn_model.item_embeddings` 或 `memory[node_id + num_users]`。 |
| `_all_item_embeddings` | `model` | `torch.Tensor` | 返回所有物品嵌入（完整嵌入表），尝试相同的属性名。 |
| `reset_model_state` | `model`, `disable_reset=False` | `None` | 除非禁用，否则调用 `model.reset_state()`。 |
| `_num_items` | `model` | `int` | 从 `model.num_items` 或 `model.rnn_model.num_items` 返回物品数量。 |
| `_normalize_partitions` | `interactions`, `partitions=None` | `List[TemporalPartition]` | 返回给定的分区或构建一个包含全部交互的单一分区。 |
| `_partition_seed` | `base_seed`, `partition_id`, `epoch` | `Optional[int]` | 推导确定性种子：`base_seed + epoch*100000 + partition_id`。 |

**训练函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `train_partition_bpr` | `model`, `partition: TemporalPartition`, `optimizer`, `criterion`, `neg_sample_size=5`, `graph_ctx=None`, `seed=None`, `progress_every=0`, `progress_callback=None` | `float` | 串行 BPR 训练：逐个交互。每个交互采样负样本，调用 `model.forward`，计算 BPR 损失，使用 `retain_graph=True` 反向传播，执行 optimizer.step。 |
| `train_partition_ce` | `model`, `partition`, `optimizer`, `graph_ctx=None`, `progress_every=0`, `progress_callback=None` | `float` | 串行 CE/L2 训练：与 BPR 结构相同但使用 `L2(pred - target)`。 |
| `train_model` | `model`, `interactions`, `num_epochs=3`, `lr=1e-3`, `neg_sample_size=5`, `graph_ctx`, `seed`, `partitions`, `batch_training`, `batch_size=32`, `batch_mode="serial"`, `tgn_loss_mode`, `tgn_window_size` | `None` | 顶层 BPR 训练器。基于 `batch_mode` 分发到 serial/tbatch/TGN 循环。每个 epoch 重置模型状态和图状态。 |
| `train_model_ce` | 与 `train_model` 相同（不含 `neg_sample_size`） | `None` | 顶层 CE/L2 训练器。与 `train_model` 相同的分发逻辑。 |
| `train_partition_bpr_batch` | `model`, `partition`, `optimizer`, `neg_sample_size=5`, `batch_size=32`, `seed=None`, `graph_ctx` | `float` | t-Batch BPR 训练。从分区交互创建 t-Batch，在批次内逐个处理交互（唯一节点 ID 保证无冲突），然后聚合损失并在每个批次后执行一次反向传播。 |
| `train_partition_ce_batch` | `model`, `partition`, `optimizer`, `batch_size=32`, `seed=None`, `graph_ctx` | `float` | t-Batch CE/L2 训练。相同的 t-Batch 结构，使用 L2 损失。 |
| `train_partition_bpr_tgn` | `model`, `partition`, `optimizer`, `criterion`, `time_window_size`, `aggregator="mean"`, `loss_mode="all"`, `neg_sample_size=5`, `seed=None`, `graph_ctx` | `float` | TGN 风格窗口批量训练（BPR）。每个窗口：(1) 通过 `model.compute_message` 收集消息，(2) 按节点聚合消息（mean/sum/last）并通过 `model.apply_aggregated_message` 更新，(3) 在所有或仅每个节点的最后一个交互上计算 BPR 损失，(4) 反向传播并更新。 |
| `train_partition_ce_tgn` | `model`, `partition`, `optimizer`, `time_window_size`, `aggregator="mean"`, `loss_mode="all"`, `seed=None`, `graph_ctx` | `float` | TGN 风格窗口批量训练（CE/L2）。与 BPR 变体结构相同。 |

---

#### `jodie/training/metrics.py`

**用途**：在分区和聚合级别提供基于排序和基于类型的 recall/MRR 评估。

**依赖**（导入的模块）：`time`、`typing`、`torch`、`jodie.data.synthetic`、`jodie.data.temporal_partition`、`.loops`
**被依赖**（被导入）：`jodie/nas/trainer.py`、`jodie/nas/data_parallel.py`、`train.py`

**函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `evaluate_partition_ranking` | `model`, `partition: TemporalPartition`, `k=10`, `graph_ctx`, `progress_label=""`, `progress_every=0`, `progress_callback`, `frozen=False` | `Dict[str, float]` | 在单个分区上评估排序指标。对每个交互：前向传播，计算与所有物品嵌入的 L2 距离，如果真实物品在前 K 中则计为命中，累计 MRR。如果 `frozen=True`，为在线评估保存和恢复模型嵌入。 |
| `evaluate_ranking_metrics` | `model`, `test_interactions`, `k=10`, `graph_ctx`, `partitions`, `frozen=False` | `Dict[str, float]` | 跨所有分区评估，返回 `{"recall_at_k": float, "mrr": float}`。 |
| `evaluate_recall_at_k` | `model`, `test_interactions`, `k=10`, `graph_ctx`, `partitions` | `float` | 仅返回 Recall@K 的便捷包装器。 |
| `evaluate_partition_type_recall` | `model`, `partition`, `item_type`, `user_type_prefs`, `k=10`, `graph_ctx`, `progress_label`, `progress_every`, `progress_callback` | `Dict[str, int]` | 基于类型的 recall：如果前 K 个物品中至少有一个物品的类型与用户的任何首选类型匹配，则计为命中。 |
| `evaluate_recall_by_type` | `model`, `test_interactions`, `item_type`, `user_type_prefs`, `k=10`, `graph_ctx`, `partitions` | `float` | 跨所有分区的基于类型的 recall，返回命中比例。 |

---

### 4.4 `jodie/nas/` -- NAS 框架

---

#### `jodie/nas/search_space.py`

**用途**：定义 NAS 搜索空间并提供配置清理和去重工具。

**依赖**（导入的模块）：`json`、`typing`
**被依赖**（被导入）：`jodie/nas/controller.py`、`jodie/nas/trainer.py`、`search.py`

**常量**：
- `TEMPORAL_MODEL_NAME = "temporal_event_gnn_jodie"`
- `PURE_JODIE_MODEL_NAME = "jodie_rnn"`

**搜索空间函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `get_small_search_space` | -- | `Dict[str, List]` | 完整的 18 参数搜索空间（包含所有超参数的所有选项）。 |
| `get_paper_compare_search_space` | -- | `Dict[str, List]` | 用于与 JODIE 论文公平比较的精简空间（受限选项）。 |
| `get_rnn_only_search_space` | -- | `Dict[str, List]` | 仅搜索 JODIERNN 超参数（4 维，适合流水线基准测试）。 |
| `get_mixed_search_space` | -- | `Dict[str, List]` | 跨两个模型家族的混合搜索，较轻的 GNN 选项（约 432 个架构）。 |
| `get_search_space` | `space_name: str` | `Dict[str, List]` | 分发器：返回 `"small"`、`"paper_compare"`、`"rnn_only"` 或 `"mixed"` 的搜索空间。 |

**清理函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `sanitize_config` | `config: Dict` | `Dict` | 规范化配置字典：对 `jodie_rnn` 去除图特定键，规范化开关值，处理条件逻辑（如 `message_mode=peer` 禁用事件聚合，`event_agg!=attn` 强制 `attn_type=dot`）。 |
| `canonical_config_signature` | `config: Dict` | `str` | 返回清理后配置的排序键 JSON 字符串，用于去重。 |

---

#### `jodie/nas/controller.py`

**用途**：提供用于采样架构的 NAS 控制器：随机搜索和基于 REINFORCE 的 RL 搜索。

**依赖**（导入的模块）：`random`、`typing`、`torch`、`.search_space`
**被依赖**（被导入）：`jodie/nas/trainer.py`、`search.py`

**类**：

| 类 | 父类 | 描述 |
|---|---|---|
| `GraphNASController` | -- | 基类，包含 `topk` 排序工具。 |
| `RandomGraphNASController` | `GraphNASController` | 随机采样：从每个搜索空间维度中均匀随机取值。 |
| `RLGraphNASController` | `GraphNASController` | REINFORCE 控制器：维护每个维度的可学习 logits，从分类分布中采样，通过策略梯度更新。 |

**`GraphNASController.topk`**：
- `results: List[Dict]`, `k: int = 3`
- 返回：`List[Dict]`——按 `(score, -params, -time_sec)` 排序

**`RandomGraphNASController.__init__`**：
- `search_space: Dict[str, List]`, `seed: int = 42`

**`RandomGraphNASController` 方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `sample_arch` | -- | `Dict` | 从搜索空间中均匀采样一个架构，应用 `sanitize_config`。 |
| `sample_arch_batch` | `batch_size: int` | `List[Dict]` | 采样 `batch_size` 个架构。 |

**`RLGraphNASController.__init__`**：
- `search_space: Dict[str, List]`, `seed: int = 42`, `lr: float = 1e-2`

**`RLGraphNASController` 方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `sample_arch` | -- | `Dict` | 从可学习策略中采样一个架构（已分离）。 |
| `sample_arch_with_logprob` | -- | `Tuple[Dict, torch.Tensor]` | 采样一个架构并返回其对数概率用于 REINFORCE。 |
| `compute_logprob` | `arch_config: Dict` | `torch.Tensor` | 离线策略：在当前策略下重新计算给定架构的对数概率（Smart 流水线用于异步更新）。 |
| `sample_arch_batch` | `batch_size: int` | `List[Dict]` | 采样一批架构（无 logprobs）。 |
| `sample_arch_batch_with_logprob` | `batch_size: int` | `List[Tuple[Dict, torch.Tensor]]` | 带 logprobs 的批量采样。 |
| `reinforce_step` | `logprob: torch.Tensor`, `reward: float` | `None` | 单样本 REINFORCE 更新。通过 EMA（alpha=0.9）更新 `reward_baseline`，计算 advantage，反向传播 `-logprob * advantage`。 |
| `reinforce_step_batch` | `samples: List[Tuple[torch.Tensor, float]]` | `None` | 批量 REINFORCE 更新：跨所有样本求和损失，然后执行一次反向传播。 |

---

#### `jodie/nas/trainer.py`

**用途**：主导 NAS 编排器（`GraphNASTrainer`），驱动架构搜索：数据准备、架构评估（串行、流水线或数据并行）、粗搜索、重新排序和最终测试评估。

**依赖**（导入的模块）：`csv`、`json`、`os`、`random`、`subprocess`、`atexit`、`threading`、`time`、`typing`、`numpy`、`torch`、`jodie.data.public_dataset`、`jodie.data.synthetic`、`jodie.data.temporal_partition`、`jodie.models.factory`、`.data_parallel`、`.ray_pipeline`、`.search_space`

**类**：

| 类 | 父类 | 描述 |
|---|---|---|
| `GraphNASTrainer` | -- | 顶层 NAS 编排器：管理数据准备、架构评估、搜索循环。 |

**常量**：
- `FINAL_RETRAIN_SEED_OFFSET = 20000`

**`GraphNASTrainer.__init__`**：
- `base_config: Dict`——与每个采样架构配置合并的基础配置字典。

**`GraphNASTrainer` 方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `_cleanup_monitor` | -- | `None` | 停止效率监控子进程并可选地生成报告。 |
| `_time_budget_reached` | `search_start_time: Optional[float]`, `time_budget_sec: float` | `bool` | 检查搜索时间预算是否已超出。 |
| `_set_seed` | `seed: int` | `None` | 设置 Python、numpy 和 PyTorch 的随机种子。 |
| `_sample_unique_arch` | `controller`, `seen_signatures: Set[str]`, `max_attempts=64` | `Tuple[Dict, Optional[torch.Tensor]]` | 采样尚未见过的架构，最多尝试 `max_attempts` 次。如果失败则回退到重复架构以保持搜索运行。 |
| `_sample_unique_arch_batch` | `controller`, `batch_size: int`, `seen_signatures: Set[str]` | `List[Tuple[Dict, Optional[torch.Tensor]]]` | 采样一批唯一架构。 |
| `_prepare_data` | -- | 7 元组 | 加载或生成数据，拆分为 train/val/test，构建分区计划，初始化图模板。 |
| `_train_and_eval` | `config`, `train_data`, `eval_data`, `user_type_prefs`, `item_type`, `graph_template`, `epochs`, `trial_seed`, `train_partitions`, `eval_partitions` | `Dict[str, float]` | 训练模型并评估。合成数据：使用 BPR + 类型 recall。公共数据：使用 CE + 排序指标。 |
| `_selection_score` | `config: Dict`, `metrics: Dict[str, float]` | `float` | 从指标中提取选择分数（合成数据用 recall@k，公共数据用可配置指标）。 |
| `_distribution_metadata` | `train_data`, `val_data`, `test_data` | `Dict` | 返回用于日志记录的数据集分布元数据。 |
| `_family_balanced_candidates` | `coarse_sorted`, `rerank_top_k`, `min_per_model` | `List[Dict]` | 通过保证每个模型家族至少有 `min_per_model` 个候选来确保重排序候选的多样性。 |
| `_evaluate_arch_multi_seed` | `arch_config`, `train_data`, `eval_data`, `user_type_prefs`, `item_type`, `graph_template`, `epochs`, `eval_seeds`, `default_seed`, `phase`, `eval_split` | `Dict` | 在多个种子上评估架构并返回平均指标。 |
| `evaluate_arch` | `arch_config`, `train_data`, `eval_data`, ... | `Dict` | 单种子架构评估：训练、评估，返回 config + metrics + params + timing。 |
| `evaluate_arch_pipeline` | `arch_configs`, `partition_plan`, `user_type_prefs`, `item_type`, `phase`, `eval_split`, `epochs`, `executor`, `time_budget_sec`, `search_start_time` | `List[Dict]` | 使用 Ray 流水线执行器评估架构。如果未提供执行器则创建一个。 |
| `_search_pipeline_async` | `controller`, `pipeline_executor`, `coarse_trials`, `architectures_per_step`, `coarse_epochs`, `seen_signatures`, ... | `List[Dict]` | Smart 异步搜索循环：维护持久化 Ray 工作池，持续提交架构，处理完成结果，通过离线策略 REINFORCE 更新控制器。预填充 2 倍架构，然后保持流水线满载。 |
| `search_pipeline` | `controller`, `coarse_trials`, `architectures_per_step`, `coarse_epochs`, `rerank_top_k=0`, `rerank_epochs=1`, `family_balanced_rerank=False`, `family_balance_per_model=1`, `time_budget_sec=0.0` | `Tuple[Dict, List[Dict]]` | 完整流水线搜索：可选第一阶段自动配置（启发式）和第二阶段（基于成本的 DP 优化），然后通过流水线粗搜索，可选重排序，最终在 train+val → test 上进行测试评估。 |
| `search` | `controller`, `coarse_trials`, `coarse_epochs`, `rerank_top_k=0`, `rerank_epochs=1`, `eval_seeds`, `family_balanced_rerank`, `family_balance_per_model`, `time_budget_sec=0.0` | `Tuple[Dict, List[Dict]]` | 串行搜索循环：逐个采样架构，通过 `_evaluate_arch_multi_seed` 评估，更新控制器（RL 步骤），用更多 epoch 重排序前 K 个候选，最终测试评估。 |
| `search_data_parallel` | `controller`, `coarse_trials`, `coarse_epochs`, `num_workers=3`, `rerank_top_k=0`, `rerank_epochs=1`, `time_budget_sec=0.0` | `Tuple[Dict, List[Dict]]` | 数据并行搜索：使用 `DataParallelExecutor` 加速每个 trial。整体搜索结构相同（采样 → 评估 → RL 更新 → 重排序 → 最终测试）。 |

---

#### `jodie/nas/ray_pipeline.py`

**用途**：通过 Ray actor 实现流水线并行。分区被分组为阶段；模型负载流经阶段（训练 → 训练 → ... → 评估），每个阶段都有工作池。

**依赖**（导入的模块）：`collections.deque`、`dataclasses`、`os`、`typing`、`time`、`torch`、`jodie.data.synthetic`、`jodie.data.temporal_partition`、`jodie.models.factory`、`.config_optimizer`、`ray`（可选）

**被依赖**（被导入）：`jodie/nas/trainer.py`

**数据类**：

| 类 | 字段 | 描述 |
|---|---|---|
| `PipelineModelPayload` | `trial_id: int`, `arch_config: Dict`, `model_state_dict: Dict[str, torch.Tensor]`, `runtime_state: Optional[Dict]`, `graph_state: Optional[Dict]`, `optimizer_state: Optional[Dict]`, `seed: int` | 携带单个架构 trial 的完整状态流经流水线阶段。 |

**类**：

| 类 | 父类 | 描述 |
|---|---|---|
| `PartitionShardWorker` | -- | Ray actor，拥有一部分分区，可以在负载上运行 train/eval 阶段。 |
| `RayPipelineExecutor` | -- | 管理流水线阶段创建、工作池生命周期和 train/eval 流水线流程。 |

**`PartitionShardWorker.__init__`**：
- `partitions: List[TemporalPartition]`, `base_config: Dict`

**`PartitionShardWorker` 方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `_build_model` | `payload: PipelineModelPayload` | `Tuple[nn.Module, Dict]` | 从配置构建模型，加载 state dict 和运行时状态。 |
| `_append_trace_line` | `line: str` | `None` | 向流水线追踪日志追加一行。 |
| `_trace_progress` | `message: str` | `None` | 写入带时间戳的进度追踪行。 |
| `run_train_stage_batch` | `payload`, `partition_ids: List[int]`, `use_bpr=True`, `num_epochs=1` | `PipelineModelPayload` | 在多个 epoch 中跨多个分区训练。支持 serial/tbatch/TGN 批量模式。返回更新后的负载。 |
| `run_eval_stage_batch` | `payload`, `partition_ids`, `item_type`, `user_type_prefs`, `k`, `synthetic_mode` | `Dict` | 在多个分区上评估。合成数据使用类型 recall，否则使用排序指标。返回 `{payload, hits, total, mrr_sum}`。 |
| `run_train_stage` | 委托到 `run_train_stage_batch`（单分区）。 |
| `run_eval_stage` | 委托到 `run_eval_stage_batch`（单分区）。 |

**`create_ray_worker`**（独立函数）：
- `partitions: List[TemporalPartition]`, `base_config: Dict`
- 返回：`PartitionShardWorker` 的 Ray 远程 actor 句柄

**`RayPipelineExecutor.__init__`**：
- `base_config: Dict`, `partition_plan: TemporalPartitionPlan`

**`RayPipelineExecutor` 关键方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `_scan_worker_progress_events` | -- | `Tuple[int, Optional[str]]` | 非阻塞扫描流水线追踪日志中的新进度事件。 |
| `_resolve_stage_worker_counts` | `key`, `num_stages`, `fallback` | `List[int]` | 解析逗号分隔的 worker 数量字符串，处理单值展开和回退。 |
| `_trace_event` | `phase`, `event`, `trial_id`, `stage_idx`, `stage_total` | `None` | 写入带时间戳的结构化流水线追踪事件。 |
| `_estimate_partition_costs` | `partitions: List[TemporalPartition]` | `List[float]` | 委托到 `CostModel.estimate_partition_costs`。 |
| `_group_partitions_by_count` | `partitions`, `num_stages` | `List[List[TemporalPartition]]` | 按数量均匀分布分区到各阶段。 |
| `_group_partitions_by_cost` | `partitions`, `num_stages` | `List[List[TemporalPartition]]` | 通过 `CostModel.optimize_partition_grouping` 使用基于 DP 优化的成本分组。 |
| `_group_partitions` | `split`, `num_stages` | `List[List[TemporalPartition]]` | 基于 `stage_balance_strategy` 路由到数量或成本分组。 |
| `_run_train_pipeline` | `payloads`, `train_groups`, `stage_workers`, `use_bpr`, `num_train_epochs` | `List[PipelineModelPayload]` | 运行训练流水线：负载流经阶段 0 → 阶段 1 → ... → 阶段 N。多 epoch：在 epoch 之间重置运行时状态。使用 Ray `ray.wait()` 实现异步完成。 |
| `_run_eval_pipeline` | `payloads`, `eval_groups`, `eval_stage_workers`, `item_type`, `user_type_prefs`, `k`, `synthetic_mode` | `Dict[int, Dict]` | 运行评估流水线：相同的阶段流程，累积每个 trial 的分数。 |
| `_run_train_eval_pipeline` | `payloads`, `train_groups`, `eval_groups`, `stage_workers`, `use_bpr`, `num_train_epochs`, ... | `Dict[int, Dict]` | 组合的 train+eval 流水线：负载流经所有训练阶段然后直接进入评估阶段，无需全局同步点。多 epoch 变体正确处理 epoch 边界。 |
| `_shutdown_worker_pool` | `stage_workers` | `None` | 终止工作池中的所有 Ray actor。 |
| `shutdown` | -- | `None` | 调用 `shutdown_persistent_pool`。 |
| `start_persistent_pool` | `eval_kwargs: Dict` | `None` | 初始化异步（smart）流水线模式的持久化 Ray 工作池。为每个阶段创建 worker，每个 worker 加载训练和评估分区。自动检测 GPU。 |
| `submit_arch` | `arch_config: Dict` | `int` | 向异步流水线提交一个架构。返回 `trial_id`。内部创建负载，推送到 train_pending[0]，排空工作池。 |
| `_drain_pool` | -- | `None` | 将待处理任务非阻塞分发到空闲 worker。评估优先于训练。 |
| `poll_completed` | `timeout=0.05` | `List[Dict]` | 轮询异步流水线中已完成的 trial。返回带有 `trial_id`、`config`、`recall_at_k`、`mrr`、`score`、`params`、`time_sec` 的结果字典列表。 |
| `shutdown_persistent_pool` | -- | `None` | 终止所有持久化池 worker。 |
| `_make_payload` | `arch_config`, `trial_id`, `seed` | `PipelineModelPayload` | 从配置创建新负载：构建模型，导出运行时状态，创建空图状态。 |
| `_print_pipeline_summary` | `train_groups`, `eval_groups`, `train_worker_counts`, `eval_worker_counts`, `payloads` | `None` | 打印带效率提示的详细流水线配置摘要。 |
| `run` | `arch_configs`, `user_type_prefs`, `item_type`, `num_train_epochs=1`, `eval_split="val"`, `time_budget_sec=0.0`, `search_start_time` | `List[Dict]` | 主流水线入口：初始化 Ray，创建阶段，自动分配 worker，使用组合的 train+eval 分区创建工作池，运行统一的 train+eval 流水线，返回结果。 |

---

#### `jodie/nas/data_parallel.py`

**用途**：使用 Ray actor 实现数据并行的 NAS 执行。支持标准数据并行（`DataParallelExecutor`）和 MemShare 热点感知状态合并（`MemShareDPExecutor`）。

**依赖**（导入的模块）：`time`、`collections.Counter`、`typing`、`ray`、`torch`、`torch.optim`、`jodie.data.synthetic`、`jodie.data.temporal_partition`、`jodie.models.factory`、`jodie.training.loops`、`jodie.training.metrics`

**被依赖**（被导入）：`jodie/nas/trainer.py`

**辅助函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `_merge_runtime_states` | `states: List[Optional[Dict]]` | `Optional[Dict]` | 合并 worker 运行时状态：对每个用户/物品，保留最新的（最大 `last_time`）。也使用相同策略合并 LSTM 状态。 |
| `_apply_averaged_gradients` | `model_state_dict`, `avg_grads`, `arch_config`, `base_config`, `runtime_state`, `optimizer_state` | `Tuple[Dict, Dict]` | 构建模型，加载状态，将平均梯度设置到 `.grad`，调用 `optimizer.step()`，返回新的 state dict 和优化器状态。 |
| `_identify_hot_nodes` | `partitions: List[TemporalPartition]`, `top_k_ratio=0.1` | `Tuple[Set[int], Set[int]]` | 按交互频率识别前 K% 的用户和物品（热节点）。 |
| `_merge_runtime_states_memshare` | `states`, `interaction_counts`, `hot_users`, `hot_items` | `Optional[Dict]` | MemShare 风格合并：热节点使用按交互计数的加权平均（平滑聚合），冷节点使用最大时间戳胜出。 |

**Ray worker 类**：

| 类 | 父类 | 描述 |
|---|---|---|
| `_DataParallelWorker` | `ray.remote` | 使用 BPR 损失训练分区的一个块，返回梯度和运行时状态。 |
| `_MemShareWorker` | `ray.remote` | 与 `_DataParallelWorker` 相同但被 `MemShareDPExecutor` 使用。 |

**`_DataParallelWorker` 方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `train_single_interaction` | `model_state_dict`, `runtime_state`, `interaction`, `arch_config`, `base_config` | `Dict` | 在单个交互上训练，返回梯度。 |
| `train_chunk` | `model_state_dict`, `runtime_state`, `interactions: List[Interaction]`, `arch_config`, `base_config` | `Dict` | 在一组交互上训练。支持 serial、tbatch 和 TGN 批量模式。返回 `{gradients, runtime_state, loss, num_interactions}`。 |

**执行器类**：

| 类 | 父类 | 描述 |
|---|---|---|
| `DataParallelExecutor` | -- | 数据并行执行器：将分区拆分为微批次，分发到 Ray worker，平均梯度，合并状态。 |
| `MemShareDPExecutor` | -- | 带 MemShare 热点感知内存共享的数据并行执行器：热节点平滑聚合，冷节点最大时间戳。 |

**`DataParallelExecutor.__init__`**：
- `base_config: Dict`, `partition_plan`, `num_workers: int = 3`

**`DataParallelExecutor` 方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `shutdown` | -- | `None` | 终止所有 Ray worker actor。 |
| `run` | `arch_configs`, `user_type_prefs=None`, `item_type=None`, `num_train_epochs=1` | `List[Dict]` | 顺序评估每个架构配置，每个 trial 内使用数据并行。 |
| `_run_trial` | `arch_config`, `trial_id`, `num_train_epochs` | `Dict` | 单 trial 执行：迭代 epoch 和分区，将每个分区拆分为分发到 Ray worker 的微批次，用 `_merge_runtime_states` 平均梯度，用 `_apply_averaged_gradients` 应用，在 CPU 上顺序评估，返回 `{trial_id, config, score, mrr, recall_at_k, time_sec}`。 |

**`MemShareDPExecutor.__init__`**：
- `base_config: Dict`, `partition_plan`, `num_workers: int = 3`, `hot_node_ratio: float = 0.1`

**`MemShareDPExecutor` 方法**：与 `DataParallelExecutor`（shutdown、run、_run_trial）相同的接口。`_run_trial` 的区别：使用 `_merge_runtime_states_memshare` 而不是 `_merge_runtime_states`，并使用 `_MemShareWorker` 实例。

---

#### `jodie/nas/config_optimizer.py`

**用途**：基于 GPU 数量、事件数量和估计工作负载，为流水线参数（阶段数量、worker 分配、分区大小）提供自动配置。

**依赖**（导入的模块）：`typing`、`math`
**被依赖**（被导入）：`jodie/nas/trainer.py`、`jodie/nas/ray_pipeline.py`

**类**：

| 类 | 父类 | 描述 |
|---|---|---|
| `CostModel` | -- | 事件成本估计和基于 DP 的分区到阶段分组。 |
| `ConfigOptimizer` | -- | 自动流水线配置的静态方法。 |

**`CostModel.__init__`**：
- `user_weight: float = 0.25`, `item_weight: float = 0.25`, `span_weight: float = 0.0`

**`CostModel` 方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `estimate_partition_costs` | `partition_info_list: List[Dict]` | `List[float]` | 估计每个分区的成本为 `num_events + user_weight*(unique_users+new_users) + item_weight*(unique_items+new_items) + span_weight*time_span`，下限 1.0。 |
| `optimize_partition_grouping` | `partition_costs: List[float]`, `num_stages: int` | `List[Tuple[int, int]]` | DP 算法，最小化与每阶段目标成本的平方偏差之和。失败时回退到均匀分组。 |
| `_uniform_grouping` | `n: int`, `num_stages: int` | `List[Tuple[int, int]]` | 将 N 个项目均匀拆分为 K 组。 |

**`ConfigOptimizer` 静态方法**：

| 方法 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `_distribute_workers` | `gpu_count`, `num_stages` | `List[int]` | **已弃用。** 跨阶段分配 GPU。 |
| `_choose_stage_count` | `gpu_count`, `num_events`, `max_stages` | `int` | 基于每个 GPU 的事件数量启发式选择阶段数。规则：<5k 事件→2 阶段，<20k→3 阶段，否则 4 阶段。确保阶段数 < GPU 数量。 |
| `_allocate_stage_workers` | `num_stages`, `gpu_count`, `stage_weights` | `List[int]` | 按阶段权重（或偏向早期阶段的衰减权重）跨阶段分配 worker。确保每个阶段至少 1 个。 |
| `parse_gpu_list` | `gpu_list_str: str` | `List[int]` | 将 "0,1,2" 解析为 `[0, 1, 2]`。 |
| `auto_allocate_config` | `gpu_count`, `num_events`, `num_partitions`, `architectures_per_step=2`, `min_workers_per_stage=1`, `max_stages=8`, `coarse_trials=6` | `Dict` | 基本启发式分配：选择阶段数，均匀分配 worker，基于事件数量选择分区大小。 |
| `auto_allocate_config_with_cost_model` | `gpu_count`, `num_events`, `partition_costs`, `architectures_per_step=2`, `coarse_trials=6`, `user_weight=0.25`, `item_weight=0.25`, `span_weight=0.0` | `Dict` | 高级分配：使用 `CostModel` 从实际分区成本计算阶段权重，然后按比例分配 worker。 |
| `_optimal_worker_allocation` | `stage_costs: List[float]`, `gpu_count: int` | `List[int]` | 拉格朗日乘子最优分配：`w_i = m * T_i / sum(T_j)`。每个阶段至少 1 个 worker。 |
| `auto_allocate_config_advanced` | `gpu_count`, `num_events`, `num_partitions`, `architectures_per_step=2`, `coarse_trials=6`, `epochs=1`, `partition_costs`, `num_users=0`, `num_items=0`, `max_embedding_dim=128`, `max_neighbors=10`, `gpu_memory_mb=0` | `Dict` | Smart 分配：基于 events/GPU 选择阶段数，通过 `_optimal_worker_allocation` 分配 worker，计算分区大小目标为 `S*5*(epochs/2)` 个分区，从 GPU 内存估算最大批量大小。 |

---

### 4.5 `jodie/baseline/` -- 基线适配器

---

#### `jodie/baseline/official_jodie.py`

**用途**：用于将官方 JODIE 代码库作为子进程运行，并将其输出规范化为结构化格式的适配器。

**依赖**（导入的模块）：`json`、`os`、`shutil`、`subprocess`、`dataclasses`、`typing`
**被依赖**（被导入）：（无直接引用，设计为独立使用）

**数据类**：

| 类 | 字段 | 描述 |
|---|---|---|
| `OfficialJodieResult` | `status: str`, `reason: str`, `mrr: Optional[float]`, `recall_at_10: Optional[float]`, `repo_path: str`, `commit: str`, `result_json_path: str` | 运行官方 JODIE 基线后的结构化结果。 |

**独立函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `_git_commit` | `repo_path: str` | `str` | 从 git 仓库获取当前 HEAD 提交哈希。 |
| `_normalize_result` | `result_json_path: str` | `Dict[str, float]` | 读取 JSON 结果文件并提取 `mrr` 和 `recall_at_10`（顶层或在 `metrics` 键下）。 |
| `_run_script_with_xrange` | `py: str`, `cwd: str`, `script_name: str`, `argv: List[str]` | `Tuple[int, str, str]` | 使用 `xrange` 兼容补丁运行 Python 脚本以兼容 Python 2，并可选 GPU monkey-patching。 |
| `_parse_interaction_results` | `result_file: str` | `Tuple[float, float]` | 解析官方 JODIE 输出文件以提取每 epoch 的验证/测试指标，选择验证 MRR 最佳的 epoch，返回 `(test_mrr, test_recall)`。 |
| `_run_builtin_official` | `repo: str`, `py: str`, `protocol: Dict`, `result_json_path: str` | `OfficialJodieResult` | 运行内置 JODIE 脚本（`jodie.py` + `evaluate_interaction_prediction.py`）。复制 CSV 数据，运行训练，运行每 epoch 评估，解析结果。 |
| `run_official_jodie_baseline` | `protocol_json_path: str`, `result_json_path: str`, `official_jodie_repo: str`, `official_python: str`, `official_cmd_template: str`, `require_official: bool` | `OfficialJodieResult` | 主入口点。依次尝试三种模式：(1) 自定义命令模板，(2) 内置脚本，(3) `official_compare_adapter.py`。返回状态为 `"ok"`、`"skipped"` 或 `"error"` 的结果。 |

---

### 4.6 入口点

---

#### `search.py`

**用途**：主 NAS 搜索 CLI 入口点。解析参数，设置搜索空间和控制器，分发到适当的执行模式，并保存结果。

**依赖**（导入的模块）：`argparse`、`csv`、`json`、`os`、`time`、`dataclasses`、`typing`、`jodie.nas.controller`、`jodie.nas.search_space`、`jodie.nas.trainer`

**数据类**：

| 类 | 字段 | 描述 |
|---|---|---|
| `SearchConfig` | 约 50 个字段（见第 6 节） | 完整的 NAS 搜索配置，包含所有参数和默认值。 |

**`SearchConfig.eval_seeds`**（属性）：将 `eval_seeds_str`（逗号分隔）解析为 `Optional[List[int]]`。

**独立函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `parse_args` | -- | `SearchConfig` | 将约 50 个 CLI 参数解析为 `SearchConfig` 数据类。 |
| `save_results` | `best: dict`, `results: list`, `output_dir: str` | `None` | 保存 `best_arch.json` 和 `leaderboard.csv` 到输出目录。CSV 包含 rank, phase, eval_split, score, val_score, test_score, mrr, recall_at_k, params, time_sec, model, config_json。 |
| `main` | -- | `None` | 主入口：解析参数，创建搜索空间 + 控制器，构建训练器，基于 `execution_mode` 分发到 `search_pipeline`、`search_data_parallel` 或 `search`，保存结果。 |

---

#### `train.py`

**用途**：单架构的独立训练入口点。加载公共数据集，使用 CE 损失训练，在测试集上评估。

**依赖**（导入的模块）：`argparse`、`json`、`time`、`pathlib`、`numpy`、`torch`、`jodie.data.public_dataset`、`jodie.data.temporal_partition`、`jodie.models.factory`、`jodie.training.loops`、`jodie.training.metrics`

**独立函数**：

| 函数 | 参数 | 返回值 | 描述 |
|---|---|---|---|
| `parse_args` | -- | `argparse.Namespace` | 解析 CLI 参数：`--model`、`--embedding-dim`、`--memory-cell`、`--time-proj`、`--use-static-embeddings`、`--normalize-state`、`--partition-size`、`--event-agg`、`--max-neighbors`、`--batch-mode`、`--train-batch-size`、`--dataset`、`--local-data-path`、`--max-events`、`--epochs`、`--seed`、`--output-dir`、`--eval-frozen`。 |
| `main` | -- | `None` | 主入口：解析参数，加载数据集（支持 public_csv），拆分 train/val/test（70/10/20），构建模型，使用 `train_model_ce` 训练，使用 `evaluate_ranking_metrics` 评估，保存 `result.json`。 |

---

## 5. 数据流

### 5.1 交互生命周期：CSV → Interaction → Partition → Model → Loss → Metric

```
CSV 文件（原始行）
    │
    ▼
public_dataset.py: load_public_dataset()
    │  - 解析 CSV 列（user_id, item_id, timestamp, label, features...）
    │  - 将原始 ID 重新映射为连续 0 索引 ID
    │  - 将特征向量填充/截断到 feature_dim
    │  - 按 (timestamp, line_no) 排序
    ▼
List[Interaction]  +  num_users  +  num_items
    │
    ▼
temporal_partition.py: build_partition_plan()
    │  - 按时间戳排序
    │  - 按比例拆分为 train/val/test
    │  - 将每个划分切块为 TemporalPartition 对象
    │  - 从第一个/最后一个交互确定 partition_start_ts / end_ts
    ▼
TemporalPartitionPlan
    ├── partitions: List[TemporalPartition]
    ├── split_to_partition_ids: Dict[str, List[int]]
    └── split_sizes: Dict[str, int]
        │
        ▼
GraphNASTrainer._train_and_eval()
    │  - 对每个 epoch：
    │    - 克隆图状态模板（重置动态图）
    │    - 重置模型状态（清零内存缓冲区）
    │    - 对每个训练分区：
    │      │
    │      ▼
    │    训练循环（serial / tbatch / TGN）：
    │      │
    │      ▼
    │    对每个交互（或批次）：
    │      │  uid=Int 张量, iid=Int 张量, t=Float 张量, f=Feature 张量
    │      ▼
    │    model.forward(uid, iid, t, f, query_time, graph_ctx)
    │      │  - _node_ids: user_node = uid, item_node = iid + num_users
    │      │  - _project_time: emb *= (1 + time_proj(delta_t))
    │      │  - event_aggregate（如果 message_mode="agg"）：
    │      │    - 从内存获取邻居嵌入
    │      │    - 从 edge_last_time 计算时间差
    │      │    - 应用 decay_weight（exp/inverse/none）
    │      │    - 如果 event_agg="attn" 则应用注意力（dot/MLP）
    │      │    - 聚合（mean/sum/attn）
    │      │  - 拼接 [proj_user, item_msg, features] 作为 RNN 输入
    │      │  - _memory_update: RNN/GRU/LSTM/add 单元步
    │      │  - _apply_gate: sigmoid(old, new) 插值
    │      │  - _predict_item_embedding: linear(user_state) → pred_item_emb
    │      │  - 将新嵌入写入 memory（如果非 deferred）
    │      │  - 更新 graph_state 的邻接/edge_last_time/edge_weight
    │      │
    │      ▼
    │    返回 (pred_item_emb, new_user_emb, new_item_emb)
    │      │
    │      ▼
    │    BPRLoss(pred, pos_emb, neg_emb)  →  标量损失
    │    loss.backward() → optimizer.step()
    │      │
    │      ▼
    │  训练后：在 val/test 数据上评估
    │    │
    │    ▼
    │  evaluate_ranking_metrics(model, eval_data, ...)
    │    │  对每个交互：
    │    │    - model.forward → pred_item_emb
    │    │    - 到所有物品嵌入的 L2 距离
    │    │    - argsort → top-K
    │    │    - 如果真实 item_id 在 top-K 中则命中
    │    │    - MRR = 每个的 1/rank
    │    ▼
    返回 {"recall_at_k": float, "mrr": float}
```

### 5.2 NAS 搜索循环

```
search.py: main()
    │
    ├── get_search_space(cfg.space)
    │     → Dict[str, List] 可搜索超参数
    │
    ├── 创建控制器：
    │     RandomGraphNASController(search_space, seed)
    │       或 RLGraphNASController(search_space, seed, lr)
    │
    ├── 创建 GraphNASTrainer(base_config)
    │
    └── 分发到执行模式：
         │
         ├── serial：
         │   trainer.search(controller, coarse_trials, coarse_epochs, ...)
         │     │
         │     ├── _prepare_data() → train/val/test 划分、分区、图模板
         │     │
         │     ├── 循环 trial=1..coarse_trials：
         │     │     ├── controller.sample_arch() → 架构配置 (Dict)
         │     │     ├── _evaluate_arch_multi_seed(arch, ...)
         │     │     │     ├── _train_and_eval(...) → 指标
         │     │     │     └── 返回 {"score", "mrr", "recall_at_k", "params", "time_sec"}
         │     │     ├── controller.reinforce_step(logprob, score)  [如果 RL]
         │     │     └── 追加到结果列表
         │     │
         │     ├── 按 (score, -params, -time_sec) 排序结果
         │     ├── [可选] 用更多 epoch 重排序前 K 个
         │     ├── 最终测试：在 train+val 上训练最佳架构，在 test 上评估
         │     └── 返回 (best_result, all_results)
         │
         ├── data_parallel：
         │   trainer.search_data_parallel(controller, ..., num_workers)
         │     │  相同结构，但每个 _evaluate_arch 调用使用
         │     │  DataParallelExecutor，在 Ray worker 之间
         │     │  分布微批次。
         │
         └── ray_pipeline：
             trainer.search_pipeline(controller, ..., architectures_per_step)
               │
               ├── [可选] 第一阶段和第二阶段自动配置
               ├── _prepare_data() → 分区
               ├── 创建 RayPipelineExecutor
               │
               ├── Naive 模式：
               │   循环批次：
               │     ├── 采样 batch_size=arch_per_step 个架构
               │     ├── evaluate_arch_pipeline(arch_batch, ...)
               │     │     └── Executor.run(payloads) → 流水线流
               │     └── controller.reinforce_step_batch(...)
               │
               ├── Smart 模式：
               │     └── _search_pipeline_async(controller, executor, ...)
               │           ├── start_persistent_pool（持久化 Ray actor）
               │           ├── 预填充 2×arch_per_step 个架构
               │           ├── 主循环：
               │           │     ├── poll_completed() → 结果
               │           │     ├── 更新控制器（离线策略）
               │           │     ├── 提交新架构以填补余量
               │           │     └── 连续运行，直到达到 coarse_trials
               │           └── shutdown_persistent_pool
               │
               ├── [可选] 通过流水线重排序前 K 个
               ├── 最终测试（串行，train+val → test）
               └── 返回 (best_result, all_results)
```

### 5.3 流水线并行流

```
流水线设置：
  1. TemporalPartitionPlan 有 N_train 个训练分区 + N_eval 个评估分区
  2. 分区按数量或 DP 最小化成本分组为 S 个阶段
  3. 每个阶段有 W_s 个 Ray PartitionShardWorker actor
     （每个加载其阶段的分区）

负载生命周期：
  ┌─────────────────────────────────────────────────────────────────┐
  │ PipelineModelPayload                                            │
  │ { trial_id, arch_config, model_state_dict, runtime_state,        │
  │   graph_state, optimizer_state, seed }                          │
  └─────────────────────────────────────────────────────────────────┘

流程（naive 批量模式）：
  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
  │架构 0│───→│阶段0 │───→│阶段1 │───→│ 评估 │───→ 分数
  └──────┘    └──────┘    └──────┘    └──────┘
  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
  │架构 1│───→│阶段0 │───→│阶段1 │───→│ 评估 │───→ 分数
  └──────┘    └──────┘    └──────┘    └──────┘

  阶段 0：在分区组 0（如最早时间范围）上训练
    ─ 负载以新模型到达 → 在 W0 worker 上训练 → 传递到阶段 1
    
  阶段 1：在分区组 1（如更晚时间范围）上训练
    ─ 接收阶段 0 更新后的模型 → 继续训练 → 传递到评估
    
  评估：在评估分区上评估
    ─ 接收训练好的模型 → 计算排序指标 → 返回分数
    
统一的 train+eval 流（无全局同步）：
  负载 → [训练 S0 → 训练 S1 → ... → 训练 S(N-1)] → [评估 S0 → 评估 S1 → ... → 评估 S(M-1)]

Smart 异步流：
  - 持久化 Ray actor 池在架构评估之间保持存活
  - 控制器持续提交架构（submit_arch）
  - 池自动排空：训练完成 → 推送到评估队列 → worker 释放用于下一个训练
  - poll_completed() 返回已完成的分数
  - 训练和评估阶段之间没有全局同步点
```

---

## 6. 配置参考

### 6.1 `SearchConfig` 字段（来自 `search.py`）

| 字段 | 类型 | 默认值 | CLI 标志 | 描述 |
|---|---|---|---|---|
| `space` | `str` | `"small"` | `--space` | 搜索空间预设：`small`、`paper_compare`、`rnn_only`、`mixed` |
| `search_mode` | `str` | `"rl"` | `--search-mode` | 控制器策略：`random` 或 `rl` |
| `execution_mode` | `str` | `"serial"` | `--execution-mode` | 执行后端：`serial`、`ray_pipeline`、`data_parallel` |
| `coarse_trials` | `int` | `0` | `--coarse-trials` | 粗搜索阶段要评估的架构数量（回退到 `trials`） |
| `coarse_epochs` | `int` | `0` | `--coarse-epochs` | 粗搜索阶段每个架构的训练 epoch 数（回退到 `epochs_per_trial`） |
| `trials` | `int` | `6` | `--trials` | 向后兼容的粗搜索 trial 数 |
| `epochs_per_trial` | `int` | `1` | `--epochs-per-trial` | 向后兼容的每个 trial 的 epoch 数 |
| `rerank_top_k` | `int` | `0` | `--rerank-top-k` | 重新排序的前 K 个架构数（0=跳过） |
| `rerank_epochs` | `int` | `0` | `--rerank-epochs` | 重排序阶段的 epoch 数（回退到 coarse_epochs） |
| `controller_lr` | `float` | `1e-2` | `--controller-lr` | RL 控制器的学习率 |
| `dataset` | `str` | `"synthetic"` | `--dataset` | 数据集：`synthetic`、`wikipedia`、`reddit`、`public_csv` |
| `dataset_dir` | `str` | `"data/public"` | `--dataset-dir` | 数据集 CSV 文件目录 |
| `local_data_path` | `str` | `""` | `--local-data-path` | 本地 CSV 文件路径（用于 `public_csv`） |
| `train_ratio` | `float` | `0.7` | `--train-ratio` | 训练数据比例 |
| `val_ratio` | `float` | `0.1` | `--val-ratio` | 验证数据比例 |
| `max_events` | `int` | `0` | `--max-events` | 最大事件数（0=全部） |
| `num_users` | `int` | `500` | `--num-users` | 用户数量（合成数据） |
| `num_items` | `int` | `1000` | `--num-items` | 物品数量（合成数据） |
| `num_interactions` | `int` | `3000` | `--num-interactions` | 交互数量（合成数据） |
| `feature_dim` | `int` | `8` | `--feature-dim` | 输入特征维度 |
| `lr` | `float` | `1e-3` | `--lr` | 训练学习率 |
| `neg_sample_size` | `int` | `5` | `--neg-sample-size` | BPR 负采样数量 |
| `k` | `int` | `10` | `--k` | Recall@K 的 K 值 |
| `selection_metric` | `str` | `"mrr"` | `--selection-metric` | 架构选择的指标：`mrr` 或 `recall_at_k` |
| `batch_training` | `bool` | `False` | `--batch-training` | 启用批量训练（旧版） |
| `train_batch_size` | `int` | `32` | `--train-batch-size` | tbatch/TGN 模式的批量大小 |
| `batch_mode` | `str` | `"tbatch"` | `--batch-mode` | 训练模式：`serial`、`tbatch`、`tgn` |
| `tgn_loss_mode` | `str` | `"all"` | `--tgn-loss-mode` | TGN 损失计算：`all` 或 `last` |
| `tgn_window_size` | `float` | `10.0` | `--tgn-window-size` | TGN 时间窗口时长 |
| `eval_frozen` | `bool` | `False` | `--eval-frozen` | 在线（false）或离线（true）评估 |
| `device` | `str` | `"auto"` | `--device` | 设备：`auto`、`cpu`、`cuda` |
| `seed` | `int` | `42` | `--seed` | 随机种子 |
| `output_dir` | `str` | `"outputs"` | `--output-dir` | 输出目录 |
| `partition_size` | `int` | `0` | `--partition-size` | 分区大小（0=自动） |
| `partition_strategy` | `str` | `"count"` | `--partition-strategy` | 分区策略（仅支持 `count`） |
| `partition_overlap_ratio` | `float` | `0.0` | `--partition-overlap-ratio` | 分区之间的重叠比例 |
| `architectures_per_step` | `int` | `2` | `--architectures-per-step` | 每个流水线批次或 RL 更新步骤的架构数 |
| `num_pipeline_stages` | `int` | `2` | `--num-pipeline-stages` | 流水线阶段数 |
| `pipeline_worker_gpus` | `float` | `0.0` | `--pipeline-worker-gpus` | 每个流水线 worker 的 GPU 数（0=自动） |
| `pipeline_worker_cpus` | `float` | `1.0` | `--pipeline-worker-cpus` | 每个流水线 worker 的 CPU 数 |
| `pipeline_stage_train_workers` | `str` | `""` | `--pipeline-stage-train-workers` | 逗号分隔的每阶段训练 worker 数 |
| `pipeline_stage_eval_workers` | `str` | `""` | `--pipeline-stage-eval-workers` | 逗号分隔的每阶段评估 worker 数 |
| `stage_balance_strategy` | `str` | `"cost"` | `--stage-balance-strategy` | 分区分组策略：`cost` 或 `count` |
| `stage_balance_user_weight` | `float` | `0.25` | `--stage-balance-user-weight` | 成本模型中用户数量的权重 |
| `stage_balance_item_weight` | `float` | `0.25` | `--stage-balance-item-weight` | 成本模型中物品数量的权重 |
| `stage_balance_span_weight` | `float` | `0.0` | `--stage-balance-span-weight` | 成本模型中时间跨度的权重 |
| `pipeline_mode` | `str` | `"naive"` | `--pipeline-mode` | 流水线模式：`naive` 或 `smart` |
| `pipeline_trace` | `bool` | `False` | `--pipeline-trace` | 启用流水线追踪 |
| `ray_address` | `str` | `""` | `--ray-address` | Ray 集群地址 |
| `gpu_list` | `str` | `"0,1,2"` | `--gpu-list` | 逗号分隔的 GPU ID，用于自动配置 |
| `enable_auto_pipeline_config` | `bool` | `False` | `--enable-auto-pipeline-config` | 启用启发式自动配置 |
| `data_parallel_workers` | `int` | `3` | `--data-parallel-workers` | 数据并行 worker 数量 |
| `data_parallel_worker_gpus` | `float` | `1.0` | `--data-parallel-worker-gpus` | 每个 DP worker 的 GPU 数 |
| `data_parallel_visible_gpus` | `str` | `"0,1,2"` | `--data-parallel-visible-gpus` | DP 可见 GPU 列表 |
| `data_parallel_sync_level` | `str` | `"micro_batch"` | `--data-parallel-sync-level` | DP 同步粒度 |
| `data_parallel_micro_batch_size` | `int` | `0` | `--data-parallel-micro-batch-size` | 微批次大小（0=自动） |
| `eval_seeds_str` | `str` | `""` | `--eval-seeds` | 逗号分隔的评估种子 |
| `family_balanced_rerank` | `bool` | `False` | `--family-balanced-rerank` | 在重排序中确保模型家族多样性 |
| `family_balance_per_model` | `int` | `1` | `--family-balance-per-model` | 重排序每个家族的最小候选数 |
| `enable_efficiency_monitor` | `bool` | `False` | `--enable-efficiency-monitor` | 启用效率监控子进程 |
| `efficiency_monitor_interval` | `int` | `10` | `--efficiency-monitor-interval` | 监控轮询间隔（秒） |
| `time_budget_sec` | `float` | `0.0` | `--time-budget-sec` | 搜索时间预算（0=无限制） |

### 6.2 `base_config` 内部键

`base_config` 字典（在 `search.py:main()` 中构建，由 `GraphNASTrainer` 消费）包含所有 `SearchConfig` 字段作为键，外加：

| 键 | 来源 | 描述 |
|---|---|---|
| `dataset` | `SearchConfig` | 数据集名称 |
| `dataset_dir` | `SearchConfig` | 数据集目录 |
| `local_data_path` | `SearchConfig` | 本地 CSV 路径 |
| `train_ratio` | `SearchConfig` | 训练划分比例 |
| `val_ratio` | `SearchConfig` | 验证划分比例 |
| `max_events` | `SearchConfig` | 事件限制 |
| `num_users` | `SearchConfig` / 自动检测 | 用户数量 |
| `num_items` | `SearchConfig` / 自动检测 | 物品数量 |
| `num_interactions` | `SearchConfig` | 合成数据交互数量 |
| `feature_dim` | `SearchConfig` | 特征维度 |
| `lr` | `SearchConfig` | 学习率 |
| `neg_sample_size` | `SearchConfig` | 负采样数 |
| `k` | `SearchConfig` | 评估 Top-K |
| `selection_metric` | `SearchConfig` | 选择指标 |
| `device` | `SearchConfig` / 自动 | 计算设备 |
| `seed` | `SearchConfig` | 随机种子 |
| `partition_size` | `SearchConfig` | 分区大小 |
| `partition_strategy` | `SearchConfig` | 分区策略 |
| `partition_overlap_ratio` | `SearchConfig` | 分区重叠比例 |
| `num_pipeline_stages` | `SearchConfig` | 流水线阶段数 |
| `pipeline_worker_gpus` | `SearchConfig` | 每个 worker 的 GPU |
| `pipeline_worker_cpus` | `SearchConfig` | 每个 worker 的 CPU |
| `pipeline_stage_train_workers` | `SearchConfig` | 每阶段训练 worker |
| `pipeline_stage_eval_workers` | `SearchConfig` | 每阶段评估 worker |
| `stage_balance_strategy` | `SearchConfig` | 阶段均衡策略 |
| `stage_balance_user_weight` | `SearchConfig` | 用户成本权重 |
| `stage_balance_item_weight` | `SearchConfig` | 物品成本权重 |
| `stage_balance_span_weight` | `SearchConfig` | 时间跨度成本权重 |
| `ray_address` | `SearchConfig` | Ray 地址 |
| `pipeline_trace` | `SearchConfig` | 启用追踪 |
| `pipeline_trace_log_path` | 在 `main()` 中生成 | 追踪日志文件路径 |
| `output_dir` | `SearchConfig` | 输出目录 |
| `enable_efficiency_monitor` | `SearchConfig` | 启用监控 |
| `efficiency_monitor_interval` | `SearchConfig` | 监控间隔 |
| `data_parallel_workers` | `SearchConfig` | DP worker 数量 |
| `data_parallel_worker_gpus` | `SearchConfig` | DP 每个 worker GPU |
| `data_parallel_visible_gpus` | `SearchConfig` | DP 可见 GPU |
| `data_parallel_sync_level` | `SearchConfig` | DP 同步级别 |
| `data_parallel_micro_batch_size` | `SearchConfig` | DP 微批次大小 |
| `gpu_list` | `SearchConfig` | GPU 列表 |
| `enable_auto_pipeline_config` | `SearchConfig` | 启用自动配置 |
| `pipeline_mode` | `SearchConfig` | 流水线模式 |
| `batch_training` | `SearchConfig` | 旧版批量标志 |
| `train_batch_size` | `SearchConfig` | 训练批次大小 |
| `batch_mode` | `SearchConfig` | 批量模式（serial/tbatch/tgn） |
| `tgn_loss_mode` | `SearchConfig` | TGN 损失模式 |
| `tgn_window_size` | `SearchConfig` | TGN 窗口大小 |
| `eval_frozen` | `SearchConfig` | 冻结评估标志 |
