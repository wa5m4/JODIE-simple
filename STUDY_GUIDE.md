# JODIE-simple-refactored 阅读指引

> **目标**：从零开始，逐步熟悉整个项目的每个模块、每个函数和关键逻辑。
> **方法**：按数据流自底向上阅读 —— 数据 → 模型 → 训练 → NAS → 执行策略 → 入口。
> **每个阶段**：列出要读的文件/函数、重点关注内容，以及自测问题。

---

## 前置知识速查

在开始之前，确保理解以下概念：

| 概念 | 一句话解释 |
|------|----------|
| **JODIE** | 时序动态图嵌入模型：用户和物品的 embedding 随时间推移通过 RNN 更新 |
| **BPR Loss** | 贝叶斯个性化排序损失：`-log(sigmoid(pos_score - neg_score))`，让正样本分数 > 负样本 |
| **t-Batch** | 贪心批处理：保证同一 batch 内没有重复的 user/item ID，避免读写冲突 |
| **TGN** | 时序图网络批处理：按时间窗口分组，先收集消息再批量更新节点状态 |
| **NAS** | 神经架构搜索：自动搜索最优超参数组合（聚合函数、RNN类型、时间衰减等） |
| **REINFORCE** | 策略梯度 RL 算法：用奖励信号更新采样策略，让高分架构更可能被采样 |
| **Ray** | 分布式计算框架：本项目用它实现数据并行和流水线并行 |
| **MRR** | 平均倒数排名：`Mean(1/rank)`，衡量预测排名的质量 |
| **Recall@K** | 前 K 个预测中包含正确答案的比例 |

---

## 第一阶段：数据结构基础（预计 30 分钟）

> 目标：理解交互数据如何表示、加载和分区。

### 1.1 阅读 `jodie/data/synthetic.py`（约 80 行）

**核心内容**：
- `Interaction` 数据类（第 10-15 行）—— 项目的"原子数据单位"
- `generate_synthetic_data()`（第 18-80 行）—— 如何生成带用户偏好的合成交互

**重点关注**：
- `Interaction` 的 4 个字段：`timestamp`, `user_id`, `item_id`, `features`
- 合成数据的生成逻辑：80% 偏好驱动 + 20% 随机
- RNG 状态的保存/恢复模式

**自测问题 1.1**：
> 1. `Interaction` 的 `features` 字段是什么类型？维度由什么决定？
> 2. 合成数据中 `user_type_prefs` 字典的 key 和 value 分别是什么？
> 3. 为什么要保存和恢复 RNG 状态？

---

### 1.2 阅读 `jodie/data/public_dataset.py`（约 110 行）

**核心内容**：
- `_load_public_dataset()`（第 66-110 行）—— CSV 加载主函数
- `_resolve_dataset_path()`（第 21-35 行）—— 数据集路径解析（含自动下载）

**重点关注**：
- CSV 期望列格式：`user_id, item_id, timestamp, label, features...`
- ID 重映射逻辑：原始 ID → 连续 0 索引
- 特征填充/截断到 `feature_dim`
- 三种数据集来源：`public_csv`（本地）、`wikipedia`/`reddit`（自动下载）

**自测问题 1.2**：
> 1. 如果 CSV 中 user_id 列的值是 `[100, 200, 300]`，重映射后变成什么？
> 2. `_to_int` 为什么用 `int(float(value))` 而不是 `int(value)`？
> 3. `max_events=0` 和 `max_events=1000` 的区别是什么？
> 4. Bug 1.1（public_csv 路径解析）的根因是什么？修复方案是什么？

---

### 1.3 阅读 `jodie/data/temporal_partition.py`（约 120 行）

**核心内容**：
- `TemporalPartition` 数据类（第 10-18 行）
- `TemporalPartitionPlan` 数据类（第 21-30 行）
- `build_temporal_partitions()`（第 65-90 行）
- `build_partition_plan()`（第 93-115 行）
- `_build_count_partitions()`（第 32-60 行）—— 包含 overlap 逻辑

**重点关注**：
- `TemporalPartition` 的 5 个字段，特别是 `split` 的含义（"train"/"val"/"test"）
- 分区策略：按 count（固定大小块）分，支持 overlap
- `partition_id` 跨 train/val/test 递增（保证全局唯一）
- overlap 的 `step` 计算公式：`step = int(partition_size * (1 - overlap_ratio))`

**自测问题 1.3**：
> 1. 如果 `partition_size=100, overlap_ratio=0.2`，相邻两个分区的起始位置差多少？
> 2. `TemporalPartitionPlan.get_split_partitions("val")` 返回什么？
> 3. `split_partition_interactions` 和 `_build_count_partitions` 的区别是什么？
> 4. Bug 1.2（无限循环风险）的触发条件是什么？如何修复的？

---

## 第二阶段：模型架构（预计 45 分钟）

> 目标：理解两个核心模型的内部结构、异同点，以及图聚合操作。

### 2.1 阅读 `jodie/models/jodie_rnn.py`（约 220 行）

**核心内容**：
- `JODIERNN.__init__()`（第 12-80 行）—— 模型结构和参数初始化
- `compute_message()`（约第 150-190 行）—— 计算 RNN 输入向量
- `process_interaction()`（约第 192-230 行）—— 完整的状态更新
- `forward()`（约第 240-260 行）—— 对外接口
- `export_runtime_state()` / `import_runtime_state()`（第 100-123 行）

**重点关注**：
- 三个关键缓冲区：`user_embeddings`, `item_embeddings`, `user_last_time`, `item_last_time`
- `cell_type` 的三种选项（rnn/gru/lstm）如何影响网络结构
- `compute_message` 的输入拼接顺序（共 6-7 个分量）
- `use_time_proj` 的时间投影公式：`emb * (1 + proj(delta_t))`
- `_delta_feature` 的计算：`log1p(clamp(delta_t, min=0))` —— 为什么用 log1p？
- 运行时状态的序列化/反序列化（GPU↔CPU 传输）

**自测问题 2.1**：
> 1. JODIERNN 和 TemporalEventGNNJODIE 最本质的区别是什么？（提示：有没有图结构？）
> 2. `reset_state()` 重置后用户 embedding 的初始值是什么？从哪里来？
> 3. `compute_message` 返回的两个消息向量各用于更新什么？
> 4. 如果 `use_static_embeddings=True`，`compute_message` 的输入会多出什么？
> 5. LSTM cell_state 只在 `cell_type=="lstm"` 时分配，这个修复节省了多少内存？

---

### 2.2 阅读 `jodie/models/gnn_encoder.py`（约 130 行）

**核心内容**：
- `EventGraphOperator.__init__()`（第 15-45 行）
- `event_aggregate()`（约第 90-130 行）—— 核心聚合逻辑
- `_decay_weight()`（第 50-58 行）
- `_attention_score()`（第 60-85 行）

**重点关注**：
- 三种聚合模式：`mean`（加权平均）、`sum`（求和）、`attn`（注意力加权）
- 三种时间衰减函数：`none`, `exp(-t)`, `1/(1+t)`
- `msg_linear` 参数：聚合前是否对邻居消息做线性变换
- `event_aggregate` 的完整计算图：邻居 embedding → time delta → decay weight → message linear → aggregate → activation

**自测问题 2.2**：
> 1. `event_agg="attn"` 时，注意力分数由哪两部分组成？
> 2. `time_decay="inverse"` 和 `time_decay="exp"` 的区别是什么？哪个衰减更快？
> 3. `event_agg="mean"` vs `event_agg="sum"` 的计算差异在哪里？（看代码）
> 4. `agg_activation` 在聚合之前还是之后应用？

---

### 2.3 阅读 `jodie/models/hybrid_jodie.py`（约 580 行，重点看结构）

**核心内容**：
- `TemporalEventGNNJODIE.__init__()`（第 30-100 行）
- `compute_message()`（约第 210-270 行）—— 包含邻居聚合调用
- `process_interaction()`（约第 290-390 行）
- `forward()`（约第 470-575 行）

**重点关注**：
- 与 JODIERNN 的核心差异：`memory` 是统一的用户+物品拼接缓冲区，用 `num_users` 做偏移
- `message_mode` 两种模式：`"agg"`（聚合邻居）vs `"peer"`（直接用对方嵌入）
- `memory_cell` 四种选项：`rnn`, `gru`, `lstm`, `add`
- `_apply_gate`（第 440-460 行）—— Sigmoid 门控：`gate*new + (1-gate)*old`
- `compute_message` 两阶段 API（为 TGN 批量模式服务）
- `enable_event_agg` 和 `enable_graph_update` 开关的作用

**自测问题 2.3**：
> 1. `user_embeddings` 属性和 `item_embeddings` 属性各对应 `memory` 的哪一段？
> 2. `message_mode="peer"` 时，聚合邻居的步骤被跳过了吗？（看 `compute_message` 代码）
> 3. `_node_ids` 方法中 item_node 为什么要加 `num_users`？
> 4. `forward()` 的 `deferred=True` 参数的作用是什么？Bug 1.3 是如何修复的？
> 5. `memory_gate="off"` 时，`_apply_gate` 还执行吗？

---

### 2.4 阅读 `jodie/models/factory.py`（约 30 行）

**核心内容**：
- `build_model()` —— 根据配置字典创建模型实例

**重点关注**：
- 如何根据 `config["model"]` 选择模型类型
- 哪些参数是两个模型共用的（`num_users`, `num_items`, `embedding_dim`, `feature_dim`）

**自测问题 2.4**：
> 1. 如果 `config["model"] = "jodie_rnn"`，`build_model` 会忽略哪些参数？为什么？
> 2. `sanitize_config` 在 sample 架构后做了什么？

---

## 第三阶段：训练与评估（预计 45 分钟）

> 目标：理解训练循环的三种粒度、BPR/CE 两种损失、评估指标的计算。

### 3.1 阅读 `jodie/training/batching.py`（约 80 行）

**核心内容**：
- `_create_t_batches()`（第 10-45 行）
- `_create_time_windows()`（第 48-80 行）

**重点关注**：
- t-Batch 的核心不变量：batch 内无重复 user/item ID
- t-Batch 的贪心构建算法：新交互来了，检查是否与当前 batch 冲突
- 时间窗口的划分标准：累计时间超过 `window_size` 就新开窗口

**自测问题 3.1**：
> 1. 如果有 10 个交互，全是同一个用户对不同的物品，t-Batch 会产生几个 batch？
> 2. `_create_time_windows` 中窗口边界是精确还是近似？为什么？
> 3. t-Batch 和 TGN 窗口的根本区别是什么？

---

### 3.2 阅读 `jodie/training/loops.py`（约 900 行，重点读关键函数）

**核心内容**（按重要程度排列）：

必读：
- `BPRLoss` 类（第 25-45 行）—— 损失函数定义
- `train_partition_bpr()`（约第 100-170 行）—— 串行 BPR 训练
- `train_partition_bpr_batch()`（约第 200-270 行）—— t-Batch BPR 训练
- `train_partition_bpr_tgn()`（约第 320-450 行）—— TGN 窗口 BPR 训练
- `train_model()`（约第 210-280 行）—— 顶层训练调度器

选读：
- `train_partition_ce()`（约第 174-200 行）—— CE/L2 训练
- `_item_embeddings_for_loss()` 和 `_all_item_embeddings()` —— 跨模型兼容辅助

**重点关注**：
- 串行 BPR：逐个交互 → neg 采样 → forward → loss → backward（`retain_graph=True`）
- t-Batch BPR：先创建 t-Batch → 逐个处理交互 → 最后统一 backward
- TGN BPR 四阶段：收集消息 → 聚合消息 → 更新节点 → 计算 loss → backward
- `reset_model_state` 在每个 epoch 开始时调用
- `_partition_seed` 的确定性种子公式：`base_seed + epoch*100000 + partition_id`

**自测问题 3.2**：
> 1. BPRLoss 的输入 `pred_emb, pos_emb, neg_emb` 各是什么维度？neg_emb 为什么是 3D？
> 2. 串行 BPR 中为什么用 `retain_graph=True`？
> 3. TGN 训练中 `compute_message` 和 `apply_aggregated_message` 为什么分开调用？
> 4. `train_model` 如何根据 `batch_mode` 分发到不同的训练循环？
> 5. `_partition_seed` 加上 `epoch*100000` 的目的是什么？

---

### 3.3 阅读 `jodie/training/metrics.py`（约 180 行）

**核心内容**：
- `evaluate_partition_ranking()`（第 30-110 行）—— 排序评估核心
- `evaluate_ranking_metrics()`（第 113-140 行）—— 跨分区汇总
- `evaluate_partition_type_recall()`（第 143-175 行）—— 合成数据类型召回

**重点关注**：
- MRR 计算：`mrr_sum += 1/(rank+1)`（如果 ground truth 在 top-K 中）
- Recall@K 计算：`hits / total`（hit = ground truth 在 top-K 中）
- `frozen=True` 时的状态保存/恢复逻辑
- 距离计算：L2 距离 `torch.norm(all_item_emb - pred_emb, p=2, dim=-1)`
- 合成数据用 `evaluate_partition_type_recall`（按用户偏好类型匹配）

**自测问题 3.3**：
> 1. 如果 ground truth item 排第 3 名，MRR 贡献是多少？如果不在 top-10 内呢？
> 2. `frozen=True` 时保存和恢复了哪些状态？
> 3. 为什么合成数据用 type recall 而不是 ranking metrics？
> 4. `_all_item_embeddings` 为什么有多个 fallback 尝试？

---

## 第四阶段：NAS 搜索框架（预计 60 分钟）

> 目标：理解搜索空间定义、控制器采样/更新策略、trainer 编排逻辑。

### 4.1 阅读 `jodie/nas/search_space.py`（约 120 行）

**核心内容**：
- `get_search_space()`（第 100-110 行）—— 四个搜索空间预设
- `get_small_search_space()` / `get_rnn_only_search_space()` / `get_mixed_search_space()`
- `sanitize_config()`（约第 50-80 行）—— 配置规范化
- `canonical_config_signature()`（第 85-90 行）—— 去重签名

**重点关注**：
- 四个搜索空间的维度差异和适用场景
- `sanitize_config` 的条件逻辑：`message_mode=peer` → 禁用 event_agg
- `jodie_rnn` 模型跳过图相关参数的清理

**自测问题 4.1**：
> 1. `rnn_only` 搜索空间只有多少维？为什么比 `small` 少？
> 2. `sanitize_config` 中 `message_mode="peer"` 会触发哪几个键的修改？
> 3. `canonical_config_signature` 和普通 `json.dumps` 有什么区别？

---

### 4.2 阅读 `jodie/nas/controller.py`（约 100 行）

**核心内容**：
- `RandomGraphNASController.sample_arch()`（第 25-35 行）—— 均匀随机采样
- `RLGraphNASController.sample_arch_with_logprob()`（约第 60-75 行）—— 策略梯度采样
- `RLGraphNASController.reinforce_step()`（约第 85-95 行）—— REINFORCE 更新
- `RLGraphNASController.compute_logprob()`（约第 100-115 行）—— 离线策略 logprob

**重点关注**：
- 控制器如何维护每个搜索维度的 logits（`nn.ParameterDict`）
- `sample_arch_with_logprob` 中 Categorical 分布的采样过程
- `reinforce_step` 的 EMA baseline：`baseline = 0.9*baseline + 0.1*reward`
- advantage 计算：`reward - baseline`
- 梯度公式：`-(logprob * advantage).backward()`

**自测问题 4.2**：
> 1. 随机控制器和 RL 控制器的 `sample_arch` 有什么本质区别？
> 2. EMA baseline 的参数 0.9 是什么意思？降低到 0.5 会怎样？
> 3. `compute_logprob` 和 `sample_arch_with_logprob` 返回的 logprob 有什么不同？（为什么 Bug B 发生了）
> 4. RL 控制器的 logits 初始值是什么？对初始采样有影响吗？

---

### 4.3 阅读 `jodie/nas/trainer.py`（约 1320 行，分段读）

**这是项目的核心文件，需要仔细阅读。**

#### 第一部分：辅助方法（第 1-310 行）

- `_prepare_data()`（第 114-175 行）—— 数据加载和分区
- `_train_and_eval()`（第 177-249 行）—— 单架构训练+评估
- `_selection_score()`（第 251-257 行）—— 提取选择分数
- `_sample_unique_arch()`（第 71-100 行）—— 去重采样
- `_family_balanced_candidates()`（第 275-306 行）—— 均衡候选选择
- `_evaluate_arch_multi_seed()`（第 307-365 行）—— 多种子评估
- `evaluate_arch()`（第 367-417 行）—— 单种子架构评估

**自测问题 4.3a**：
> 1. `_prepare_data` 中 train/val/test 的划分比例是如何保证 `train_ratio + val_ratio < 1` 的？
> 2. `_train_and_eval` 中 synthetic 和 public 数据集走的分支有何不同？
> 3. `evaluate_arch` 中的 `score` 是什么？和 `mrr`/`recall_at_k` 的关系？

#### 第二部分：Serial 搜索（第 957-1120 行）—— `search()`

- 主循环结构：采样 → 多种子评估 → RL 更新 → 下一个
- 重排序逻辑
- Final test 逻辑（train+val 训练，test 评估）

**自测问题 4.3b**：
> 1. Serial 搜索中架构的 trial_seed 是如何计算的？
> 2. 重排序时候选架构从哪里来？
> 3. Final test 的 `eval_split` 是什么？`epochs` 用哪个值？

#### 第三部分：Data Parallel 搜索（第 1122-1318 行）—— `search_data_parallel()`

- 与 serial 的结构对比
- DataParallelExecutor 的使用方式
- Final test 的 epochs 修复前的问题

**自测问题 4.3c**：
> 1. Data Parallel 搜索中每个 trial 的 `score` 从哪里来？
> 2. 为什么 DP 的重排序阶段创建了新的 executor？
> 3. DP 的 final test 之前用的 epochs 是什么（Bug C）？修复后改成了什么？

#### 第四部分：Pipeline 搜索（第 623-955 行）—— `search_pipeline()` 和 `_search_pipeline_async()`

- Naive vs Smart 两种模式的调度
- 自动配置的阶段 1 和阶段 2
- Smart 模式的异步循环：预填充 → poll → RL 更新 → 补充提交
- `evaluate_arch_pipeline()`（第 419-470 行）

**自测问题 4.3d**：
> 1. `evaluate_arch_pipeline` 中 `time_sec` 的计算为什么有问题？
> 2. Smart 模式的预填充（prefill）策略是什么？为什么要 2×arch_per_step？
> 3. Smart 模式中 `remaining` 变量有什么问题？
> 4. Naive 和 Smart 模式在 RL 更新方式上有什么不同？
> 5. `_search_pipeline_async` 的 `pending_logprobs` 用途是什么？Bug B 修复前后有何变化？

---

## 第五阶段：并行执行后端（预计 60 分钟）

> 目标：理解 Ray 如何实现数据并行和流水线并行。

### 5.1 阅读 `jodie/nas/data_parallel.py`（约 820 行，重点看核心逻辑）

**核心内容**：
- `_merge_runtime_states()`（第 40-66 行）—— 状态合并策略
- `_apply_averaged_gradients()`（第 68-105 行）—— 梯度平均+应用
- `_DataParallelWorker.train_chunk()`（约第 200-385 行）—— Worker 训练逻辑
- `DataParallelExecutor._run_trial()`（第 446-561 行）—— 单 trial 执行

**重点关注**：
- 微批次并行：每个 partition 被拆分为 `batch_size * num_workers` 的微批次
- `_merge_runtime_states` 的 max-timestamp-wins 策略
- `_apply_averaged_gradients` 中 `.grad` 的手动设置
- 评估在 CPU 上串行执行（`eval_device = torch.device("cpu")`）

**自测问题 5.1**：
> 1. 如果有 3 个 worker，微批次大小是 32，每个 partition 有多少交互同时被处理？
> 2. `_merge_runtime_states` 如何处理 LSTM cell state？
> 3. DP worker 的 `train_chunk` 支持哪三种 batch_mode？代码路径有何不同？
> 4. `MemShareDPExecutor` 和 `DataParallelExecutor` 的主要区别是什么？

---

### 5.2 阅读 `jodie/nas/ray_pipeline.py`（约 1340 行，分阶段读）

#### 第一部分：基础结构（第 1-310 行）

- `PipelineModelPayload` 数据类（第 38-45 行）
- `PartitionShardWorker.__init__()` 和 `_build_model()`
- `run_train_stage_batch()`（第 81-195 行）—— Worker 训练
- `run_eval_stage_batch()`（第 197-283 行）—— Worker 评估

**自测问题 5.2a**：
> 1. `PipelineModelPayload` 携带了哪些状态？为什么需要 optimizer_state？
> 2. `run_train_stage_batch` 中 epoch 循环现在的 model.reset_state 行为是怎样的？
> 3. `run_eval_stage_batch` 中 synthetic 和 public 数据评估函数有何不同？

#### 第二部分：Naive Pipeline 执行（第 323-752 行）

- `_run_train_pipeline()`（第 522-643 行）—— 多阶段训练流水线
- `_run_eval_pipeline()`（第 645-751 行）—— 多阶段评估流水线
- `_run_train_eval_pipeline()`（第 753-932 行）—— 统一训练+评估
- `run()`（第 1181-1337 行）—— Naive 模式主入口

**重点关注**：
- 三个 pipeline 方法的心跳循环结构（while True → dispatch → ray.wait → process）
- `_run_train_eval_pipeline` 中 train 完成 → 立即进 eval stage 0 的逻辑
- 多 epoch 时 epoch 间 runtime_state 重置机制
- `run()` 的 worker 创建策略：每个 stage 的 worker 同时加载 train+eval partitions

**自测问题 5.2b**：
> 1. `_run_train_pipeline` 和 `_run_eval_pipeline` 的心跳循环有哪些相同和不同？
> 2. `_run_train_eval_pipeline` 时间预算超出时发生了什么？
> 3. `run()` 中 score 是如何从 `eval_scores` 计算出来的？Bug D 修复前后有什么变化？

#### 第三部分：Smart Pipeline 执行（第 947-1113 行）

- `start_persistent_pool()`（第 947-989 行）—— 创建持久化 worker 池
- `submit_arch()`（第 991-1004 行）—— 提交架构到池
- `_drain_pool()`（第 1006-1044 行）—— 非阻塞任务调度
- `poll_completed()`（第 1060-1113 行）—— 轮询完成结果

**重点关注**：
- 持久化池 vs 临时池的区别（worker 不释放，复用）
- `_drain_pool` 中 eval 优先于 train 的调度策略
- `poll_completed` 中 train 完成 → 下一 stage 或 eval 队列的路由逻辑
- Bug A 修复前后 `_drain_pool` 的 `num_epochs` 变化

**自测问题 5.2c**：
> 1. `start_persistent_pool` 中每个 worker 加载了哪些 partitions？
> 2. Smart 模式如何保证 eval 优先执行？
> 3. `poll_completed` 中 trial 的 `time_sec` 是怎么计算的？与 naive 模式有何不同？
> 4. Bug A 的根因是什么？影响有多大？

---

### 5.3 阅读 `jodie/nas/config_optimizer.py`（约 400 行）

**核心内容**：
- `CostModel.estimate_partition_costs()`（第 23-44 行）
- `CostModel.optimize_partition_grouping()`（第 46-112 行）—— DP 算法
- `ConfigOptimizer._optimal_worker_allocation()`（约第 320 行）
- `ConfigOptimizer.auto_allocate_config_advanced()`（约第 342 行）
- `_aggregate_stage_costs()`（新增）

**重点关注**：
- 分区成本公式的 5 个分量（events, users, items, new_users, new_items, time_span）
- DP 算法最小化阶段间成本方差
- 拉格朗日乘子最优 worker 分配公式：`w_i ∝ T_i`
- 修复后的 `auto_allocate_config_advanced` 如何利用 `partition_costs`

**自测问题 5.3**：
> 1. 成本模型中将 `span_weight` 默认设为 0 意味着什么？
> 2. DP 算法的目标函数是什么？
> 3. `_optimal_worker_allocation` 如何保证每个 stage 至少 1 个 worker？

---

## 第六阶段：入口和整体流程（预计 30 分钟）

> 目标：理解 CLI 参数、配置传递、结果保存，以及四种策略的完整生命周期。

### 6.1 阅读 `search.py`（约 486 行）

**核心内容**：
- `SearchConfig` 数据类（第 36-135 行）—— 所有 CLI 参数的集中定义
- `parse_args()`（第 142-303 行）—— argparse 定义和映射
- `save_results()`（第 310-348 行）—— 结果保存
- `main()`（第 354-485 行）—— 主流程

**自测问题 6.1**：
> 1. `SearchConfig.eval_seeds` 属性的作用是什么？
> 2. `coarse_trials` 和 `trials` 的关系是什么？
> 3. 设备 `"auto"` 的解析逻辑在哪里？

---

### 6.2 阅读 `run_all.py`（约 500 行）

**核心内容**：
- 文件头顶部的参数文档和配置区
- `build_base_config()` —— 四个策略共享的配置构建
- `run_serial()`, `run_data_parallel()`, `run_pipeline_naive()`, `run_pipeline_smart()`
- `generate_comparison()` —— 对比报告生成
- `main()` —— 依次执行所有策略

**自测问题 6.2**：
> 1. 四个策略的 final test 是如何保证分数可比的？
> 2. `generate_comparison` 生成的 comparison.md 包含哪些对比维度？
> 3. 如果某个策略执行失败，`main()` 如何处理？

---

## 第七阶段：数据流全景追踪（预计 30 分钟）

> 目标：能够从"一个 CSV 文件"出发，追踪数据在四个策略中的完整流动路径。

### 7.1 追踪 Serial 模式的一条完整路径

从 `run_all.py` → `trainer.search()` → `_prepare_data()` → `build_model()` → `train_model()` → `train_partition_bpr()` → `BPRLoss.forward()` → `evaluate_partition_ranking()` → `_selection_score()` → `_evaluate_arch_multi_seed()`（final test）

### 7.2 追踪 Pipeline Smart 模式的一条完整路径

从 `run_all.py` → `trainer.search_pipeline()` → `_search_pipeline_async()` → `start_persistent_pool()` → `submit_arch()` → `_drain_pool()` → `PartitionShardWorker.run_train_stage_batch()` → `poll_completed()` → `controller.reinforce_step()` → final test via `_evaluate_arch_multi_seed()`

### 7.3 对比四种策略的关键差异

| 维度 | Serial | Data Parallel | Pipeline Naive | Pipeline Smart |
|------|--------|---------------|----------------|----------------|
| 架构并行度 | 1 | 1 | `arch_per_step` | `total_workers` |
| 训练并行度 | 1 | `num_workers` (微批次) | `total_workers` (阶段内) | `total_workers` (阶段内) |
| Worker 生命周期 | 无 | 单 trial | 单 batch | 持久化池 |
| RL 更新时机 | 每 trial | 每 trial | 每 batch | 累积 `arch_per_step` |
| 状态传递 | 无（每次独立） | 微批次间合并 | Payload 阶段间传递 | Payload 阶段间传递 |
| Final test | Serial | Serial | Serial | Serial |

---

## 自测答案参考

<details>
<summary>点击展开各阶段自测答案</summary>

### 第一阶段
1.1.1: `torch.Tensor`，由 `feature_dim` 决定
1.1.2: key=user_id, value=该用户偏好的 item_type 集合
1.1.3: 避免修改全局 numpy 随机状态影响其他代码
1.2.1: `[0, 1, 2]`（按出现顺序映射）
1.2.2: CSV 中可能 `"3.0"` 这种浮点格式的整数
1.2.3: 0=加载全部事件，1000=只取前 1000 条
1.2.4: `_resolve_dataset_path("public_csv")` 不在 URL 字典中，直接 raise
1.3.1: `step = int(100*(1-0.2)) = 80`，起始差 80
1.3.2: 所有 split 为 "val" 的 TemporalPartition 列表
1.3.3: 前者按时间范围拆分（减少重叠），后者按固定数量拆分
1.3.4: `overlap_ratio >= 1.0 → step=0 → 无限循环`

### 第二阶段
2.1.1: JODIERNN 无图操作，用对方投影嵌入；HybridJODIE 有图邻居聚合
2.1.2: 从 `self.user_init`/`self.item_init`（nn.Parameter，初始为 0）
2.1.3: user_msg 用于更新用户嵌入，item_msg 用于更新物品嵌入
2.1.4: 用户和物品的静态嵌入（`user_static(uid)`, `item_static(iid)`）
2.1.5: `2 * num_nodes * embedding_dim * 4 bytes`，对 20K 节点 dim=128 约 20MB
2.2.1: 学习到的注意力分数 + log(时间衰减权重)
2.2.2: exp 衰减为 `e^(-t)`，inverse 为 `1/(1+t)`，exp 衰减更快
2.2.3: mean 除以 sum(decay_weight)，sum 直接求和
2.2.4: 聚合之后
2.3.1: `memory[:num_users]` 和 `memory[num_users:num_users+num_items]`
2.3.2: 是的，直接使用 peer 的 projected embedding 而不聚合邻居
2.3.3: 因为 memory 是 user+item 拼接的，item 节点索引需要偏移
2.3.4: 原 bug 是 deferred=True 时仍写入 memory；修复加了 `if not deferred:` 守卫
2.3.5: 不执行，`_apply_gate` 直接返回 new_state
2.4.1: event_agg, attn_type, max_neighbors, enable_event_agg 等图相关参数
2.4.2: 确保 jodie_rnn 配置不含图键，hybrid 配置不含无关项

### 第三阶段
3.1.1: 每个交互一个 batch（因为 user_id 相同会冲突），共 10 个 batch
3.1.2: 近似——按累计时间，不是固定时间点
3.1.3: t-Batch 保证无重复 ID（数学等价于串行），TGN 按时序窗口（允许重复）
3.2.1: pred_emb [1,dim], pos_emb [1,dim], neg_emb [1,N,dim]（N=neg_sample_size）
3.2.2: 因为同一 batch 中多个交互共享计算图，需要保留
3.2.3: compute_message 不更新状态（可批量计算），apply 才更新（需逐个节点）
3.2.4: batch_mode 参数：serial→串行，tbatch→t-Batch，tgn→TGN 窗口
3.2.5: 确保不同 epoch 不同 partition 的种子完全不同
3.3.1: MRR 贡献=1/3≈0.333；不在 top-10=0
3.3.2: user/item embeddings 和 LSTM cell state
3.3.3: 合成数据有用户偏好类型标签，可以评估类型级别的预测准确性
3.3.4: 因为两个模型的嵌入存储方式不同（统一 memory vs 分离 buffer）

### 第四阶段
4.1.1: rnn_only 仅 4-6 维（模型类型、cell_type、time_proj、embedding_dim 等），不含图参数
4.1.2: enable_event_agg=False, event_agg="none"
4.1.3: 先用 sanitize_config 规范化，再对排序后的键做 JSON 序列化
4.2.1: 随机控制器的采样概率固定（均匀），RL 控制器根据学习到的 logits 采样
4.2.2: 控制 baseline 对历史 reward 的记忆长度；0.5 会让 baseline 更快响应近期变化
4.2.3: sample 返回采样时的 logprob（正确），compute 返回当前策略的 logprob（用于离线策略，有偏）
4.2.4: 初始化为均匀分布（logits=0），所以初始采样也是均匀的
4.3a.1: `train_ratio + val_ratio >= 1` 直接 raise ValueError
4.3a.2: synthetic→BPR+type_recall；public→CE+ranking_metrics
4.3a.3: score 是架构选择的排序依据，synthetic=recall，public=selection_metric（默认 mrr）
4.3b.1: `seed + trial_idx`（base_config 中的 seed 全局种子）
4.3b.2: 从 coarse_sorted 中取前 K 个（或 family_balanced 选择）
4.3b.3: eval_split="test", epochs=rerank_epochs（如有 rerank）否则 coarse_epochs
4.3c.1: 从 DataParallelExecutor 的 `_run_trial` 返回的 metrics[selection_metric]
4.3c.2: 因为前面的 executor 已经 shutdown，需要新 worker 做重排序
4.3c.3: 之前是 coarse_epochs，修复后是 `rerank_epochs if rerank_top_k > 0 else coarse_epochs`
4.3d.1: 总时间除以结果数得到平均值，而非每个架构的实际执行时间
4.3d.2: 确保 pipeline 始终有足够架构在流水线中流动，避免 worker 空闲
4.3d.3: `remaining` 始终等于 `coarse_trials` 不递减，名字误导；实际用 `total_submitted < remaining` 控制
4.3d.4: Naive 每 batch 做一次 batch RL 更新；Smart 累积 arch_per_step 个结果后做 offline RL 更新
4.3d.5: 存储采样时的 logprob 用于 RL 更新；Bug B 修复前被丢弃并用 compute_logprob 重新计算

### 第五阶段
5.1.1: 3 个 worker × 32 batch_size = 96 个交互同时被处理
5.1.2: 按 max-timestamp-wins 选择最"新"的 cell state
5.1.3: serial/tbatch/tgn，通过 if-elif 分支调用不同的训练函数
5.1.4: MemShare 用热点感知合并（热节点加权平均+冷节点 max-timestamp）
5.2a.1: model_state_dict, runtime_state, graph_state, optimizer_state 四种状态
5.2a.2: epoch=0 不 reset，epoch>0 调用 model.reset_state()，匹配 serial 行为
5.2a.3: synthetic→evaluate_partition_type_recall，public→evaluate_partition_ranking
5.2b.1: 结构相同（dispatch→wait→heartbeat→process），train 完成后进下一 stage，eval 完成后累加分数
5.2b.2: 清除所有 train_pending，取消在途 train 任务，eval 仍继续执行
5.2b.3: `hits/total` 或 `mrr_sum/total`；修复后还考虑了 selection_metric 配置
5.2c.1: 该 stage 的 train partitions + 所有 eval partitions（便于任意 worker 做 eval）
5.2c.2: `_drain_pool` 中先遍历 eval_pending 再处理 train_pending
5.2c.3: `time.time() - submit_time`（每个 trial 的独立耗时）；naive 模式用 `elapsed / n`
5.2c.4: `num_epochs=1` 硬编码；所有架构只训 1 epoch，导致分数完全不可比
5.3.1: 时间跨度不影响分区成本估计（时间权重为 0）
5.3.2: 最小化 sum(每个 stage 的实际成本 - 目标平均成本)²
5.3.3: `max(1, round(m * T_i / sum(T_j)))` 保证至少为 1

</details>

---

## 推荐的阅读顺序总览

```
第一遍（快速通读，建立全局认知）：
  jodie/data/synthetic.py         → 理解 Interaction
  jodie/models/jodie_rnn.py        → 理解简单模型
  jodie/nas/search_space.py        → 理解搜索空间
  jodie/nas/trainer.py             → 理解 search() 主循环
  search.py                        → 理解 CLI 入口

第二遍（深入理解，逐个模块击破）：
  jodie/data/temporal_partition.py → 分区机制
  jodie/models/hybrid_jodie.py     → GNN 模型细节
  jodie/training/loops.py          → 三种训练循环
  jodie/training/metrics.py        → 评估指标
  jodie/nas/controller.py          → RL 采样逻辑
  jodie/nas/trainer.py (全文)      → 三种搜索策略

第三遍（并行后端和性能优化）：
  jodie/nas/data_parallel.py       → 微批次梯度同步
  jodie/nas/ray_pipeline.py        → 阶段流水线+持久化池
  jodie/nas/config_optimizer.py    → 自动配置和成本模型
  run_all.py                       → 四策略对比执行

第四遍（收尾和验证）：
  ARCHITECTURE.md                  → 验证理解是否与文档一致
  ISSUES.md                        → 理解已修复的 12 个 bug
  jodie/baseline/official_jodie.py → 了解基线对比方式（可选）
```

---

## 学习进度检查表

- [ ] 第一阶段：能说出 Interaction 包含哪些字段，分区是怎么构建的
- [ ] 第二阶段：能画出 JODIERNN 和 HybridJODIE 的前向传播计算图
- [ ] 第三阶段：能解释 BPR Loss 的三个输入、t-Batch 的不变量、TGN 四阶段
- [ ] 第四阶段：能解释 REINFORCE 的梯度流向、serial 搜索的主循环
- [ ] 第五阶段：能画出 Pipeline 的 stage 间 payload 流动图
- [ ] 第六阶段：能修改 run_all.py 的配置跑一次完整对比
- [ ] 第七阶段：能从 CSV 出发，描述数据在四种策略中的完整路径
