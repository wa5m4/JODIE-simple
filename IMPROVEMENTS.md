# JODIE-simple-refactored 待改进清单

> **创建日期**：2026-07-20
> **目的**：记录已讨论但尚未实施的改进思路，按优先级和可行性排序。

---

## 一、Pipeline 存在性论证（论文核心）

### 1.1 核心问题

**现状**：在当前的默认配置下（20000 事件、rnn_only 小模型），单 stage 架构并行的速度 ≈ 甚至快于多 stage 流水线。如果 pipeline 没有加速，论文的核心贡献会被动摇。

**根本原因分析**：

| 因素 | 详细说明 |
|------|----------|
| 数据量小 | 20000 事件 → 每 stage 少数几个分区，stage 间 payload 传递的开销占比太高 |
| 模型过小 | rnn_only 只有 embedding + RNN，GPU 远未饱和 |
| 图状态小 | 用户/物品数少，图状态（邻居列表）内存微不足道 |
| 无内存压力 | 单 GPU 轻松装下全部数据 + 模型 + 状态 |

### 1.2 改进方案

#### 方案 A：大规模实验（让单 GPU 不够用）

通过数据/模型规模让单 GPU 真的 OOM：

```python
# 建议配置
MAX_EVENTS = 200000           # 10x 当前
SEARCH_SPACE = "small"        # 18 维大模型
max_neighbors = 50            # 图邻居列表增大
embedding_dim 最大 256        # 嵌入维度增大
```

#### 方案 B：显式内存约束（模拟"硬件受限"环境）

**思路**：Ray worker 在创建时接受 `memory` 参数限制可用内存。在此基础上，图状态需要跨多 GPU 分片存储，pipeline 成为唯一可行方案。

**实现位置**：`jodie/nas/ray_pipeline.py` `create_ray_worker()` 和 `jodie/nas/config_optimizer.py`

```python
# 新增参数
PIPELINE_WORKER_MEMORY_MB = 4000   # 每个 worker 最多 4GB 显存

# 在 create_ray_worker 中
ray.remote(
    num_gpus=worker_gpus,
    memory=PIPELINE_WORKER_MEMORY_MB * 1024 * 1024  # Ray 内存限制
)(PartitionShardWorker)

# ConfigOptimizer 新增方法
def auto_allocate_stages_by_memory(
    gpu_count, num_users, num_items,
    max_neighbors, embedding_dim, memory_per_worker_mb
) -> int:
    """基于图状态内存需求自动计算最小 stage 数量"""
    graph_state_per_event = (num_users + num_items) * max_neighbors * 16
    model_memory = ...
    max_stages_with_data = ...
    return min_stages_needed
```

#### 方案 C：论文叙事 — 不靠 OOM，靠"资源受限"论证

**核心叙事**：

> 在边缘计算 / 多租户 GPU 集群 / 资源受限环境下，
> 单 GPU 的内存配额有限（如 2-4 GB）。
> 完整图状态超出配额时，单 stage 数据并行会 OOM。
> 流水线并行通过时间分片，每个 stage 只维护 1/N 时间范围的图状态，
> 将内存压力线性分散到 N 个 GPU，是唯一可行的方案。

**需要的实验对比**：

| 实验条件 | Serial | Data Parallel | Pipeline |
|----------|--------|---------------|----------|
| 资源充足（无限制） | 基线 | 速度提升 | 速度略低（开销大） |
| 内存受限（2GB/worker） | 可行 | **OOM 不可行** | ✅ 唯一可行 |
| 大数据+大模型 | 极慢 | **OOM 不可行** | ✅ 唯一可行 |

---

## 二、自动配置逻辑修复

### 2.1 Smart 模式的 stage/work 检测逻辑

**文件**：`jodie/nas/trainer.py` `_search_pipeline_async()` 第 501-503 行

**问题**：
```python
user_stages = self.base_config.get("num_pipeline_stages")   # = 3
user_workers = str(self.base_config.get("pipeline_stage_train_workers", "")).strip()  # = ""
if user_stages and user_workers:  # 3 and "" = False → 走自动配置
```

用户设了 `stages=3` 但 `workers=""` → 自动配置覆盖 stages 为 1。

**建议修复**：如果用户设了 stages，就按手动模式；workers 为空时使用 `_optimal_worker_allocation` 自动分配。

```python
if user_stages:
    # 用户设了 stages → 手动模式
    # workers 为空则自动均匀分配
    if not user_workers:
        worker_list = [1] * user_stages  # 每 stage 至少 1 个
        self.base_config["pipeline_stage_train_workers"] = ",".join(str(w) for w in worker_list)
else:
    # 完全自动配置
    auto_config = ...
```

### 2.2 自动配置的阈值调整

**文件**：`jodie/nas/config_optimizer.py` `auto_allocate_config_advanced()` 第 384-392 行

**问题**：`events_per_gpu < 10000 → S=1` 过于保守。20000 事件 / 3 GPU = 6667/GPU 就退化为单 stage。

**建议**：
```python
# 当前 (过于保守)
if m <= 1 or events_per_gpu < 10000:
    S = 1

# 建议 (更积极的流水线)
if m <= 1:
    S = 1
elif events_per_gpu < 2000:
    S = 1
elif events_per_gpu < 10000:
    S = min(2, m)    # 至少 2 stage
elif events_per_gpu < 50000:
    S = min(3, m)
```

---

## 三、剩余架构问题（来自 ISSUES.md）

### 3.1 心跳循环重复（架构问题 4.2）

**文件**：`jodie/nas/ray_pipeline.py`
- `_run_train_pipeline`（~120 行）
- `_run_eval_pipeline`（~106 行）
- `_run_train_eval_pipeline`（~179 行）

**问题**：三个方法 ~80% 结构相同，~400 行重复代码。

**建议重构方案**：
```python
def _run_pipeline_core(
    self,
    payloads, groups, stage_workers,
    stage_fn,           # 每个 stage 的调用函数
    result_handler,     # 完成后的结果处理
    phase_label: str,
) -> Any:
    """通用流水线核心循环：dispatch → ray.wait → heartbeat → process"""
    # 统一的心跳/分发/处理逻辑
```

### 3.2 MemShareDPExecutor 模式支持（架构问题 4.3）

**文件**：`jodie/nas/data_parallel.py` `MemShareDPExecutor` 类

**问题**：只支持 serial 模式，不支持 tbatch/tgn。

| 功能 | DataParallelExecutor | MemShareDPExecutor |
|------|---------------------|-------------------|
| Serial | ✅ | ✅ |
| t-Batch | ✅ | ❌ |
| TGN | ✅ | ❌ |

### 3.3 TemporalEventGNNJODIE batch_size=1 限制（代码坏味 3.8）

**文件**：`jodie/models/hybrid_jodie.py`

**问题**：每个交互只处理单节点 `uid = int(user_nodes[0].item())`。真正的批量 GNN 消息传递不可行。

**建议**：为 `compute_message` 增加批量版本，支持一次处理多个交互的节点（需要更复杂的邻居索引）。

### 3.4 魔数种子偏移量（代码坏味 3.1）

**文件**：`jodie/nas/trainer.py`

**问题**：
- `FINAL_RETRAIN_SEED_OFFSET = 20000`（已具名但无解释）
- `seed + 10000 + idx`（10000 硬编码无解释）
- `_partition_seed` 中的 `epoch*100000`

**建议**：统一使用命名常量并添加推导说明。

### 3.5 evaluate_arch_pipeline 时间统计（代码坏味 3.9）

**文件**：`jodie/nas/trainer.py` 第 466 行

**问题**：`time_sec = round(elapsed / max(len(pipeline_results), 1), 4)` 把总时间除以架构数，不是实际训练时间。

**建议**：在 `run()` 返回时携带每个 payload 的实际执行时间。

---

## 四、评估方法改进

### 4.1 可选训练交互过滤

**文件**：`jodie/training/metrics.py` `evaluate_partition_ranking()`

**背景**：Bug 1.5 的分析确认当前评估符合 JODIE 论文标准（所有物品作为候选）。但某些场景下可能需要严格冷启动评估。

**建议**：新增可选参数 `--exclude-train-items`：
- 评估时为每个用户过滤掉训练期间见过的物品
- 仅用于 cold-start 场景实验

### 4.2 多种子评估默认值

**当前**：`EVAL_SEEDS = ""` 表示单种子评估。

**建议**：默认至少用 3 个种子，降低单种子偶然性。同时增加标准差输出。

---

## 五、代码质量

### 5.1 jodie/training/__init__.py 仍未定义公共 API

**文件**：`jodie/training/__init__.py`（空文件）

**建议**：添加包级别重导出，明确公共 API：
```python
from jodie.training.loops import BPRLoss, train_model, train_model_ce
from jodie.training.metrics import evaluate_ranking_metrics, evaluate_recall_by_type
from jodie.training.batching import _create_t_batches, _create_time_windows
```

### 5.2 _trace_key 使用不一致（代码坏味 3.3）

**文件**：`jodie/nas/ray_pipeline.py`

**问题**：`_trace_key` 方法已定义但大部分调用点以内联方式构建 key。

**建议**：要么全局使用 `_trace_key`，要么删除它并标准化内联方式。

### 5.3 类型标注覆盖率

当前代码只有部分类型标注（`typing` 导入已存在但未全面使用）。

**建议**：为关键接口添加完整类型标注，特别是：
- `trainer.py` 中的所有 public 方法
- `ray_pipeline.py` 中的 `PipelineModelPayload` 相关方法

---

## 六、功能增强

### 6.1 搜索进度可恢复性

**当前**：搜索中断后无法恢复，所有已评估架构丢失。

**建议**：每 N 个 trial 自动保存 checkpoint（包含 `seen_signatures`、已评估结果、controller 状态），支持 `--resume` 恢复。

### 6.2 实时可视化

**建议**：搜索过程中输出 JSON-lines 格式的进度日志，配套简单的 Python 脚本生成：
- 每 trial 的 score 趋势图
- 四种策略的实时对比曲线
- 搜索空间探索热力图（显示哪些参数组合被采样过）

### 6.3 早停策略

**建议**：如果连续 N 个 trial 的 best_score 不再提升，提前结束搜索。

---

## 七、文档完善

### 7.1 贡献者入门指南

在 README.md 中补充：
- 如何添加新的搜索空间
- 如何添加新的模型类型
- 如何添加新的执行模式

### 7.2 配置文件替代

当前所有参数在 `run_all.py` 中以 Python 变量定义。未来可考虑：
- YAML 配置文件支持
- 多组参数批量实验（grid search over configs）

---

## 八、未完成的问题清单（快速索引）

| 编号 | 类别 | 简述 | 优先级 |
|------|------|------|--------|
| 1.2-B | 论文 | 内存约束驱动的 Pipeline 必要性论证 | **最高** |
| 2.1 | Bug | Smart 模式 stage 检测逻辑修复 | 高 |
| 2.2 | 参数 | 自动配置阈值调整 | 高 |
| 3.1 | 架构 | 心跳循环去重 | 中 |
| 3.2 | 架构 | MemShareDPExecutor 补齐 batch 模式 | 低 |
| 3.3 | 架构 | 批量消息传递 | 低 |
| 3.4 | 代码 | 魔数种子统一命名 | 低 |
| 3.5 | 代码 | Pipeline 时间统计修复 | 中 |
| 4.1 | 评估 | 可选训练交互过滤 | 低 |
| 4.2 | 评估 | 多种子评估默认 | 低 |
| 5.1 | 代码 | training/__init__.py 公共 API | 低 |
| 5.2 | 代码 | _trace_key 统一 | 低 |
| 5.3 | 代码 | 类型标注覆盖率 | 低 |
| 6.1 | 功能 | 搜索可恢复性 | 中 |
| 6.2 | 功能 | 实时可视化 | 中 |
| 6.3 | 功能 | 早停策略 | 中 |
| 7.1 | 文档 | 贡献者入门指南 | 低 |
| 7.2 | 功能 | 配置文件支持 | 中 |

---

## 优先级排序（建议实施顺序）

```
第一优先（论文必需）:
  ├── 1.2-B: 内存约束 pipeline 必要性
  ├── 2.1: Smart 模式 stage 检测修复
  └── 2.2: 自动配置阈值调整

第二优先（代码质量）:
  ├── 3.1: 心跳循环去重
  ├── 3.5: Pipeline 时间统计
  └── 5.1: training 公共 API

第三优先（功能增强）:
  ├── 6.1: 搜索可恢复
  ├── 6.2: 实时可视化
  └── 6.3: 早停

第四优先（锦上添花）:
  └── 其余所有
```
