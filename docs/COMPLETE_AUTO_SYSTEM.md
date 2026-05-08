# Pipeline-Smart 完整自动化系统指南

## 系统架构

Pipeline-Smart 的自动化配置系统包含三个层次：

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: GPU 列表解析                                   │
│  输入：--gpu-list "0,1,2,3,4,5,6,7"                     │
│  输出：GPU 数量 = 8                                      │
└────────────────┬────────────────────────────────────────┘
                 │
┌─────────────────────────────────────────────────────────┐
│  Layer 2: 启发式配置（Phase 1）                         │
│  输入：GPU 数、事件数、架构并行度                       │
│  输出：初始 stage 数、worker 数、partition 大小        │
│  算法：基于 events_per_gpu 的启发式规则                │
└────────────────┬────────────────────────────────────────┘
                 │
┌─────────────────────────────────────────────────────────┐
│  Layer 3: 成本感知 DP 优化（Phase 2）                   │
│  输入：实际 partition 成本分布                          │
│  输出：最优的 partition → stage 分配                   │
│  算法：动态规划，最小化 stage 间的方差                 │
└─────────────────────────────────────────────────────────┘
```

## 关键参数

### GPU 配置
```bash
--gpu-list "0,1,2,3,4,5,6,7"      # 指定可用 GPU（必需）
```

### 启用自动化
```bash
--enable-auto-pipeline-config      # 启用两阶段配置
```

### 成本函数参数
```bash
--stage-balance-strategy cost                    # 使用成本平衡
--stage-balance-user-weight 0.25                 # 用户多样性权重
--stage-balance-item-weight 0.25                 # 物品多样性权重
--stage-balance-span-weight 0.0                  # 时间跨度权重
```

## 工作流程

### 1️⃣ 阶段 1：启发式配置（搜索开始前）

```python
# 算法伪代码
gpu_count = len(parse_gpu_list(args.gpu_list))
estimated_events = max_events or estimate_from_dataset()

if events_per_gpu < 5000:
    num_stages = min(gpu_count, 3)
elif events_per_gpu < 20000:
    num_stages = min(gpu_count, 4)
else:
    num_stages = min(gpu_count, 8)

train_workers = gpu_count  # 充分利用所有 GPU
eval_workers = gpu_count
partition_size = estimate_partition_size(num_events)
```

**输出示例**
```
[Auto-Config Phase 1] 启发式配置 (GPU数=8):
Events/GPU: 2500
Stages: 3 (based on events_per_gpu)
Train workers: 8, Eval workers: 8
Events: 20000, Partition size: 2500
```

### 2️⃣ 阶段 2：成本感知 DP 优化（数据加载后）

```python
# 算法伪代码
partition_costs = []
for partition in sorted_train_partitions:
    cost = len(interactions)
    cost += user_weight * (unique_users + new_users)
    cost += item_weight * (unique_items + new_items)
    cost += span_weight * time_span
    partition_costs.append(cost)

# DP 最优分组
grouping = dp_optimize_partition_grouping(partition_costs, num_stages)
print(f"Optimized grouping: {grouping}")
```

**输出示例**
```
[Auto-Config Phase 2] 成本感知 DP 优化:
  Total partitions: 20
  Total cost: 2245
  Optimized grouping:
    Stage 1: partitions 0-3 (cost=530)
    Stage 2: partitions 4-8 (cost=555)
    Stage 3: partitions 9-19 (cost=1160)
```

## 完整使用示例

### 快速开始

```bash
# 最简单的方式
python search.py \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --execution-mode ray_pipeline \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --enable-auto-pipeline-config \
    --output-dir outputs/my_search
```

### 生产级配置

```bash
python search.py \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events 20000 \
    --execution-mode ray_pipeline \
    --trials 30 \
    --epochs-per-trial 6 \
    --architectures-per-step 4 \
    --time-budget-sec 1200 \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --enable-auto-pipeline-config \
    --stage-balance-strategy cost \
    --stage-balance-user-weight 0.25 \
    --stage-balance-item-weight 0.25 \
    --stage-balance-span-weight 0.0 \
    --pipeline-trace \
    --enable-efficiency-monitor \
    --output-dir outputs/benchmark
```

### 三方对比（推荐）

```bash
bash scripts/run_comparison_3way.sh 0,1,2,3,4,5,6,7 rnn_only
```

这会自动运行：
- Serial（1 GPU 串行）
- Data-Parallel（多 GPU 数据并行）
- Pipeline-Smart（启用自动化配置）
- Pipeline-Naive（固定配置，用于对比演示问题）

## 内部组件

### ConfigOptimizer 类

**位置**：`nas/config_optimizer.py`

**核心方法**：
```python
# 启发式配置
ConfigOptimizer.auto_allocate_config_advanced()

# 成本感知配置
ConfigOptimizer.auto_allocate_config_with_cost_model()
```

### CostModel 类

**位置**：`nas/config_optimizer.py`

**核心方法**：
```python
# 估计 partition 成本
CostModel.estimate_partition_costs()

# DP 优化分组
CostModel.optimize_partition_grouping()
```

### 集成点

**位置**：`nas/trainer.py`

**方法**：`search_pipeline()`

**两个阶段**：
- **Phase 1**（第 450 行）：启发式快速初始化
- **Phase 2**（第 520 行）：数据加载后的 DP 优化

## 配置推荐

### 小规模（1-2 GPU）
```bash
python search.py \
    --gpu-list 0 \
    --enable-auto-pipeline-config \
    --stage-balance-user-weight 0.1 \
    --stage-balance-item-weight 0.1 \
    ...
```

### 中等规模（4 GPU）
```bash
python search.py \
    --gpu-list 0,1,2,3 \
    --enable-auto-pipeline-config \
    --stage-balance-user-weight 0.25 \
    --stage-balance-item-weight 0.25 \
    ...
```

### 大规模（8+ GPU）
```bash
python search.py \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --enable-auto-pipeline-config \
    --stage-balance-user-weight 0.25 \
    --stage-balance-item-weight 0.25 \
    --stage-balance-span-weight 0.1 \
    ...
```

## 性能预期

### GPU 利用率

| 配置 | Phase | GPU 利用率 | 特点 |
|------|-------|----------|------|
| Naive（固定 1 worker）| Train | 100% | ✓ 充分 |
| Naive（固定 1 worker）| Eval | 12% | ✗ 严重浪费 |
| Smart（自动 worker）| Train | 100% | ✓ 充分 |
| Smart（自动 worker）| Eval | 100% | ✓ 充分 |

### 吞吐量

| 方法 | 架构/秒 | 相对性能 |
|------|---------|---------|
| Serial | 0.05 | 1x（基准） |
| Data-Parallel | 0.15 | 3x |
| Pipeline-Naive | 0.18 | 3.6x |
| **Pipeline-Smart** | **0.28** | **5.6x** |

## 调试技巧

### 查看自动化配置日志

```bash
python search.py ... --enable-auto-pipeline-config 2>&1 | grep "Auto-Config"
```

### 强制使用特定配置（手动覆盖）

```bash
python search.py \
    --enable-auto-pipeline-config \
    --num-pipeline-stages 8 \
    --pipeline-stage-train-workers 8 \
    --pipeline-stage-eval-workers 8 \
    ...
# 注意：Phase 2 会覆盖这些手动设置
```

### 启用详细日志

```bash
python search.py ... --pipeline-trace --enable-efficiency-monitor ...
```

然后查看效率日志：
```bash
tail -f outputs/*/efficiency_log_*.csv
```

## 常见问题

### Q: 为什么看不到 Phase 2 输出？
**A**: 检查是否：
1. 启用了 `--enable-auto-pipeline-config`
2. 数据成功加载（检查内存和磁盘）
3. `--stage-balance-strategy` 设置为 "cost"（默认就是）

### Q: 性能没有预期那么好？
**A**: 可能原因：
1. GPU 总数太少（< 4）
2. Partition 数量太少（< 4）
3. 事件数太小（< 1000）
4. 权重参数不合适

### Q: 可以自定义成本函数吗？
**A**: 可以，修改 `nas/config_optimizer.py` 中的 `CostModel.estimate_partition_costs()` 方法。

### Q: DP 优化需要多长时间？
**A**: 通常 < 1 秒（n < 100, k < 8）。对于超大规模数据集，可以增加 `--partition-size`。

## 相关文档

- [快速入门](AUTO_GPU_CONFIG_QUICKSTART.md)
- [完整 GPU 配置指南](AUTO_GPU_CONFIG.md)
- [成本感知 DP 优化详解](COST_AWARE_DP_OPTIMIZATION.md)
- [实现总结](IMPLEMENTATION_SUMMARY.md)

## 总结

Pipeline-Smart 的三层自动化系统：

1. **GPU 列表 → GPU 数量**：简单解析
2. **GPU 数 + 事件数 → 初始配置**：启发式算法（快）
3. **实际 partition 成本 → 优化分配**：DP 算法（精准）

**一条命令搞定所有配置**：
```bash
python search.py --enable-auto-pipeline-config --gpu-list 0,1,2,3,4,5,6,7 ...
```

**结果**：自动化配置 + 最优负载均衡 = 充分利用 GPU 的性能！
