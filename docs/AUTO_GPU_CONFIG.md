# 自动化 GPU Worker 配置指南

## 概述

自动化 GPU Worker 配置是 Pipeline-Smart 的核心智能特性，可以根据以下因素自动分配资源：

- **GPU 数量**：系统可用的 GPU 个数
- **数据规模**：事件数、分区数等
- **搜索配置**：试验数、架构并行度等

## 核心特性

### 1. 智能化 Worker 分配
- **自动计算 Stage 数量**：基于 GPU 数量和数据规模
- **动态 Worker 分配**：train/eval stage 的 worker 数自动调整
- **分区大小优化**：根据事件数自动推算最优分区大小

### 2. GPU 列表支持
用户可通过 `--gpu-list` 参数指定可用 GPU，格式为逗号分隔的 GPU ID：
```bash
# 指定 GPU 0,1,2
python search.py --gpu-list 0,1,2 ...

# 指定 GPU 0,1,2,3,4,5,6,7（8GPU 服务器）
python search.py --gpu-list 0,1,2,3,4,5,6,7 ...
```

### 3. 启用自动化配置
使用 `--enable-auto-pipeline-config` 标志启用自动化配置：
```bash
python search.py \
    --execution-mode ray_pipeline \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --enable-auto-pipeline-config \
    ...
```

## 使用方式

### 方式 1：三方对比脚本（推荐）

脚本已经集成自动化配置到 Pipeline-Smart：

```bash
# 手动指定 GPU
bash scripts/run_comparison_3way.sh 0,1,2

# 自动检测所有可用 GPU
bash scripts/run_comparison_3way.sh

# 指定搜索空间
bash scripts/run_comparison_3way.sh 0,1,2,3,4,5,6,7 rnn_only
```

在运行时，你会看到类似的输出：

```
[Auto-Config] 自动化 Pipeline 配置 (GPU数=8):
GPUs: 8, Stages: 8
Train workers: 8, Eval workers: 8
Events: 20000, Partitions: 0, Partition size: 5000
Trials: 6, Architectures/step: 2
```

### 方式 2：直接调用单个搜索

```bash
python search.py \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events 20000 \
    --space rnn_only \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --trials 10 \
    --epochs-per-trial 3 \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --enable-auto-pipeline-config \
    --output-dir outputs/my_search
```

### 方式 3：测试脚本

使用提供的测试脚本快速验证自动化配置：

```bash
# 快速测试（小数据集）
bash scripts/test_auto_gpu_config.sh 0,1,2

# 测试 8GPU 配置
bash scripts/test_auto_gpu_config.sh 0,1,2,3,4,5,6,7

# 自动检测 GPU
bash scripts/test_auto_gpu_config.sh
```

## 自动化配置算法

### 输入参数
- `gpu_count`: 从 `--gpu-list` 解析得到的 GPU 数量
- `num_events`: 从 `--max-events` 获取，或根据 `--num-interactions` 估计
- `num_partitions`: 从数据准备阶段获取
- `architectures_per_step`: 从 `--architectures-per-step` 获取
- `coarse_trials`: 从 `--trials` 获取

### 输出配置
自动化算法生成以下配置参数（无需手动指定）：

```python
{
    'num_pipeline_stages': int,           # Pipeline 的 stage 数量
    'pipeline_stage_train_workers': str,  # Train stage 的 worker 数
    'pipeline_stage_eval_workers': str,   # Eval stage 的 worker 数
    'partition_size': int,                # 时间分区大小
    'architectures_per_step': int,        # 每步并行架构数
}
```

### 算法原理

#### 1. Stage 数量确定
```
- 基准：max_stages = min(gpu_count, 8)
- 原则：充分利用 GPU 级别的管道并行
- 最小值：2（train + eval）
```

#### 2. Worker 分配策略
```
当前实现：所有 stage 共用所有 GPU
- train_workers_per_stage = gpu_count
- eval_workers_per_stage = gpu_count

优点：充分利用 GPU，避免阶段不平衡
```

#### 3. 分区大小推断
```
启发式：num_events / partition_size >= architectures_per_step × num_stages

具体算法：
- 如果 num_events < 10,000：partition_size = max(500, num_events / 4)
- 如果 num_events < 100,000：partition_size = max(2000, num_events / 8)
- 否则：partition_size = max(5000, num_events / 16)
```

## 配置示例

### 小规模 (1-2 GPU)
```
GPU: 1
自动配置：
  - Stages: 2
  - Train workers: 1
  - Eval workers: 1
  - Partition size: 500-1000
```

### 中等规模 (4 GPU)
```
GPU: 4
自动配置：
  - Stages: 4
  - Train workers: 4
  - Eval workers: 4
  - Partition size: 2000-5000
```

### 大规模 (8 GPU)
```
GPU: 8
自动配置：
  - Stages: 8
  - Train workers: 8
  - Eval workers: 8
  - Partition size: 5000+
```

## 与手动配置的对比

### 自动化配置（推荐）
```bash
python search.py \
    --execution-mode ray_pipeline \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --enable-auto-pipeline-config \
    ...
```

**优点：**
- 无需手动计算 worker 数、stage 数
- 根据 GPU 和数据自动调整
- 避免 stage imbalance（阶段不平衡）

### 手动配置
```bash
python search.py \
    --execution-mode ray_pipeline \
    --num-pipeline-stages 8 \
    --pipeline-stage-train-workers 8 \
    --pipeline-stage-eval-workers 8 \
    --partition-size 5000 \
    ...
```

**缺点：**
- 需要用户手动调整参数
- 不同 GPU 配置需要修改参数
- 容易出现阶段不平衡

## 性能优化建议

### 1. 最大化 GPU 利用
- 确保 `train_workers` 和 `eval_workers` 都等于 GPU 数
- 这样避免在 eval 阶段 GPU 空闲

### 2. 分区大小选择
- 太小：同步开销大，通信成为瓶颈
- 太大：分区间工作量差异大，某些 worker 提前完成
- 推荐：1000-5000 事件/分区

### 3. 架构并行度
- 增加 `--architectures-per-step` 可以更好地利用管道并行
- 但需要足够的分区数支持

## 故障排除

### 问题 1：GPU 利用率低
**可能原因：**
- train_workers < gpu_count 导致 GPU 未充分利用
- eval_workers < gpu_count 导致 eval 阶段 GPU 空闲

**解决方案：**
- 确保启用了 `--enable-auto-pipeline-config`
- 检查自动化配置输出中的 worker 数

### 问题 2：Pipeline 效率低于 Data-Parallel
**可能原因：**
- 阶段不平衡（train 和 eval worker 数不匹配）
- 分区大小不合适

**解决方案：**
- 使用自动化配置
- 检查效率日志 (efficiency_log_*.csv)

### 问题 3：内存溢出
**可能原因：**
- partition_size 太大，单分区数据过多

**解决方案：**
- 减小 `--partition-size`
- 或减少 `--max-events`

## 查看详细配置

搜索运行时，`[Auto-Config]` 日志会输出详细的配置信息：

```
[Auto-Config] 自动化 Pipeline 配置 (GPU数=8):
GPUs: 8, Stages: 8
Train workers: 8, Eval workers: 8
Events: 20000, Partitions: 0, Partition size: 1000
Trials: 6, Architectures/step: 2
```

此外，可以查看效率监控日志：
```bash
# 查看实时 GPU 利用率
tail -f outputs/test_auto_config_small/efficiency_log_*.csv
```

## 相关参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu-list` | "0,1,2" | GPU 列表，逗号分隔 |
| `--enable-auto-pipeline-config` | False | 启用自动化配置 |
| `--max-events` | 0 | 最大事件数（0=无限制） |
| `--num-pipeline-stages` | 2 | 手动设置 stage 数（自动化配置时被覆盖） |
| `--pipeline-stage-train-workers` | "" | 手动设置 train worker 数（自动化配置时被覆盖） |
| `--pipeline-stage-eval-workers` | "" | 手动设置 eval worker 数（自动化配置时被覆盖） |
| `--partition-size` | 0 | 手动设置分区大小（自动化配置时被覆盖） |

## 总结

自动化 GPU Worker 配置是 Pipeline-Smart 相比 Pipeline-Naive 的核心智能特性：

- **Pipeline-Smart**：自动配置 stages、workers、partitions，充分利用 GPU
- **Pipeline-Naive**：固定 1 worker/stage，导致 GPU 利用率低

使用 `--enable-auto-pipeline-config` 可以在各种 GPU 配置上获得最优性能。
