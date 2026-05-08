# 快速开始：自动化 GPU 配置

## 5 分钟快速上手

### 1. 基本用法

**最简单的方式** - 让系统自动检测 GPU 并自动配置：

```bash
python search.py \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events 20000 \
    --execution-mode ray_pipeline \
    --trials 10 \
    --epochs-per-trial 3 \
    --enable-auto-pipeline-config \
    --output-dir outputs/my_search
```

### 2. 指定特定 GPU

如果你只想使用某些 GPU（例如 GPU 0, 1, 2）：

```bash
python search.py \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events 20000 \
    --execution-mode ray_pipeline \
    --trials 10 \
    --epochs-per-trial 3 \
    --gpu-list 0,1,2 \
    --enable-auto-pipeline-config \
    --output-dir outputs/my_search
```

### 3. 对比实验（推荐）

运行三方对比（Serial vs Data-Parallel vs Pipeline-Smart）：

```bash
# 自动检测 GPU
bash scripts/run_comparison_3way.sh

# 或指定 GPU（例如 8 GPU）
bash scripts/run_comparison_3way.sh 0,1,2,3,4,5,6,7

# 或指定搜索空间
bash scripts/run_comparison_3way.sh 0,1,2,3,4,5,6,7 rnn_only
```

## 输出示例

运行时，你会看到类似的自动配置日志：

```
[Auto-Config] 自动化 Pipeline 配置 (GPU数=8):
GPUs: 8, Stages: 3
Train workers: 8, Eval workers: 8
Events: 20000, Partitions: 0, Partition size: 2500
Trials: 10, Architectures/step: 2
```

这表示系统已经自动配置：
- 使用 8 个 GPU
- 创建 3 个 pipeline stage
- Train 和 Eval stage 各使用 8 个 worker（充分利用所有 GPU）
- 自动计算分区大小为 2500

## 关键参数说明

| 参数 | 作用 | 示例 |
|------|------|------|
| `--gpu-list` | 指定可用 GPU（逗号分隔） | `0,1,2` 或 `0,1,2,3,4,5,6,7` |
| `--enable-auto-pipeline-config` | 启用自动化配置 | 添加此标志即可 |
| `--max-events` | 限制数据大小（用于测试） | `20000` |
| `--execution-mode ray_pipeline` | 使用 Pipeline 执行模式 | 必须 |

## 什么是自动化配置做的？

自动化配置会根据 GPU 数量和数据规模自动计算：

✅ **Pipeline Stages 数量** - 多少个计算阶段  
✅ **Worker 数量** - 每个 stage 使用多少 worker  
✅ **分区大小** - 数据分成多大的块  

这确保了充分利用 GPU，避免阶段不平衡导致的性能下降。

## 不同规模的自动配置示例

### 小规模（1-2 GPU）
```bash
python search.py --gpu-list 0 --enable-auto-pipeline-config ...
# 自动配置：2 stages，1 worker/stage
```

### 中等规模（4 GPU）
```bash
python search.py --gpu-list 0,1,2,3 --enable-auto-pipeline-config ...
# 自动配置：3 stages，4 workers/stage
```

### 大规模（8 GPU）
```bash
python search.py --gpu-list 0,1,2,3,4,5,6,7 --enable-auto-pipeline-config ...
# 自动配置：3-8 stages，8 workers/stage
```

## 对比：自动化 vs 手动配置

### 自动化（推荐，Pipeline-Smart）✅
```bash
python search.py \
    --execution-mode ray_pipeline \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --enable-auto-pipeline-config \
    ...
```
- ✅ 无需计算参数
- ✅ 自动适配不同 GPU 配置
- ✅ 避免阶段不平衡
- ✅ 最优性能

### 手动配置（Pipeline-Naive，用于对比）
```bash
python search.py \
    --execution-mode ray_pipeline \
    --num-pipeline-stages 8 \
    --pipeline-stage-train-workers 1 \
    --pipeline-stage-eval-workers 1 \
    ...
```
- ❌ 需要手动计算
- ❌ 容易出错
- ❌ 阶段不平衡导致低效率
- ❌ 对比时用来演示问题

## 常见用例

### 用例 1：快速测试（小数据）
```bash
python search.py \
    --dataset synthetic \
    --num-interactions 1000 \
    --execution-mode ray_pipeline \
    --trials 3 \
    --epochs-per-trial 1 \
    --gpu-list 0,1,2 \
    --enable-auto-pipeline-config \
    --time-budget-sec 60 \
    --output-dir outputs/quick_test
```

### 用例 2：完整对比（标准配置）
```bash
bash scripts/run_comparison_3way.sh 0,1,2,3,4,5,6,7 rnn_only
```

### 用例 3：公开数据集基准测试
```bash
python search.py \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events 20000 \
    --execution-mode ray_pipeline \
    --trials 20 \
    --epochs-per-trial 3 \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --enable-auto-pipeline-config \
    --time-budget-sec 1200 \
    --output-dir outputs/benchmark
```

## 检查配置是否生效

搜索时查看日志中的 `[Auto-Config]` 部分：

```bash
python search.py ... --enable-auto-pipeline-config ... 2>&1 | grep -A5 "Auto-Config"
```

应该看到：
```
[Auto-Config] 自动化 Pipeline 配置 (GPU数=8):
GPUs: 8, Stages: 3
Train workers: 8, Eval workers: 8
Events: 20000, Partitions: 0, Partition size: 2500
Trials: 10, Architectures/step: 2
```

## 更多信息

详细文档请见：[docs/AUTO_GPU_CONFIG.md](AUTO_GPU_CONFIG.md)
