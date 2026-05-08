# 🚀 run_comparison_3way.sh CLI 参数指南

## 快速开始

```bash
# 显示帮助
bash scripts/run_comparison_3way.sh --help

# 使用默认参数（自动检测 GPU）
bash scripts/run_comparison_3way.sh

# 快速测试
bash scripts/run_comparison_3way.sh --max-events 5000 --time-budget 60 --trials 10

# 完整对比（8 个 GPU）
bash scripts/run_comparison_3way.sh --gpu-list 0,1,2,3,4,5,6,7 --trials 50
```

## 所有可用参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--gpu-list` | GPU ID 列表（逗号分隔） | 自动检测 | `0,1,2,3,4,5,6,7` |
| `--space` | 搜索空间类型 | `rnn_only` | `small`, `large` |
| `--dataset` | 数据集类型 | `public_csv` | `synthetic` |
| `--data-file` | 数据文件路径 | `data/public/mooc.csv` | `data/synthetic.csv` |
| `--max-events` | 最大事件数 | `20000` | `5000`, `50000` |
| `--time-budget` | 每个方法的时间预算（秒） | `1200` | `60`, `600` |
| `--epochs` | 每个 trial 的 epoch 数 | `3` | `1`, `5` |
| `--trials` | 试验上限（所有方法相同） | `30`* | `10`, `50` |
| `--seeds` | 种子列表（逗号分隔） | `42,43` | `42,43,44,45` |
| `--output-dir` | 输出目录 | `outputs/comparison_TIMESTAMP` | `outputs/my_exp` |
| `--help` | 显示帮助信息 | - | - |

*默认值 30 实际是 999，由 `--time-budget` 控制

## 常见用法示例

### 1. 快速验证（5 分钟）

```bash
bash scripts/run_comparison_3way.sh \
    --max-events 5000 \
    --time-budget 60 \
    --trials 10 \
    --epochs 1 \
    --seeds 42
```

**说明**：
- 只用 5000 个事件（小规模数据）
- 60 秒时间预算（快速测试）
- 最多 10 个 trials
- 1 个 epoch（快速训练）
- 单个种子

### 2. 小规模对比（10-15 分钟）

```bash
bash scripts/run_comparison_3way.sh \
    --max-events 10000 \
    --time-budget 300 \
    --trials 20 \
    --epochs 2 \
    --seeds 42,43
```

**说明**：
- 10k 事件
- 300 秒（5 分钟）时间预算
- 20 个 trials
- 2 个 epochs
- 2 个种子（2 次重复）

### 3. 标准对比（1 小时）

```bash
bash scripts/run_comparison_3way.sh \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --max-events 20000 \
    --time-budget 1200 \
    --trials 50 \
    --epochs 3 \
    --seeds 42,43,44
```

**说明**：
- 使用全部 8 个 GPU
- 20k 事件（标准数据量）
- 1200 秒（20 分钟）时间预算 × 4 个方法
- 最多 50 个 trials
- 3 个 epochs
- 3 个种子（3 次重复）

### 4. 不同搜索空间对比

```bash
# 小搜索空间
bash scripts/run_comparison_3way.sh \
    --space small \
    --gpu-list 0,1,2,3 \
    --time-budget 300

# 大搜索空间
bash scripts/run_comparison_3way.sh \
    --space large \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --time-budget 1800
```

### 5. 自定义数据集

```bash
bash scripts/run_comparison_3way.sh \
    --dataset synthetic \
    --data-file data/synthetic_large.csv \
    --max-events 100000 \
    --time-budget 600 \
    --trials 30
```

### 6. 多个独立实验

```bash
# 实验 1: RNN-only
bash scripts/run_comparison_3way.sh \
    --space rnn_only \
    --output-dir outputs/exp_rnn_only

# 实验 2: GNN
bash scripts/run_comparison_3way.sh \
    --space gnn \
    --output-dir outputs/exp_gnn

# 实验 3: Hybrid
bash scripts/run_comparison_3way.sh \
    --space hybrid \
    --output-dir outputs/exp_hybrid
```

### 7. GPU 部分可用时

```bash
# 只用 GPU 0,2,4
bash scripts/run_comparison_3way.sh \
    --gpu-list 0,2,4 \
    --time-budget 600

# 只用 GPU 0
bash scripts/run_comparison_3way.sh \
    --gpu-list 0 \
    --time-budget 300
```

## 输出结构

每次运行会在输出目录下生成如下结构：

```
outputs/comparison_TIMESTAMP/
├── seed_42/
│   ├── serial/                  # Serial 方法结果
│   ├── data_parallel/           # Data-Parallel 方法结果
│   ├── pipeline/                # Pipeline-Smart 方法结果
│   ├── pipeline_naive/          # Pipeline-Naive 方法结果
│   ├── report_3way.txt          # 四方对比报告
│   └── report_pipeline_smart_vs_naive.txt  # 智能优化效果对比
├── seed_43/
│   └── [同上结构]
├── seed_times.csv               # 所有 seeds 的时间汇总
└── aggregate_report_3way.txt    # 多 seeds 汇总报告
```

## 性能预期

| 数据量 | 时间预算 | 预期时间 | 用例 |
|-------|---------|---------|------|
| 5k 事件 | 60s | ~10 min | 快速验证 |
| 10k 事件 | 300s | ~30 min | 小规模对比 |
| 20k 事件 | 1200s | ~2 hours | 标准对比 |
| 50k 事件 | 1800s | ~3 hours | 大规模对比 |

*实际时间取决于 GPU、CPU 和数据集复杂度

## 故障排除

### 参数格式错误

```bash
# ❌ 错误：GPU 列表中有空格
bash scripts/run_comparison_3way.sh --gpu-list "0, 1, 2"

# ✅ 正确：无空格
bash scripts/run_comparison_3way.sh --gpu-list "0,1,2"
```

### Seed 参数

```bash
# ✅ 单个 seed
bash scripts/run_comparison_3way.sh --seeds 42

# ✅ 多个 seeds（逗号分隔）
bash scripts/run_comparison_3way.sh --seeds 42,43,44

# ✅ 或用引号
bash scripts/run_comparison_3way.sh --seeds "42,43,44"
```

### 查看正在运行的进程

```bash
# 查看 Python 进程
ps aux | grep search.py

# 查看 GPU 使用
nvidia-smi
```

## 参数组合建议

### 最小化测试
```bash
--max-events 1000 --time-budget 30 --epochs 1 --trials 3 --seeds 42
```

### 快速验证
```bash
--max-events 5000 --time-budget 60 --epochs 1 --trials 10 --seeds 42,43
```

### 标准对比
```bash
--max-events 20000 --time-budget 300 --epochs 3 --trials 50 --seeds 42,43,44
```

### 发表级结果
```bash
--max-events 20000 --time-budget 600 --epochs 5 --trials 100 --seeds 42,43,44,45,46
```

## 实验记录

建议保存实验参数到文件以便复现：

```bash
# 创建实验记录
cat > exp_config.sh << EOF
# 实验：Pipeline-Smart vs Naive 对比
GPU_LIST="0,1,2,3,4,5,6,7"
MAX_EVENTS=20000
TIME_BUDGET=600
TRIALS=50
EPOCHS=3
SEEDS="42,43,44"
EOF

# 运行实验
bash scripts/run_comparison_3way.sh \
    --gpu-list "$GPU_LIST" \
    --max-events "$MAX_EVENTS" \
    --time-budget "$TIME_BUDGET" \
    --trials "$TRIALS" \
    --epochs "$EPOCHS" \
    --seeds "$SEEDS"
```

## 常见问题

**Q: 如何在多个 GPU 上测试？**

A: 指定 `--gpu-list` 参数：
```bash
bash scripts/run_comparison_3way.sh --gpu-list 0,1,2,3
```

**Q: 如何调整搜索速度？**

A: 调整 `--time-budget` 和 `--trials` 参数：
```bash
# 快速搜索（少试验）
bash scripts/run_comparison_3way.sh --time-budget 60 --trials 10

# 深度搜索（多试验）
bash scripts/run_comparison_3way.sh --time-budget 1800 --trials 100
```

**Q: 如何运行多个独立实验？**

A: 使用 `--output-dir` 为每个实验指定不同目录：
```bash
bash scripts/run_comparison_3way.sh --output-dir outputs/exp1
bash scripts/run_comparison_3way.sh --output-dir outputs/exp2
```

**Q: 实验结果保存在哪里？**

A: 默认保存在 `outputs/comparison_TIMESTAMP/` 下，可用 `--output-dir` 自定义。
