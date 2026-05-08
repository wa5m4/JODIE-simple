# 🎯 run_comparison_3way.sh 改进总结

## 改进内容

### ✅ 已完成的更新

脚本 [scripts/run_comparison_3way.sh](scripts/run_comparison_3way.sh) 已升级为支持完整的命令行参数配置。

#### 1. **命令行参数支持**

原来：
```bash
bash run_comparison_3way.sh [GPU_LIST] [SEARCH_SPACE]
# 其他参数硬编码在脚本中
```

现在：
```bash
bash run_comparison_3way.sh [OPTIONS]

# 所有参数均可通过 --xxx 指定
bash scripts/run_comparison_3way.sh \
    --gpu-list 0,1,2,3 \
    --max-events 10000 \
    --time-budget 300 \
    --trials 20 \
    --epochs 2 \
    --seeds 42,43
```

#### 2. **支持的参数列表**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gpu-list` | GPU ID 列表（逗号分隔，无空格） | 自动检测所有 GPU |
| `--space` | 搜索空间类型 | `rnn_only` |
| `--dataset` | 数据集类型 | `public_csv` |
| `--data-file` | 数据文件路径 | `data/public/mooc.csv` |
| `--max-events` | 最大事件数 | `20000` |
| `--time-budget` | 每个方法的时间预算（秒） | `1200` |
| `--epochs` | 每个 trial 的 epoch 数 | `3` |
| `--trials` | 试验上限 | `999`（实际由 time-budget 控制） |
| `--seeds` | 种子列表（逗号分隔） | `42,43` |
| `--output-dir` | 输出目录 | `outputs/comparison_TIMESTAMP` |
| `--help` | 显示帮助信息 | - |

#### 3. **智能默认值**

- **自动 GPU 检测**：如果未指定 `--gpu-list`，脚本自动检测可用的 GPU 数量
- **时间戳输出目录**：默认输出目录包含时间戳，防止覆盖历史结果
- **灵活的 trials 控制**：通过 `--trials` 统一设置所有方法的试验上限

#### 4. **向后兼容性**

保持了原有的大部分核心逻辑，确保不破坏现有功能。

### 📝 新增文档

#### 1. **[docs/CLI_PARAMETERS_GUIDE.md](docs/CLI_PARAMETERS_GUIDE.md)** - 详细 CLI 参数指南
- 快速开始部分
- 所有参数的详细说明
- 5+ 个常见用法示例
- 参数组合建议
- 故障排除指南

#### 2. **[demo_cli_examples.sh](demo_cli_examples.sh)** - 参数演示脚本
- 展示 5 种常见使用场景
- 参数快速参考表
- 易于学习和复制

### 🚀 使用示例

#### 快速验证（5 分钟）
```bash
bash scripts/run_comparison_3way.sh \
    --max-events 5000 \
    --time-budget 60 \
    --trials 10 \
    --epochs 1
```

#### 标准对比（2 小时）
```bash
bash scripts/run_comparison_3way.sh \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --max-events 20000 \
    --time-budget 600 \
    --trials 50 \
    --epochs 3 \
    --seeds 42,43,44
```

#### 自定义实验
```bash
bash scripts/run_comparison_3way.sh \
    --space small \
    --dataset synthetic \
    --output-dir outputs/my_experiment
```

### 📊 输出结构不变

脚本仍然在输出目录下生成相同的结构：
```
outputs/comparison_TIMESTAMP/
├── seed_42/
│   ├── serial/
│   ├── data_parallel/
│   ├── pipeline/
│   ├── pipeline_naive/
│   ├── report_3way.txt
│   └── report_pipeline_smart_vs_naive.txt
├── seed_43/
│   └── [同上]
├── seed_times.csv
└── aggregate_report_3way.txt
```

### 🔧 技术细节

#### 参数解析实现
脚本使用 `while [[ $# -gt 0 ]]` 循环逐个解析参数，支持：
- 标准 `--key value` 格式
- 灵活的参数顺序
- 智能的默认值处理

#### 数组转换
```bash
# Seed 和 GPU 列表均支持逗号分隔，脚本内部转换为 bash 数组
IFS=',' read -ra SEEDS <<< "$SEEDS"
IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST"
```

### ✨ 特殊功能

#### 1. **自动 GPU 检测**
```bash
# 不指定 GPU，自动检测所有可用 GPU
bash scripts/run_comparison_3way.sh
```

#### 2. **灵活的 GPU 指定**
```bash
# 显式指定 GPU
bash scripts/run_comparison_3way.sh --gpu-list 0,1,2,3

# 部分 GPU（如 GPU 1,3,5 被占用）
bash scripts/run_comparison_3way.sh --gpu-list 0,2,4,6
```

#### 3. **多种子实验**
```bash
# 运行 3 次实验，每次不同的随机种子
bash scripts/run_comparison_3way.sh --seeds 42,43,44
```

### 📈 实验推荐参数组合

| 用途 | 命令 | 预期耗时 |
|------|------|---------|
| 快速检查 | `--max-events 1000 --time-budget 30 --epochs 1` | 5 min |
| 快速验证 | `--max-events 5000 --time-budget 60 --trials 10` | 10 min |
| 小规模对比 | `--max-events 10000 --time-budget 300 --trials 20` | 30 min |
| 标准对比 | `--max-events 20000 --time-budget 600 --trials 50` | 2 hours |
| 发表级结果 | `--max-events 20000 --time-budget 600 --trials 100 --seeds 42,43,44,45,46` | 10 hours |

### 🎓 学习资源

1. **快速开始**：`bash scripts/run_comparison_3way.sh --help`
2. **查看参数演示**：`bash demo_cli_examples.sh`
3. **详细指南**：参考 [docs/CLI_PARAMETERS_GUIDE.md](docs/CLI_PARAMETERS_GUIDE.md)

### 🔄 迁移指南（从旧脚本）

#### 旧用法 → 新用法

```bash
# 旧：指定 GPU 和搜索空间
bash run_comparison_3way.sh 0,1,2,3 rnn_only

# 新：使用 --xxx 参数
bash scripts/run_comparison_3way.sh \
    --gpu-list 0,1,2,3 \
    --space rnn_only
```

```bash
# 旧：所有参数硬编码
# DATASET="public_csv"
# MAX_EVENTS=20000
# ...

# 新：命令行指定
bash scripts/run_comparison_3way.sh \
    --dataset public_csv \
    --max-events 20000 \
    --time-budget 600
```

### ✅ 验证改进

运行以下命令验证改进功能：

```bash
# 1. 查看帮助
bash scripts/run_comparison_3way.sh --help

# 2. 查看参数演示
bash demo_cli_examples.sh

# 3. 运行快速测试（无需实际执行全流程）
bash test_cli_params.sh
```

### 📋 注意事项

1. **GPU 列表格式**：逗号分隔，无空格
   ```bash
   ✅ --gpu-list 0,1,2,3
   ❌ --gpu-list "0, 1, 2, 3"
   ```

2. **Seeds 格式**：逗号分隔
   ```bash
   ✅ --seeds 42,43,44
   ✅ --seeds "42,43,44"
   ```

3. **时间预算**：单位是秒，四个方法会并行运行
   ```bash
   # 总耗时约为 time-budget × 4 + 开销
   bash scripts/run_comparison_3way.sh --time-budget 300
   ```

### 🎯 下一步

所有参数现已支持命令行配置。你可以：

1. **运行快速测试**
   ```bash
   bash scripts/run_comparison_3way.sh --max-events 5000 --time-budget 60
   ```

2. **创建实验记录**
   ```bash
   cat > exp_config.sh << EOF
   GPU_LIST="0,1,2,3,4,5,6,7"
   MAX_EVENTS=20000
   TIME_BUDGET=600
   TRIALS=50
   EPOCHS=3
   SEEDS="42,43,44"
   EOF
   
   bash scripts/run_comparison_3way.sh \
       --gpu-list "$GPU_LIST" \
       --max-events "$MAX_EVENTS" \
       --time-budget "$TIME_BUDGET" \
       --trials "$TRIALS" \
       --epochs "$EPOCHS" \
       --seeds "$SEEDS"
   ```

3. **批量运行多个实验**
   ```bash
   # 对比不同搜索空间
   for space in rnn_only small large; do
       bash scripts/run_comparison_3way.sh \
           --space "$space" \
           --output-dir "outputs/exp_${space}"
   done
   ```

---

## 🔍 相关文件

- [scripts/run_comparison_3way.sh](scripts/run_comparison_3way.sh) - 改进的比较脚本
- [docs/CLI_PARAMETERS_GUIDE.md](docs/CLI_PARAMETERS_GUIDE.md) - 详细 CLI 参数指南
- [demo_cli_examples.sh](demo_cli_examples.sh) - 参数演示脚本
- [test_cli_params.sh](test_cli_params.sh) - 参数解析测试脚本

---

**最后更新**：$(date)
**改进状态**：✅ 完成
