# 使用命令行参数启用实时效率监控

## 新增参数

在 `search.py` 中添加了两个新的命令行参数：

### 1. `--enable-efficiency-monitor`
- **类型**: 布尔标志（action="store_true"）
- **说明**: 启用实时效率监控
- **默认值**: False（不启用）

### 2. `--efficiency-monitor-interval`  
- **类型**: 整数
- **说明**: 效率监控的采样间隔（秒）
- **默认值**: 10

---

## 使用方法

### 方法 1：启用监控，使用默认间隔（10秒）

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --search-mode rl \
  --execution-mode ray_pipeline \
  --coarse-trials 6 \
  --coarse-epochs 1 \
  --architectures-per-step 6 \
  --pipeline-trace \
  --enable-efficiency-monitor \
  --output-dir outputs_monitored
```

### 方法 2：启用监控，自定义间隔（5秒采样一次）

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --search-mode rl \
  --execution-mode ray_pipeline \
  --coarse-trials 6 \
  --coarse-epochs 1 \
  --architectures-per-step 6 \
  --pipeline-trace \
  --enable-efficiency-monitor \
  --efficiency-monitor-interval 5 \
  --output-dir outputs_monitored
```

### 方法 3：启用监控，30秒采样一次（更疏稀）

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --search-mode rl \
  --execution-mode ray_pipeline \
  --coarse-trials 6 \
  --coarse-epochs 1 \
  --architectures-per-step 6 \
  --pipeline-trace \
  --enable-efficiency-monitor \
  --efficiency-monitor-interval 30 \
  --output-dir outputs_monitored
```

### 方法 4：不启用监控（默认行为）

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --search-mode rl \
  --execution-mode ray_pipeline \
  --coarse-trials 6 \
  --coarse-epochs 1 \
  --architectures-per-step 6 \
  --pipeline-trace \
  --output-dir outputs_baseline
```

---

## 工作流程

当启用 `--enable-efficiency-monitor` 时，Pipeline 会自动执行以下步骤：

### 1️⃣ 启动 Pipeline（主进程）
```
search.py 
  → GraphNASTrainer.search_pipeline()
    → 启动 Ray Pipeline 执行
    → 生成 pipeline_trace_*.log
```

### 2️⃣ 启动效率监控子进程
```
当 Pipeline 开始时，自动启动：
  python monitor_pipeline_efficiency.py <trace_file> <interval>
  
该子进程会：
  • 每隔 N 秒读取 trace 日志
  • 计算当前效率指标
  • 写入 efficiency_log_*.csv
```

### 3️⃣ 实时输出效率数据
```
在终端看到实时更新：
[Efficiency Monitor] GPU利用率: 33.3% | GPU效率: 30.7% | Speedup: 1.05x
[Efficiency Monitor] GPU利用率: 35.2% | GPU效率: 32.1% | Speedup: 1.08x
...
```

### 4️⃣ Pipeline 完成后生成报告
```
当 Pipeline 完成时，自动：
  • 停止监控子进程
  • 运行 visualize_efficiency_log.py 生成摘要报告
  • 保存报告到 efficiency_log_*_report.txt
```

---

## 输出文件

使用 `--enable-efficiency-monitor` 后，输出目录中会生成：

```
outputs_monitored/
├── pipeline_trace_20260428_xxxxxx.log        # Pipeline trace 日志
├── efficiency_log_20260428_xxxxxx.csv        # 🆕 实时效率指标数据
├── efficiency_log_20260428_xxxxxx_report.txt # 🆕 效率分析报告
├── best_arch.json                            # 最佳架构
├── leaderboard.csv                           # 排行榜
└── ... 其他输出
```

---

## 示例：对比两个配置

### 配置 A：基准（不启用监控）
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --search-mode rl \
  --execution-mode ray_pipeline \
  --architectures-per-step 3 \
  --pipeline-stage-train-workers 2,1 \
  --pipeline-trace \
  --output-dir outputs_config_a
```

### 配置 B：优化（启用监控）
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --search-mode rl \
  --execution-mode ray_pipeline \
  --architectures-per-step 6 \
  --num-pipeline-stages 3 \
  --pipeline-stage-train-workers 3,2,2 \
  --pipeline-trace \
  --enable-efficiency-monitor \
  --efficiency-monitor-interval 10 \
  --output-dir outputs_config_b
```

### 比较效果
```bash
echo "=== 配置 A（基准）==="
tail -20 outputs_config_a/efficiency_log_*.csv

echo ""
echo "=== 配置 B（优化）==="
tail -20 outputs_config_b/efficiency_log_*.csv

# 查看报告
cat outputs_config_a/efficiency_log_*_report.txt
cat outputs_config_b/efficiency_log_*_report.txt
```

---

## 常见问题

### Q1: 监控不启动？
A: 确保：
1. 使用了 `--pipeline-trace` 参数（生成 trace 日志）
2. 使用了 `--enable-efficiency-monitor` 参数
3. `monitor_pipeline_efficiency.py` 在当前目录或 Python path 中

### Q2: 效率指标何时开始有数据？
A: 通常在 Pipeline 运行 10-30 秒后，当有足够的 dispatch/complete 事件时。

### Q3: 监控会影响 Pipeline 性能吗？
A: 影响很小，因为监控运行在单独的子进程中。如果发现性能问题，可以：
- 增加 `--efficiency-monitor-interval` 到 30-60 秒
- 关闭监控，使用 `--pipeline-trace` 后手动分析

### Q4: 能否在运行中修改监控间隔？
A: 不能，监控参数在 Pipeline 启动时确定。需要重新启动 Pipeline。

### Q5: 监控失败会影响 Pipeline 吗？
A: 不会。如果监控无法启动，Pipeline 会继续正常运行，只是不会生成效率日志。

---

## 最佳实践

### 1. 快速测试（小规模）
```bash
--enable-efficiency-monitor \
--efficiency-monitor-interval 5      # 采样更频繁
--coarse-trials 3                    # 少数 trial 快速测试
```

### 2. 完整测试（中等规模）
```bash
--enable-efficiency-monitor \
--efficiency-monitor-interval 10     # 标准间隔
--coarse-trials 6-12
```

### 3. 大规模搜索（生产级）
```bash
--enable-efficiency-monitor \
--efficiency-monitor-interval 30     # 采样更疏稀，减少开销
--coarse-trials 50+
```

### 4. 多配置并行测试
开多个终端，分别运行不同配置的 Pipeline，同时对比效率。

---

## 集成到脚本

如果你想在自己的脚本中调用 search.py，可以这样：

```bash
#!/bin/bash

# 配置参数
DATASET="public_csv"
DATA_PATH="data/public/mooc.csv"
TRIALS="6"
WORKERS="3,2"
MONITOR_INTERVAL="10"

# 构建命令
CMD="CUDA_VISIBLE_DEVICES=0,1,2 python search.py \
  --dataset $DATASET \
  --local-data-path $DATA_PATH \
  --search-mode rl \
  --execution-mode ray_pipeline \
  --coarse-trials $TRIALS \
  --coarse-epochs 1 \
  --architectures-per-step 6 \
  --pipeline-stage-train-workers $WORKERS \
  --pipeline-trace \
  --enable-efficiency-monitor \
  --efficiency-monitor-interval $MONITOR_INTERVAL \
  --output-dir outputs_test"

echo "运行命令："
echo "$CMD"

# 执行
eval "$CMD"
```

---

**🎉 现在你可以在运行 Pipeline 时直接启用实时效率监控！**
