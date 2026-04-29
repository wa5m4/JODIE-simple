# 实时效率监控系统 - 完整总结

## 📦 系统组成

本系统由 4 个核心组件组成，实现了 **Pipeline 训练过程中的实时效率监控**：

### 1. **命令行参数集成** (`search.py`)
- 新增 `--enable-efficiency-monitor` 参数（布尔值）
- 新增 `--efficiency-monitor-interval` 参数（整数，默认 10 秒）
- 参数自动传递到 trainer 中

### 2. **自动启动和停止** (`nas/trainer.py`)
- 在 `search_pipeline()` 开始时自动启动监控子进程
- 在 Pipeline 完成时自动停止监控
- 自动生成效率分析报告

### 3. **实时数据采集** (`monitor_pipeline_efficiency.py`)
- 持续读取 trace 日志文件
- 每隔 N 秒计算一次效率指标
- 将数据写入 CSV 格式的效率日志

### 4. **可视化和报告** (`visualize_efficiency_log.py`)
- 读取效率日志并生成走势图
- 输出统计摘要和最终报告
- 支持导出为文本文件

---

## 🚀 快速开始

### 最简单的用法（一行命令）

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
  --output-dir outputs_test
```

**就这样！** 该命令会：
1. ✅ 启动 Pipeline
2. ✅ 自动启动效率监控
3. ✅ 每 10 秒采样一次效率指标
4. ✅ 实时输出效率数据
5. ✅ Pipeline 完成后自动生成报告

---

## 📊 输出示例

### 实时终端输出

```
[Efficiency Monitor] Starting efficiency monitor (interval: 10s)
[Efficiency Monitor] Monitor process started (PID: 12345)
...
[Efficiency Monitor] GPU利用率: 33.3% | GPU效率: 30.7% | Speedup: 1.05x | 完成任务: 2
[Efficiency Monitor] GPU利用率: 35.2% | GPU效率: 32.1% | Speedup: 1.08x | 完成任务: 3
[Efficiency Monitor] GPU利用率: 40.5% | GPU效率: 35.8% | Speedup: 1.12x | 完成任务: 4
...
[Efficiency Monitor] Monitor process stopped
[Efficiency Monitor] Report saved to: outputs_test/efficiency_log_*_report.txt
```

### 生成的文件

```
outputs_test/
├── pipeline_trace_20260428_xxxxxx.log        # Pipeline trace
├── efficiency_log_20260428_xxxxxx.csv        # 实时效率数据（CSV）
├── efficiency_log_20260428_xxxxxx_report.txt # 最终效率报告
├── best_arch.json                            # 最佳架构
└── leaderboard.csv                           # 排行榜
```

### CSV 数据格式

```csv
timestamp,elapsed_time_s,wall_time,avg_concurrent_gpus,gpu_util_ratio,gpu_efficiency,...
2026-04-28T19:50:30.123456,10.2,8.5,0.8,0.267,0.213,...
2026-04-28T19:50:40.234567,20.5,18.3,1.2,0.4,0.32,...
```

---

## 🎯 主要功能

### 1. 实时监控效率指标
- **GPU 利用率**: 平均有多少个 GPU 在同时工作
- **GPU 效率**: 实际 GPU 工作时间 / 理想值
- **Pipeline Speedup**: 流水线加速比
- **Trial 吞吐量**: 每秒完成多少个 trial
- **Stage 利用率**: 每个 stage 的工作效率

### 2. 自动化工作流
- ✅ 无需手动启动监控脚本
- ✅ 无需等待 Pipeline 完成后再分析
- ✅ 无需手动整合各种脚本

### 3. 对比分析
```bash
# 快速对比两个配置的效率
python visualize_efficiency_log.py outputs_config_a/efficiency_log_*.csv
python visualize_efficiency_log.py outputs_config_b/efficiency_log_*.csv
```

### 4. 参数调优引导
- 直观看到参数改变对效率的影响
- 实时反馈优化效果
- 数据驱动的调参决策

---

## 💡 使用场景

### 场景 1：快速诊断问题
问题：GPU 利用率很低，不知道为什么
解决：启用监控，看实时数据
```bash
--enable-efficiency-monitor --efficiency-monitor-interval 5
```

### 场景 2：参数优化
问题：想知道增加 worker 数是否真的有帮助
解决：对比两个配置的效率日志
```bash
# 配置 A：baseline
python search.py --pipeline-stage-train-workers 2,1 --enable-efficiency-monitor

# 配置 B：优化
python search.py --pipeline-stage-train-workers 4,3 --enable-efficiency-monitor

# 对比
python visualize_efficiency_log.py outputs_a/efficiency_log_*.csv
python visualize_efficiency_log.py outputs_b/efficiency_log_*.csv
```

### 场景 3：生产环境监控
问题：大规模搜索要跑 12+ 小时，想看进展
解决：启用监控，偶尔查看效率报告
```bash
--enable-efficiency-monitor --efficiency-monitor-interval 30
```

---

## 📋 参数说明

### `--enable-efficiency-monitor`
- **类型**: 布尔标志
- **必需**: 否（默认不启用）
- **说明**: 启用实时效率监控
- **依赖**: 必须同时使用 `--pipeline-trace` 和 `--execution-mode ray_pipeline`

### `--efficiency-monitor-interval`
- **类型**: 整数（秒）
- **默认值**: 10
- **范围**: 建议 5-60 秒
- **说明**: 效率监控的采样间隔
  - 5-10 秒：适合调试，数据精细但略微增加开销
  - 15-30 秒：适合中等规模，平衡精度和开销
  - 30-60 秒：适合大规模，降低开销

---

## 🔧 配置建议

### 快速测试（开发调试）
```bash
--coarse-trials 3
--enable-efficiency-monitor
--efficiency-monitor-interval 5
```
效果：快速看到效率趋势，5 分钟内完成

### 常规测试（参数调优）
```bash
--coarse-trials 6-12
--enable-efficiency-monitor
--efficiency-monitor-interval 10
```
效果：完整的效率数据，20-40 分钟完成

### 生产运行（大规模搜索）
```bash
--coarse-trials 50-100
--enable-efficiency-monitor
--efficiency-monitor-interval 30
```
效果：最小化开销，几小时内完成

---

## 📈 分析工作流

### Step 1: 运行 Pipeline 并启用监控
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python search.py \
  ... \
  --enable-efficiency-monitor \
  --output-dir outputs_test
```

### Step 2: 实时查看进度（可选）
在另一个终端：
```bash
watch -n 5 'tail -10 outputs_test/efficiency_log_*.csv'
```

### Step 3: Pipeline 完成后查看报告
```bash
cat outputs_test/efficiency_log_*_report.txt
```

### Step 4: 深度分析
```bash
python visualize_efficiency_log.py outputs_test/efficiency_log_*.csv --limit 20
```

### Step 5: 导出数据（用于进一步分析）
```bash
python visualize_efficiency_log.py outputs_test/efficiency_log_*.csv --export summary.txt
```

---

## 🎓 理解效率指标

### GPU 利用率 (gpu_util_ratio)
- 范围: 0-1
- 含义: 平均有多少个 GPU 在工作（相对于 3 个 GPU）
- 目标: 尽可能接近 1（理想情况）
- 改进方向:
  - 增加 `--architectures-per-step`
  - 增加 `--num-pipeline-stages`
  - 增加 `--pipeline-stage-train-workers`

### Pipeline Speedup
- 范围: 1.0 ~ trial_数量
- 含义: 流水线相比顺序执行的加速倍数
- 目标: 越高越好（接近 trial 数量最理想）
- 改进方向:
  - 确保 stage 有时间重叠
  - 增加流水线深度（stage 数）

### Speedup 效率
- 范围: 0-1
- 含义: 实际加速 / 理想加速
- 目标: 接近 1（100%）表示完美利用
- 改进方向:
  - 同 Pipeline Speedup

---

## 💻 集成到自动化脚本

```bash
#!/bin/bash

DATASET="public_csv"
DATA_PATH="data/public/mooc.csv"
TRIALS=6
WORKERS="3,2"

for INTERVAL in 5 10 15; do
    echo "Testing with monitor interval: $INTERVAL seconds"
    
    CUDA_VISIBLE_DEVICES=0,1,2 python search.py \
      --dataset $DATASET \
      --local-data-path $DATA_PATH \
      --search-mode rl \
      --execution-mode ray_pipeline \
      --coarse-trials $TRIALS \
      --architectures-per-step 6 \
      --pipeline-stage-train-workers $WORKERS \
      --pipeline-trace \
      --enable-efficiency-monitor \
      --efficiency-monitor-interval $INTERVAL \
      --output-dir outputs_interval_${INTERVAL}
    
    echo "Results saved to outputs_interval_${INTERVAL}/"
done

echo "All tests completed! Compare results:"
for dir in outputs_interval_*; do
    echo ""
    echo "=== $dir ==="
    tail -5 $dir/efficiency_log_*.csv
done
```

---

## 🐛 故障排查

### 问题: 监控不启动
```
[Efficiency Monitor] ⚠️ Failed to start monitor: No such file or directory
```
解决:
- 确保 `monitor_pipeline_efficiency.py` 在当前目录
- 确保 Python 路径正确

### 问题: 效率指标都是 0
```
GPU利用率: 0.0% | GPU效率: 0.0%
```
解决:
- 等待足够的时间（30 秒+）
- 确保 Pipeline 真的在运行

### 问题: CSV 文件为空
解决:
- 检查 trace 日志是否有数据
- 等待更长时间再查看

### 问题: 监控占用 CPU 过高
解决:
- 增加采样间隔: `--efficiency-monitor-interval 30`

---

## 📚 相关文档

- [EFFICIENCY_MONITOR_USAGE.md](EFFICIENCY_MONITOR_USAGE.md) - 详细使用指南
- [MONITORING_GUIDE.md](MONITORING_GUIDE.md) - 完整工作流指南
- [METRICS_EXPLANATION.md](METRICS_EXPLANATION.md) - 指标计算方法说明

---

## ✨ 总结

这个系统使得 Pipeline 效率优化变得 **简单、直观、数据驱动**：

| 特性 | 优势 |
|------|------|
| 🔄 **集成参数** | 一行命令启用，无需额外脚本 |
| 📊 **实时数据** | 训练中即时查看效率指标 |
| 📈 **自动报告** | Pipeline 完成自动生成报告 |
| 🎯 **对比分析** | 轻松对比不同配置的效率 |
| 🛠️ **数据驱动** | 基于实际数据做出优化决策 |

**立即开始使用：** `--enable-efficiency-monitor`

