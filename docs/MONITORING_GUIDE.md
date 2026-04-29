# Pipeline 实时效率监控系统

本系统可以在 Pipeline 训练过程中**实时记录**效率参数，并生成可视化报告。

## 📋 系统组成

### 1. 监控脚本 (`monitor_pipeline_efficiency.py`)
- **功能**：持续监控 trace 日志，每隔 N 秒计算一次效率指标
- **输出**：CSV 格式的效率日志（`efficiency_log_*.csv`）
- **指标**：GPU 利用率、GPU 效率、Pipeline Speedup 等

### 2. 可视化脚本 (`visualize_efficiency_log.py`)
- **功能**：读取效率日志，生成走势图和统计摘要
- **输出**：终端输出或导出 txt 文件

### 3. 一键启动脚本 (`run_pipeline_with_monitoring.sh`)
- **功能**：同时启动 Pipeline 和监控器，自动生成报告

---

## 🚀 快速开始

### 方案 A：一键启动（推荐）

```bash
bash run_pipeline_with_monitoring.sh outputs_test_v2 10
```

参数说明：
- `outputs_test_v2`：输出目录
- `10`：监控间隔（秒）

这个脚本会：
1. ✅ 启动 Pipeline（后台运行）
2. ✅ 自动等待 trace 文件生成
3. ✅ 启动效率监控（每 10 秒采样一次）
4. ✅ Pipeline 完成后自动生成可视化报告

### 方案 B：手动启动（三个终端）

**终端 1：启动 Pipeline**
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
  --output-dir outputs_test_v2
```

**终端 2：启动监控**
```bash
# 等待 trace 文件生成后，找到文件路径
python monitor_pipeline_efficiency.py outputs_test_v2/pipeline_trace_*.log 10
```

**终端 3：实时查看效率日志（可选）**
```bash
# 每 5 秒刷新一次
watch -n 5 'python visualize_efficiency_log.py outputs_test_v2/efficiency_log_*.csv --limit 10'
```

---

## 📊 输出文件说明

### Pipeline 输出
```
outputs_test_v2/
├── pipeline_trace_20260428_xxxxxx.log      # Ray pipeline trace
├── efficiency_log_20260428_xxxxxx.csv      # 🆕 实时效率指标日志
├── efficiency_summary.txt                   # 🆕 效率摘要报告
└── ... 其他输出文件
```

### efficiency_log_*.csv 格式

```csv
timestamp,elapsed_time_s,wall_time,avg_concurrent_gpus,gpu_util_ratio,gpu_efficiency,...
2026-04-28T19:50:30.123456,10.2,8.5,0.8,0.267,0.213,...
2026-04-28T19:50:40.234567,20.5,18.3,1.2,0.4,0.32,...
...
```

列说明：
- **timestamp**: 采样时刻
- **elapsed_time_s**: 从 Pipeline 开始到现在的耗时
- **wall_time**: 从第一个 dispatch 到现在的时间跨度
- **avg_concurrent_gpus**: 平均并发 GPU actors 数
- **gpu_util_ratio**: GPU 利用率 (0-1)
- **gpu_efficiency**: GPU 效率 (0-1)
- **avg_stage_util**: Stage 平均利用率
- **max_stage_util**: Stage 最大利用率
- **trial_throughput**: Trial 吞吐量 (trials/sec)
- **pipeline_speedup**: Pipeline 加速比
- **speedup_efficiency**: Speedup 效率
- **num_completed_tasks**: 已完成的任务数

---

## 📈 监控指标详解

### GPU 利用率 (gpu_util_ratio)
- **范围**: 0-1（理想值接近 1）
- **含义**: 平均有多少个 GPU 在同时工作（相对于 3 个 GPU 总数）
- **0.33** 表示平均只有 1 个 GPU 在工作
- **1.0** 表示理想情况，3 个 GPU 全满负荷

### GPU 效率 (gpu_efficiency)
- **范围**: 0-1
- **含义**: 实际 GPU 工作时间 / 理想 GPU 工作时间
- **低值**表示大量 GPU 空闲时间

### Pipeline Speedup (pipeline_speedup)
- **理想值**: 接近 trial 数
- **例子**: 6 个 trial 通过流水线，理想 speedup 应接近 2-3x
- **低值**表示流水线中没有有效重叠

### Speedup 效率 (speedup_efficiency)
- **范围**: 0-1（理想值接近 1）
- **计算**: speedup / num_completed_trials
- **高值**表示流水线利用率高

---

## 🔧 自定义配置

### 修改监控间隔
```bash
# 每 5 秒采样一次（更频繁）
python monitor_pipeline_efficiency.py trace_file.log 5

# 每 30 秒采样一次（更疏稀）
python monitor_pipeline_efficiency.py trace_file.log 30
```

### 修改启动脚本的 Pipeline 参数

编辑 `run_pipeline_with_monitoring.sh`，修改 `python search.py` 部分的参数：

```bash
--architectures-per-step 10       # 增加试验并发
--num-pipeline-stages 4            # 增加流水线深度
--pipeline-stage-train-workers 4,3,2  # 增加 worker 数
```

---

## 📊 示例：对比不同配置

### 配置 1：基准配置
```bash
bash run_pipeline_with_monitoring.sh outputs_config_baseline 10
```

### 配置 2：高并发配置
修改脚本，改为：
- `--architectures-per-step 10`
- `--num-pipeline-stages 4`
- `--pipeline-stage-train-workers 4,3,2`

```bash
bash run_pipeline_with_monitoring.sh outputs_config_highconcurrency 10
```

### 对比效果
```bash
echo "=== 配置 1（基准）==="
python visualize_efficiency_log.py outputs_config_baseline/efficiency_log_*.csv

echo ""
echo "=== 配置 2（高并发）==="
python visualize_efficiency_log.py outputs_config_highconcurrency/efficiency_log_*.csv
```

---

## 💡 常见问题

### Q1: Trace 文件何时生成？
A: 在 Pipeline 开始处理第一个任务时。通常需要等待几秒钟。

### Q2: 效率指标何时开始有数据？
A: 当至少有一个 dispatch 和一个 complete 事件时。通常 10-20 秒后。

### Q3: 如何在运行中查看最新效率？
A: 使用 `watch` 命令定期刷新：
```bash
watch -n 5 'python visualize_efficiency_log.py outputs_test/efficiency_log_*.csv --limit 5'
```

### Q4: 效率指标何时稳定？
A: 通常在完成 3-5 个 trial 后效率指标会稳定。

### Q5: 能否在 Pipeline 还在运行时停止监控？
A: 可以，按 Ctrl+C 停止监控，Pipeline 会继续运行。

---

## 🎯 最佳实践

1. **基准测试**：先用默认配置跑一次，记录基准效率
2. **单一变量**：每次只改一个参数，看效果
3. **足够长的运行**：至少让 Pipeline 完成 6+ 个 trial，效率指标才能稳定
4. **对比分析**：保存多次运行的效率日志，对比改进
5. **监控间隔**：10-15 秒是比较合理的间隔（太频繁会有计算开销）

---

## 📋 完整工作流示例

```bash
# 1. 启动基准测试
bash run_pipeline_with_monitoring.sh outputs_baseline 10 &

# 2. 等待完成后（比如 30 分钟）
# 3. 启动改进版本测试
bash run_pipeline_with_monitoring.sh outputs_improved 10 &

# 4. 对比效果
python visualize_efficiency_log.py outputs_baseline/efficiency_log_*.csv --export baseline_report.txt
python visualize_efficiency_log.py outputs_improved/efficiency_log_*.csv --export improved_report.txt

# 5. 查看报告
cat baseline_report.txt
cat improved_report.txt
```

---

## 🐛 故障排查

### 监控脚本报错：FileNotFoundError
确保 trace 文件路径正确：
```bash
ls -la outputs_test/pipeline_trace_*.log
```

### 效率指标一直为 0
等待足够的时间让 Pipeline 生成事件（通常需要 20+ 秒）

### 监控脚本占用 CPU 过高
减少监控频率（增加 interval 参数）：
```bash
python monitor_pipeline_efficiency.py trace_file.log 30  # 改成 30 秒
```

---

**🎉 祝你调试愉快！**
