# NAS搜索和评估完整流程脚本

## 功能说明

`run_full_nas_pipeline.py` 是一个端到端的NAS搜索和评估脚本，自动完成以下流程：

1. **NAS搜索**：对四种执行模式进行架构搜索
   - Serial
   - Data Parallel
   - Pipeline Naive
   - Pipeline Smart

2. **实时日志**：搜索过程中显示实时进度

3. **自动重训练**：每个搜索完成后，用Serial T-Batch模式重训练最优架构

4. **综合报告**：生成包含时间、准确率等多维度对比的报告

## 使用方法

### 基本用法

```bash
python run_full_nas_pipeline.py \
  --gpu-list 0,1,2 \
  --max-events 20000 \
  --trials 27 \
  --epochs 3 \
  --seed 42
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu-list` | 0,1,2 | 使用的GPU列表 |
| `--data-path` | data/public/mooc.csv | 数据集路径 |
| `--max-events` | 20000 | 最大事件数 |
| `--trials` | 27 | NAS搜索的trials数量 |
| `--epochs` | 3 | 训练的epochs数量 |
| `--seed` | 42 | 随机种子 |
| `--output-dir` | outputs/full_pipeline | 输出目录 |
| `--modes` | all | 执行模式（all或逗号分隔的列表） |

### 运行特定模式

只运行部分执行模式：

```bash
# 只运行Serial和Data Parallel
python run_full_nas_pipeline.py --modes serial,data_parallel

# 只运行Pipeline模式
python run_full_nas_pipeline.py --modes pipeline_naive,pipeline_smart
```

## 输出结构

```
outputs/full_pipeline/
├── serial_tbatch/
│   ├── best_arch.json          # NAS搜索的最优架构
│   └── ...
├── serial_retrain/
│   └── result.json             # 重训练结果
├── data_parallel_tbatch/
│   └── ...
├── data_parallel_retrain/
│   └── ...
├── pipeline_naive_tbatch/
│   └── ...
├── pipeline_naive_retrain/
│   └── ...
├── pipeline_smart_tbatch/
│   └── ...
├── pipeline_smart_retrain/
│   └── ...
└── comprehensive_report.md     # 综合对比报告
```

## 示例

### 完整运行（默认参数）

```bash
python run_full_nas_pipeline.py
```

### 自定义参数

```bash
python run_full_nas_pipeline.py \
  --gpu-list 0,1,2,3 \
  --max-events 50000 \
  --trials 50 \
  --epochs 5 \
  --seed 123 \
  --output-dir outputs/my_experiment
```

### 快速测试（少量trials）

```bash
python run_full_nas_pipeline.py \
  --trials 9 \
  --epochs 2 \
  --max-events 10000
```

## 预计运行时间

基于MOOC数据集（20000事件，27 trials，3 epochs）：

- Serial搜索: ~30-40分钟
- Data Parallel搜索: ~15-20分钟
- Pipeline Naive搜索: ~20-30分钟
- Pipeline Smart搜索: ~15-20分钟
- 每个重训练: ~2-3分钟

**总计: 约90-120分钟**

## 注意事项

1. **GPU资源**：确保指定的GPU可用且有足够显存
2. **磁盘空间**：每个搜索会生成约100-500MB的输出文件
3. **实时日志**：使用`python -u`确保实时输出，可以用`tee`保存日志：
   ```bash
   python run_full_nas_pipeline.py 2>&1 | tee pipeline.log
   ```
4. **中断恢复**：如果中断，可以用`--modes`参数只运行未完成的模式

## 输出报告示例

综合报告包含：

- **结果汇总表**：所有模式的搜索时间、NAS性能、重训性能对比
- **详细分析**：每个模式的最优架构配置和性能差异
- **时间统计**：搜索时间、重训时间、总时间

## 故障排查

### 搜索失败

如果某个模式搜索失败，脚本会跳过该模式的重训练并继续其他模式。

### GPU内存不足

减少`--max-events`或使用更少的GPU。

### 日志不实时显示

确保使用`python -u`（脚本中已包含）。
