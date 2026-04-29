# 项目结构指南

## 📁 项目文件夹组织

```
JODIE-simple/
├── 📄 README.md                 # 项目主文档
├── 🐍 search.py                 # 主程序：图神经网络架构搜索入口
├── 
├── 📚 docs/                     # 📖 文档（所有说明文档集中在这里）
│   ├── USAGE.md                 # 基本使用指南
│   ├── DEPLOYMENT_COMPLETE.md   # 实时效率监控部署说明
│   ├── EFFICIENCY_MONITOR_SUMMARY.md      # 效率监控功能总结
│   ├── EFFICIENCY_MONITOR_USAGE.md        # 效率监控详细使用指南
│   ├── METRICS_EXPLANATION.md   # 效率指标计算方法说明
│   └── MONITORING_GUIDE.md      # 完整监控工作流程
│
├── 🛠️  tools/                    # 🔧 工具脚本（所有 Python 分析工具）
│   ├── monitor_pipeline_efficiency.py     # 实时效率监控
│   ├── visualize_efficiency_log.py        # 效率数据可视化和报告生成
│   ├── analyze_pipeline_efficiency.py     # Pipeline 效率分析
│   ├── visualize_pipeline_metrics.py      # Pipeline 指标可视化
│   └── compare_public_dataset.py          # 公开数据集对比分析
│
├── 🧪 scripts/                   # 🧪 测试脚本（所有 Shell 测试脚本）
│   ├── test_single_gpu.sh        # 单 GPU 效率测试
│   ├── test_multi_gpu.sh         # 多 GPU 效率对比测试
│   ├── test_efficiency_monitor.sh # 效率监控功能验证
│   ├── compare_efficiency.sh     # 效率对比分析脚本
│   └── run_pipeline_with_monitoring.sh    # Pipeline + 监控集成脚本
│
├── 📊 data/                      # 数据文件（不动）
│   ├── public/
│   │   ├── mooc.csv
│   │   └── tiny_jodie.csv
│   ├── public_dataset.py
│   ├── synthetic.py
│   └── temporal_partition.py
│
├── 🧠 models/                    # 模型实现（不动）
│   ├── factory.py
│   ├── gnn_encoder.py
│   ├── hybrid_jodie.py
│   ├── jodie_rnn.py
│   └── training.py
│
├── 🔍 nas/                       # NAS 框架核心（不动）
│   ├── controller.py
│   ├── ray_pipeline.py
│   ├── search_space.py
│   └── trainer.py
│
├── 📈 baselines/                 # 基准实现（不动）
│   └── official_jodie_adapter.py
│
└── 📁 outputs/                   # 输出目录（运行结果）
    └── ... 各种实验输出目录
```

---

## 🎯 文件说明

### 📚 docs/ - 文档目录

| 文件 | 用途 |
|------|------|
| `USAGE.md` | 基本使用指南 |
| `DEPLOYMENT_COMPLETE.md` | 实时效率监控系统部署说明 |
| `EFFICIENCY_MONITOR_SUMMARY.md` | 效率监控功能概览 |
| `EFFICIENCY_MONITOR_USAGE.md` | 效率监控详细使用方法和参数说明 |
| `METRICS_EXPLANATION.md` | GPU 利用率、效率、Speedup 等指标的计算方法 |
| `MONITORING_GUIDE.md` | 完整的监控工作流程和最佳实践 |

### 🛠️ tools/ - 工具脚本

| 文件 | 用途 |
|------|------|
| `monitor_pipeline_efficiency.py` | **实时监控工具** - 在 Pipeline 训练中持续采集效率指标 |
| `visualize_efficiency_log.py` | **报告生成工具** - 分析效率日志并生成报告 |
| `analyze_pipeline_efficiency.py` | Pipeline trace 分析工具 |
| `visualize_pipeline_metrics.py` | Pipeline 指标可视化工具 |
| `compare_public_dataset.py` | 公开数据集对比分析工具 |

### 🧪 scripts/ - 测试脚本

| 脚本 | 用途 |
|------|------|
| `test_single_gpu.sh` | **单 GPU 基准测试** - 效率应接近 100% |
| `test_multi_gpu.sh` | **多 GPU 对比测试** - 用于效率对比 |
| `compare_efficiency.sh` | **对比分析脚本** - 自动分析两次测试的效率数据 |
| `test_efficiency_monitor.sh` | 监控功能验证脚本 |
| `run_pipeline_with_monitoring.sh` | Pipeline + 实时监控集成脚本 |

---

## 🚀 快速开始

### 1. 查看文档
```bash
# 基本使用
cat docs/USAGE.md

# 效率监控详细使用
cat docs/EFFICIENCY_MONITOR_USAGE.md

# 指标说明
cat docs/METRICS_EXPLANATION.md
```

### 2. 运行测试
```bash
# 单 GPU 效率测试（预期 ~100%）
bash scripts/test_single_gpu.sh

# 多 GPU 效率测试（用于对比）
bash scripts/test_multi_gpu.sh

# 对比分析
bash scripts/compare_efficiency.sh
```

### 3. 启动主程序
```bash
# 标准模式
python search.py --dataset public_csv --local-data-path data/public/mooc.csv ...

# 启用实时效率监控
python search.py --dataset public_csv --local-data-path data/public/mooc.csv \
  --enable-efficiency-monitor --efficiency-monitor-interval 10 ...
```

---

## 📝 文件整理规则

整理后的目录按以下方式组织：

- **docs/** - 所有文档（.md 文件）
- **tools/** - 所有 Python 工具和分析脚本
- **scripts/** - 所有 Shell 测试脚本
- **根目录** - 只保留核心程序和配置

---

## 💡 使用建议

### 学习路径
1. 先阅读 `docs/USAGE.md` 了解基本使用
2. 查看 `docs/DEPLOYMENT_COMPLETE.md` 了解效率监控功能
3. 参考 `docs/EFFICIENCY_MONITOR_USAGE.md` 学习具体使用方法
4. 查阅 `docs/METRICS_EXPLANATION.md` 理解各指标含义

### 实验流程
1. 运行 `scripts/test_single_gpu.sh` 建立基准
2. 运行 `scripts/test_multi_gpu.sh` 进行对比
3. 执行 `scripts/compare_efficiency.sh` 分析结果
4. 根据结果调整参数，优化配置

### 生产使用
```bash
# 启用效率监控的标准命令
CUDA_VISIBLE_DEVICES=0,1,2 python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --search-mode rl \
  --execution-mode ray_pipeline \
  --coarse-trials 6 \
  --pipeline-trace \
  --enable-efficiency-monitor \
  --efficiency-monitor-interval 10 \
  --output-dir outputs_demo
```

---

## 🔄 路径更新说明

由于文件整理后目录结构改变，以下内容已自动更新：

- ✅ `nas/trainer.py` - Python 脚本路径已指向 `tools/`
- ✅ `scripts/test_efficiency_monitor.sh` - 脚本路径已更新
- ✅ `scripts/run_pipeline_with_monitoring.sh` - 脚本路径已更新

所有脚本和程序都可以直接使用，无需手动调整。

---

**整理完成！项目现在结构更清晰。** ✨
