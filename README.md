# JODIE-simple

时序图神经网络架构搜索 (NAS) 框架 —— 为 JODIE 风格的时序动态图模型搜索最优架构。

## 概述

JODIE-simple 通过神经架构搜索（NAS）为事件级时序 GNN 自动发现最佳超参数组合。项目支持三种执行模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `serial` | 单机串行搜索 | 开发调试、小规模实验 |
| `data_parallel` | Ray 数据并行 | 单机多 GPU 加速 |
| `ray_pipeline` | Ray 流水线并行 | 多 GPU 流水线，最大化吞吐 |

## 项目结构

```
jodie-simple/
├── jodie/                    # 核心包
│   ├── data/                 # 数据加载与分区
│   ├── models/               # GNN 模型定义
│   ├── training/             # 训练循环与评估指标
│   ├── nas/                  # NAS 框架（搜索空间、控制器、训练器）
│   └── baseline/             # 官方 JODIE 基线适配器
├── search.py                 # NAS 搜索入口
├── train.py                  # 单架构训练入口
├── data/public/              # 数据集文件
└── docs/                     # 详细文档
    ├── ARCHITECTURE.md       # 架构说明（逐文件逐函数）
    ├── ISSUES.md             # 问题记录（bug、重复代码、代码坏味）
    └── REFACTORING_GUIDE.md  # 重构指南
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

依赖：`torch>=1.10.0`, `numpy>=1.21.0`, `ray>=2.0.0`

### 冒烟测试

```bash
python search.py --space small --execution-mode serial --trials 2 --epochs-per-trial 1
```

### 在真实数据上搜索

```bash
python search.py \
    --space rnn_only \
    --execution-mode serial \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --coarse-trials 32 \
    --coarse-epochs 4 \
    --rerank-top-k 8 \
    --output-dir outputs/my_search
```

### 使用 Ray 流水线并行

```bash
python search.py \
    --space rnn_only \
    --execution-mode ray_pipeline \
    --pipeline-mode smart \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --coarse-trials 32 \
    --coarse-epochs 4 \
    --num-pipeline-stages 2 \
    --gpu-list 0,1,2 \
    --enable-auto-pipeline-config \
    --output-dir outputs/pipeline_search
```

### 训练单个架构

```bash
python train.py \
    --model jodie_rnn \
    --embedding-dim 32 \
    --memory-cell gru \
    --local-data-path data/public/mooc.csv \
    --epochs 5 \
    --output-dir outputs/single_run
```

## 搜索空间

| 空间名 | 模型 | 搜索维度 |
|--------|------|----------|
| `small` | TemporalEventGNNJODIE | 16 维 |
| `paper_compare` | TemporalEventGNNJODIE | 受限子集 |
| `rnn_only` | JODIERNN | 6 维 |
| `mixed` | 两者混合 | 17 维 |

## 输出

每次搜索在 `--output-dir` 下生成：
- `best_arch.json` — 最佳架构配置
- `leaderboard.csv` — 所有评估架构的排名
- `pipeline_trace_*.log` — 流水线追踪（仅 pipeline 模式）

## 核心模型

### TemporalEventGNNJODIE (HybridJODIE)
事件级时序 GNN，结合图邻域聚合与循环记忆更新。可搜索的维度包括：
- 聚合函数 (mean/sum/attention)
- 时间衰减 (none/exp/inverse)
- 记忆单元 (RNN/GRU/LSTM/add)
- 时间投影 (linear/MLP/off)

### JODIERNN
经典 JODIE 互递归动态嵌入，用户和物品的投影嵌入直接作为彼此的输入。

## 文档

- [ARCHITECTURE.md](ARCHITECTURE.md) — 逐目录、逐文件、逐函数的详细架构说明
- [ISSUES.md](ISSUES.md) — 已知 bug、代码重复、代码坏味、架构问题
- [REFACTORING_GUIDE.md](../jodie-simple/REFACTORING_GUIDE.md) — 重构指南（在原项目中）
