# Pipeline NAS 三因素完全交叉实验

## 实验目录结构

```
outputs/three_factor_test/
├── baseline/           # 基准组
│   ├── serial/         ✅ (已复制)
│   └── data_parallel/  ✅ (已复制)
├── async/              # 异步实验（smart模式）
│   ├── 1stage_overlap0/    ⏳ 需补充
│   ├── 1stage_overlap20/   ✅ 已复制 (smart_1stage)
│   ├── 2stage_overlap0/    ⏳ 需补充
│   ├── 2stage_overlap20/   ✅ 已复制 (smart_overlap20)
│   ├── 3stage_overlap0/    ⏳ 需补充
│   └── 3stage_overlap20/   ⏳ 需补充
└── sync/               # 同步实验（naive模式）
    ├── 1stage_overlap0/    ⏳ 需补充
    ├── 1stage_overlap20/   ⏳ 需补充
    ├── 2stage_overlap0/    ✅ 已复制 (naive_no_overlap)
    ├── 2stage_overlap20/   ⏳ 需补充
    ├── 3stage_overlap0/    ✅ 已复制 (naive_3stages)
    └── 3stage_overlap20/   ⏳ 需补充
```

## 三因素完全交叉设计

### 因素定义

**因素A - Stage数量**:
- 1 stage
- 2 stages
- 3 stages

**因素B - Overlap比例**:
- 0% (无overlap)
- 20% (有overlap)

**因素C - 架构生成模式**:
- 异步 (async): 边训边生成新架构，pipeline满载
- 同步 (sync): 批次执行，一批完成才生成下一批

### Worker分配策略 (3个GPU)

- 1 stage: `[3]` - 3个worker在stage 0
- 2 stages: `[3, 3]` - 每stage 3个worker
- 3 stages: `[1, 1, 1]` - 每stage 1个worker

## 完整实验矩阵

| ID | Stage | Overlap | 模式 | Worker | 状态 | 目录 |
|----|-------|---------|------|--------|------|------|
| **B1** | - | - | Serial | - | ✅ | baseline/serial |
| **B2** | - | - | DataParallel | - | ✅ | baseline/data_parallel |
| **A1** | 1 | 0% | 异步 | [3] | ⏳ | async/1stage_overlap0 |
| **A2** | 1 | 20% | 异步 | [3] | ✅ | async/1stage_overlap20 |
| **A3** | 2 | 0% | 异步 | [3,3] | ⏳ | async/2stage_overlap0 |
| **A4** | 2 | 20% | 异步 | [3,3] | ✅ | async/2stage_overlap20 |
| **A5** | 3 | 0% | 异步 | [1,1,1] | ⏳ | async/3stage_overlap0 |
| **A6** | 3 | 20% | 异步 | [1,1,1] | ⏳ | async/3stage_overlap20 |
| **S1** | 1 | 0% | 同步 | [3] | ⏳ | sync/1stage_overlap0 |
| **S2** | 1 | 20% | 同步 | [3] | ⏳ | sync/1stage_overlap20 |
| **S3** | 2 | 0% | 同步 | [3,3] | ✅ | sync/2stage_overlap0 |
| **S4** | 2 | 20% | 同步 | [3,3] | ⏳ | sync/2stage_overlap20 |
| **S5** | 3 | 0% | 同步 | [1,1,1] | ✅ | sync/3stage_overlap0 |
| **S6** | 3 | 20% | 同步 | [1,1,1] | ⏳ | sync/3stage_overlap20 |

## 已有数据 (4个实验)

从 `outputs/50k_comparison/seed_42/` 复制：

| 原路径 | 新路径 | 配置 |
|--------|--------|------|
| smart_1stage | async/1stage_overlap20 | 1s + 20% + 异步 |
| smart_overlap20 | async/2stage_overlap20 | 2s + 20% + 异步 |
| naive_no_overlap | sync/2stage_overlap0 | 2s + 0% + 同步 |
| naive_3stages | sync/3stage_overlap0 | 3s + 0% + 同步 |

## 需补充实验 (8个)

### 异步实验 (4个)
- A1: 1 stage + 0% overlap
- A3: 2 stages + 0% overlap
- A5: 3 stages + 0% overlap
- A6: 3 stages + 20% overlap

### 同步实验 (4个)
- S1: 1 stage + 0% overlap
- S2: 1 stage + 20% overlap
- S4: 2 stages + 20% overlap
- S6: 3 stages + 20% overlap

## 固定参数

- Dataset: MOOC (public_csv)
- Max events: 50,000
- Seed: 42
- Trials: 50
- Search space: rnn_only
- Search mode: RL
- Partition size: 12,500
- Coarse epochs: 1

## 实验命令参数

### 异步实验 (--pipeline-mode smart)
```bash
python search.py \
  --dataset public_csv \
  --max-events 50000 \
  --seed 42 \
  --space rnn_only \
  --coarse-trials 50 \
  --coarse-epochs 1 \
  --execution-mode ray_pipeline \
  --search-mode rl \
  --num-pipeline-stages {1|2|3} \
  --partition-size 12500 \
  --partition-overlap-ratio {0.0|0.2} \
  --pipeline-mode smart \
  --architectures-per-step 2 \
  --output-dir outputs/three_factor_test/async/{stage}stage_overlap{overlap}
```

### 同步实验 (--pipeline-mode naive)
```bash
python search.py \
  --dataset public_csv \
  --max-events 50000 \
  --seed 42 \
  --space rnn_only \
  --coarse-trials 50 \
  --coarse-epochs 1 \
  --execution-mode ray_pipeline \
  --search-mode rl \
  --num-pipeline-stages {1|2|3} \
  --partition-size 12500 \
  --partition-overlap-ratio {0.0|0.2} \
  --pipeline-mode naive \
  --architectures-per-step 2 \
  --output-dir outputs/three_factor_test/sync/{stage}stage_overlap{overlap}
```

## 预期研究问题

### Q1: Stage数量的主效应
固定其他因素，比较1/2/3 stages对架构选择准确性的影响。

### Q2: Overlap的主效应
固定其他因素，比较0%/20% overlap对架构选择准确性的影响。

### Q3: 异步/同步的主效应
固定其他因素，比较async/sync模式对架构选择准确性的影响。

### Q4: 因素交互效应
- Stage × Overlap
- Stage × 异步/同步
- Overlap × 异步/同步
- 三因素交互

## 评估指标

**主指标**：
- 选出的最佳架构 (time_proj/use_static_embeddings)
- 是否正确选出off/off

**诊断指标**：
- off/off架构的Val MRR (用于诊断评估偏差)
- 最佳架构的Test MRR (最终性能)

**成功标准**：
与Serial/Data Parallel基准一致，选出off/off架构。
