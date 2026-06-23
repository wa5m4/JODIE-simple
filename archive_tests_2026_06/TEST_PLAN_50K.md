# 50K Events 测试计划

**测试日期：** 2026-06-20  
**目的：** 在更大规模数据上验证各策略的性能和准确性

---

## 测试配置

| 参数 | 值 |
|------|---|
| 数据量 | 50000 events |
| Trials | 50 |
| Epochs | 3 |
| GPU配置 | 0,1,2 (3 GPUs) |
| 数据集 | MOOC (public_csv) |
| 搜索空间 | rnn_only |
| Batch模式 | tbatch |
| Seed | 42 |

---

## 策略列表

1. **Serial** - 串行基准
2. **数据并行改进** - Micro-batch级数据并行
3. **Smart Pipeline + 20%预热** - 成本优化的流水线
4. **Naive Pipeline** - 均匀划分的流水线

---

## 1. Serial策略

### 描述
- 单机串行训练
- 无分区（partition_size=0）
- 作为准确率基准

### 完整命令

```bash
#!/bin/bash

SEED=42
MAX_EVENTS=50000
COARSE_TRIALS=50
COARSE_EPOCHS=3
OUTPUT_DIR="outputs/50k_comparison/seed_42/serial"

mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode serial \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events $MAX_EVENTS \
    --seed $SEED \
    --coarse-trials $COARSE_TRIALS \
    --coarse-epochs $COARSE_EPOCHS \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false \
    2>&1 | tee "${OUTPUT_DIR}.log"
```

### 关键参数说明
- `--execution-mode serial`: 串行模式
- `--max-events 50000`: 使用50K events
- `--coarse-trials 50`: 搜索50个架构
- `--coarse-epochs 3`: 每个架构训练3个epochs

### 预期
- **Test MRR**: ~0.85 (基于20K经验)
- **时间**: ~300-400秒 (50K比20K约2.5倍)
- **架构**: off/off (预期)

---

## 2. 数据并行改进策略

### 描述
- Micro-batch级数据并行
- Partition级串行（保持时序）
- Micro-batch级并行（多GPU加速）
- 3个workers并行处理micro-batches

### 完整命令

```bash
#!/bin/bash

SEED=42
MAX_EVENTS=50000
COARSE_TRIALS=50
COARSE_EPOCHS=3
OUTPUT_DIR="outputs/50k_comparison/seed_42/data_parallel_improved"

mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode data_parallel \
    --data-parallel-workers 3 \
    --gpu-list "0,1,2" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events $MAX_EVENTS \
    --seed $SEED \
    --coarse-trials $COARSE_TRIALS \
    --coarse-epochs $COARSE_EPOCHS \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    --partition-size 7500 \
    --partition-strategy count \
    --data-parallel-micro-batch-size 32 \
    2>&1 | tee "${OUTPUT_DIR}.log"
```

### 关键参数说明
- `--execution-mode data_parallel`: 数据并行模式
- `--data-parallel-workers 3`: 3个Ray workers
- `--gpu-list "0,1,2"`: 使用GPU 0,1,2
- `--partition-size 7500`: 每个partition约7500 events (创建6-7个partitions)
- `--data-parallel-micro-batch-size 32`: Micro-batch大小32

### 核心机制
```
Partition级串行:
  partition 0 → partition 1 → partition 2 → ... (保持时序)

Micro-batch级并行:
  在每个partition内:
    Worker0: batch[0:32]   }
    Worker1: batch[32:64]  } 并行
    Worker2: batch[64:96]  }
    → 梯度平均 → 更新模型
```

### 预期
- **Test MRR**: ~0.85 (与Serial一致)
- **时间**: 接近Serial (micro-batch并行加速有限)
- **架构**: off/off (应该正确)
- **多GPU利用率**: 高 (3个GPUs同时工作)


---

## 3. Smart Pipeline + 20%预热策略

### 描述
- 成本优化的Pipeline并行
- 按partition成本智能分组stages
- 20%数据预热（overlap_ratio=0.2）
- 3个stages并行执行不同trials
- 100%准确率（20K验证）

### 完整命令

```bash
#!/bin/bash

SEED=42
MAX_EVENTS=50000
COARSE_TRIALS=50
COARSE_EPOCHS=3
OUTPUT_DIR="outputs/50k_comparison/seed_42/smart_overlap20"

mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode smart \
    --pipeline-stage-train-workers 3 \
    --gpu-list "0,1,2" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events $MAX_EVENTS \
    --seed $SEED \
    --coarse-trials $COARSE_TRIALS \
    --coarse-epochs $COARSE_EPOCHS \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    --partition-size 12500 \
    --partition-strategy count \
    --partition-overlap-ratio 0.2 \
    --stage-balance-strategy cost \
    2>&1 | tee "${OUTPUT_DIR}.log"
```

### 关键参数说明
- `--execution-mode ray_pipeline`: Pipeline模式
- `--pipeline-mode smart`: 智能成本优化分stage
- `--pipeline-stage-train-workers 3`: 3个pipeline stages
- `--partition-size 12500`: 每个partition约12500 events (创建4个partitions)
- `--partition-overlap-ratio 0.2`: 20%预热数据重叠
- `--stage-balance-strategy cost`: 按成本平衡stages

### 核心机制
```
Stage划分（成本优化）:
  Stage 0: partitions with high cost (many new users/items)
  Stage 1: partitions with medium cost
  Stage 2: partitions with low cost (mostly existing users/items)

流水线并行:
  Trial A: Stage 0 → Stage 1 → Stage 2
  Trial B:          Stage 0 → Stage 1 → Stage 2
  Trial C:                   Stage 0 → Stage 1 → ...
  
  3个trials同时在不同stages执行
```

### 预期
- **Test MRR**: ~0.85 (与Serial一致，20K验证100%准确)
- **时间**: ~120-150秒 (预期3倍加速)
- **架构**: off/off (应该正确)
- **加速比**: 3.0× vs Serial


---

## 4. Naive Pipeline策略

### 描述
- 均匀划分的Pipeline并行
- Partitions均匀分配到stages
- 无预热（overlap_ratio=0）
- 3个stages并行执行不同trials
- 67%准确率（20K验证，seed依赖）

### 完整命令

```bash
#!/bin/bash

SEED=42
MAX_EVENTS=50000
COARSE_TRIALS=50
COARSE_EPOCHS=3
OUTPUT_DIR="outputs/50k_comparison/seed_42/naive_no_overlap"

mkdir -p "$OUTPUT_DIR"

python search.py \
    --search-mode rl \
    --execution-mode ray_pipeline \
    --pipeline-mode naive \
    --pipeline-stage-train-workers 3 \
    --gpu-list "0,1,2" \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events $MAX_EVENTS \
    --seed $SEED \
    --coarse-trials $COARSE_TRIALS \
    --coarse-epochs $COARSE_EPOCHS \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    --partition-size 12500 \
    --partition-strategy count \
    --partition-overlap-ratio 0.0 \
    2>&1 | tee "${OUTPUT_DIR}.log"
```

### 关键参数说明
- `--execution-mode ray_pipeline`: Pipeline模式
- `--pipeline-mode naive`: 均匀划分stages
- `--pipeline-stage-train-workers 3`: 3个pipeline stages
- `--partition-size 12500`: 每个partition约12500 events (创建4个partitions)
- `--partition-overlap-ratio 0.0`: 无预热

### 核心机制
```
Stage划分（均匀）:
  4个partitions均匀分配到3个stages:
  Stage 0: partition 0
  Stage 1: partition 1
  Stage 2: partitions 2, 3

流水线并行:
  与Smart相同的流水线执行
  但stage负载可能不均衡
```

### 预期
- **Test MRR**: ~0.85 或更低 (67%准确率，可能选错架构)
- **时间**: ~200-250秒 (加速比1.5-2.0×)
- **架构**: off/off 或其他 (取决于seed)
- **加速比**: 1.5-2.0× vs Serial

---

## 测试执行计划

### 执行顺序
1. **Serial** (基准) - 先运行，建立准确率基准
2. **数据并行改进** - 验证多GPU数据并行
3. **Smart Pipeline + 20%** - 验证最优Pipeline方案
4. **Naive Pipeline** - 对比基准Pipeline

### 脚本生成

所有测试脚本保存在 `tests_50k/` 目录：

```bash
mkdir -p tests_50k
cd tests_50k

# 从TEST_PLAN_50K.md提取命令创建独立脚本
# test_serial.sh
# test_data_parallel.sh
# test_smart_pipeline.sh
# test_naive_pipeline.sh
```

### 结果对比

测试完成后，创建对比表格：

| 策略 | 架构 | Test MRR | 时间(s) | 加速比 | GPU利用率 |
|------|------|----------|---------|--------|----------|
| Serial | ? | ? | ? | 1.0× | 低 |
| 数据并行改进 | ? | ? | ? | ~1.0× | 高 |
| Smart+20% | ? | ? | ? | ~3.0× | 高 |
| Naive | ? | ? | ? | ~1.5× | 高 |

### 验证目标

1. ✅ **数据并行准确性**: Test MRR = Serial ± 0.001
2. ✅ **Smart Pipeline稳定性**: 架构选择正确（off/off）
3. ✅ **加速效果**: Smart达到3倍加速
4. ✅ **大规模可扩展性**: 50K数据下所有策略正常运行

---

## Retrain计划

所有策略完成后，使用统一的final test进行retrain：

```bash
# 对每个策略的best架构进行retrain
for strategy in serial data_parallel_improved smart_overlap20 naive_no_overlap; do
    python retrain_final_test.py \
        --best-arch-json outputs/50k_comparison/seed_42/${strategy}/best_arch.json \
        --output-dir outputs/50k_comparison/seed_42/${strategy}/retrain \
        --seed 42 \
        --epochs 3
done
```

确保所有选出相同架构的策略，retrain后Test MRR完全一致。

---

**文档创建时间**: 2026-06-20  
**预计总测试时间**: 约1.5-2小时  
**输出路径**: `outputs/50k_comparison/seed_42/`

