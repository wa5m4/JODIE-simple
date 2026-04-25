# JODIE + GraphNAS

这个项目用于搜索并训练事件级时序图模型（Temporal Event GNN + JODIE memory），支持以下数据源：

- `synthetic`：合成交互数据（快速验证）
- `wikipedia`：JODIE 原论文公开数据集
- `reddit`：JODIE 原论文公开数据集
- `public_csv`：本地 JODIE 格式 CSV，例如 `data/public/mooc.csv`

## 项目结构

```text
jodie-simple/
├── search.py
├── README.md
├── USAGE.md
├── data/
│   ├── synthetic.py
│   └── public_dataset.py
├── models/
│   ├── factory.py
│   ├── gnn_encoder.py
│   ├── hybrid_jodie.py
│   ├── jodie_rnn.py
│   └── training.py
└── nas/
    ├── controller.py
    ├── search_space.py
    └── trainer.py
```

## 快速开始

### 1) 合成数据快速验证

```bash
python search.py --dataset synthetic --search-mode random --trials 1 --epochs-per-trial 1 --num-users 20 --num-items 30 --num-interactions 80 --output-dir outputs_test
```

### 2) Wikipedia 训练（JODIE）

```bash
python search.py --dataset wikipedia --search-mode rl --trials 24 --epochs-per-trial 3 --k 10 --selection-metric mrr --output-dir outputs_wikipedia
```

### 3) Reddit 训练（JODIE）

```bash
python search.py --dataset reddit --search-mode rl --trials 24 --epochs-per-trial 3 --k 10 --selection-metric mrr --output-dir outputs_reddit
```

### 4) 本地 CSV 训练

```bash
python search.py --dataset public_csv --local-data-path data/public/mooc.csv --search-mode rl --trials 24 --epochs-per-trial 3 --k 10 --selection-metric mrr --output-dir outputs_csv
```

### 5) Ray pipeline 训练（MOOC）

适合希望把多个架构在多个 stage 间流水线推进的场景。当前 pipeline 模式支持 `--coarse-epochs`，并且每个 epoch 都会重置动态图状态，和串行模式保持一致。

从本版本开始，pipeline stage 切分默认使用 cost-aware 连续切分（`--stage-balance-strategy cost`），以降低 stage 间负载不均导致的空转等待。也可以通过 `--stage-balance-strategy count` 回退为按分区数量均分。
`cost` 估计同时考虑交互量、用户/物品活跃度，以及时序新颖度（new user/item）。

另外支持每个 stage 配置多个 worker（trial 级并行），例如让 stage1 用 2 个 worker、stage2 用 1 个 worker：`--pipeline-stage-train-workers 2,1`。
该并行只发生在不同 trial 之间；单个 trial 仍然保持 stage 顺序和时间分区顺序，不会破坏单模型时序约束。

```bash
python search.py --dataset public_csv --local-data-path data/public/mooc.csv --search-mode rl --execution-mode ray_pipeline --coarse-trials 30 --coarse-epochs 6 --architectures-per-step 4 --partition-size 256 --num-pipeline-stages 2 --pipeline-worker-gpus 0 --pipeline-worker-cpus 2 --pipeline-stage-train-workers 2,1 --pipeline-stage-eval-workers 1,1 --stage-balance-strategy cost --stage-balance-user-weight 0.25 --stage-balance-item-weight 0.25 --stage-balance-span-weight 0.0 --max-events 10000 --lr 3e-4 --k 10 --seed 42 --pipeline-trace --output-dir outputs_mooc_pipeline_small
```

常用负载均衡参数：

- `--stage-balance-strategy cost|count`
- `--stage-balance-user-weight`：用户多样性权重
- `--stage-balance-item-weight`：物品多样性权重
- `--stage-balance-span-weight`：时间跨度权重
- `--pipeline-stage-train-workers`：每个 train stage 的 worker 数（单值或逗号列表）
- `--pipeline-stage-eval-workers`：每个 eval stage 的 worker 数（单值或逗号列表）

## 数据格式说明（public_csv）

`public_csv` 需要 JODIE 风格 CSV，每行至少 5 列，例如 `mooc.csv`：

```text
user_id,item_id,timestamp,label,f1[,f2,...]
```

说明：
- `user_id` / `item_id` 会自动重映射为连续整数 ID。
- 数据会按 `timestamp` 升序处理。
- 特征列会按 `--feature-dim` 自动对齐：
  - 原始维度大于 `feature_dim`：截断
  - 原始维度小于 `feature_dim`：0 填充

## 自动下载与本地优先

- 对 `wikipedia` / `reddit`：若 `--local-data-path` 为空，程序会先检查 `--dataset-dir`（默认 `data/public`）下是否已有同名 CSV。
- 如果本地不存在，会自动尝试下载官方公开 CSV。
- 若网络不可用，建议手动下载后使用 `--local-data-path`。

## 输出文件

每次 `search.py` 运行会输出到 `--output-dir`：

- `best_arch.json`：最佳架构
- `leaderboard.csv`：全部候选架构排名

## 与 jodie_rnn 基线对比

当前对齐协议：
- 训练目标：预测嵌入与目标嵌入的 L2 回归损失
- 评估指标：MRR + Recall@10
- 对比约束：同一数据集、同一时间切分、同一训练轮数与学习率、同一特征处理、同一候选物品全集（并使用相同 seed）
- 基线来源：`models/jodie_rnn.py`（不再依赖外部官方仓库）

补充说明：
- `TemporalEventGNNJODIE` 需要动态图上下文 `graph_ctx`，因此 pipeline 模式会为每个 trial 初始化独立图状态。
- pipeline 模式下每个 epoch 会重置 `graph_state`，避免把上一轮的图结构累积到下一轮，和串行模式保持一致。

`compare_public_dataset.py` 支持两种 jodie_rnn 基线模式：
- `match_best`（默认）：复制搜索最优配置的可用超参，再切到 `jodie_rnn`。
- `default_arch`：使用固定的 jodie_rnn 架构配置（可通过命令行显式指定）。

当你在公开数据集上完成架构搜索后，可以把最佳架构与 `jodie_rnn` 做同集对比：

```bash
python compare_public_dataset.py --dataset wikipedia --best-arch-path outputs_wikipedia/best_arch.json --epochs 3 --output-dir outputs/public_compare_wikipedia
```

如果你要做“搜索最优 vs 固定 jodie_rnn 架构”对比：

```bash
python compare_public_dataset.py --dataset wikipedia --best-arch-path outputs_wikipedia/best_arch.json --epochs 3 --baseline-jodie-mode default_arch --baseline-embedding-dim 32 --baseline-cell-type rnn --baseline-time-proj on --output-dir outputs/public_compare_wikipedia_vs_default_jodie
```

也可以使用本地 CSV：

```bash
python compare_public_dataset.py --dataset public_csv --local-data-path data/public/wikipedia.csv --best-arch-path outputs/best_arch.json --epochs 3 --output-dir outputs/public_compare_csv
```

对比输出文件：

- `comparison_result.json`：包含 `searched_model.*`、`jodie_rnn.*`、`delta_mrr`、`delta_recall_at_10`

## 常见问题

- `ModuleNotFoundError: data.public_dataset`：确认项目目录下存在 `data/public_dataset.py`。
- `dataset=public_csv requires --local-data-path`：`public_csv` 模式必须提供本地 CSV 路径。
- `expected at least 5 columns`：CSV 列数不足，需符合 JODIE 格式。

更多参数和完整示例见 `USAGE.md`。
