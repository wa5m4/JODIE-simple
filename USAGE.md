# USAGE

本文件统一说明以下两个入口脚本的功能、全部命令行参数、以及常用运行命令：

- `search.py`：搜索事件级时序模型架构（支持串行与 Ray pipeline 执行）
- `compare_public_dataset.py`：将搜索到的最佳架构与 `jodie_rnn` 基线在公开数据集上对比

## 1. 环境准备

最小依赖：

```powershell
pip install torch numpy

MOOC + Ray pipeline 小型搜索示例（CPU）：

python search.py `
  --dataset public_csv `
  --local-data-path data/public/mooc.csv `
  --search-mode rl `
  --execution-mode ray_pipeline `
  --coarse-trials 30 `
  --coarse-epochs 6 `
  --rerank-top-k 8 `
  --rerank-epochs 10 `
  --architectures-per-step 4 `
  --partition-size 256 `
  --num-pipeline-stages 2 `
  --pipeline-worker-gpus 0 `
  --pipeline-worker-cpus 2 `
  --pipeline-stage-train-workers 2,1 `
  --pipeline-stage-eval-workers 1,1 `
  --stage-balance-strategy cost `
  --stage-balance-user-weight 0.25 `
  --stage-balance-item-weight 0.25 `
  --stage-balance-span-weight 0.0 `
  --max-events 10000 `
  --lr 3e-4 `
  --k 10 `
  --seed 42 `
  --pipeline-trace `
  --output-dir outputs_mooc_pipeline_small


| 参数 | 类型 | 默认值 | 适用范围 | 说明 |
|---|---|---|---|---|
| `--space` | str | `small` | 全部 | 搜索空间名称（当前仅支持 `small`） |
| `--search-mode` | str | `rl` | 全部 | 架构搜索模式：`random` 或 `rl` |
| `--execution-mode` | str | `serial` | 全部 | 执行后端：`serial` 或 `ray_pipeline` |
| `--controller-lr` | float | `1e-2` | 全部 | RL controller 学习率 |
| `--dataset` | str | `synthetic` | 全部 | 数据集：`synthetic/wikipedia/reddit/public_csv` |
| `--dataset-dir` | str | `data/public` | 全部 | 公开数据集目录 |
| `--local-data-path` | str | `""` | 全部 | 本地 CSV 路径（提供后优先于自动下载/默认目录） |
| `--train-ratio` | float | `0.7` | 全部 | 训练集比例 |
| `--val-ratio` | float | `0.1` | 全部 | 验证集比例 |
| `--max-events` | int | `0` | 全部 | 仅使用前 N 条事件，`0` 表示全部 |
| `--trials` | int | `6` | 全部 | 兼容参数；当 `--coarse-trials=0` 时生效 |
| `--epochs-per-trial` | int | `1` | 全部 | 兼容参数；当 `--coarse-epochs=0` 时生效 |
| `--coarse-trials` | int | `0` | 全部 | coarse 阶段采样架构数，`0` 回退到 `--trials` |
| `--coarse-epochs` | int | `0` | 全部 | 每个架构训练轮数；`0` 回退到 `--epochs-per-trial`。在 `serial` 与 `ray_pipeline` 下都生效 |
| `--rerank-top-k` | int | `0` | 全部 | rerank 阶段候选数，`0` 表示关闭 rerank |
| `--rerank-epochs` | int | `0` | 全部 | rerank 阶段训练轮数，`0` 回退到 coarse epochs |
| `--architectures-per-step` | int | `2` | ray_pipeline | 每步并发评估架构数 |
| `--partition-size` | int | `0` | 全部 | 时序分区大小；`0` 表示每个 split 只保留 1 个分区 |
| `--partition-strategy` | str | `count` | 全部 | 分区策略（当前仅支持 `count`） |
| `--num-pipeline-stages` | int | `2` | ray_pipeline | pipeline worker stage 数 |
| `--pipeline-worker-gpus` | float | `0.0` | ray_pipeline | 每个 pipeline stage worker 申请的 GPU 资源 |
| `--pipeline-worker-cpus` | float | `1.0` | ray_pipeline | 每个 pipeline stage worker 申请的 CPU 资源 |
| `--pipeline-stage-train-workers` | str | `""` | ray_pipeline | train 每个 stage 的 worker 数。可用单值（如 `2`）或逗号列表（如 `2,1,1`） |
| `--pipeline-stage-eval-workers` | str | `""` | ray_pipeline | eval 每个 stage 的 worker 数。可用单值或逗号列表 |
| `--stage-balance-strategy` | str | `cost` | ray_pipeline | stage 切分策略：`cost`(按代价连续切分，综合交互量/活跃度/时序新颖度) 或 `count`(按分区数量均分) |
| `--stage-balance-user-weight` | float | `0.25` | ray_pipeline | `cost` 策略下用户多样性权重 |
| `--stage-balance-item-weight` | float | `0.25` | ray_pipeline | `cost` 策略下物品多样性权重 |
| `--stage-balance-span-weight` | float | `0.0` | ray_pipeline | `cost` 策略下时间跨度权重 |
| `--ray-address` | str | `""` | ray_pipeline | Ray 集群地址；空字符串表示本地 `ray.init()` |
| `--pipeline-trace` | flag | `False` | ray_pipeline | 打印每个 trial 在每个 stage 的 dispatch/complete 时间日志 |
| `--eval-seeds` | str | `""` | serial | 多种子评估，格式如 `42,43,44` |
| `--family-balanced-rerank` | flag | `False` | 全部 | 开启 family-balanced rerank |
| `--family-balance-per-model` | int | `1` | 全部 | family-balanced rerank 时每个模型家族最少候选数 |
| `--num-users` | int | `500` | synthetic | synthetic 专用：用户数 |
| `--num-items` | int | `1000` | synthetic | synthetic 专用：物品数 |
| `--num-interactions` | int | `3000` | synthetic | synthetic 专用：交互数 |
| `--feature-dim` | int | `8` | 全部 | 输入特征维度 |
| `--lr` | float | `1e-3` | 全部 | 模型训练学习率 |
| `--neg-sample-size` | int | `5` | 全部 | BPR 训练负采样数 |
| `--k` | int | `10` | 全部 | Recall@K 的 K |
| `--selection-metric` | str | `mrr` | 全部 | 公开数据集上的架构选择主指标：`mrr` 或 `recall_at_k` |
| `--seed` | int | `42` | 全部 | 随机种子 |
| `--output-dir` | str | `outputs` | 全部 | 搜索输出目录 |

模式说明：

1. `execution-mode=serial` 支持 coarse + rerank，并支持 `--eval-seeds`。
2. `execution-mode=ray_pipeline` 支持 coarse pipeline + rerank pipeline，支持 `--coarse-epochs` 与 `--rerank-epochs`。
3. `ray_pipeline` 默认使用 `--stage-balance-strategy cost` 做 stage 负载均衡，也可切回 `count`。
4. `ray_pipeline` 支持每 stage 多 worker（`--pipeline-stage-train-workers/--pipeline-stage-eval-workers`），并采用 trial 级动态派发。
5. `ray_pipeline` 中每个 epoch 都会重置 `graph_state`，和串行模式保持一致。
6. 单个 trial 仍保持 stage 顺序和时间分区顺序；多 worker 只并行不同 trial，不破坏单模型时序约束。

### 2.3 `search.py` 输出文件

运行完成后在 `--output-dir` 下生成：

- `best_arch.json`
- `leaderboard.csv`

## 3. `compare_public_dataset.py` 功能概览

`compare_public_dataset.py` 用于把搜索到的最佳架构与 `jodie_rnn` 基线在同一数据切分和训练配置下进行公平比较。

- 支持数据集：`wikipedia`、`reddit`、`public_csv`
- 支持单种子和多种子评估
- 支持两种 jodie_rnn 基线构建策略：
  - `match_best`：复制最佳架构可用超参数，再切换到 `jodie_rnn`
  - `default_arch`：使用固定 jodie_rnn 架构参数

### 3.1 常用命令

对比搜索结果与 jodie_rnn（默认 `match_best`）：

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe compare_public_dataset.py `
  --dataset wikipedia `
  --dataset-dir data/public `
  --best-arch-path outputs_wikipedia/best_arch.json `
  --epochs 3 `
  --train-ratio 0.7 `
  --val-ratio 0.1 `
  --k 10 `
  --seed 42 `
  --output-dir outputs/public_compare_wikipedia
```

固定 jodie_rnn 架构（`default_arch`）多种子对比：

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe compare_public_dataset.py `
  --dataset wikipedia `
  --best-arch-path outputs_wikipedia_midbudget_v4/best_arch.json `
  --max-events 20000 `
  --epochs 6 `
  --lr 3e-4 `
  --seeds 42,43,44,45,46 `
  --baseline-jodie-mode default_arch `
  --baseline-embedding-dim 32 `
  --baseline-cell-type rnn `
  --baseline-time-proj on `
  --strict-meta-check `
  --output-dir outputs/public_compare_wikipedia_midbudget_v4_vs_default_jodie
```

### 3.2 `compare_public_dataset.py` 全参数表

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--dataset` | str | `wikipedia` | 数据集：`wikipedia/reddit/public_csv` |
| `--dataset-dir` | str | `data/public` | 公开数据目录 |
| `--local-data-path` | str | `""` | 本地 CSV 路径（`public_csv` 必填） |
| `--best-arch-path` | str | `outputs/best_arch.json` | 搜索得到的最佳架构文件路径 |
| `--epochs` | int | `3` | 每个模型的训练轮数 |
| `--train-ratio` | float | `0.7` | 训练集比例 |
| `--val-ratio` | float | `0.1` | 验证集比例 |
| `--max-events` | int | `0` | 使用前 N 条事件，`0` 表示全部 |
| `--feature-dim` | int | `8` | 输入特征维度 |
| `--lr` | float | `1e-3` | 学习率 |
| `--k` | int | `10` | 排名评估的 K |
| `--seed` | int | `42` | 单种子模式默认种子 |
| `--seeds` | str | `""` | 多种子列表，格式如 `42,43,44` |
| `--baseline-jodie-mode` | str | `match_best` | `jodie_rnn` 基线模式：`match_best/default_arch` |
| `--baseline-embedding-dim` | int | `32` | `default_arch` 下 jodie_rnn embedding 维度 |
| `--baseline-cell-type` | str | `rnn` | `default_arch` 下 jodie_rnn cell 类型（常用：`rnn/gru/lstm/add`） |
| `--baseline-time-proj` | str | `on` | `default_arch` 下是否启用 time projection（`on/off`） |
| `--output-dir` | str | `outputs/public_compare` | 对比输出目录 |
| `--strict-meta-check` | flag | `False` | 若当前配置与 `best_arch` 元信息不一致则报错 |

### 3.3 `compare_public_dataset.py` 输出文件

在 `--output-dir` 下输出：

- `comparison_result.json`：包含 searched 模型与 jodie_rnn 的指标及差值

## 4. 数据格式说明

`public_csv` 需要 JODIE 风格 CSV，每行至少 5 列：

```text
user_id,item_id,timestamp,label,f1[,f2,...]
```

程序会自动：

1. 将 `user_id` 和 `item_id` 重映射为连续整数
2. 按 `timestamp` 升序排序
3. 根据 `--feature-dim` 截断或零填充特征

## 5. 常见问题

- 报错：`dataset=public_csv requires --local-data-path`
  - 处理：为 `public_csv` 模式提供本地 CSV 路径

- 报错：`expected at least 5 columns`
  - 处理：检查 CSV 是否符合 JODIE 格式

- 报错：`invalid user_id/item_id/timestamp/feature`
  - 处理：检查是否有非数值、NaN、Inf

- 使用 Ray pipeline 时报错找不到 `ray`
  - 处理：安装 `ray` 并确认 VS Code 选中的 Python 环境正确

## 6. 参数模板速查

下面按目标给出可直接复制的模板。你只需要替换数据集和输出目录。

### 6.1 模板 A：5 分钟冒烟测试（最小成本）

适用：先确认环境、数据流、模型训练都能跑通。

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe search.py `
  --dataset synthetic `
  --search-mode random `
  --execution-mode serial `
  --trials 2 `
  --epochs-per-trial 1 `
  --num-users 50 `
  --num-items 80 `
  --num-interactions 400 `
  --output-dir outputs_smoke
```

### 6.2 模板 B：中小预算串行搜索（结果更稳）

适用：单机 CPU/GPU 都可跑，关注可复现和结果稳定性。

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe search.py `
  --dataset wikipedia `
  --search-mode rl `
  --execution-mode serial `
  --coarse-trials 32 `
  --coarse-epochs 3 `
  --rerank-top-k 8 `
  --rerank-epochs 6 `
  --eval-seeds 42,43,44 `
  --family-balanced-rerank `
  --family-balance-per-model 1 `
  --selection-metric mrr `
  --k 10 `
  --lr 3e-4 `
  --seed 42 `
  --output-dir outputs_serial_mid
```

### 6.3 模板 C：3-stage 架构流水线

适用：你希望出现 “TGNN1->stage2 时，stage1 同时开始 TGNN2” 这种流动。

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe search.py `
  --dataset wikipedia `
  --search-mode rl `
  --execution-mode ray_pipeline `
  --coarse-trials 48 `
  --architectures-per-step 10 `
  --partition-size 256 `
  --num-pipeline-stages 3 `
  --pipeline-worker-gpus 1 `
  --pipeline-worker-cpus 1 `
  --pipeline-stage-train-workers 2,1,1 `
  --pipeline-stage-eval-workers 1,1,1 `
  --stage-balance-strategy cost `
  --stage-balance-user-weight 0.25 `
  --stage-balance-item-weight 0.25 `
  --stage-balance-span-weight 0.0 `
  --pipeline-trace `
  --output-dir outputs_pipeline_3stage
```

### 6.4 模板 D：Ray 集群多机流水线

适用：已有 Ray 集群，希望把 stage 分配到不同机器。

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe search.py `
  --dataset wikipedia `
  --search-mode rl `
  --execution-mode ray_pipeline `
  --coarse-trials 64 `
  --architectures-per-step 12 `
  --partition-size 256 `
  --num-pipeline-stages 4 `
  --pipeline-worker-gpus 1 `
  --pipeline-worker-cpus 1 `
  --pipeline-stage-train-workers 2,1,1,1 `
  --pipeline-stage-eval-workers 1,1,1,1 `
  --stage-balance-strategy cost `
  --stage-balance-user-weight 0.25 `
  --stage-balance-item-weight 0.25 `
  --stage-balance-span-weight 0.0 `
  --ray-address auto `
  --pipeline-trace `
  --output-dir outputs_pipeline_cluster
```

### 6.5 模板 E：搜索后固定基线对比

适用：验证“搜索最优架构 vs 固定 jodie_rnn”的差值。

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe compare_public_dataset.py `
  --dataset wikipedia `
  --best-arch-path outputs_pipeline_3stage/best_arch.json `
  --epochs 6 `
  --lr 3e-4 `
  --train-ratio 0.7 `
  --val-ratio 0.1 `
  --max-events 20000 `
  --seeds 42,43,44,45,46 `
  --baseline-jodie-mode default_arch `
  --baseline-embedding-dim 32 `
  --baseline-cell-type rnn `
  --baseline-time-proj on `
  --strict-meta-check `
  --output-dir outputs/public_compare_pipeline_vs_default_jodie
```

### 6.6 调参建议（简版）

1. GPU 不够时优先降低 `--architectures-per-step`，其次降低 `--num-pipeline-stages`。
2. 吞吐不高时优先减小 `--partition-size`，让流水线颗粒更细。
3. stage 负载不均时优先保持 `--stage-balance-strategy cost`，再微调三个权重。
4. 如果代价估计失真明显，可临时切回 `--stage-balance-strategy count` 做对照。
5. 对明显慢的 stage（常见 stage1）先增加该 stage worker 数，如 `--pipeline-stage-train-workers 2,1`。
6. 想更稳的最佳架构，优先增加 `--coarse-trials` 和 `--rerank-epochs`。
7. 串行模式下想更稳分数，启用 `--eval-seeds`。
