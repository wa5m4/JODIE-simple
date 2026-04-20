# 运行示例与参数说明

本文档给出 `search.py` 的常见运行方式、参数说明和公开数据集使用方法。

---



***************************************************************

# 快速架构搜索模型

你的Python解释器路径 search.py   
--dataset 你的数据集名称(wikipedia/reddit/public_csv/synthetic)
--dataset-dir 你的公开数据目录路径   
--local-data-path 你的本地CSV路径(仅public_csv时需要)
--max-events 你的最大事件数(如20000)   
--train-ratio 你的训练集比例(如0.7)
--val-ratio 你的验证集比例(如0.1)   
--search-mode 你的搜索模式(random/rl)
--coarse-trials 你的粗搜候选数(如24)   
--coarse-epochs 你的粗搜训练轮数(如2)
--rerank-top-k 你的重排候选数(如6)   
--rerank-epochs 你的重排训练轮数(如6)
--selection-metric 你的选择指标(mrr/recall_at_k)   
--k 你的Recall@K(如10)
--lr 你的学习率(如3e-4)   
--seed 你的随机种子(如42)
--output-dir 你的输出目录路径


例如：

c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe search.py `
  --dataset wikipedia `
  --dataset-dir data/public `
  --max-events 20000 `
  --train-ratio 0.7 `
  --val-ratio 0.1 `
  --search-mode rl `
  --coarse-trials 24 `
  --coarse-epochs 2 `
  --rerank-top-k 6 `
  --rerank-epochs 6 `
  --selection-metric mrr `
  --k 10 `
  --lr 3e-4 `
  --seed 42 `
  --output-dir outputs_wikipedia_midbudget_v3



# 快速与简单模型jodie_rnn作为基准对比


你的Python解释器路径 compare_public_dataset.py   
--dataset 你的数据集名称(wikipedia/reddit/public_csv)
--dataset-dir 你的公开数据目录路径   
--local-data-path 你的本地CSV路径(仅public_csv时需要)
--best-arch-path 你的best_arch.json路径   
--max-events 你的最大事件数(如20000)
--train-ratio 你的训练集比例(如0.7)   
--val-ratio 你的验证集比例(如0.1)
--epochs 你的训练轮数(如6)   
--lr 你的学习率(如3e-4)
--k 你的Recall@K(如10)   
--seeds 你的多随机种子列表(如42,43,44,45,46)
--strict-meta-check `
--output-dir 你的对比结果输出目录路径

如：

c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe compare_public_dataset.py `
  --dataset wikipedia `
  --dataset-dir data/public `
  --best-arch-path outputs_wikipedia_midbudget_v2/best_arch.json `
  --max-events 20000 `
  --train-ratio 0.7 `
  --val-ratio 0.1 `
  --epochs 6 `
  --lr 3e-4 `
  --k 10 `
  --seeds 42,43,44,45,46 `
  --strict-meta-check `
  --output-dir outputs/public_compare_wikipedia_midbudget_v3




***************************************************************


## 前置要求

```powershell
pip install torch numpy
```

Windows venv Python 示例：

```text
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe
```

---

## 1. 一键快速验证（synthetic）

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe search.py `
  --dataset synthetic `
  --search-mode random `
  --trials 1 `
  --epochs-per-trial 1 `
  --num-users 20 `
  --num-items 30 `
  --num-interactions 80 `
  --feature-dim 8 `
  --output-dir outputs_test
```

用途：快速确认环境和训练流程可跑通。

---

## 2. JODIE-Wikipedia 训练

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe search.py `
  --dataset wikipedia `
  --dataset-dir data/public `
  --search-mode rl `
  --trials 6 `
  --epochs-per-trial 1 `
  --max-events 5000 `
  --output-dir outputs_wikipedia
```

说明：
- 若 `data/public/wikipedia.csv` 存在则直接读取。
- 若不存在则尝试自动下载。

---

## 3. JODIE-Reddit 训练

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe search.py `
  --dataset reddit `
  --dataset-dir data/public `
  --search-mode rl `
  --trials 6 `
  --epochs-per-trial 1 `
  --max-events 5000 `
  --output-dir outputs_reddit
```

---

## 4. 使用本地 JODIE CSV（public_csv）

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe search.py `
  --dataset public_csv `
  --local-data-path data/public/wikipedia.csv `
  --search-mode rl `
  --trials 6 `
  --epochs-per-trial 1 `
  --output-dir outputs_csv
```

`public_csv` 模式下必须提供 `--local-data-path`。

---

## 5. 参数表（search.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--space` | `small` | 搜索空间名称（当前仅 `small`） |
| `--search-mode` | `rl` | 搜索模式：`random` 或 `rl` |
| `--controller-lr` | `1e-2` | RL 控制器学习率 |
| `--dataset` | `synthetic` | 数据集：`synthetic/wikipedia/reddit/public_csv` |
| `--dataset-dir` | `data/public` | 公开数据目录 |
| `--local-data-path` | 空 | 本地 CSV 路径（提供时优先级最高） |
| `--train-ratio` | `0.8` | 训练集比例 |
| `--max-events` | `0` | 使用前 N 条事件，`0` 表示全部 |
| `--trials` | `6` | 搜索候选架构数 |
| `--epochs-per-trial` | `1` | 每个候选训练轮数 |
| `--num-users` | `500` | synthetic 专用：用户数 |
| `--num-items` | `1000` | synthetic 专用：物品数 |
| `--num-interactions` | `3000` | synthetic 专用：交互数 |
| `--feature-dim` | `8` | 输入特征维度 |
| `--lr` | `1e-3` | 模型训练学习率 |
| `--neg-sample-size` | `5` | 负采样数 |
| `--k` | `10` | Recall@K 的 K |
| `--seed` | `42` | 随机种子 |
| `--output-dir` | `outputs` | 输出目录 |

---

## 6. 数据转换规则（公开数据）

公开数据在加载时会自动转换为模型可训练格式：

1. 用户/物品 ID 重映射到连续整数
2. 事件按时间戳升序排序
3. 特征向量按 `--feature-dim` 截断或零填充
4. 输出为内部 `Interaction(timestamp, user_id, item_id, features)` 序列

JODIE CSV 每行至少应为：

```text
user_id,item_id,timestamp,label,f1[,f2,...]
```

---

## 7. 公开数据集：最优架构 vs jodie_rnn 对比

先完成一次搜索拿到 `best_arch.json`，再运行：

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe compare_public_dataset.py `
  --dataset wikipedia `
  --best-arch-path outputs_wikipedia/best_arch.json `
  --epochs 3 `
  --lr 1e-3 `
  --train-ratio 0.8 `
  --seed 42 `
  --output-dir outputs/public_compare_wikipedia
```

本地 CSV 对比：

```powershell
c:/Users/17789/Desktop/jodie-simple/.venv/Scripts/python.exe compare_public_dataset.py `
  --dataset public_csv `
  --local-data-path data/public/wikipedia.csv `
  --best-arch-path outputs/best_arch.json `
  --epochs 3 `
  --lr 1e-3 `
  --train-ratio 0.8 `
  --seed 42 `
  --output-dir outputs/public_compare_csv
```

`comparison_result.json` 关键字段：

- `searched_model.mrr`
- `searched_model.recall_at_10`
- `jodie_rnn.mrr`
- `jodie_rnn.recall_at_10`
- `delta_mrr`
- `delta_recall_at_10`

---

## 8. 输出文件

每次运行在 `--output-dir` 下生成：

- `best_arch.json`
- `leaderboard.csv`

公开数据集对比会额外生成：

- `comparison_result.json`

---

## 9. 故障排查

- `dataset=public_csv requires --local-data-path`
  - 解决：给 `public_csv` 模式提供本地 CSV 路径。

- `expected at least 5 columns`
  - 解决：检查 CSV 是否满足 JODIE 格式列数。

- `invalid user_id/item_id/timestamp/feature`
  - 解决：检查对应行是否含非数值或 NaN/Inf。

- 自动下载失败
  - 解决：手动下载后使用 `--local-data-path`。
