# 运行示例与参数说明

本文档给出 `search.py` 的常见运行方式、参数说明和公开数据集使用方法。

---

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

## 7. 输出文件

每次运行在 `--output-dir` 下生成：

- `best_arch.json`
- `leaderboard.csv`

---

## 8. 故障排查

- `dataset=public_csv requires --local-data-path`
  - 解决：给 `public_csv` 模式提供本地 CSV 路径。

- `expected at least 5 columns`
  - 解决：检查 CSV 是否满足 JODIE 格式列数。

- `invalid user_id/item_id/timestamp/feature`
  - 解决：检查对应行是否含非数值或 NaN/Inf。

- 自动下载失败
  - 解决：手动下载后使用 `--local-data-path`。
