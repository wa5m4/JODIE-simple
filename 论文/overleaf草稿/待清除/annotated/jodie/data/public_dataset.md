# public_dataset.md — 对应源码 [public_dataset.py](../../jodie/data/public_dataset.py)

> **定位**:把真实世界的 CSV(Wikipedia 编辑、Reddit 发帖、MOOC 选课)清洗、规范化成项目能吃的 `Interaction` 列表。**实验真实性的第一道工序**。

---

## 🔑 load_public_dataset——CSV → Interaction 流水线(源码 72-171 行)

函数签名先扫一眼:

```python
def load_public_dataset(
    dataset_name: str,        # "wikipedia" / "reddit" / "public_csv"
    dataset_dir: str,         # 下载目录
    feature_dim: int,         # 特征维度(统一截断/补齐到这个长度)
    max_events: int = 0,      # 只取前 N 条(0=全部),用于小规模实验
    local_data_path: str = "",
    precompute_neg_seed: int | None = None,      # ← 负样本预生成的随机种子
    precompute_neg_epochs: int = 20,
    precompute_neg_sample_size: int = 5,
) -> Tuple[List[Interaction], int, int]          # (交互列表, 用户数, 物品数)
```

### 第一步:解析 CSV 行 + ID 重映射(源码 92-126 行)

```python
raw_rows: List[Tuple[int, int, float, List[float], int]] = []
user_map: Dict[int, int] = {}
item_map: Dict[int, int] = {}

with open(dataset_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for line_no, row in enumerate(reader, start=1):
        if not row:
            continue
        # 跳过可选的标题行，例如：user_id,item_id,timestamp,state_label,...
        first_col = row[0].strip().lower()
        if first_col in {"user", "user_id"}:
            continue

        if len(row) < 5:
            raise ValueError(...)  # 期望至少 5 列

        raw_uid = _to_int(row[0], dataset_path, line_no, "user_id")
        raw_iid = _to_int(row[1], dataset_path, line_no, "item_id")
        ts = _to_float(row[2], dataset_path, line_no, "timestamp")

        features = [_to_float(v, ...) for v in row[4:]]

        if raw_uid not in user_map:
            user_map[raw_uid] = len(user_map)       # ← ID 重映射
        if raw_iid not in item_map:
            item_map[raw_iid] = len(item_map)

        raw_rows.append((user_map[raw_uid], item_map[raw_iid], ts, features, line_no))
```

**CSV 的列格式**(必须记住,你自己实验数据也要按这个格式):

```
列0       列1      列2        列3      列4...
user_id, item_id, timestamp, label,  features...
```

- 列 3 是 `label`(表示正/负交互状态),**这里被忽略**——因为训练用 BPR 自造负样本,不需要数据集提供标签
- 列 4 起全是特征列

**ID 重映射为什么是必须的?** 真实数据的 user_id 可能是 `[100, 200, 300]` 这种任意整数。但模型的嵌入表 `nn.Embedding(num_users, dim)` 只接受 **0 到 num_users-1** 的连续编号(它内部就是"按行号查表")。`user_map[raw_uid] = len(user_map)` 这个写法把"第一次见到的 ID"编号为 0、下一个为 1……最终 `[100,200,300]` → `[0,1,2]`,**不重不漏、保持连续**。

### 第二步:排序 + 截断(源码 131-133 行)

```python
raw_rows.sort(key=lambda x: (x[2], x[4]))   # 按 (时间戳, 原始行号) 排序
if max_events > 0:
    raw_rows = raw_rows[:max_events]
```

`max_events` 是科研调试的好朋友:小规模实验时只取前 1000 条,几秒出结果,验证逻辑没问题再跑全量。

### 第三步:特征对齐 + 构造 Interaction(源码 135-154 行)

```python
for uid, iid, ts, feats, _ in raw_rows:
    if len(feats) >= feature_dim:
        aligned = feats[:feature_dim]                          # 太长 → 截断
    else:
        aligned = feats + [0.0] * (feature_dim - len(feats))   # 太短 → 补零
    interactions.append(
        Interaction(timestamp=ts, user_id=uid, item_id=iid,
                    features=torch.tensor(aligned, dtype=torch.float32))
    )

used_users = {ev.user_id for ev in interactions}
used_items = {ev.item_id for ev in interactions}
num_users = max(used_users) + 1
num_items = max(used_items) + 1
```

特征**统一长度**(截断/补零到 `feature_dim`)——因为神经网络要接收**固定形状**的张量。`num_users = max+1` 是因为 ID 从 0 开始连续编号,最大 ID 加一就是总数。

---

## 🔑 负样本预生成——本文件与论文关系最紧的一段(源码 156-169 行)

```python
# ── 预生成负样本（解决 Pipeline RNG 重置偏差）──────────────────
if precompute_neg_seed is not None:
    for epoch in range(precompute_neg_epochs):
        rng = np.random.default_rng(precompute_neg_seed + epoch * 100000)
        for inter in interactions:
            negs: List[int] = []
            while len(negs) < precompute_neg_sample_size:
                neg = int(rng.integers(0, num_items))
                if neg != inter.item_id:
                    negs.append(neg)
            inter.neg_samples_by_epoch[epoch] = negs
```

**读法**:对每个 epoch、每条交互,用**只依赖 (seed, epoch) 的确定性随机数生成器**抽 5 个负样本,存进 `neg_samples_by_epoch`。

**为什么这是论文级别的设计**:

1. `default_rng(seed + epoch*100000)` 是**可复现的**:任何进程、任何时刻算,epoch 3 的负样本都完全一样。训练代码里不再调用随机数生成负样本 → **随机性的来源被收编到数据层**,训练循环变得"无随机"。
2. 这是 ATTEMPTS_LOG 排障战役的成果之一:`pipeline_analysis/test_precomputed_negs.py` 验证过,负样本不一致会直接导致 Serial 与 Pipeline 训练轨迹分叉。**分布式训练中,任何一处 RNG 不一致,最后都会体现为分数对不上。**
3. 注意它"必要但不充分"(ATTEMPTS_LOG 原话):负样本统一后 Pipeline 评分仍有偏差,主因在 optimizer 状态重建(执行层)。

📄 **论文可用句式**:为了保证不同执行后端的评估一致性,我们将训练数据中的随机负采样**冻结为数据加载阶段的确定性预生成**,消除跨进程 RNG 漂移这一混淆因素。

---

## 📖 辅助函数(源码 22-69 行)

- `_resolve_dataset_path`:路径解析的三级逻辑——`public_csv` 直接用本地路径(**这里修过一个 bug:原来会去 `_JODIE_URLS` 里查 "public_csv" 导致 KeyError**);给了本地路径就检查存在性;否则从 Stanford SNAP 下载(URL 也是修过的,GitHub → SNAP)。⏭️ 了解流程即可。
- `_to_int` 用 `int(float(value))` 而不是 `int(value)`:CSV 里常出现 `"1.0"` 这种浮点写法,`int("1.0")` 直接报错,`int(float("1.0"))` 能正确得 1。📖 记住"防御性解析"这个习惯。
- `_to_float` 额外检查 `math.isfinite`:挡住 `NaN`/`Inf` 的脏数据,**错误在数据入口就暴露**,而不是训练到一半才爆炸。

---

## ❓ 检查题

**D5.** 原始 user_id 是 `[100, 200, 300]`,经过重映射后分别变成什么?为什么模型不能直接用原始 ID?(提示:回忆 `nn.Embedding` 的查表机制)

答：重映射后变成 `[0,1,2]`，因为模型内部是按行号查表的，所以不能直接用原始 ID 


**D6.** 负样本"预生成"与"训练时现场随机抽"相比,在**数据读取层面**的差别是什么?为什么这个差别在 Serial 与 Pipeline 对比时尤其致命?("必要但不充分"指的是它没解决什么?)

答：是提前做好，需要用直接读取 和 需要用现场抽取读取的区别
因为这个差别导致了两个策略实际评估的时候负样本有没有区别，会导致细微差异，这个还是没有解决两个策略评估分数不一致的问题

---

## ✅ 批改(2026-08-14)

- **D5 ✅ 对**,表述准确(nn.Embedding 按行号查表,ID 必须连续从 0 开始)。
- **D6 ✅ 两问全对**。数据层面的差别是"确定性数据"与"随机数据"的差别;为什么 Serial/Pipeline 对比时致命——两种模式抽到的负样本不同,等于在训练不同的数据,评分差异里混入了数据差异这个混淆因素;而"必要但不充分"指的是它没解决 optimizer 状态重建问题(主因,执行层详解)。