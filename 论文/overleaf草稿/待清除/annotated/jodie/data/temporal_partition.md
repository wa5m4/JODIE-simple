# temporal_partition.md — 对应源码 [temporal_partition.py](../../jodie/data/temporal_partition.py)

> **定位**:把一条条 `Interaction` 按时间切成"分区"(partition),再给分区贴上 train/val/test 标签。这是**时序泄漏防线**的落地代码——第 2 课概念在这里变成实现。

---

## 🔑 TemporalPartition——分区对象(源码 7-13 行)

```python
@dataclass
class TemporalPartition: #数据分区
    partition_id: int #分区id
    split: str #数据集类型（train/val/test）
    start_ts: float #分区开始时间戳
    end_ts: float #分区结束时间戳
    interactions: List[Interaction] #分区内的交互数据列表
```

**逐字段白话**:

- `partition_id` —— 分区在**全局**的唯一编号(见下方 D4 的伏笔)
- `split` —— `"train"` / `"val"` / `"test"` 标签。注意:分区本身不知道自己是训练还是评估,是**使用者**(训练循环/评估循环)根据这个标签决定怎么用它
- `start_ts / end_ts` —— 分区覆盖的时间范围(训练/评估时可用作提示信息,数据并行切块也用得上)
- `interactions` —— 一段**按时间升序排好**的交互列表

---

## 🔑 TemporalPartitionPlan——分区总表(源码 16-30 行)

```python
@dataclass
class TemporalPartitionPlan:
    partitions: List[TemporalPartition]              # 所有分区的列表
    split_to_partition_ids: Dict[str, List[int]]     # {"train": [0,1,2], "val": [3], "test": [4]}
    split_sizes: Dict[str, int]                      # {"train": 800, ...}

    def get_split_partitions(self, split: str) -> List[TemporalPartition]:
        ids = set(self.split_to_partition_ids.get(split, []))
        return [partition for partition in self.partitions if partition.partition_id in ids]
```

就是一张"目录表":想要所有 train 分区?调 `get_split_partitions("train")` 拿列表。**为什么需要这张表?** 因为训练循环需要遍历"所有 train 分区",评估循环需要"所有 test 分区",而所有分区都混在一个 `partitions` 列表里——这张表就是索引。

---

## 🔑 _build_count_partitions——按数量切块,支持重叠(源码 81-111 行)

```python
if overlap_ratio == 0:
    # 无重叠：原有逻辑
    return [interactions[i : i + partition_size] for i in range(0, len(interactions), partition_size)]

# 有重叠：改进逻辑
step = int(partition_size * (1 - overlap_ratio))
if step == 0:
    raise ValueError(...)   # 防止无限循环

partitions = []
start = 0
while start < len(interactions):
    end = min(start + partition_size, len(interactions))
    partitions.append(interactions[start:end])
    if end >= len(interactions):
        break
    start += step
return partitions
```

**无重叠**(默认):简单切片,每块 `partition_size` 条,互不重叠。

**有重叠**(`overlap_ratio > 0`):滑动窗口。比如 `partition_size=100, overlap_ratio=0.2`:

- `step = 100 × (1-0.2) = 80` → 第 0 块取 [0,100),第 1 块从 80 开始取 [80,180),第 2 块从 160 开始……
- 相邻两块共享 20 条交互(尾部/头部重叠)

**为什么要有重叠?** 分区的边界是"人造的刀口"。模型在一个分区里更新嵌入,到下个分区开头,嵌入携带的历史是完整连续的——但**没有重叠时,分区之间的过渡是生硬的**;重叠让边界区域的交互在相邻分区里都被处理,边界效应被缓冲。

**step==0 的检查**(源码 96-102 行)是一个真实的**防死循环修复**:如果 `overlap_ratio` 接近 1,`step` 算出来是 0,`start` 永远不前进,`while` 循环永不退出。科研代码里这种边界防御很重要——因为超参数是 NAS 在搜的,什么奇怪值都可能出现。

---

## 🔑 build_temporal_partitions——切块 + 包装(源码 136-179 行)

```python
ordered = sort_interactions_by_time(interactions)   # ① 先按时间升序排序
if not ordered:
    return []

if strategy != "count":
    raise ValueError(...)                            # 目前只支持 count 策略

if partition_size is not None:
    chunks = _build_count_partitions(ordered, partition_size, overlap_ratio)
elif num_partitions is not None:
    chunks = _build_num_partitions(ordered, num_partitions)
else:
    raise ValueError(...)

for local_idx, chunk in enumerate(chunks):
    partitions.append(
        TemporalPartition(
            partition_id=partition_id_offset + local_idx,  # ② 全局唯一 ID
            split=split,
            start_ts=float(chunk[0].timestamp),
            end_ts=float(chunk[-1].timestamp),
            interactions=list(chunk),
        )
    )
```

**核心流程就三步**:① 排序(时间顺序是时序模型的生命线,绝不能乱)→ ② 切块 → ③ 包装成带 `split` 标签和全局 ID 的 `TemporalPartition`。

注意 `start_ts` 取块内**第一条**的时间、`end_ts` 取**最后一条**——因为已排序,这确实就是块的时间范围。

---

## 🔑 build_partition_plan——train/val/test 的完整规划(源码 182-238 行)

```python
offset = 0
for split, split_interactions in (
    ("train", train_interactions),
    ("val", val_interactions),
    ("test", test_interactions),
):
    split_partitions = build_temporal_partitions(
        interactions=split_interactions,
        split=split,
        partition_size=partition_size,
        num_partitions=num_partitions,
        partition_id_offset=offset,      # ← 关键:ID 偏移
        overlap_ratio=overlap_ratio,
    )
    partitions.extend(split_partitions)
    split_to_partition_ids[split] = [p.partition_id for p in split_partitions]
    offset += len(split_partitions)      # ← 偏移累加
```

**ID 偏移机制**:train 用了 0,1,2 三个 ID,val 就从 `offset=3` 开始编号,test 再往后接。**结果:`partition_id` 在 train/val/test 之间全局递增、绝不重复。** 为什么必须这样?见检查题 D4。

---

## 📖 split_partition_interactions / _build_num_partitions(源码 33-72、114-133 行)

- `_build_num_partitions`:按"指定分区数量"均分(和按大小切是两种输入习惯,逻辑类似),余数摊给前面的分区。
- `split_partition_interactions`:把**一个分区**再切成 `num_workers` 份,给数据并行(DataParallel)的多 worker 用。有意思的细节:它**按时间范围切**(`chunk_duration = time_span / num_workers`),而不是按条数均分——**为了减少用户/物品在多个 worker 间的重叠**(源码 42 行注释)。为什么重叠有害?等执行层讲 DataParallel 的状态合并时会回头解释。

---

## ❓ 检查题

**D3.** `partition_size=100, overlap_ratio=0.2` 时,`step` 是多少?第 0 块和第 1 块的交互范围各是什么、重叠几条?

答：20条重叠，是第80到第100之间的

**D4.** 为什么 `partition_id` 要跨 train/val/test **全局唯一**?如果每个 split 都从 0 开始编号(出现两个 id=0 的分区),`TemporalPartitionPlan.get_split_partitions` 或执行层的日志追踪会出什么问题?(提示:想想"按 ID 找分区"的场景)

答：因为这样可以通过 `partition_id` 找到对应的分区，而不会因为 ID 重复而导致错误。如果重复会出现有多个分区的 ID 相同的情况，导致查找错误

---

## ✅ 批改(2026-08-14)

- **D3 ⚠️ 半对**。重叠 20 条、位置 [80,100) 都答对了,但漏答了 `step` 本身 = **80**。完整答案:第 0 块 [0,100)、第 1 块 [80,180)、第 2 块 [160,260)……相邻块起点差 80,尾部/头部重叠 20 条。
- **D4 ✅ 对**。补充具体场景:执行层(ray_pipeline)按 `partition_id` 派发任务、记追踪日志,如果 train 和 val 都出现 id=0,日志和调度就分不清说的是哪个分区。