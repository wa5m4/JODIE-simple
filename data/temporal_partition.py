from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from data.synthetic import Interaction


@dataclass
class TemporalPartition: #数据分区
    partition_id: int #分区id
    split: str #数据集类型（train/val/test）
    start_ts: float #分区开始时间戳
    end_ts: float #分区结束时间戳
    interactions: List[Interaction] #分区内的交互数据列表


@dataclass
class TemporalPartitionPlan:
    partitions: List[TemporalPartition] 
    #所有分区的列表
    split_to_partition_ids: Dict[str, List[int]] 
    #将数据集划分类型（train/val/test）映射到对应的分区 ID 列表
    split_sizes: Dict[str, int] 
    #每个划分类型包含的交互总数

    def get_split_partitions(self, split: str) -> List[TemporalPartition]:
        #根据指定的划分类型（train/val/test）获取对应的分区列表
        ids = set(self.split_to_partition_ids.get(split, []))
        #根据分区 ID 列表筛选出对应的分区
        #返回指定划分类型的分区TemporalPartitionPlan的列表
        return [partition for partition in self.partitions if partition.partition_id in ids]


def sort_interactions_by_time(interactions: Sequence[Interaction]) -> List[Interaction]:
    #对交互列表按时间戳进行升序排序
    #返回排序后的交互列表
    return sorted(interactions, key=lambda x: x.timestamp)


def _build_count_partitions(interactions: List[Interaction], partition_size: int) -> List[List[Interaction]]:
    #根据指定的分区大小将交互列表分割成多个分区
    #返回分区列表
    if partition_size <= 0:
        raise ValueError("partition_size must be > 0")
    return [interactions[i : i + partition_size] for i in range(0, len(interactions), partition_size)]


def _build_num_partitions(interactions: List[Interaction], num_partitions: int) -> List[List[Interaction]]:
    #根据指定的分区数量将交互列表分割成多个分区
    #返回分区列表
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")
    total = len(interactions)
    if total == 0:
        return []
    base = total // num_partitions
    remainder = total % num_partitions
    chunks: List[List[Interaction]] = []
    start = 0
    for idx in range(num_partitions):
        chunk_size = base + (1 if idx < remainder else 0)
        if chunk_size == 0:
            continue
        end = start + chunk_size
        chunks.append(interactions[start:end])
        start = end
    return chunks


def build_temporal_partitions(
    interactions: Sequence[Interaction],    #原始交互数据
    split: str,                             #数据集划分类型（train/val/test）
    partition_size: Optional[int] = None,   #分区大小
    num_partitions: Optional[int] = None,   #分区数量
    strategy: str = "count",                #分区策略（count/num）
    partition_id_offset: int = 0,           #分区 ID 的偏移量
) -> List[TemporalPartition]: #返回分区列表
    #从原始交互数据构建 TemporalPartition 对象列表

    #按时间戳升序排序
    ordered = sort_interactions_by_time(interactions)
    if not ordered:
        return []

    #目前只支持 "count" 策略（按数量划分）
    if strategy != "count":
        raise ValueError(f"Unsupported partition strategy: {strategy}")

    if partition_size is not None:#如果指定了分区大小，则按数量划分
        chunks = _build_count_partitions(ordered, partition_size)
    elif num_partitions is not None:#如果指定了分区数量，则按数量划分
        chunks = _build_num_partitions(ordered, num_partitions)
    else:#如果没有未指定分区大小和分区数量，则抛出错误
        raise ValueError("Either partition_size or num_partitions must be provided")

    partitions: List[TemporalPartition] = []
    #创建一个空列表，用于存储即将创建的 TemporalPartition 对象
    for local_idx, chunk in enumerate(chunks):
        #chunks 是一个列表List[List[Interaction]]，每个元素是一个交互列表，代表一个分区
        #enumerate(chunks) 会返回一个迭代器，产生 (index, chunk) 对，每次迭代时 index 是当前分区的索引，chunk 是当前分区的交互列表
        #遍历每个分区，创建一个 TemporalPartition 对象
        partitions.append(
            TemporalPartition(
                partition_id=partition_id_offset + local_idx, 
                #分区 ID 由 partition_id_offset 和当前分区的索引 local_idx 组成，确保全局唯一
                split=split, #数据集划分类型（train/val/test）
                start_ts=float(chunk[0].timestamp), #分区开始时间戳
                end_ts=float(chunk[-1].timestamp), #分区结束时间戳
                interactions=list(chunk), #分区内的交互数据列表，转换为 list 类型以确保一致性
            )
        )
    return partitions #返回分区列表


def build_partition_plan(
    train_interactions: Sequence[Interaction],   # 训练集交互数据
    val_interactions: Sequence[Interaction],     # 验证集交互数据
    test_interactions: Sequence[Interaction],    # 测试集交互数据
    partition_size: Optional[int] = None,        # 每个分区的大小
    num_partitions: Optional[int] = None,        # 分区的数量
    strategy: str = "count",                     # 分区策略
) -> TemporalPartitionPlan:                      # 返回规划对象
    
    #构建完整的时间分区规划，将训练集、验证集、测试集统一组织成一个 TemporalPartitionPlan 对象

    if partition_size is None and num_partitions is None:
        partition_size = max(1, len(train_interactions))
    #设置partition_size

    partitions: List[TemporalPartition] = []
    #创建一个空列表，用于存储即将创建的 TemporalPartition 对象
    split_to_partition_ids: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    #创建一个字典，用于存储每个数据集划分类型（train/val/test）的分区 ID 列表
    offset = 0
    #分区ID偏移量

    for split, split_interactions in (
        #遍历训练集、验证集、测试集
        ("train", train_interactions),
        ("val", val_interactions),
        ("test", test_interactions),
    ):
        split_partitions = build_temporal_partitions(
            interactions=split_interactions,
            split=split,
            partition_size=partition_size,
            num_partitions=num_partitions,
            strategy=strategy,
            partition_id_offset=offset,
        )
        partitions.extend(split_partitions)
        #将当前划分类型的分区添加到总分区列表中
        split_to_partition_ids[split] = [partition.partition_id for partition in split_partitions]
        #将当前划分类型的分区 ID 添加到对应的数据集划分类型列表中
        offset += len(split_partitions)
        #更新分区ID偏移量

    return TemporalPartitionPlan(
        partitions=partitions, 
        #返回总分区列表
        split_to_partition_ids=split_to_partition_ids,
        #返回每个数据集划分类型（train/val/test）的分区 ID 列表
        split_sizes={
            #返回每个数据集划分类型（train/val/test）的交互数量
            "train": len(train_interactions),
            "val": len(val_interactions),
            "test": len(test_interactions),
        },
    )
