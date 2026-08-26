"""
用于时序图训练的小批量构建工具。

t-Batch
-------
t-Batch 创建的每个批次中，用户或物品最多出现一次。这保证了无损并行性——
每个交互的正向传播结果不依赖于交互的分组方式，因为同一批次中没有两个
交互共享同一个会导致模型内存中出现写后读冲突的节点。

算法（贪心）：
  1. 按时序顺序遍历交互。
  2. 用当前批次中尚未出现的 user_id 和 item_id 填充当前批次。
  3. 当发生冲突或批次达到 batch_size 时，关闭当前批次并开始一个新批次。

TGN 时间窗口
----------------
TGN 风格的小批量将交互序列分割为固定时长的时间窗口。窗口内的所有交互
用于计算消息，然后按节点聚合。这实现了批量消息传递，是 TGN 训练的基础
（例如 train_partition_bpr_tgn）。
"""

from typing import List

from jodie.data.synthetic import Interaction


def _create_t_batches(interactions: List, batch_size: int) -> List[List]:
    """
    将交互序列分割为 t-Batch 组。

    每个批次保证用户或物品 ID 不重复出现，
    从而实现无损并行训练。

    Args:
        interactions: 按时间排序的交互列表。
        batch_size:   每个批次的最大交互数。

    Returns:
        批次列表，每个批次是一个交互列表。
    """
    batches = []
    current_batch = []
    seen_users: set = set()
    seen_items: set = set()

    for interaction in interactions:
        uid = interaction.user_id
        iid = interaction.item_id
        if uid in seen_users or iid in seen_items or len(current_batch) >= batch_size:
            if current_batch:
                batches.append(current_batch)
            current_batch = []
            seen_users = set()
            seen_items = set()
        current_batch.append(interaction)
        seen_users.add(uid)
        seen_items.add(iid)

    if current_batch:
        batches.append(current_batch)
    return batches


def _chunk_batches(interactions: List, batch_size: int) -> List[List]:
    """
    将交互序列直接按连续顺序切块（不做冲突消解）。

    与 ``_create_t_batches`` 的对照：t-Batch 保证每个批次内用户/物品 ID
    不重复（冲突无关）；本函数允许同批出现重复节点——批内后出现的交互
    读到批前旧嵌入（stale read），破坏交互流上的写后读（RAW）依赖。
    这正是朴素分批实现的样子。

    Args:
        interactions: 按时间排序的交互列表。
        batch_size:   每个批次的最大交互数。

    Returns:
        批次列表，每个批次是一个交互列表。
    """
    return [
        interactions[i : i + batch_size]
        for i in range(0, len(interactions), batch_size)
    ]


def _create_time_windows(
    interactions: List[Interaction], window_size: float
) -> List[List[Interaction]]:
    """
    将交互序列分割为固定时长的时间窗口。

    交互按时间戳排序，然后分组，使每个窗口覆盖 ``window_size`` 个时间单位。
    当自当前窗口开始以来的累计时间超过 ``window_size`` 时，开始一个新窗口。

    Args:
        interactions: 交互序列（不一定已排序）。
        window_size:  每个时间窗口的时长。

    Returns:
        时间窗口列表，每个窗口是一个交互列表。
    """
    if not interactions:
        return []

    sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
    windows = []
    current_window = []
    window_start = sorted_interactions[0].timestamp

    for interaction in sorted_interactions:
        if interaction.timestamp >= window_start + window_size:
            if current_window:
                windows.append(current_window)
            current_window = [interaction]
            window_start = interaction.timestamp
        else:
            current_window.append(interaction)

    if current_window:
        windows.append(current_window)

    return windows
