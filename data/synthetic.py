"""
数据模块：生成带用户偏好的交互数据，并初始化事件级动态图状态。
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import torch


@dataclass
class Interaction:
    """一条交互记录"""

    timestamp: float # 交互发生的时间戳
    user_id: int # 用户ID
    item_id: int # 商品ID
    features: torch.Tensor # 交互特征向量


def generate_synthetic_data(
    # 这个函数是用来生成合成数据的，包含用户偏好和物品类型的交互记录。

    num_users: int, # 用户数量
    num_items: int, # 商品数量
    num_interactions: int, # 交互数量
    feature_dim: int, # 输入特征维度
    seed: int = 42, # 随机数种子
) -> Tuple[List[Interaction], Dict[int, Set[int]], np.ndarray]:
    """生成有偏好的交互序列与类型标签。"""

    np.random.seed(seed) # 设置随机数种子
    torch.manual_seed(seed) # 设置PyTorch随机数种子

    num_types = 10 # 物品类型数量
    item_type = np.random.randint(0, num_types, num_items) # 每个物品随机分配一个类型

    user_type_prefs: Dict[int, Set[int]] = {}
    # 为每个用户随机分配一些物品类型作为偏好
    for uid in range(num_users):
        n_types = np.random.randint(2, 4)
        user_type_prefs[uid] = set(np.random.choice(num_types, n_types, replace=False))
        #np.random.choice(num_types, n_types, replace=False)：从 0 到 num_types - 1 中随机选择 n_types 个不重复的类型
        #set(...)：转换为集合（自动去重，虽然已经是不重复选择

    interactions: List[Interaction] = []
    for i in range(num_interactions):
    # 循环生成交互记录
        uid = np.random.randint(0, num_users)
        # 随机选择一个用户

        if np.random.random() < 0.8: # 80%偏好驱动
            allowed_types = user_type_prefs[uid] # 获取用户 uid 的偏好“类型”
            candidates = [iid for iid in range(num_items) if item_type[iid] in allowed_types]
            #遍历所有物品，item_type[iid]是物品类型，allowed_types是用户偏好类型
            #candidates是所有偏好类型的所有物品ID
            iid = int(np.random.choice(candidates)) if candidates else np.random.randint(0, num_items)
            # 从允许的物品集中选择一个物品
        else: # 20%随机选择
            iid = np.random.randint(0, num_items)

        interactions.append(
            # 生成一条交互记录
            Interaction(
                timestamp=float(i) * 0.1, # 交互时间戳
                user_id=uid,    # 用户ID
                item_id=iid,    # 商品ID
                features=torch.randn(feature_dim),  # 输入特征向量
            )
        )

    return interactions, user_type_prefs, item_type
    #返回交互记录（字典）、用户偏好、物品类型


def init_dynamic_graph_state(num_users: int, num_items: int, max_neighbors: int) -> Dict:
    """初始化事件级动态图状态容器。"""
    #一个保存动态图状态的字典，包含用户数量、物品数量、最大邻居数，以及邻接表、边的最后时间和权重等信息。
    return {
        "num_users": num_users,         # 用户数量
        "num_items": num_items,         # 商品数量
        "max_neighbors": max_neighbors, # 每个节点的最大邻居数量
        "adj": {},                      # 邻接表
        "edge_last_time": {},           # 边的最后时间
        "edge_weight": {},              # 边的权重
    }


def clone_graph_state_template(state_template: Dict) -> Dict:
    """拷贝图状态模板，确保每次训练/评估相互独立。"""
    #从现有的图状态字典中提取配置信息，然后返回一个全新的空状态（所有动态数据都重置为空）
    return {
        "num_users": state_template["num_users"],
        "num_items": state_template["num_items"],
        "max_neighbors": state_template["max_neighbors"],
        "adj": {},
        "edge_last_time": {},
        "edge_weight": {},
    }


def snapshot_graph_state(graph_state: Dict) -> Dict:
    return {
        "num_users": int(graph_state["num_users"]),
        "num_items": int(graph_state["num_items"]),
        "max_neighbors": int(graph_state["max_neighbors"]),
        "adj": {int(node): list(neighbors) for node, neighbors in graph_state["adj"].items()},
        "edge_last_time": dict(graph_state["edge_last_time"]),
        "edge_weight": dict(graph_state["edge_weight"]),
    }


def restore_graph_state(snapshot: Dict) -> Dict:
    return {
        "num_users": int(snapshot["num_users"]),
        "num_items": int(snapshot["num_items"]),
        "max_neighbors": int(snapshot["max_neighbors"]),
        "adj": {int(node): list(neighbors) for node, neighbors in snapshot["adj"].items()},
        "edge_last_time": dict(snapshot["edge_last_time"]),
        "edge_weight": dict(snapshot["edge_weight"]),
    }
