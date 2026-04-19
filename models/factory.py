"""
模型工厂：按配置创建事件级时序 GNN-JODIE 模型。
"""

from typing import Dict

from models.hybrid_jodie import TemporalEventGNNJODIE


def build_model(config: Dict):
    model_type = config.get("model", "temporal_event_gnn_jodie")
    # 从字典获取模型类型，默认是 temporal_event_gnn_jodie
    if model_type not in {"temporal_event_gnn_jodie", "hybrid"}:
        #这俩其实是一样的，都是时序事件驱动 GNN-JODIE 模型，区别只是命名而已
        #如果model_type不是这两种，则报错
        raise ValueError(f"Only temporal_event_gnn_jodie is supported, got: {model_type}")

    return TemporalEventGNNJODIE(
        # 模型参数从配置字典中提取，提供默认值
        num_users=config["num_users"], # 用户数量
        num_items=config["num_items"], # 商品数量
        embedding_dim=config.get("embedding_dim", 32), # 嵌入维度, 默认32
        feature_dim=config.get("feature_dim", 8), # 输入特征维度, 默认8
        event_agg=config.get("event_agg", "mean"), # 事件聚合方式, 默认mean
        agg_activation=config.get("agg_activation", "none"), # 聚合后激活, 默认none
        attn_type=config.get("attn_type", "dot"), # 注意力类型, 默认dot
        time_decay=config.get("time_decay", "none"), # 时间衰减方式, 默认none
        max_neighbors=config.get("max_neighbors", 20), # 最大邻居数, 默认20
        hidden_dim=config.get("hidden_dim", 128), # 隐藏层维度, 默认128
        memory_cell=config.get("memory_cell", "gru"), # 记忆单元类型, 默认gru
        time_proj=config.get("time_proj", "linear"), # 时间投影方式, 默认linear
        memory_gate=config.get("memory_gate", "on"), # 记忆门类型, 默认on
    )
