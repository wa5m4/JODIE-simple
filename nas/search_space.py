"""
GraphNAS 搜索空间定义：事件级时序 GNN-JODIE 可搜索模块。
"""

from typing import Dict, List


MODEL_NAME = "temporal_event_gnn_jodie"


def get_small_search_space() -> Dict[str, List]:
    return {
        "model": [MODEL_NAME],
        "embedding_dim": [32, 64, 128],
        "event_agg": ["mean", "sum", "attn"],
        "agg_activation": ["none", "relu", "tanh", "gelu"],
        "attn_type": ["dot", "mlp"],
        "time_decay": ["none", "exp", "inverse"],
        "max_neighbors": [10, 20, 40],
        "memory_cell": ["rnn", "gru", "lstm", "add"],
        "time_proj": ["off", "linear", "mlp"],
        "memory_gate": ["on", "off"],
    }


def sanitize_config(config: Dict) -> Dict:
    cfg = dict(config)
    cfg["model"] = MODEL_NAME
    if cfg.get("event_agg") != "attn":
        cfg["attn_type"] = "dot"
    return cfg
