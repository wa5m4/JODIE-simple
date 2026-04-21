"""
GraphNAS 搜索空间定义：事件级时序 GNN-JODIE 可搜索模块。
"""

from typing import Dict, List


TEMPORAL_MODEL_NAME = "temporal_event_gnn_jodie"
PURE_JODIE_MODEL_NAME = "jodie_rnn"


def get_small_search_space() -> Dict[str, List]:
    return {
        "model": [TEMPORAL_MODEL_NAME, PURE_JODIE_MODEL_NAME],
        "embedding_dim": [32, 64, 128],
        "hidden_dim": [64, 128, 256],
        "event_agg": ["mean", "sum", "attn"],
        "agg_activation": ["none", "relu", "tanh", "gelu"],
        "attn_type": ["dot", "mlp"],
        "time_decay": ["none", "exp", "inverse"],
        "max_neighbors": [10, 20, 40],
        "memory_cell": ["rnn", "gru", "lstm", "add"],
        "time_proj": ["off", "linear", "mlp"],
        "memory_gate": ["on", "off"],
        "enable_event_agg": ["on", "off"],
        "enable_graph_update": ["on", "off"],
        "message_mode": ["agg", "peer"],
        "msg_linear": ["on", "off"],
    }


def sanitize_config(config: Dict) -> Dict:
    cfg = dict(config)
    model_name = cfg.get("model", TEMPORAL_MODEL_NAME)
    if model_name not in {TEMPORAL_MODEL_NAME, PURE_JODIE_MODEL_NAME}:
        raise ValueError(f"Unsupported model in search config: {model_name}")

    if model_name == PURE_JODIE_MODEL_NAME:
        cfg["event_agg"] = "none"
        cfg["agg_activation"] = "none"
        cfg["attn_type"] = "dot"
        cfg["time_decay"] = "none"
        cfg["max_neighbors"] = 0
        cfg["hidden_dim"] = 0
        cfg["memory_gate"] = "off"
        cfg["enable_event_agg"] = "off"
        cfg["enable_graph_update"] = "off"
        cfg["message_mode"] = "peer"
        cfg["msg_linear"] = "off"
    else:
        cfg.setdefault("enable_event_agg", "on")
        cfg.setdefault("enable_graph_update", "on")
        cfg.setdefault("message_mode", "agg")
        cfg.setdefault("msg_linear", "on")

        if cfg.get("message_mode") == "peer":
            cfg["enable_event_agg"] = "off"

        if cfg.get("enable_event_agg") == "off":
            cfg["event_agg"] = "none"
            cfg["agg_activation"] = "none"
            cfg["attn_type"] = "dot"
            cfg["time_decay"] = "none"
        elif cfg.get("event_agg") != "attn":
            cfg["attn_type"] = "dot"

    return cfg
