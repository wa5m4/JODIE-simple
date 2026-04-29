"""
GraphNAS 搜索空间定义：事件级时序 GNN-JODIE 可搜索模块。
"""

import json
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
        "use_static_embeddings": ["on", "off"],
        "normalize_state": ["on", "off"],
    }


def get_paper_compare_search_space() -> Dict[str, List]:
    """更偏向论文可比性的搜索空间。"""

    return {
        "model": [TEMPORAL_MODEL_NAME, PURE_JODIE_MODEL_NAME],
        "embedding_dim": [32, 64, 128],
        "hidden_dim": [64, 128, 256],
        "event_agg": ["mean", "attn"],
        "agg_activation": ["none", "relu", "tanh"],
        "attn_type": ["dot", "mlp"],
        "time_decay": ["none", "exp"],
        "max_neighbors": [10, 20, 40],
        "memory_cell": ["rnn", "gru", "lstm"],
        "time_proj": ["off", "linear"],
        "memory_gate": ["on", "off"],
        "enable_event_agg": ["on", "off"],
        "enable_graph_update": ["on", "off"],
        "message_mode": ["agg", "peer"],
        "msg_linear": ["on", "off"],
        "use_static_embeddings": ["on", "off"],
        "normalize_state": ["on", "off"],
    }


def get_search_space(space_name: str) -> Dict[str, List]:
    if space_name == "small":
        return get_small_search_space()
    if space_name == "paper_compare":
        return get_paper_compare_search_space()
    raise ValueError(f"Unsupported search space: {space_name}")


def sanitize_config(config: Dict) -> Dict:
    cfg = dict(config)
    model_name = cfg.get("model", TEMPORAL_MODEL_NAME)
    if model_name not in {TEMPORAL_MODEL_NAME, PURE_JODIE_MODEL_NAME}:
        raise ValueError(f"Unsupported model in search config: {model_name}")

    def _on_off(value, default="off"):
        if value is None:
            return default
        s = str(value).lower()
        return "on" if s in {"on", "true", "1", "yes"} else "off"

    def _choose(value, allowed, default):
        return value if value in allowed else default

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

        cfg["memory_cell"] = _choose(str(cfg.get("memory_cell", "gru")).lower(), {"rnn", "gru", "lstm"}, "gru")
        cfg["time_proj"] = _choose(str(cfg.get("time_proj", "linear")).lower(), {"off", "linear"}, "linear")
        cfg["use_static_embeddings"] = _on_off(cfg.get("use_static_embeddings", "on"), default="on")
        cfg["normalize_state"] = _on_off(cfg.get("normalize_state", "on"), default="on")
    else:
        cfg.setdefault("enable_event_agg", "on")
        cfg.setdefault("enable_graph_update", "on")
        cfg.setdefault("message_mode", "agg")
        cfg.setdefault("msg_linear", "on")

        cfg["enable_event_agg"] = _on_off(cfg.get("enable_event_agg", "on"), default="on")
        cfg["enable_graph_update"] = _on_off(cfg.get("enable_graph_update", "on"), default="on")
        cfg["msg_linear"] = _on_off(cfg.get("msg_linear", "on"), default="on")

        if cfg.get("message_mode") == "peer":
            cfg["enable_event_agg"] = "off"

        if cfg.get("enable_event_agg") == "off":
            cfg["event_agg"] = "none"
            cfg["agg_activation"] = "none"
            cfg["attn_type"] = "dot"
            cfg["time_decay"] = "none"
        elif cfg.get("event_agg") != "attn":
            cfg["attn_type"] = "dot"

        cfg["use_static_embeddings"] = "off"
        cfg["normalize_state"] = "off"

    return cfg


def canonical_config_signature(config: Dict) -> str:
    """用于去重的标准化签名。"""

    return json.dumps(sanitize_config(config), ensure_ascii=True, sort_keys=True)
