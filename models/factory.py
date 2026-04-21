"""
模型工厂：按配置创建事件级时序 GNN-JODIE 模型。
"""

from typing import Dict

from models.hybrid_jodie import TemporalEventGNNJODIE
from models.jodie_rnn import JODIERNN


def build_model(config: Dict):
    model_type = config.get("model", "temporal_event_gnn_jodie")

    if model_type in {"temporal_event_gnn_jodie", "hybrid"}:
        return TemporalEventGNNJODIE(
            num_users=config["num_users"],
            num_items=config["num_items"],
            embedding_dim=config.get("embedding_dim", 32),
            feature_dim=config.get("feature_dim", 8),
            event_agg=config.get("event_agg", "mean"),
            agg_activation=config.get("agg_activation", "none"),
            attn_type=config.get("attn_type", "dot"),
            time_decay=config.get("time_decay", "none"),
            max_neighbors=config.get("max_neighbors", 20),
            hidden_dim=config.get("hidden_dim", 128),
            memory_cell=config.get("memory_cell", "gru"),
            time_proj=config.get("time_proj", "linear"),
            memory_gate=config.get("memory_gate", "on"),
            enable_event_agg=str(config.get("enable_event_agg", "on")).lower() != "off",
            enable_graph_update=str(config.get("enable_graph_update", "on")).lower() != "off",
            message_mode=config.get("message_mode", "agg"),
            msg_linear=str(config.get("msg_linear", "on")).lower() != "off",
        )

    if model_type == "jodie_rnn":
        return JODIERNN(
            num_users=config["num_users"],
            num_items=config["num_items"],
            embedding_dim=config.get("embedding_dim", 32),
            feature_dim=config.get("feature_dim", 8),
            cell_type=config.get("memory_cell", "gru"),
            use_time_proj=str(config.get("time_proj", "linear")).lower() not in {"off", "none"},
        )

    raise ValueError(f"Unsupported model type: {model_type}")
