"""
RNN 基线：JODIE 动态嵌入模型。
"""

from typing import Tuple

import torch
import torch.nn as nn


class JODIERNN(nn.Module):
    """JODIE 模型（RNN 版本）"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        feature_dim: int,
        cell_type: str = "rnn",
        use_time_proj: bool = True,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.use_time_proj = use_time_proj

        self.register_buffer("user_embeddings", torch.zeros(num_users, embedding_dim))
        self.register_buffer("item_embeddings", torch.zeros(num_items, embedding_dim))
        self.register_buffer("user_last_time", torch.zeros(num_users))
        self.register_buffer("item_last_time", torch.zeros(num_items))

        user_rnn_input = embedding_dim * 2 + feature_dim
        item_rnn_input = embedding_dim * 2 + feature_dim

        if cell_type == "gru":
            self.user_cell = nn.GRUCell(user_rnn_input, embedding_dim)
            self.item_cell = nn.GRUCell(item_rnn_input, embedding_dim)
        else:
            self.user_cell = nn.RNNCell(user_rnn_input, embedding_dim, nonlinearity="tanh")
            self.item_cell = nn.RNNCell(item_rnn_input, embedding_dim, nonlinearity="tanh")

        self.time_proj = nn.Linear(1, embedding_dim, bias=False)
        self.predict_layer = nn.Linear(embedding_dim, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def reset_state(self):
        self.user_embeddings.zero_()
        self.item_embeddings.zero_()
        self.user_last_time.zero_()
        self.item_last_time.zero_()

    def export_runtime_state(self):
        return {
            "user_embeddings": self.user_embeddings.detach().clone(),
            "item_embeddings": self.item_embeddings.detach().clone(),
            "user_last_time": self.user_last_time.detach().clone(),
            "item_last_time": self.item_last_time.detach().clone(),
        }

    def import_runtime_state(self, state) -> None:
        self.user_embeddings.copy_(state["user_embeddings"].to(self.user_embeddings.device))
        self.item_embeddings.copy_(state["item_embeddings"].to(self.item_embeddings.device))
        self.user_last_time.copy_(state["user_last_time"].to(self.user_last_time.device))
        self.item_last_time.copy_(state["item_last_time"].to(self.item_last_time.device))

    def get_projected_embedding(self, node_embedding: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        if not self.use_time_proj:
            return node_embedding
        time_factor = self.time_proj(delta_t)
        return node_embedding * (1 + time_factor)

    def process_interaction(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        timestamps: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_embeddings[user_ids]
        item_emb = self.item_embeddings[item_ids]

        delta_user = (timestamps - self.user_last_time[user_ids]).unsqueeze(-1)
        delta_item = (timestamps - self.item_last_time[item_ids]).unsqueeze(-1)

        user_emb = self.get_projected_embedding(user_emb, delta_user)
        item_emb = self.get_projected_embedding(item_emb, delta_item)

        user_rnn_input = torch.cat([user_emb, item_emb, features], dim=-1)
        item_rnn_input = torch.cat([item_emb, user_emb, features], dim=-1)

        new_user_emb = self.user_cell(user_rnn_input, user_emb)
        new_item_emb = self.item_cell(item_rnn_input, item_emb)

        self.user_embeddings[user_ids] = new_user_emb.detach()
        self.item_embeddings[item_ids] = new_item_emb.detach()

        self.user_last_time[user_ids] = timestamps
        self.item_last_time[item_ids] = timestamps

        return new_user_emb, new_item_emb

    def predict(self, user_ids: torch.Tensor, query_time: float) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_embeddings[user_ids]
        delta_t = (query_time - self.user_last_time[user_ids]).unsqueeze(-1)
        projected = self.get_projected_embedding(user_emb, delta_t)
        pred_item_emb = self.predict_layer(projected)
        return pred_item_emb, projected

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        timestamps: torch.Tensor,
        features: torch.Tensor,
        query_time: float,
        graph_ctx=None,
    ):
        pred_item_emb, _ = self.predict(user_ids, query_time)
        new_user_emb, new_item_emb = self.process_interaction(user_ids, item_ids, timestamps, features)
        return pred_item_emb, new_user_emb, new_item_emb
