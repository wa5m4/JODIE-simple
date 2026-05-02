"""JODIE-style mutual-recursive dynamic embedding model."""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class JODIERNN(nn.Module):
    """JODIE-style model with mutual user/item recurrent updates."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        feature_dim: int,
        cell_type: str = "rnn",
        use_time_proj: bool = True,
        use_static_embeddings: bool = True,
        normalize_state: bool = True,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.cell_type = str(cell_type).lower()
        self.use_time_proj = use_time_proj
        self.use_static_embeddings = use_static_embeddings
        self.normalize_state = normalize_state

        self.register_buffer("user_embeddings", torch.zeros(num_users, embedding_dim))
        self.register_buffer("item_embeddings", torch.zeros(num_items, embedding_dim))
        self.register_buffer("user_last_time", torch.zeros(num_users))
        self.register_buffer("item_last_time", torch.zeros(num_items))
        self.register_buffer("user_cell_state", torch.zeros(num_users, embedding_dim))
        self.register_buffer("item_cell_state", torch.zeros(num_items, embedding_dim))

        self.user_init = nn.Parameter(torch.zeros(embedding_dim))
        self.item_init = nn.Parameter(torch.zeros(embedding_dim))

        if self.use_static_embeddings:
            self.user_static = nn.Embedding(num_users, embedding_dim)
            self.item_static = nn.Embedding(num_items, embedding_dim)
            update_static_dim = embedding_dim * 2
            predict_static_dim = embedding_dim
        else:
            self.user_static = None
            self.item_static = None
            update_static_dim = 0
            predict_static_dim = 0

        # Include explicit user/item time-gap scalars as inputs.
        user_rnn_input = embedding_dim * 2 + update_static_dim + feature_dim + 2
        item_rnn_input = embedding_dim * 2 + update_static_dim + feature_dim + 2

        if self.cell_type == "gru":
            self.user_cell = nn.GRUCell(user_rnn_input, embedding_dim)
            self.item_cell = nn.GRUCell(item_rnn_input, embedding_dim)
        elif self.cell_type == "lstm":
            self.user_cell = nn.LSTMCell(user_rnn_input, embedding_dim)
            self.item_cell = nn.LSTMCell(item_rnn_input, embedding_dim)
        else:
            self.user_cell = nn.RNNCell(user_rnn_input, embedding_dim, nonlinearity="tanh")
            self.item_cell = nn.RNNCell(item_rnn_input, embedding_dim, nonlinearity="tanh")

        self.user_time_proj = nn.Linear(1, embedding_dim, bias=False)
        self.item_time_proj = nn.Linear(1, embedding_dim, bias=False)

        predict_in_dim = embedding_dim + predict_static_dim
        self.predict_layer = nn.Sequential(
            nn.Linear(predict_in_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self._init_weights()
        self.reset_state()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.zeros_(self.user_init)
        nn.init.zeros_(self.item_init)

    def reset_state(self):
        self.user_embeddings.copy_(self.user_init.detach().unsqueeze(0).expand(self.num_users, -1))
        self.item_embeddings.copy_(self.item_init.detach().unsqueeze(0).expand(self.num_items, -1))
        self.user_last_time.zero_()
        self.item_last_time.zero_()
        self.user_cell_state.zero_()
        self.item_cell_state.zero_()

    def export_runtime_state(self) -> Dict[str, torch.Tensor]:
        state = {
            "user_embeddings": self.user_embeddings.detach().clone(),
            "item_embeddings": self.item_embeddings.detach().clone(),
            "user_last_time": self.user_last_time.detach().clone(),
            "item_last_time": self.item_last_time.detach().clone(),
        }
        if self.cell_type == "lstm":
            state["user_cell_state"] = self.user_cell_state.detach().clone()
            state["item_cell_state"] = self.item_cell_state.detach().clone()
        return state

    def import_runtime_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.user_embeddings.copy_(state["user_embeddings"].to(self.user_embeddings.device))
        self.item_embeddings.copy_(state["item_embeddings"].to(self.item_embeddings.device))
        self.user_last_time.copy_(state["user_last_time"].to(self.user_last_time.device))
        self.item_last_time.copy_(state["item_last_time"].to(self.item_last_time.device))
        if self.cell_type == "lstm":
            if "user_cell_state" in state and "item_cell_state" in state:
                self.user_cell_state.copy_(state["user_cell_state"].to(self.user_cell_state.device))
                self.item_cell_state.copy_(state["item_cell_state"].to(self.item_cell_state.device))
            else:
                self.user_cell_state.zero_()
                self.item_cell_state.zero_()

    def _normalize(self, emb: torch.Tensor) -> torch.Tensor:
        if not self.normalize_state:
            return emb
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def get_projected_embedding(
        self,
        node_embedding: torch.Tensor,
        delta_t: torch.Tensor,
        projection_layer: nn.Module,
    ) -> torch.Tensor:
        if not self.use_time_proj:
            return node_embedding
        time_factor = projection_layer(delta_t)
        return node_embedding * (1 + time_factor)

    def _delta_feature(self, delta_t: torch.Tensor) -> torch.Tensor:
        # log1p stabilizes long-tail timestamp gaps.
        return torch.log1p(torch.clamp(delta_t, min=0.0))

    def process_interaction(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        timestamps: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_embeddings[user_ids].detach().clone()
        item_emb = self.item_embeddings[item_ids].detach().clone()

        delta_user = (timestamps - self.user_last_time[user_ids].detach().clone()).unsqueeze(-1)
        delta_item = (timestamps - self.item_last_time[item_ids].detach().clone()).unsqueeze(-1)

        delta_user_feat = self._delta_feature(delta_user)
        delta_item_feat = self._delta_feature(delta_item)

        user_emb = self.get_projected_embedding(user_emb, delta_user, self.user_time_proj)
        item_emb = self.get_projected_embedding(item_emb, delta_item, self.item_time_proj)

        if self.use_static_embeddings:
            user_static = self.user_static(user_ids)
            item_static = self.item_static(item_ids)
        else:
            user_static = None
            item_static = None

        user_inputs = [user_emb, item_emb]
        item_inputs = [item_emb, user_emb]
        if self.use_static_embeddings:
            user_inputs.extend([user_static, item_static])
            item_inputs.extend([item_static, user_static])
        user_inputs.extend([features, delta_user_feat, delta_item_feat])
        item_inputs.extend([features, delta_item_feat, delta_user_feat])

        user_rnn_input = torch.cat(user_inputs, dim=-1)
        item_rnn_input = torch.cat(item_inputs, dim=-1)

        if self.cell_type == "lstm":
            user_c = self.user_cell_state[user_ids].detach().clone()
            item_c = self.item_cell_state[item_ids].detach().clone()
            new_user_emb, new_user_c = self.user_cell(user_rnn_input, (user_emb, user_c))
            new_item_emb, new_item_c = self.item_cell(item_rnn_input, (item_emb, item_c))
            self.user_cell_state[user_ids] = new_user_c.detach()
            self.item_cell_state[item_ids] = new_item_c.detach()

        else:
            new_user_emb = self.user_cell(user_rnn_input, user_emb)
            new_item_emb = self.item_cell(item_rnn_input, item_emb)

        new_user_emb = self._normalize(new_user_emb)
        new_item_emb = self._normalize(new_item_emb)

        self.user_embeddings[user_ids] = new_user_emb.detach()
        self.item_embeddings[item_ids] = new_item_emb.detach()

        self.user_last_time[user_ids] = timestamps
        self.item_last_time[item_ids] = timestamps

        return new_user_emb, new_item_emb

    def predict(self, user_ids: torch.Tensor, query_time: float) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_embeddings[user_ids]
        delta_t = (query_time - self.user_last_time[user_ids]).unsqueeze(-1)
        projected = self.get_projected_embedding(user_emb, delta_t, self.user_time_proj)

        if self.use_static_embeddings:
            predictor_input = torch.cat([projected, self.user_static(user_ids)], dim=-1)
        else:
            predictor_input = projected
        pred_item_emb = self.predict_layer(predictor_input)
        pred_item_emb = self._normalize(pred_item_emb)
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
