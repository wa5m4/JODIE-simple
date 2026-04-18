"""
训练与评估公用模块（事件级动态图版本）。
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.synthetic import Interaction, clone_graph_state_template


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss"""

    def forward(self, pred_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
        pos_score = (pred_emb * pos_emb).sum(dim=-1, keepdim=True)
        neg_scores = torch.bmm(neg_emb, pred_emb.unsqueeze(-1)).squeeze(-1)
        return -F.logsigmoid(pos_score - neg_scores).mean()


def _item_embeddings_for_loss(model, item_ids: torch.Tensor):
    # 从模型中提取物品嵌入
    if hasattr(model, "item_embeddings"):
        return model.item_embeddings[item_ids]
    if hasattr(model, "item_base"):
        return model.item_base(item_ids)
    if hasattr(model, "rnn_model"):
        return model.rnn_model.item_embeddings[item_ids]
    if hasattr(model, "memory") and hasattr(model, "num_users"):
        return model.memory[item_ids + model.num_users]
    raise ValueError("Model does not expose item embeddings for loss.")


def _all_item_embeddings(model):
    # 从模型中提取所有物品嵌入
    if hasattr(model, "item_embeddings"):
        return model.item_embeddings
    if hasattr(model, "item_base"):
        return model.item_base.weight
    if hasattr(model, "rnn_model"):
        return model.rnn_model.item_embeddings
    if hasattr(model, "memory") and hasattr(model, "num_users"):
        return model.memory[model.num_users : model.num_users + model.num_items]
    raise ValueError("Model does not expose item embeddings for evaluation.")


def reset_model_state(model):
    # 重置模型状态
    if hasattr(model, "reset_state"):
        model.reset_state()


def _num_items(model) -> int:
    # 获取模型中物品的数量
    if hasattr(model, "num_items"):
        return model.num_items
    if hasattr(model, "rnn_model") and hasattr(model.rnn_model, "num_items"):
        return model.rnn_model.num_items
    raise ValueError("Model does not expose num_items.")


def train_model(
    model,
    interactions: List[Interaction],
    num_epochs: int = 3,
    lr: float = 1e-3,
    neg_sample_size: int = 5,
    graph_ctx: Optional[Dict] = None,
) -> None:
    interactions = sorted(interactions, key=lambda x: x.timestamp)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BPRLoss()

    for epoch in range(num_epochs):
        reset_model_state(model)
        model.train()
        total_loss = 0.0

        epoch_graph_ctx = clone_graph_state_template(graph_ctx) if graph_ctx is not None else None

        for interaction in interactions:
            uid = torch.tensor([interaction.user_id], dtype=torch.long)
            iid = torch.tensor([interaction.item_id], dtype=torch.long)
            t = torch.tensor([interaction.timestamp], dtype=torch.float32)
            f = interaction.features.unsqueeze(0)

            neg_items = []
            while len(neg_items) < neg_sample_size:
                neg = np.random.randint(0, _num_items(model))
                if neg != interaction.item_id:
                    neg_items.append(neg)
            neg_ids = torch.tensor(neg_items, dtype=torch.long)

            optimizer.zero_grad()
            pred_emb, _, _ = model(uid, iid, t, f, interaction.timestamp, graph_ctx=epoch_graph_ctx)
            pos_emb = _item_embeddings_for_loss(model, iid)
            neg_emb = _item_embeddings_for_loss(model, neg_ids).unsqueeze(0)
            loss = criterion(pred_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(interactions), 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")


@torch.no_grad()
def evaluate_recall_at_k(model, test_interactions: List[Interaction], k: int = 10, graph_ctx=None) -> float:
    model.eval()
    hits = 0

    eval_graph_ctx = clone_graph_state_template(graph_ctx) if graph_ctx is not None else None

    for interaction in test_interactions:
        uid = torch.tensor([interaction.user_id], dtype=torch.long)
        pred_emb, _, _ = model(
            uid,
            torch.tensor([interaction.item_id], dtype=torch.long),
            torch.tensor([interaction.timestamp], dtype=torch.float32),
            interaction.features.unsqueeze(0),
            interaction.timestamp,
            graph_ctx=eval_graph_ctx,
        )
        all_item_emb = _all_item_embeddings(model)
        scores = (pred_emb @ all_item_emb.T).squeeze(0)
        top_k = scores.topk(k).indices.tolist()
        if interaction.item_id in top_k:
            hits += 1

    return hits / max(len(test_interactions), 1)


@torch.no_grad()
def evaluate_recall_by_type(model, test_interactions, item_type, user_type_prefs, k=10, graph_ctx=None) -> float:
    model.eval()
    hits = 0

    eval_graph_ctx = clone_graph_state_template(graph_ctx) if graph_ctx is not None else None

    for interaction in test_interactions:
        uid = interaction.user_id
        pred_emb, _, _ = model(
            torch.tensor([uid], dtype=torch.long),
            torch.tensor([interaction.item_id], dtype=torch.long),
            torch.tensor([interaction.timestamp], dtype=torch.float32),
            interaction.features.unsqueeze(0),
            interaction.timestamp,
            graph_ctx=eval_graph_ctx,
        )
        all_item_emb = _all_item_embeddings(model)
        scores = (pred_emb @ all_item_emb.T).squeeze(0)
        top_k_items = scores.topk(k).indices.tolist()
        top_k_types = set(item_type[iid] for iid in top_k_items)
        if top_k_types & user_type_prefs[uid]:
            hits += 1

    return hits / max(len(test_interactions), 1)
