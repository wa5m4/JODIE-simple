"""
训练与评估公用模块（事件级动态图版本）。
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.synthetic import Interaction, clone_graph_state_template
from data.temporal_partition import TemporalPartition


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss"""

    def forward(self, pred_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
        pos_score = (pred_emb * pos_emb).sum(dim=-1, keepdim=True)
        neg_scores = torch.bmm(neg_emb, pred_emb.unsqueeze(-1)).squeeze(-1)
        return -F.logsigmoid(pos_score - neg_scores).mean()


def _item_embeddings_for_loss(model, item_ids: torch.Tensor):
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
    if hasattr(model, "reset_state"):
        model.reset_state()


def _num_items(model) -> int:
    if hasattr(model, "num_items"):
        return model.num_items
    if hasattr(model, "rnn_model") and hasattr(model.rnn_model, "num_items"):
        return model.rnn_model.num_items
    raise ValueError("Model does not expose num_items.")


def _normalize_partitions(interactions: List[Interaction], partitions: Optional[List[TemporalPartition]] = None) -> List[TemporalPartition]:
    if partitions is not None:
        return partitions
    ordered = sorted(interactions, key=lambda x: x.timestamp)
    if not ordered:
        return []
    return [
        TemporalPartition(
            partition_id=0,
            split="all",
            start_ts=float(ordered[0].timestamp),
            end_ts=float(ordered[-1].timestamp),
            interactions=ordered,
        )
    ]


def _partition_seed(base_seed: Optional[int], partition_id: int, epoch: int) -> Optional[int]:
    if base_seed is None:
        return None
    return int(base_seed) + epoch * 100000 + partition_id


def train_partition_bpr(
    model,
    partition: TemporalPartition,
    optimizer,
    criterion,
    neg_sample_size: int = 5,
    graph_ctx: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> float:
    rng = np.random.default_rng(seed)
    total_loss = 0.0

    for interaction in partition.interactions:
        uid = torch.tensor([interaction.user_id], dtype=torch.long)
        iid = torch.tensor([interaction.item_id], dtype=torch.long)
        t = torch.tensor([interaction.timestamp], dtype=torch.float32)
        f = interaction.features.unsqueeze(0)

        neg_items = []
        while len(neg_items) < neg_sample_size:
            neg = int(rng.integers(0, _num_items(model)))
            if neg != interaction.item_id:
                neg_items.append(neg)
        neg_ids = torch.tensor(neg_items, dtype=torch.long)

        optimizer.zero_grad()
        pred_emb, _, _ = model(uid, iid, t, f, interaction.timestamp, graph_ctx=graph_ctx)
        pos_emb = _item_embeddings_for_loss(model, iid)
        neg_emb = _item_embeddings_for_loss(model, neg_ids).unsqueeze(0)
        loss = criterion(pred_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def train_partition_ce(
    model,
    partition: TemporalPartition,
    optimizer,
    graph_ctx: Optional[Dict] = None,
) -> float:
    total_loss = 0.0

    for interaction in partition.interactions:
        uid = torch.tensor([interaction.user_id], dtype=torch.long)
        iid = torch.tensor([interaction.item_id], dtype=torch.long)
        t = torch.tensor([interaction.timestamp], dtype=torch.float32)
        f = interaction.features.unsqueeze(0)

        optimizer.zero_grad()
        pred_emb, _, _ = model(uid, iid, t, f, interaction.timestamp, graph_ctx=graph_ctx)
        target_emb = _item_embeddings_for_loss(model, iid)
        loss = ((pred_emb - target_emb) ** 2).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def train_model(
    model,
    interactions: List[Interaction],
    num_epochs: int = 3,
    lr: float = 1e-3,
    neg_sample_size: int = 5,
    graph_ctx: Optional[Dict] = None,
    seed: Optional[int] = None,
    partitions: Optional[List[TemporalPartition]] = None,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BPRLoss()
    ordered_partitions = _normalize_partitions(interactions, partitions=partitions)
    total_events = sum(len(partition.interactions) for partition in ordered_partitions)

    for epoch in range(num_epochs):
        reset_model_state(model)
        model.train()
        total_loss = 0.0
        epoch_graph_ctx = clone_graph_state_template(graph_ctx) if graph_ctx is not None else None

        for partition in ordered_partitions:
            total_loss += train_partition_bpr(
                model=model,
                partition=partition,
                optimizer=optimizer,
                criterion=criterion,
                neg_sample_size=neg_sample_size,
                graph_ctx=epoch_graph_ctx,
                seed=_partition_seed(seed, partition.partition_id, epoch),
            )

        avg_loss = total_loss / max(total_events, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")


def train_model_ce(
    model,
    interactions: List[Interaction],
    num_epochs: int = 3,
    lr: float = 1e-3,
    graph_ctx: Optional[Dict] = None,
    seed: Optional[int] = None,
    partitions: Optional[List[TemporalPartition]] = None,
) -> None:
    if seed is not None:
        torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ordered_partitions = _normalize_partitions(interactions, partitions=partitions)
    total_events = sum(len(partition.interactions) for partition in ordered_partitions)

    for epoch in range(num_epochs):
        reset_model_state(model)
        model.train()
        total_loss = 0.0
        epoch_graph_ctx = clone_graph_state_template(graph_ctx) if graph_ctx is not None else None

        for partition in ordered_partitions:
            total_loss += train_partition_ce(
                model=model,
                partition=partition,
                optimizer=optimizer,
                graph_ctx=epoch_graph_ctx,
            )

        avg_loss = total_loss / max(total_events, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | L2 Loss: {avg_loss:.4f}")


@torch.no_grad()
def evaluate_partition_ranking(model, partition: TemporalPartition, k: int = 10, graph_ctx=None) -> Dict[str, float]:
    hits = 0
    mrr_sum = 0.0

    for interaction in partition.interactions:
        uid = torch.tensor([interaction.user_id], dtype=torch.long)
        pred_emb, _, _ = model(
            uid,
            torch.tensor([interaction.item_id], dtype=torch.long),
            torch.tensor([interaction.timestamp], dtype=torch.float32),
            interaction.features.unsqueeze(0),
            interaction.timestamp,
            graph_ctx=graph_ctx,
        )
        all_item_emb = _all_item_embeddings(model)
        distances = torch.norm(all_item_emb - pred_emb, p=2, dim=-1)
        item_count = int(distances.shape[0])
        top_k = torch.argsort(distances, descending=False)[: min(k, item_count)].tolist()
        if interaction.item_id in top_k:
            hits += 1

        sorted_indices = torch.argsort(distances, descending=False)
        rank = int((sorted_indices == interaction.item_id).nonzero(as_tuple=False)[0].item()) + 1
        mrr_sum += 1.0 / rank

    total = max(len(partition.interactions), 1)
    return {
        "hits": hits,
        "mrr_sum": mrr_sum,
        "total": total,
    }


@torch.no_grad()
def evaluate_ranking_metrics(
    model,
    test_interactions: List[Interaction],
    k: int = 10,
    graph_ctx=None,
    partitions: Optional[List[TemporalPartition]] = None,
) -> Dict[str, float]:
    model.eval()
    eval_graph_ctx = clone_graph_state_template(graph_ctx) if graph_ctx is not None else None
    ordered_partitions = _normalize_partitions(test_interactions, partitions=partitions)

    hits = 0
    mrr_sum = 0.0
    total = 0
    for partition in ordered_partitions:
        metrics = evaluate_partition_ranking(model, partition, k=k, graph_ctx=eval_graph_ctx)
        hits += int(metrics["hits"])
        mrr_sum += float(metrics["mrr_sum"])
        total += int(metrics["total"])

    total = max(total, 1)
    return {
        "recall_at_k": hits / total,
        "mrr": mrr_sum / total,
    }


@torch.no_grad()
def evaluate_recall_at_k(model, test_interactions: List[Interaction], k: int = 10, graph_ctx=None, partitions: Optional[List[TemporalPartition]] = None) -> float:
    return evaluate_ranking_metrics(model, test_interactions, k=k, graph_ctx=graph_ctx, partitions=partitions)["recall_at_k"]


@torch.no_grad()
def evaluate_partition_type_recall(model, partition: TemporalPartition, item_type, user_type_prefs, k=10, graph_ctx=None) -> Dict[str, int]:
    hits = 0

    for interaction in partition.interactions:
        uid = interaction.user_id
        pred_emb, _, _ = model(
            torch.tensor([uid], dtype=torch.long),
            torch.tensor([interaction.item_id], dtype=torch.long),
            torch.tensor([interaction.timestamp], dtype=torch.float32),
            interaction.features.unsqueeze(0),
            interaction.timestamp,
            graph_ctx=graph_ctx,
        )
        all_item_emb = _all_item_embeddings(model)
        distances = torch.norm(all_item_emb - pred_emb, p=2, dim=-1)
        top_k_items = torch.argsort(distances, descending=False)[: min(k, distances.shape[0])].tolist()
        top_k_types = set(item_type[iid] for iid in top_k_items)
        if top_k_types & user_type_prefs[uid]:
            hits += 1

    return {
        "hits": hits,
        "total": max(len(partition.interactions), 1),
    }


@torch.no_grad()
def evaluate_recall_by_type(model, test_interactions, item_type, user_type_prefs, k=10, graph_ctx=None, partitions: Optional[List[TemporalPartition]] = None) -> float:
    model.eval()
    eval_graph_ctx = clone_graph_state_template(graph_ctx) if graph_ctx is not None else None
    ordered_partitions = _normalize_partitions(test_interactions, partitions=partitions)

    hits = 0
    total = 0
    for partition in ordered_partitions:
        metrics = evaluate_partition_type_recall(
            model,
            partition,
            item_type,
            user_type_prefs,
            k=k,
            graph_ctx=eval_graph_ctx,
        )
        hits += int(metrics["hits"])
        total += int(metrics["total"])

    return hits / max(total, 1)
