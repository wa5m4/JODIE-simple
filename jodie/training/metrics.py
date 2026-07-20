"""
时序图模型的评估指标。

提供基于排序和基于类型的召回率评估，支持多种粒度
（按分区评估和聚合所有分区）。
"""

import time
from typing import Dict, List, Optional

import torch

from jodie.data.synthetic import Interaction, clone_graph_state_template
from jodie.data.temporal_partition import TemporalPartition

from .loops import _model_device, _all_item_embeddings, _normalize_partitions


@torch.no_grad()
def evaluate_partition_ranking(
    model,
    partition: TemporalPartition,
    k: int = 10,
    graph_ctx=None,
    progress_label: str = "",
    progress_every: int = 0,
    progress_callback=None,
    frozen: bool = False,
) -> Dict[str, float]:
    """评估单个分区上的排序指标（Recall@k / MRR）。

    对于每个交互：
      1. 通过模型正向传播获取预测嵌入。
      2. 计算与所有物品嵌入的 L2 距离。
      3. 如果真实物品在 top-k 最近物品中，则计为命中。
      4. 累计 MRR。

    Args:
        frozen: 如果为 ``True``，保存并恢复模型的节点嵌入，
                使评估不会永久改变它们（在线评估模式）。

    Returns:
        包含键 ``hits``、``mrr_sum`` 和 ``total`` 的字典
        （评估的交互数量）。
    """
    device = _model_device(model)

    # 如果 frozen=True，保存原始嵌入
    if frozen and hasattr(model, "user_embeddings"):
        original_user_emb = model.user_embeddings.data.clone()
        original_item_emb = model.item_embeddings.data.clone()
        original_user_time = model.user_last_time.data.clone()
        original_item_time = model.item_last_time.data.clone()
        if hasattr(model, "user_cell_state") and model.user_cell_state is not None:
            original_user_cell = model.user_cell_state.data.clone()
            original_item_cell = model.item_cell_state.data.clone()
        else:
            original_user_cell = None
            original_item_cell = None

    hits = 0
    mrr_sum = 0.0
    interaction_total = len(partition.interactions)

    start_time = time.time()
    for idx, interaction in enumerate(partition.interactions, start=1):
        if progress_every > 0 and (idx == 1 or idx % max(progress_every, 100) == 0 or idx == interaction_total):
            if progress_callback is not None:
                progress_callback(idx, interaction_total)
            elapsed = time.time() - start_time
            rate = idx / max(elapsed, 0.1)
            remaining = (interaction_total - idx) / max(rate, 0.1)
            pct = 100.0 * idx / max(interaction_total, 1)
            prefix = f"[{progress_label}] " if progress_label else ""
            print(f"{prefix}[Interaction {idx}/{interaction_total} ({pct:.1f}%)] elapsed={elapsed:.1f}s, est.remain={remaining:.1f}s, partition={partition.partition_id}", flush=True)

        uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)

        if frozen and hasattr(model, "user_embeddings"):
            pred_emb, _, _ = model(
                uid,
                torch.tensor([interaction.item_id], dtype=torch.long, device=device),
                torch.tensor([interaction.timestamp], dtype=torch.float32, device=device),
                interaction.features.unsqueeze(0).to(device),
                interaction.timestamp,
                graph_ctx=graph_ctx,
                deferred=True,
            )
        else:
            pred_emb, _, _ = model(
                uid,
                torch.tensor([interaction.item_id], dtype=torch.long, device=device),
                torch.tensor([interaction.timestamp], dtype=torch.float32, device=device),
                interaction.features.unsqueeze(0).to(device),
                interaction.timestamp,
                graph_ctx=graph_ctx,
            )

        all_item_emb = _all_item_embeddings(model).to(device)
        distances = torch.norm(all_item_emb - pred_emb, p=2, dim=-1)
        item_count = int(distances.shape[0])
        top_k = torch.argsort(distances, descending=False)[: min(k, item_count)].tolist()
        if interaction.item_id in top_k:
            hits += 1

        sorted_indices = torch.argsort(distances, descending=False)
        rank = int((sorted_indices == interaction.item_id).nonzero(as_tuple=False)[0].item()) + 1
        mrr_sum += 1.0 / rank

    # 如果 frozen，恢复原始嵌入
    if frozen and hasattr(model, "user_embeddings"):
        model.user_embeddings.data = original_user_emb
        model.item_embeddings.data = original_item_emb
        model.user_last_time.data = original_user_time
        model.item_last_time.data = original_item_time
        if original_user_cell is not None:
            model.user_cell_state.data = original_user_cell
            model.item_cell_state.data = original_item_cell

    total = max(interaction_total, 1)
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
    frozen: bool = False,
) -> Dict[str, float]:
    """跨所有分区评估排序指标（Recall@k 和 MRR）。

    Args:
        frozen: 如果为 ``True``，在每个分区前保存模型的节点嵌入，
                之后恢复，允许在线评估而不永久改变状态。

    Returns:
        包含键 ``recall_at_k`` 和 ``mrr`` 的字典。
    """
    model.eval()
    eval_graph_ctx = clone_graph_state_template(graph_ctx) if graph_ctx is not None else None
    ordered_partitions = _normalize_partitions(test_interactions, partitions=partitions)

    hits = 0
    mrr_sum = 0.0
    total = 0
    for partition in ordered_partitions:
        metrics = evaluate_partition_ranking(model, partition, k=k, graph_ctx=eval_graph_ctx, frozen=frozen)
        hits += int(metrics["hits"])
        mrr_sum += float(metrics["mrr_sum"])
        total += int(metrics["total"])

    total = max(total, 1)
    return {
        "recall_at_k": hits / total,
        "mrr": mrr_sum / total,
    }


@torch.no_grad()
def evaluate_recall_at_k(
    model,
    test_interactions: List[Interaction],
    k: int = 10,
    graph_ctx=None,
    partitions: Optional[List[TemporalPartition]] = None,
) -> float:
    """便捷包装函数，仅返回 Recall@k。"""
    return evaluate_ranking_metrics(model, test_interactions, k=k, graph_ctx=graph_ctx, partitions=partitions)["recall_at_k"]


@torch.no_grad()
def evaluate_partition_type_recall(
    model,
    partition: TemporalPartition,
    item_type,
    user_type_prefs,
    k: int = 10,
    graph_ctx=None,
    progress_label: str = "",
    progress_every: int = 0,
    progress_callback=None,
) -> Dict[str, int]:
    """评估单个分区上基于类型的召回率。

    如果 top-k 最近物品中至少有一个物品的类型
    与用户的偏好类型匹配，则该交互被视为"命中"。

    Returns:
        包含键 ``hits`` 和 ``total`` 的字典。
    """
    device = _model_device(model)
    hits = 0
    interaction_total = len(partition.interactions)

    for idx, interaction in enumerate(partition.interactions, start=1):
        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == interaction_total):
            if progress_callback is not None:
                progress_callback(idx, interaction_total)
            prefix = f"[{progress_label}] " if progress_label else ""
            print(f"{prefix}eval type progress {idx}/{interaction_total} partition={partition.partition_id}", flush=True)
        uid = interaction.user_id
        pred_emb, _, _ = model(
            torch.tensor([uid], dtype=torch.long, device=device),
            torch.tensor([interaction.item_id], dtype=torch.long, device=device),
            torch.tensor([interaction.timestamp], dtype=torch.float32, device=device),
            interaction.features.unsqueeze(0).to(device),
            interaction.timestamp,
            graph_ctx=graph_ctx,
        )
        all_item_emb = _all_item_embeddings(model).to(device)
        distances = torch.norm(all_item_emb - pred_emb, p=2, dim=-1)
        top_k_items = torch.argsort(distances, descending=False)[: min(k, distances.shape[0])].tolist()
        top_k_types = set(item_type[iid] for iid in top_k_items)
        if top_k_types & user_type_prefs[uid]:
            hits += 1

    return {
        "hits": hits,
        "total": max(interaction_total, 1),
    }


@torch.no_grad()
def evaluate_recall_by_type(
    model,
    test_interactions,
    item_type,
    user_type_prefs,
    k: int = 10,
    graph_ctx=None,
    partitions: Optional[List[TemporalPartition]] = None,
) -> float:
    """跨所有分区评估基于类型的召回率。

    返回 top-k 物品中至少有一个物品的类型
    与用户偏好类型匹配的交互比例。
    """
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
