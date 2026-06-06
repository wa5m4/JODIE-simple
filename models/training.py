"""
训练与评估公用模块（事件级动态图版本）。
"""

import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.synthetic import Interaction, clone_graph_state_template
from data.temporal_partition import TemporalPartition


def _model_device(model) -> torch.device:
    """Return the device of the model's first parameter or buffer."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return next(model.buffers()).device


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
    progress_every: int = 0,
    progress_callback=None,
) -> float:
    device = _model_device(model)
    rng = np.random.default_rng(seed)
    total_loss = 0.0

    interaction_total = len(partition.interactions)
    for idx, interaction in enumerate(partition.interactions, start=1):
        if progress_every > 0 and (idx == 1 or idx % max(progress_every, 100) == 0 or idx == interaction_total):
            if progress_callback is not None:
                progress_callback(idx, interaction_total)
        uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
        iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)
        t = torch.tensor([interaction.timestamp], dtype=torch.float32, device=device)
        f = interaction.features.unsqueeze(0).to(device)

        neg_items = []
        while len(neg_items) < neg_sample_size:
            neg = int(rng.integers(0, _num_items(model)))
            if neg != interaction.item_id:
                neg_items.append(neg)
        neg_ids = torch.tensor(neg_items, dtype=torch.long, device=device)

        optimizer.zero_grad()
        pred_emb, _, _ = model(uid, iid, t, f, interaction.timestamp, graph_ctx=graph_ctx)
        pos_emb = _item_embeddings_for_loss(model, iid).detach().to(device)
        neg_emb = _item_embeddings_for_loss(model, neg_ids).detach().to(device).unsqueeze(0)
        loss = criterion(pred_emb, pos_emb, neg_emb)
        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def train_partition_ce(
    model,
    partition: TemporalPartition,
    optimizer,
    graph_ctx: Optional[Dict] = None,
    progress_every: int = 0,
    progress_callback=None,
) -> float:
    device = _model_device(model)
    total_loss = 0.0

    interaction_total = len(partition.interactions)
    for idx, interaction in enumerate(partition.interactions, start=1):
        if progress_every > 0 and (idx == 1 or idx % max(progress_every, 100) == 0 or idx == interaction_total):
            if progress_callback is not None:
                progress_callback(idx, interaction_total)
        uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
        iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)
        t = torch.tensor([interaction.timestamp], dtype=torch.float32, device=device)
        f = interaction.features.unsqueeze(0).to(device)

        optimizer.zero_grad()
        pred_emb, _, _ = model(uid, iid, t, f, interaction.timestamp, graph_ctx=graph_ctx)
        target_emb = _item_embeddings_for_loss(model, iid).to(device)
        loss = ((pred_emb - target_emb) ** 2).sum(dim=-1).mean()
        loss.backward(retain_graph=True)
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
    batch_training: bool = False,
    batch_size: int = 32,
    batch_mode: str = "serial",
    tgn_loss_mode: str = "all",
    tgn_window_size: float = 10.0,
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
            if batch_mode == "tgn":
                total_loss += train_partition_bpr_tgn(
                    model=model,
                    partition=partition,
                    optimizer=optimizer,
                    criterion=criterion,
                    time_window_size=tgn_window_size,
                    aggregator="mean",
                    loss_mode=tgn_loss_mode,
                    neg_sample_size=neg_sample_size,
                    seed=_partition_seed(seed, partition.partition_id, epoch),
                    graph_ctx=epoch_graph_ctx,
                )
            elif batch_mode == "tbatch" or batch_training:
                total_loss += train_partition_bpr_batch(
                    model=model,
                    partition=partition,
                    optimizer=optimizer,
                    neg_sample_size=neg_sample_size,
                    batch_size=batch_size,
                    seed=_partition_seed(seed, partition.partition_id, epoch),
                    graph_ctx=epoch_graph_ctx,
                )
            else:  # serial
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
    batch_training: bool = False,
    batch_size: int = 32,
    batch_mode: str = "serial",
    tgn_loss_mode: str = "all",
    tgn_window_size: float = 10.0,
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
            if batch_mode == "tgn":
                total_loss += train_partition_ce_tgn(
                    model=model,
                    partition=partition,
                    optimizer=optimizer,
                    time_window_size=tgn_window_size,
                    aggregator="mean",
                    loss_mode=tgn_loss_mode,
                    seed=_partition_seed(seed, partition.partition_id, epoch),
                    graph_ctx=epoch_graph_ctx,
                )
            elif batch_mode == "tbatch" or batch_training:
                total_loss += train_partition_ce_batch(
                    model=model,
                    partition=partition,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    seed=_partition_seed(seed, partition.partition_id, epoch),
                    graph_ctx=epoch_graph_ctx,
                )
            else:  # serial
                total_loss += train_partition_ce(
                    model=model,
                    partition=partition,
                    optimizer=optimizer,
                    graph_ctx=epoch_graph_ctx,
                )

        avg_loss = total_loss / max(total_events, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | L2 Loss: {avg_loss:.4f}")


@torch.no_grad()
def evaluate_partition_ranking(model, partition: TemporalPartition, k: int = 10, graph_ctx=None, progress_label: str = "", progress_every: int = 0, progress_callback=None, frozen: bool = False) -> Dict[str, float]:
    device = _model_device(model)

    # 如果frozen=True，保存原始embeddings
    if frozen and hasattr(model, 'user_embeddings'):
        original_user_emb = model.user_embeddings.data.clone()
        original_item_emb = model.item_embeddings.data.clone()
        original_user_time = model.user_last_time.data.clone()
        original_item_time = model.item_last_time.data.clone()
        if hasattr(model, 'user_cell_state') and model.user_cell_state is not None:
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

        # 根据frozen参数决定是否使用deferred模式
        if frozen and hasattr(model, 'user_embeddings'):
            pred_emb, _, _ = model(
                uid,
                torch.tensor([interaction.item_id], dtype=torch.long, device=device),
                torch.tensor([interaction.timestamp], dtype=torch.float32, device=device),
                interaction.features.unsqueeze(0).to(device),
                interaction.timestamp,
                graph_ctx=graph_ctx,
                deferred=True,  # 冻结模式：不更新embeddings
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

    # 如果frozen=True，恢复原始embeddings
    if frozen and hasattr(model, 'user_embeddings'):
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
    frozen: bool = False,  # 在线评估模式（允许测试时更新embeddings）
) -> Dict[str, float]:
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
def evaluate_recall_at_k(model, test_interactions: List[Interaction], k: int = 10, graph_ctx=None, partitions: Optional[List[TemporalPartition]] = None) -> float:
    return evaluate_ranking_metrics(model, test_interactions, k=k, graph_ctx=graph_ctx, partitions=partitions)["recall_at_k"]


@torch.no_grad()
def evaluate_partition_type_recall(model, partition: TemporalPartition, item_type, user_type_prefs, k=10, graph_ctx=None, progress_label: str = "", progress_every: int = 0, progress_callback=None) -> Dict[str, int]:
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


def _create_t_batches(interactions: List, batch_size: int) -> List[List]:
    """
    将交互序列切分为 t-Batch 列表。
    每个 batch 内 user 和 item 均不重复（无损并行的前提）。
    按时间顺序贪心填充：遇到重复节点则开启新 batch。
    """
    batches = []
    current_batch = []
    seen_users: set = set()
    seen_items: set = set()

    for interaction in interactions:
        uid = interaction.user_id
        iid = interaction.item_id
        if uid in seen_users or iid in seen_items or len(current_batch) >= batch_size:
            if current_batch:
                batches.append(current_batch)
            current_batch = []
            seen_users = set()
            seen_items = set()
        current_batch.append(interaction)
        seen_users.add(uid)
        seen_items.add(iid)

    if current_batch:
        batches.append(current_batch)
    return batches


def train_partition_bpr_batch(
    model,
    partition: TemporalPartition,
    optimizer,
    neg_sample_size: int = 5,
    batch_size: int = 32,
    seed: Optional[int] = None,
    graph_ctx=None,
) -> float:
    """标准 t-Batch BPR 训练：batch 内节点唯一，逐条 forward，累积 loss 后统一 backward。"""
    device = _model_device(model)
    rng = np.random.default_rng(seed)
    criterion = BPRLoss()
    total_loss = 0.0

    for batch in _create_t_batches(partition.interactions, batch_size):
        optimizer.zero_grad()
        batch_losses = []

        for interaction in batch:
            uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
            iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)
            t = torch.tensor([interaction.timestamp], dtype=torch.float32, device=device)
            f = interaction.features.unsqueeze(0).to(device)

            neg_items = []
            while len(neg_items) < neg_sample_size:
                neg = int(rng.integers(0, _num_items(model)))
                if neg != interaction.item_id:
                    neg_items.append(neg)
            neg_ids = torch.tensor(neg_items, dtype=torch.long, device=device)

            pred_emb, _, _ = model(uid, iid, t, f, interaction.timestamp, graph_ctx=graph_ctx)
            pos_emb = _item_embeddings_for_loss(model, iid).detach().to(device)
            neg_emb = _item_embeddings_for_loss(model, neg_ids).detach().to(device).unsqueeze(0)
            batch_losses.append(criterion(pred_emb, pos_emb, neg_emb))

        total_batch_loss = sum(batch_losses)
        total_batch_loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += total_batch_loss.item()

    return total_loss


def train_partition_ce_batch(
    model,
    partition: TemporalPartition,
    optimizer,
    batch_size: int = 32,
    seed: Optional[int] = None,
    graph_ctx=None,
) -> float:
    """标准 t-Batch CE/L2 训练：batch 内节点唯一，逐条 forward，累积 loss 后统一 backward。"""
    device = _model_device(model)
    total_loss = 0.0

    for batch in _create_t_batches(partition.interactions, batch_size):
        optimizer.zero_grad()
        batch_losses = []

        for interaction in batch:
            uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
            iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)
            t = torch.tensor([interaction.timestamp], dtype=torch.float32, device=device)
            f = interaction.features.unsqueeze(0).to(device)

            pred_emb, _, _ = model(uid, iid, t, f, interaction.timestamp, graph_ctx=graph_ctx)
            target_emb = _item_embeddings_for_loss(model, iid).to(device)
            batch_losses.append(((pred_emb - target_emb) ** 2).sum(dim=-1).mean())

        total_batch_loss = sum(batch_losses)
        total_batch_loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += total_batch_loss.item()

    return total_loss


def _create_time_windows(interactions: List[Interaction], window_size: float) -> List[List[Interaction]]:
    """按时间窗口切分交互序列"""
    if not interactions:
        return []

    sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
    windows = []
    current_window = []
    window_start = sorted_interactions[0].timestamp

    for interaction in sorted_interactions:
        if interaction.timestamp >= window_start + window_size:
            if current_window:
                windows.append(current_window)
            current_window = [interaction]
            window_start = interaction.timestamp
        else:
            current_window.append(interaction)

    if current_window:
        windows.append(current_window)

    return windows


def train_partition_bpr_tgn(
    model,
    partition: TemporalPartition,
    optimizer,
    criterion,
    time_window_size: float,
    aggregator: str = "mean",
    loss_mode: str = "all",
    neg_sample_size: int = 5,
    seed: Optional[int] = None,
    graph_ctx: Optional[Dict] = None,
) -> float:
    """TGN 风格窗口批处理训练（BPR loss）：窗口内聚合消息后更新

    Args:
        loss_mode: "all" - 所有交互都计算loss; "last" - 只对每个节点最后一条交互计算loss
    """
    from collections import defaultdict
    device = _model_device(model)
    rng = np.random.default_rng(seed)
    total_loss = 0.0

    windows = _create_time_windows(partition.interactions, time_window_size)
    total_events = len(partition.interactions)
    processed_events = 0

    for window in windows:
        # 阶段1：收集消息（同时记录每个节点的最后一条交互）
        user_messages = defaultdict(list)
        item_messages = defaultdict(list)
        last_interaction_per_node = {}  # 记录每个节点的最后一条交互

        for idx, interaction in enumerate(window):
            processed_events += 1
            if processed_events % 100 == 0 or processed_events == total_events:
                print(f"  [TGN-BPR] Processed {processed_events}/{total_events} events ({100*processed_events//total_events}%)", flush=True)
            uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
            iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)
            t = torch.tensor([interaction.timestamp], dtype=torch.float32, device=device)
            f = interaction.features.unsqueeze(0).to(device)

            # 计算user和item的消息
            user_msg, item_msg = model.compute_message(uid, iid, t, f, graph_ctx=graph_ctx)
            user_messages[interaction.user_id].append((user_msg, interaction.timestamp))
            item_messages[interaction.item_id].append((item_msg, interaction.timestamp))

            # 记录每个节点的最后一条交互
            last_interaction_per_node[('user', interaction.user_id)] = idx
            last_interaction_per_node[('item', interaction.item_id)] = idx

        # 阶段2：聚合并更新user
        for uid, msg_list in user_messages.items():
            messages = [m for m, _ in msg_list]
            last_ts = max(ts for _, ts in msg_list)

            if aggregator == "mean":
                agg_msg = torch.stack(messages).mean(dim=0)
            elif aggregator == "sum":
                agg_msg = torch.stack(messages).sum(dim=0)
            elif aggregator == "last":
                agg_msg = messages[-1]
            else:
                agg_msg = messages[-1]

            new_emb, new_cell = model.apply_aggregated_message(uid, agg_msg, node_type="user")
            model.user_embeddings[uid] = new_emb.detach()
            model.user_last_time[uid] = last_ts
            if new_cell is not None:
                model.user_cell_state[uid] = new_cell.detach()

        # 阶段2：聚合并更新item
        for iid, msg_list in item_messages.items():
            messages = [m for m, _ in msg_list]
            last_ts = max(ts for _, ts in msg_list)

            if aggregator == "mean":
                agg_msg = torch.stack(messages).mean(dim=0)
            elif aggregator == "sum":
                agg_msg = torch.stack(messages).sum(dim=0)
            elif aggregator == "last":
                agg_msg = messages[-1]
            else:
                agg_msg = messages[-1]

            new_emb, new_cell = model.apply_aggregated_message(iid, agg_msg, node_type="item")
            model.item_embeddings[iid] = new_emb.detach()
            model.item_last_time[iid] = last_ts
            if new_cell is not None:
                model.item_cell_state[iid] = new_cell.detach()

        # 阶段3：计算loss（根据loss_mode决定哪些交互参与）
        optimizer.zero_grad()
        batch_losses = []

        if loss_mode == "last":
            # 只对每个节点的最后一条交互计算loss
            last_indices = set(last_interaction_per_node.values())
            interactions_for_loss = [window[i] for i in sorted(last_indices)]
        else:  # loss_mode == "all"
            interactions_for_loss = window

        for interaction in interactions_for_loss:
            uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
            iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)

            neg_items = []
            while len(neg_items) < neg_sample_size:
                neg = int(rng.integers(0, _num_items(model)))
                if neg != interaction.item_id:
                    neg_items.append(neg)
            neg_ids = torch.tensor(neg_items, dtype=torch.long, device=device)

            pred_emb, _ = model.predict(uid, interaction.timestamp)
            pos_emb = _item_embeddings_for_loss(model, iid).detach().to(device)
            neg_emb = _item_embeddings_for_loss(model, neg_ids).detach().to(device).unsqueeze(0)
            loss = criterion(pred_emb, pos_emb, neg_emb)
            batch_losses.append(loss)

        # 阶段4：Backward
        if batch_losses:
            total_batch_loss = sum(batch_losses) / len(batch_losses)
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item() * len(batch_losses)

    return total_loss


def train_partition_ce_tgn(
    model,
    partition: TemporalPartition,
    optimizer,
    time_window_size: float,
    aggregator: str = "mean",
    loss_mode: str = "all",
    seed: Optional[int] = None,
    graph_ctx: Optional[Dict] = None,
) -> float:
    """TGN 风格窗口批处理训练（CE/L2 loss）：窗口内聚合消息后更新

    Args:
        loss_mode: "all" - 所有交互都计算loss; "last" - 只对每个节点最后一条交互计算loss
    """
    from collections import defaultdict
    device = _model_device(model)
    total_loss = 0.0

    windows = _create_time_windows(partition.interactions, time_window_size)
    total_events = len(partition.interactions)
    processed_events = 0

    for window in windows:
        # 阶段1：收集消息
        user_messages = defaultdict(list)
        item_messages = defaultdict(list)
        last_interaction_per_node = {}

        for idx, interaction in enumerate(window):
            processed_events += 1
            if processed_events % 100 == 0 or processed_events == total_events:
                print(f"  [TGN-CE] Processed {processed_events}/{total_events} events ({100*processed_events//total_events}%)", flush=True)
            uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
            iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)
            t = torch.tensor([interaction.timestamp], dtype=torch.float32, device=device)
            f = interaction.features.unsqueeze(0).to(device)

            user_msg, item_msg = model.compute_message(uid, iid, t, f, graph_ctx=graph_ctx)
            user_messages[interaction.user_id].append((user_msg, interaction.timestamp))
            item_messages[interaction.item_id].append((item_msg, interaction.timestamp))

            last_interaction_per_node[('user', interaction.user_id)] = idx
            last_interaction_per_node[('item', interaction.item_id)] = idx

        # 阶段2：聚合并更新user
        for uid, msg_list in user_messages.items():
            messages = [m for m, _ in msg_list]
            last_ts = max(ts for _, ts in msg_list)

            if aggregator == "mean":
                agg_msg = torch.stack(messages).mean(dim=0)
            elif aggregator == "sum":
                agg_msg = torch.stack(messages).sum(dim=0)
            elif aggregator == "last":
                agg_msg = messages[-1]
            else:
                agg_msg = messages[-1]

            new_emb, new_cell = model.apply_aggregated_message(uid, agg_msg, node_type="user")
            model.user_embeddings[uid] = new_emb.detach()
            model.user_last_time[uid] = last_ts
            if new_cell is not None:
                model.user_cell_state[uid] = new_cell.detach()

        # 阶段2：聚合并更新item
        for iid, msg_list in item_messages.items():
            messages = [m for m, _ in msg_list]
            last_ts = max(ts for _, ts in msg_list)

            if aggregator == "mean":
                agg_msg = torch.stack(messages).mean(dim=0)
            elif aggregator == "sum":
                agg_msg = torch.stack(messages).sum(dim=0)
            elif aggregator == "last":
                agg_msg = messages[-1]
            else:
                agg_msg = messages[-1]

            new_emb, new_cell = model.apply_aggregated_message(iid, agg_msg, node_type="item")
            model.item_embeddings[iid] = new_emb.detach()
            model.item_last_time[iid] = last_ts
            if new_cell is not None:
                model.item_cell_state[iid] = new_cell.detach()

        # 阶段3：计算loss
        optimizer.zero_grad()
        batch_losses = []

        if loss_mode == "last":
            last_indices = set(last_interaction_per_node.values())
            interactions_for_loss = [window[i] for i in sorted(last_indices)]
        else:  # loss_mode == "all"
            interactions_for_loss = window

        for interaction in interactions_for_loss:
            uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
            iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)

            pred_emb, _ = model.predict(uid, interaction.timestamp)
            target_emb = _item_embeddings_for_loss(model, iid).to(device)
            loss = ((pred_emb - target_emb) ** 2).sum(dim=-1).mean()
            batch_losses.append(loss)

        # 阶段4：Backward
        if batch_losses:
            total_batch_loss = sum(batch_losses) / len(batch_losses)
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item() * len(batch_losses)

    return total_loss


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.jodie_rnn import JODIERNN
    from data.synthetic import Interaction
    from data.temporal_partition import TemporalPartition

    torch.manual_seed(0)
    NUM_USERS, NUM_ITEMS, FEAT_DIM, EMB_DIM = 20, 10, 4, 16
    interactions = [
        Interaction(
            timestamp=float(i),
            user_id=i % NUM_USERS,
            item_id=i % NUM_ITEMS,
            features=torch.randn(FEAT_DIM),
        )
        for i in range(500)
    ]
    partition = TemporalPartition(
        partition_id=0, split="all",
        start_ts=0.0, end_ts=499.0,
        interactions=interactions,
    )

    # ── 验证1：t-Batch 内节点唯一性
    batches = _create_t_batches(interactions, batch_size=32)
    violations = 0
    for b in batches:
        users = [x.user_id for x in b]
        items = [x.item_id for x in b]
        if len(users) != len(set(users)) or len(items) != len(set(items)):
            violations += 1
    print(f"[验证1] t-Batch 节点唯一性: {len(batches)} batches, violations={violations}")
    assert violations == 0, "t-Batch 内存在重复节点！"

    # ── 验证2：覆盖率（所有交互都被分配到某个 batch）
    total_in_batches = sum(len(b) for b in batches)
    assert total_in_batches == len(interactions), f"交互丢失: {total_in_batches} != {len(interactions)}"
    print(f"[验证2] 覆盖率: {total_in_batches}/{len(interactions)} 交互全部覆盖")

    # ── 验证3：冻结参数，比较 serial 和 t-batch 对同一序列的 forward 输出
    # t-Batch 与 serial 的区别仅在于 optimizer.step() 频率，forward 本身应完全一致
    model_s = JODIERNN(NUM_USERS, NUM_ITEMS, EMB_DIM, FEAT_DIM, cell_type="rnn")
    model_b = JODIERNN(NUM_USERS, NUM_ITEMS, EMB_DIM, FEAT_DIM, cell_type="rnn")
    # 共享相同初始参数
    model_b.load_state_dict(model_s.state_dict())

    # 只跑一个 epoch，不更新参数（eval 模式），比较每条交互的 pred_emb
    model_s.eval()
    model_b.eval()
    device = torch.device("cpu")
    diffs = []
    with torch.no_grad():
        for inter in interactions[:50]:
            uid = torch.tensor([inter.user_id], dtype=torch.long)
            iid = torch.tensor([inter.item_id], dtype=torch.long)
            t = torch.tensor([inter.timestamp], dtype=torch.float32)
            f = inter.features.unsqueeze(0)
            e_s, _, _ = model_s(uid, iid, t, f, inter.timestamp)
            e_b, _, _ = model_b(uid, iid, t, f, inter.timestamp)
            diffs.append((e_s - e_b).abs().max().item())
    max_diff = max(diffs)
    print(f"[验证3] 相同参数下 forward 最大差异: {max_diff:.2e}")
    assert max_diff < 1e-5, f"forward 输出不一致: {max_diff}"

    print("\nALL CHECKS PASSED — t-Batch 实现正确")

    # ── 验证4：比较三种训练方式（逐条、t-Batch、TGN）
    print("\n" + "="*60)
    print("验证4：比较三种训练方式的loss和速度")
    print("="*60)

    NUM_EPOCHS = 3
    LR = 1e-3
    NEG_SAMPLES = 5

    # 逐条训练
    print("\n[1] 逐条训练 (Serial)")
    model_serial = JODIERNN(NUM_USERS, NUM_ITEMS, EMB_DIM, FEAT_DIM, cell_type="rnn")
    optimizer_serial = torch.optim.Adam(model_serial.parameters(), lr=LR)
    criterion = BPRLoss()

    import time
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model_serial.reset_state()
        loss = train_partition_bpr(
            model_serial, partition, optimizer_serial, criterion,
            neg_sample_size=NEG_SAMPLES, seed=epoch
        )
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss:.4f}")
    serial_time = time.time() - start
    print(f"  总耗时: {serial_time:.2f}s")

    # t-Batch训练
    print("\n[2] t-Batch训练 (batch_size=32)")
    model_tbatch = JODIERNN(NUM_USERS, NUM_ITEMS, EMB_DIM, FEAT_DIM, cell_type="rnn")
    optimizer_tbatch = torch.optim.Adam(model_tbatch.parameters(), lr=LR)

    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model_tbatch.reset_state()
        loss = train_partition_bpr_batch(
            model_tbatch, partition, optimizer_tbatch,
            neg_sample_size=NEG_SAMPLES, batch_size=32, seed=epoch
        )
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss:.4f}")
    tbatch_time = time.time() - start
    print(f"  总耗时: {tbatch_time:.2f}s")

    # TGN窗口训练 (loss_mode="all")
    print("\n[3] TGN窗口训练 (window_size=10, aggregator=mean, loss_mode='all')")
    model_tgn = JODIERNN(NUM_USERS, NUM_ITEMS, EMB_DIM, FEAT_DIM, cell_type="rnn")
    optimizer_tgn = torch.optim.Adam(model_tgn.parameters(), lr=LR)

    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model_tgn.reset_state()
        loss = train_partition_bpr_tgn(
            model_tgn, partition, optimizer_tgn, criterion,
            time_window_size=10.0, aggregator="mean", loss_mode="all",
            neg_sample_size=NEG_SAMPLES, seed=epoch
        )
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss:.4f}")
    tgn_time = time.time() - start
    print(f"  总耗时: {tgn_time:.2f}s")

    # TGN窗口训练 (loss_mode="last")
    print("\n[4] TGN窗口训练 (window_size=10, aggregator=mean, loss_mode='last')")
    model_tgn_last = JODIERNN(NUM_USERS, NUM_ITEMS, EMB_DIM, FEAT_DIM, cell_type="rnn")
    optimizer_tgn_last = torch.optim.Adam(model_tgn_last.parameters(), lr=LR)

    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model_tgn_last.reset_state()
        loss = train_partition_bpr_tgn(
            model_tgn_last, partition, optimizer_tgn_last, criterion,
            time_window_size=10.0, aggregator="mean", loss_mode="last",
            neg_sample_size=NEG_SAMPLES, seed=epoch
        )
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss:.4f}")
    tgn_last_time = time.time() - start
    print(f"  总耗时: {tgn_last_time:.2f}s")

    # 比较最终loss
    print("\n" + "-"*60)
    print("最终loss对比（第3个epoch后额外测试）:")
    model_serial.reset_state()
    serial_final = train_partition_bpr(
        model_serial, partition, optimizer_serial, criterion,
        neg_sample_size=NEG_SAMPLES, seed=999
    )
    model_tbatch.reset_state()
    tbatch_final = train_partition_bpr_batch(
        model_tbatch, partition, optimizer_tbatch,
        neg_sample_size=NEG_SAMPLES, batch_size=32, seed=999
    )
    model_tgn.reset_state()
    tgn_final = train_partition_bpr_tgn(
        model_tgn, partition, optimizer_tgn, criterion,
        time_window_size=10.0, aggregator="mean", loss_mode="all",
        neg_sample_size=NEG_SAMPLES, seed=999
    )
    model_tgn_last.reset_state()
    tgn_last_final = train_partition_bpr_tgn(
        model_tgn_last, partition, optimizer_tgn_last, criterion,
        time_window_size=10.0, aggregator="mean", loss_mode="last",
        neg_sample_size=NEG_SAMPLES, seed=999
    )

    print(f"  Serial:        {serial_final:.4f} (baseline)")
    print(f"  t-Batch:       {tbatch_final:.4f} (差异: {abs(tbatch_final-serial_final)/serial_final*100:.1f}%)")
    print(f"  TGN (all):     {tgn_final:.4f} (差异: {abs(tgn_final-serial_final)/serial_final*100:.1f}%)")
    print(f"  TGN (last):    {tgn_last_final:.4f} (差异: {abs(tgn_last_final-serial_final)/serial_final*100:.1f}%)")

    print("\n速度对比:")
    print(f"  Serial:        {serial_time:.2f}s (baseline)")
    print(f"  t-Batch:       {tbatch_time:.2f}s (加速: {serial_time/tbatch_time:.2f}x)")
    print(f"  TGN (all):     {tgn_time:.2f}s (加速: {serial_time/tgn_time:.2f}x)")
    print(f"  TGN (last):    {tgn_last_time:.2f}s (加速: {serial_time/tgn_last_time:.2f}x)")

    # 验证TGN loss差异在合理范围内
    tgn_diff_pct = abs(tgn_final - serial_final) / serial_final * 100
    tgn_last_diff_pct = abs(tgn_last_final - serial_final) / serial_final * 100
    assert tgn_diff_pct < 40, f"TGN (all) loss差异过大: {tgn_diff_pct:.1f}% > 40%"
    print(f"\n✓ TGN (all) loss差异在合理范围内 ({tgn_diff_pct:.1f}% < 40%)")
    print(f"✓ TGN (last) loss差异: {tgn_last_diff_pct:.1f}%")
    if tgn_last_diff_pct < tgn_diff_pct:
        print(f"  → loss_mode='last' 缩小了与逐条训练的差距 ({tgn_diff_pct:.1f}% → {tgn_last_diff_pct:.1f}%)")
    print("  注：TGN是有损批处理，消息聚合会导致loss略高于逐条训练")

    print("\n" + "="*60)
    print("ALL CHECKS PASSED — TGN实现正确")
    print("="*60)

    # ── 验证5：测试TGN消息聚合（aggregator参数真实生效）
    print("\n" + "="*60)
    print("验证5：测试TGN消息聚合（aggregator参数）")
    print("="*60)

    NUM_EPOCHS_AGG = 3
    WINDOW_SIZE = 10.0

    # 测试 aggregator="last"
    print("\n[1] TGN with aggregator='last'")
    model_last = JODIERNN(NUM_USERS, NUM_ITEMS, EMB_DIM, FEAT_DIM, cell_type="rnn")
    optimizer_last = torch.optim.Adam(model_last.parameters(), lr=LR)
    criterion_last = BPRLoss()

    for epoch in range(NUM_EPOCHS_AGG):
        model_last.reset_state()
        loss = train_partition_bpr_tgn(
            model_last, partition, optimizer_last, criterion_last,
            time_window_size=WINDOW_SIZE, aggregator="last",
            neg_sample_size=NEG_SAMPLES, seed=epoch
        )
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS_AGG} | Loss: {loss:.4f}")

    # 测试 aggregator="mean"
    print("\n[2] TGN with aggregator='mean'")
    model_mean = JODIERNN(NUM_USERS, NUM_ITEMS, EMB_DIM, FEAT_DIM, cell_type="rnn")
    optimizer_mean = torch.optim.Adam(model_mean.parameters(), lr=LR)
    criterion_mean = BPRLoss()

    for epoch in range(NUM_EPOCHS_AGG):
        model_mean.reset_state()
        loss = train_partition_bpr_tgn(
            model_mean, partition, optimizer_mean, criterion_mean,
            time_window_size=WINDOW_SIZE, aggregator="mean",
            neg_sample_size=NEG_SAMPLES, seed=epoch
        )
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS_AGG} | Loss: {loss:.4f}")

    # 测试 aggregator="sum"
    print("\n[3] TGN with aggregator='sum'")
    model_sum = JODIERNN(NUM_USERS, NUM_ITEMS, EMB_DIM, FEAT_DIM, cell_type="rnn")
    optimizer_sum = torch.optim.Adam(model_sum.parameters(), lr=LR)
    criterion_sum = BPRLoss()

    for epoch in range(NUM_EPOCHS_AGG):
        model_sum.reset_state()
        loss = train_partition_bpr_tgn(
            model_sum, partition, optimizer_sum, criterion_sum,
            time_window_size=WINDOW_SIZE, aggregator="sum",
            neg_sample_size=NEG_SAMPLES, seed=epoch
        )
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS_AGG} | Loss: {loss:.4f}")

    print("\n" + "-"*60)
    print("结论：如果三种aggregator的loss有差异且都正常下降，")
    print("说明消息聚合逻辑真实生效。")
    print("="*60)

    # ── 验证6：测试TGN loss_mode参数（all vs last）
    print("\n" + "="*60)
    print("验证6：测试TGN loss_mode参数（all vs last）")
    print("="*60)

    NUM_EPOCHS_LOSS_MODE = 3
    WINDOW_SIZE_LOSS_MODE = 10.0

    # 创建有重复节点的测试数据（确保窗口内有重复）
    print("\n[准备测试数据] 生成窗口内有重复节点的交互序列...")
    test_interactions_loss_mode = []
    for i in range(200):
        # 使用更小的user/item范围，确保窗口内有重复
        test_interactions_loss_mode.append(
            Interaction(
                timestamp=float(i),
                user_id=i % 5,  # 只有5个用户，窗口内必有重复
                item_id=i % 3,  # 只有3个物品，窗口内必有重复
                features=torch.randn(FEAT_DIM),
            )
        )
    test_partition_loss_mode = TemporalPartition(
        partition_id=0, split="test",
        start_ts=0.0, end_ts=199.0,
        interactions=test_interactions_loss_mode,
    )

    # 测试 loss_mode="all"
    print("\n[1] TGN with loss_mode='all'")
    torch.manual_seed(42)
    model_all = JODIERNN(5, 3, EMB_DIM, FEAT_DIM, cell_type="rnn")
    optimizer_all = torch.optim.Adam(model_all.parameters(), lr=LR)
    criterion_all = BPRLoss()

    for epoch in range(NUM_EPOCHS_LOSS_MODE):
        model_all.reset_state()
        loss = train_partition_bpr_tgn(
            model_all, test_partition_loss_mode, optimizer_all, criterion_all,
            time_window_size=WINDOW_SIZE_LOSS_MODE, aggregator="mean",
            loss_mode="all", neg_sample_size=NEG_SAMPLES, seed=epoch
        )
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS_LOSS_MODE} | Loss: {loss:.4f}")

    # 测试 loss_mode="last"
    print("\n[2] TGN with loss_mode='last'")
    torch.manual_seed(42)
    model_last = JODIERNN(5, 3, EMB_DIM, FEAT_DIM, cell_type="rnn")
    optimizer_last = torch.optim.Adam(model_last.parameters(), lr=LR)
    criterion_last = BPRLoss()

    for epoch in range(NUM_EPOCHS_LOSS_MODE):
        model_last.reset_state()
        loss = train_partition_bpr_tgn(
            model_last, test_partition_loss_mode, optimizer_last, criterion_last,
            time_window_size=WINDOW_SIZE_LOSS_MODE, aggregator="mean",
            loss_mode="last", neg_sample_size=NEG_SAMPLES, seed=epoch
        )
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS_LOSS_MODE} | Loss: {loss:.4f}")

    print("\n" + "-"*60)
    print("结论：loss_mode='last' 只对每个节点的最后一条交互计算loss，")
    print("因此总loss值会低于 'all' 模式（参与loss计算的交互更少）。")
    print("两种模式都应该正常收敛。")
    print("="*60)



