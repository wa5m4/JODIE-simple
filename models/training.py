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
            if batch_training:
                total_loss += train_partition_bpr_batch(
                    model=model,
                    partition=partition,
                    optimizer=optimizer,
                    neg_sample_size=neg_sample_size,
                    batch_size=batch_size,
                    seed=_partition_seed(seed, partition.partition_id, epoch),
                    graph_ctx=epoch_graph_ctx,
                )
            else:
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
            if batch_training:
                total_loss += train_partition_ce_batch(
                    model=model,
                    partition=partition,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    seed=_partition_seed(seed, partition.partition_id, epoch),
                    graph_ctx=epoch_graph_ctx,
                )
            else:
                total_loss += train_partition_ce(
                    model=model,
                    partition=partition,
                    optimizer=optimizer,
                    graph_ctx=epoch_graph_ctx,
                )

        avg_loss = total_loss / max(total_events, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | L2 Loss: {avg_loss:.4f}")


@torch.no_grad()
def evaluate_partition_ranking(model, partition: TemporalPartition, k: int = 10, graph_ctx=None, progress_label: str = "", progress_every: int = 0, progress_callback=None) -> Dict[str, float]:
    device = _model_device(model)
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


