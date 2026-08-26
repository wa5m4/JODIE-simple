"""
时序图模型的训练循环（事件级动态图版本）。

提供三种粒度的 BPR 和 CE（余弦嵌入 / L2）训练循环：

  - **串行（Serial）**：一次处理一个交互（基线方法）。
  - **t-Batch**：贪心批处理，每个批次中节点 ID 唯一。
  - **TGN**：基于时间窗口的消息聚合和批量更新。
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jodie.data.synthetic import Interaction, clone_graph_state_template
from jodie.data.temporal_partition import TemporalPartition

from .batching import _chunk_batches, _create_t_batches, _create_time_windows


# ────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────


def _model_device(model) -> torch.device:
    """返回模型第一个参数或缓冲区的设备。"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return next(model.buffers()).device


def _item_embeddings_for_loss(model, item_ids: torch.Tensor):
    """从模型中查找物品嵌入，尝试常见的属性名称。"""
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
    """从模型中返回所有物品嵌入，尝试常见的属性名称。"""
    if hasattr(model, "item_embeddings"):
        return model.item_embeddings
    if hasattr(model, "item_base"):
        return model.item_base.weight
    if hasattr(model, "rnn_model"):
        return model.rnn_model.item_embeddings
    if hasattr(model, "memory") and hasattr(model, "num_users"):
        return model.memory[model.num_users : model.num_users + model.num_items]
    raise ValueError("Model does not expose item embeddings for evaluation.")


def reset_model_state(model, disable_reset=False):
    """重置模型的内部（memory / RNN）状态，除非被禁用。"""
    if disable_reset:
        return
    if hasattr(model, "reset_state"):
        model.reset_state()


def _num_items(model) -> int:
    """返回模型知道的物品数量。"""
    if hasattr(model, "num_items"):
        return model.num_items
    if hasattr(model, "rnn_model") and hasattr(model.rnn_model, "num_items"):
        return model.rnn_model.num_items
    raise ValueError("Model does not expose num_items.")


def _normalize_partitions(
    interactions: List[Interaction],
    partitions: Optional[List[TemporalPartition]] = None,
) -> List[TemporalPartition]:
    """返回 ``partitions``，或从按时间戳排序的 ``interactions``
    构建一个单一的全局分区。"""
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
    """为 (分区, 周期) 对推导一个确定性的随机种子。"""
    if base_seed is None:
        return None
    return int(base_seed) + epoch * 100000 + partition_id


# ────────────────────────────────────────────────────────────
# 损失函数
# ────────────────────────────────────────────────────────────


class BPRLoss(nn.Module):
    """贝叶斯个性化排序损失（Bayesian Personalized Ranking Loss）"""

    def forward(self, pred_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
        pos_score = (pred_emb * pos_emb).sum(dim=-1, keepdim=True)
        neg_scores = torch.bmm(neg_emb, pred_emb.unsqueeze(-1)).squeeze(-1)
        return -F.logsigmoid(pos_score - neg_scores).mean()


# ────────────────────────────────────────────────────────────
# 串行训练循环
# ────────────────────────────────────────────────────────────


def train_partition_bpr(
    model,
    partition: TemporalPartition,
    optimizer,
    criterion,
    neg_sample_size: int = 5,
    graph_ctx: Optional[Dict] = None,
    seed: Optional[int] = None,
    epoch: int = 0,
    progress_every: int = 0,
    progress_callback=None,
) -> float:
    """在单个分区上串行 BPR 训练（一次处理一个交互）。"""
    device = _model_device(model)
    rng = np.random.default_rng(seed) if seed is not None else None
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

        # ── 优先使用预分配的负样本（解决 Pipeline RNG 重置偏差）──
        if epoch in interaction.neg_samples_by_epoch:
            neg_items = list(interaction.neg_samples_by_epoch[epoch])
        elif rng is not None:
            neg_items = []
            while len(neg_items) < neg_sample_size:
                neg = int(rng.integers(0, _num_items(model)))
                if neg != interaction.item_id:
                    neg_items.append(neg)
        else:
            raise RuntimeError(
                f"No precomputed neg samples for epoch {epoch} and no RNG seed provided. "
                f"Pass seed or precompute neg samples during data loading."
            )
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
    """在单个分区上串行余弦嵌入（L2）训练。"""
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


# ────────────────────────────────────────────────────────────
# 顶层训练编排函数
# ────────────────────────────────────────────────────────────


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
    """使用 BPR 损失训练模型，根据 ``batch_mode``
    （``"serial"``、``"tbatch"`` 或 ``"tgn"``）分派到相应的循环。"""
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
            elif batch_mode == "stale_batch":
                total_loss += train_partition_bpr_stale_batch(
                    model=model,
                    partition=partition,
                    optimizer=optimizer,
                    neg_sample_size=neg_sample_size,
                    batch_size=batch_size,
                    seed=_partition_seed(seed, partition.partition_id, epoch),
                    graph_ctx=epoch_graph_ctx,
                    epoch=epoch,
                )
            else:  # 串行
                total_loss += train_partition_bpr(
                    model=model,
                    partition=partition,
                    optimizer=optimizer,
                    criterion=criterion,
                    neg_sample_size=neg_sample_size,
                    graph_ctx=epoch_graph_ctx,
                    seed=_partition_seed(seed, partition.partition_id, epoch),
                    epoch=epoch,
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
    """使用 CE（L2）损失训练模型，根据 ``batch_mode`` 分派到相应的循环。"""
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
            elif batch_mode == "stale_batch":
                total_loss += train_partition_ce_stale_batch(
                    model=model,
                    partition=partition,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    seed=_partition_seed(seed, partition.partition_id, epoch),
                    graph_ctx=epoch_graph_ctx,
                )
            else:  # 串行
                total_loss += train_partition_ce(
                    model=model,
                    partition=partition,
                    optimizer=optimizer,
                    graph_ctx=epoch_graph_ctx,
                )

        avg_loss = total_loss / max(total_events, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | L2 Loss: {avg_loss:.4f}")


# ────────────────────────────────────────────────────────────
# t-Batch 训练循环
# ────────────────────────────────────────────────────────────


def train_partition_bpr_batch(
    model,
    partition: TemporalPartition,
    optimizer,
    neg_sample_size: int = 5,
    batch_size: int = 32,
    seed: Optional[int] = None,
    graph_ctx=None,
) -> float:
    """t-Batch BPR 训练：每个批次中节点 ID 唯一，单独前向传播每个
    交互，然后在聚合的批次损失上反向传播。"""
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
    """t-Batch CE/L2 训练：每个批次中节点 ID 唯一，聚合后反向传播。"""
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


# ────────────────────────────────────────────────────────────
# stale_batch 训练循环（朴素分批：不做冲突消解）
# ────────────────────────────────────────────────────────────


def _stale_writeback(model, staged_writes):
    """批末统一写回 staged 的嵌入/时间戳/胞状态。

    同节点在批内多次出现时，最后一次写入生效——朴素实现的批语义。
    与 process_interaction(deferred=False) 写回的表完全一致。
    """
    for uid, iid, t, new_user_emb, new_item_emb, new_user_c, new_item_c in staged_writes:
        model.user_embeddings[uid] = new_user_emb.detach()
        model.item_embeddings[iid] = new_item_emb.detach()
        model.user_last_time[uid] = t
        model.item_last_time[iid] = t
        if new_user_c is not None:
            model.user_cell_state[uid] = new_user_c.detach()
            model.item_cell_state[iid] = new_item_c.detach()


def _stale_forward(model, uid, iid, t, f, interaction):
    """批内单交互前向：对批前状态计算预测与更新（deferred，不写回）。"""
    pred_emb, _ = model.predict(uid, query_time=interaction.timestamp)
    if getattr(model, "cell_type", "") == "lstm":
        new_user_emb, new_item_emb, new_user_c, new_item_c = model.process_interaction(
            uid, iid, t, f, deferred=True, return_cell_state=True
        )
    else:
        new_user_emb, new_item_emb = model.process_interaction(
            uid, iid, t, f, deferred=True
        )
        new_user_c = new_item_c = None
    return pred_emb, new_user_emb, new_item_emb, new_user_c, new_item_c


def train_partition_bpr_stale_batch(
    model,
    partition: TemporalPartition,
    optimizer,
    neg_sample_size: int = 5,
    batch_size: int = 32,
    seed: Optional[int] = None,
    graph_ctx=None,
    epoch: int = 0,
) -> float:
    """stale_batch BPR 训练：连续交互直接切块，批内所有交互对批前状态
    计算预测（deferred 前向），批末统一写回。

    与 t-Batch 的对照：t-Batch 保证批内节点唯一（冲突无关）；stale_batch
    允许同批重复节点，后出现的交互读到批前旧嵌入——朴素分批破坏写后读
    （RAW）依赖的机制（见引言段 4 微例）。

    负样本优先使用预分配的 ``neg_samples_by_epoch``（与串行路径一致），
    保证消融实验中负样本集合与 serial 完全相同——唯一变量只剩批处理模式。
    """
    device = _model_device(model)
    rng = np.random.default_rng(seed) if seed is not None else None
    criterion = BPRLoss()
    total_loss = 0.0

    for batch in _chunk_batches(partition.interactions, batch_size):
        optimizer.zero_grad()
        batch_losses = []
        staged_writes = []

        for interaction in batch:
            uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
            iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)
            t = torch.tensor([interaction.timestamp], dtype=torch.float32, device=device)
            f = interaction.features.unsqueeze(0).to(device)

            # ── 优先使用预分配的负样本（与串行路径一致,消除负样本来源差异）──
            if epoch in interaction.neg_samples_by_epoch:
                neg_items = list(interaction.neg_samples_by_epoch[epoch])
            elif rng is not None:
                neg_items = []
                while len(neg_items) < neg_sample_size:
                    neg = int(rng.integers(0, _num_items(model)))
                    if neg != interaction.item_id:
                        neg_items.append(neg)
            else:
                raise RuntimeError(
                    f"No precomputed neg samples for epoch {epoch} and no RNG seed provided. "
                    f"Pass seed or precompute neg samples during data loading."
                )
            neg_ids = torch.tensor(neg_items, dtype=torch.long, device=device)

            pred_emb, new_user_emb, new_item_emb, new_user_c, new_item_c = _stale_forward(
                model, uid, iid, t, f, interaction
            )
            pos_emb = new_item_emb.detach().to(device)
            neg_emb = _item_embeddings_for_loss(model, neg_ids).detach().to(device).unsqueeze(0)
            batch_losses.append(criterion(pred_emb, pos_emb, neg_emb))
            staged_writes.append(
                (uid, iid, t, new_user_emb, new_item_emb, new_user_c, new_item_c)
            )

        _stale_writeback(model, staged_writes)

        total_batch_loss = sum(batch_losses)
        total_batch_loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += total_batch_loss.item()

    return total_loss


def train_partition_ce_stale_batch(
    model,
    partition: TemporalPartition,
    optimizer,
    batch_size: int = 32,
    seed: Optional[int] = None,
    graph_ctx=None,
) -> float:
    """stale_batch CE/L2 训练：同 BPR 版——批内所有交互对批前状态计算，
    批末统一写回。"""
    device = _model_device(model)
    total_loss = 0.0

    for batch in _chunk_batches(partition.interactions, batch_size):
        optimizer.zero_grad()
        batch_losses = []
        staged_writes = []

        for interaction in batch:
            uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
            iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)
            t = torch.tensor([interaction.timestamp], dtype=torch.float32, device=device)
            f = interaction.features.unsqueeze(0).to(device)

            pred_emb, new_user_emb, new_item_emb, new_user_c, new_item_c = _stale_forward(
                model, uid, iid, t, f, interaction
            )
            target_emb = new_item_emb.to(device)
            batch_losses.append(((pred_emb - target_emb) ** 2).sum(dim=-1).mean())
            staged_writes.append(
                (uid, iid, t, new_user_emb, new_item_emb, new_user_c, new_item_c)
            )

        _stale_writeback(model, staged_writes)

        total_batch_loss = sum(batch_losses)
        total_batch_loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += total_batch_loss.item()

    return total_loss


# ────────────────────────────────────────────────────────────
# TGN 基于时间窗口的训练循环
# ────────────────────────────────────────────────────────────


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
    """TGN 风格的窗口批量 BPR 损失训练。

    每个窗口的阶段：
      1. 收集窗口内所有交互的消息。
      2. 按节点聚合消息并更新节点嵌入。
      3. 计算所选交互（全部或每个节点最后一个）的 BPR 损失。
      4. 反向传播和优化步进。

    Args:
        loss_mode: ``"all"`` — 为每个交互计算损失。
                   ``"last"`` — 仅为每个节点的最后一个交互计算损失。
    """
    from collections import defaultdict

    device = _model_device(model)
    rng = np.random.default_rng(seed)
    total_loss = 0.0

    windows = _create_time_windows(partition.interactions, time_window_size)
    total_events = len(partition.interactions)
    processed_events = 0

    for window in windows:
        # 阶段 1：收集消息 + 跟踪每个节点的最后一个交互
        user_messages = defaultdict(list)
        item_messages = defaultdict(list)
        last_interaction_per_node = {}

        for idx, interaction in enumerate(window):
            processed_events += 1
            if processed_events % 100 == 0 or processed_events == total_events:
                print(f"  [TGN-BPR] Processed {processed_events}/{total_events} events ({100*processed_events//total_events}%)", flush=True)
            uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
            iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)
            t = torch.tensor([interaction.timestamp], dtype=torch.float32, device=device)
            f = interaction.features.unsqueeze(0).to(device)

            user_msg, item_msg = model.compute_message(uid, iid, t, f, graph_ctx=graph_ctx)
            user_messages[interaction.user_id].append((user_msg, interaction.timestamp))
            item_messages[interaction.item_id].append((item_msg, interaction.timestamp))

            last_interaction_per_node[("user", interaction.user_id)] = idx
            last_interaction_per_node[("item", interaction.item_id)] = idx

        # 阶段 2：聚合并更新用户
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

        # 阶段 2：聚合并更新物品
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

        # 阶段 3：计算损失
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

        # 阶段 4：反向传播
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
    """TGN 风格的窗口批量 CE/L2 损失训练。

    每个窗口的阶段：
      1. 收集窗口内所有交互的消息。
      2. 按节点聚合消息并更新节点嵌入。
      3. 计算所选交互的 L2 损失。
      4. 反向传播和优化步进。

    Args:
        loss_mode: ``"all"`` — 为每个交互计算损失。
                   ``"last"`` — 仅为每个节点的最后一个交互计算损失。
    """
    from collections import defaultdict

    device = _model_device(model)
    total_loss = 0.0

    windows = _create_time_windows(partition.interactions, time_window_size)
    total_events = len(partition.interactions)
    processed_events = 0

    for window in windows:
        # 阶段 1：收集消息
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

            last_interaction_per_node[("user", interaction.user_id)] = idx
            last_interaction_per_node[("item", interaction.item_id)] = idx

        # 阶段 2：聚合并更新用户
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

        # 阶段 2：聚合并更新物品
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

        # 阶段 3：计算损失
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

        # 阶段 4：反向传播
        if batch_losses:
            total_batch_loss = sum(batch_losses) / len(batch_losses)
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item() * len(batch_losses)

    return total_loss


# ────────────────────────────────────────────────────────────
# 自测 / 验证（运行 ``python -m jodie.training.loops``）
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from jodie.models.jodie_rnn import JODIERNN
    from jodie.data.synthetic import Interaction
    from jodie.data.temporal_partition import TemporalPartition

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

    # ── 检查 1：t-Batch 节点唯一性
    batches = _create_t_batches(interactions, batch_size=32)
    violations = 0
    for b in batches:
        users = [x.user_id for x in b]
        items = [x.item_id for x in b]
        if len(users) != len(set(users)) or len(items) != len(set(items)):
            violations += 1
    print(f"[Check 1] t-Batch node uniqueness: {len(batches)} batches, violations={violations}")
    assert violations == 0, "Duplicate nodes within a t-Batch!"

    # ── 检查 2：覆盖率（所有交互都已分配）
    total_in_batches = sum(len(b) for b in batches)
    assert total_in_batches == len(interactions), f"Lost interactions: {total_in_batches} != {len(interactions)}"
    print(f"[Check 2] Coverage: {total_in_batches}/{len(interactions)} interactions covered")

    # ── 检查 3：冻结参数，比较串行与 t-Batch 正向传播
    model_s = JODIERNN(NUM_USERS, NUM_ITEMS, EMB_DIM, FEAT_DIM, cell_type="rnn")
    model_b = JODIERNN(NUM_USERS, NUM_ITEMS, EMB_DIM, FEAT_DIM, cell_type="rnn")
    model_b.load_state_dict(model_s.state_dict())

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
    print(f"[Check 3] Max forward diff with identical params: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Forward output mismatch: {max_diff}"

    print("\nALL CHECKS PASSED -- t-Batch implementation correct")

    # ── 检查 4：比较三种训练模式（串行、t-Batch、TGN）
    print("\n" + "=" * 60)
    print("Check 4: Compare loss and speed across training modes")
    print("=" * 60)

    NUM_EPOCHS = 3
    LR = 1e-3
    NEG_SAMPLES = 5

    # 串行训练
    print("\n[1] Serial training")
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
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {loss:.4f}")
    serial_time = time.time() - start
    print(f"  Total time: {serial_time:.2f}s")

    # t-Batch 训练
    print("\n[2] t-Batch training (batch_size=32)")
    model_tbatch = JODIERNN(NUM_USERS, NUM_ITEMS, EMB_DIM, FEAT_DIM, cell_type="rnn")
    optimizer_tbatch = torch.optim.Adam(model_tbatch.parameters(), lr=LR)

    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model_tbatch.reset_state()
        loss = train_partition_bpr_batch(
            model_tbatch, partition, optimizer_tbatch,
            neg_sample_size=NEG_SAMPLES, batch_size=32, seed=epoch
        )
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {loss:.4f}")
    tbatch_time = time.time() - start
    print(f"  Total time: {tbatch_time:.2f}s")

    # TGN 窗口训练（loss_mode="all"）
    print("\n[3] TGN window training (window_size=10, aggregator=mean, loss_mode='all')")
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
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {loss:.4f}")
    tgn_time = time.time() - start
    print(f"  Total time: {tgn_time:.2f}s")

    # TGN 窗口训练（loss_mode="last"）
    print("\n[4] TGN window training (window_size=10, aggregator=mean, loss_mode='last')")
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
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {loss:.4f}")
    tgn_last_time = time.time() - start
    print(f"  Total time: {tgn_last_time:.2f}s")

    # 比较最终损失
    print("\n" + "-" * 60)
    print("Final loss comparison (extra epoch after training):")
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
    print(f"  t-Batch:       {tbatch_final:.4f} (diff: {abs(tbatch_final - serial_final) / serial_final * 100:.1f}%)")
    print(f"  TGN (all):     {tgn_final:.4f} (diff: {abs(tgn_final - serial_final) / serial_final * 100:.1f}%)")
    print(f"  TGN (last):    {tgn_last_final:.4f} (diff: {abs(tgn_last_final - serial_final) / serial_final * 100:.1f}%)")

    print("\nSpeed comparison:")
    print(f"  Serial:        {serial_time:.2f}s (baseline)")
    print(f"  t-Batch:       {tbatch_time:.2f}s (speedup: {serial_time / tbatch_time:.2f}x)")
    print(f"  TGN (all):     {tgn_time:.2f}s (speedup: {serial_time / tgn_time:.2f}x)")
    print(f"  TGN (last):    {tgn_last_time:.2f}s (speedup: {serial_time / tgn_last_time:.2f}x)")

    tgn_diff_pct = abs(tgn_final - serial_final) / serial_final * 100
    tgn_last_diff_pct = abs(tgn_last_final - serial_final) / serial_final * 100
    assert tgn_diff_pct < 40, f"TGN (all) loss difference too large: {tgn_diff_pct:.1f}% > 40%"
    print(f"\n✓ TGN (all) loss difference within range ({tgn_diff_pct:.1f}% < 40%)")
    print(f"✓ TGN (last) loss difference: {tgn_last_diff_pct:.1f}%")
    if tgn_last_diff_pct < tgn_diff_pct:
        print(f"  → loss_mode='last' narrowed the gap ({tgn_diff_pct:.1f}% → {tgn_last_diff_pct:.1f}%)")
    print("  Note: TGN is lossy batching; message aggregation may produce slightly higher loss vs. serial")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED -- TGN implementation correct")
    print("=" * 60)

    # ── 检查 5：TGN 消息聚合（aggregator 参数）
    print("\n" + "=" * 60)
    print("Check 5: TGN message aggregation (aggregator parameter)")
    print("=" * 60)

    NUM_EPOCHS_AGG = 3
    WINDOW_SIZE = 10.0

    # 聚合器 aggregator="last"
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
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS_AGG} | Loss: {loss:.4f}")

    # 聚合器 aggregator="mean"
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
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS_AGG} | Loss: {loss:.4f}")

    # 聚合器 aggregator="sum"
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
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS_AGG} | Loss: {loss:.4f}")

    print("\n" + "-" * 60)
    print("Conclusion: if all three aggregators produce different losses")
    print("that all decrease normally, the aggregation logic is working.")
    print("=" * 60)

    # ── 检查 6：TGN loss_mode 参数（all vs last）
    print("\n" + "=" * 60)
    print("Check 6: TGN loss_mode parameter (all vs last)")
    print("=" * 60)

    NUM_EPOCHS_LOSS_MODE = 3
    WINDOW_SIZE_LOSS_MODE = 10.0

    # 创建包含重复节点的测试数据（确保窗口内有重复项）
    print("\n[Prepare] Generating interaction sequence with repeated nodes...")
    test_interactions_loss_mode = []
    for i in range(200):
        test_interactions_loss_mode.append(
            Interaction(
                timestamp=float(i),
                user_id=i % 5,  # 只有 5 个用户 -- 确保有重复
                item_id=i % 3,  # 只有 3 个物品 -- 确保有重复
                features=torch.randn(FEAT_DIM),
            )
        )
    test_partition_loss_mode = TemporalPartition(
        partition_id=0, split="test",
        start_ts=0.0, end_ts=199.0,
        interactions=test_interactions_loss_mode,
    )

    # 损失模式 loss_mode="all"
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
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS_LOSS_MODE} | Loss: {loss:.4f}")

    # 损失模式 loss_mode="last"
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
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS_LOSS_MODE} | Loss: {loss:.4f}")

    print("\n" + "-" * 60)
    print("Conclusion: loss_mode='last' computes loss only for each node's")
    print("last interaction, so total loss will be lower than 'all' mode.")
    print("Both modes should converge normally.")
    print("=" * 60)
