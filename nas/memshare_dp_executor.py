"""
MemShare-DP Executor: Data-parallel baseline augmented with MemShare's
Hotspot Memory Sharing idea (VLDB 2025, p3093-wang.pdf).

Key difference from plain DataParallelExecutor:
  - Identifies "hot nodes" (top-k% most frequent users/items across partitions)
  - Hot-node embeddings are shared across workers and merged with smooth
    aggregation (weighted average by interaction count) instead of max-timestamp.
  - Cold-node embeddings still use max-timestamp merge (same as baseline DP).

This isolates the contribution of MemShare's core idea on top of data parallelism.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set

import ray
import torch
import torch.optim as optim

from data.synthetic import Interaction, init_dynamic_graph_state
from data.temporal_partition import TemporalPartition, split_partition_interactions
from models.factory import build_model
from models.training import BPRLoss, _item_embeddings_for_loss, _num_items, evaluate_ranking_metrics


# ---------------------------------------------------------------------------
# Hot-node identification
# ---------------------------------------------------------------------------

def _identify_hot_nodes(
    partitions: List[TemporalPartition],
    top_k_ratio: float = 0.1,
) -> tuple[Set[int], Set[int]]:
    """Return (hot_user_ids, hot_item_ids) — top-k% by interaction frequency."""
    user_cnt: Counter = Counter()
    item_cnt: Counter = Counter()
    for p in partitions:
        for ix in p.interactions:
            user_cnt[ix.user_id] += 1
            item_cnt[ix.item_id] += 1

    def _top_k(cnt: Counter, ratio: float) -> Set[int]:
        if not cnt:
            return set()
        k = max(1, int(len(cnt) * ratio))
        return {uid for uid, _ in cnt.most_common(k)}

    return _top_k(user_cnt, top_k_ratio), _top_k(item_cnt, top_k_ratio)


# ---------------------------------------------------------------------------
# Smooth aggregation for hot nodes
# ---------------------------------------------------------------------------

def _merge_runtime_states_memshare(
    states: List[Optional[Dict[str, Any]]],
    interaction_counts: List[int],
    hot_users: Set[int],
    hot_items: Set[int],
) -> Optional[Dict[str, Any]]:
    """
    Merge worker runtime states with MemShare's smooth aggregation:
    - Hot nodes: weighted average by interaction count (smooth aggregation)
    - Cold nodes: max-timestamp wins (same as baseline DP)
    """
    valid = [(s, c) for s, c in zip(states, interaction_counts) if s is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return {k: v.clone() for k, v in valid[0][0].items()}

    total_count = sum(c for _, c in valid)
    weights = [c / total_count if total_count > 0 else 1.0 / len(valid) for _, c in valid]

    merged = {k: v.clone() for k, v in valid[0][0].items()}

    for prefix in ("user", "item"):
        lt_key = f"{prefix}_last_time"
        emb_key = f"{prefix}_embeddings"
        hot_ids = hot_users if prefix == "user" else hot_items

        if emb_key not in valid[0][0]:
            continue

        n = merged[emb_key].shape[0]
        hot_mask = torch.zeros(n, dtype=torch.bool)
        for hid in hot_ids:
            if hid < n:
                hot_mask[hid] = True

        # Hot nodes: smooth aggregation (weighted average)
        if hot_mask.any():
            smooth_emb = torch.zeros_like(merged[emb_key])
            for (state, _), w in zip(valid, weights):
                if emb_key in state:
                    smooth_emb[hot_mask] += w * state[emb_key][hot_mask]
            merged[emb_key][hot_mask] = smooth_emb[hot_mask]

            # For last_time of hot nodes: use max (most recent)
            if lt_key in merged:
                for state, _ in valid[1:]:
                    if lt_key in state:
                        mask = hot_mask & (state[lt_key] > merged[lt_key])
                        merged[lt_key][mask] = state[lt_key][mask]

        # Cold nodes: max-timestamp wins
        cold_mask = ~hot_mask
        if cold_mask.any() and lt_key in merged:
            for state, _ in valid[1:]:
                if lt_key in state and emb_key in state:
                    mask = cold_mask & (state[lt_key] > merged[lt_key])
                    merged[lt_key][mask] = state[lt_key][mask]
                    merged[emb_key][mask] = state[emb_key][mask]

    # LSTM cell states: same hot/cold split
    for prefix in ("user", "item"):
        ht_key = f"{prefix}_h"
        ct_key = f"{prefix}_c"
        lt_key = f"{prefix}_last_time"
        hot_ids = hot_users if prefix == "user" else hot_items

        if ht_key not in valid[0][0]:
            continue

        n = merged[ht_key].shape[0]
        hot_mask = torch.zeros(n, dtype=torch.bool)
        for hid in hot_ids:
            if hid < n:
                hot_mask[hid] = True

        if hot_mask.any():
            smooth_h = torch.zeros_like(merged[ht_key])
            smooth_c = torch.zeros_like(merged.get(ct_key, merged[ht_key]))
            for (state, _), w in zip(valid, weights):
                if ht_key in state:
                    smooth_h[hot_mask] += w * state[ht_key][hot_mask]
                if ct_key in state:
                    smooth_c[hot_mask] += w * state[ct_key][hot_mask]
            merged[ht_key][hot_mask] = smooth_h[hot_mask]
            if ct_key in merged:
                merged[ct_key][hot_mask] = smooth_c[hot_mask]

        cold_mask = ~hot_mask
        if cold_mask.any() and lt_key in merged:
            for state, _ in valid[1:]:
                if lt_key in state and ht_key in state:
                    mask = cold_mask & (state[lt_key] > merged[lt_key])
                    if ht_key in merged:
                        merged[ht_key][mask] = state[ht_key][mask]
                    if ct_key in merged and ct_key in state:
                        merged[ct_key][mask] = state[ct_key][mask]

    return merged


# ---------------------------------------------------------------------------
# Ray worker (reuses same train_chunk logic as DataParallelExecutor)
# ---------------------------------------------------------------------------

@ray.remote
class _MemShareWorker:
    def train_chunk(
        self,
        model_state_dict: Dict[str, Any],
        runtime_state: Optional[Dict[str, Any]],
        interactions: List[Interaction],
        arch_config: Dict[str, Any],
        base_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        import numpy as np

        if not interactions:
            return {"gradients": {}, "runtime_state": runtime_state, "loss": 0.0, "num_interactions": 0}

        config = dict(base_config)
        config.update(arch_config)
        device = torch.device(config.get("device", "cpu"))

        model = build_model(config)
        model.to(device)
        model.load_state_dict({k: v.to(device) for k, v in model_state_dict.items()})
        if runtime_state is not None and hasattr(model, "import_runtime_state"):
            model.import_runtime_state({k: v.to(device) for k, v in runtime_state.items()})

        model_name = config.get("model", "jodie_rnn")
        graph_ctx = None if model_name == "jodie_rnn" else init_dynamic_graph_state(
            num_users=config.get("num_users", 1),
            num_items=config.get("num_items", 1),
            max_neighbors=config.get("max_neighbors", 20),
        )

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
        optimizer.zero_grad()
        criterion = BPRLoss()
        neg_sample_size = config.get("neg_sample_size", 5)
        num_items = _num_items(model)
        rng = np.random.default_rng(None)
        total_loss = 0.0

        for interaction in interactions:
            uid = torch.tensor([interaction.user_id], dtype=torch.long, device=device)
            iid = torch.tensor([interaction.item_id], dtype=torch.long, device=device)
            t   = torch.tensor([interaction.timestamp], dtype=torch.float32, device=device)
            f   = interaction.features.unsqueeze(0).to(device)

            neg_items = []
            while len(neg_items) < neg_sample_size:
                neg = int(rng.integers(0, num_items))
                if neg != interaction.item_id:
                    neg_items.append(neg)
            neg_ids = torch.tensor(neg_items, dtype=torch.long, device=device)

            pred_emb, _, _ = model(uid, iid, t, f, interaction.timestamp, graph_ctx=graph_ctx)
            pos_emb = _item_embeddings_for_loss(model, iid).detach().to(device)
            neg_emb = _item_embeddings_for_loss(model, neg_ids).detach().to(device).unsqueeze(0)
            loss = criterion(pred_emb, pos_emb, neg_emb)
            loss.backward(retain_graph=True)
            total_loss += loss.item()

        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone().cpu()

        runtime_out = None
        if hasattr(model, "export_runtime_state"):
            runtime_out = {k: v.cpu() for k, v in model.export_runtime_state().items()}

        return {
            "gradients": grads,
            "runtime_state": runtime_out,
            "loss": total_loss,
            "num_interactions": len(interactions),
        }


def _apply_averaged_gradients(
    model_state_dict, avg_grads, arch_config, base_config, runtime_state, optimizer_state
):
    device = torch.device("cpu")
    config = dict(base_config)
    config.update(arch_config)
    model = build_model(config)
    model.to(device)
    model.load_state_dict({k: v.to(device) for k, v in model_state_dict.items()})
    if runtime_state is not None and hasattr(model, "import_runtime_state"):
        model.import_runtime_state({k: v.to(device) for k, v in runtime_state.items()})

    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception:
            pass

    optimizer.zero_grad()
    for name, param in model.named_parameters():
        if name in avg_grads:
            param.grad = avg_grads[name].to(device)
    optimizer.step()

    return (
        {k: v.cpu() for k, v in model.state_dict().items()},
        optimizer.state_dict(),
    )


# ---------------------------------------------------------------------------
# MemShareDPExecutor
# ---------------------------------------------------------------------------

class MemShareDPExecutor:
    """
    Data-parallel executor with MemShare's Hotspot Memory Sharing.

    Compared to DataParallelExecutor:
    - Hot nodes (top-k% by frequency) use smooth aggregation (weighted avg)
      instead of max-timestamp merge, reducing memory staleness.
    - Cold nodes keep max-timestamp merge for efficiency.
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        partition_plan,
        num_workers: int = 3,
        hot_node_ratio: float = 0.1,
    ):
        self.base_config = base_config
        self.partition_plan = partition_plan
        self.num_workers = num_workers
        self.hot_node_ratio = hot_node_ratio

        if not ray.is_initialized():
            import os
            visible = str(base_config.get("data_parallel_visible_gpus", "0,1,2"))
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
            ray.init(ignore_reinit_error=True)

        gpu_frac = base_config.get("data_parallel_worker_gpus", 1.0)
        self._workers = [
            _MemShareWorker.options(num_cpus=1, num_gpus=gpu_frac).remote()
            for _ in range(num_workers)
        ]

        # Pre-compute hot nodes from training partitions
        train_partitions = partition_plan.get_split_partitions("train")
        self.hot_users, self.hot_items = _identify_hot_nodes(train_partitions, hot_node_ratio)
        print(
            f"[MemShare-DP] Hot nodes: {len(self.hot_users)} users, "
            f"{len(self.hot_items)} items (top {hot_node_ratio*100:.0f}%)",
            flush=True,
        )

    def shutdown(self) -> None:
        for w in self._workers:
            try:
                ray.kill(w)
            except Exception:
                pass
        self._workers = []

    def __del__(self):
        self.shutdown()

    def run(
        self,
        arch_configs: List[Dict[str, Any]],
        user_type_prefs=None,
        item_type=None,
        num_train_epochs: int = 1,
    ) -> List[Dict[str, Any]]:
        return [self._run_trial(cfg, i, num_train_epochs) for i, cfg in enumerate(arch_configs)]

    def _run_trial(self, arch_config, trial_id, num_train_epochs):
        config = dict(self.base_config)
        config.update(arch_config)

        model = build_model(config)
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer_state: Optional[Dict] = None

        train_partitions = self.partition_plan.get_split_partitions("train")
        val_partitions = self.partition_plan.get_split_partitions("val")
        t_start = time.time()

        micro_batch_size = int(self.base_config.get("data_parallel_micro_batch_size", 50))

        for _epoch in range(num_train_epochs):
            runtime_state: Optional[Dict] = None

            for partition in train_partitions:
                if not partition.interactions:
                    continue

                interactions = partition.interactions
                total = len(interactions)

                for start_idx in range(0, total, micro_batch_size):
                    end_idx = min(start_idx + micro_batch_size, total)
                    micro_batch = interactions[start_idx:end_idx]

                    chunks = split_partition_interactions(
                        TemporalPartition(
                            partition.partition_id, partition.split,
                            partition.start_ts, partition.end_ts, micro_batch
                        ),
                        self.num_workers,
                    )
                    while len(chunks) < self.num_workers:
                        chunks.append([])

                    refs = [
                        self._workers[i].train_chunk.remote(
                            model_state_dict, runtime_state, chunks[i],
                            arch_config, self.base_config,
                        )
                        for i in range(self.num_workers)
                    ]
                    worker_results = ray.get(refs)

                    total_interactions = sum(r["num_interactions"] for r in worker_results)
                    if total_interactions == 0:
                        continue

                    # AllReduce: weighted gradient average
                    avg_grads: Dict[str, Any] = {}
                    for r in worker_results:
                        w = r["num_interactions"] / total_interactions
                        for name, g in r["gradients"].items():
                            avg_grads[name] = avg_grads.get(name, 0) + g * w

                    model_state_dict, optimizer_state = _apply_averaged_gradients(
                        model_state_dict, avg_grads, arch_config,
                        self.base_config, runtime_state, optimizer_state,
                    )

                    # MemShare: smooth aggregation for hot nodes, max-ts for cold
                    runtime_state = _merge_runtime_states_memshare(
                        [r["runtime_state"] for r in worker_results],
                        [r["num_interactions"] for r in worker_results],
                        self.hot_users,
                        self.hot_items,
                    )

                    if (end_idx % 100 == 0) or (end_idx == total):
                        print(
                            f"[MemShare-DP] trial={trial_id} epoch={_epoch+1} "
                            f"partition={partition.partition_id} "
                            f"interactions={end_idx}/{total}",
                            flush=True,
                        )

        # Evaluation
        eval_device = torch.device("cpu")
        val_model = build_model(config)
        val_model.to(eval_device)
        val_model.load_state_dict({k: v.to(eval_device) for k, v in model_state_dict.items()})
        if runtime_state is not None and hasattr(val_model, "import_runtime_state"):
            val_model.import_runtime_state({k: v.to(eval_device) for k, v in runtime_state.items()})

        model_name = config.get("model", "jodie_rnn")
        eval_graph_ctx = None if model_name == "jodie_rnn" else init_dynamic_graph_state(
            num_users=config.get("num_users", 1),
            num_items=config.get("num_items", 1),
            max_neighbors=config.get("max_neighbors", 20),
        )

        val_interactions: List[Interaction] = []
        for p in val_partitions:
            val_interactions.extend(p.interactions)

        metrics = evaluate_ranking_metrics(
            val_model, val_interactions,
            k=config.get("k", 10),
            graph_ctx=eval_graph_ctx,
            partitions=val_partitions if val_partitions else None,
        )

        selection_metric = config.get("selection_metric", "mrr")
        score = float(metrics.get(selection_metric, metrics["mrr"]))
        elapsed = time.time() - t_start

        return {
            "trial_id": trial_id,
            "config": arch_config,
            "score": score,
            "mrr": float(metrics["mrr"]),
            "recall_at_k": float(metrics["recall_at_k"]),
            "time_sec": round(elapsed, 4),
        }
