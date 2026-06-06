"""
Data-Parallel NAS Executor.

Fine-grained batch-level data parallelism: each batch is split among N workers,
gradients are averaged after each batch, and model is updated immediately.
This ensures the model sees all data with the same gradient sync frequency as Serial.

Architecture-level search is still sequential (same as serial), so the number of
architectures explored per unit time equals the serial baseline. This is the key
contrast with Pipeline, which achieves architecture-level concurrency.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import ray
import torch
import torch.optim as optim

from data.synthetic import Interaction, clone_graph_state_template, init_dynamic_graph_state
from data.temporal_partition import TemporalPartition, split_partition_interactions
from models.factory import build_model
from models.training import BPRLoss, _item_embeddings_for_loss, _num_items, evaluate_ranking_metrics, _model_device


# ---------------------------------------------------------------------------
# Ray worker actor
# ---------------------------------------------------------------------------

@ray.remote
class _DataParallelWorker:
    """Trains one chunk of a partition using BPR loss; returns updated weights and state."""

    def train_single_interaction(
        self,
        model_state_dict: Dict[str, Any],
        runtime_state: Optional[Dict[str, Any]],
        interaction: Interaction,
        arch_config: Dict[str, Any],
        base_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.train_chunk(model_state_dict, runtime_state, [interaction], arch_config, base_config)

    def train_chunk(
        self,
        model_state_dict: Dict[str, Any],
        runtime_state: Optional[Dict[str, Any]],
        interactions: List[Interaction],
        arch_config: Dict[str, Any],
        base_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        import numpy as np
        from models.training import train_partition_bpr_tgn, train_partition_bpr_batch

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

        batch_mode = config.get("batch_mode", "serial")

        if batch_mode == "tgn":
            # TGN batch processing
            partition = TemporalPartition(
                partition_id=0,
                split="train",
                start_ts=float(interactions[0].timestamp),
                end_ts=float(interactions[-1].timestamp),
                interactions=interactions,
            )
            tgn_loss_mode = config.get("tgn_loss_mode", "last")
            tgn_window_size = config.get("tgn_window_size", 10.0)
            total_loss = train_partition_bpr_tgn(
                model=model,
                partition=partition,
                optimizer=optimizer,
                criterion=criterion,
                time_window_size=tgn_window_size,
                aggregator="mean",
                loss_mode=tgn_loss_mode,
                neg_sample_size=neg_sample_size,
                seed=None,
                graph_ctx=graph_ctx,
            )
        elif batch_mode == "tbatch":
            # T-batch processing - compute gradients only, don't update parameters
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
        else:
            # Serial mode: event-by-event training
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

        # Collect accumulated gradients
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_runtime_states(states: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Merge worker runtime states: per user/item keep the most recent (max last_time)."""
    valid = [s for s in states if s is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    merged = {k: v.clone() for k, v in valid[0].items()}

    for state in valid[1:]:
        for prefix in ("user", "item"):
            lt_key = f"{prefix}_last_time"
            emb_key = f"{prefix}_embeddings"
            if lt_key not in state or emb_key not in state:
                continue
            mask = state[lt_key] > merged[lt_key]
            merged[lt_key][mask] = state[lt_key][mask]
            merged[emb_key][mask] = state[emb_key][mask]

        # LSTM cell state: same max-timestamp strategy
        for prefix in ("user", "item"):
            ht_key = f"{prefix}_h"
            ct_key = f"{prefix}_c"
            lt_key = f"{prefix}_last_time"
            if ht_key in state:
                mask = state[lt_key] > merged.get(lt_key, torch.zeros_like(state[lt_key]))
                if mask.any():
                    merged[ht_key] = merged.get(ht_key, state[ht_key]).clone()
                    merged[ht_key][mask] = state[ht_key][mask]
            if ct_key in state:
                mask = state[lt_key] > merged.get(lt_key, torch.zeros_like(state[lt_key]))
                if mask.any():
                    merged[ct_key] = merged.get(ct_key, state[ct_key]).clone()
                    merged[ct_key][mask] = state[ct_key][mask]

    return merged


def _apply_averaged_gradients(
    model_state_dict: Dict[str, Any],
    avg_grads: Dict[str, Any],
    arch_config: Dict[str, Any],
    base_config: Dict[str, Any],
    runtime_state: Optional[Dict[str, Any]],
    optimizer_state: Optional[Dict[str, Any]],
) -> tuple:
    """Apply averaged gradients to model and return (new_state_dict, new_optimizer_state)."""
    device = torch.device("cpu")
    config = dict(base_config)
    config.update(arch_config)

    model = build_model(config)
    model.to(device)
    model.load_state_dict({k: v.to(device) for k, v in model_state_dict.items()})
    if runtime_state is not None and hasattr(model, "import_runtime_state"):
        rs = {k: v.to(device) for k, v in runtime_state.items()}
        model.import_runtime_state(rs)

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
# DataParallelExecutor
# ---------------------------------------------------------------------------

class DataParallelExecutor:
    """
    Evaluates NAS architectures sequentially (same count as serial), but uses
    N Ray workers to parallelize training *within* each temporal partition.

    This gives a ~N× speedup per trial, so N architectures can be evaluated
    in roughly the same wall-time as serial — but the search *coverage* is
    identical (same N architectures).  Pipeline, by contrast, keeps N workers
    busy evaluating N *different* architectures concurrently, doubling coverage.
    """

    def __init__(self, base_config: Dict[str, Any], partition_plan, num_workers: int = 3):
        self.base_config = base_config
        self.partition_plan = partition_plan
        self.num_workers = num_workers

        if not ray.is_initialized():
            import os
            visible = str(base_config.get("data_parallel_visible_gpus", "0,1,2"))
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
            ray.init(ignore_reinit_error=True)

        gpu_frac = base_config.get("data_parallel_worker_gpus", 1.0)
        self._workers = [
            _DataParallelWorker.options(  # type: ignore[attr-defined]
                num_cpus=1,
                num_gpus=gpu_frac,
            ).remote()
            for _ in range(num_workers)
        ]

    def shutdown(self) -> None:
        """Kill all Ray worker actors so they release resources before the next executor."""
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
        results = []
        for trial_id, arch_config in enumerate(arch_configs):
            result = self._run_trial(arch_config, trial_id, num_train_epochs)
            results.append(result)
        return results

    def _run_trial(
        self,
        arch_config: Dict[str, Any],
        trial_id: int,
        num_train_epochs: int,
    ) -> Dict[str, Any]:
        config = dict(self.base_config)
        config.update(arch_config)

        model = build_model(config)
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer_state: Optional[Dict] = None

        train_partitions = self.partition_plan.get_split_partitions("train")
        val_partitions = self.partition_plan.get_split_partitions("val")

        t_start = time.time()
        batch_size = config.get("train_batch_size", 32)

        for epoch_idx in range(num_train_epochs):
            runtime_state: Optional[Dict] = None

            for partition in train_partitions:
                if not partition.interactions:
                    continue

                interactions = partition.interactions
                total = len(interactions)

                # Process partition in batches, splitting each batch among workers
                for start_idx in range(0, total, batch_size):
                    end_idx = min(start_idx + batch_size, total)
                    batch = interactions[start_idx:end_idx]

                    # Split batch among workers
                    chunks = split_partition_interactions(
                        TemporalPartition(
                            partition.partition_id, partition.split,
                            batch[0].timestamp if batch else partition.start_ts,
                            batch[-1].timestamp if batch else partition.end_ts,
                            batch
                        ),
                        self.num_workers
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

                    # Average gradients weighted by chunk size
                    avg_grads = {}
                    for r in worker_results:
                        w = r["num_interactions"] / total_interactions
                        for name, g in r["gradients"].items():
                            avg_grads[name] = avg_grads.get(name, 0) + g * w

                    # Update model parameters with averaged gradients
                    model_state_dict, optimizer_state = _apply_averaged_gradients(
                        model_state_dict, avg_grads, arch_config,
                        self.base_config, runtime_state, optimizer_state,
                    )

                    # Merge embedding states from workers
                    runtime_state = _merge_runtime_states([r["runtime_state"] for r in worker_results])

                    if (end_idx % 500 == 0) or (end_idx == total):
                        print(
                            f"[DataParallel batch-sync] trial={trial_id} epoch={epoch_idx+1}/{num_train_epochs} "
                            f"partition={partition.partition_id} interactions={end_idx}/{total}",
                            flush=True,
                        )

        # Evaluation (sequential to preserve temporal order)
        # Evaluation runs in the executor process (no GPU assigned by Ray here), use CPU.
        eval_device = torch.device("cpu")
        val_model = build_model(config)
        val_model.to(eval_device)
        val_model.load_state_dict({k: v.to(eval_device) for k, v in model_state_dict.items()})
        if runtime_state is not None and hasattr(val_model, "import_runtime_state"):
            val_model.import_runtime_state({k: v.to(eval_device) for k, v in runtime_state.items()})

        model_name = config.get("model", "jodie_rnn")
        if model_name == "jodie_rnn":
            eval_graph_ctx = None
        else:
            eval_graph_ctx = init_dynamic_graph_state(
                num_users=config.get("num_users", 1),
                num_items=config.get("num_items", 1),
                max_neighbors=config.get("max_neighbors", 20),
            )

        val_interactions: List[Interaction] = []
        for p in val_partitions:
            val_interactions.extend(p.interactions)

        metrics = evaluate_ranking_metrics(
            val_model,
            val_interactions,
            k=config.get("k", 10),
            graph_ctx=eval_graph_ctx,
            partitions=val_partitions if val_partitions else None,
            frozen=config.get("eval_frozen", False),
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
