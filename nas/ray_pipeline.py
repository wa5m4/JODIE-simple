from collections import deque
from dataclasses import dataclass
import os
from typing import Deque, Dict, List, Optional, Tuple
import time

import torch

from data.synthetic import (
    clone_graph_state_template,
    init_dynamic_graph_state,
    restore_graph_state,
    snapshot_graph_state,
)
from data.temporal_partition import TemporalPartition, TemporalPartitionPlan
from models.factory import build_model
from models.training import (
    BPRLoss,
    evaluate_partition_ranking,
    evaluate_partition_type_recall,
    train_partition_bpr,
    train_partition_ce,
)

try:
    import ray
except ImportError:  # pragma: no cover
    ray = None


@dataclass
class PipelineModelPayload:
    trial_id: int
    arch_config: Dict
    model_state_dict: Dict[str, torch.Tensor]
    runtime_state: Optional[Dict]
    graph_state: Optional[Dict]
    optimizer_state: Optional[Dict]
    seed: int


class PartitionShardWorker:
    def __init__(self, partitions: List[TemporalPartition], base_config: Dict):
        self.partitions = {partition.partition_id: partition for partition in partitions}
        self.base_config = dict(base_config)
        self.pipeline_trace = bool(self.base_config.get("pipeline_trace", False))
        self.pipeline_trace_log_path = str(self.base_config.get("pipeline_trace_log_path", "")).strip()
        self._trace_start = time.perf_counter()

    def _build_model(self, payload: PipelineModelPayload):
        config = dict(self.base_config)
        config.update(payload.arch_config)
        model = build_model(config)
        model.load_state_dict(payload.model_state_dict)
        if payload.runtime_state is not None and hasattr(model, "import_runtime_state"):
            model.import_runtime_state(payload.runtime_state)
        return model, config

    def _append_trace_line(self, line: str) -> None:
        if not self.pipeline_trace_log_path:
            return
        with open(self.pipeline_trace_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _trace_progress(self, message: str) -> None:
        elapsed = time.perf_counter() - self._trace_start
        wall_ts = time.strftime("%H:%M:%S", time.localtime())
        line = f"[pipeline-trace] wall={wall_ts} elapsed={elapsed:.3f}s {message}"
        self._append_trace_line(line)
        if self.pipeline_trace:
            print(line, flush=True)

    def run_train_stage_batch(
        self,
        payload: PipelineModelPayload,
        partition_ids: List[int],
        use_bpr: bool = True,
        num_epochs: int = 1,
    ) -> PipelineModelPayload:
        if not partition_ids:
            return payload

        model, config = self._build_model(payload)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
        if payload.optimizer_state is not None:
            optimizer.load_state_dict(payload.optimizer_state)

        epochs = max(int(num_epochs), 1)
        progress_every = int(self.base_config.get("pipeline_train_progress_every", 100))
        graph_state = restore_graph_state(payload.graph_state) if payload.graph_state is not None else None
        for epoch in range(epochs):
            # Do NOT call reset_state() here. State is cleared at epoch boundaries by the
            # outer loop (runtime_state=None in payloads), and should flow between stages
            # so that Stage 1 sees Stage 0's accumulated user_embeddings/user_last_time.
            epoch_graph_state = clone_graph_state_template(graph_state) if graph_state is not None else None

            for partition_id in partition_ids:
                partition = self.partitions[partition_id]
                if use_bpr:
                    criterion = BPRLoss()
                    train_partition_bpr(
                        model=model,
                        partition=partition,
                        optimizer=optimizer,
                        criterion=criterion,
                        neg_sample_size=config.get("neg_sample_size", 5),
                        graph_ctx=epoch_graph_state,
                        seed=payload.seed + epoch * 100000 + partition_id,
                        progress_every=progress_every,
                        progress_callback=lambda idx, total, current_partition_id=partition_id: self._trace_progress(
                            f"phase=train event=interaction_progress trial={payload.trial_id} partition={current_partition_id} processed={idx} total={total} metric=bpr epoch={epoch + 1}/{epochs}"
                        ),
                    )
                else:
                    train_partition_ce(
                        model=model,
                        partition=partition,
                        optimizer=optimizer,
                        graph_ctx=epoch_graph_state,
                        progress_every=progress_every,
                        progress_callback=lambda idx, total, current_partition_id=partition_id: self._trace_progress(
                            f"phase=train event=interaction_progress trial={payload.trial_id} partition={current_partition_id} processed={idx} total={total} metric=ce epoch={epoch + 1}/{epochs}"
                        ),
                    )

        runtime_state = model.export_runtime_state() if hasattr(model, "export_runtime_state") else None
        return PipelineModelPayload(
            trial_id=payload.trial_id,
            arch_config=payload.arch_config,
            model_state_dict=model.state_dict(),
            runtime_state=runtime_state,
            graph_state=snapshot_graph_state(graph_state) if graph_state is not None else None,
            optimizer_state=optimizer.state_dict(),
            seed=payload.seed,
        )

    def run_eval_stage_batch(
        self,
        payload: PipelineModelPayload,
        partition_ids: List[int],
        item_type,
        user_type_prefs,
        k: int,
        synthetic_mode: bool,
    ) -> Dict:
        if not partition_ids:
            return {
                "payload": payload,
                "hits": 0,
                "total": 0,
                "mrr_sum": 0.0,
            }

        model, _ = self._build_model(payload)
        graph_state = restore_graph_state(payload.graph_state) if payload.graph_state is not None else None
        progress_every = int(self.base_config.get("pipeline_eval_progress_every", 100))
        stage_label = f"trial={payload.trial_id} partitions={partition_ids}"
        self._trace_progress(f"phase=eval event=batch_start {stage_label} synthetic={synthetic_mode}")

        total_hits = 0
        total_count = 0
        total_mrr = 0.0
        for partition_id in partition_ids:
            partition = self.partitions[partition_id]
            partition_start = time.perf_counter()
            self._trace_progress(
                f"phase=eval event=partition_start trial={payload.trial_id} partition={partition_id} interactions={len(partition.interactions)}"
            )
            if synthetic_mode:
                metrics = evaluate_partition_type_recall(
                    model,
                    partition,
                    item_type=item_type,
                    user_type_prefs=user_type_prefs,
                    k=k,
                    graph_ctx=graph_state,
                    progress_label=stage_label,
                    progress_every=progress_every,
                    progress_callback=lambda idx, total, current_partition_id=partition_id: self._trace_progress(
                        f"phase=eval event=interaction_progress trial={payload.trial_id} partition={current_partition_id} processed={idx} total={total} metric=type"
                    ),
                )
                total_hits += int(metrics["hits"])
                total_count += int(metrics["total"])
                total_mrr += float(metrics["hits"])
            else:
                metrics = evaluate_partition_ranking(
                    model,
                    partition,
                    k=k,
                    graph_ctx=graph_state,
                    progress_label=stage_label,
                    progress_every=progress_every,
                    progress_callback=lambda idx, total, current_partition_id=partition_id: self._trace_progress(
                        f"phase=eval event=interaction_progress trial={payload.trial_id} partition={current_partition_id} processed={idx} total={total} metric=ranking"
                    ),
                )
                total_hits += int(metrics["hits"])
                total_count += int(metrics["total"])
                total_mrr += float(metrics["mrr_sum"])
            self._trace_progress(
                f"phase=eval event=partition_complete trial={payload.trial_id} partition={partition_id} elapsed_sec={time.perf_counter() - partition_start:.3f} hits={int(metrics['hits'])} total={int(metrics['total'])}"
            )

        runtime_state = model.export_runtime_state() if hasattr(model, "export_runtime_state") else None
        updated_payload = PipelineModelPayload(
            trial_id=payload.trial_id,
            arch_config=payload.arch_config,
            model_state_dict=model.state_dict(),
            runtime_state=runtime_state,
            graph_state=snapshot_graph_state(graph_state) if graph_state is not None else None,
            optimizer_state=payload.optimizer_state,
            seed=payload.seed,
        )
        return {
            "payload": updated_payload,
            "hits": total_hits,
            "total": total_count,
            "mrr_sum": total_mrr,
        }

    def run_train_stage(self, payload: PipelineModelPayload, partition_id: int, use_bpr: bool = True, num_epochs: int = 1) -> PipelineModelPayload:
        return self.run_train_stage_batch(payload, [partition_id], use_bpr=use_bpr, num_epochs=num_epochs)

    def run_eval_stage(
        self,
        payload: PipelineModelPayload,
        partition_id: int,
        item_type,
        user_type_prefs,
        k: int,
        synthetic_mode: bool,
    ) -> Dict:
        batch_result = self.run_eval_stage_batch(
            payload=payload,
            partition_ids=[partition_id],
            item_type=item_type,
            user_type_prefs=user_type_prefs,
            k=k,
            synthetic_mode=synthetic_mode,
        )
        return {
            "hits": int(batch_result["hits"]),
            "total": int(batch_result["total"]),
            "mrr_sum": float(batch_result["mrr_sum"]),
        }


def create_ray_worker(partitions: List[TemporalPartition], base_config: Dict):
    if ray is None:
        raise ImportError("ray is required for execution_mode=ray_pipeline")
    num_gpus = float(base_config.get("pipeline_worker_gpus", 0.0))
    num_cpus = float(base_config.get("pipeline_worker_cpus", 1.0))
    if num_cpus <= 0:
        num_cpus = 1.0
    worker_cls = ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)(PartitionShardWorker)
    return worker_cls.remote(partitions, base_config)


class RayPipelineExecutor:
    def __init__(self, base_config: Dict, partition_plan: TemporalPartitionPlan):
        self.base_config = dict(base_config)    
        
        self.partition_plan = partition_plan
        self.pipeline_trace = bool(self.base_config.get("pipeline_trace", False))
        self.pipeline_trace_log_path = str(self.base_config.get("pipeline_trace_log_path", "")).strip()
        self.stage_balance_strategy = str(self.base_config.get("stage_balance_strategy", "cost")).strip().lower()
        self.stage_balance_user_weight = float(self.base_config.get("stage_balance_user_weight", 0.25))
        self.stage_balance_item_weight = float(self.base_config.get("stage_balance_item_weight", 0.25))
        self.stage_balance_span_weight = float(self.base_config.get("stage_balance_span_weight", 0.0))
        self._trace_start = time.perf_counter()
        self._stage_dispatch_times: Dict[str, float] = {}
        self._trace_scan_pos = 0

    def _scan_worker_progress_events(self) -> Tuple[int, Optional[str]]:
        if not self.pipeline_trace_log_path:
            return 0, None
        if not os.path.exists(self.pipeline_trace_log_path):
            return 0, None

        progress_count = 0
        last_progress_line = None
        try:
            with open(self.pipeline_trace_log_path, "r", encoding="utf-8") as f:
                f.seek(self._trace_scan_pos)
                for line in f:
                    if (
                        "event=interaction_progress" in line
                        or "event=partition_start" in line
                        or "event=partition_complete" in line
                    ):
                        progress_count += 1
                        last_progress_line = line.strip()
                self._trace_scan_pos = f.tell()
        except OSError:
            return 0, None

        return progress_count, last_progress_line

    def _resolve_stage_worker_counts(self, key: str, num_stages: int, fallback: Optional[List[int]] = None) -> List[int]:
        raw = str(self.base_config.get(key, "")).strip()
        if not raw:
            if fallback is not None and len(fallback) == num_stages:
                return list(fallback)
            return [1 for _ in range(num_stages)]

        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if not parts:
            if fallback is not None and len(fallback) == num_stages:
                return list(fallback)
            return [1 for _ in range(num_stages)]

        values = [max(1, int(part)) for part in parts]
        if len(values) == 1:
            return [values[0] for _ in range(num_stages)]
        if len(values) != num_stages:
            raise ValueError(f"{key} expects one value or {num_stages} comma-separated values")
        return values

    def _trace_key(self, phase: str, trial_id: int, stage_idx: int) -> str:
        return f"{phase}:{trial_id}:{stage_idx}"

    def _append_trace_line(self, line: str) -> None:
        if not self.pipeline_trace_log_path:
            return
        with open(self.pipeline_trace_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _trace_event(
        self,
        phase: str,
        event: str,
        trial_id: int,
        stage_idx: int,
        stage_total: int,
    ) -> None:
        elapsed = time.perf_counter() - self._trace_start
        wall_ts = time.strftime("%H:%M:%S", time.localtime())
        key = self._trace_key(phase, trial_id, stage_idx)
        stage_duration = None
        if event == "dispatch":
            self._stage_dispatch_times[key] = elapsed
        elif event == "complete":
            start_elapsed = self._stage_dispatch_times.pop(key, None)
            if start_elapsed is not None:
                stage_duration = max(0.0, elapsed - start_elapsed)

        duration_part = ""
        if stage_duration is not None:
            duration_part = f" stage_duration_sec={stage_duration:.3f}"
        line = (
            f"[pipeline-trace] wall={wall_ts} elapsed={elapsed:.3f}s "
            f"phase={phase} event={event} trial={trial_id} stage={stage_idx + 1}/{stage_total}{duration_part}"
        )
        self._append_trace_line(line)
        if self.pipeline_trace:
            print(line)

    def _estimate_partition_costs(self, partitions: List[TemporalPartition]) -> List[float]:
        costs: List[float] = []
        seen_users = set()
        seen_items = set()

        for partition in partitions:
            interactions = partition.interactions
            if not interactions:
                costs.append(1.0)
                continue

            unique_users = {interaction.user_id for interaction in interactions}
            unique_items = {interaction.item_id for interaction in interactions}
            new_users = unique_users.difference(seen_users)
            new_items = unique_items.difference(seen_items)
            span = max(float(partition.end_ts) - float(partition.start_ts), 0.0)

            cost = float(len(interactions))
            # Novel users/items often imply heavier updates in early temporal shards.
            cost += self.stage_balance_user_weight * float(len(unique_users) + len(new_users))
            cost += self.stage_balance_item_weight * float(len(unique_items) + len(new_items))
            cost += self.stage_balance_span_weight * span
            costs.append(max(cost, 1.0))

            seen_users.update(unique_users)
            seen_items.update(unique_items)

        return costs

    def _group_partitions_by_count(self, partitions: List[TemporalPartition], num_stages: int) -> List[List[TemporalPartition]]:
        num_stages = max(1, min(num_stages, len(partitions)))
        grouped = []
        base = len(partitions) // num_stages
        remainder = len(partitions) % num_stages
        start = 0
        for idx in range(num_stages):
            chunk_size = base + (1 if idx < remainder else 0)
            end = start + chunk_size
            grouped.append(partitions[start:end])
            start = end
        return grouped

    def _group_partitions_by_cost(self, partitions: List[TemporalPartition], num_stages: int) -> List[List[TemporalPartition]]:
        num_stages = max(1, min(num_stages, len(partitions)))
        if num_stages == 1:
            return [partitions]

        costs = self._estimate_partition_costs(partitions)
        total_cost = sum(costs)
        target_cost = total_cost / num_stages
        prefix_costs = [0.0]
        for cost in costs:
            prefix_costs.append(prefix_costs[-1] + cost)

        def segment_cost(start_idx: int, end_idx: int) -> float:
            return prefix_costs[end_idx] - prefix_costs[start_idx]

        n = len(partitions)
        inf = float("inf")
        dp = [[inf] * (num_stages + 1) for _ in range(n + 1)]
        backtrack = [[-1] * (num_stages + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0

        for end_idx in range(1, n + 1):
            max_stages = min(num_stages, end_idx)
            for stage_count in range(1, max_stages + 1):
                best_score = inf
                best_start = -1
                min_start = stage_count - 1
                for start_idx in range(min_start, end_idx):
                    prev_score = dp[start_idx][stage_count - 1]
                    if prev_score == inf:
                        continue
                    current_cost = segment_cost(start_idx, end_idx)
                    score = prev_score + (current_cost - target_cost) ** 2
                    if score < best_score:
                        best_score = score
                        best_start = start_idx
                dp[end_idx][stage_count] = best_score
                backtrack[end_idx][stage_count] = best_start

        if dp[n][num_stages] == inf:
            return self._group_partitions_by_count(partitions, num_stages)

        grouped: List[List[TemporalPartition]] = []
        end_idx = n
        stage_count = num_stages
        while stage_count > 0:
            start_idx = backtrack[end_idx][stage_count]
            if start_idx < 0:
                return self._group_partitions_by_count(partitions, num_stages)
            grouped.append(partitions[start_idx:end_idx])
            end_idx = start_idx
            stage_count -= 1

        grouped.reverse()
        return grouped

    def _group_partitions(self, split: str, num_stages: int) -> List[List[TemporalPartition]]:
        partitions = self.partition_plan.get_split_partitions(split)
        if not partitions:
            return []
        ordered_partitions = sorted(partitions, key=lambda partition: (float(partition.start_ts), partition.partition_id))
        if self.stage_balance_strategy == "count":
            return self._group_partitions_by_count(ordered_partitions, num_stages)
        return self._group_partitions_by_cost(ordered_partitions, num_stages)

    def _run_train_pipeline(
        self,
        payloads: List[PipelineModelPayload],
        train_groups: List[List[TemporalPartition]],
        stage_workers,
        use_bpr: bool,
        num_train_epochs: int,
    ) -> List[PipelineModelPayload]:
        if not payloads or not train_groups:
            return payloads
        # 每个 epoch 独立走一遍完整的 stage 流水线（每 stage 只训 1 epoch），
        # 等价于 serial 训练的多 epoch：epoch 循环在外，partition 顺序在内。
        # 若只有 1 epoch 则与之前行为完全相同。
        if num_train_epochs > 1:
            current = payloads
            for _ in range(num_train_epochs):
                # Reset runtime state at each epoch boundary so Stage 0 starts with zero
                # user_embeddings/user_last_time (equivalent to serial's reset_model_state).
                # Stages within an epoch do NOT reset — state flows stage0→stage1→stage2
                # so that later stages see temporal context from earlier partitions.
                epoch_start = [
                    PipelineModelPayload(
                        trial_id=p.trial_id,
                        arch_config=p.arch_config,
                        model_state_dict=p.model_state_dict,
                        runtime_state=None,
                        graph_state=p.graph_state,
                        optimizer_state=p.optimizer_state,
                        seed=p.seed,
                    )
                    for p in current
                ]
                current = self._run_train_pipeline(
                    epoch_start, train_groups, stage_workers, use_bpr, num_train_epochs=1
                )
            return current

        stage_partition_ids = [[p.partition_id for p in partitions] for partitions in train_groups]
        in_flight: Dict[object, Tuple[int, int]] = {}
        completed: List[PipelineModelPayload] = []
        last_stage_idx = len(train_groups) - 1
        stage_total = len(train_groups)
        pending_by_stage: List[Deque[PipelineModelPayload]] = [deque() for _ in range(stage_total)]
        pending_by_stage[0] = deque(payloads)
        idle_workers: List[Deque[int]] = [deque(range(len(pool))) for pool in stage_workers]
        last_heartbeat = time.perf_counter()
        heartbeat_interval = float(self.base_config.get("pipeline_heartbeat_interval_sec", 5.0))
        last_progress_events_seen = 0

        while True:
            progress = False
            for stage_idx in range(stage_total):
                stage_pending = pending_by_stage[stage_idx]
                stage_idle = idle_workers[stage_idx]
                while stage_pending and stage_idle:
                    payload = stage_pending.popleft()
                    worker_idx = stage_idle.popleft()
                    self._trace_event("train", "dispatch", payload.trial_id, stage_idx, stage_total)
                    ref = stage_workers[stage_idx][worker_idx].run_train_stage_batch.remote(
                        payload,
                        stage_partition_ids[stage_idx],
                        use_bpr=use_bpr,
                        num_epochs=num_train_epochs,
                    )
                    in_flight[ref] = (stage_idx, worker_idx)
                    progress = True

            if not in_flight:
                if progress:
                    continue
                break

            done_refs, _ = ray.wait(list(in_flight.keys()), num_returns=1, timeout=5.0)
            if not done_refs:
                now = time.perf_counter()
                progress_delta, last_progress_line = self._scan_worker_progress_events()
                if progress_delta > 0:
                    last_progress_events_seen += progress_delta
                # 固定间隔打印有意义的心跳：只依据真实进度事件变化判断状态
                if now - last_heartbeat >= heartbeat_interval:
                    active_by_stage: Dict[int, int] = {}
                    for stage_idx, _ in in_flight.values():
                        active_by_stage[stage_idx] = active_by_stage.get(stage_idx, 0) + 1
                    active_desc = ", ".join(
                        f"stage {stage_idx + 1}={count}" for stage_idx, count in sorted(active_by_stage.items())
                    ) or "none"
                    pending_desc = ", ".join(
                        f"stage {stage_idx + 1}={len(queue)}" for stage_idx, queue in enumerate(pending_by_stage)
                    )
                    if progress_delta > 0:
                        status = "advancing"
                    else:
                        status = "no_new_progress"
                    print(
                        f"[pipeline-heartbeat] wall={time.strftime('%H:%M:%S', time.localtime())} "
                        f"phase=train status={status} progress_events_total={last_progress_events_seen} progress_events_delta={progress_delta} "
                        f"in_flight={len(in_flight)} active={active_desc} pending={pending_desc}"
                        + (f" last_progress={last_progress_line}" if last_progress_line else ""),
                        flush=True,
                    )
                    last_heartbeat = now
                continue
            done_ref = done_refs[0]
            stage_idx, worker_idx = in_flight.pop(done_ref)
            idle_workers[stage_idx].append(worker_idx)
            try:
                updated_payload = ray.get(done_ref)
            except Exception as e:
                print(f"[ERROR] Ray task failed for train pipeline: {str(e)}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            self._trace_event("train", "complete", updated_payload.trial_id, stage_idx, stage_total)

            if stage_idx >= last_stage_idx:
                completed.append(updated_payload)
                continue

            next_stage = stage_idx + 1
            pending_by_stage[next_stage].append(updated_payload)

        return sorted(completed, key=lambda x: x.trial_id)

    def _run_eval_pipeline(
        self,
        payloads: List[PipelineModelPayload],
        eval_groups: List[List[TemporalPartition]],
        eval_stage_workers,
        item_type,
        user_type_prefs,
        k: int,
        synthetic_mode: bool,
    ) -> Dict[int, Dict[str, float]]:
        scores = {
            payload.trial_id: {"hits": 0, "total": 0, "mrr_sum": 0.0}
            for payload in payloads
        }
        if not payloads or not eval_groups:
            return scores

        stage_partition_ids = [[p.partition_id for p in partitions] for partitions in eval_groups]
        in_flight: Dict[object, Tuple[int, int]] = {}
        last_stage_idx = len(eval_groups) - 1
        stage_total = len(eval_groups)
        pending_by_stage: List[Deque[PipelineModelPayload]] = [deque() for _ in range(stage_total)]
        pending_by_stage[0] = deque(payloads)
        idle_workers: List[Deque[int]] = [deque(range(len(pool))) for pool in eval_stage_workers]
        last_progress_events_seen = 0
        last_heartbeat = time.perf_counter()
        heartbeat_interval = float(self.base_config.get("pipeline_heartbeat_interval_sec", 5.0))

        while True:
            progress = False
            for stage_idx in range(stage_total):
                stage_pending = pending_by_stage[stage_idx]
                stage_idle = idle_workers[stage_idx]
                while stage_pending and stage_idle:
                    payload = stage_pending.popleft()
                    worker_idx = stage_idle.popleft()
                    self._trace_event("eval", "dispatch", payload.trial_id, stage_idx, stage_total)
                    ref = eval_stage_workers[stage_idx][worker_idx].run_eval_stage_batch.remote(
                        payload,
                        stage_partition_ids[stage_idx],
                        item_type=item_type,
                        user_type_prefs=user_type_prefs,
                        k=k,
                        synthetic_mode=synthetic_mode,
                    )
                    in_flight[ref] = (stage_idx, worker_idx)
                    progress = True

            if not in_flight:
                if progress:
                    continue
                break

            done_refs, _ = ray.wait(list(in_flight.keys()), num_returns=1, timeout=5.0)
            if not done_refs:
                now = time.perf_counter()
                progress_delta, last_progress_line = self._scan_worker_progress_events()
                if progress_delta > 0:
                    last_progress_events_seen += progress_delta
                # 固定间隔打印有意义的心跳：只依据真实进度事件变化判断状态
                if now - last_heartbeat >= heartbeat_interval:
                    active_by_stage: Dict[int, int] = {}
                    for stage_idx, _ in in_flight.values():
                        active_by_stage[stage_idx] = active_by_stage.get(stage_idx, 0) + 1
                    active_desc = ", ".join(
                        f"stage {stage_idx + 1}={count}" for stage_idx, count in sorted(active_by_stage.items())
                    ) or "none"
                    pending_desc = ", ".join(
                        f"stage {stage_idx + 1}={len(queue)}" for stage_idx, queue in enumerate(pending_by_stage)
                    )
                    if progress_delta > 0:
                        status = "advancing"
                    else:
                        status = "no_new_progress"
                    print(
                        f"[pipeline-heartbeat] wall={time.strftime('%H:%M:%S', time.localtime())} "
                        f"phase=eval status={status} progress_events_total={last_progress_events_seen} progress_events_delta={progress_delta} "
                        f"in_flight={len(in_flight)} active={active_desc} pending={pending_desc}"
                        + (f" last_progress={last_progress_line}" if last_progress_line else ""),
                        flush=True,
                    )
                    last_heartbeat = now
                continue
            done_ref = done_refs[0]
            stage_idx, worker_idx = in_flight.pop(done_ref)
            idle_workers[stage_idx].append(worker_idx)
            try:
                stage_result = ray.get(done_ref)
            except Exception as e:
                print(f"[ERROR] Ray task failed for eval pipeline: {str(e)}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            payload = stage_result["payload"]
            self._trace_event("eval", "complete", payload.trial_id, stage_idx, stage_total)
            trial_score = scores[payload.trial_id]
            trial_score["hits"] += int(stage_result["hits"])
            trial_score["total"] += int(stage_result["total"])
            trial_score["mrr_sum"] += float(stage_result["mrr_sum"])

            if stage_idx >= last_stage_idx:
                continue

            next_stage = stage_idx + 1
            pending_by_stage[next_stage].append(payload)

        return scores

    def _shutdown_worker_pool(self, stage_workers) -> None:
        if ray is None:
            return
        for pool in stage_workers:
            for worker in pool:
                try:
                    ray.kill(worker, no_restart=True)
                except Exception:
                    pass

    def _make_payload(self, arch_config: Dict, trial_id: int, seed: int) -> PipelineModelPayload:
        config = dict(self.base_config)
        config.update(arch_config)
        model = build_model(config)
        runtime_state = model.export_runtime_state() if hasattr(model, "export_runtime_state") else None
        graph_state = init_dynamic_graph_state(
            num_users=int(self.base_config.get("num_users", 0)),
            num_items=int(self.base_config.get("num_items", 0)),
            max_neighbors=int(self.base_config.get("max_neighbors", 20)),
        )
        return PipelineModelPayload(
            trial_id=trial_id,
            arch_config=arch_config,
            model_state_dict=model.state_dict(),
            runtime_state=runtime_state,
            graph_state=graph_state,
            optimizer_state=None,
            seed=seed,
        )

    def _print_pipeline_summary(self, train_groups, eval_groups, train_worker_counts, eval_worker_counts, payloads):
        """打印 pipeline 配置摘要，帮助用户诊断并行效率瓶颈"""
        num_trials = len(payloads)
        total_train_workers = sum(train_worker_counts)
        total_eval_workers = sum(eval_worker_counts)
        gpu_count = int(torch.cuda.device_count() if torch.cuda.is_available() else 0)

        print(f"\n{'='*60}", flush=True)
        print(f"[Pipeline Summary]", flush=True)
        print(f"  Trials: {num_trials}", flush=True)
        print(f"  Train stages: {len(train_groups)}, workers: {train_worker_counts} (total={total_train_workers})", flush=True)
        print(f"  Eval stages:  {len(eval_groups)}, workers: {eval_worker_counts} (total={total_eval_workers})", flush=True)
        if gpu_count > 0:
            print(f"  GPUs: {gpu_count} (train workers {'fit' if total_train_workers <= gpu_count else 'oversubscribed'})", flush=True)

        # Per-stage cost summary
        for si, group in enumerate(train_groups):
            n_int = sum(len(p.interactions) for p in group)
            n_users = len({i.user_id for p in group for i in p.interactions})
            n_items = len({i.item_id for p in group for i in p.interactions})
            workers = train_worker_counts[si] if si < len(train_worker_counts) else 1
            print(f"  Train S{si+1}: {len(group)} partitions, {n_int} interactions, "
                  f"users={n_users}, items={n_items}, workers={workers}", flush=True)

        for si, group in enumerate(eval_groups):
            n_int = sum(len(p.interactions) for p in group)
            workers = eval_worker_counts[si] if si < len(eval_worker_counts) else 1
            print(f"  Eval S{si+1}:  {len(group)} partitions, {n_int} interactions, workers={workers}", flush=True)

        # Efficiency hints
        if max(train_worker_counts) <= 1 and len(train_groups) > 1:
            print(f"  hint: only 1 worker per train stage, pipeline has no intra-stage parallelism", flush=True)
        if gpu_count > 0 and total_train_workers < gpu_count:
            print(f"  hint: {gpu_count - total_train_workers} GPUs unused during training; "
                  f"add --pipeline-stage-train-workers to use them", flush=True)
        if gpu_count > 0 and total_eval_workers < gpu_count:
            print(f"  hint: {gpu_count - total_eval_workers} GPUs unused during evaluation", flush=True)
        print(f"{'='*60}\n", flush=True)

    def run(self, arch_configs: List[Dict], user_type_prefs, item_type, num_train_epochs: int = 1) -> List[Dict]:
        if ray is None:
            raise ImportError("ray is required for execution_mode=ray_pipeline")

        if not ray.is_initialized():
            ray_address = str(self.base_config.get("ray_address", "")).strip()
            if ray_address:
                ray.init(address=ray_address, ignore_reinit_error=True)
            else:
                ray.init(ignore_reinit_error=True)

        num_stages = int(self.base_config.get("num_pipeline_stages", 1))
        train_groups = self._group_partitions("train", num_stages)
        # prefer "test" split if available (final evaluation), fall back to "val"
        eval_split = "test" if self.partition_plan.get_split_partitions("test") else "val"
        eval_groups = self._group_partitions(eval_split, num_stages)
        stage_total = len(train_groups)

        # 自动检测 GPU
        try:
            gpu_count = int(torch.cuda.device_count() if torch.cuda.is_available() else 0)
        except Exception:
            gpu_count = 0

        specified_worker_gpus = float(self.base_config.get("pipeline_worker_gpus", 0.0))
        if gpu_count > 0 and specified_worker_gpus <= 0.0:
            self.base_config["pipeline_worker_gpus"] = 1.0

        # Smart worker count defaults: scale with GPU count when user hasn't specified
        user_specified_train_workers = bool(str(self.base_config.get("pipeline_stage_train_workers", "")).strip())
        if gpu_count > 0 and not user_specified_train_workers:
            # Distribute workers across stages, more to earlier stages (bottleneck-heavy)
            train_worker_counts = []
            remaining_gpus = gpu_count
            for si in range(stage_total):
                stage_share = max(1, remaining_gpus // (stage_total - si))
                train_worker_counts.append(stage_share)
                remaining_gpus -= stage_share
            print(
                f"[RayPipeline] auto worker allocation: {train_worker_counts} workers across {stage_total} stages (GPUs={gpu_count})",
                flush=True,
            )
        else:
            train_worker_counts = self._resolve_stage_worker_counts("pipeline_stage_train_workers", stage_total)

        eval_worker_counts = self._resolve_stage_worker_counts("pipeline_stage_eval_workers", len(eval_groups), fallback=[1 for _ in range(len(eval_groups))])

        # Warn if GPU count is underutilized
        total_train_workers = sum(train_worker_counts)
        if gpu_count > 0:
            print(
                f"[RayPipeline] detected {gpu_count} GPUs; pipeline_worker_gpus={self.base_config.get('pipeline_worker_gpus', 0.0)}",
                flush=True,
            )
            if total_train_workers > gpu_count:
                print(
                    f"[RayPipeline] warning: total train workers={total_train_workers} > GPUs={gpu_count}; "
                    f"Ray will schedule up to {gpu_count} concurrent GPU-actors.",
                    flush=True,
                )
            elif total_train_workers < gpu_count:
                print(
                    f"[RayPipeline] note: total train workers={total_train_workers} < GPUs={gpu_count}; "
                    f"consider increasing --pipeline-stage-train-workers to use all GPUs.",
                    flush=True,
                )

        # Stage cost diagnostics
        for si, group in enumerate(train_groups):
            costs = self._estimate_partition_costs(group)
            n_interactions = sum(len(p.interactions) for p in group)
            print(
                f"[RayPipeline] train stage {si+1}: {len(group)} partitions, "
                f"{n_interactions} interactions, "
                f"cost range=[{min(costs):.0f}-{max(costs):.0f}] sum={sum(costs):.0f}, "
                f"workers={train_worker_counts[si] if si < len(train_worker_counts) else 1}",
                flush=True,
            )

        print(
            f"[RayPipeline] train worker counts={train_worker_counts}, train groups={[ [p.partition_id for p in group] for group in train_groups ]}",
            flush=True,
        )
        print(
            f"[RayPipeline] eval worker counts={eval_worker_counts}, eval groups={[ [p.partition_id for p in group] for group in eval_groups ]}",
            flush=True,
        )

        payloads = [self._make_payload(arch, trial_id=idx, seed=int(self.base_config.get("seed", 42)) + idx) for idx, arch in enumerate(arch_configs)]
        synthetic_mode = self.base_config.get("dataset", "synthetic") == "synthetic"
        k = int(self.base_config.get("k", 10))
        use_bpr = synthetic_mode
        train_epochs = max(int(num_train_epochs), 1)

        train_stage_workers = []
        eval_stage_workers = []
        try:
            train_stage_workers = [
                [create_ray_worker(partitions, self.base_config) for _ in range(train_worker_counts[idx])]
                for idx, partitions in enumerate(train_groups)
            ]

            print(
                f"[RayPipeline] created train worker pools={[len(pool) for pool in train_stage_workers]}",
                flush=True,
            )

            payloads = self._run_train_pipeline(
                payloads,
                train_groups,
                train_stage_workers,
                use_bpr=use_bpr,
                num_train_epochs=train_epochs,
            )

            # Free train actors first to avoid GPU resource contention when scheduling eval actors.
            self._shutdown_worker_pool(train_stage_workers)
            train_stage_workers = []

            eval_stage_workers = [
                [create_ray_worker(partitions, self.base_config) for _ in range(eval_worker_counts[idx])]
                for idx, partitions in enumerate(eval_groups)
            ] if eval_groups else []
            print(
                f"[RayPipeline] created eval worker pools={[len(pool) for pool in eval_stage_workers]}",
                flush=True,
            )
            eval_scores = self._run_eval_pipeline(
                payloads,
                eval_groups,
                eval_stage_workers,
                item_type=item_type,
                user_type_prefs=user_type_prefs,
                k=k,
                synthetic_mode=synthetic_mode,
            )
        finally:
            self._shutdown_worker_pool(train_stage_workers)
            self._shutdown_worker_pool(eval_stage_workers)

        # Print pipeline utilization summary
        self._print_pipeline_summary(train_groups, eval_groups, train_worker_counts, eval_worker_counts, payloads)

        results = []
        for payload in payloads:
            trial_metrics = eval_scores.get(payload.trial_id, {"hits": 0, "total": 0, "mrr_sum": 0.0})
            total_hits = int(trial_metrics["hits"])
            total_count = int(trial_metrics["total"])
            total_mrr = float(trial_metrics["mrr_sum"])
            denom = max(total_count, 1)
            score = total_hits / denom
            results.append(
                {
                    "trial_id": payload.trial_id,
                    "config": payload.arch_config,
                    "recall_at_k": score,
                    "mrr": total_mrr / denom,
                    "score": score if synthetic_mode else total_mrr / denom,
                }
            )
        return results
