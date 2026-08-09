"""
最小回归脚本：比较 serial 与 Ray worker 在第一个 partition 后的 optimizer state。

目的：确保 optimizer state 的跨进程传输确实恢复到了同一语义，
而不是只恢复了模型参数。
"""

import copy

import torch

from jodie.data.synthetic import generate_synthetic_data
from jodie.data.temporal_partition import build_partition_plan
from jodie.models.factory import build_model
from jodie.nas.ray_pipeline import PartitionShardWorker, PipelineModelPayload
from jodie.nas.search_space import sanitize_config
from jodie.training.loops import BPRLoss, train_partition_bpr


def _make_config(seed: int = 42):
    config = {
        "dataset": "synthetic",
        "num_users": 50,
        "num_items": 30,
        "max_events": 300,
        "feature_dim": 4,
        "lr": 1e-3,
        "neg_sample_size": 5,
        "k": 10,
        "selection_metric": "mrr",
        "device": "cpu",
        "seed": seed,
        "partition_size": 50,
        "partition_strategy": "count",
        "batch_mode": "serial",
        "train_batch_size": 32,
        "max_neighbors": 0,
        "model": "jodie_rnn",
        "embedding_dim": 16,
        "memory_cell": "rnn",
        "time_proj": "off",
        "use_static_embeddings": "off",
        "normalize_state": "off",
        "event_agg": "none",
        "agg_activation": "none",
        "attn_type": "dot",
        "time_decay": "none",
        "hidden_dim": 0,
        "memory_gate": "off",
        "enable_event_agg": "off",
        "enable_graph_update": "off",
        "message_mode": "peer",
        "msg_linear": "off",
    }
    return sanitize_config(config)


def _load_optimizer_state(state_dict, optimizer, model):
    if state_dict.get("format") == "fqn":
        from jodie.nas.ray_pipeline import _optimizer_state_from_fqn

        _optimizer_state_from_fqn(state_dict, optimizer, model)
    else:
        optimizer.load_state_dict(state_dict)


def main() -> int:
    seed = 42
    config = _make_config(seed)

    interactions, _, _ = generate_synthetic_data(50, 30, 300, 4, seed=seed)
    interactions = sorted(interactions, key=lambda item: item.timestamp)
    train_ints = interactions[:240]
    plan = build_partition_plan(train_ints, [], [], partition_size=50, strategy="count")
    first_partition = sorted(plan.get_split_partitions("train"), key=lambda p: (float(p.start_ts), p.partition_id))[0]

    base_model = build_model(config)
    base_state = {name: tensor.detach().clone() for name, tensor in base_model.state_dict().items()}
    base_runtime = base_model.export_runtime_state() if hasattr(base_model, "export_runtime_state") else None

    serial_model = build_model(config)
    serial_model.load_state_dict(base_state)
    if base_runtime is not None and hasattr(serial_model, "import_runtime_state"):
        serial_model.import_runtime_state(copy.deepcopy(base_runtime))
    elif hasattr(serial_model, "reset_state"):
        serial_model.reset_state()
    serial_optimizer = torch.optim.Adam(serial_model.parameters(), lr=config["lr"])
    criterion = BPRLoss()
    train_partition_bpr(
        model=serial_model,
        partition=first_partition,
        optimizer=serial_optimizer,
        criterion=criterion,
        neg_sample_size=config["neg_sample_size"],
        graph_ctx=None,
        seed=seed + first_partition.partition_id,
    )

    worker = PartitionShardWorker([first_partition], config)
    payload = PipelineModelPayload(
        trial_id=0,
        arch_config=config,
        model_state_dict=base_state,
        runtime_state=copy.deepcopy(base_runtime),
        graph_state=None,
        optimizer_state=None,
        seed=seed,
    )
    worker_payload = worker.run_train_stage_batch(payload, [first_partition.partition_id], use_bpr=True, num_epochs=1)

    worker_model, _ = worker._build_model(payload)
    worker_optimizer = torch.optim.Adam(worker_model.parameters(), lr=config["lr"])
    _load_optimizer_state(worker_payload.optimizer_state, worker_optimizer, worker_model)

    serial_state = serial_optimizer.state_dict()
    worker_state = worker_optimizer.state_dict()

    max_state_diff = 0.0
    for serial_param_id, serial_slot in serial_state["state"].items():
        worker_slot = worker_state["state"].get(serial_param_id)
        if worker_slot is None:
            print(f"missing worker optimizer state for param_index={serial_param_id}")
            return 2
        for key, serial_value in serial_slot.items():
            worker_value = worker_slot.get(key)
            if isinstance(serial_value, torch.Tensor) and isinstance(worker_value, torch.Tensor):
                max_state_diff = max(max_state_diff, float((serial_value - worker_value).abs().max().item()))
            elif serial_value != worker_value:
                print(f"mismatch param_index={serial_param_id} field={key} serial={serial_value} worker={worker_value}")
                return 3

    print(f"optimizer_state_roundtrip_max_diff {max_state_diff}")
    return 0 if max_state_diff == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())