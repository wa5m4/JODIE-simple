"""Compare searched architecture against jodie_rnn baseline on public datasets."""

from __future__ import annotations

import argparse
import json
import os
import random
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import numpy as np
import torch

from data.public_dataset import load_public_dataset
from data.synthetic import init_dynamic_graph_state
from models.factory import build_model
from models.jodie_rnn import JODIERNN
from models.training import evaluate_ranking_metrics, train_model_ce


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare searched model and jodie_rnn baseline on public datasets")
    parser.add_argument("--dataset", choices=["wikipedia", "reddit", "public_csv"], default="wikipedia")
    parser.add_argument("--dataset-dir", type=str, default="data/public")
    parser.add_argument("--local-data-path", type=str, default="")
    parser.add_argument("--best-arch-path", type=str, default="outputs/best_arch.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds for multi-seed evaluation. Example: 42,43,44",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/public_compare")
    parser.add_argument("--strict-meta-check", action="store_true", help="Fail when run config mismatches best_arch metadata.")
    return parser.parse_args()


def _parse_seeds(seed: int, seeds_text: str) -> List[int]:
    if not seeds_text.strip():
        return [seed]
    values = [s.strip() for s in seeds_text.split(",") if s.strip()]
    if not values:
        return [seed]
    return [int(v) for v in values]


def _split(interactions: List, train_ratio: float, val_ratio: float) -> Tuple[List, List, List]:
    interactions = sorted(interactions, key=lambda x: x.timestamp)
    total = len(interactions)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_end = max(1, min(train_end, total - 2))
    val_end = max(train_end + 1, min(val_end, total - 1))
    return interactions[:train_end], interactions[train_end:val_end], interactions[val_end:]


def _load_best_arch(path: str) -> Tuple[Dict, Dict]:
    if not os.path.exists(path):
        raise ValueError(f"best_arch.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid best architecture file: {path}")
    metadata = payload.get("distribution_metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid distribution metadata in: {path}")
    return cfg, metadata


def _validate_metadata(args, metadata: Dict, train_data: List, val_data: List, test_data: List) -> None:
    if not metadata:
        return
    checks = {
        "dataset": args.dataset,
        "max_events": int(args.max_events),
        "feature_dim": int(args.feature_dim),
        "lr": float(args.lr),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "k": int(args.k),
        "num_train_events": len(train_data),
        "num_val_events": len(val_data),
        "num_test_events": len(test_data),
    }
    mismatches = []
    for key, expected in checks.items():
        if key in metadata and metadata[key] != expected:
            mismatches.append(f"{key}: best_arch={metadata[key]} current={expected}")
    if mismatches:
        text = "Search/compare distribution mismatch: " + "; ".join(mismatches)
        if args.strict_meta_check:
            raise ValueError(text)
        print(f"[Warning] {text}")


def _build_jodie_rnn_config(best_config: Dict, num_users: int, num_items: int, feature_dim: int) -> Dict:
    config = dict(best_config)
    config["num_users"] = num_users
    config["num_items"] = num_items
    config["feature_dim"] = feature_dim
    config["model"] = "jodie_rnn"
    return config


def _build_jodie_rnn_model(config: Dict) -> JODIERNN:
    time_proj_mode = str(config.get("time_proj", "linear")).lower()
    use_time_proj = time_proj_mode not in {"off", "none"}
    return JODIERNN(
        num_users=config["num_users"],
        num_items=config["num_items"],
        embedding_dim=config.get("embedding_dim", 32),
        feature_dim=config.get("feature_dim", 8),
        cell_type=config.get("memory_cell", "rnn"),
        use_time_proj=use_time_proj,
    )


def _aggregate_metric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(pstdev(values))}


def _evaluate_single_seed(
    seed: int,
    best_config_template: Dict,
    train_data: List,
    test_data: List,
    num_users: int,
    num_items: int,
    feature_dim: int,
    epochs: int,
    lr: float,
    k: int,
) -> Dict:
    _set_seed(seed)

    best_config = dict(best_config_template)
    best_config["num_users"] = num_users
    best_config["num_items"] = num_items
    best_config["feature_dim"] = feature_dim

    searched_model = build_model(best_config)
    searched_graph = None
    if best_config.get("model", "temporal_event_gnn_jodie") != "jodie_rnn":
        searched_graph = init_dynamic_graph_state(
            num_users=num_users,
            num_items=num_items,
            max_neighbors=best_config.get("max_neighbors", 20),
        )
    train_model_ce(
        searched_model,
        train_data,
        num_epochs=epochs,
        lr=lr,
        graph_ctx=searched_graph,
        seed=seed,
    )
    searched_metrics = evaluate_ranking_metrics(
        searched_model,
        test_data,
        k=k,
        graph_ctx=searched_graph,
    )

    jodie_rnn_config = _build_jodie_rnn_config(best_config, num_users, num_items, feature_dim)
    jodie_rnn_model = _build_jodie_rnn_model(jodie_rnn_config)
    train_model_ce(
        jodie_rnn_model,
        train_data,
        num_epochs=epochs,
        lr=lr,
        graph_ctx=None,
        seed=seed,
    )
    jodie_rnn_metrics = evaluate_ranking_metrics(
        jodie_rnn_model,
        test_data,
        k=k,
        graph_ctx=None,
    )

    return {
        "seed": seed,
        "searched_model": {
            "model": best_config.get("model", "temporal_event_gnn_jodie"),
            "mrr": float(searched_metrics["mrr"]),
            "recall_at_k": float(searched_metrics["recall_at_k"]),
        },
        "jodie_rnn": {
            "model": "jodie_rnn",
            "cell_type": jodie_rnn_config.get("memory_cell", "rnn"),
            "use_time_proj": str(jodie_rnn_config.get("time_proj", "linear")).lower() not in {"off", "none"},
            "mrr": float(jodie_rnn_metrics["mrr"]),
            "recall_at_k": float(jodie_rnn_metrics["recall_at_k"]),
        },
        "delta_mrr": float(searched_metrics["mrr"] - jodie_rnn_metrics["mrr"]),
        "delta_recall_at_k": float(searched_metrics["recall_at_k"] - jodie_rnn_metrics["recall_at_k"]),
    }


def main():
    args = parse_args()

    interactions, num_users, num_items = load_public_dataset(
        dataset_name=args.dataset,
        dataset_dir=args.dataset_dir,
        feature_dim=args.feature_dim,
        max_events=args.max_events,
        local_data_path=args.local_data_path,
    )
    train_data, val_data, test_data = _split(interactions, args.train_ratio, args.val_ratio)
    fit_data = train_data + val_data

    best_config, best_metadata = _load_best_arch(args.best_arch_path)
    _validate_metadata(args, best_metadata, train_data, val_data, test_data)
    seeds = _parse_seeds(args.seed, args.seeds)

    per_seed_results = [
        _evaluate_single_seed(
            seed=seed,
            best_config_template=best_config,
            train_data=fit_data,
            test_data=test_data,
            num_users=num_users,
            num_items=num_items,
            feature_dim=args.feature_dim,
            epochs=args.epochs,
            lr=args.lr,
            k=args.k,
        )
        for seed in seeds
    ]

    searched_mrr_values = [r["searched_model"]["mrr"] for r in per_seed_results]
    searched_recall_values = [r["searched_model"]["recall_at_k"] for r in per_seed_results]
    baseline_mrr_values = [r["jodie_rnn"]["mrr"] for r in per_seed_results]
    baseline_recall_values = [r["jodie_rnn"]["recall_at_k"] for r in per_seed_results]
    delta_mrr_values = [r["delta_mrr"] for r in per_seed_results]
    delta_recall_values = [r["delta_recall_at_k"] for r in per_seed_results]

    summary = {
        "searched_model": {
            "mrr": _aggregate_metric(searched_mrr_values),
            "recall_at_k": _aggregate_metric(searched_recall_values),
        },
        "jodie_rnn": {
            "mrr": _aggregate_metric(baseline_mrr_values),
            "recall_at_k": _aggregate_metric(baseline_recall_values),
        },
        "delta": {
            "mrr": _aggregate_metric(delta_mrr_values),
            "recall_at_k": _aggregate_metric(delta_recall_values),
        },
    }

    result = {
        "dataset": args.dataset,
        "best_arch_path": args.best_arch_path,
        "epochs": args.epochs,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "lr": args.lr,
        "feature_dim": args.feature_dim,
        "num_users": num_users,
        "num_items": num_items,
        "num_train_events": len(train_data),
        "num_val_events": len(val_data),
        "num_fit_events": len(fit_data),
        "num_test_events": len(test_data),
        "k": args.k,
        "candidate_policy": "global_item_set",
        "seeds": seeds,
        "per_seed": per_seed_results,
        "summary": summary,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "comparison_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Dataset: {args.dataset}")
    print(f"Seeds: {seeds}")
    print(f"Searched model MRR (mean±std): {summary['searched_model']['mrr']['mean']:.4f}±{summary['searched_model']['mrr']['std']:.4f}")
    print(f"Searched model Recall@{args.k} (mean±std): {summary['searched_model']['recall_at_k']['mean']:.4f}±{summary['searched_model']['recall_at_k']['std']:.4f}")
    print(f"jodie_rnn MRR (mean±std): {summary['jodie_rnn']['mrr']['mean']:.4f}±{summary['jodie_rnn']['mrr']['std']:.4f}")
    print(f"jodie_rnn Recall@{args.k} (mean±std): {summary['jodie_rnn']['recall_at_k']['mean']:.4f}±{summary['jodie_rnn']['recall_at_k']['std']:.4f}")
    print(f"Delta MRR (searched - jodie_rnn): {summary['delta']['mrr']['mean']:.4f}±{summary['delta']['mrr']['std']:.4f}")
    print(f"Delta Recall@{args.k} (searched - jodie_rnn): {summary['delta']['recall_at_k']['mean']:.4f}±{summary['delta']['recall_at_k']['std']:.4f}")
    print(f"Saved comparison to: {out_path}")


if __name__ == "__main__":
    main()
