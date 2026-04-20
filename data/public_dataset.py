"""Load and normalize public JODIE-style datasets into Interaction events."""

from __future__ import annotations

import csv
import math
import os
import urllib.request
from typing import Dict, List, Tuple

import torch

from data.synthetic import Interaction

_JODIE_URLS = {
    "wikipedia": "https://raw.githubusercontent.com/claws-lab/jodie/master/data/wikipedia.csv",
    "reddit": "https://raw.githubusercontent.com/claws-lab/jodie/master/data/reddit.csv",
}


def _resolve_dataset_path(dataset_name: str, dataset_dir: str, local_data_path: str) -> str:
    if local_data_path:
        if not os.path.exists(local_data_path):
            raise ValueError(f"local_data_path does not exist: {local_data_path}")
        return local_data_path

    if dataset_name not in _JODIE_URLS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Use one of: wikipedia, reddit, public_csv."
        )

    os.makedirs(dataset_dir, exist_ok=True)
    target_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
    if os.path.exists(target_path):
        return target_path

    try:
        urllib.request.urlretrieve(_JODIE_URLS[dataset_name], target_path)
    except Exception as exc:  # pragma: no cover
        raise ValueError(
            f"Failed to download dataset '{dataset_name}' to {target_path}. "
            "Please provide --local-data-path manually."
        ) from exc

    return target_path


def _to_float(value: str, path: str, line_no: int, field_name: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no} invalid {field_name}: '{value}'") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path}:{line_no} {field_name} is not finite: '{value}'")
    return number


def _to_int(value: str, path: str, line_no: int, field_name: str) -> int:
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no} invalid {field_name}: '{value}'") from exc


def load_public_dataset(
    dataset_name: str,
    dataset_dir: str,
    feature_dim: int,
    max_events: int = 0,
    local_data_path: str = "",
) -> Tuple[List[Interaction], int, int]:
    if feature_dim <= 0:
        raise ValueError("feature_dim must be > 0")

    if dataset_name == "public_csv":
        if not local_data_path:
            raise ValueError("dataset=public_csv requires --local-data-path")
        dataset_path = _resolve_dataset_path("public_csv", dataset_dir, local_data_path)
    else:
        dataset_path = _resolve_dataset_path(dataset_name, dataset_dir, local_data_path)

    raw_rows: List[Tuple[int, int, float, List[float], int]] = []
    user_map: Dict[int, int] = {}
    item_map: Dict[int, int] = {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue

            # Skip optional header rows like: user_id,item_id,timestamp,state_label,...
            first_col = row[0].strip().lower()
            if first_col in {"user", "user_id"}:
                continue

            if len(row) < 5:
                raise ValueError(
                    f"{dataset_path}:{line_no} expected at least 5 columns "
                    "(user_id,item_id,timestamp,label,features...)"
                )

            raw_uid = _to_int(row[0], dataset_path, line_no, "user_id")
            raw_iid = _to_int(row[1], dataset_path, line_no, "item_id")
            ts = _to_float(row[2], dataset_path, line_no, "timestamp")

            features = [_to_float(v, dataset_path, line_no, "feature") for v in row[4:]]
            if not features:
                raise ValueError(f"{dataset_path}:{line_no} requires at least one feature column")

            if raw_uid not in user_map:
                user_map[raw_uid] = len(user_map)
            if raw_iid not in item_map:
                item_map[raw_iid] = len(item_map)

            raw_rows.append((user_map[raw_uid], item_map[raw_iid], ts, features, line_no))

    if not raw_rows:
        raise ValueError(f"No valid events found in dataset: {dataset_path}")

    raw_rows.sort(key=lambda x: (x[2], x[4]))
    if max_events > 0:
        raw_rows = raw_rows[:max_events]

    interactions: List[Interaction] = []
    for uid, iid, ts, feats, _ in raw_rows:
        if len(feats) >= feature_dim:
            aligned = feats[:feature_dim]
        else:
            aligned = feats + [0.0] * (feature_dim - len(feats))

        interactions.append(
            Interaction(
                timestamp=ts,
                user_id=uid,
                item_id=iid,
                features=torch.tensor(aligned, dtype=torch.float32),
            )
        )

    used_users = {ev.user_id for ev in interactions}
    used_items = {ev.item_id for ev in interactions}
    num_users = max(used_users) + 1
    num_items = max(used_items) + 1
    return interactions, num_users, num_items
