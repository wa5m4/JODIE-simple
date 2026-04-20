"""Adapter for running official JODIE baseline and normalizing outputs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class OfficialJodieResult:
    status: str
    reason: str
    mrr: Optional[float]
    recall_at_10: Optional[float]
    repo_path: str
    commit: str
    result_json_path: str


def _git_commit(repo_path: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _normalize_result(result_json_path: str) -> Dict[str, float]:
    if not os.path.exists(result_json_path):
        raise ValueError(f"official result json not found: {result_json_path}")
    with open(result_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "mrr" in payload and "recall_at_10" in payload:
        return {
            "mrr": float(payload["mrr"]),
            "recall_at_10": float(payload["recall_at_10"]),
        }

    metrics = payload.get("metrics", {})
    if "mrr" in metrics and "recall_at_10" in metrics:
        return {
            "mrr": float(metrics["mrr"]),
            "recall_at_10": float(metrics["recall_at_10"]),
        }

    raise ValueError(
        "official result json must contain mrr and recall_at_10 (top-level or under metrics)."
    )


def _run_script_with_xrange(py: str, cwd: str, script_name: str, argv: List[str]) -> Tuple[int, str, str]:
    launcher = (
        "import builtins,runpy,sys,torch;"
        "builtins.xrange=range;"
        "cuda_ok=bool(getattr(torch,'cuda',None)) and torch.cuda.is_available();"
        "import library_models as _lm;"
        "_lm.select_free_gpu=(lambda:'0');"
        "(cuda_ok) or setattr(torch.nn.Module,'cuda',lambda self,*a,**k:self);"
        "(cuda_ok) or setattr(torch.Tensor,'cuda',lambda self,*a,**k:self);"
        f"sys.argv={[script_name] + argv!r};"
        f"runpy.run_path('{script_name}', run_name='__main__')"
    )
    proc = subprocess.run(
        [py, "-c", launcher],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _parse_interaction_results(result_file: str) -> Tuple[float, float]:
    val_by_epoch: Dict[int, Dict[str, float]] = {}
    test_by_epoch: Dict[int, Dict[str, float]] = {}
    current_epoch = -1

    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "Validation performance of epoch" in line:
                current_epoch = int(line.split("epoch ")[1].split()[0])
                val_by_epoch.setdefault(current_epoch, {})
                test_by_epoch.setdefault(current_epoch, {})
                continue
            if line.startswith("Validation: Mean Reciprocal Rank:") and current_epoch >= 0:
                val_by_epoch[current_epoch]["mrr"] = float(line.split(":")[-1].strip())
                continue
            if line.startswith("Validation: Recall@10:") and current_epoch >= 0:
                val_by_epoch[current_epoch]["recall"] = float(line.split(":")[-1].strip())
                continue
            if line.startswith("Test: Mean Reciprocal Rank:") and current_epoch >= 0:
                test_by_epoch[current_epoch]["mrr"] = float(line.split(":")[-1].strip())
                continue
            if line.startswith("Test: Recall@10:") and current_epoch >= 0:
                test_by_epoch[current_epoch]["recall"] = float(line.split(":")[-1].strip())
                continue

    if not val_by_epoch:
        raise ValueError(f"No validation metrics found in: {result_file}")

    best_epoch = max(val_by_epoch.keys(), key=lambda ep: val_by_epoch[ep].get("mrr", -1.0))
    best_test = test_by_epoch.get(best_epoch, {})
    if "mrr" not in best_test or "recall" not in best_test:
        raise ValueError(f"Missing test metrics for selected epoch {best_epoch} in: {result_file}")
    return float(best_test["mrr"]), float(best_test["recall"])


def _run_builtin_official(repo: str, py: str, protocol: Dict, result_json_path: str) -> OfficialJodieResult:
    commit = _git_commit(repo)
    if abs(float(protocol.get("lr", 1e-3)) - 1e-3) > 1e-12:
        return OfficialJodieResult(
            status="skipped",
            reason="official jodie.py uses fixed lr=1e-3; set --lr 1e-3 for fair comparison",
            mrr=None,
            recall_at_10=None,
            repo_path=repo,
            commit=commit,
            result_json_path=result_json_path,
        )

    source_csv = protocol.get("official_dataset_csv", "")
    network_name = protocol.get("official_network_name", "official_jodie_compare")
    if not source_csv or not os.path.exists(source_csv):
        return OfficialJodieResult(
            status="error",
            reason=f"official dataset csv missing: {source_csv}",
            mrr=None,
            recall_at_10=None,
            repo_path=repo,
            commit=commit,
            result_json_path=result_json_path,
        )

    data_dir = os.path.join(repo, "data")
    saved_models_dir = os.path.join(repo, "saved_models", network_name)
    results_dir = os.path.join(repo, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    target_csv = os.path.join(data_dir, f"{network_name}.csv")
    shutil.copyfile(source_csv, target_csv)

    train_prop = str(protocol.get("train_ratio", 0.8))
    epochs = int(protocol.get("epochs", 3))
    emb = int(protocol.get("official_embedding_dim", 128))

    rc, _, err = _run_script_with_xrange(
        py,
        repo,
        "jodie.py",
        [
            "--network",
            network_name,
            "--model",
            "jodie",
            "--epochs",
            str(epochs),
            "--embedding_dim",
            str(emb),
            "--train_proportion",
            train_prop,
        ],
    )
    if rc != 0:
        return OfficialJodieResult(
            status="error",
            reason=f"official training failed: {err.strip()[:500]}",
            mrr=None,
            recall_at_10=None,
            repo_path=repo,
            commit=commit,
            result_json_path=result_json_path,
        )

    result_file = os.path.join(results_dir, f"interaction_prediction_{network_name}.txt")
    if os.path.exists(result_file):
        os.remove(result_file)

    for ep in range(epochs):
        rc, _, err = _run_script_with_xrange(
            py,
            repo,
            "evaluate_interaction_prediction.py",
            [
                "--network",
                network_name,
                "--model",
                "jodie",
                "--epoch",
                str(ep),
                "--embedding_dim",
                str(emb),
                "--train_proportion",
                train_prop,
            ],
        )
        if rc != 0:
            return OfficialJodieResult(
                status="error",
                reason=f"official evaluation failed at epoch {ep}: {err.strip()[:500]}",
                mrr=None,
                recall_at_10=None,
                repo_path=repo,
                commit=commit,
                result_json_path=result_json_path,
            )

    mrr, recall = _parse_interaction_results(result_file)
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump({"mrr": mrr, "recall_at_10": recall}, f, ensure_ascii=False, indent=2)

    return OfficialJodieResult(
        status="ok",
        reason="",
        mrr=mrr,
        recall_at_10=recall,
        repo_path=repo,
        commit=commit,
        result_json_path=result_json_path,
    )


def run_official_jodie_baseline(
    protocol_json_path: str,
    result_json_path: str,
    official_jodie_repo: str,
    official_python: str,
    official_cmd_template: str,
    require_official: bool,
) -> OfficialJodieResult:
    repo = official_jodie_repo.strip()
    if not repo:
        reason = "official repo path is not provided"
        if require_official:
            raise ValueError(reason)
        return OfficialJodieResult("skipped", reason, None, None, "", "unknown", result_json_path)

    if not os.path.isdir(repo):
        reason = f"official repo does not exist: {repo}"
        if require_official:
            raise ValueError(reason)
        return OfficialJodieResult("skipped", reason, None, None, repo, "unknown", result_json_path)

    with open(protocol_json_path, "r", encoding="utf-8") as f:
        protocol = json.load(f)

    commit = _git_commit(repo)
    cmd_template = official_cmd_template.strip()

    if cmd_template:
        cmd = cmd_template.format(
            python=official_python,
            repo=repo,
            protocol_json=protocol_json_path,
            result_json=result_json_path,
        )
        proc = subprocess.run(cmd, shell=True, cwd=repo, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            reason = (
                "official JODIE command failed: "
                f"returncode={proc.returncode}; stderr={proc.stderr.strip()[:500]}"
            )
            if require_official:
                raise ValueError(reason)
            return OfficialJodieResult("error", reason, None, None, repo, commit, result_json_path)

        metrics = _normalize_result(result_json_path)
        return OfficialJodieResult(
            status="ok",
            reason="",
            mrr=metrics["mrr"],
            recall_at_10=metrics["recall_at_10"],
            repo_path=repo,
            commit=commit,
            result_json_path=result_json_path,
        )

    has_builtin = os.path.exists(os.path.join(repo, "jodie.py")) and os.path.exists(
        os.path.join(repo, "evaluate_interaction_prediction.py")
    )
    if has_builtin:
        result = _run_builtin_official(repo, official_python, protocol, result_json_path)
        if require_official and result.status != "ok":
            raise ValueError(result.reason)
        return result

    default_entry = os.path.join(repo, "official_compare_adapter.py")
    if not os.path.exists(default_entry):
        reason = "official repo missing supported entrypoints (jodie.py/evaluate_interaction_prediction.py)"
        if require_official:
            raise ValueError(reason)
        return OfficialJodieResult("skipped", reason, None, None, repo, commit, result_json_path)

    cmd = (
        f"{official_python} official_compare_adapter.py "
        f"--protocol-json {protocol_json_path} --result-json {result_json_path}"
    )
    proc = subprocess.run(cmd, shell=True, cwd=repo, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        reason = (
            "official JODIE command failed: "
            f"returncode={proc.returncode}; stderr={proc.stderr.strip()[:500]}"
        )
        if require_official:
            raise ValueError(reason)
        return OfficialJodieResult("error", reason, None, None, repo, commit, result_json_path)

    metrics = _normalize_result(result_json_path)
    return OfficialJodieResult(
        status="ok",
        reason="",
        mrr=metrics["mrr"],
        recall_at_10=metrics["recall_at_10"],
        repo_path=repo,
        commit=commit,
        result_json_path=result_json_path,
    )
