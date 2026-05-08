#!/usr/bin/env python3
"""
两方多种子汇总：A vs B（通用）
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple


def load_best_arch(d: str) -> Dict:
    p = os.path.join(d, "best_arch.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_timing_count(d: str) -> int:
    p = os.path.join(d, "timing_log.csv")
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        return max(0, sum(1 for _ in f) - 1)


def mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    std = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0.0
    return m, std


def fv(v, spec=".4f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(float(v), spec)
    except Exception:
        return "N/A"


def ms_str(m: float, s: float, spec=".4f") -> str:
    return f"{format(m, spec)} ± {format(s, spec)}"


def load_seed_times(root: str) -> Dict[int, Tuple]:
    times: Dict[int, Tuple] = {}
    p = os.path.join(root, "seed_times.csv")
    if not os.path.exists(p):
        return times
    with open(p) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                try:
                    times[int(parts[0])] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    pass
    return times


def generate_report(
    root: str, seeds: List[int],
    a_label: str, b_label: str,
    a_subdir: str, b_subdir: str,
    title: str,
) -> str:
    seed_times = load_seed_times(root)
    records = []
    a_counts, b_counts = [], []

    for seed in seeds:
        seed_dir = os.path.join(root, f"seed_{seed}")
        ab = load_best_arch(os.path.join(seed_dir, a_subdir))
        bb = load_best_arch(os.path.join(seed_dir, b_subdir))
        times = seed_times.get(seed, (None, None))
        a_time = times[0] if len(times) > 0 else None
        b_time = times[1] if len(times) > 1 else None
        ac = load_timing_count(os.path.join(seed_dir, a_subdir))
        bc = load_timing_count(os.path.join(seed_dir, b_subdir))
        if ac: a_counts.append(ac)
        if bc: b_counts.append(bc)
        records.append({
            "seed":    seed,
            "a_score": ab.get("score") or ab.get("mrr"),
            "b_score": bb.get("score") or bb.get("mrr"),
            "a_mrr":   ab.get("mrr"),
            "b_mrr":   bb.get("mrr"),
            "a_r10":   ab.get("recall_at_k"),
            "b_r10":   bb.get("recall_at_k"),
            "a_time":  a_time,
            "b_time":  b_time,
        })

    def get_vals(key):
        return [r[key] for r in records if r.get(key) is not None]

    avg_a = int(sum(a_counts) / len(a_counts)) if a_counts else 0
    avg_b = int(sum(b_counts) / len(b_counts)) if b_counts else 0

    L = []
    W = 70
    L.append("╔" + "═"*W + "╗")
    L.append("║{:^{}}║".format(title, W))
    L.append("╚" + "═"*W + "╝")
    L.append("")
    L.append(f"  Seeds: {seeds}")
    L.append(f"  {a_label}: ~{avg_a} trials/seed  |  {b_label}: ~{avg_b} trials/seed")
    L.append("")

    # ── 1. Per-seed
    L.append("┌─ 1. Per-seed Results " + "─"*48 + "┐")
    L.append("")
    L.append(f"  {'Seed':>5}  {a_label:>12}  {b_label:>12}  {'Winner':>10}")
    L.append(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*10}")
    for r in records:
        av, bv = r["a_score"], r["b_score"]
        try:
            winner = a_label if float(av) >= float(bv) else b_label
        except Exception:
            winner = "?"
        L.append(f"  {r['seed']:>5}  {fv(av):>12}  {fv(bv):>12}  {winner:>10} ◀")
    L.append("")
    L.append("└" + "─"*W + "┘")
    L.append("")

    # ── 2. 均值±标准差
    L.append("┌─ 2. Aggregate Statistics " + "─"*44 + "┐")
    L.append("")
    L.append(f"  {'Metric':<18}  {a_label:>22}  {b_label:>22}  {'Best':>6}")
    L.append(f"  {'─'*18}  {'─'*22}  {'─'*22}  {'─'*6}")

    for label, ak, bk in [
        ("Best Score (MRR)", "a_score", "b_score"),
        ("MRR",              "a_mrr",   "b_mrr"),
        ("Recall@10",        "a_r10",   "b_r10"),
    ]:
        av_list, bv_list = get_vals(ak), get_vals(bk)
        if not av_list and not bv_list:
            continue
        am, as_ = mean_std(av_list) if av_list else (None, None)
        bm, bs_ = mean_std(bv_list) if bv_list else (None, None)
        try:
            winner = a_label if float(am) >= float(bm) else b_label
        except Exception:
            winner = "?"
        a_str = ms_str(am, as_) if am is not None else "N/A"
        b_str = ms_str(bm, bs_) if bm is not None else "N/A"
        L.append(f"  {label:<18}  {a_str:>22}  {b_str:>22}  {winner + ' ◀':>6}")

    L.append("")
    a_times = get_vals("a_time")
    b_times = get_vals("b_time")
    if a_times:
        am_t, as_t = mean_std(a_times)
        L.append(f"  {'Wall Time (s)':<18}  {ms_str(am_t, as_t, '.1f'):>22}")
    if b_times:
        bm_t, bs_t = mean_std(b_times)
        L.append(f"  {'(' + b_label + ')':<18}  {'':>22}  {ms_str(bm_t, bs_t, '.1f'):>22}")

    if a_times and b_times and avg_a and avg_b:
        L.append("")
        a_tph_list = [3600 / t * avg_a for t in a_times if t > 0]
        b_tph_list = [3600 / t * avg_b for t in b_times if t > 0]
        if a_tph_list and b_tph_list:
            am_tph, _ = mean_std(a_tph_list)
            bm_tph, bs_tph = mean_std(b_tph_list)
            ratio = bm_tph / am_tph if am_tph > 0 else 0
            L.append(f"  {'Throughput (tph)':<18}  {ms_str(am_tph, 0, '.1f'):>22}  {ms_str(bm_tph, bs_tph, '.1f'):>22}")
            L.append(f"  {'Throughput Ratio':<18}  {'1.00x':>22}  {f'{ratio:.2f}x':>22}  {b_label + ' ◀':>6}")

    L.append("")
    L.append("└" + "─"*W + "┘")
    L.append("")

    # ── 3. Win Count
    L.append("┌─ 3. Win Count " + "─"*55 + "┐")
    L.append("")
    n = len(records)
    for method in [a_label, b_label]:
        wins = sum(
            1 for r in records
            if r.get("a_score") is not None and r.get("b_score") is not None
            and (float(r["a_score"]) >= float(r["b_score"])) == (method == a_label)
        )
        bar_str = "█" * wins + "░" * (n - wins)
        L.append(f"  {method:<14} wins : {wins}/{n}  {bar_str}")
    L.append("")
    L.append("└" + "─"*W + "┘")

    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",     required=True)
    parser.add_argument("--seeds",    required=True)
    parser.add_argument("--a-label",  default="A")
    parser.add_argument("--b-label",  default="B")
    parser.add_argument("--a-subdir", default=None)
    parser.add_argument("--b-subdir", default=None)
    parser.add_argument("--title",    default="NAS 2-Way Multi-Seed")
    parser.add_argument("--output",   default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    a_subdir = args.a_subdir or args.a_label.lower().replace("-", "_").replace(" ", "_")
    b_subdir = args.b_subdir or args.b_label.lower().replace("-", "_").replace(" ", "_")

    report = generate_report(
        root=args.root, seeds=seeds,
        a_label=args.a_label, b_label=args.b_label,
        a_subdir=a_subdir, b_subdir=b_subdir,
        title=args.title,
    )

    print(report)
    out = args.output or os.path.join(args.root, "aggregate_report_2way.txt")
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Aggregate report saved → {out}")


if __name__ == "__main__":
    main()
