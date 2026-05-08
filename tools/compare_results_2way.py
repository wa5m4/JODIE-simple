#!/usr/bin/env python3
"""
两方对比报告：A vs B（通用，支持 Serial vs Pipeline 或 DataParallel vs Pipeline）
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional


def load_best_arch(d: str) -> Dict:
    p = os.path.join(d, "best_arch.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_leaderboard(d: str) -> List[Dict]:
    p = os.path.join(d, "leaderboard.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_timing_log(d: str) -> List[Dict]:
    p = os.path.join(d, "timing_log.csv")
    if not os.path.exists(p):
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "trial_id": int(row["trial_id"]),
                    "end_time_s": float(row["end_time_s"]),
                    "cumulative_best_score": float(row["cumulative_best_score"]),
                })
            except (KeyError, ValueError):
                pass
    return rows


def fv(v, spec=".4f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(float(v), spec)
    except Exception:
        return "N/A"


def bar(v: float, maxv: float, w: int = 36) -> str:
    ratio = min(max(float(v) / max(float(maxv), 1e-9), 0.0), 1.0)
    n = int(ratio * w)
    return "█" * n + "░" * (w - n)


def top_scores(lb: List[Dict], k: int = 5) -> List[float]:
    vals = []
    for row in lb:
        try:
            vals.append(float(row.get("score") or row.get("mrr") or 0))
        except Exception:
            pass
    return sorted(vals, reverse=True)[:k]


def arch_diversity(lb: List[Dict]) -> Dict:
    models, cells, projs = set(), set(), set()
    for row in lb:
        cfg_str = row.get("config_json", "")
        if not cfg_str:
            continue
        try:
            cfg = json.loads(cfg_str)
        except Exception:
            continue
        models.add(cfg.get("model", "?"))
        cells.add(cfg.get("memory_cell", "?"))
        projs.add(cfg.get("time_proj", "?"))
    return {
        "num_archs": len(lb),
        "unique_models": len(models),
        "unique_memory_cell": len(cells),
        "unique_time_proj": len(projs),
    }


def render_curve_2way(
    a_timing: List[Dict], b_timing: List[Dict],
    a_label: str, b_label: str,
    width: int = 52, height: int = 10,
) -> str:
    ax = [r["end_time_s"] for r in a_timing]
    ay = [r["cumulative_best_score"] for r in a_timing]
    bx = [r["end_time_s"] for r in b_timing]
    by = [r["cumulative_best_score"] for r in b_timing]
    all_x = ax + bx
    all_y = ay + by
    if not all_x:
        return "  (no timing data)\n"

    max_x = max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    yr = max(max_y - min_y, 1e-6)

    grid = [[" "] * width for _ in range(height)]
    a_ch, b_ch = a_label[0].upper(), b_label[0].upper()
    if a_ch == b_ch:
        b_ch = "B"

    def plot(xs, ys, ch):
        for x, y in zip(xs, ys):
            c = int((x / max_x) * (width - 1)) if max_x > 0 else 0
            r = height - 1 - int(((y - min_y) / yr) * (height - 1))
            c, r = max(0, min(c, width - 1)), max(0, min(r, height - 1))
            if grid[r][c] == " ":
                grid[r][c] = ch
            elif grid[r][c] != ch:
                grid[r][c] = "*"

    plot(ax, ay, a_ch)
    plot(bx, by, b_ch)

    lines = [f"  Best Score vs Wall Time  [{a_ch}={a_label}  {b_ch}={b_label}  *=overlap]", ""]
    ticks = [max_y - yr * i / (height - 1) for i in range(height)]
    for i, row in enumerate(grid):
        lines.append(f"  {ticks[i]:.4f} │{''.join(row)}")
    lines.append(f"  {'':8s} └{'─'*width}")
    lines.append(f"  {'0s':10s}{'':>{width//2-4}s}{max_x:.0f}s")
    return "\n".join(lines)


def generate_report(
    a_dir: str, b_dir: str,
    a_label: str, b_label: str,
    a_time: Optional[float], b_time: Optional[float],
    title: str, conclusion: str,
) -> str:
    ab = load_best_arch(a_dir)
    bb = load_best_arch(b_dir)
    a_lb = load_leaderboard(a_dir)
    b_lb = load_leaderboard(b_dir)
    a_timing = load_timing_log(a_dir)
    b_timing = load_timing_log(b_dir)

    a_score = ab.get("score") or ab.get("mrr")
    b_score = bb.get("score") or bb.get("mrr")
    a_mrr   = ab.get("mrr")
    b_mrr   = bb.get("mrr")
    a_r10   = ab.get("recall_at_k")
    b_r10   = bb.get("recall_at_k")

    if a_time is None and a_timing:
        a_time = max(r["end_time_s"] for r in a_timing)
    if b_time is None and b_timing:
        b_time = max(r["end_time_s"] for r in b_timing)

    a_trials = len(a_timing) if a_timing else 0
    b_trials = len(b_timing) if b_timing else 0

    a_tph = (3600 / a_time * a_trials) if a_time and a_trials else None
    b_tph = (3600 / b_time * b_trials) if b_time and b_trials else None

    def best2(av, bv, higher=True):
        try:
            fa, fb = float(av), float(bv)
            if higher:
                return a_label if fa >= fb else b_label
            else:
                return a_label if fa <= fb else b_label
        except Exception:
            return "?"

    L = []
    W = 70

    L.append("╔" + "═"*W + "╗")
    L.append("║{:^{}}║".format(title, W))
    L.append("╚" + "═"*W + "╝")
    L.append("")

    # ── 1. 搜索质量
    L.append("┌─ 1. Search Quality  架构搜索质量 " + "─"*36 + "┐")
    L.append("")
    L.append(f"  {'Metric':<18}  {a_label:>14}  {b_label:>14}  {'Best':>6}")
    L.append(f"  {'─'*18}  {'─'*14}  {'─'*14}  {'─'*6}")
    for label, av, bv in [
        ("Best Score (MRR)", a_score, b_score),
        ("MRR",              a_mrr,   b_mrr),
        ("Recall@10",        a_r10,   b_r10),
    ]:
        try:
            winner = best2(av, bv)
            marker = f"{winner} ◀"
        except Exception:
            marker = "?"
        L.append(f"  {label:<18}  {fv(av):>14}  {fv(bv):>14}  {marker:>6}")

    L.append("")
    L.append("  Top-5 val scores per method:")
    for label, lb in [(a_label, a_lb), (b_label, b_lb)]:
        ts = top_scores(lb)
        L.append(f"    {label:<14}: {' | '.join(fv(v) for v in ts) or 'N/A'}")
    L.append("")

    for label, best in [(a_label, ab), (b_label, bb)]:
        cfg = best.get("config", {})
        if cfg:
            L.append(f"  Best Architecture ({label}):")
            L.append(f"    model={cfg.get('model','?')}  memory_cell={cfg.get('memory_cell','?')}  "
                     f"embedding_dim={cfg.get('embedding_dim','?')}")
    L.append("")
    L.append("└" + "─"*W + "┘")
    L.append("")

    # ── 2. 搜索效率
    L.append("┌─ 2. Search Efficiency  搜索效率 " + "─"*37 + "┐")
    L.append("")
    L.append(f"  {'Metric':<26}  {a_label:>14}  {b_label:>14}  {'Best':>6}")
    L.append(f"  {'─'*26}  {'─'*14}  {'─'*14}  {'─'*6}")

    eff_rows = [
        ("Architectures Explored", a_trials,  b_trials,  True,  "d"),
        ("Total Wall Time (s)",    a_time,    b_time,    False, ".0f"),
        ("Avg Time/Trial (s)",
         (a_time/a_trials if a_time and a_trials else None),
         (b_time/b_trials if b_time and b_trials else None),
         False, ".1f"),
        ("Trial Throughput (tph)", a_tph,     b_tph,     True,  ".1f"),
    ]
    for label, av, bv, higher, spec in eff_rows:
        try:
            winner = best2(av, bv, higher)
            marker = f"{winner} ◀"
        except Exception:
            marker = "?"
        L.append(f"  {label:<26}  {fv(av, spec):>14}  {fv(bv, spec):>14}  {marker:>6}")

    L.append("")
    max_tc = max(t for t in [a_trials, b_trials] if t) or 1
    for label, cnt in [(a_label, a_trials), (b_label, b_trials)]:
        L.append(f"  {label:<14} [{bar(cnt, max_tc)}] {cnt} trials")
    L.append("")
    L.append("└" + "─"*W + "┘")
    L.append("")

    # ── 3. 架构多样性
    L.append("┌─ 3. Architecture Diversity  架构多样性 " + "─"*30 + "┐")
    L.append("")
    a_div = arch_diversity(a_lb)
    b_div = arch_diversity(b_lb)
    L.append(f"  {'Metric':<24}  {a_label:>14}  {b_label:>14}")
    L.append(f"  {'─'*24}  {'─'*14}  {'─'*14}")
    for key, label in [
        ("num_archs",          "Architectures Evaluated"),
        ("unique_models",      "Unique Model Types"),
        ("unique_memory_cell", "Unique Memory Cell"),
        ("unique_time_proj",   "Unique Time Proj"),
    ]:
        L.append(f"  {label:<24}  {a_div.get(key,0):>14}  {b_div.get(key,0):>14}")
    L.append("")
    L.append("└" + "─"*W + "┘")
    L.append("")

    # ── 4. 收敛曲线
    L.append("┌─ 4. Search Convergence Curve  搜索收敛曲线 " + "─"*26 + "┐")
    L.append("")
    L.append(render_curve_2way(a_timing, b_timing, a_label, b_label))
    L.append("")
    L.append("└" + "─"*W + "┘")
    L.append("")

    # ── 5. 结论
    L.append("┌─ 5. Summary  综合评估 " + "─"*47 + "┐")
    L.append("")
    if a_score and b_score:
        try:
            diff = float(b_score) - float(a_score)
            pct  = diff / float(a_score) * 100
            L.append(f"  {b_label} vs {a_label} — Best Score: {fv(a_score)} → {fv(b_score)} ({diff:+.4f}, {pct:+.1f}%)")
        except Exception:
            pass
    L.append("")
    if a_tph and b_tph:
        ratio = b_tph / a_tph if a_tph > 0 else 1
        L.append(f"  Throughput (tph):")
        L.append(f"    {a_label:<14}: {a_tph:.1f}  (1.00x, {a_trials} archs explored)")
        L.append(f"    {b_label:<14}: {b_tph:.1f}  ({ratio:.2f}x, {b_trials} archs explored)")
        L.append("")
    L.append("  结论：")
    for line in conclusion.strip().split("\n"):
        L.append(f"    {line}")
    L.append("")
    L.append("└" + "─"*W + "┘")

    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-dir",      required=True)
    parser.add_argument("--b-dir",      required=True)
    parser.add_argument("--a-label",    default="A")
    parser.add_argument("--b-label",    default="B")
    parser.add_argument("--a-time",     type=float, default=None)
    parser.add_argument("--b-time",     type=float, default=None)
    parser.add_argument("--title",      default="NAS 2-Way Comparison")
    parser.add_argument("--conclusion", default="Pipeline 在架构级并行上优于对比方法。")
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    report = generate_report(
        a_dir=args.a_dir, b_dir=args.b_dir,
        a_label=args.a_label, b_label=args.b_label,
        a_time=args.a_time, b_time=args.b_time,
        title=args.title, conclusion=args.conclusion,
    )

    print(report)
    out = args.output or "outputs/compare_2way_report.txt"
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report saved → {out}")


if __name__ == "__main__":
    main()
