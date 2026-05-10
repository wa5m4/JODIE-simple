#!/usr/bin/env python3
"""
固定 Trial 数对比报告：Serial vs Data-Parallel vs Pipeline-Naive vs Pipeline-Smart
核心指标：相同架构数下，谁更快（时间）、谁质量更好（MRR）
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional


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
                    "score": float(row["score"]),
                    "cumulative_best_score": float(row["cumulative_best_score"]),
                })
            except (KeyError, ValueError):
                pass
    return rows


def best_score(lb: List[Dict]) -> Optional[float]:
    valid = [float(r["score"]) for r in lb if float(r.get("score", 0)) > 0]
    return max(valid) if valid else None


def top5_scores(lb: List[Dict]) -> List[float]:
    valid = sorted([float(r["score"]) for r in lb if float(r.get("score", 0)) > 0], reverse=True)
    return valid[:5]


def best_arch_str(lb: List[Dict]) -> str:
    valid = [r for r in lb if float(r.get("score", 0)) > 0]
    if not valid:
        return "N/A"
    best = max(valid, key=lambda r: float(r["score"]))
    try:
        cfg = json.loads(best.get("config_json", "{}"))
        return f"model={cfg.get('model','?')}  memory_cell={cfg.get('memory_cell','?')}  embedding_dim={cfg.get('embedding_dim','?')}"
    except Exception:
        return "N/A"


def wall_time(timing: List[Dict], fallback: Optional[float]) -> Optional[float]:
    if timing:
        return max(r["end_time_s"] for r in timing)
    return fallback


def fv(v, spec=".4f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(float(v), spec)
    except Exception:
        return "N/A"


def bar(v: Optional[float], best: Optional[float], width: int = 20) -> str:
    if v is None or best is None or best == 0:
        return "░" * width
    ratio = min(v / best, 1.0)
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)


def generate_report(
    serial_dir, dp_dir, naive_dir, smart_dir,
    serial_time, dp_time, naive_time, smart_time,
    trials: int,
) -> str:
    s_lb = load_leaderboard(serial_dir)
    d_lb = load_leaderboard(dp_dir)
    n_lb = load_leaderboard(naive_dir)
    p_lb = load_leaderboard(smart_dir)

    s_timing = load_timing_log(serial_dir)
    d_timing = load_timing_log(dp_dir)
    n_timing = load_timing_log(naive_dir)
    p_timing = load_timing_log(smart_dir)

    s_score = best_score(s_lb)
    d_score = best_score(d_lb)
    n_score = best_score(n_lb)
    p_score = best_score(p_lb)

    s_time = wall_time(s_timing, serial_time)
    d_time = wall_time(d_timing, dp_time)
    n_time = wall_time(n_timing, naive_time)
    p_time = wall_time(p_timing, smart_time)

    s_trials = len(s_timing) if s_timing else trials
    d_trials = len(d_timing) if d_timing else trials
    n_trials = len(n_timing) if n_timing else trials
    p_trials = len(p_timing) if p_timing else trials

    methods = ["Serial", "DataParallel", "Pipeline-Naive", "Pipeline-Smart"]
    scores = [s_score, d_score, n_score, p_score]
    times = [s_time, d_time, n_time, p_time]
    trial_counts = [s_trials, d_trials, n_trials, p_trials]
    lbs = [s_lb, d_lb, n_lb, p_lb]

    best_s = max((s for s in scores if s is not None), default=None)
    best_t = min((t for t in times if t is not None), default=None)

    def winner_tag(vals, best_fn=max):
        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        if not valid:
            return [""] * len(vals)
        best_v = best_fn(v for _, v in valid)
        return ["◀" if v == best_v else "" for _, v in [(i, vals[i]) for i in range(len(vals))]]

    score_tags = winner_tag(scores, max)
    time_tags = winner_tag(times, min)

    lines = []
    W = 72
    lines.append("╔" + "═" * W + "╗")
    lines.append("║" + "  固定 Trial 数对比：Serial vs DP vs Pipeline-Naive vs Pipeline-Smart  ".center(W) + "║")
    lines.append("╚" + "═" * W + "╝")
    lines.append("")

    # ── 1. 搜索质量
    lines.append("┌─ 1. Search Quality  搜索质量（相同 trial 数下谁找到更好的架构）─────────┐")
    lines.append("")
    col = 16
    lines.append(f"  {'Metric':<{col}}  {'Serial':>12}  {'DataParallel':>12}  {'Pipeline-Naive':>14}  {'Pipeline-Smart':>14}  {'Best'}")
    lines.append(f"  {'─'*col}  {'─'*12}  {'─'*12}  {'─'*14}  {'─'*14}  {'─'*6}")
    lines.append(f"  {'Best Score (MRR)':<{col}}  {fv(s_score):>12}  {fv(d_score):>12}  {fv(n_score):>14}  {fv(p_score):>14}  {methods[score_tags.index('◀')] if '◀' in score_tags else 'N/A'} ◀")

    lines.append("")
    lines.append("  Top-5 val scores per method:")
    for m, lb in zip(methods, lbs):
        t5 = top5_scores(lb)
        s = " | ".join(fv(v) for v in t5) if t5 else "N/A"
        lines.append(f"    {m:<16}: {s}")

    lines.append("")
    lines.append("  Best Architecture:")
    for m, lb in zip(methods, lbs):
        lines.append(f"    {m:<16}: {best_arch_str(lb)}")
    lines.append("")
    lines.append("└" + "─" * W + "┘")
    lines.append("")

    # ── 2. 搜索效率
    lines.append("┌─ 2. Search Efficiency  搜索效率（相同架构数下谁更快）──────────────────┐")
    lines.append("")
    lines.append(f"  {'Metric':<26}  {'Serial':>12}  {'DataParallel':>12}  {'Pipeline-Naive':>14}  {'Pipeline-Smart':>14}  {'Best'}")
    lines.append(f"  {'─'*26}  {'─'*12}  {'─'*12}  {'─'*14}  {'─'*14}  {'─'*6}")

    def speedup(t):
        if t is None or s_time is None or s_time == 0:
            return "N/A"
        return f"{s_time/t:.2f}x"

    lines.append(f"  {'Total Wall Time (s)':<26}  {fv(s_time,',.0f'):>12}  {fv(d_time,',.0f'):>12}  {fv(n_time,',.0f'):>14}  {fv(p_time,',.0f'):>14}  {methods[time_tags.index('◀')] if '◀' in time_tags else 'N/A'} ◀")
    lines.append(f"  {'Speedup vs Serial':<26}  {'1.00x':>12}  {speedup(d_time):>12}  {speedup(n_time):>14}  {speedup(p_time):>14}")

    avg_times = [t/c if t and c else None for t, c in zip(times, trial_counts)]
    lines.append(f"  {'Avg Time/Trial (s)':<26}  {fv(avg_times[0],'.1f'):>12}  {fv(avg_times[1],'.1f'):>12}  {fv(avg_times[2],'.1f'):>14}  {fv(avg_times[3],'.1f'):>14}")

    lines.append("")
    lines.append(f"  Trials completed: {s_trials} / {d_trials} / {n_trials} / {p_trials}  (target={trials})")
    lines.append("")

    # 时间条形图
    max_t = max((t for t in times if t is not None), default=1)
    for m, t in zip(methods, times):
        b = bar(t, max_t, 36) if t else "░" * 36
        lines.append(f"  {m:<16} [{b}] {fv(t,',.0f')}s")
    lines.append("")
    lines.append("└" + "─" * W + "┘")
    lines.append("")

    # ── 3. 综合结论
    lines.append("┌─ 3. Summary  综合评估 ─────────────────────────────────────────────────┐")
    lines.append("")
    lines.append(f"  固定 {trials} 个 trial，比较各方法的运行时间和搜索质量：")
    lines.append("")
    for m, t, s in zip(methods, times, scores):
        sp = speedup(t) if m != "Serial" else "baseline"
        lines.append(f"    {m:<16}: 时间={fv(t,',.0f')}s ({sp}),  best_score={fv(s)}")
    lines.append("")

    # 找最快和最好
    valid_times = [(m, t) for m, t in zip(methods, times) if t is not None]
    valid_scores = [(m, s) for m, s in zip(methods, scores) if s is not None]
    if valid_times:
        fastest = min(valid_times, key=lambda x: x[1])
        lines.append(f"  最快方法: {fastest[0]} ({fv(fastest[1],',.0f')}s)")
    if valid_scores:
        best_q = max(valid_scores, key=lambda x: x[1])
        lines.append(f"  最高质量: {best_q[0]} (MRR={fv(best_q[1])})")
    lines.append("")
    lines.append("└" + "─" * W + "┘")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-dir",   required=True)
    parser.add_argument("--dp-dir",       required=True)
    parser.add_argument("--naive-dir",    required=True)
    parser.add_argument("--smart-dir",    required=True)
    parser.add_argument("--serial-time",  type=float, default=None)
    parser.add_argument("--dp-time",      type=float, default=None)
    parser.add_argument("--naive-time",   type=float, default=None)
    parser.add_argument("--smart-time",   type=float, default=None)
    parser.add_argument("--trials",       type=int, default=27)
    parser.add_argument("--output",       default=None)
    args = parser.parse_args()

    report = generate_report(
        args.serial_dir, args.dp_dir, args.naive_dir, args.smart_dir,
        args.serial_time, args.dp_time, args.naive_time, args.smart_time,
        args.trials,
    )
    print(report)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)


if __name__ == "__main__":
    main()
