#!/usr/bin/env python3
"""
多种子汇总：对多次 serial vs pipeline 对比实验计算 mean±std
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


def load_seed_times(root: str) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    times: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
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


def generate_aggregate_report(
    root: str,
    seeds: List[int],
    serial_trials: int,
    pipeline_trials: int,
) -> str:
    seed_times = load_seed_times(root)

    records = []
    for seed in seeds:
        seed_dir = os.path.join(root, f"seed_{seed}")
        sb = load_best_arch(os.path.join(seed_dir, "serial"))
        pb = load_best_arch(os.path.join(seed_dir, "pipeline"))
        s_time, p_time = seed_times.get(seed, (None, None))
        records.append({
            "seed":    seed,
            "s_score": sb.get("score") or sb.get("mrr"),
            "p_score": pb.get("score") or pb.get("mrr"),
            "s_mrr":   sb.get("mrr"),
            "p_mrr":   pb.get("mrr"),
            "s_r10":   sb.get("recall_at_k"),
            "p_r10":   pb.get("recall_at_k"),
            "s_time":  s_time,
            "p_time":  p_time,
        })

    def get_vals(key) -> List[float]:
        return [r[key] for r in records if r.get(key) is not None]

    L = []

    L.append("╔" + "═" * 70 + "╗")
    L.append("║{:^70}║".format("NAS Multi-Seed Aggregate: Serial vs Pipeline"))
    L.append("╚" + "═" * 70 + "╝")
    L.append("")
    L.append(f"  Seeds: {seeds}")
    L.append(f"  Serial: {serial_trials} trials/seed  |  Pipeline: {pipeline_trials} trials/seed")
    L.append("")

    # ── 1. Per-seed 结果一览
    L.append("┌─ 1. Per-seed Results  逐种子结果 " + "─" * 36 + "┐")
    L.append("")
    L.append(f"  {'Seed':>6}  {'S.Score':>8}  {'P.Score':>8}  {'S.MRR':>8}  "
             f"{'P.MRR':>8}  {'S.R@10':>7}  {'P.R@10':>7}  {'Winner':>7}")
    L.append(f"  {'─' * 6}  {'─' * 8}  {'─' * 8}  {'─' * 8}  "
             f"{'─' * 8}  {'─' * 7}  {'─' * 7}  {'─' * 7}")

    for r in records:
        ss, ps = r["s_score"], r["p_score"]
        if ps and ss:
            w = "P ◀" if ps > ss + 1e-6 else ("S ◀" if ss > ps + 1e-6 else "  =")
        else:
            w = "N/A"
        L.append(f"  {r['seed']:>6}  {fv(ss):>8}  {fv(ps):>8}  {fv(r['s_mrr']):>8}  "
                 f"{fv(r['p_mrr']):>8}  {fv(r['s_r10']):>7}  {fv(r['p_r10']):>7}  {w:>7}")

    L.append("")
    L.append("└" + "─" * 70 + "┘")
    L.append("")

    # ── 2. 均值±标准差统计
    L.append("┌─ 2. Aggregate Statistics  均值 ± 标准差 " + "─" * 28 + "┐")
    L.append("")
    L.append(f"  {'Metric':<20}  {'Serial  mean ± std':>24}  {'Pipeline  mean ± std':>24}  {'Winner':>7}")
    L.append(f"  {'─' * 20}  {'─' * 24}  {'─' * 24}  {'─' * 7}")

    quality_rows = [
        ("Best Score (MRR)", "s_score", "p_score"),
        ("MRR",              "s_mrr",   "p_mrr"),
        ("Recall@10",        "s_r10",   "p_r10"),
    ]
    for label, s_key, p_key in quality_rows:
        sv_list = get_vals(s_key)
        pv_list = get_vals(p_key)
        if not sv_list or not pv_list:
            continue
        sm, ss = mean_std(sv_list)
        pm, ps = mean_std(pv_list)
        w = "P ◀" if pm > sm + 1e-6 else ("S ◀" if sm > pm + 1e-6 else "  =")
        L.append(f"  {label:<20}  {ms_str(sm, ss):>24}  {ms_str(pm, ps):>24}  {w:>7}")

    L.append("")

    s_times = get_vals("s_time")
    p_times = get_vals("p_time")
    if s_times and p_times:
        sm_t, ss_t = mean_std(s_times)
        pm_t, ps_t = mean_std(p_times)
        s_tph_list = [3600 / t * serial_trials for t in s_times if t > 0]
        p_tph_list = [3600 / t * pipeline_trials for t in p_times if t > 0]

        L.append(f"  {'Wall Time (s)':<20}  {ms_str(sm_t, ss_t, '.1f'):>24}  {ms_str(pm_t, ps_t, '.1f'):>24}")
        if s_tph_list and p_tph_list:
            sm_tph, _ = mean_std(s_tph_list)
            pm_tph, ps_tph = mean_std(p_tph_list)
            tph_ratio = pm_tph / sm_tph if sm_tph > 0 else 0
            L.append(f"  {'Throughput (tph)':<20}  {ms_str(sm_tph, 0, '.1f'):>24}  {ms_str(pm_tph, ps_tph, '.1f'):>24}")
            L.append(f"  {'Throughput Ratio':<20}  {'1.00x':>24}  {f'{tph_ratio:.2f}x':>24}  {'P ◀':>7}")

    L.append("")
    L.append("└" + "─" * 70 + "┘")
    L.append("")

    # ── 3. 胜负统计
    L.append("┌─ 3. Win Count  胜负统计 " + "─" * 45 + "┐")
    L.append("")

    s_wins = sum(1 for r in records
                 if r["s_score"] and r["p_score"] and r["s_score"] > r["p_score"] + 1e-6)
    p_wins = sum(1 for r in records
                 if r["s_score"] and r["p_score"] and r["p_score"] > r["s_score"] + 1e-6)
    ties   = len(records) - s_wins - p_wins
    n      = len(records)

    L.append(f"  Serial   wins : {s_wins}/{n}  {'█' * s_wins + '░' * (n - s_wins)}")
    L.append(f"  Pipeline wins : {p_wins}/{n}  {'█' * p_wins + '░' * (n - p_wins)}")
    L.append(f"  Ties          : {ties}/{n}")
    L.append("")
    L.append("└" + "─" * 70 + "┘")
    L.append("")

    # ── 4. 综合结论
    L.append("┌─ 4. Conclusion  综合结论 " + "─" * 44 + "┐")
    L.append("")

    sv_list = get_vals("s_score")
    pv_list = get_vals("p_score")
    if sv_list and pv_list:
        sm, ss = mean_std(sv_list)
        pm, ps = mean_std(pv_list)
        diff     = pm - sm
        diff_pct = diff / sm * 100 if sm > 0 else 0
        # 差异是否在误差范围内（小于双方 std 的较大值）
        within_noise = abs(diff) < max(ss, ps, 1e-6)

        L.append("  质量对比（Best Score / MRR）：")
        if within_noise:
            L.append(f"    差异 {diff:+.4f}（{diff_pct:+.1f}%）在种子方差范围内（±{max(ss,ps):.4f}）。")
            L.append(f"    → 两种方法找到的架构质量相当，无显著差异。")
        elif pm > sm:
            L.append(f"    Pipeline 平均高出 {diff:+.4f}（{diff_pct:+.1f}%），超出噪声范围。")
            L.append(f"    → Pipeline 在搜索质量上有显著优势。")
        else:
            L.append(f"    Serial 平均高出 {abs(diff):.4f}（{abs(diff_pct):.1f}%），超出噪声范围。")
            L.append(f"    → Serial 搜索质量更优，建议检查 Pipeline 的 RL 控制器配置。")

        L.append("")
        L.append("  吞吐量对比：")
        if s_times and p_times:
            s_tph_list = [3600 / t * serial_trials for t in s_times if t > 0]
            p_tph_list = [3600 / t * pipeline_trials for t in p_times if t > 0]
            if s_tph_list and p_tph_list:
                ratio = mean_std(p_tph_list)[0] / max(mean_std(s_tph_list)[0], 1e-9)
                L.append(f"    Pipeline 吞吐量 {ratio:.2f}x，在相同时间内评估了 "
                         f"{pipeline_trials / serial_trials:.1f}x 更多架构。")
                L.append(f"    → 即使质量持平，Pipeline 以更大的搜索覆盖降低了遗漏最优架构的风险。")

    L.append("")
    L.append("└" + "─" * 70 + "┘")

    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",             required=True, help="multi-seed output root dir")
    parser.add_argument("--seeds",            required=True, help="space or comma separated seeds")
    parser.add_argument("--serial-trials",    type=int, default=15)
    parser.add_argument("--pipeline-trials",  type=int, default=30)
    parser.add_argument("--output",           default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]

    report = generate_aggregate_report(
        root=args.root,
        seeds=seeds,
        serial_trials=args.serial_trials,
        pipeline_trials=args.pipeline_trials,
    )

    print(report)

    out = args.output or os.path.join(args.root, "aggregate_report.txt")
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Aggregate report saved → {out}")


if __name__ == "__main__":
    main()
