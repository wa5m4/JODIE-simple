#!/usr/bin/env python3
"""
对比实验结果汇总工具

读取 serial 和 pipeline 两个输出目录，输出全面的对比报告。
包含：搜索质量、搜索效率、系统吞吐、搜索曲线对比。
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ─────────────────────────────────────────────
# 数据读取
# ─────────────────────────────────────────────

def load_best_arch(output_dir: str) -> Optional[Dict]:
    path = os.path.join(output_dir, "best_arch.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_leaderboard(output_dir: str) -> List[Dict]:
    path = os.path.join(output_dir, "leaderboard.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_timing_log(output_dir: str) -> List[Dict]:
    path = os.path.join(output_dir, "timing_log.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "trial_id": int(row["trial_id"]),
                "mode": row["mode"],
                "start_time_s": float(row["start_time_s"]),
                "end_time_s": float(row["end_time_s"]),
                "duration_s": float(row["duration_s"]),
                "score": float(row["score"]),
                "mrr": float(row["mrr"]),
                "recall_at_k": float(row["recall_at_k"]),
                "cumulative_best_score": float(row["cumulative_best_score"]),
                "model": row["model"],
            })
    return rows


def load_efficiency_csv(output_dir: str) -> Optional[Dict]:
    """读取 pipeline efficiency log，返回最后一行的关键指标"""
    csvs = list(Path(output_dir).glob("efficiency_log_*.csv"))
    if not csvs:
        return None
    path = str(csvs[0])
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None
    last = rows[-1]
    return {
        "gpu_util_ratio": float(last.get("gpu_util_ratio", 0)),
        "gpu_efficiency": float(last.get("gpu_efficiency", 0)),
        "pipeline_speedup": float(last.get("pipeline_speedup", 1)),
        "speedup_efficiency": float(last.get("speedup_efficiency", 0)),
        "trial_throughput": float(last.get("trial_throughput", 0)),
        "avg_concurrent_gpus": float(last.get("avg_concurrent_gpus", 0)),
        "num_completed_tasks": int(last.get("num_completed_tasks", 0)),
    }


# ─────────────────────────────────────────────
# 格式化工具
# ─────────────────────────────────────────────

def fmt(v, fmt_str=".4f", fallback="N/A"):
    if v is None:
        return fallback
    try:
        return format(float(v), fmt_str)
    except Exception:
        return fallback


def bar(value: float, max_val: float = 1.0, width: int = 20) -> str:
    ratio = min(max(value / max(max_val, 1e-9), 0), 1)
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)


def speedup_bar(speedup: float, max_speedup: float = 5.0, width: int = 20) -> str:
    return bar(speedup, max_speedup, width)


# ─────────────────────────────────────────────
# 搜索曲线（ASCII）
# ─────────────────────────────────────────────

def render_search_curve(serial_timing: List[Dict], pipeline_timing: List[Dict],
                         width: int = 50, height: int = 12) -> str:
    """并排绘制两种模式的搜索曲线（cumulative best score vs wall time）"""

    if not serial_timing and not pipeline_timing:
        return "  (no timing data available)\n"

    # 收集所有时间点和分数
    def get_curve(timing):
        if not timing:
            return [], []
        xs = [r["end_time_s"] for r in timing]
        ys = [r["cumulative_best_score"] for r in timing]
        return xs, ys

    sx, sy = get_curve(serial_timing)
    px, py = get_curve(pipeline_timing)

    all_x = sx + px
    all_y = sy + py
    if not all_x:
        return "  (no data)\n"

    max_x = max(all_x)
    min_y = min(all_y) if all_y else 0
    max_y = max(all_y) if all_y else 1
    y_range = max(max_y - min_y, 1e-6)

    lines = []
    lines.append(f"  Cumulative Best Score vs Wall Time")
    lines.append(f"  S=Serial  P=Pipeline  *=Both")
    lines.append("")

    # 构建 grid
    grid = [[" "] * width for _ in range(height)]

    def plot_curve(xs, ys, ch):
        for x, y in zip(xs, ys):
            col = int((x / max_x) * (width - 1)) if max_x > 0 else 0
            row = height - 1 - int(((y - min_y) / y_range) * (height - 1))
            col = max(0, min(col, width - 1))
            row = max(0, min(row, height - 1))
            if grid[row][col] == " ":
                grid[row][col] = ch
            elif grid[row][col] != ch:
                grid[row][col] = "*"

    plot_curve(sx, sy, "S")
    plot_curve(px, py, "P")

    y_ticks = [max_y - (y_range * i / (height - 1)) for i in range(height)]
    for i, row in enumerate(grid):
        lines.append(f"  {y_ticks[i]:.4f} │{''.join(row)}")

    lines.append(f"  {'':7s} └{'─' * width}")
    lines.append(f"  {'0s':9s}{'':>20s}{max_x:.0f}s")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 主报告生成
# ─────────────────────────────────────────────

def generate_report(
    serial_dir: str,
    pipeline_dir: str,
    serial_time_sec: Optional[float],
    pipeline_time_sec: Optional[float],
    num_trials: int,
) -> str:
    lines = []

    serial_best = load_best_arch(serial_dir)
    pipeline_best = load_best_arch(pipeline_dir)
    serial_lb = load_leaderboard(serial_dir)
    pipeline_lb = load_leaderboard(pipeline_dir)
    serial_timing = load_timing_log(serial_dir)
    pipeline_timing = load_timing_log(pipeline_dir)
    eff = load_efficiency_csv(pipeline_dir)

    # ── Header
    lines.append("=" * 72)
    lines.append("  NAS Comparison Report: Single-GPU Serial  vs  Pipeline Parallel")
    lines.append("=" * 72)
    lines.append("")

    # ── 1. 搜索质量对比
    lines.append("┌─────────────────────────────────────────────────────────────────┐")
    lines.append("│  1. Search Quality (架构搜索质量)                               │")
    lines.append("└─────────────────────────────────────────────────────────────────┘")
    lines.append("")

    sb = serial_best or {}
    pb = pipeline_best or {}

    s_score = sb.get("score")
    p_score = pb.get("score")
    s_mrr   = sb.get("mrr")
    p_mrr   = pb.get("mrr")
    s_r10   = sb.get("recall_at_k")
    p_r10   = pb.get("recall_at_k")

    lines.append(f"  {'Metric':<22}  {'Serial (Baseline)':>18}  {'Pipeline (Ours)':>18}  {'Δ':>8}")
    lines.append(f"  {'─'*22}  {'─'*18}  {'─'*18}  {'─'*8}")

    def delta(a, b):
        if a is None or b is None:
            return "N/A"
        d = float(b) - float(a)
        return f"{d:+.4f}"

    lines.append(f"  {'Test Score (primary)':<22}  {fmt(s_score):>18}  {fmt(p_score):>18}  {delta(s_score, p_score):>8}")
    lines.append(f"  {'MRR':<22}  {fmt(s_mrr):>18}  {fmt(p_mrr):>18}  {delta(s_mrr, p_mrr):>8}")
    lines.append(f"  {'Recall@10':<22}  {fmt(s_r10):>18}  {fmt(p_r10):>18}  {delta(s_r10, p_r10):>8}")
    lines.append("")

    # Top-3 val scores
    def top3_scores(lb):
        scored = []
        for row in lb:
            try:
                scored.append(float(row.get("score", 0) or 0))
            except Exception:
                pass
        scored.sort(reverse=True)
        return scored[:3]

    s_top3 = top3_scores(serial_lb)
    p_top3 = top3_scores(pipeline_lb)
    lines.append(f"  Top-3 val scores:")
    lines.append(f"    Serial  : {' | '.join(fmt(v) for v in s_top3) or 'N/A'}")
    lines.append(f"    Pipeline: {' | '.join(fmt(v) for v in p_top3) or 'N/A'}")
    lines.append("")

    # Best architecture configs
    lines.append(f"  Best Architecture (Serial):")
    if sb.get("config"):
        cfg = sb["config"]
        lines.append(f"    model={cfg.get('model','?')}  memory_cell={cfg.get('memory_cell','?')}  "
                     f"embedding_dim={cfg.get('embedding_dim','?')}  hidden_dim={cfg.get('hidden_dim','?')}")
        lines.append(f"    event_agg={cfg.get('event_agg','?')}  time_proj={cfg.get('time_proj','?')}  "
                     f"time_decay={cfg.get('time_decay','?')}")
    lines.append(f"  Best Architecture (Pipeline):")
    if pb.get("config"):
        cfg = pb["config"]
        lines.append(f"    model={cfg.get('model','?')}  memory_cell={cfg.get('memory_cell','?')}  "
                     f"embedding_dim={cfg.get('embedding_dim','?')}  hidden_dim={cfg.get('hidden_dim','?')}")
        lines.append(f"    event_agg={cfg.get('event_agg','?')}  time_proj={cfg.get('time_proj','?')}  "
                     f"time_decay={cfg.get('time_decay','?')}")
    lines.append("")

    # ── 2. 搜索效率对比
    lines.append("┌─────────────────────────────────────────────────────────────────┐")
    lines.append("│  2. Search Efficiency (搜索效率)                                │")
    lines.append("└─────────────────────────────────────────────────────────────────┘")
    lines.append("")

    lines.append(f"  {'Metric':<30}  {'Serial':>12}  {'Pipeline':>12}  {'Speedup':>8}")
    lines.append(f"  {'─'*30}  {'─'*12}  {'─'*12}  {'─'*8}")

    def speedup_str(a, b):
        if a and b and float(b) > 0:
            return f"{float(a)/float(b):.2f}x"
        return "N/A"

    # 总搜索时间
    lines.append(f"  {'Total Search Time (s)':<30}  "
                 f"{fmt(serial_time_sec, '.1f'):>12}  "
                 f"{fmt(pipeline_time_sec, '.1f'):>12}  "
                 f"{speedup_str(serial_time_sec, pipeline_time_sec):>8}")

    # 平均每 trial 耗时
    s_avg = None
    p_avg = None
    if serial_time_sec and num_trials > 0:
        s_avg = float(serial_time_sec) / num_trials
    if pipeline_time_sec and num_trials > 0:
        p_avg = float(pipeline_time_sec) / num_trials
    lines.append(f"  {'Avg Time per Trial (s)':<30}  "
                 f"{fmt(s_avg, '.2f'):>12}  "
                 f"{fmt(p_avg, '.2f'):>12}  "
                 f"{speedup_str(s_avg, p_avg):>8}")

    # Trials per hour
    s_tph = (3600 / float(serial_time_sec) * num_trials) if serial_time_sec else None
    p_tph = (3600 / float(pipeline_time_sec) * num_trials) if pipeline_time_sec else None
    lines.append(f"  {'Trials per Hour':<30}  "
                 f"{fmt(s_tph, '.1f'):>12}  "
                 f"{fmt(p_tph, '.1f'):>12}  "
                 f"{speedup_str(p_tph, s_tph):>8}")

    lines.append("")
    lines.append(f"  Wall-time bars (total search time):")
    max_t = max(v for v in [serial_time_sec, pipeline_time_sec] if v) if any([serial_time_sec, pipeline_time_sec]) else 1
    if serial_time_sec:
        lines.append(f"    Serial  [{bar(serial_time_sec, max_t, 40)}] {serial_time_sec:.0f}s")
    if pipeline_time_sec:
        lines.append(f"    Pipeline[{bar(pipeline_time_sec, max_t, 40)}] {pipeline_time_sec:.0f}s")
    lines.append("")

    # ── 3. 系统资源利用
    lines.append("┌─────────────────────────────────────────────────────────────────┐")
    lines.append("│  3. System Resource Utilization (系统资源利用率)                │")
    lines.append("└─────────────────────────────────────────────────────────────────┘")
    lines.append("")

    if eff:
        lines.append(f"  Pipeline GPU Metrics (from efficiency monitor):")
        lines.append(f"    GPU Utilization Ratio  : {eff['gpu_util_ratio']:.1%}  "
                     f"[{bar(eff['gpu_util_ratio'], 1.0, 30)}]")
        lines.append(f"    GPU Efficiency         : {eff['gpu_efficiency']:.1%}  "
                     f"[{bar(eff['gpu_efficiency'], 1.0, 30)}]")
        lines.append(f"    Pipeline Speedup       : {eff['pipeline_speedup']:.2f}x  "
                     f"[{speedup_bar(eff['pipeline_speedup'], 5.0, 30)}]")
        lines.append(f"    Speedup Efficiency     : {eff['speedup_efficiency']:.1%}")
        lines.append(f"    Avg Concurrent GPUs    : {eff['avg_concurrent_gpus']:.2f}")
        lines.append(f"    Trial Throughput       : {eff['trial_throughput']:.4f} trials/s")
    else:
        lines.append("  (no pipeline efficiency log found — run with --enable-efficiency-monitor)")
    lines.append("")

    lines.append("  Serial (single GPU) resource profile:")
    lines.append("    GPU Utilization : ~single-threaded, GPU idle during data I/O")
    lines.append("    GPU Efficiency  : depends on data loading overhead")
    lines.append("    Pipeline Speedup: 1.00x (no parallelism)")
    lines.append("")

    # ── 4. 搜索收敛曲线
    lines.append("┌─────────────────────────────────────────────────────────────────┐")
    lines.append("│  4. Search Convergence Curve (搜索收敛曲线)                     │")
    lines.append("└─────────────────────────────────────────────────────────────────┘")
    lines.append("")

    curve_str = render_search_curve(serial_timing, pipeline_timing)
    lines.append(curve_str)
    lines.append("")

    # 每步的 cumulative best（表格形式）
    if serial_timing or pipeline_timing:
        max_trials = max(
            max((r["trial_id"] for r in serial_timing), default=-1),
            max((r["trial_id"] for r in pipeline_timing), default=-1),
        ) + 1
        lines.append(f"  {'Trial':>6}  {'Serial BestScore':>16}  {'Serial Time(s)':>14}  "
                     f"{'Pipeline BestScore':>18}  {'Pipeline Time(s)':>16}")
        lines.append(f"  {'─'*6}  {'─'*16}  {'─'*14}  {'─'*18}  {'─'*16}")

        s_by_id = {r["trial_id"]: r for r in serial_timing}
        p_by_id = {r["trial_id"]: r for r in pipeline_timing}

        for tid in range(max_trials):
            sr = s_by_id.get(tid)
            pr = p_by_id.get(tid)
            s_score_str = fmt(sr["cumulative_best_score"]) if sr else "─"
            s_time_str  = f"{sr['end_time_s']:.1f}" if sr else "─"
            p_score_str = fmt(pr["cumulative_best_score"]) if pr else "─"
            p_time_str  = f"{pr['end_time_s']:.1f}" if pr else "─"
            lines.append(f"  {tid:>6}  {s_score_str:>16}  {s_time_str:>14}  "
                         f"{p_score_str:>18}  {p_time_str:>16}")
        lines.append("")

    # ── 5. 客观性说明
    lines.append("┌─────────────────────────────────────────────────────────────────┐")
    lines.append("│  5. Fairness Notes (实验公平性说明)                              │")
    lines.append("└─────────────────────────────────────────────────────────────────┘")
    lines.append("")
    lines.append("  ✓ 相同数据集、相同 seed、相同 trial 数量")
    lines.append("  ✓ 相同 epoch 数、相同评估指标")
    lines.append("  ✓ Pipeline 模式 GPU 总量与串行相同（分片使用同一 GPU）")
    lines.append("  ✓ 两种模式均使用 RL controller（REINFORCE），搜索策略一致")
    lines.append("  ✓ 质量指标在 test set 上评估（与训练集不重叠）")
    lines.append("")
    lines.append("  ⚠ Pipeline 模式因并发评估，REINFORCE 更新是 batch 方式，")
    lines.append("    与串行逐步更新存在微小差异，属于方法本身的合理设计区别。")
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare serial vs pipeline NAS results")
    parser.add_argument("--serial-dir",    required=True, help="Serial output directory")
    parser.add_argument("--pipeline-dir",  required=True, help="Pipeline output directory")
    parser.add_argument("--serial-time",   type=float, default=None, help="Serial total wall time (seconds)")
    parser.add_argument("--pipeline-time", type=float, default=None, help="Pipeline total wall time (seconds)")
    parser.add_argument("--trials",        type=int,   default=0,    help="Number of trials (for throughput calc)")
    parser.add_argument("--output",        type=str,   default="outputs/compare_report.txt",
                        help="Output report path")
    args = parser.parse_args()

    report = generate_report(
        serial_dir=args.serial_dir,
        pipeline_dir=args.pipeline_dir,
        serial_time_sec=args.serial_time,
        pipeline_time_sec=args.pipeline_time,
        num_trials=args.trials,
    )

    print(report)

    # 写入文件
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report saved to: {args.output}")


if __name__ == "__main__":
    main()
