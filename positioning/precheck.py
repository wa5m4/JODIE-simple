"""启动后 ~90 秒的自动预检 —— 指南 §3"1 分钟内看 log"步骤的自动化。

检查两项:
 1) 最新 results/<时间戳>/config.json 配置快照与预期值逐项一致
    (策略名、MAX_EVENTS、SEARCH_SPACE、GPU_LIST、BATCH_MODE=serial、seed 42、
     COARSE_TRIALS、RERANK_TOP_K、各 pipeline/DP workers 等)
 2) log 中出现关键启动标志行(数据文件 ✓、CUDA ✓、启用策略、结果目录)

用法(在仓库根目录):
  python positioning/precheck.py --cell D --strategy smart --log run_pos_cellD_smart.log [--timeout 240]
退出码: 0 = 预检通过;1 = 失败(launch.sh / chain_all.sh 会杀掉进程并中止)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import REPO_ROOT, STRATEGY_KEY, expected_snapshot  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"


def find_latest_snapshot(timeout: int):
    """轮询等待最新 config.json 快照出现,返回 (路径, dict) 或 (None, None)。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        dirs = sorted(
            [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()],
            key=lambda d: d.name,
        )
        for d in reversed(dirs):
            snap = d / "config.json"
            if snap.exists():
                try:
                    return snap, json.loads(snap.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"  [预检] ⚠ 快照解析失败 {snap}: {e}")
                    return None, None
        time.sleep(5)
    return None, None


def check_log(log_path: Path):
    """grep 启动标志行,返回 (ok, 详情列表)。"""
    if not log_path.exists():
        return False, [f"log 不存在: {log_path}"]
    text = log_path.read_text(encoding="utf-8", errors="replace")
    marks = [
        ("数据文件", r"\[预检\] ✓ 数据文件"),
        ("CUDA", r"\[预检\] ✓ CUDA 可用"),
        ("Ray 已安装", r"\[预检\] ✓ Ray 已安装"),
        ("结果目录", r"结果目录: "),
        ("启用策略", r"启用策略: "),
    ]
    details = []
    ok = True
    for label, pat in marks:
        found = re.search(pat, text)
        details.append(f"{'✓' if found else '✗'} log: {label}")
        ok = ok and bool(found)
    return ok, details


def check_gpu_visibility(log_path: Path, gpu_list: str):
    """核对 run_all.py 预检打印的可见GPU数 == GPU_LIST 卡数。

    2026-09-07 事故:pipeline 路径不自行设置 CUDA_VISIBLE_DEVICES,
    未限定可见卡时 worker 直连物理 cuda:0/1,必须靠此项拦住。
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"可见GPU数: (\d+)", text)
    if not m:
        return False, "✗ log: 未找到「可见GPU数」行(运行可能启动失败)"
    visible = int(m.group(1))
    expect = len([x for x in gpu_list.split(",") if x.strip()])
    if visible != expect:
        return (False,
                f"✗ 可见GPU数 {visible} ≠ 期望 {expect} —— CUDA_VISIBLE_DEVICES 未生效,"
                f"worker 会落到错误物理卡上!立即终止")
    return True, f"✓ 可见GPU数 {visible} = 期望 {expect}(CUDA_VISIBLE_DEVICES 已生效)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["D", "B"])
    ap.add_argument("--strategy", required=True, choices=["smart", "naive", "dp", "serial"])
    ap.add_argument("--log", required=True)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--gpu-list", default=None,
                    help="覆盖期望的 GPU_LIST(动态选卡时由 chain_all.sh 传入实际卡号)")
    args = ap.parse_args()

    exp = expected_snapshot(args.cell, args.strategy)
    if args.gpu_list:
        exp["GPU_LIST"] = args.gpu_list
    elif exp.get("GPU_LIST") == "auto":
        print("✗ 期望 GPU_LIST=auto,预检需 --gpu-list 指定实际卡号")
        sys.exit(1)
    key = STRATEGY_KEY[args.strategy]
    print(f"=== 自动预检: cell {args.cell} / {args.strategy} (期望 {key}) ===")

    snap_path, snap = find_latest_snapshot(args.timeout)
    if snap is None:
        print("  [预检] ✗ 超时:未找到 config.json 快照(运行可能启动失败,看 log)")
        sys.exit(1)
    print(f"  快照: {snap_path}")

    fails = []
    for k, v in exp.items():
        got = snap.get(k)
        ok = got == v
        if not ok:
            fails.append(f"{k}: 期望 {v!r}, 实际 {got!r}")
        print(f"  {'✓' if ok else '✗'} {k} = {got!r}" + ("" if ok else f" (期望 {v!r})"))

    log_ok, log_details = check_log(Path(args.log))
    for d in log_details:
        print(f"  {d}")

    gpu_ok, gpu_detail = check_gpu_visibility(Path(args.log), exp["GPU_LIST"])
    print(f"  {gpu_detail}")

    if fails or not log_ok or not gpu_ok:
        print("\n  [预检] ✗ 未通过,见上。")
        if fails:
            print("  快照不一致项: " + "; ".join(fails))
        sys.exit(1)
    print("\n  [预检] ✓ 全部通过 — 配置与指南一致,运行继续。")
    sys.exit(0)


if __name__ == "__main__":
    main()
