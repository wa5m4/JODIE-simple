"""把某个 (cell, strategy) 的配置写入 run_all.py 顶部配置区(指南 §2、§3)。

用法(在仓库根目录):
  python positioning/apply_config.py --cell D --strategy smart            # dry-run:打印将要改的行,不写盘
  python positioning/apply_config.py --cell D --strategy smart --apply    # 真正写入 run_all.py
  python positioning/apply_config.py --cell D --strategy smart --check    # 只打印该 run 的预检摘要

每次恰好修改 run_all.py 配置区中对应的赋值行;若某参数匹配不到唯一一行则报错拒绝写入。
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import REPO_ROOT, RUNS, STRATEGY_KEY, FIXED, get_config  # noqa: E402

RUN_ALL = REPO_ROOT / "run_all.py"

# 行级替换规则:(参数名 → 匹配该行赋值前缀的正则,re.MULTILINE)。
# 只替换值部分,保留行尾注释。
_INT_PATTERNS = {
    "MAX_EVENTS": r"(^MAX_EVENTS\s*=\s*)\d+",
    "COARSE_TRIALS": r"(^COARSE_TRIALS\s*=\s*)\d+",
    "RERANK_TOP_K": r"(^RERANK_TOP_K\s*=\s*)\d+",
    "DATA_PARALLEL_WORKERS": r"(^DATA_PARALLEL_WORKERS\s*=\s*)\d+",
    "NUM_PIPELINE_STAGES": r"(^NUM_PIPELINE_STAGES\s*=\s*)\d+",
    "SMART_NUM_PIPELINE_STAGES": r"(^SMART_NUM_PIPELINE_STAGES\s*=\s*)\d+",
}
_STR_PATTERNS = {
    "GPU_LIST": r'(^GPU_LIST\s*=\s*)"[0-9,\s]*"',
    "SEARCH_SPACE": r'(^SEARCH_SPACE\s*=\s*)"[a-z_]*"',
    "PIPELINE_STAGE_TRAIN_WORKERS": r'(^PIPELINE_STAGE_TRAIN_WORKERS\s*=\s*)"[0-9,\s]*"',
    "PIPELINE_STAGE_EVAL_WORKERS": r'(^PIPELINE_STAGE_EVAL_WORKERS\s*=\s*)"[0-9,\s]*"',
    "SMART_PIPELINE_STAGE_TRAIN_WORKERS": r'(^SMART_PIPELINE_STAGE_TRAIN_WORKERS\s*=\s*)"[0-9,\s]*"',
    "SMART_PIPELINE_MODE": r'(^SMART_PIPELINE_MODE\s*=\s*)"[a-z]*"',
    "NAIVE_PIPELINE_MODE": r'(^NAIVE_PIPELINE_MODE\s*=\s*)"[a-z]*"',
}
# ENABLE_STRATEGIES 是多行列表,整体替换(从赋值到首个独占行 "]")
_ENABLE_RE = re.compile(r"^ENABLE_STRATEGIES\s*=\s*\[.*?^\]", re.M | re.S)


def _fmt(v) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return str(v)
    return str(v)


def apply_to_text(text: str, cfg: dict, strategy: str):
    """返回 (新文本, 变更记录列表)。任何参数匹配数 != 1 时抛 AssertionError。

    strategy 是简名(smart_sync/naive_alloc 与 smart/naive 共用内部策略名,
    不能用内部名反查简名,由调用方直接传入)。
    """
    new_text = text
    changes = []

    for name, pat in _INT_PATTERNS.items():
        if name not in cfg:
            continue
        rx = re.compile(pat, re.M)
        new_text, n = rx.subn(rf"\g<1>{cfg[name]}", new_text)
        assert n == 1, f"参数 {name} 在 run_all.py 中应恰好匹配 1 处,实际 {n} 处"
        changes.append(f"{name} = {cfg[name]}")

    for name, pat in _STR_PATTERNS.items():
        if name not in cfg:
            continue
        rx = re.compile(pat, re.M)
        new_text, n = rx.subn(rf'\g<1>"{cfg[name]}"', new_text)
        assert n == 1, f"参数 {name} 在 run_all.py 中应恰好匹配 1 处,实际 {n} 处"
        changes.append(f'{name} = "{cfg[name]}"')

    key = cfg["ENABLE_STRATEGIES"][0]  # 内部名,如 pipeline_smart
    replacement = f'ENABLE_STRATEGIES = [\n    "{key}",\n]'
    new_text, n = _ENABLE_RE.subn(replacement, new_text)
    assert n == 1, f"ENABLE_STRATEGIES 应恰好匹配 1 处,实际 {n} 处"
    changes.append(f"ENABLE_STRATEGIES = [{key}]  ({strategy})")
    return new_text, changes


def print_precheck_summary(cell: str, strategy: str, cfg: dict):
    """打印指南 §3 要求的预检要素(策略名/MAX_EVENTS/SEARCH_SPACE/GPU_LIST/BATCH_MODE/seed)。"""
    print("── 预检摘要(启动后 precheck.py 会照此核对)──")
    print(f"  策略        : {STRATEGY_KEY[strategy]}  (cell {cell}-{strategy})")
    print(f"  MAX_EVENTS  : {cfg['MAX_EVENTS']}")
    print(f"  SEARCH_SPACE: {cfg['SEARCH_SPACE']}")
    print(f"  GPU_LIST    : {cfg['GPU_LIST']}")
    print(f"  BATCH_MODE  : {FIXED['BATCH_MODE']} (固定, 保真协议)")
    print(f"  SEED        : {FIXED['SEED']} (固定)")
    print(f"  COARSE_TRIALS: {cfg['COARSE_TRIALS']}   RERANK_TOP_K: {cfg['RERANK_TOP_K']}")
    if strategy in ("naive", "naive_alloc", "naive_async"):
        print(f"  Naive stages: {cfg['NUM_PIPELINE_STAGES']} × "
              f"train[{cfg['PIPELINE_STAGE_TRAIN_WORKERS']}] eval[{cfg['PIPELINE_STAGE_EVAL_WORKERS']}]")
        if strategy == "naive_async":
            print(f"  驱动        : NAIVE_PIPELINE_MODE={cfg['NAIVE_PIPELINE_MODE']} (异步池, 多阶段流水线)")
    elif strategy in ("smart", "smart_sync"):
        print(f"  Smart stages: {cfg['SMART_NUM_PIPELINE_STAGES']} × "
              f"train[{cfg['SMART_PIPELINE_STAGE_TRAIN_WORKERS']}] (自动分配=OFF)")
        if strategy == "smart_sync":
            print(f"  驱动        : SMART_PIPELINE_MODE={cfg['SMART_PIPELINE_MODE']} (批同步, 去异步)")
    if strategy == "dp":
        print(f"  DP workers   : {cfg['DATA_PARALLEL_WORKERS']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["D", "B", "E", "F", "C", "S", "G", "H"])
    ap.add_argument("--strategy", required=True,
                    choices=["smart", "smart_sync", "naive", "naive_alloc", "naive_async", "dp", "serial"])
    ap.add_argument("--apply", action="store_true", help="真正写入 run_all.py(默认只 dry-run)")
    ap.add_argument("--gpu-list", default=None,
                    help="覆盖配置中的 GPU_LIST(动态选卡时由 chain_all.sh 传入实际卡号)")
    args = ap.parse_args()

    cfg = get_config(args.cell, args.strategy)
    if args.gpu_list:
        cfg["GPU_LIST"] = args.gpu_list
    elif cfg.get("GPU_LIST") == "auto":
        print("✗ GPU_LIST=auto 需要显式指定 --gpu-list(chain_all.sh 会自动传入)")
        sys.exit(1)
    text = RUN_ALL.read_text(encoding="utf-8")
    new_text, changes = apply_to_text(text, cfg, args.strategy)

    print(f"=== cell {args.cell} / {args.strategy} (内部名 {STRATEGY_KEY[args.strategy]}) ===")
    if args.apply:
        RUN_ALL.write_text(new_text, encoding="utf-8")
        print("已写入 run_all.py,变更如下:")
    else:
        print("DRY-RUN(未写盘),将发生如下变更:")
    for c in changes:
        print(f"  · {c}")
    print_precheck_summary(args.cell, args.strategy, cfg)
    print()


if __name__ == "__main__":
    main()
