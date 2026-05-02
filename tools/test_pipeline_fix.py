#!/usr/bin/env python3
"""
验证 pipeline epoch 修复是否生效。

原理：random search 固定同一 seed，serial 和 pipeline 会采样到完全相同的
架构序列。对比同一架构在两种训练方式下的得分，直接验证训练质量是否等价。
"""
import subprocess, json, sys, os, tempfile, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_EVENTS     = 10000
EPOCHS         = 3
SEED           = 43
TRIALS         = 16     # 增加 trial 确保覆盖 on/off
PARTITION_SIZE = 300    # 10000/300 ≈ 33 partitions → 3 stages 各约 11 片
NUM_STAGES     = 3
WORKER_GPUS    = 0.33
DATA_FILE      = "data/public/mooc.csv"

OUTDIR = tempfile.mkdtemp(prefix="pipeline_fix_")
print(f"\n{'='*64}")
print(f"Pipeline fix validation")
print(f"events={MAX_EVENTS}  epochs={EPOCHS}  seed={SEED}  trials={TRIALS}")
print(f"Output: {OUTDIR}")
print('='*64)


def run_search(mode, out_subdir, extra_args=None):
    out = os.path.join(OUTDIR, out_subdir)
    os.makedirs(out, exist_ok=True)
    cmd = [
        sys.executable, "search.py",
        "--dataset",          "public_csv",
        "--local-data-path",  DATA_FILE,
        "--max-events",       str(MAX_EVENTS),
        "--search-mode",      "random",     # 固定 seed → 确定性架构序列
        "--execution-mode",   mode,
        "--trials",           str(TRIALS),
        "--epochs-per-trial", str(EPOCHS),
        "--seed",             str(SEED),
        "--k",                "10",
        "--selection-metric", "mrr",
        "--output-dir",       out,
    ]
    if extra_args:
        cmd += extra_args
    print(f"\n▶  Running {mode.upper()} ({TRIALS} trials, {EPOCHS} epochs) ...")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR]\n{result.stderr[-3000:]}")
        return None
    lb_path = os.path.join(out, "leaderboard.csv")
    if not os.path.exists(lb_path):
        return None
    with open(lb_path) as f:
        return list(csv.DictReader(f))


def arch_key(cfg):
    """用几个关键维度做架构指纹（seed 相同时 rank 顺序一致）。"""
    return (
        cfg.get("model", ""),
        cfg.get("memory_cell", ""),
        str(cfg.get("embedding_dim", "")),
        cfg.get("use_static_embeddings", ""),
        cfg.get("time_proj", ""),
        cfg.get("normalize_state", ""),
    )


# ── 运行两种模式
serial_lb = run_search("serial", "serial")
pipeline_lb = run_search(
    "ray_pipeline", "pipeline",
    extra_args=[
        "--architectures-per-step", "1",
        "--num-pipeline-stages",    str(NUM_STAGES),
        "--pipeline-worker-gpus",   str(WORKER_GPUS),
        "--partition-size",         str(PARTITION_SIZE),
        "--stage-balance-strategy", "cost",
    ],
)

if not serial_lb or not pipeline_lb:
    print("运行失败，请检查错误信息。")
    sys.exit(1)


# ── 对齐同一架构
def parse_lb(rows):
    result = {}
    for r in rows:
        try:
            cfg = json.loads(r.get("config_json", "{}"))
            key = arch_key(cfg)
            score = float(r.get("score", 0))
            if key not in result or score > result[key]["score"]:
                result[key] = {"score": score, "cfg": cfg}
        except Exception:
            pass
    return result

s_map = parse_lb(serial_lb)
p_map = parse_lb(pipeline_lb)
common_keys = set(s_map) & set(p_map)

print(f"\n{'='*64}")
print(f"逐架构得分对比  (共同架构 {len(common_keys)}/{TRIALS} 个)")
print('='*64)
print(f"  {'static_emb':<10} {'mem':<5} {'dim':<5}  {'Serial':>8}  {'Pipeline':>9}  {'Diff':>7}  {'OK?'}")
print(f"  {'-'*10} {'-'*5} {'-'*5}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*4}")

diffs_on, diffs_off = [], []
for key in sorted(common_keys, key=lambda k: -s_map[k]["score"]):
    cfg = s_map[key]["cfg"]
    s = s_map[key]["score"]
    p = p_map[key]["score"]
    diff = p - s
    static = cfg.get("use_static_embeddings", "?")
    mem    = cfg.get("memory_cell", "?")
    dim    = str(cfg.get("embedding_dim", "?"))
    ok = "✓" if abs(diff) < 0.08 else "△"
    print(f"  {static:<10} {mem:<5} {dim:<5}  {s:>8.4f}  {p:>9.4f}  {diff:>+7.4f}  {ok}")
    if static == "on":
        diffs_on.append(diff)
    else:
        diffs_off.append(diff)

print(f"\n  use_static_embeddings=on  architectures: n={len(diffs_on)}", end="")
if diffs_on:
    print(f"  mean_diff={sum(diffs_on)/len(diffs_on):+.4f}")
else:
    print("  (未采样到)")

print(f"  use_static_embeddings=off architectures: n={len(diffs_off)}", end="")
if diffs_off:
    print(f"  mean_diff={sum(diffs_off)/len(diffs_off):+.4f}")
else:
    print("  (未采样到)")

# ── 结论
print(f"\n{'='*64}")
if diffs_on:
    avg_on = sum(diffs_on) / len(diffs_on)
    if avg_on > -0.05:
        print("✓  修复有效：use_static_embeddings=on 在 pipeline 下得分与 serial 相当")
    else:
        print(f"✗  修复可能未完全生效：on 架构 pipeline 仍平均低 {avg_on:.4f}")
else:
    all_diffs = diffs_on + diffs_off
    avg = sum(all_diffs) / len(all_diffs) if all_diffs else 0
    if abs(avg) < 0.05:
        print(f"✓  所有架构得分差异均在 ±0.05 以内（avg diff={avg:+.4f}），训练质量等价")
    else:
        print(f"△  平均差异 {avg:+.4f}，建议用更多 trials 确认")

print(f"\n临时输出: {OUTDIR}  （可手动删除）")
