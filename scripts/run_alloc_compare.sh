#!/bin/bash
set -e
cd "$(dirname "$0")/.."

SEED=${1:-42}  # 支持命令行传入 seed，默认 42
MAX_EVENTS=20000
TRIALS=27
EPOCHS=3
GPU_LIST="0,1,2"
SPACE="mixed"
OUT="outputs/alloc_compare"

mkdir -p "$OUT/seed_${SEED}"

run() {
    local name=$1; shift
    local outdir="$OUT/seed_${SEED}/$name"
    if [[ -f "$outdir/leaderboard.csv" ]]; then
        echo "  ↪ $name 已存在，跳过"
        return
    fi
    echo "  [running] $name"
    python search.py \
        --dataset public_csv --local-data-path data/public/mooc.csv \
        --max-events $MAX_EVENTS --space $SPACE \
        --search-mode rl --execution-mode ray_pipeline \
        --trials $TRIALS --epochs-per-trial $EPOCHS \
        --pipeline-worker-gpus 1.0 \
        --stage-balance-strategy cost \
        --gpu-list $GPU_LIST --seed $SEED \
        --output-dir "$outdir" \
        "$@"
    echo "  ✅ $name done"
}

echo "=== seed=$SEED ==="

# 异步 1stage 3workers（当前 Smart 默认）
run async_1stage_3w \
    --pipeline-mode smart

# 异步 2stage 2,1
run async_2stage_2_1 \
    --pipeline-mode smart \
    --num-pipeline-stages 2 \
    --pipeline-stage-train-workers 2,1

# 异步 2stage 1,2
run async_2stage_1_2 \
    --pipeline-mode smart \
    --num-pipeline-stages 2 \
    --pipeline-stage-train-workers 1,2

# 同步 1stage 3workers
run sync_1stage_3w \
    --pipeline-mode naive \
    --num-pipeline-stages 1 \
    --pipeline-stage-train-workers 3 \
    --architectures-per-step 9

# 同步 2stage 2,1
run sync_2stage_2_1 \
    --pipeline-mode naive \
    --num-pipeline-stages 2 \
    --pipeline-stage-train-workers 2,1 \
    --architectures-per-step 9

# 同步 2stage 1,2
run sync_2stage_1_2 \
    --pipeline-mode naive \
    --num-pipeline-stages 2 \
    --pipeline-stage-train-workers 1,2 \
    --architectures-per-step 9

echo ""
echo "=== 生成对比报告 ==="
python3 - <<'EOF'
import csv, os, json

configs = [
    ("async_1stage_3w",  "异步 1stage [3]"),
    ("async_2stage_2_1", "异步 2stage [2,1]"),
    ("async_2stage_1_2", "异步 2stage [1,2]"),
    ("sync_1stage_3w",   "同步 1stage [3]"),
    ("sync_2stage_2_1",  "同步 2stage [2,1]"),
    ("sync_2stage_1_2",  "同步 2stage [1,2]"),
]
root = "outputs/alloc_compare/seed_42"

print(f"{'配置':<22} {'时间(s)':>8} {'best_MRR':>9} {'s/trial':>8} {'trials':>7}")
print("-" * 58)
for d, label in configs:
    lb_path = f"{root}/{d}/leaderboard.csv"
    tl_path = f"{root}/{d}/timing_log.csv"
    if not os.path.exists(lb_path):
        print(f"{label:<22} {'N/A':>8}")
        continue
    lb = list(csv.DictReader(open(lb_path)))
    tl = list(csv.DictReader(open(tl_path))) if os.path.exists(tl_path) else []
    best = max((float(r["score"]) for r in lb if float(r.get("score", 0)) > 0), default=0)
    t = max(float(r["end_time_s"]) for r in tl) if tl else 0
    n = len([r for r in lb if float(r.get("score", 0)) > 0])
    print(f"{label:<22} {t:>8.0f} {best:>9.4f} {t/27:>8.1f} {n:>7}")
EOF
     