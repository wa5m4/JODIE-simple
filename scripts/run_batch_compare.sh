#!/bin/bash
set -e
cd "$(dirname "$0")/.."

OUT="outputs/batch_vs_sequential"
REF="outputs/medium_4way_multiseed"
mkdir -p "$OUT"

for SEED in 42 43 44; do
  mkdir -p "$OUT/seed_${SEED}"

  # 批处理 Smart
  if [[ ! -f "$OUT/seed_${SEED}/smart_batch/leaderboard.csv" ]]; then
    echo "=== seed=$SEED Smart-Batch ==="
    python search.py \
      --dataset public_csv --local-data-path data/public/mooc.csv \
      --max-events 20000 --space mixed \
      --search-mode rl --execution-mode ray_pipeline \
      --trials 27 --epochs-per-trial 3 \
      --pipeline-mode smart \
      --batch-training --train-batch-size 32 \
      --pipeline-worker-gpus 1.0 \
      --gpu-list 0,1,2 --seed $SEED \
      --output-dir "$OUT/seed_${SEED}/smart_batch"
  else
    echo "=== seed=$SEED Smart-Batch 已存在，跳过 ==="
  fi

  # 批处理 Naive
  if [[ ! -f "$OUT/seed_${SEED}/naive_batch/leaderboard.csv" ]]; then
    echo "=== seed=$SEED Naive-Batch ==="
    python search.py \
      --dataset public_csv --local-data-path data/public/mooc.csv \
      --max-events 20000 --space mixed \
      --search-mode rl --execution-mode ray_pipeline \
      --trials 27 --epochs-per-trial 3 \
      --pipeline-mode naive \
      --num-pipeline-stages 3 --pipeline-stage-train-workers 1,1,1 \
      --architectures-per-step 3 \
      --partition-size 1000 \
      --batch-training --train-batch-size 32 \
      --pipeline-worker-gpus 1.0 \
      --gpu-list 0,1,2 --seed $SEED \
      --output-dir "$OUT/seed_${SEED}/naive_batch"
  else
    echo "=== seed=$SEED Naive-Batch 已存在，跳过 ==="
  fi
done

echo ""
echo "=== 生成对比报告 ==="
python3 - <<'EOF'
import csv, os

configs = [
    ("sequential_smart", "pipeline_smart",  "outputs/medium_4way_multiseed"),
    ("sequential_naive", "pipeline_naive",  "outputs/medium_4way_multiseed"),
    ("batch_smart",      "smart_batch",     "outputs/batch_vs_sequential"),
    ("batch_naive",      "naive_batch",     "outputs/batch_vs_sequential"),
]
seeds = [42, 43, 44]

print("%-22s %8s %9s %8s" % ("配置", "avg时间(s)", "avg_MRR", "seeds"))
print("-" * 52)
for label, dirname, root in configs:
    times, scores = [], []
    for seed in seeds:
        lb = os.path.join(root, f"seed_{seed}", dirname, "leaderboard.csv")
        tl = os.path.join(root, f"seed_{seed}", dirname, "timing_log.csv")
        if not os.path.exists(lb):
            continue
        lb_rows = list(csv.DictReader(open(lb)))
        tl_rows = list(csv.DictReader(open(tl))) if os.path.exists(tl) else []
        best = max((float(r["score"]) for r in lb_rows if float(r.get("score", 0)) > 0), default=0)
        t = max(float(r["end_time_s"]) for r in tl_rows) if tl_rows else 0
        scores.append(best)
        times.append(t)
    if scores:
        print("%-22s %8.0f %9.4f %8d" % (
            label, sum(times)/len(times), sum(scores)/len(scores), len(scores)))
    else:
        print("%-22s %8s" % (label, "N/A"))
EOF
