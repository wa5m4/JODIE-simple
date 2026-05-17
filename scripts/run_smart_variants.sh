#!/bin/bash
set -e
cd "$(dirname "$0")/.."

for SEED in 42 43 44; do
  echo "=== seed=$SEED ==="

  python search.py \
    --dataset public_csv --local-data-path data/public/mooc.csv \
    --max-events 20000 --space mixed \
    --search-mode rl --execution-mode ray_pipeline \
    --trials 27 --epochs-per-trial 3 \
    --num-pipeline-stages 1 --pipeline-stage-train-workers 3 \
    --architectures-per-step 9 --pipeline-worker-gpus 1.0 \
    --stage-balance-strategy cost --pipeline-mode naive \
    --gpu-list 0,1,2 --seed $SEED \
    --output-dir outputs/medium_4way_multiseed/seed_${SEED}/pipeline_sync_1stage_3w

  python search.py \
    --dataset public_csv --local-data-path data/public/mooc.csv \
    --max-events 20000 --space mixed \
    --search-mode rl --execution-mode ray_pipeline \
    --trials 27 --epochs-per-trial 3 \
    --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1 \
    --architectures-per-step 9 --pipeline-worker-gpus 1.0 \
    --stage-balance-strategy cost --pipeline-mode naive \
    --gpu-list 0,1,2 --seed $SEED \
    --output-dir outputs/medium_4way_multiseed/seed_${SEED}/pipeline_sync_2stage_2_1

  python search.py \
    --dataset public_csv --local-data-path data/public/mooc.csv \
    --max-events 20000 --space mixed \
    --search-mode rl --execution-mode ray_pipeline \
    --trials 27 --epochs-per-trial 3 \
    --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1 \
    --pipeline-worker-gpus 1.0 \
    --stage-balance-strategy cost --pipeline-mode smart \
    --gpu-list 0,1,2 --seed $SEED \
    --output-dir outputs/medium_4way_multiseed/seed_${SEED}/pipeline_2stage_2_1

  python search.py \
    --dataset public_csv --local-data-path data/public/mooc.csv \
    --max-events 20000 --space mixed \
    --search-mode rl --execution-mode ray_pipeline \
    --trials 27 --epochs-per-trial 3 \
    --num-pipeline-stages 2 --pipeline-stage-train-workers 1,2 \
    --pipeline-worker-gpus 1.0 \
    --stage-balance-strategy cost --pipeline-mode smart \
    --gpu-list 0,1,2 --seed $SEED \
    --output-dir outputs/medium_4way_multiseed/seed_${SEED}/pipeline_2stage_1_2

  echo "=== seed=$SEED done ==="
done
