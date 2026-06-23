#!/bin/bash

SEED=42
MAX_EVENTS=50000
COARSE_TRIALS=50
COARSE_EPOCHS=3
OUTPUT_DIR="outputs/50k_comparison/seed_42/serial"

mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "测试1/4: Serial基准"
echo "========================================================================"

python search.py \
    --search-mode rl \
    --execution-mode serial \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events $MAX_EVENTS \
    --seed $SEED \
    --coarse-trials $COARSE_TRIALS \
    --coarse-epochs $COARSE_EPOCHS \
    --output-dir "$OUTPUT_DIR" \
    --space rnn_only \
    --batch-mode tbatch \
    --eval-frozen false \
    2>&1 | tee "${OUTPUT_DIR}.log"
