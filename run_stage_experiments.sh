#!/bin/bash
# 补充缺失的Stage实验

set -e

DATASET="public_csv"
MAX_EVENTS=50000
SEED=42
TRIALS=50
SPACE="rnn_only"
PARTITION_SIZE=12500
OVERLAP=0.2
OUTPUT_BASE="outputs/stage_test"

mkdir -p "$OUTPUT_BASE"

echo "=== 补充Stage数量实验 ==="
echo ""

# S3: 3 stages + 20% overlap
if [ ! -d "$OUTPUT_BASE/S3_3stages" ]; then
  echo "[S3] 3 stages + 20% overlap..."
  python search.py \
    --dataset "$DATASET" \
    --max-events $MAX_EVENTS \
    --seed $SEED \
    --space "$SPACE" \
    --coarse-trials $TRIALS \
    --coarse-epochs 1 \
    --execution-mode ray_pipeline \
    --search-mode rl \
    --num-pipeline-stages 3 \
    --partition-size $PARTITION_SIZE \
    --partition-overlap-ratio $OVERLAP \
    --pipeline-mode smart \
    --architectures-per-step 2 \
    --output-dir "$OUTPUT_BASE/S3_3stages" \
    > "$OUTPUT_BASE/S3_3stages.log" 2>&1 &
fi

# S4: 4 stages + 20% overlap
if [ ! -d "$OUTPUT_BASE/S4_4stages" ]; then
  echo "[S4] 4 stages + 20% overlap..."
  python search.py \
    --dataset "$DATASET" \
    --max-events $MAX_EVENTS \
    --seed $SEED \
    --space "$SPACE" \
    --coarse-trials $TRIALS \
    --coarse-epochs 1 \
    --execution-mode ray_pipeline \
    --search-mode rl \
    --num-pipeline-stages 4 \
    --partition-size $PARTITION_SIZE \
    --partition-overlap-ratio $OVERLAP \
    --pipeline-mode smart \
    --architectures-per-step 2 \
    --output-dir "$OUTPUT_BASE/S4_4stages" \
    > "$OUTPUT_BASE/S4_4stages.log" 2>&1 &
fi

echo ""
echo "等待实验完成..."
wait

echo "完成! 运行分析: python analyze_stage_experiments.py"
