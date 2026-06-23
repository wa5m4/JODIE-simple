#!/bin/bash
# 三因素完全交叉实验 - 后台逐个运行

set -e

DATASET="public_csv"
DATA_PATH="data/public/mooc.csv"
MAX_EVENTS=50000
SEED=42
TRIALS=50
EPOCHS=3
SPACE="rnn_only"
PARTITION_SIZE=12500
BASE="outputs/three_factor_test"

echo "=== 补充三因素实验（共8个，后台逐个运行） ==="
echo ""

# A1
echo "[1/8] A1: 1 stage + 0% (async)..."
python search.py --dataset "$DATASET" --local-data-path "$DATA_PATH" --max-events $MAX_EVENTS --seed $SEED \
  --space "$SPACE" --coarse-trials $TRIALS --coarse-epochs $EPOCHS \
  --execution-mode ray_pipeline --search-mode rl \
  --num-pipeline-stages 1 --partition-size $PARTITION_SIZE \
  --partition-overlap-ratio 0.0 --pipeline-mode smart --architectures-per-step 2 \
  --output-dir "$BASE/async/1stage_overlap0" > "$BASE/async/1stage_overlap0.log" 2>&1 &
wait $!
echo "✓"

# A3
echo "[2/8] A3: 2 stages + 0% (async)..."
python search.py --dataset "$DATASET" --local-data-path "$DATA_PATH" --max-events $MAX_EVENTS --seed $SEED \
  --space "$SPACE" --coarse-trials $TRIALS --coarse-epochs $EPOCHS \
  --execution-mode ray_pipeline --search-mode rl \
  --num-pipeline-stages 2 --partition-size $PARTITION_SIZE \
  --partition-overlap-ratio 0.0 --pipeline-mode smart --architectures-per-step 2 \
  --output-dir "$BASE/async/2stage_overlap0" > "$BASE/async/2stage_overlap0.log" 2>&1 &
wait $!
echo "✓"

# A5
echo "[3/8] A5: 3 stages + 0% (async)..."
python search.py --dataset "$DATASET" --local-data-path "$DATA_PATH" --max-events $MAX_EVENTS --seed $SEED \
  --space "$SPACE" --coarse-trials $TRIALS --coarse-epochs $EPOCHS \
  --execution-mode ray_pipeline --search-mode rl \
  --num-pipeline-stages 3 --partition-size $PARTITION_SIZE \
  --partition-overlap-ratio 0.0 --pipeline-mode smart --architectures-per-step 2 \
  --output-dir "$BASE/async/3stage_overlap0" > "$BASE/async/3stage_overlap0.log" 2>&1 &
wait $!
echo "✓"

# A6
echo "[4/8] A6: 3 stages + 20% (async)..."
python search.py --dataset "$DATASET" --local-data-path "$DATA_PATH" --max-events $MAX_EVENTS --seed $SEED \
  --space "$SPACE" --coarse-trials $TRIALS --coarse-epochs $EPOCHS \
  --execution-mode ray_pipeline --search-mode rl \
  --num-pipeline-stages 3 --partition-size $PARTITION_SIZE \
  --partition-overlap-ratio 0.2 --pipeline-mode smart --architectures-per-step 2 \
  --output-dir "$BASE/async/3stage_overlap20" > "$BASE/async/3stage_overlap20.log" 2>&1 &
wait $!
echo "✓"

# S1
echo "[5/8] S1: 1 stage + 0% (sync)..."
python search.py --dataset "$DATASET" --local-data-path "$DATA_PATH" --max-events $MAX_EVENTS --seed $SEED \
  --space "$SPACE" --coarse-trials $TRIALS --coarse-epochs $EPOCHS \
  --execution-mode ray_pipeline --search-mode rl \
  --num-pipeline-stages 1 --partition-size $PARTITION_SIZE \
  --partition-overlap-ratio 0.0 --pipeline-mode naive --architectures-per-step 2 \
  --output-dir "$BASE/sync/1stage_overlap0" > "$BASE/sync/1stage_overlap0.log" 2>&1 &
wait $!
echo "✓"

# S2
echo "[6/8] S2: 1 stage + 20% (sync)..."
python search.py --dataset "$DATASET" --local-data-path "$DATA_PATH" --max-events $MAX_EVENTS --seed $SEED \
  --space "$SPACE" --coarse-trials $TRIALS --coarse-epochs $EPOCHS \
  --execution-mode ray_pipeline --search-mode rl \
  --num-pipeline-stages 1 --partition-size $PARTITION_SIZE \
  --partition-overlap-ratio 0.2 --pipeline-mode naive --architectures-per-step 2 \
  --output-dir "$BASE/sync/1stage_overlap20" > "$BASE/sync/1stage_overlap20.log" 2>&1 &
wait $!
echo "✓"

# S4
echo "[7/8] S4: 2 stages + 20% (sync)..."
python search.py --dataset "$DATASET" --local-data-path "$DATA_PATH" --max-events $MAX_EVENTS --seed $SEED \
  --space "$SPACE" --coarse-trials $TRIALS --coarse-epochs $EPOCHS \
  --execution-mode ray_pipeline --search-mode rl \
  --num-pipeline-stages 2 --partition-size $PARTITION_SIZE \
  --partition-overlap-ratio 0.2 --pipeline-mode naive --architectures-per-step 2 \
  --output-dir "$BASE/sync/2stage_overlap20" > "$BASE/sync/2stage_overlap20.log" 2>&1 &
wait $!
echo "✓"

# S6
echo "[8/8] S6: 3 stages + 20% (sync)..."
python search.py --dataset "$DATASET" --local-data-path "$DATA_PATH" --max-events $MAX_EVENTS --seed $SEED \
  --space "$SPACE" --coarse-trials $TRIALS --coarse-epochs $EPOCHS \
  --execution-mode ray_pipeline --search-mode rl \
  --num-pipeline-stages 3 --partition-size $PARTITION_SIZE \
  --partition-overlap-ratio 0.2 --pipeline-mode naive --architectures-per-step 2 \
  --output-dir "$BASE/sync/3stage_overlap20" > "$BASE/sync/3stage_overlap20.log" 2>&1 &
wait $!
echo "✓"

echo ""
echo "所有实验完成! 运行分析: python analyze_three_factors.py"
