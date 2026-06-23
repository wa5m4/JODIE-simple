#!/bin/bash

# Pipeline NAS架构选择准确性综合实验
# 实验设计：系统验证Stage划分、Overlap、搜索模式对架构选择的影响

set -e

DATASET="public_csv"
MAX_EVENTS=50000
SEED=42
TRIALS=50
SPACE="rnn_only"
PARTITION_SIZE=12500
BASE_OUTPUT="outputs/comprehensive_experiment"

echo "=== Pipeline NAS 综合实验 ==="
echo "开始时间: $(date)"
echo "输出目录: $BASE_OUTPUT"
echo ""

# 创建输出目录
mkdir -p "$BASE_OUTPUT"

# =============================================================================
# 第一部分：基准组 (2个实验)
# =============================================================================
echo "【第一部分：基准组】"
echo ""

# B1: Serial baseline
echo "[B1] Serial baseline..."
python search.py \
  --dataset "$DATASET" \
  --max-events $MAX_EVENTS \
  --seed $SEED \
  --space "$SPACE" \
  --coarse-trials $TRIALS \
  --coarse-epochs 1 \
  --execution-mode serial \
  --search-mode rl \
  --output-dir "$BASE_OUTPUT/B1_serial_baseline" \
  > "$BASE_OUTPUT/B1_serial_baseline.log" 2>&1 &
PID_B1=$!

# B2: Data parallel
echo "[B2] Data parallel..."
python search.py \
  --dataset "$DATASET" \
  --max-events $MAX_EVENTS \
  --seed $SEED \
  --space "$SPACE" \
  --coarse-trials $TRIALS \
  --coarse-epochs 1 \
  --execution-mode data_parallel \
  --search-mode rl \
  --data-parallel-workers 3 \
  --output-dir "$BASE_OUTPUT/B2_data_parallel" \
  > "$BASE_OUTPUT/B2_data_parallel.log" 2>&1 &
PID_B2=$!

echo "基准组实验已启动 (后台运行)"
echo ""

# =============================================================================
# 第二部分：Stage数量实验 (4个实验)
# =============================================================================
echo "【第二部分：Stage数量实验】(固定Overlap=20%)"
echo ""

OVERLAP=0.2

for STAGES in 1 2 3 4; do
  EXP_ID="S${STAGES}"
  echo "[${EXP_ID}] Pipeline ${STAGES}-stage(s) + ${OVERLAP} overlap..."

  python search.py \
    --dataset "$DATASET" \
    --max-events $MAX_EVENTS \
    --seed $SEED \
    --space "$SPACE" \
    --coarse-trials $TRIALS \
    --coarse-epochs 1 \
    --execution-mode ray_pipeline \
    --search-mode rl \
    --num-pipeline-stages $STAGES \
    --partition-size $PARTITION_SIZE \
    --partition-overlap-ratio $OVERLAP \
    --pipeline-mode smart \
    --architectures-per-step 2 \
    --output-dir "$BASE_OUTPUT/${EXP_ID}_pipeline_${STAGES}stage_overlap${OVERLAP}" \
    > "$BASE_OUTPUT/${EXP_ID}_pipeline_${STAGES}stage.log" 2>&1 &

  eval "PID_${EXP_ID}=$!"
done

echo "Stage数量实验已启动 (后台运行)"
echo ""

# =============================================================================
# 第三部分：Overlap比例实验 (3个实验)
# =============================================================================
echo "【第三部分：Overlap比例实验】(固定Stage=2)"
echo ""

STAGES=2

for OVERLAP in 0.0 0.1 0.2; do
  OVERLAP_PCT=$(echo "$OVERLAP * 100" | bc | cut -d. -f1)
  EXP_ID="O$((OVERLAP_PCT / 10 + 1))"

  echo "[${EXP_ID}] Pipeline 2-stages + ${OVERLAP} overlap..."

  python search.py \
    --dataset "$DATASET" \
    --max-events $MAX_EVENTS \
    --seed $SEED \
    --space "$SPACE" \
    --coarse-trials $TRIALS \
    --coarse-epochs 1 \
    --execution-mode ray_pipeline \
    --search-mode rl \
    --num-pipeline-stages $STAGES \
    --partition-size $PARTITION_SIZE \
    --partition-overlap-ratio $OVERLAP \
    --pipeline-mode smart \
    --architectures-per-step 2 \
    --output-dir "$BASE_OUTPUT/${EXP_ID}_pipeline_2stage_overlap${OVERLAP}" \
    > "$BASE_OUTPUT/${EXP_ID}_overlap${OVERLAP_PCT}pct.log" 2>&1 &

  eval "PID_${EXP_ID}=$!"
done

echo "Overlap比例实验已启动 (后台运行)"
echo ""

# =============================================================================
# 第四部分：Stage×Overlap交叉验证 (4个实验)
# =============================================================================
echo "【第四部分：Stage×Overlap交叉验证】"
echo ""

# C1: 1-stage + 0% overlap
echo "[C1] 1-stage + 0% overlap..."
python search.py \
  --dataset "$DATASET" \
  --max-events $MAX_EVENTS \
  --seed $SEED \
  --space "$SPACE" \
  --coarse-trials $TRIALS \
  --coarse-epochs 1 \
  --execution-mode ray_pipeline \
  --search-mode rl \
  --num-pipeline-stages 1 \
  --partition-size $PARTITION_SIZE \
  --partition-overlap-ratio 0.0 \
  --pipeline-mode smart \
  --architectures-per-step 2 \
  --output-dir "$BASE_OUTPUT/C1_1stage_no_overlap" \
  > "$BASE_OUTPUT/C1_1stage_no_overlap.log" 2>&1 &
PID_C1=$!

# C2: 3-stages + 0% overlap (已有数据，但重新验证)
echo "[C2] 3-stages + 0% overlap..."
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
  --partition-overlap-ratio 0.0 \
  --pipeline-mode smart \
  --architectures-per-step 2 \
  --output-dir "$BASE_OUTPUT/C2_3stages_no_overlap" \
  > "$BASE_OUTPUT/C2_3stages_no_overlap.log" 2>&1 &
PID_C2=$!

# C3: 3-stages + 20% overlap
echo "[C3] 3-stages + 20% overlap..."
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
  --partition-overlap-ratio 0.2 \
  --pipeline-mode smart \
  --architectures-per-step 2 \
  --output-dir "$BASE_OUTPUT/C3_3stages_overlap20" \
  > "$BASE_OUTPUT/C3_3stages_overlap20.log" 2>&1 &
PID_C3=$!

# C4: 4-stages + 10% overlap
echo "[C4] 4-stages + 10% overlap..."
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
  --partition-overlap-ratio 0.1 \
  --pipeline-mode smart \
  --architectures-per-step 2 \
  --output-dir "$BASE_OUTPUT/C4_4stages_overlap10" \
  > "$BASE_OUTPUT/C4_4stages_overlap10.log" 2>&1 &
PID_C4=$!

echo "交叉验证实验已启动 (后台运行)"
echo ""

# =============================================================================
# 第五部分：搜索模式对比 (2个实验)
# =============================================================================
echo "【第五部分：搜索模式对比】"
echo ""

# M1: Random搜索
echo "[M1] 2-stages + 20% overlap + Random搜索..."
python search.py \
  --dataset "$DATASET" \
  --max-events $MAX_EVENTS \
  --seed $SEED \
  --space "$SPACE" \
  --coarse-trials $TRIALS \
  --coarse-epochs 1 \
  --execution-mode ray_pipeline \
  --search-mode random \
  --num-pipeline-stages 2 \
  --partition-size $PARTITION_SIZE \
  --partition-overlap-ratio 0.2 \
  --pipeline-mode smart \
  --architectures-per-step 2 \
  --output-dir "$BASE_OUTPUT/M1_2stage_random" \
  > "$BASE_OUTPUT/M1_2stage_random.log" 2>&1 &
PID_M1=$!

# M2: RL搜索 (与S2重复，作为对照)
echo "[M2] 2-stages + 20% overlap + RL搜索 (参考S2结果)"
echo ""

# =============================================================================
# 等待所有实验完成
# =============================================================================
echo "=========================================="
echo "所有实验已启动 (共18个实验 + S2/M2共用)"
echo ""
echo "等待实验完成..."
echo "可以使用 'tail -f $BASE_OUTPUT/*.log' 查看进度"
echo ""

# 等待所有后台进程
wait

echo ""
echo "=========================================="
echo "所有实验已完成!"
echo "结束时间: $(date)"
echo ""
echo "结果分析脚本: python analyze_comprehensive_experiments.py"
