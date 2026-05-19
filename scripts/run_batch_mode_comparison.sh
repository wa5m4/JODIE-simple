#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# 比较四种批处理策略：serial, t-batch, TGN(all), TGN(last)
# 使用 smart mode pipeline, 3个GPU, 20000数据, 27 trials, 3 epochs

SEED=42
MAX_EVENTS=20000
TRIALS=27
EPOCHS=3
GPU_LIST="0,1,2"

echo "=== 批处理策略对比实验 ==="
echo "配置: ${MAX_EVENTS} events, ${TRIALS} trials, ${EPOCHS} epochs, GPUs: ${GPU_LIST}"
echo ""

# 1. Serial (逐条训练)
echo "[1/4] Running Serial mode..."
python search.py \
  --dataset public_csv --local-data-path data/public/mooc.csv \
  --max-events ${MAX_EVENTS} --space mixed \
  --search-mode rl --execution-mode ray_pipeline \
  --trials ${TRIALS} --epochs-per-trial ${EPOCHS} \
  --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1 \
  --architectures-per-step 9 --pipeline-worker-gpus 1.0 \
  --stage-balance-strategy cost --pipeline-mode smart \
  --gpu-list ${GPU_LIST} --seed ${SEED} \
  --batch-mode serial \
  --output-dir outputs/batch_mode_comparison/serial

# 2. t-Batch
echo "[2/4] Running t-Batch mode..."
python search.py \
  --dataset public_csv --local-data-path data/public/mooc.csv \
  --max-events ${MAX_EVENTS} --space mixed \
  --search-mode rl --execution-mode ray_pipeline \
  --trials ${TRIALS} --epochs-per-trial ${EPOCHS} \
  --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1 \
  --architectures-per-step 9 --pipeline-worker-gpus 1.0 \
  --stage-balance-strategy cost --pipeline-mode smart \
  --gpu-list ${GPU_LIST} --seed ${SEED} \
  --batch-mode tbatch --train-batch-size 32 \
  --output-dir outputs/batch_mode_comparison/tbatch

# 3. TGN (loss_mode=all)
echo "[3/4] Running TGN (all) mode..."
python search.py \
  --dataset public_csv --local-data-path data/public/mooc.csv \
  --max-events ${MAX_EVENTS} --space mixed \
  --search-mode rl --execution-mode ray_pipeline \
  --trials ${TRIALS} --epochs-per-trial ${EPOCHS} \
  --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1 \
  --architectures-per-step 9 --pipeline-worker-gpus 1.0 \
  --stage-balance-strategy cost --pipeline-mode smart \
  --gpu-list ${GPU_LIST} --seed ${SEED} \
  --batch-mode tgn --tgn-loss-mode all --tgn-window-size 10.0 \
  --output-dir outputs/batch_mode_comparison/tgn_all

# 4. TGN (loss_mode=last)
echo "[4/4] Running TGN (last) mode..."
python search.py \
  --dataset public_csv --local-data-path data/public/mooc.csv \
  --max-events ${MAX_EVENTS} --space mixed \
  --search-mode rl --execution-mode ray_pipeline \
  --trials ${TRIALS} --epochs-per-trial ${EPOCHS} \
  --num-pipeline-stages 2 --pipeline-stage-train-workers 2,1 \
  --architectures-per-step 9 --pipeline-worker-gpus 1.0 \
  --stage-balance-strategy cost --pipeline-mode smart \
  --gpu-list ${GPU_LIST} --seed ${SEED} \
  --batch-mode tgn --tgn-loss-mode last --tgn-window-size 10.0 \
  --output-dir outputs/batch_mode_comparison/tgn_last

echo ""
echo "=== 所有实验完成 ==="
echo "结果保存在: outputs/batch_mode_comparison/"
