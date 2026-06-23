#!/bin/bash
# 验证 Data Parallel 模式下 batch_mode 参数是否生效

echo "=========================================="
echo "Testing Data Parallel with TGN batch mode"
echo "=========================================="

python search.py \
  --execution-mode data_parallel \
  --batch-mode tgn \
  --tgn-loss-mode last \
  --tgn-window-size 10.0 \
  --space rnn_only \
  --dataset synthetic \
  --num-users 100 \
  --num-items 200 \
  --num-interactions 5000 \
  --trials 2 \
  --epochs-per-trial 1 \
  --output-dir outputs/test_dp_tgn_last \
  --data-parallel-workers 2 \
  --seed 42

echo ""
echo "=========================================="
echo "Test completed. Check the output above for:"
echo "  [Verify] train_model() called with batch_mode=tgn, tgn_loss_mode=last, tgn_window_size=10.0"
echo "=========================================="
