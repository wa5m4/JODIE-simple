#!/bin/bash
# 最小测试：验证所有批处理模式都能正常工作

echo "=== 测试1: 单GPU + 批处理 ==="
python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --max-events 1000 \
  --trials 2 \
  --epochs 1 \
  --batch-training \
  --train-batch-size 32 \
  --output-dir outputs/test_single_gpu_batch \
  --seed 42

if [ $? -ne 0 ]; then
    echo "❌ 单GPU批处理测试失败"
    exit 1
fi
echo "✅ 单GPU批处理测试通过"

echo ""
echo "=== 测试2: Native pipeline + 批处理 ==="
python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --max-events 1000 \
  --trials 2 \
  --epochs 1 \
  --pipeline-mode naive \
  --batch-training \
  --train-batch-size 32 \
  --output-dir outputs/test_naive_batch \
  --seed 42

if [ $? -ne 0 ]; then
    echo "❌ Native pipeline批处理测试失败"
    exit 1
fi
echo "✅ Native pipeline批处理测试通过"

echo ""
echo "=== 测试3: Smart pipeline + 批处理 ==="
python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --max-events 1000 \
  --trials 2 \
  --epochs 1 \
  --pipeline-mode smart \
  --batch-training \
  --train-batch-size 32 \
  --output-dir outputs/test_smart_batch \
  --seed 42

if [ $? -ne 0 ]; then
    echo "❌ Smart pipeline批处理测试失败"
    exit 1
fi
echo "✅ Smart pipeline批处理测试通过"

echo ""
echo "=== 所有测试通过！ ==="
