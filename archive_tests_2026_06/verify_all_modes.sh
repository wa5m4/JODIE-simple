#!/bin/bash
# 快速验证所有模式（批处理+非批处理）

echo "=== 验证1: 单GPU 非批处理 ==="
python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --max-events 500 \
  --trials 1 \
  --epochs 1 \
  --output-dir outputs/verify_single_nobatch \
  --seed 42

[ $? -ne 0 ] && echo "❌ 失败" && exit 1
echo "✅ 通过"

echo ""
echo "=== 验证2: 单GPU 批处理 ==="
python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --max-events 500 \
  --trials 1 \
  --epochs 1 \
  --batch-training \
  --train-batch-size 32 \
  --output-dir outputs/verify_single_batch \
  --seed 42

[ $? -ne 0 ] && echo "❌ 失败" && exit 1
echo "✅ 通过"

echo ""
echo "=== 验证3: Native pipeline 非批处理 ==="
python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --max-events 500 \
  --trials 1 \
  --epochs 1 \
  --pipeline-mode naive \
  --output-dir outputs/verify_naive_nobatch \
  --seed 42

[ $? -ne 0 ] && echo "❌ 失败" && exit 1
echo "✅ 通过"

echo ""
echo "=== 验证4: Smart pipeline 非批处理 ==="
python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --max-events 500 \
  --trials 1 \
  --epochs 1 \
  --pipeline-mode smart \
  --output-dir outputs/verify_smart_nobatch \
  --seed 42

[ $? -ne 0 ] && echo "❌ 失败" && exit 1
echo "✅ 通过"

echo ""
echo "=== 所有验证通过！ ==="
