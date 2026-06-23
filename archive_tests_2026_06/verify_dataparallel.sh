#!/bin/bash
# 验证数据并行模式

echo "=== 验证: 数据并行 非批处理 ==="
python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --max-events 500 \
  --trials 1 \
  --epochs 1 \
  --execution-mode data_parallel \
  --data-parallel-workers 3 \
  --data-parallel-visible-gpus "0,1,2" \
  --output-dir outputs/verify_dataparallel_nobatch \
  --seed 42

[ $? -ne 0 ] && echo "❌ 失败" && exit 1
echo "✅ 通过"

echo ""
echo "=== 验证: 数据并行 批处理 ==="
python search.py \
  --dataset public_csv \
  --local-data-path data/public/mooc.csv \
  --max-events 500 \
  --trials 1 \
  --epochs 1 \
  --execution-mode data_parallel \
  --data-parallel-workers 3 \
  --data-parallel-visible-gpus "0,1,2" \
  --batch-training \
  --train-batch-size 32 \
  --output-dir outputs/verify_dataparallel_batch \
  --seed 42

[ $? -ne 0 ] && echo "❌ 失败" && exit 1
echo "✅ 通过"

echo ""
echo "=== 数据并行验证完成！ ==="
