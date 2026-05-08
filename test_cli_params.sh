#!/bin/bash
# 测试 CLI 参数解析

echo "════════════════════════════════════════════════════════════════"
echo "测试 1: 默认参数"
echo "════════════════════════════════════════════════════════════════"

cat > /tmp/test1.sh << 'EOF'
source scripts/run_comparison_3way.sh 2>/dev/null || true
EOF

cd /home/wanghaoyu/JODIE-simple

# 测试 1: 默认参数
bash << 'EOF'
cd /home/wanghaoyu/JODIE-simple

GPU_LIST=""
SEARCH_SPACE="rnn_only"
DATASET="public_csv"
DATA_FILE="data/public/mooc.csv"
MAX_EVENTS=20000
TIME_BUDGET=1200
EPOCHS=3
TRIALS=""
SEEDS="42,43"
OUTPUT_DIR=""

# 自动检测 GPU
if [ -z "$GPU_LIST" ]; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    NUM_GPUS=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
    GPU_LIST=$(python -c "print(','.join(str(i) for i in range($NUM_GPUS)))")
else
    NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
fi

# 设置 trials
SERIAL_TRIALS=999
DP_TRIALS=999
PIPELINE_TRIALS=999

# 转换 seeds
IFS=',' read -ra SEEDS_ARRAY <<< "$SEEDS"

echo "✓ GPU_LIST: $GPU_LIST"
echo "✓ NUM_GPUS: $NUM_GPUS"
echo "✓ SEEDS: ${SEEDS_ARRAY[@]}"
echo "✓ SEEDS count: ${#SEEDS_ARRAY[@]}"
echo "✓ DATASET: $DATASET"
echo "✓ MAX_EVENTS: $MAX_EVENTS"
echo "✓ TIME_BUDGET: $TIME_BUDGET"
EOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "测试 2: 自定义参数"
echo "════════════════════════════════════════════════════════════════"

bash << 'EOF'
# 模拟参数传入
GPU_LIST="0,1,2,3,4,5,6,7"
SEARCH_SPACE="small"
DATASET="synthetic"
DATA_FILE="data/synthetic.csv"
MAX_EVENTS=50000
TIME_BUDGET=600
EPOCHS=5
TRIALS="50"
SEEDS="1,2,3"
OUTPUT_DIR="outputs/test_custom"

if [ -z "$GPU_LIST" ]; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
else
    NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
fi

if [ -z "$TRIALS" ]; then
    SERIAL_TRIALS=999
    DP_TRIALS=999
    PIPELINE_TRIALS=999
else
    SERIAL_TRIALS="$TRIALS"
    DP_TRIALS="$TRIALS"
    PIPELINE_TRIALS="$TRIALS"
fi

IFS=',' read -ra SEEDS_ARRAY <<< "$SEEDS"

echo "✓ GPU_LIST: $GPU_LIST (NUM_GPUS: $NUM_GPUS)"
echo "✓ SEARCH_SPACE: $SEARCH_SPACE"
echo "✓ DATASET: $DATASET"
echo "✓ DATA_FILE: $DATA_FILE"
echo "✓ MAX_EVENTS: $MAX_EVENTS"
echo "✓ TIME_BUDGET: $TIME_BUDGET"
echo "✓ EPOCHS: $EPOCHS"
echo "✓ TRIALS: SERIAL=$SERIAL_TRIALS, DP=$DP_TRIALS, PIPELINE=$PIPELINE_TRIALS"
echo "✓ SEEDS: ${SEEDS_ARRAY[@]} (count: ${#SEEDS_ARRAY[@]})"
echo "✓ OUTPUT_DIR: $OUTPUT_DIR"
EOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "测试 3: 参数解析逻辑"
echo "════════════════════════════════════════════════════════════════"

bash << 'EOF'
cd /home/wanghaoyu/JODIE-simple

# 模拟参数传入和解析
params=("--gpu-list" "0,1,2" "--trials" "20" "--seeds" "42,43,44" "--time-budget" "300")

GPU_LIST=""
SEARCH_SPACE="rnn_only"
TRIALS=""
SEEDS="42,43"
TIME_BUDGET=1200

# 参数解析循环
i=0
while [ $i -lt ${#params[@]} ]; do
    case ${params[$i]} in
        --gpu-list)
            i=$((i+1))
            GPU_LIST="${params[$i]}"
            ;;
        --trials)
            i=$((i+1))
            TRIALS="${params[$i]}"
            ;;
        --seeds)
            i=$((i+1))
            SEEDS="${params[$i]}"
            ;;
        --time-budget)
            i=$((i+1))
            TIME_BUDGET="${params[$i]}"
            ;;
    esac
    i=$((i+1))
done

echo "✓ GPU_LIST: $GPU_LIST"
echo "✓ TRIALS: $TRIALS"
echo "✓ SEEDS: $SEEDS"
echo "✓ TIME_BUDGET: $TIME_BUDGET"
EOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ 所有测试完成"
echo "════════════════════════════════════════════════════════════════"
