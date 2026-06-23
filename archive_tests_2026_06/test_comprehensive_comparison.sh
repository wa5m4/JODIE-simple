#!/bin/bash

# 全面对比实验：多种Pipeline配置 + Serial重训练验证

SEEDS=(100 1000)
GPU_LIST="0,1,2"
MAX_EVENTS=20000
TRIALS=50
EPOCHS=3
PARTITION_SIZE=5000

echo "========================================================================"
echo "全面对比实验"
echo "========================================================================"
echo "配置:"
echo "  种子: ${SEEDS[@]}"
echo "  数据: $MAX_EVENTS events"
echo "  搜索: $TRIALS trials × $EPOCHS epochs"
echo "  Partition: $PARTITION_SIZE"
echo "  GPU: $GPU_LIST"
echo "========================================================================"

BASE_OUTPUT="outputs/comprehensive_comparison"
mkdir -p "$BASE_OUTPUT"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "种子 $SEED 开始"
    echo "========================================================================"

    SEED_DIR="$BASE_OUTPUT/seed_$SEED"
    mkdir -p "$SEED_DIR"

    # ============================================================
    # 1. Serial (无Pipeline)
    # ============================================================
    echo ""
    echo "[1/7] Serial (无Pipeline)"
    python search.py \
        --search-mode rl \
        --execution-mode serial \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$TRIALS" \
        --coarse-epochs "$EPOCHS" \
        --output-dir "${SEED_DIR}/serial" \
        --space rnn_only \
        --batch-mode tbatch \
        2>&1 | tee "${SEED_DIR}/serial.log"

    # ============================================================
    # 2. 数据并行
    # ============================================================
    echo ""
    echo "[2/7] 数据并行"
    python search.py \
        --search-mode rl \
        --execution-mode data_parallel \
        --data-parallel-workers 3 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$TRIALS" \
        --coarse-epochs "$EPOCHS" \
        --output-dir "${SEED_DIR}/data_parallel" \
        --space rnn_only \
        --batch-mode tbatch \
        2>&1 | tee "${SEED_DIR}/data_parallel.log"

    # ============================================================
    # 3. Pipeline Smart + 20%预热
    # ============================================================
    echo ""
    echo "[3/7] Pipeline Smart (异步 + 20%预热)"
    python search.py \
        --search-mode rl \
        --execution-mode ray_pipeline \
        --pipeline-mode smart \
        --num-pipeline-stages 1 \
        --pipeline-stage-train-workers 3 \
        --pipeline-worker-gpus 1.0 \
        --partition-size "$PARTITION_SIZE" \
        --partition-overlap-ratio 0.2 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$TRIALS" \
        --coarse-epochs "$EPOCHS" \
        --output-dir "${SEED_DIR}/smart_overlap20" \
        --space rnn_only \
        --batch-mode tbatch \
        2>&1 | tee "${SEED_DIR}/smart_overlap20.log"

    # ============================================================
    # 4. Pipeline Smart + 无预热
    # ============================================================
    echo ""
    echo "[4/7] Pipeline Smart (异步 + 无预热)"
    python search.py \
        --search-mode rl \
        --execution-mode ray_pipeline \
        --pipeline-mode smart \
        --num-pipeline-stages 1 \
        --pipeline-stage-train-workers 3 \
        --pipeline-worker-gpus 1.0 \
        --partition-size "$PARTITION_SIZE" \
        --partition-overlap-ratio 0.0 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$TRIALS" \
        --coarse-epochs "$EPOCHS" \
        --output-dir "${SEED_DIR}/smart_no_overlap" \
        --space rnn_only \
        --batch-mode tbatch \
        2>&1 | tee "${SEED_DIR}/smart_no_overlap.log"

    # ============================================================
    # 5. Pipeline Naive + 20%预热
    # ============================================================
    echo ""
    echo "[5/7] Pipeline Naive (同步 + 20%预热)"
    python search.py \
        --search-mode rl \
        --execution-mode ray_pipeline \
        --pipeline-mode naive \
        --num-pipeline-stages 1 \
        --pipeline-stage-train-workers 3 \
        --pipeline-worker-gpus 1.0 \
        --partition-size "$PARTITION_SIZE" \
        --partition-overlap-ratio 0.2 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$TRIALS" \
        --coarse-epochs "$EPOCHS" \
        --output-dir "${SEED_DIR}/naive_overlap20" \
        --space rnn_only \
        --batch-mode tbatch \
        2>&1 | tee "${SEED_DIR}/naive_overlap20.log"

    # ============================================================
    # 6. Pipeline Naive + 无预热
    # ============================================================
    echo ""
    echo "[6/7] Pipeline Naive (同步 + 无预热)"
    python search.py \
        --search-mode rl \
        --execution-mode ray_pipeline \
        --pipeline-mode naive \
        --num-pipeline-stages 1 \
        --pipeline-stage-train-workers 3 \
        --pipeline-worker-gpus 1.0 \
        --partition-size "$PARTITION_SIZE" \
        --partition-overlap-ratio 0.0 \
        --gpu-list "$GPU_LIST" \
        --dataset public_csv \
        --local-data-path data/public/mooc.csv \
        --max-events "$MAX_EVENTS" \
        --seed "$SEED" \
        --coarse-trials "$TRIALS" \
        --coarse-epochs "$EPOCHS" \
        --output-dir "${SEED_DIR}/naive_no_overlap" \
        --space rnn_only \
        --batch-mode tbatch \
        2>&1 | tee "${SEED_DIR}/naive_no_overlap.log"

    echo ""
    echo "✓ 种子 $SEED NAS搜索完成"

    # ============================================================
    # 7. Serial重训练所有策略的best架构
    # ============================================================
    echo ""
    echo "[7/7] Serial重训练各策略的best架构"

    STRATEGIES=(
        "serial"
        "data_parallel"
        "smart_overlap20"
        "smart_no_overlap"
        "naive_overlap20"
        "naive_no_overlap"
    )

    for STRATEGY in "${STRATEGIES[@]}"; do
        BEST_FILE="${SEED_DIR}/${STRATEGY}/best_arch.json"

        if [ ! -f "$BEST_FILE" ]; then
            echo "  跳过 $STRATEGY: best_arch.json 不存在"
            continue
        fi

        echo "  重训练: $STRATEGY"

        # 提取架构参数
        MODEL=$(python3 -c "import json; print(json.load(open('$BEST_FILE'))['config']['model'])")
        EMB_DIM=$(python3 -c "import json; print(json.load(open('$BEST_FILE'))['config']['embedding_dim'])")
        MEMORY_CELL=$(python3 -c "import json; print(json.load(open('$BEST_FILE'))['config']['memory_cell'])")
        TIME_PROJ=$(python3 -c "import json; print(json.load(open('$BEST_FILE'))['config']['time_proj'])")
        USE_STATIC=$(python3 -c "import json; print(json.load(open('$BEST_FILE'))['config']['use_static_embeddings'])")

        RETRAIN_DIR="${SEED_DIR}/${STRATEGY}/retrain"
        mkdir -p "$RETRAIN_DIR"

        python train_single_arch.py \
            --model "$MODEL" \
            --embedding-dim "$EMB_DIM" \
            --memory-cell "$MEMORY_CELL" \
            --time-proj "$TIME_PROJ" \
            --use-static-embeddings "$USE_STATIC" \
            --dataset public_csv \
            --local-data-path data/public/mooc.csv \
            --max-events "$MAX_EVENTS" \
            --epochs "$EPOCHS" \
            --seed "$SEED" \
            --batch-mode tbatch \
            --eval-frozen false \
            --output-dir "$RETRAIN_DIR" \
            2>&1 | tee "${RETRAIN_DIR}.log"
    done

    echo ""
    echo "========================================================================"
    echo "种子 $SEED 完成（NAS + Serial重训练）"
    echo "========================================================================"
done

echo ""
echo "========================================================================"
echo "所有实验完成！生成汇总报告..."
echo "========================================================================"

# 生成详细对比报告
python3 << 'PYEOF'
import json
from pathlib import Path

print()
print("=" * 80)
print("全面对比实验结果报告")
print("=" * 80)
print()

seeds = [100, 1000]
strategies = [
    ("serial", "Serial (无Pipeline)"),
    ("data_parallel", "数据并行"),
    ("smart_overlap20", "Smart + 20%预热"),
    ("smart_no_overlap", "Smart + 无预热"),
    ("naive_overlap20", "Naive + 20%预热"),
    ("naive_no_overlap", "Naive + 无预热"),
]

for seed in seeds:
    print(f"【种子 {seed}】")
    print("=" * 80)
    print()
    
    base_dir = Path(f"outputs/comprehensive_comparison/seed_{seed}")
    
    # 表头
    print(f"{'策略':<25} {'NAS架构选择':<25} {'NAS Test':<10} {'Retrain Test':<12} {'差距':<8}")
    print("-" * 85)
    
    results = []
    
    for strategy_dir, strategy_name in strategies:
        nas_path = base_dir / strategy_dir / "best_arch.json"
        retrain_path = base_dir / strategy_dir / "retrain" / "result.json"
        
        if nas_path.exists():
            with open(nas_path) as f:
                nas_data = json.load(f)
            
            config = nas_data['config']
            arch_str = f"{config['time_proj']}/{config['use_static_embeddings'][:2]}"
            nas_test = nas_data.get('test_mrr', 0)
            
            retrain_test = 0
            if retrain_path.exists():
                with open(retrain_path) as f:
                    retrain_data = json.load(f)
                retrain_test = retrain_data.get('test_mrr', 0)
            
            gap = retrain_test - nas_test if retrain_test > 0 else 0
            gap_str = f"{gap:+.4f}" if retrain_test > 0 else "N/A"
            retrain_str = f"{retrain_test:.4f}" if retrain_test > 0 else "N/A"
            
            is_correct = config['time_proj'] == 'off' and config['use_static_embeddings'] == 'off'
            status = "✅" if is_correct else "❌"
            
            print(f"{status} {strategy_name:<23} {arch_str:<25} {nas_test:<10.4f} {retrain_str:<12} {gap_str:<8}")
            
            results.append({
                'strategy': strategy_name,
                'arch': arch_str,
                'nas_test': nas_test,
                'retrain_test': retrain_test,
                'correct': is_correct
            })
        else:
            print(f"⏳ {strategy_name:<23} 未完成")
    
    print()
    
    # 分析
    if results:
        print("分析:")
        correct = [r for r in results if r['correct']]
        wrong = [r for r in results if not r['correct']]
        
        print(f"  选对架构: {len(correct)}/{len(results)}")
        print(f"  选错架构: {len(wrong)}/{len(results)}")
        
        if correct:
            best_retrain = max(r['retrain_test'] for r in correct if r['retrain_test'] > 0)
            print(f"  最佳Retrain性能: {best_retrain:.4f}")
        
        if wrong:
            print(f"  错误选择:")
            for r in wrong:
                print(f"    • {r['strategy']}: {r['arch']} (Test={r['nas_test']:.4f})")
    
    print()
    print()

print("=" * 80)
print("关键结论")
print("=" * 80)
print()
print("对比维度:")
print("  1. Pipeline模式: Smart(异步) vs Naive(同步)")
print("  2. 预热策略: 20%重叠 vs 无重叠")
print("  3. NAS vs Retrain: 架构选择准确性 vs 最终性能")
print()
print("预期验证:")
print("  • Naive无预热 应与 Serial 选出相同架构")
print("  • Smart有预热 在种子100失败、种子1000成功")
print("  • Retrain能恢复Serial训练的完整性能")

PYEOF

echo ""
echo "报告完成！"
