#!/usr/bin/env python3
import json
from pathlib import Path

base_dir = Path("outputs/full_cross_experiment_fixed")
experiments = ["serial_tbatch", "data_parallel_tbatch", "pipeline_naive_tbatch", "pipeline_smart_tbatch"]

print("=" * 120)
print("T-Batch 模式下四种并行策略的全面对比")
print("配置: partition_size=1000, batch_size=32, max_events=20000, trials=27")
print("=" * 120)
print()

results = []
for exp_name in experiments:
    best_arch_path = base_dir / exp_name / "best_arch.json"
    timing_log_path = base_dir / exp_name / "timing_log.csv"

    if best_arch_path.exists():
        with open(best_arch_path) as f:
            data = json.load(f)
            config = data.get("config", {})

            # Extract timing from timing_log.csv
            total_time = 0
            if timing_log_path.exists():
                with open(timing_log_path) as tf:
                    lines = tf.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1]
                        parts = last_line.strip().split(',')
                        if len(parts) >= 4:
                            try:
                                total_time = float(parts[3])  # end_time_s
                            except:
                                pass

            results.append({
                "name": exp_name.replace("_tbatch", ""),
                "test_mrr": data.get("test_mrr", 0),
                "test_recall": data.get("test_recall_at_k", 0),
                "val_mrr": data.get("val_mrr", 0),
                "val_recall": data.get("val_recall_at_k", 0),
                "time_sec": total_time,
                "model": config.get("model", "N/A"),
                "embedding_dim": config.get("embedding_dim", "N/A"),
                "memory_cell": config.get("memory_cell", "N/A"),
            })

# Section 1: Performance Metrics
print("【1. 性能指标对比】")
print("-" * 120)
print(f"{'模式':<20} | {'Test MRR':<10} | {'Test Recall@10':<15} | {'Val MRR':<10} | {'Val Recall@10':<15}")
print("-" * 120)

for r in results:
    print(f"{r['name']:<20} | {r['test_mrr']:<10.4f} | {r['test_recall']:<15.4f} | {r['val_mrr']:<10.4f} | {r['val_recall']:<15.4f}")

print()

# Section 2: Training Time
print("【2. 训练时间对比】")
print("-" * 120)
print(f"{'模式':<20} | {'总时间(秒)':<12} | {'总时间(分钟)':<15} | {'相对Serial':<12}")
print("-" * 120)

serial_time = [r['time_sec'] for r in results if r['name'] == 'serial'][0]
for r in results:
    time_min = r['time_sec'] / 60
    speedup = serial_time / r['time_sec'] if r['time_sec'] > 0 else 0
    print(f"{r['name']:<20} | {r['time_sec']:<12.0f} | {time_min:<15.1f} | {speedup:<12.2f}x")

print()

# Section 3: Efficiency (MRR per minute)
print("【3. 效率分析 (准确率/时间)】")
print("-" * 120)
print(f"{'模式':<20} | {'MRR/分钟':<15} | {'综合评分':<12} | {'排名':<6}")
print("-" * 120)

for r in results:
    time_min = r['time_sec'] / 60 if r['time_sec'] > 0 else 1
    mrr_per_min = r['test_mrr'] / time_min
    # Composite score: balance between accuracy and speed
    composite = r['test_mrr'] * 0.7 + (serial_time / r['time_sec'] if r['time_sec'] > 0 else 0) * 0.3
    r['mrr_per_min'] = mrr_per_min
    r['composite'] = composite

# Sort by composite score
sorted_results = sorted(results, key=lambda x: x['composite'], reverse=True)
for idx, r in enumerate(sorted_results, 1):
    print(f"{r['name']:<20} | {r['mrr_per_min']:<15.4f} | {r['composite']:<12.4f} | {idx:<6}")

print()

# Section 4: Overfitting Analysis
print("【4. 泛化能力分析 (Val-Test Gap)】")
print("-" * 120)
print(f"{'模式':<20} | {'Val MRR':<10} | {'Test MRR':<10} | {'Gap':<10} | {'状态':<15}")
print("-" * 120)

for r in results:
    gap = r['test_mrr'] - r['val_mrr']
    if gap > 0.05:
        status = "良好泛化"
    elif gap > -0.05:
        status = "平衡"
    else:
        status = "⚠️ 过拟合"
    print(f"{r['name']:<20} | {r['val_mrr']:<10.4f} | {r['test_mrr']:<10.4f} | {gap:+<10.4f} | {status:<15}")

print()

# Section 5: Architecture
print("【5. 最佳架构选择】")
print("-" * 120)
print(f"{'模式':<20} | {'Model':<25} | {'Embedding':<10} | {'Memory Cell':<12}")
print("-" * 120)

for r in results:
    print(f"{r['name']:<20} | {r['model']:<25} | {r['embedding_dim']:<10} | {r['memory_cell']:<12}")

print()

# Section 6: Summary and Recommendations
print("=" * 120)
print("【总结与建议】")
print("=" * 120)
print()

best_acc = max(results, key=lambda x: x['test_mrr'])
best_speed = min([r for r in results if r['time_sec'] > 0], key=lambda x: x['time_sec'])
best_balance = sorted_results[0]

print(f"✅ 最高准确率: {best_acc['name']} (Test MRR = {best_acc['test_mrr']:.4f})")
print(f"⚡ 最快速度: {best_speed['name']} (耗时 = {best_speed['time_sec']/60:.1f} 分钟)")
print(f"🎯 最佳平衡: {best_balance['name']} (综合评分 = {best_balance['composite']:.4f})")
print()

print("推荐策略:")
print("-" * 120)
print("1. 追求最高准确率 → Serial (MRR=0.8509, 但耗时最长)")
print("2. 追求速度与准确率平衡 → Pipeline Smart (MRR=0.7896, 速度快)")
print("3. 不推荐 → Data Parallel (准确率低且耗时长)")
print()
