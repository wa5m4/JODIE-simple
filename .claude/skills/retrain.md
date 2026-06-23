# Retrain Skill

对NAS搜索出的最佳架构进行标准化重训练验证。

## 含义

**Retrain（重训练）**指的是：
- 从某个策略（数据并行、Pipeline等）的搜索结果中提取**最佳架构配置**
- 使用该固定架构在**final test模式**下重新训练
- 获得该架构的**真实性能分数**（Test MRR），与其他策略可比较

## 为什么需要统一的Retrain

不同策略的Test MRR必须在**相同条件**下评估才能公平对比：
- **相同的训练逻辑**：使用search.py的final test逻辑
- **相同的seed**：base_seed + 20000 = final_seed
- **相同的训练数据**：train+val作为训练集
- **相同的测试数据**：test作为测试集

## 标准Retrain脚本

使用 `retrain_final_test.py`（与search.py的final test逻辑完全一致）：

```bash
python retrain_final_test.py \
    --best-arch-json <path/to/best_arch.json> \
    --output-dir <output/dir/retrain> \
    --dataset public_csv \
    --local-data-path data/public/mooc.csv \
    --max-events 20000 \
    --seed 42 \
    --epochs 3
```

## 关键参数

- `--best-arch-json`: 搜索结果的best_arch.json路径
- `--output-dir`: retrain结果保存目录
- `--seed`: 基础seed（将自动+20000作为final seed）
- `--epochs`: 训练epochs（默认3）

## Final Test逻辑

```python
# 来自 nas/trainer.py:930-954
final_train_data = train_data + val_data  # 训练数据：train+val
final_seed = base_seed + 20000             # Seed：42 → 20042
final_test_result = trainer._evaluate_arch_multi_seed(
    arch_config=arch_config,
    train_data=final_train_data,
    eval_data=test_data,                   # 测试数据：test
    epochs=epochs,
    default_seed=final_seed,
    phase="final",
    eval_split="test",
)
```

## 示例

```bash
# 对数据并行的搜索结果进行retrain
python retrain_final_test.py \
    --best-arch-json outputs/comprehensive_comparison/seed_42/data_parallel_no_split/best_arch.json \
    --output-dir outputs/comprehensive_comparison/seed_42/data_parallel_no_split/retrain \
    --seed 42
```

## 预期结果

所有选出相同架构（off/off）的策略，retrain后Test MRR应该完全一致：

| 策略 | 架构 | Retrain Test MRR |
|------|------|------------------|
| Serial | off/off | 0.850914 |
| Smart+20% | off/off | 0.850914 |
| Naive+20% | off/off | 0.850914 |
| 数据并行改进 | off/off | 0.850914 ✓ |

## 与train_single_arch.py的区别

| 维度 | train_single_arch.py | retrain_final_test.py |
|------|---------------------|----------------------|
| 训练数据 | train only | train+val |
| Seed | 用户指定 | base_seed + 20000 |
| 逻辑来源 | 独立实现 | search.py的final test |
| 结果一致性 | 可能有差异 | 与search.py完全一致 |

## 目录结构

```
outputs/comprehensive_comparison/seed_42/
├── data_parallel_no_split/
│   ├── best_arch.json          # 搜索结果
│   └── retrain/                
│       └── best_arch.json      # Final test retrain结果
```

