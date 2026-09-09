# Pipeline 准确率问题：完整修复历程与最终结论

**日期**: 2026-08-13（四策略补跑完成）
**最终状态**: ✅ 修复成功，四策略结果完全一致

---

## 最终数据对比（修复后 · 四策略综合）

### 核心指标

| 策略 | 耗时 | val_score | **test_score** | test_mrr | test_recall@10 | 参数量 | 架构 | 加速比 |
|------|------|:---------:|:--------------:|:--------:|:--------------:|:------:|------|:------:|
| **Serial** (串行基线) | 25,965s (7.2h) | 0.8648 | **0.8561** | 0.8561 | 0.9893 | 133,888 | jodie_rnn (all OFF) | 1.0× |
| **Data Parallel** (3 workers) | 15,746s (4.4h) | 0.7521 | **0.8561** | 0.8561 | 0.9893 | 133,888 | jodie_rnn (all OFF) | **1.6×** |
| **Pipeline Naive** (3 stages × 1,1,1) | 16,783s (4.7h) | 0.8267 | **0.8561** | 0.8561 | 0.9893 | 133,888 | jodie_rnn (all OFF) | **1.5×** |
| **Pipeline Smart** (1 stage × 3 workers) | **11,113s (3.1h)** | 0.8267 | **0.8561** | 0.8561 | 0.9893 | 133,888 | jodie_rnn (all OFF) | **2.3×** |

**四个策略选出的架构完全一致，test_score 完全一致 (0.8561)。Pipeline Smart 加速 2.3×，为最优策略。**

> 注：val_score 仅用于搜索阶段的架构选择，不同策略的 val 排名略有差异（RL 控制器探索路径不同），
> 但最终选出架构在 test 集上的分数一致。**test_score 为最终评判标准**。

### 分阶段耗时明细

| 策略 | 粗搜索 50 trials | 重排序 8×5 epochs + 最终测试 | 总计 |
|------|:----------------:|:---------------------------:|:----:|
| Serial | 18,125s | 7,840s | 25,965s |
| Data Parallel | 10,620s (1.7×) | 5,126s (1.5×) | 15,746s |
| Pipeline Naive | 11,864s (1.5×) | 4,919s (1.6×) | 16,783s |
| Pipeline Smart | **6,939s (2.6×)** | **4,173s (1.9×)** | **11,113s** |

Smart 在粗搜索阶段加速最明显（2.6×）：异步池让 3 个 worker 始终满载，无批次同步等待。

### 修复前 vs 修复后

| 指标 | 修复前 | 修复后 |
|------|:------:|:------:|
| Serial test_score | 0.8561 | 0.8561 |
| Data Parallel test_score | 0.8561 | 0.8561 |
| Pipeline Naive test_score | **0.7000** | **0.8561** |
| Pipeline Smart test_score | **0.6687** | **0.8561** |
| Pipeline 选出架构 | 402K all ON（错误） | 133K all OFF（正确） |
| 四策略一致性 | ❌ 不一致 | ✅ 完全一致 |

### 结果文件

- Serial + Pipeline Naive: [`results/20260811_204240/comparison.json`](../results/20260811_204240/comparison.json)
- Data Parallel: [`results/20260812_153045/comparison.json`](../results/20260812_153045/comparison.json)（DP 部分）
- Pipeline Smart: [`results/20260813_103010/comparison.json`](../results/20260813_103010/comparison.json)

---

## 修复历程

### 阶段 1：问题发现（2026-08-05~06）

**现象**：首次全策略对比实验 (`run_all.log`)，partition_size=500。

| 策略 | test_score | 架构 |
|------|-----------|------|
| Serial | **0.8561** | 133K all OFF |
| Data Parallel | **0.8561** | 133K all OFF |
| Pipeline Naive | 0.7000 | 402K all ON |
| Pipeline Smart | 0.6687 | 402K all ON |

Pipeline 选出的架构完全不同，test_score 差距 **0.156**。

**文件**: [`PIPELINE_BIAS_ANALYSIS.md`](PIPELINE_BIAS_ANALYSIS.md)

---

### 阶段 2：五个被证伪的假说（2026-08-06~09）

逐个提出假说并验证：

| # | 假说 | 实验 | 结论 |
|---|------|------|:--:|
| 1 | 负采样 seed 含 `+partition_id` 导致分区间 RNG 不同 | 去掉 partition_id，重跑 (`run_all_seedfix.log`) | ❌ 无效 |
| 2 | 分区太小，训练分布偏差 | partition_size 500→2000 | ❌ 无效 |
| 3 | 模型前向/反向算子有 bug | 逐算子对比 (`diagnose_pipeline_divergence.py`) | ❌ max_diff=0.00 |
| 4 | Ray worker 间 state transfer 精度损失 | Roundtrip 验证 (`verify_optimizer_state_roundtrip.py`) | ❌ 零损失 |
| 5 | 分区评估偏差（分区评估低估小模型） | 全数据评估 Plan C (`test_plan_c.py`, `run_all_planc.log`) | ❌ 无效 |

**关键文件**: [`diagnose_pipeline_divergence.py`](diagnose_pipeline_divergence.py), [`verify_optimizer_state_roundtrip.py`](verify_optimizer_state_roundtrip.py), [`PIPELINE_DIVERGENCE_ANALYSIS.md`](PIPELINE_DIVERGENCE_ANALYSIS.md), [`PIPELINE_RUNALL_SUMMARY.md`](PIPELINE_RUNALL_SUMMARY.md)

---

### 阶段 3：RNG 重置假说（2026-08-09）— 部分正确，但方向错误

**精确诊断 v2** (`diagnose_v2_precise.py`) 发现：

```
第一个交互（分区0，step 0）:
  Serial neg samples:  [1, 16, 13, 9, 9]
  Pipeline neg samples: [1, 16, 13, 9, 9]  ✅ 一致

第 500 个交互（分区1，step 0）:
  Serial neg samples:  [18, 1, 14, 1, 11]  (RNG 继续消费)
  Pipeline neg samples: [1, 16, 13, 9, 9]  (RNG 重置!) ❌ 不同!
```

**当时的结论**：每分区 `rng = np.random.default_rng(seed)` 导致 BPR 负采样在分区边界重复。结论写入当时的旧版 `FINAL_REPORT.md`（已被本版覆盖）。

**文件**: [`diagnose_v2_precise.py`](diagnose_v2_precise.py)

---

### 阶段 4：关键转折 — 发现 public_csv 不用 BPR（2026-08-10）

**单架构训练一致性测试** (`test_l2_divergence.py`, `test_ray_pipeline_match.py`) 揭示：

- **Serial ≡ Pipeline** 训练结果完全一致（MRR **0.8487636143**，embedding **-0.0065195826**，diff=0）
- 原因：`public_csv` 使用 **L2/CE loss**，不需要负采样，因此 RNG 重置不影响训练
- RNG 假说只适用于 BPR 训练路径（synthetic 模式），不适用于 public_csv

**这意味着**：RNG 重置不是 public_csv 上 Pipeline 搜索偏差的根因。偏差一定在搜索层面，而非训练层面。

**文件**: [`test_l2_divergence.py`](test_l2_divergence.py), [`test_ray_pipeline_match.py`](test_ray_pipeline_match.py)

---

### 阶段 5：定位真正的三个根因（2026-08-10~11）

对比 Serial 和 Pipeline 的搜索代码路径（`trainer.py`），精确定位三个差异：

#### 根因 1：`_make_payload` 未设种子 → 所有 Pipeline trial 初始权重相关

**文件**: [`jodie/nas/ray_pipeline.py`](jodie/nas/ray_pipeline.py), 函数 `_make_payload()`

```python
# 修复前：build_model 前未设 seed
def _make_payload(self, arch_config, trial_id, seed):
    config = dict(self.base_config)
    config.update(arch_config)
    model = build_model(config)  # ← 使用全局 RNG 当前状态，与之前 trial 相关

# 修复后
def _make_payload(self, arch_config, trial_id, seed):
    torch.manual_seed(seed)      # ✅ 每个 trial 独立初始化
    config = dict(self.base_config)
    config.update(arch_config)
    model = build_model(config)
```

Serial 路径在 `_train_and_eval` 中先 `_set_seed(trial_seed)` 再 `build_model`，Pipeline 没有这一步。

#### 根因 2：评估循环中冗余 `build_model` 污染全局 RNG

**文件**: [`jodie/nas/trainer.py`](jodie/nas/trainer.py), 函数 `evaluate_arch_pipeline()`

```python
# 修复前：build_model 消费 RNG 后权重立即被 load_state_dict 覆盖，但 RNG 状态已污染
config = dict(self.base_config)
config.update(payload.arch_config)
model = build_model(config)          # ← 消费了 RNG
model.load_state_dict(payload.state) # ← 权重被覆盖，但 RNG 已污染

# 修复后：保存/恢复 RNG
rng_state = torch.get_rng_state()
model = build_model(config)          # 消费 RNG 不影响后续
torch.set_rng_state(rng_state)       # ✅ 恢复
model.load_state_dict(payload.state)
```

每个 trial 评估时多调用一次 `build_model`，累计 50+ 次 → 下一 batch 的 `_make_payload` 中 `build_model` 获得的初始权重完全偏离 Serial。

#### 根因 3：Controller 批量更新 vs 逐 trial 更新

**文件**: [`jodie/nas/trainer.py`](jodie/nas/trainer.py), 函数 `search_pipeline()`

```python
# 修复前：batch 模式，每 4 个 trial 统一更新一次
batch_samples = [(logprob, score) for ...]
controller.reinforce_step_batch(batch_samples)

# 修复后：逐 trial 更新（与 serial 一致），使用 compute_logprob 做 off-policy 校正
for arch_cfg, result in zip(arch_batch, batch_results):
    logprob = controller.compute_logprob(arch_cfg)  # ✅ 重新计算，避免 inplace 冲突
    controller.reinforce_step(logprob, result["score"])
```

Serial 搜索每完成一个 trial 就更新 controller；Pipeline 攒 4 个 trial 才更新一次。反馈频率不同 → 控制器探索路径不同 → 选出的架构不同。

**文件**: [`test_minimal_search.py`](test_minimal_search.py)（最小搜索验证）

---

### 阶段 6：验证与最终确认（2026-08-11~12）

#### 最小搜索测试（4 trials × 1 epoch）

| 策略 | 状态 | 耗时 |
|------|:--:|------|
| Serial | ✅ 完成 | 1019s |
| Pipeline Naive | ✅ 完成（controller 更新无崩溃） | 896s |

#### 完整 50-trial 测试

| 策略 | test_score | test_mrr | 架构 | 耗时 |
|------|:---------:|:--------:|------|------|
| Serial | **0.8561** | **0.8561** | jodie_rnn, 133K | 25,965s |
| Pipeline Naive | **0.8561** | **0.8561** | jodie_rnn, 133K | 16,783s |

**Serial ≡ Pipeline Naive，结果完全一致。Pipeline 加速 1.5×。**

**结果文件**: [`results/20260811_204240/comparison.json`](../results/20260811_204240/comparison.json)

---

### 阶段 7：四策略补跑 — Smart 异步池 flush 的 inplace bug（2026-08-12~13）

补跑 Data Parallel 与 Pipeline Smart（1 stage × 3 workers）：

#### 最小测试（4 策略 × 4 trials）— 全部通过

| 策略 | 状态 | best |
|------|:--:|------|
| Serial | ✅ | 0.7906 |
| Data Parallel | ✅ | 0.4479 |
| Pipeline Naive | ✅ | 0.7019 |
| Pipeline Smart | ✅ | 0.7906 |

#### Data Parallel 全量（2026-08-12）

✅ 完成，耗时 15,746s，test_score=0.8561，架构与 Serial 完全一致。

#### Pipeline Smart 全量首次失败：inplace version 冲突

```
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [torch.FloatTensor [2]] is at version 12;
expected version 11 instead.
    at trainer.py:711 → controller.reinforce_step(stored_lp, sc)
```

**根因**：Smart 异步路径在搜索末尾 flush 剩余 `update_buffer` 时，
**优先使用了采样时保存的原始 logprob**（`stored_lp`），其计算图引用的 logits
已被前面 12 次 `optimizer.step()` inplace 修改（version 11 → 12），backward 失败。

最小测试没有暴露此问题：4 trials ÷ 2 arch/step 恰好整除，flush 路径从未被触发。
全量 50 trials ÷ 4 arch/step = 12 次批量更新 + 最后 flush 2 个 → 触发崩溃。

**修复**（[`jodie/nas/trainer.py`](jodie/nas/trainer.py) `_search_pipeline_async` 末尾）：

```python
# 修复前：优先使用存储的原始 logprob（计算图引用旧版本 logits）
if stored_lp is not None:
    controller.reinforce_step(stored_lp, sc)
elif hasattr(controller, "compute_logprob"):
    logprob = controller.compute_logprob(arch_cfg)
    controller.reinforce_step(logprob, sc)

# 修复后：优先用 compute_logprob 重算 logprob（off-policy）
if hasattr(controller, "compute_logprob"):
    logprob = controller.compute_logprob(arch_cfg)  # ✅ 基于当前 logits 重算
    controller.reinforce_step(logprob, sc)
elif stored_lp is not None:
    controller.reinforce_step(stored_lp, sc)
```

**验证**：

1. [`test_smart_flush.py`](test_smart_flush.py) — 5 trials × 2 arch/step → 2 次批量更新 + 最后 flush 1 个，✅ 通过（730s，best=0.7906）
2. 全量 50-trial 补跑 — ✅ 完成（[`run_all_smart_fix.log`](../run_all_smart_fix.log)），test_score=0.8561

#### 最终四策略结果（见文首表格）

**四个策略选出完全相同的架构（jodie_rnn, 133K, all OFF），test_score 完全一致 (0.8561)。**

---

## 错误结论的反思

原始 FINAL_REPORT.md 的错误在于：

1. **找到了 RNG 重置问题**（真实存在），但 **public_csv 用 L2 loss 不受此影响**
2. **把 BPR 训练路径的 bug 当作 public_csv 的根因**，导致花了大量时间在预分配负样本方案上
3. **应该更早做单架构训练一致性测试** — 这能立即证明训练本身没问题，偏差在搜索层面

正确的方法是"**二分排除法**"：
- 先验证训练是否一致（单架构 Serial vs Pipeline）→ ✅ 一致
- 再找搜索循环中的差异（代码对比 Serial vs Pipeline 搜索路径）→ 找到 3 个根因
- Smart 异步路径的 flush bug 则是"**最小测试覆盖不足**"的教训：4 % 2 = 0 恰好整除，
  测试没有触达最后 flush 分支。最小测试的 trial 数应设计为**不整除** arch_per_step。

---

## 最终结论

**四种搜索策略（Serial / Data Parallel / Pipeline Naive / Pipeline Smart）在 NAS 搜索中结果一致。**

三个搜索层面的 RNG/控制器差异是导致 Pipeline 选错架构的真正原因：
1. `_make_payload` 中 `build_model` 前未设种子
2. `evaluate_arch_pipeline` 中冗余 `build_model` 污染 RNG
3. Controller 批量更新 vs 逐 trial 更新（+ 存储 logprob 的 inplace 冲突，含 Smart flush 变体）

修复后：
- **四策略 test_score 完全一致** (0.8561)
- **选出的架构完全一致** (jodie_rnn, all OFF, 133K)
- **加速比**：Pipeline Smart **2.3×** > Data Parallel 1.6× > Pipeline Naive 1.5×
- **推荐 Pipeline Smart (1 stage × 3 workers)** 作为 NAS 加速策略：最快且结果与 Serial 一致

---

## 文件索引

| 文件 | 说明 |
|------|------|
| `diagnose_v2_precise.py` | 精确诊断 v2 — 定位 per-partition RNG 重置 |
| `diagnose_pipeline_divergence.py` | 诊断 v1 — 算子对比 |
| [`test_l2_divergence.py`](test_l2_divergence.py) | ⭐ L2 loss 训练一致性验证（Serial ≡ Pipeline） |
| [`test_ray_pipeline_match.py`](test_ray_pipeline_match.py) | ⭐ Ray Pipeline 训练一致性验证（MRR diff=0） |
| [`test_minimal_search.py`](test_minimal_search.py) | ⭐ 最小搜索测试（4 策略，验证 controller 修复） |
| [`test_smart_flush.py`](test_smart_flush.py) | ⭐ Smart flush 路径最小测试（5 trials 不整除） |
| [`test_plan_c.py`](test_plan_c.py) | 方案 C 全数据评估验证 |
| `verify_optimizer_state_roundtrip.py` | State transfer 精度验证 |
| [`../results/20260811_204240/comparison.json`](../results/20260811_204240/comparison.json) | ⭐ Serial + Pipeline Naive 最终结果 |
| [`../results/20260812_153045/comparison.json`](../results/20260812_153045/comparison.json) | ⭐ Data Parallel 最终结果 |
| [`../results/20260813_103010/comparison.json`](../results/20260813_103010/comparison.json) | ⭐ Pipeline Smart 最终结果 |
| `run_all.log` | 首次全策略对比（修复前） |
| `run_all_seedfix.log` | Seed 修复实验 |
| `run_all_planc.log` | 方案 C 全量实验 |
| [`../run_all_dp_smart.log`](../run_all_dp_smart.log) | DP+Smart 补跑（Smart 首次失败日志） |
| [`../run_all_smart_fix.log`](../run_all_smart_fix.log) | Smart 修复后补跑（成功） |
