# DP 保真度专项:种子流分析(2026-09-09,代码级)

> **背景**:Phase 1 发现 DP 在 B(20K)/D(100K)两个 cell 都偏离家族架构、val 质量最差;
> 与 E1「四策略一致」的历史结论矛盾。draft_v2_实验结果_C2.md 猜想「trial 调度顺序不同 → RL 探索路径分叉」。
> 本文档给出**代码级根因链 + 修复方向 + 验证实验设计**。
> **状态(2026-09-09)**:Phase 2 链条已按用户决定终止于 E-dp(原因=本 bug),
> 最小修复已实施并带日志;冒烟验证 + B-cell 复测挂在五臂合并链(configs.py RUNS_P3)里。

---

## 一、根因:DP 路径完全没有种子纪律

对照 serial 路径(`trainer.search` → `_evaluate_arch_multi_seed` → `_train_and_eval` 先 `_set_seed(trial_seed)`,trainer.py:66-69),DP 路径有**三个缺口**:

### 缺口 1:每个 trial 的初始权重无种子

- [data_parallel.py:489](jodie/nas/data_parallel.py#L489) `_run_trial` 里 `model = build_model(config)` 构建初始模型,**前后没有任何 seed 设置**。
- 后果:DP 每个 trial 的初始权重来自「当时的全局 RNG 状态」,与 serial(seed = 42 + trial_idx)完全不同 → 训练起点不同 → 分数不同。

### 缺口 2:记录在案但从未使用的 trial seed

- [trainer.py:1279](jodie/nas/trainer.py#L1279) 结果里写了 `"seed": base_seed + trial_idx`,但 [trainer.py:1265](jodie/nas/trainer.py#L1265) `executor.run([arch], ...)` **根本不传 seed**。
- 后果:日志里看起来有 seed 纪律,实际上没有——典型「判读落到代码、不落到日志文案」。

### 缺口 3:driver 上三次无种子 build_model 消耗全局 RNG,污染控制器采样流

- [data_parallel.py:489](jodie/nas/data_parallel.py#L489)(初始模型)、[data_parallel.py:555](jodie/nas/data_parallel.py#L555)(评估模型)、[trainer.py:1271](jodie/nas/trainer.py#L1271)(算 params 的模型)——三次 `build_model` 都会消耗 torch 全局 RNG(embedding/Linear 初始化)。
- `RLGraphNASController.sample_arch` 的 categorical 采样用的正是 torch 全局 RNG(controller.py:56-67)。
- 后果:**每跑完一个 trial,下一个 trial 的采样起点就被上一次的无种子初始化随机推走** → DP 的 coarse 采样序列与 serial 完全不同 → RL 探索路径分叉。
- 这正是 draft_v2 猜想的精确版:不是「调度顺序」,而是「无种子初始化消耗了采样 RNG 流」。

### 连锁反应

初始权重不同 + 采样路径不同 → 分数不同 → RL 奖励不同 → 双重分叉。12-trial 的 D cell 里分叉直接表现为选出 time_linear+normalize 变体;50-trial 的 B cell 里 rerank 也没能救回来(val 0.77 垫底)。

---

## 二、修复方向(已实施,2026-09-09)

最小修复 = 让 DP 复刻 serial 的 RNG 消耗顺序:**采样 → 设种子 → 训练/评估 → RL 更新**。

1. ✅ `search_data_parallel` 循环内、`executor.run` 前:`self._set_seed(base_seed + trial_idx)`(与 serial 同款),并打印 `[DataParallel] trial=i seed=... (与 serial 同源)` 便于日志核对。
2. ✅ `_run_trial` 里两次 build_model + trainer.py 的 params build_model 都落在这个种子域内 → 初始权重与 serial 同源(缺口 2 的「记录但未用」随 1 一并解决:种子真正生效)。
3. (可选加固,未做)params 统计改为从 `model_state_dict` 计算 numel——保留 build 是因为 serial 每 trial 的种子域内同样有 params build,结构一致,不再构成污染。
4. 不追求 DP 分数与 serial 分数相等(DP 是梯度平均的近似训练,分数本来就有差);目标是**初始权重同源 + 采样流确定性 + 选择级保真**,让 RL 探索路径与 serial 可比。

**已知限制**:worker 侧 `train_chunk` 的 dropout RNG 未按 trial 设种子(改动面大,暂不做)——DP 分数在 run 间有微小噪声,但不影响选择级判读与 driver 侧采样流。

**注意**:micro_batch 梯度平均本身改变训练数学(DP 与 serial 分数不可能位级一致),所以修复后 DP 的保真度标准是「选择级」——选出同一家族架构、test 达标,而不是分数相等。这与 C1 已软化的承诺一致。

## 三、验证实验设计(修复后,已实施)

1. **冒烟测试·确定性对照**(快,~15 分钟 ×2):COARSE_TRIALS=12、rnn_only、5K 事件的 S cell(configs.py CELL_S),修复后 DP 连跑两遍,对比 coarse 采样序列。
   ⚠️ **预期修正(2026-09-09 代码级推演)**:序列应与「另一次修复后 DP」完全一致(确定性),但**不等于 serial 的序列**——driver 侧 RNG 消耗模式不同(DP 每 trial 三次 build_model 且训练 RNG 消耗发生在 worker 进程,serial 在 driver 进程逐交互消耗),逐 trial 序列相等在数学上不可达。选择级保真以第 2 步为准。
2. **B-cell 复测**(慢,约 5.5h):修复后重跑 B-DP(20K×2GPU、50 trials、rerank 8),看是否回到 133K 家族、test≈0.8561。若回到家族 → DP 保真度缺口修复,Phase 1 的 B-DP 数字需重测替换;若仍偏离 → 分叉另有原因(需查 micro_batch 训练的分数偏差本身是否系统性偏好 time_linear/normalize 变体)。
3. 若 2 仍偏离:做「固定架构分数对照」——取 Phase 1 B 轮 50 个 coarse 架构,serial 与 DP 各自打分,看 DP 分数是否有系统性偏置(类似双臂消融的评分污染分析)。

## 四、对论文的影响(预判)

- 修复成功 → C1 的「任何后端复现串行选择」对 DP 也成立,Phase 1 的 DP 保真度异常可以写成「发现并修复了 DP 后端的种子纪律缺口」——反而是一个加分的小故事。
- 修复失败(DP 分数天然偏置)→ C1 承诺需要范围限定(明确 DP 是「近似后端」),DP 在 Section 6 只报加速比、并列报保真度限制。
