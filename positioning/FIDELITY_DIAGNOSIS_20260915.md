# 保真度问题诊断(2026-09-15,实验已全部停掉后)

**问题**:同步流水线(naive)在 D/E/G(100K)选出与 serial 不同的架构(D/E: serial 8896 vs 同步族 338240);
naive_async 在 B/D/F/G 与 serial 位级一致;DP 在大负载全分叉。B/C/F 的 serial 与 naive 选择恰好一致。
**方法**:对比 B/D/F 的 serial/naive/naive_async leaderboard(experiments/results/ + results/20260913_*)
+ 阅读 trainer.py / ray_pipeline.py / loops.py / controller.py / metrics.py 训练与采样路径。

## 一、数据发现(leaderboard 对比)

1. **候选集分叉**:D serial 10 个候选 {8896,1280640,28864,...} vs D naive 9 个 {338240,334592,110976,...}
   重叠仅 4/10;D naive 首候选 338240 ≠ D serial 首候选 8896。**B 的首两个候选一致(133888,133888),
   第 3 个起分叉**——采样流在两次 draw 后分叉。
2. **naive_async ≈ serial**:D naive_async 12 候选与 serial 重叠 9/10,首候选 8896 一致(off-policy
   逐完成更新 + 逐个采样,与 serial 同构)。
3. **同架构分数不同**:
   - D 8896:serial 0.476019(seed 42)vs naive 0.21672(seed 45,批次内位置 3)vs naive_async
     **0.500769(seed 42!)**——同种子同架构,流水线分数仍 ≠ serial。
   - F 34176:serial 0.5335 vs naive 0.2697;F 564608:0.0189 vs 0.2858。
   - 排名大体保持(133888 在 B/F 都登顶),100K 处接近对决被翻转。

## 二、代码层面:三个分叉机制(按影响排序)

### 机制 1 · 控制器采样流未锚定(首要,解释候选集分叉)

- 采样 = `dist.sample()`(torch 全局 RNG,controller.py:62),每维一次 draw。
- serial 逐个采样+逐 trial 更新(trainer.py:1097,1137);同步流水线**先整批采 4 个再更新**
  (trainer.py:916,965)——第 3 个 draw 时 serial 已做过 1 次 reinforce_step,θ 已微移 → 分叉(B 实测:前 2 个一致、第 3 个分叉)。
- 首 draw 分叉(D):控制器创建与首次采样之间,流水线路径多创建了 RayPipelineExecutor(874 行),
  其初始化若消费 torch RNG 即改变首 draw;`_sample_unique_arch` 的去重重试(64 次上限)也让 RNG 消耗
  与已见候选集耦合,路径差异被放大。
- **异步引擎天然规避**:`_sample_unique_arch_batch(controller, 1, ...)`(632/699 行)逐个采样 + 逐完成
  off-policy 更新 → 采样流与 serial 重合(实测 9/10)。

### 机制 2 · 评估种子错位(解释同架构分数不可比)

- serial:trial_seed = 42 + 全局 trial 序号(trainer.py:1102)。
- 流水线:`_make_payload(arch, trial_id=idx, seed=42+idx)`,idx 是**批次内位置**(ray_pipeline.py:1532),
  每批从 0 重新计数 → 全局第 5 个候选在 serial 得 seed 47、在流水线得 seed 42 → 初始化不同 → 分数不同。

### 机制 3 · 同种子下流水线训练仍 ≠ serial(解释 8896 seed42 的 0.5008 vs 0.4760)

已核对并对齐的:负样本预分配(public_dataset.py:160,数据加载期按流序生成,两路径同一份数据)、
epoch 结构(run_train_only 逐 epoch 重建 payload、runtime_state=None 重置,与 serial reset 一致)、
optimizer 状态 FQN 交接(ray_pipeline.py:333)、eval 语义(两路径均 frozen=False 在线评估,metrics.py
无 RNG)。**静态分析未定位**,剩余嫌疑:
- optimizer FQN 状态加载 bug(`_optimizer_state_from_fqn`,状态错位 = 每 stage 重置动量);
- `_make_payload` 只 `torch.manual_seed`,而 serial 的 `_set_seed` 还设 python/numpy 种子;
- RayPipelineExecutor / worker 侧其他 torch RNG 消耗(build_model 初始化、CostModel 等)。

## 三、能否解决:能,三修复 + 一个定位实验

| 修复 | 内容 | 效果 |
|---|---|---|
| Fix A(采样锚定) | 控制器改用专用 torch.Generator(或每步采样前按协议 seed);同步循环改为**逐个采样+逐完成更新**(与 serial 同构;流水线并发不受影响,只是控制器不再整批采样) | 消除机制 1,候选集与 serial 位级重合 |
| Fix B(种子对齐) | `_make_payload` 的 seed 改为 42 + **全局 trial 序号**(调用方已有 total_generated / trial_id) | 消除机制 2,同候选同种子可比 |
| Fix C(训练等价) | 20K 微对比实验定位机制 3:同种子同架构,serial vs pipeline,逐步 diff state_dict(首步后、首 epoch 后、终态)→ 锁定 FQN 加载或 RNG 消耗点后修复 | 消除机制 3,同种子位级同分 |

**验证方案(修复后)**:B(20K)重跑 serial / naive / naive_async 各 1 次,leaderboard 应位级一致;
再跑 D(100K)验证 12-trial 全位级。预算:3-6 个小型 run。

**对论文的意义**:修复成功则 C1 从「异步族 4/4 位级一致 + 同步族族内互证」升级为「**所有策略全 cell
位级一致**」——保真执行从经验观察变成构造性保证(protocol-guaranteed),当前「任何后端按构造复现串行
选择」的表述才字面成立。若机制 3 深挖失败,退路是保留现有表述(异步族 4/4 + 同步族互证)。

## 四、附带说明

- 与 CPU 洪水无关:leaderboard 分数与采样由种子确定,洪水只污染计时。本诊断基于已完成 run 的选择数据,有效。
- 已停掉的 run(H-naive、链 #3=H-smart_async attempt2、E cron)数据状态:选择数据仍可用(如 H naive_async
  provisional 438656),计时作废。
