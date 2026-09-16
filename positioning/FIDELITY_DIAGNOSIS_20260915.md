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

## 五、修复实施(2026-09-15 晚,用户批准,基线 commit 3576127)

**已落代码**:

1. **Fix A** — controller.py:RLGraphNASController 采样改用专用 `torch.Generator`
   (`_rng.manual_seed(seed)`),`sample_arch_with_logprob` 用 `torch.multinomial(probs, 1,
   generator=self._rng)`。控制器采样流与进程内任何其他 torch 全局 RNG 消耗脱钩。
   `torch.manual_seed(seed)` 保留(维持进程其余部分的既有确定性行为)。
2. **Fix B** — ray_pipeline.run_train_only 增加 `trial_ids` 参数(seed = 42 + 全局 trial 序号);
   trainer.evaluate_arch_pipeline 透传;同步批循环传 `[total_generated+i]`,rerank 传
   `[10000+i]`(对齐 serial 的 rerank_seed,之前少 10000 偏移)。异步池 submit_arch 本来就是
   全局计数,无需改。
3. **Fix C** — `_make_payload` 改设 python/numpy/torch 三种子(对齐 serial `_set_seed`);
   `run_train_stage_batch` 补 `model.train()`。

**两处结论修正**(深读代码后):

- **异步池的 eval ≈ 方案C**:`run_eval_stage_batch` 在单个 worker 内按序评估全部 eval 分区、
  状态跨分区累积(frozen=False 活评估)→ 语义与 driver 侧全数据评估等价,eval 不是分叉源。
- **同步臂的批采样协议无法逐位复现 serial**:批内 4 个候选全部在批首 θ₀ 下采样,而 serial
  逐 trial 更新后采样——这是批同步执行(同步臂的论文定位与其加速比来源)的结构性属性,不是 bug。
  因此保真度结论维持三档:serial 参考;naive(同步流水线)分数经 Fix B/C 变为诚实可比、
  轨迹可复现(与 serial 是否逐位一致由重跑如实记录);naive_async(异步流水线)= serial 位级
  一致(4/4,经验性,经 Fix A 后对 RNG 漂移免疫、逐构造可复现)。

4. **Fix D(2026-09-15 晚)** — 异步池全局 epoch-major 编排。旧池 `run_train_stage_batch`
   内部 `for epoch in range(epochs)` 在每个 stage 局部跑多轮,更新顺序 p0e0,p0e1,p1e0…
   ≠ serial 全局 epoch-major(p0e0…pNe0, p0e1…pNe1)——旧 D async 8896 同种子分叉
   (0.500769 vs serial 0.476019)的代码级原因。改动:
   - `_drain_pool` train 调度 `num_epochs=1`(每次 stage 只跑 1 个 epoch)+
     `seed_epoch_offset=已完成全局 epoch 数`;
   - `poll_completed` 最后 train stage 完成分支:全局 epoch 计数 +1,未满 `epochs` 则构造
     边界 payload(runtime_state=None 触发 stage 0 重置嵌入缓冲、graph_state 换全新空模板、
     model_state_dict 与 optimizer_state 跨 epoch 保留)回 `_pool_train_pending[0]`,满了进 eval;
   - `run_eval_stage_batch` eval 从全新空图模板开始(此前恢复训练后图快照,混合模型 eval
     邻居集与 serial 不同);
   - `run_train_only` / `_run_train_pipeline` / `_run_train_eval_pipeline` 三处 epoch 边界
     payload 的 graph_state 同样重置为空模板(此前跨 epoch 携带累积图;五臂 cell 全为
     rnn_only 无图,属防御性正确)。

**验证(微对比实验,positioning/fidelity_microcheck.py)**:同架构(8896,取自 D serial 真值)、
同种子(42),serial 路径 vs 执行路径,比较 val score 位级 + 终态 state_dict 指纹。
- **B cell 同步路径(run_train_only,方案C)已通过**:serial 0.5139392453105059 逐位复现;
  state_dict 指纹 a756412612f120ae 两路径一致 → 机制 2/3(同步路径)消除。
- **B cell 异步池路径(--pool,start_persistent_pool/submit_arch/poll_completed)**:
  验证 Fix D 全局 epoch-major 后池训练+池内评估 ≡ serial 位级(2026-09-15 晚跑,GPU 1,7)。
**注意:机器 CPU 洪水未退(load ~180,dongyu yolov13),流水线 worker 慢但可完成;
选择数据由种子确定,不受洪水影响。**

**修复后全矩阵需重跑**:Fix A 换专用 Generator 后控制器采样流与旧全局流不同 → 旧选择
数据全部作废。保真结论三档不变(serial 参考;naive 同步批协议分数诚实可比、轨迹可复现;
naive_async = serial 位级),重跑按 B(20K)位级验证 → D(100K)全位级验证的顺序执行。

---

## 2026-09-16 追加:phase5 中断 + DP 负采样第二层修复

**B-dp 保真门槛失败(结果目录 20260916_101424)**:DP 选出 (128,linear,off,on) 133888 /
val 0.7658930176060056 / test 0.6686594469069215,serial 真值为 (128,linear,on,on) 402176 /
val 0.7655034809596591 / test 0.6999869339656702。50/50 trial 全分叉;trial 0 同架构
(rnn)serial 0.6678 vs DP 0.3643(Δ=0.30)。根因两层:
1. **可修(已修,commit c53f3bc)**:worker `train_chunk` 用 `np.random.default_rng(None)`
   现场抽负样本(完全无种子、连确定性都没有),serial 用数据加载期预计算负样本
   `neg_samples_by_epoch`(seed=42+epoch*100000,public_dataset.py)。修复:train_chunk 加
   `epoch_idx`、负样本「预计算优先 → 同公式种子回退」三态镜像 loops.py:156-168、
   `_run_trial` 注入 trial seed(42+trial_idx,与 serial payload.seed 同源)。
   冒烟 positioning/dp_negfix_smoke.py 六项全过(预计算路径零 RNG 调用、数值流入损失、
   回退确定性、种子公式、epoch 区分、两路径区分)。
2. **固有(不可修)**:微批梯度平均+一步 Adam(批量梯度步)≠ serial 逐事件在线更新。
   09-10 旧数据显示批量步本身偏差约 Δ0.03,负采样异流可能占大头——B-dp 复测定夺。

**smart_sync 免于同类问题(代码级结论)**:「1 stage × 2 train workers」是 payload 级并行
(每个 payload 整体交给一个 worker,顺序训练本 stage 全量 partitions;ray_pipeline.py
`_run_train_pipeline`→`_single_epoch` 706-717),非 DP 式梯度切分平均;与金丝雀位级验证的
naive 同同步路径、更简单特例。修复范围仅 data_parallel.py。

**CPU 洪水中断**:09-16 白天 loadavg 228/48 核,B-smart_sync 8/50 trials 耗时 6.7h
(金丝雀全量 2.76h,慢 12-18×),用户批准 kill 链条。B-dp/B-smart_sync 旧日志改名
`*_pre_fix`/`*_partial`;夜间负载 <96 自动重启 phase5(cron 17e890af),B-dp 复测为门槛:
**通过 → 续跑全矩阵;仍翻盘 → DP 降级「近似后端」:Section 6 只报加速比、并列报保真度
限制(见 DP_FIDELITY_ANALYSIS.md §四预案)。**
