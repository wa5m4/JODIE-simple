# 单因素消融实验运行说明(服务器版)

**实验名称**:保真度三修复 · 单因素消融(每次只关一个修复)
**目的**:检验段 4 第 5-6 句的两个断言——"not a single large error" 和 "Each deviation is negligible in isolation"。已完成的"全关消融"(402K/0.7000 复现)只证明三个因素**整体**足以导致失败;本实验逐个检验每个因素**单独**的作用。
**日期**:2026-08-24

---

## 0. 核心原则:不要只看最终选出的架构

**即使单关某因素后最终选出的架构与修复全开时相同(133K/0.8561),搜索过程也可能已经被干扰**——采样序列变了、分数被污染了、只是恰好终点没翻。所以判据分三层:

| 层 | 观察什么 | 回答什么 |
|---|---|---|
| 第一层 · 终点 | 最终架构签名 + test_score | 该因素单独是否致命 |
| 第二层 · 轨迹 | 50 个 coarse 架构采样序列、top-8 rerank 集合、leaderboard 分数分布(与基准对比) | 搜索路径是否被扰动 |
| 第三层 · 评分 | 两次运行中都出现过的**同一架构**,val 分数差多少 | 评分本身是否被污染(还是只改了采样路径) |

三层结论组合起来才决定段 4 句子怎么改(见第 6 节判读表)。

---

## 1. 三个分支(每次只关一个修复,其余两个保持开启)

| 分支 | 关闭的修复 | 验证 commit |
|------|-----------|-------------|
| `ablation/f1-seed-off` | ① trial 种子纪律(`ray_pipeline.py` 不设种子) | `1c89fb3` |
| `ablation/f2-rng-off` | ② RNG 保存/恢复(`trainer.py` 评估处) | `b22135a` |
| `ablation/f3-batch-off` | ③ controller 逐 trial 更新(`trainer.py` 搜索处) | `309ece8` |

三者的其余配置与全关消融完全一致:serial、rnn_only、50×2 + 8×5、MOOC、`pipeline_naive`。

---

## 2. 服务器部署与运行(一条链式命令,依次跑完三个)

```bash
cd <项目目录>
git fetch origin
nohup bash -c '
git checkout ablation/f1-seed-off && PYTHONIOENCODING=utf-8 python run_all.py > run_f1_seed_off.log 2>&1; echo "=== F1 END ==="
git checkout ablation/f2-rng-off && PYTHONIOENCODING=utf-8 python run_all.py > run_f2_rng_off.log 2>&1; echo "=== F2 END ==="
git checkout ablation/f3-batch-off && PYTHONIOENCODING=utf-8 python run_all.py > run_f3_batch_off.log 2>&1; echo "=== F3 END ==="
git checkout refactored; echo "=== ALL DONE ==="
' > run_factors_master.log 2>&1 &
```

- 每个约 4.5h,三个依次约 **15h**(一个晚上)
- 某个失败不影响后两个(没有 set -e);`run_factors_master.log` 里按顺序打 === 分隔标记,哪个失败一目了然
- 启动后 1 分钟内看 `run_f1_seed_off.log` 预检(应显示 8 GPU、serial、pipeline_naive),再确认 `run_factors_master.log` 里没有报错

---

## 3. 跑完带回来的东西

### 3.1 每个运行(×3)

从各自的日志里找到结果目录时间戳:`grep "结果目录" run_f1_seed_off.log`,然后带回:

1. `results/<时间戳>/pipeline_naive/pipeline_naive/summary.txt`
2. `results/<时间戳>/pipeline_naive/pipeline_naive/best_arch.json`
3. `results/<时间戳>/pipeline_naive/pipeline_naive/leaderboard.csv`
4. `run_f1_seed_off.log` / `run_f2_rng_off.log` / `run_f3_batch_off.log`(完整)
5. `run_factors_master.log`

### 3.2 基准数据(修复全开,必需!)

三层观察需要对照基准——**修复全开**的 pipeline_naive 运行:

- `results/20260811_204240/pipeline_naive/`(对应本地 `run_all_fix_full.log`,2026-08-11 20:42 启动)
- 带回来:`leaderboard.csv`(若目录已不在服务器上,找 08-11 后含 pipeline_naive 的最新结果目录,并说明用哪个替代)

### 3.3 一句总结

每个运行一行:**f1/f2/f3 → 最终参数量 / test_score / 结果目录时间戳**。

---

## 4. 判据(三层判读表)

### 第一层:终点(每个单关运行)

| 最终架构 + test | 初步含义 |
|---|---|
| 133K / 0.8561 | 该因素单独**不致命** → 继续看第二三层 |
| 402K / 0.7000 | 该因素单独**致命** → 段 4 第 5 句 "not a single large error" 必须改写 |
| 其他架构 | 该因素单独改变了终点 → 同上,句子改写 |

### 第二层:轨迹(对比基准 leaderboard)

- 50 个 coarse 架构的**采样序列**与基准的重合度(位置是否错开、集合交集多大)
- top-8 rerank 集合与基准的交集
- leaderboard 的 val 分数分布(均值/中位/最高)

### 第三层:评分(对比基准 leaderboard)

- 基准与单关运行**都出现过的同一架构**,val 分数差多少(≈0 还是明显偏离)

### 组合判读

| 终点 | 轨迹 | 评分 | 结论 → 对段 4 的影响 |
|---|---|---|---|
| 133K | 与基准一致 | 一致 | 该因素在此配置下无影响 → 句子考虑删掉/降级该因素 |
| 133K | 有扰动 | 任意 | 有影响但不致命 → "negligible in isolation" 措辞需斟酌(受扰≠可忽略) |
| 133K | 一致 | 不一致 | 评分被污染但没翻终点 → 同上,并记录污染幅度 |
| 402K / 其他 | — | — | 单独致命 → 句子改写 |

具体数值分析(重合度、分数差阈值、句子措辞)拿到数据后由 Claude 计算并给出改法,你在服务器上只需**把文件带回来**。

---

## 5. 跑完恢复

链式命令末尾已自动 `git checkout refactored`。验证:

```bash
git status                # 应干净(或只有 untracked 结果文件)
grep -A2 ENABLE_STRATEGIES run_all.py | head -4   # 应显示 "pipeline_smart"
```

三个消融分支原封不动保留在 GitHub 上,随时可重跑。

---

## 6. 时间线备忘

- 全关消融(已完成):07fe26e,402K/0.7000 → 三因素整体效应成立
- 单关消融(本次):f1/f2/f3 → 逐因素检验,决定段 4 第 5-6 句逐条措辞
