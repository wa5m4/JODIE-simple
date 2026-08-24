# 消融实验运行说明(服务器版)

**实验名称**:保真度三修复关闭消融(Fidelity Fixes OFF)
**目的**:论文引言段 4 case study 的机制归因判据——在**最终配置(serial 模式)**下,仅关闭三处搜索态修复,验证 Pipeline Naive 是否会重新选出错误架构。
**日期**:2026-08-24

---

## 1. 实验逻辑(为什么跑这个)

修复前(2026-08-06,tbatch 时代)首次对比中,Pipeline Naive 选出 402K 全开架构、test_score=0.7000,而 Serial 选出 133K、test_score=0.8561。但那次跑法混杂了两类因素:

- **训练态偏差**(tbatch 每 epoch 仅 ~14 步,epoch 边界 optimizer 重建的动量断裂被放大)——旧证据,最终 serial 配置下已证明无效(单架构训练 diff=0);
- **搜索态三修复**(①trial 种子纪律 ②RNG 保存/恢复 ③controller 逐 trial 更新)——FINAL_REPORT 盖章的根因。

本实验只关掉三处搜索态修复、其余配置与现有四策略数据完全一致,回答:**"搜索态三修复单独是否足以导致选错架构?"**

### 判据(跑完对照这张表)

| 结果 | 含义 | 下一步 |
|------|------|--------|
| 选出 402K 全开架构,test≈0.7000 | 三修复效应独立成立 → 段 4 机制归因干净 | 段 4 定稿,消融进实验表 C2 |
| 选出 133K 全关架构,test≈0.8561 | 三修复单独不够致命 → 段 4 必须改写(tbatch 复合故事) | 重写段 4,讨论 tbatch 因子 |
| 选出其他架构 | 同上,归因不成立 | 同上 |

---

## 2. 服务器部署(方案 A:检出消融分支)

**已采用方案 A**:消融代码(4 处改动)已推送到 GitHub,服务器检出分支即可,**不需要手动改任何代码**。

```bash
cd <项目目录>
git fetch origin
git checkout ablation/fidelity-off    # 消融分支:4 处改动 + 本说明
git status                            # 应干净;若有本地改动先 git stash 保存
```

检出后验证:

```bash
git log --oneline -1        # 应看到 90245bf(消融实验提交)或包含它的 merge 提交
sed -n '/ENABLE_STRATEGIES/,/]/p' run_all.py   # 应只出现 "pipeline_naive"
```

> 服务器原有的 refactored 分支保持不动;跑完 `git checkout refactored` 即自动恢复(见第 8 节)。

---

## 3. 应用改动(共 4 处,只改这些,别的任何配置都不许动)

> **方案 A 下本节已由分支完成,直接跳到第 4 节核对清单。** 本节仅作改动内容存档,供"手动改回"或对照参考。

### 改动 1:run_all.py —— 只跑 pipeline_naive

把 `ENABLE_STRATEGIES` 改成:

```python
ENABLE_STRATEGIES = [
    "pipeline_naive",      # 消融:fidelity 三修复关闭,serial 配置,MOOC
]
```

### 改动 2:jodie/nas/trainer.py 评估处(约 462 行)—— 删掉 RNG 保存/恢复两行

删掉这两行(连同注释):

```python
                # ★ 修复：保存/恢复 RNG 状态，避免 build_model 污染下一个 trial 的初始化
                rng_state = torch.get_rng_state()
                model = build_model(config)
                torch.set_rng_state(rng_state)
```

改成只剩:

```python
                model = build_model(config)
```

### 改动 3:jodie/nas/trainer.py 搜索处(约 951 行)—— controller 更新改回批量版

把这整段(含注释):

```python
                # ★ 修复：逐 trial 更新 controller（与 serial 路径一致）
                # 必须用 compute_logprob 重新计算 logprob，因为上一个 trial 的
                # optimizer.step() 已修改 logits，batch 中后续的原始 logprob
                # computation graph 会失效（inplace version mismatch）
                for arch_cfg, result in zip(arch_batch, batch_results):
                    if hasattr(controller, "reinforce_step") and hasattr(controller, "compute_logprob"):
                        logprob = controller.compute_logprob(arch_cfg)
                        controller.reinforce_step(logprob, result["score"])
```

替换成:

```python
                batch_samples = [
                    (logprob, result["score"])
                    for logprob, result in zip(logprobs, batch_results)
                    if logprob is not None
                ]
                if batch_samples and hasattr(controller, "reinforce_step_batch"):
                    controller.reinforce_step_batch(batch_samples)
                else:
                    for logprob, score in batch_samples:
                        if hasattr(controller, "reinforce_step"):
                            controller.reinforce_step(logprob, score)
```

### 改动 4:jodie/nas/ray_pipeline.py `_make_payload`(约 1235 行)—— 删掉设种子一行

删掉这两行(含注释):

```python
        # ★ 修复：设种子后建模型，保证每个 trial 初始权重独立且可复现
        torch.manual_seed(seed)
```

改成只剩:

```python
        model = build_model(config)
```

> 备选方案:也可直接 `git apply ablation_fidelity_off.patch`(补丁文件与本文档同目录),效果相同。

---

## 4. 配置核对清单(改动后,启动前逐项确认 run_all.py)

| 配置项 | 必须是 | 说明 |
|--------|--------|------|
| `DATASET` | `"public_csv"` | 数据文件 mooc.csv(39.5 MB) |
| `BATCH_MODE` | `"serial"` | 最终配置,与四策略数据一致 |
| `SEARCH_SPACE` | `"rnn_only"` | 与四策略数据一致 |
| `COARSE_TRIALS` / `COARSE_EPOCHS` | `50` / `2` | 与四策略数据一致 |
| `RERANK_TOP_K` / `RERANK_EPOCHS` | `8` / `5` | 与四策略数据一致 |
| `ENABLE_STRATEGIES` | 只含 `"pipeline_naive"` | 改动 1 |
| 其他一切 | **不动** | 对照纪律:唯一变量是"三修复在不在" |

⚠️ 千万不能顺手改搜索空间、trials 数、epochs、数据集——那样消融就废了(变量不唯一,结果无法归因)。

---

## 5. 启动

```bash
nohup python run_all.py > run_ablation_fidelity_off.log 2>&1 &
# 若日志中文乱码,改用:
# PYTHONIOENCODING=utf-8 nohup python run_all.py > run_ablation_fidelity_off.log 2>&1 &
```

**启动后 1 分钟内检查日志**,预检应显示:

```
[预检] ✓ CUDA 可用: True, 可见GPU数: 8
[预检] ✓ Pipeline Naive: 3 stages × [1, 1, 1] workers (固定)
...
启用策略: ['pipeline_naive']
```

**预计时长**:约 3~5 小时(参考:修复后 Pipeline Naive 总耗时 16,783s ≈ 4.7h;服务器 8 卡可能更快)。

---

## 6. 跑完看什么

结果目录:`results/<时间戳>/pipeline_naive/`

关键文件:
- `summary.txt` —— 看 `参数量`、`test_score`、`模型`、`架构配置`(对照第 1 节判据表)
- `best_arch.json` —— 最终选出架构的完整配置
- `leaderboard.csv` —— 所有 trial 的分数轨迹(可选,写论文用)

---

## 7. 带回来的东西

1. `results/<时间戳>/pipeline_naive/summary.txt`
2. `results/<时间戳>/pipeline_naive/best_arch.json`
3. `results/<时间戳>/pipeline_naive/leaderboard.csv`
4. `run_ablation_fidelity_off.log`(完整日志)
5. 一句结果总结:最终参数量 / test_score / 架构开关状态

---

## 8. 跑完恢复(重要!)

跑完后**立即把第 3 节的四处改动改回去**(恢复三修复 + ENABLE_STRATEGIES 改回原值),或直接 `git checkout -- run_all.py jodie/nas/trainer.py jodie/nas/ray_pipeline.py`,并用 `git status` 确认服务器代码回到干净状态,再跑其他实验。

否则之后的正式实验会在"修复被关掉"的代码上跑,数据全废。

**方案 A 更简单**:直接 `git checkout refactored` 切回原分支即可(消融分支原封不动保留,随时可切回来)。
