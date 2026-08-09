# 重新分析：Pipeline 为什么会改变架构排名

## 结论先说

现在的判断要比最初保守，但也更接近事实：

- 不是所有架构都会被 pipeline 系统性低估。
- 不是 `optimizer.state_dict()` 单点坏掉就能解释全部偏差。
- 真正的问题是 pipeline 的 epoch / stage 边界曾经改变了训练连续性，导致某些依赖动态状态的架构被更明显地扰动。

修复后，pipeline 在最近的定向验证里已经能和 serial 保持一致的架构排名；这说明之前会影响选型的那类偏差已经被压住了。

## 实验结论

### 1. 四策略对比里，serial 和 data parallel 的最终选择一致

在 `run_all` 配置家族下，历史结果里 serial 和 data parallel 的最终 `test_score` 一致，而 pipeline naive 明显更低：

- Serial: `test_score = 0.8793`
- Data Parallel: `test_score = 0.8793`
- Pipeline Naive: `test_score = 0.7382`

这说明数据并行本身不是问题来源，pipeline 才是排名偏差的来源。

### 2. pipeline smart 不是和 naive 同一种固定执行形态

smart 策略会走自动配置，最终 stage 数和分区方式可能被改写，所以它更适合单独看“pipeline smart 这条路径能跑多好”，而不是拿来和固定 stage 的 naive 做一一对应的实现对照。

### 3. 修复后的定向验证已经恢复排名一致性

在最近的验证脚本里，pipeline 和 serial 的候选架构排名已经一致。这一点很关键，因为它说明当前实现已经不再引入会改排序的系统性偏移。

## 已验证的事实

### 1. 同进程重建模型 + fresh optimizer 会和 Serial 产生可见偏差

我做了一个最小复现，比较了几种训练路径：

- `serial`：同一个模型 / optimizer 跨 epoch 持续训练
- `fresh_opt`：epoch 边界重建模型，并新建 optimizer
- `restore_opt`：epoch 边界重建模型，但恢复 optimizer state
- `same_obj`：保持对象连续，只在 epoch 边界重置运行时状态

结果是：

- `fresh_opt` 的参数差异不为 0
- `restore_opt` 的参数差异为 0
- `same_obj` 的参数差异为 0

这说明核心问题不是“state_dict 无法恢复”，而是“训练路径一旦被拆段，轨迹就会变”。

### 2. Ray 的 payload round-trip 数值是无损的

单独验证过 payload 传输后：

- `payload_roundtrip_max_diff = 0.0`

所以，把模型状态从主进程传到 Ray worker 的那一步本身没有发生明显数值损坏。

### 3. 训练轨迹的连续性曾经被打断

Serial 中：

- 模型对象持续存在
- optimizer 持续存在
- 运行时状态持续累积

Pipeline 中：

- epoch / stage 边界会重新构造执行对象
- 运行时状态需要显式传递
- optimizer state 也需要跨进程恢复

这会把原本连续的训练轨迹切成多段，对动态状态依赖强的架构影响更大。

## 根因是什么

根因是几个执行层问题叠加，不是单点故障：

1. optimizer state 跨进程映射不稳定。
	 - PyTorch optimizer state 依赖参数标识。
	 - 直接搬运旧 id / index 容易把 slot state 对错对象。

2. epoch / stage 边界的 seed 和 runtime state 传递不完整。
	 - 负采样、动态图状态、以及 stage 间连续性都可能偏移。

3. multi-epoch 调度里存在边界处理不清晰的问题。
	 - 递归式处理容易引入 payload 复用和对象别名。

4. pipeline 本身对动态状态架构更敏感。
	 - `static=off` 的架构更依赖时序记忆，最容易受边界扰动。

## 我改了什么

主要改动都在 [jodie/nas/ray_pipeline.py](jodie/nas/ray_pipeline.py)：

- 把 optimizer state 改成了基于 FQN 的传输与恢复。
	- 先把 optimizer state 从参数 id 转成参数名。
	- worker 端再按当前模型参数名反向恢复。

- 补齐了 epoch / stage 的 seed 传递。
	- 训练阶段按 `seed + epoch_offset + partition_id` 保持确定性。

- 修掉了 multi-epoch 调度中的边界问题。
	- 去掉了递归式调度，改成显式 epoch 循环。

- 增加了 payload 拷贝。
	- 避免 epoch 边界对象共享带来的别名污染。

## 改动效果如何

### 已经验证通过的点

- payload round-trip 数值无损，差异为 `0.0`。
- optimizer state 的 FQN round-trip 差异为 `0.0`。
- worker 级别的单分区、两 epoch 训练可以和 serial 对齐。
- 最近的定向验证中，pipeline 与 serial 的架构排名一致。

### 还需要保留的谨慎

- 这不代表所有未来数据集、所有搜索空间、所有随机种子都自动保证完全一致。
- pipeline smart 会走 auto-config，和 naive 的固定 stage 形态不是同一个执行面。
- 如果要做正式报告，最好把四策略结果固定到同一批次再归档。

## 对“pipeline 会低估某些架构”的更准确表述

过去的说法太绝对了。当前更稳妥的描述是：

> Pipeline 通过 epoch / stage 边界改变了训练连续性，曾经让架构评估偏离 Serial；对高度依赖动态状态的架构，这种偏差更明显。修复后，pipeline 已能恢复与 serial 一致的排序。

## 进一步整理

更完整的四策略对比和当前结果我单独整理在 [PIPELINE_RUNALL_SUMMARY.md](PIPELINE_RUNALL_SUMMARY.md)。那份文档更适合直接拿去给别人看，这份文档更适合保留为“根因分析 + 修复记录”。

## 下一步建议

1. 把 serial vs pipeline 的一致性做成回归测试，防止以后再退化。
2. 如果要继续做正式 benchmark，最好在同一次 `run_all` 里固定四策略结果再归档。
3. 若后面还想继续缩小数值微差，可以继续做更细粒度的 payload diff，但它目前已经不再表现为会改排名的系统性偏差。

