# 四种NAS策略的时间记录方式说明

## 1. Serial模式
**记录方式**: 每个trial独立记录
- **start_time**: 单个trial开始前的时间戳
- **end_time**: 单个trial完成后的时间戳  
- **duration**: end_time - start_time（单个trial的完整训练+评估时间）

**特点**: 顺序执行，时间记录最直接准确

## 2. Data Parallel模式
**记录方式**: 每个trial独立记录
- **start_time**: trial_end_time - result["time_sec"]（反推开始时间）
- **end_time**: 从search开始的累计时间
- **duration**: result["time_sec"]（单个trial的实际执行时间）

**特点**: 并行执行，但每个trial的时间独立记录

## 3. Pipeline Naive模式
**记录方式**: **以batch为单位记录**
- **start_time**: batch开始时间（batch内所有trial共享）
- **end_time**: batch结束时间（batch内所有trial共享）
- **duration**: batch_end - batch_start（整个batch的时间）

**问题**: batch内多个trial的时间戳相同，无法区分单个trial耗时

## 4. Pipeline Smart模式（修复后）
**记录方式**: 异步提交，完成时记录
- **start_time**: submit_arch()时记录
- **end_time**: poll_completed()返回结果时记录
- **duration**: end_time - start_time（从提交到完成的elapsed time）

**问题**: 异步模式下，duration包含了等待+执行时间，但由于并行，不能直接相加

## 关键差异

| 模式 | 单trial时间准确性 | wall-clock时间准确性 | 问题 |
|------|-------------------|----------------------|------|
| Serial | ✓ 准确 | ✓ 准确 | 无 |
| Data Parallel | ✓ 准确 | ✓ 准确 | 无 |
| Pipeline Naive | ✗ batch共享 | ✓ 准确 | duration不准 |
| Pipeline Smart | ⚠️ 含等待时间 | ✓ 准确 | duration包含排队 |

## 结论

**对比总NAS时间应该使用**: timing_log最后一行的`end_time_s`（从搜索开始到完成的总时间）

**不应该使用**: duration之和，因为各模式的duration含义不同
