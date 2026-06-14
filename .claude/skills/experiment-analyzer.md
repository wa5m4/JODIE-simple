---
name: experiment-analyzer
description: 分析实验结果，生成对比表格和可视化，用于论文写作
---

# 实验结果分析 Skill

当用户需要分析实验结果时：

1. **读取实验输出**：
   - 扫描 `outputs/` 目录下的实验结果
   - 读取 `leaderboard.csv`、`best_arch.json`、`comparison_result.json` 等
   - 读取 `timing_log.csv` 和 pipeline trace 日志

2. **对比分析**：
   - 多种子实验对比（mean ± std）
   - 不同配置对比（async vs sync、不同 stage 分配）
   - 搜索架构 vs 基线（JODIE-RNN）对比

3. **生成输出**：
   - Markdown 表格（适合快速查看）
   - LaTeX 表格（适合论文）
   - 可视化建议（使用 matplotlib/seaborn）

4. **统计检验**：
   - 计算均值、标准差、置信区间
   - 必要时进行 t-test 或 Wilcoxon 检验

输出格式示例：
```
| Method | MRR | Recall@10 | Time(s) |
|--------|-----|-----------|---------|
| Searched | 0.45±0.02 | 0.68±0.03 | 125 |
| JODIE-RNN | 0.42±0.01 | 0.65±0.02 | 98 |
```
