# overleaf草稿 使用说明

本目录是论文的 Overleaf 版草稿,已按 `示例.tex`(VLDB / ACM acmart 模板)**规范化**。

## 文件

| 文件 | 说明 |
|---|---|
| `示例.tex` | 你的 VLDB 模板参考(未改动,保留) |
| `main.tex` | 论文草稿:**只填了摘要 v4 + 引言 8 段**(2026-08-28 讨论后最新版);第 2-7 节待阶段 2-3 逐节补 |
| `README.md` | 本说明 |

## 怎么编译

1. Overleaf:新建项目 → **Blank**,上传 `main.tex` + 模板自带的 `acmart.cls`(从你示例项目里下载,或新建项目时选 ACM 官方模板再覆盖)。
2. Recompile。第一页 = 标题 + 作者 + 摘要 + 引言。
3. `\cite{}` 显示 `[?]` 是**正常的**——`sample.bib` 还没补,见下方占位清单。

## 与 示例.tex 的不同(规范化时改动的三处)

1. **作者块、页脚块、文档类**:原样保留示例的写法
2. **卷号**:`\vldbvolume{14}` → `{19}`(当前 2025-2026 卷;示例注释说"投稿时用当前卷号")
3. **公开链接**:`\vldbavailabilityurl` 清空(示例里的 `URL_TO_YOUR_ARTIFACTS` 会原样打印到页脚)

## 占位清单(按优先级)

| 占位 | 位置 | 怎么补 |
|---|---|---|
| **摘要两处同步点**(挂账,等批改) | main.tex 摘要里两处 `% TODO(挂账同步点)` | 批改摘要 v4 时定(第二轮批改第 3/4 点),定完同步 `../draft_v2_摘要.md` |
| 引用 `sample.bib` | 全文 `\cite{jodie/tgn/graphnas/pygt/cacheg/pipad/esdg/dynahb/mooc}` 共 9 条 | 新建 sample.bib |
| Challenge III 的 X 倍数字 | 引言 Challenge III 段 | 定位实验的分区统计顺手取 |
| DepTGL 全称(标题) | `\title{}` | 待确认问题 #1 |
| X datasets / Y model families | 摘要末句 | 完整实验表(阶段 3) |
| 致谢 `\begin{acks}` | 正文末尾 | 基金信息确定后补 |

## 写作铁律提醒(填后续章节时不要破)

- 段 4 数字一个不能动:0.8561 / 0.6014 / 0.2547 / 0.96 / 0.62 / 0.9335 / 133K / 147K
- 处置原则:t-Batch/stale_batch 不进摘要、不进贡献列表;t-Batch 只在 Section 5 出现一次("integrated from JODIE")
- Challenge I 的"现有方法不够"句保持泛化写法,不点名、不引用
- 承诺级别:C1 = "任何后端复现串行搜索的选择"(不承诺候选级分数精确相等;位级一致 0.856121275963994 是实验章细节)
