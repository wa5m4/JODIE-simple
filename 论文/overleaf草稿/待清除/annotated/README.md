# 说明项目(Annotated Project)——带批注的代码镜像

> 本目录是真实代码的"带批注镜像":目录结构与 `jodie/` 完全一致,每个源文件对应一个 `.md` 说明文件。
> **阅读方式**:打开说明文件,旁边开一个窗口打开对应源码对照读。

## 标注约定

| 标注 | 含义 |
|------|------|
| 🔑 | **核心逻辑**:贴出源代码 + 逐段白话解释。必须读懂 |
| 📖 | **了解即可**:文字描述功能,不贴代码。知道"它干什么、为什么存在"即可 |
| ⏭️ | **可跳过**:一句话带过,不影响主线 |
| ❓ | **检查题**:读完做,做不出回头重读标注部分 |

## 阅读顺序(按数据流自底向上)

| 阶段 | 模块 | 说明文件 | 状态 |
|------|------|---------|------|
| 1 | 数据层 | [synthetic.md](jodie/data/synthetic.md)、[temporal_partition.md](jodie/data/temporal_partition.md)、[public_dataset.md](jodie/data/public_dataset.md) | ✅ 已生成 |
| 2 | 模型层 | [jodie_rnn.md](jodie/models/jodie_rnn.md)、[gnn_encoder.md](jodie/models/gnn_encoder.md)、[hybrid_jodie.md](jodie/models/hybrid_jodie.md)、[factory.md](jodie/models/factory.md) | ✅ 已生成 |
| 3 | 训练层 | batching.md、loops.md、metrics.md | ⏳ 待生成 |
| 4 | NAS 层 | search_space.md、controller.md、trainer.md | ⏳ 待生成 |
| 5 | 执行层 | data_parallel.md、ray_pipeline.md、config_optimizer.md | ⏳ 待生成 |
| 6 | 基线与入口 | baseline/official_jodie.md、入口 search.py / train.py / run_all.py | ⏳ 待生成 |

## 使用建议

1. 按阶段顺序读,每个文件先看开头"定位",再按 🔑 精读,📖 扫读
2. 做完文件末尾的 ❓ 检查题,把答案发给 Claude 批改,**通过后再进下一文件**
3. 全部读完后,这份说明就是你写论文 Method 章节的素材库

## 对应关系说明

- 源码在仓库根目录 `jodie/` 下,说明文件在 `annotated/jodie/` 下,路径一一对应
- 说明文件里标注了源码行号,如"见 [synthetic.py:13](../jodie/data/synthetic.py#L13)"
- 论文关键点会在说明中用 📄 标注,写论文前把所有 📄 串起来就是实验方法
