---
name: reproduce-check
description: 检查代码的可复现性，确保论文实验可被他人重现
---

# 可复现性检查 Skill

确保代码和实验的可复现性（论文审稿和开源的关键）：

## 检查清单

### 1. 随机种子控制
- 检查是否设置了 `torch.manual_seed()`、`numpy.random.seed()`、`random.seed()`
- 验证 `--seed` 参数是否正确传递和使用
- 检查 GPU 确定性设置（`torch.backends.cudnn.deterministic`）

### 2. 依赖和环境
- 生成/更新 `requirements.txt` 或 `environment.yml`
- 记录 Python 版本、PyTorch 版本、CUDA 版本
- 检查是否有硬编码路径或环境依赖

### 3. 数据和预处理
- 确认数据集来源和版本
- 检查数据预处理的确定性
- 验证数据划分（train/val/test）是否固定

### 4. 配置文件
- 检查关键超参数是否记录
- 确认实验配置是否完整
- 生成实验配置模板

### 5. 文档完整性
- README 是否包含完整运行说明
- 是否有快速验证示例（如你的 synthetic 数据集）
- 是否说明了预期结果范围

输出可复现性报告和改进建议。
