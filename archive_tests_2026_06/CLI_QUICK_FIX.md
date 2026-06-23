# 🔧 CLI 参数格式快速修复指南

## ❌ 常见错误和 ✅ 正确用法

### 错误 1：参数格式错误
```bash
❌ bash scripts/run_comparison_3way.sh -gpu-list 0,1,2
❌ bash scripts/run_comparison_3way.sh 0,1,2

✅ bash scripts/run_comparison_3way.sh --gpu-list 0,1,2
✅ bash scripts/run_comparison_3way.sh -g 0,1,2  (短形式)
```

### 错误 2：参数值中有空格
```bash
❌ --gpu-list "0, 1, 2"      # 有空格！
❌ --seeds "42, 43, 44"      # 有空格！

✅ --gpu-list 0,1,2          # 无空格
✅ --seeds 42,43,44          # 无空格
```

### 错误 3：路径错误
```bash
❌ bash run_comparison_3way.sh
❌ bash scripts/run_comparison_3way

✅ bash scripts/run_comparison_3way.sh
```

---

## 📚 参数快速参考

### 长形式（完整）
```bash
--gpu-list       GPU ID 列表
--space          搜索空间
--max-events     事件数
--time-budget    时间预算
--trials         试验上限
--epochs         epoch 数
--seeds          种子列表
--output-dir     输出目录
--help           帮助信息
```

### 短形式（快捷）
```bash
-g               GPU ID 列表
-m               事件数
-t               时间预算
-e               epoch 数
-s               种子列表
-o               输出目录
-h               帮助信息
```

---

## 🚀 立即可用的命令

### 最简单（1 个参数）
```bash
bash scripts/run_comparison_3way.sh -g 0,1,2
```

### 快速测试（3 个参数）
```bash
bash scripts/run_comparison_3way.sh -g 0,1,2 -m 5000 -t 60
```

### 完整配置（长形式）
```bash
bash scripts/run_comparison_3way.sh \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --max-events 20000 \
    --time-budget 600 \
    --trials 50 \
    --epochs 3 \
    --seeds 42,43,44
```

### 完整配置（短形式）
```bash
bash scripts/run_comparison_3way.sh \
    -g 0,1,2,3,4,5,6,7 \
    -m 20000 \
    -t 600 \
    -s 42,43,44 \
    -e 3
```

---

## 💡 关键提示

1. **始终使用双横杆 `--` 或单横杆 `-`（对应短形式）**
   - ❌ 不要混用单横杆和双横杆的参数名
   - ✅ `--gpu-list 0,1,2` 或 `-g 0,1,2`

2. **参数值之间不要有空格**
   - ❌ `0, 1, 2` 
   - ✅ `0,1,2`

3. **路径要正确**
   - ✅ `bash scripts/run_comparison_3way.sh`
   - ❌ `bash run_comparison_3way.sh`

4. **查看帮助**
   ```bash
   bash scripts/run_comparison_3way.sh --help
   bash scripts/run_comparison_3way.sh -h
   ```

---

## 📊 参数值说明

| 参数 | 值类型 | 示例 |
|------|--------|------|
| `--gpu-list` | 逗号分隔 | `0,1,2,3` |
| `--max-events` | 整数 | `5000`, `20000` |
| `--time-budget` | 整数（秒） | `60`, `300`, `1200` |
| `--epochs` | 整数 | `1`, `3`, `5` |
| `--trials` | 整数 | `10`, `50`, `999` |
| `--seeds` | 逗号分隔 | `42`, `42,43`, `42,43,44` |
| `--space` | 字符串 | `rnn_only`, `small`, `large` |
| `--dataset` | 字符串 | `public_csv`, `synthetic` |

---

## ✅ 修复验证

试试这个命令验证一切正常：

```bash
# 显示配置（不执行）
bash scripts/run_comparison_3way.sh -g 0,1,2 -m 1000
```

如果看到类似这样的输出（配置展示），说明参数正确：
```
╔════════════════════════════════════════════════════════════════════════╗
║     🔬 NAS 四方对比：Serial vs DP vs Pipeline-Naive vs Pipeline-Smart ║
║  📊 数据集:
│    - 类型：public_csv
│    - 事件数：1000
│  🖥️  硬件配置:
│    - GPU 列表：0,1,2
│    - GPU 数量：3
```

---

**现在您可以正确使用脚本了！** 🎉
