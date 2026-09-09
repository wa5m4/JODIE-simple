# 定位实验运行工具(配合 POSITIONING_GUIDE.md)

一套把指南 §2/§3 的手工步骤自动化的小工具。**不改实验本身,只负责:配置落盘 → nohup 启动 → 自动预检 → 顺序推进。**

## 文件

| 文件 | 作用 |
|---|---|
| `configs.py` | 7 个 (cell, strategy) 的配置表,与指南 §2 表格一一对应(固定值:serial/rl/seed42/partition2000/rnn_only) |
| `apply_config.py` | 把某个 run 的配置写入 run_all.py 顶部配置区(`--apply` 写盘,默认 dry-run) |
| `precheck.py` | 启动后 90s 自动预检:对照最新 `results/<ts>/config.json` 快照 + log 标志行,不一致退出码 1 |
| `launch.sh` | 单个 run 启动器(不做 GPU 检查,手动用) |
| `chain_all.sh` | 7 个 run 顺序链条:每次启动前查 GPU 空闲(显存<2GiB 且利用率<20%)+ 磁盘(>5GB),预检失败杀进程中止 |

## 用法(仓库根目录)

```bash
# 单个 run(D cell, smart 策略):
bash positioning/launch.sh D smart

# 全链条(nohup 常驻,GPU 空闲自动开跑,建议):
nohup bash positioning/chain_all.sh > positioning/chain_all.log 2>&1 &

# 从中途续跑(如第 3 个 run = D dp):
bash positioning/chain_all.sh 3

# 只看配置不动文件:
python positioning/apply_config.py --cell B --strategy naive
```

启动顺序 = `configs.RUNS` 顺序(2026-09-07 起):B smart→naive→dp → D smart→naive→dp→serial。
GPU 分配:B 固定 "5,6"(0/1 被长期占用);D 为 "auto"——链条启动前动态选任意 3 张空闲卡,不再绑定 0,1,2(选中的卡号会随 log 记录)。
日志名: `run_pos_cell{cell}_{strategy}.log`;结果仍在 `results/<时间戳>/`。

**铁律**:计时实验不共享 GPU。chain_all.sh 已内置检查;手动 launch.sh 前务必自己确认(launch.sh 不支持 auto 选卡,D 的手动启动需加 `--gpu-list`)。
