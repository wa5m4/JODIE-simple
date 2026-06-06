#!/usr/bin/env python3
"""
完整的NAS搜索和评估流程
- 四种执行模式的NAS搜索
- 实时进度日志
- 自动重训练最优架构
- 生成综合对比报告
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime


def run_nas_search(mode, args, output_base):
    """运行单个执行模式的NAS搜索"""
    print(f"\n{'='*80}")
    print(f"开始NAS搜索: {mode}")
    print(f"{'='*80}")

    output_dir = output_base / f"{mode}_tbatch"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 转换eval_mode为frozen参数
    eval_frozen = "true" if args.eval_mode == "offline" else "false"

    # 构建搜索命令
    cmd = [
        "python", "-u", "search.py",
        "--dataset", "public_csv",
        "--local-data-path", args.data_path,
        "--max-events", str(args.max_events),
        "--train-ratio", "0.7",
        "--val-ratio", "0.1",
        "--feature-dim", "8",
        "--lr", "0.001",
        "--k", "10",
        "--seed", str(args.seed),
        "--coarse-trials", str(args.trials),
        "--coarse-epochs", str(args.epochs),
        "--batch-mode", "tbatch",
        "--train-batch-size", "32",
        "--gpu-list", args.gpu_list,
        "--output-dir", str(output_dir),
        "--eval-frozen", eval_frozen,
    ]

    # 添加执行模式特定参数
    if mode == "serial":
        cmd.extend(["--execution-mode", "serial"])
    elif mode == "data_parallel":
        cmd.extend([
            "--execution-mode", "data_parallel",
            "--data-parallel-workers", "3",
            "--data-parallel-visible-gpus", args.gpu_list,
        ])
    elif mode == "pipeline_naive":
        cmd.extend([
            "--execution-mode", "ray_pipeline",
            "--pipeline-mode", "naive",
            "--num-pipeline-stages", "3",
            "--pipeline-stage-train-workers", "1,1,1",
        ])
    elif mode == "pipeline_smart":
        cmd.extend([
            "--execution-mode", "ray_pipeline",
            "--pipeline-mode", "smart",
            "--num-pipeline-stages", "1",
            "--pipeline-stage-train-workers", "3",
        ])

    print(f"命令: {' '.join(cmd)}")
    print(f"输出目录: {output_dir}")
    print(f"\n{'─'*60}")
    print(f"开始搜索... (实时日志)")
    print(f"{'─'*60}\n")
    import sys
    sys.stdout.flush()

    # 运行搜索 - 使用实时输出
    start_time = time.time()
    result = subprocess.run(cmd, text=True, bufsize=1)
    search_time = time.time() - start_time

    print(f"\n{'─'*60}")
    print(f"{mode} 搜索完成")
    print(f"{'─'*60}")
    print(f"耗时: {search_time:.2f}秒 ({search_time/60:.1f}分钟)")
    sys.stdout.flush()

    if result.returncode != 0:
        print(f"错误: {mode} 搜索失败")
        return None, search_time

    print(f"\n{mode} 搜索完成，耗时: {search_time:.2f}秒")

    # 读取最优架构
    best_arch_file = output_dir / "best_arch.json"
    if not best_arch_file.exists():
        print(f"警告: 未找到 {best_arch_file}")
        return None, search_time

    with open(best_arch_file) as f:
        best_arch = json.load(f)

    return best_arch, search_time


def retrain_best_arch(mode, best_arch, args, output_base):
    """用Serial T-Batch模式重训练最优架构"""
    print(f"\n{'='*80}")
    print(f"重训练最优架构: {mode}")
    print(f"{'='*80}")

    config = best_arch["config"]
    retrain_dir = output_base / f"{mode}_retrain"
    retrain_dir.mkdir(parents=True, exist_ok=True)

    # 转换eval_mode为frozen参数
    eval_frozen = "true" if args.eval_mode == "offline" else "false"

    # 构建重训练命令
    cmd = [
        "python", "-u", "train_single_arch.py",
        "--model", config["model"],
        "--embedding-dim", str(config["embedding_dim"]),
        "--batch-mode", "tbatch",
        "--train-batch-size", "32",
        "--dataset", "public_csv",
        "--local-data-path", args.data_path,
        "--max-events", str(args.max_events),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed + 20000),
        "--output-dir", str(retrain_dir),
        "--eval-frozen", eval_frozen,
    ]

    # 添加可选架构参数
    if config.get("time_proj") and config["time_proj"] != "off":
        cmd.extend(["--time-proj", config["time_proj"]])
    if config.get("memory_cell"):
        cmd.extend(["--memory-cell", config["memory_cell"]])

    print(f"架构: {config['model']}, {config['embedding_dim']}-dim, time_proj={config.get('time_proj', 'off')}")
    print(f"命令: {' '.join(cmd)}")
    print(f"\n{'─'*60}")
    print(f"开始重训练... (实时日志)")
    print(f"{'─'*60}\n")
    import sys
    sys.stdout.flush()

    # 运行重训练 - 使用实时输出
    start_time = time.time()
    env = {"CUDA_VISIBLE_DEVICES": args.gpu_list}
    result = subprocess.run(cmd, text=True, bufsize=1, env={**subprocess.os.environ, **env})
    retrain_time = time.time() - start_time

    print(f"\n{'─'*60}")
    print(f"{mode} 重训练完成")
    print(f"{'─'*60}")
    print(f"耗时: {retrain_time:.2f}秒")
    sys.stdout.flush()

    if result.returncode != 0:
        print(f"错误: {mode} 重训练失败")
        return None, retrain_time

    print(f"\n{mode} 重训练完成，耗时: {retrain_time:.2f}秒")

    # 读取重训练结果
    result_file = retrain_dir / "result.json"
    if not result_file.exists():
        print(f"警告: 未找到 {result_file}")
        return None, retrain_time

    with open(result_file) as f:
        retrain_result = json.load(f)

    return retrain_result, retrain_time


def generate_report(results, output_base):
    """生成综合对比报告"""
    print(f"\n{'='*80}")
    print("生成综合对比报告")
    print(f"{'='*80}\n")

    report_lines = []
    report_lines.append("# NAS搜索和重训练综合报告\n")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 汇总表格
    report_lines.append("## 结果汇总\n\n")
    report_lines.append("| 执行模式 | 搜索时间(s) | NAS Test MRR | NAS Test Recall | 重训时间(s) | 重训Test MRR | 重训Test Recall |\n")
    report_lines.append("|---------|------------|--------------|----------------|------------|--------------|----------------|\n")

    for mode in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]:
        if mode not in results:
            continue
        r = results[mode]
        nas_mrr = r["nas_arch"].get("test_mrr", "N/A")
        nas_recall = r["nas_arch"].get("test_recall_at_k", "N/A")
        retrain_mrr = r["retrain_result"].get("test_mrr", "N/A")
        retrain_recall = r["retrain_result"].get("test_recall_at_k", "N/A")

        nas_mrr_str = f"{nas_mrr:.4f}" if isinstance(nas_mrr, (int, float)) else nas_mrr
        nas_recall_str = f"{nas_recall:.4f}" if isinstance(nas_recall, (int, float)) else nas_recall
        retrain_mrr_str = f"{retrain_mrr:.4f}" if isinstance(retrain_mrr, (int, float)) else retrain_mrr
        retrain_recall_str = f"{retrain_recall:.4f}" if isinstance(retrain_recall, (int, float)) else retrain_recall

        report_lines.append(
            f"| {mode} | {r['search_time']:.1f} | {nas_mrr_str} | {nas_recall_str} | "
            f"{r['retrain_time']:.1f} | {retrain_mrr_str} | {retrain_recall_str} |\n"
        )

    # 详细分析
    report_lines.append("\n## 详细分析\n\n")
    for mode in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]:
        if mode not in results:
            continue
        r = results[mode]
        report_lines.append(f"### {mode}\n\n")
        report_lines.append(f"**最优架构配置:**\n")
        config = r["nas_arch"]["config"]
        report_lines.append(f"- 模型: {config['model']}\n")
        report_lines.append(f"- Embedding维度: {config['embedding_dim']}\n")
        report_lines.append(f"- Time Projection: {config.get('time_proj', 'off')}\n")
        report_lines.append(f"- Memory Cell: {config.get('memory_cell', 'rnn')}\n\n")

        report_lines.append(f"**性能对比:**\n")
        nas_mrr = r["nas_arch"].get("test_mrr", 0)
        retrain_mrr = r["retrain_result"].get("test_mrr", 0)
        if isinstance(nas_mrr, (int, float)) and isinstance(retrain_mrr, (int, float)):
            diff = retrain_mrr - nas_mrr
            pct = (diff / nas_mrr * 100) if nas_mrr > 0 else 0
            report_lines.append(f"- NAS Test MRR: {nas_mrr:.4f}\n")
            report_lines.append(f"- 重训Test MRR: {retrain_mrr:.4f}\n")
            report_lines.append(f"- 差异: {diff:+.4f} ({pct:+.1f}%)\n\n")

    # 保存报告
    report_file = output_base / "comprehensive_report.md"
    with open(report_file, "w") as f:
        f.writelines(report_lines)

    print(f"报告已保存到: {report_file}\n")

    # 打印到控制台
    print("".join(report_lines))

    return report_file


def main():
    parser = argparse.ArgumentParser(description="完整的NAS搜索和评估流程")
    parser.add_argument("--gpu-list", type=str, default="0,1,2", help="GPU列表，例如: 0,1,2")
    parser.add_argument("--data-path", type=str, default="data/public/mooc.csv", help="数据集路径")
    parser.add_argument("--max-events", type=int, default=20000, help="最大事件数")
    parser.add_argument("--trials", type=int, default=27, help="搜索trials数量")
    parser.add_argument("--epochs", type=int, default=3, help="训练epochs数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="outputs/full_pipeline", help="输出目录")
    parser.add_argument("--modes", type=str, default="all", help="执行模式: all, serial, data_parallel, pipeline_naive, pipeline_smart")
    parser.add_argument("--eval-mode", type=str, default="online", choices=["online", "offline"], help="评估模式: online(允许测试时更新embeddings), offline(冻结embeddings)")
    args = parser.parse_args()

    # 创建输出目录
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # 确定要运行的模式
    if args.modes == "all":
        modes = ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]
    else:
        modes = [m.strip() for m in args.modes.split(",")]

    print(f"\n{'='*80}")
    print("NAS搜索和评估流程")
    print(f"{'='*80}")
    print(f"GPU: {args.gpu_list}")
    print(f"数据: {args.data_path} (max_events={args.max_events})")
    print(f"搜索: {args.trials} trials, {args.epochs} epochs")
    print(f"种子: {args.seed}")
    print(f"执行模式: {', '.join(modes)}")
    print(f"评估模式: {args.eval_mode} ({'frozen=False' if args.eval_mode == 'online' else 'frozen=True'})")
    print(f"输出目录: {output_base}")
    print(f"{'='*80}\n")

    # 运行流程
    results = {}
    total_start = time.time()

    print(f"\n{'='*80}")
    print(f"开始处理 {len(modes)} 个执行模式")
    print(f"{'='*80}\n")
    import sys

    for idx, mode in enumerate(modes, 1):
        print(f"\n{'█'*80}")
        print(f"█ 模式 {idx}/{len(modes)}: {mode.upper()}")
        print(f"█ 总进度: {(idx-1)/len(modes)*100:.0f}% → {idx/len(modes)*100:.0f}%")
        print(f"{'█'*80}\n")
        sys.stdout.flush()

        mode_start = time.time()

        # 1. NAS搜索
        best_arch, search_time = run_nas_search(mode, args, output_base)
        if best_arch is None:
            print(f"跳过 {mode} 的重训练（搜索失败）")
            continue

        # 2. 重训练最优架构
        retrain_result, retrain_time = retrain_best_arch(mode, best_arch, args, output_base)
        if retrain_result is None:
            print(f"警告: {mode} 重训练失败")

        # 3. 记录结果
        results[mode] = {
            "nas_arch": best_arch,
            "search_time": search_time,
            "retrain_result": retrain_result or {},
            "retrain_time": retrain_time,
            "total_time": time.time() - mode_start,
        }

        print(f"\n{'█'*80}")
        print(f"█ {mode.upper()} 完成")
        print(f"█ 搜索时间: {search_time:.1f}s ({search_time/60:.1f}min)")
        print(f"█ 重训时间: {retrain_time:.1f}s")
        print(f"█ 总耗时: {results[mode]['total_time']:.1f}s ({results[mode]['total_time']/60:.1f}min)")
        if retrain_result:
            print(f"█ Test MRR: {retrain_result.get('test_mrr', 0):.4f}")
            print(f"█ Test Recall@10: {retrain_result.get('test_recall_at_k', 0):.4f}")
        print(f"{'█'*80}\n")
        sys.stdout.flush()

    total_time = time.time() - total_start

    # 4. 生成综合报告
    if results:
        print(f"\n{'='*80}")
        print("生成综合报告...")
        print(f"{'='*80}\n")
        import sys
        sys.stdout.flush()
        generate_report(results, output_base)

    total_time = time.time() - total_start

    print(f"\n{'='*80}")
    print(f"全部流程完成！")
    print(f"{'='*80}")
    print(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟 / {total_time/3600:.1f}小时)")
    print(f"\n结果汇总:")
    print(f"{'─'*80}")
    print(f"{'模式':<20} {'搜索(min)':<12} {'重训(s)':<10} {'Test MRR':<12} {'Test Recall':<12}")
    print(f"{'─'*80}")
    for mode in modes:
        if mode in results:
            r = results[mode]
            mrr = r['retrain_result'].get('test_mrr', 0)
            recall = r['retrain_result'].get('test_recall_at_k', 0)
            print(f"{mode:<20} {r['search_time']/60:<12.1f} {r['retrain_time']:<10.1f} {mrr:<12.4f} {recall:<12.4f}")
    print(f"{'─'*80}")
    print(f"\n详细报告: {output_base}/comprehensive_report.md")
    print(f"{'='*80}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

