"""
固定架构串行复评(基线协议)—— 双臂消融的补测

背景:双臂最终测试的 batch_mode 取自 base_config(各臂自己的模式),
日志里的 "Serial training" 是写死字符串,实际并非串行:
  - stale 臂 test 0.9335 = stale_batch 训练(与 val 污染同一机制,可能被高估)
  - tbatch 臂 test 0.8793 = tbatch 训练
因此 147K 架构(emb=64/static=on)的真实串行 test 至今未知 —— 本脚本补上。

做法:与搜索流程最终测试完全相同的调用(本分支 = origin/refactored = 08-25 基线代码):
  GraphNASTrainer._evaluate_arch_multi_seed(epochs=5, eval_seeds=None,
  default_seed=42+20000=20042) → evaluate_arch → train_model_ce(batch_mode=serial)
  fit=train+val, test=test。BATCH_MODE 由 build_base_config 提供 = "serial"。

判读:
  1) 133K 探针 ≈ 0.8561(噪声地板内)→ 本路径 = 基线协议成立,147K 可与 0.8561 直接对比:
      147K 明显低于 0.8561  → "选错架构 + 掉分"成立(段 4 原故事)
      147K ≈ 0.8561 或更高  → 走"评分污染 + 选择洗牌 + 方向不可控"版本
  2) 133K 探针 ≈ 0.8793 → refactored 与 arm 协议仍有差异,需继续排查

运行:python reeval_fixed_arch.py(约 12-20 分钟,单 GPU,不启 Ray)
"""
import json
import os
import time

from run_all import build_base_config
from jodie.nas.trainer import GraphNASTrainer

# ── 要复评的架构(rnn_only 搜索空间的 6 个字段)──
ARCHS = {
    # 协议探针:基线搜索选中的架构,串行 test 已知 = 0.856121275963994(×2 bit-identical)
    "133K_baseline_pick": {
        "model": "jodie_rnn",
        "embedding_dim": 128,
        "memory_cell": "rnn",
        "time_proj": "off",
        "use_static_embeddings": "off",
        "normalize_state": "off",
    },
    # 主角:stale_batch 臂选中的架构,真实串行 test 未知
    "147K_stale_pick": {
        "model": "jodie_rnn",
        "embedding_dim": 64,
        "memory_cell": "rnn",
        "time_proj": "off",
        "use_static_embeddings": "on",
        "normalize_state": "off",
    },
}

FINAL_EPOCHS = 5
FINAL_SEED = 42 + 20000  # FINAL_RETRAIN_SEED_OFFSET = 20000


def main() -> None:
    output_dir = "results/reeval_fixed_arch"
    os.makedirs(output_dir, exist_ok=True)

    config = build_base_config("pipeline_naive", output_dir)
    trainer = GraphNASTrainer(config)

    train_data, val_data, test_data, user_type_prefs, item_type, graph_template, partition_plan = trainer._prepare_data()
    final_train_data = train_data + val_data
    print(f"[Reeval] 数据就绪: train+val={len(final_train_data)}, test={len(test_data)}, "
          f"batch_mode={config['batch_mode']}, seed={config['seed']}", flush=True)

    summary = {}
    for name, arch in ARCHS.items():
        print(f"[Reeval] {name}: {arch}", flush=True)
        t0 = time.time()
        result = trainer._evaluate_arch_multi_seed(
            arch_config=arch,
            train_data=final_train_data,
            eval_data=test_data,
            user_type_prefs=user_type_prefs,
            item_type=item_type,
            graph_template=graph_template,
            epochs=FINAL_EPOCHS,
            eval_seeds=None,
            default_seed=FINAL_SEED,
            phase="final_pipeline",
            eval_split="test",
        )
        print(
            f"[Reeval] {name}: test_mrr={result['mrr']:.10f}  "
            f"recall@10={result['recall_at_k']:.10f}  "
            f"params={result['params']}  time={time.time() - t0:.1f}s",
            flush=True,
        )
        summary[name] = {
            "arch": arch,
            "test_mrr": result["mrr"],
            "test_recall_at_k": result["recall_at_k"],
            "params": result["params"],
            "time_sec": round(time.time() - t0, 1),
        }

    with open(os.path.join(output_dir, "reeval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Reeval] 完成 → {os.path.join(output_dir, 'reeval_summary.json')}", flush=True)


if __name__ == "__main__":
    main()
