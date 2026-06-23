"""对比train_single_arch.py构建的配置和best_arch.json的配置"""
import json

# best_arch.json的配置
with open('outputs/bug_fix_verification/seed_42/serial/best_arch.json') as f:
    best_config = json.load(f)['config']

# train_single_arch.py构建的配置
train_single_arch_config = {
    "model": "jodie_rnn",
    "embedding_dim": 128,
    "memory_cell": "rnn",
    "time_proj": "off",
    "use_static_embeddings": "off",
    "event_agg": "none",
    "max_neighbors": 0,
    "batch_mode": "tbatch",
    "train_batch_size": 32,
    "dataset": "public_csv",
    "local_data_path": "data/public/mooc.csv",
    "max_events": 20000,
    "seed": 20042,
    "lr": 0.001,
    "neg_sample_size": 5,
    "k": 10,
    "device": "cuda",
    "num_users": 1435,
    "num_items": 21,
    "feature_dim": 8,
}

print("=" * 70)
print("配置对比")
print("=" * 70)
print()

# 找出best_config有但train_single_arch_config没有的键
missing_keys = set(best_config.keys()) - set(train_single_arch_config.keys())
print(f"best_arch.json有但train_single_arch.py没有的参数 ({len(missing_keys)}个):")
for key in sorted(missing_keys):
    print(f"  {key}: {best_config[key]}")

print()

# 找出值不同的键
different_keys = []
for key in set(best_config.keys()) & set(train_single_arch_config.keys()):
    if best_config[key] != train_single_arch_config[key]:
        different_keys.append(key)

if different_keys:
    print(f"值不同的参数 ({len(different_keys)}个):")
    for key in sorted(different_keys):
        print(f"  {key}:")
        print(f"    best_arch: {best_config[key]}")
        print(f"    train_single_arch: {train_single_arch_config[key]}")
else:
    print("共同参数的值都相同")
