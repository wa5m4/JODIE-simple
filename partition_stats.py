"""分区倾斜统计:为引言 Challenge III 提供 X 倍数字。

与 public_dataset.py 的加载逻辑一致(跳过表头、按 (timestamp, line_no) 排序、
取前 max_events),再按 run_all.py 的 0.7/0.1/0.2 划分,对 train 按
PARTITION_SIZE=2000 的 count 策略切分区,统计每区:
  n_events / unique_users / unique_items / new_users / new_items
(new = 本分区首次出现、此前所有分区都没见过)。

用法: python partition_stats.py [max_events]   (默认 20000)
"""
import csv
import sys


DATASET = "data/public/mooc.csv"
PARTITION_SIZE = 2000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1


def load(path, max_events):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue
            if row[0].strip().lower() in {"user", "user_id"}:
                continue
            rows.append((int(row[0]), int(row[1]), float(row[2]), line_no))
    rows.sort(key=lambda x: (x[2], x[3]))
    if max_events > 0:
        rows = rows[:max_events]
    return rows


def partition_stats(events):
    stats = []
    seen_u, seen_i = set(), set()
    for i in range(0, len(events), PARTITION_SIZE):
        chunk = events[i : i + PARTITION_SIZE]
        users, items = set(), set()
        new_u, new_i = set(), set()
        for uid, iid, ts, _ in chunk:
            users.add(uid)
            items.add(iid)
            if uid not in seen_u:
                new_u.add(uid)
            if iid not in seen_i:
                new_i.add(iid)
            seen_u.add(uid)
            seen_i.add(iid)
        stats.append(
            dict(
                n=len(chunk),
                nu=len(users),
                ni=len(items),
                nnu=len(new_u),
                nni=len(new_i),
            )
        )
    return stats


def summarize(stats):
    keys = ["n", "nu", "ni", "nnu", "nni"]
    names = {
        "n": "事件数",
        "nu": "unique users",
        "ni": "unique items",
        "nnu": "new users",
        "nni": "new items",
    }
    print(f"{'指标':<14}{'min':>8}{'max':>8}{'mean':>8}{'max/min':>10}")
    for k in keys:
        vals = [s[k] for s in stats]
        lo, hi = min(vals), max(vals)
        mean = sum(vals) / len(vals)
        ratio = hi / lo if lo > 0 else float("nan")
        print(f"{names[k]:<14}{lo:>8}{hi:>8}{mean:>8.1f}{ratio:>10.2f}")
    print(f"\n分区数: {len(stats)}")


def main():
    max_events = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    print(f"=== MOOC 前 {max_events} 条事件 · 分区 {PARTITION_SIZE} ===")
    events = load(DATASET, max_events)
    n_train = int(len(events) * TRAIN_RATIO)
    train_events = events[:n_train]
    stats = partition_stats(train_events)
    summarize(stats)


if __name__ == "__main__":
    main()
