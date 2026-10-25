"""When does the DDQN reschedule?  Reads log_ddqn_decisions.py output (<model>_ep*.jsonl).

usage (from Testing_Tools):  python analyze_ddqn_decisions.py ../models_pth/gnn_ddqn_eval_r4/decisions demo r4
Only steps where rescheduling was allowed are analysed (the other steps are forced waits).
"""
import glob
import json
import os
import sys

import numpy as np

BINS = {
    "unscheduled": [(1, 1), (2, 3), (4, 7), (8, 15), (16, 999)],
    "idle_empty_amrs": [(0, 0), (1, 1), (2, 2), (3, 3)],
    "queued": [(0, 0), (1, 5), (6, 15), (16, 30), (31, 999)],
    "since_resched": [(0, 5), (5.01, 10), (10.01, 20), (20.01, 40), (40.01, 1e9)],
    "t": [(0, 99.99), (100, 199.99), (200, 299.99), (300, 399.99), (400, 1e9)],
}
LABELS = {
    "unscheduled": "Unscheduled jobs (arrived, not in any AMR queue)",
    "idle_empty_amrs": "AMRs idle with an empty queue",
    "queued": "Jobs already queued on AMRs",
    "since_resched": "Seconds since last reschedule",
    "t": "Episode time (s)",
}
FEATURES = ["unscheduled", "idle_empty_amrs", "queued", "since_resched", "t", "new_arrival", "new_completion"]


def load(folder, tag):
    rows = [json.loads(l) for p in sorted(glob.glob(os.path.join(folder, f"{tag}_ep*.jsonl"))) for l in open(p)]
    return [r for r in rows if r["allowed"]]


def fmt_range(lo, hi):
    if hi >= 1e8 or hi >= 999:
        return f"{lo:g}+"
    return f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"


def logistic(X, y, iters=3000, lr=0.1, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]])
    acc = ((1 / (1 + np.exp(-Z @ w)) > 0.5) == y).mean()
    return w[1:], acc


def main():
    folder, tags = sys.argv[1], sys.argv[2:]
    data = {t: load(folder, t) for t in tags}
    for t in tags:
        rows = data[t]
        y = np.array([r["action"] for r in rows])
        print(f"\n=== {t}: {len(rows)} allowed decisions, rescheduled {y.sum()} ({y.mean()*100:.0f}%)")
    for feat, bins in BINS.items():
        print(f"\n{LABELS[feat]}")
        print(f"  {'value':<8}" + "".join(f"| {t:>6} rate   (n) " for t in tags))
        for lo, hi in bins:
            cells = []
            for t in tags:
                sel = [r["action"] for r in data[t] if lo <= r[feat] <= hi]
                cells.append(f"| {np.mean(sel)*100:6.0f}%  ({len(sel):>4}) " if sel else "|      -   (   0) ")
            print(f"  {fmt_range(lo, hi):<8}" + "".join(cells))
    print("\nWhat triggered the allowed decision")
    for name, cond in [("arrival only", lambda r: r["new_arrival"] and not r["new_completion"]),
                       ("completion only", lambda r: r["new_completion"] and not r["new_arrival"]),
                       ("both", lambda r: r["new_arrival"] and r["new_completion"])]:
        cells = []
        for t in tags:
            sel = [r["action"] for r in data[t] if cond(r)]
            cells.append(f"| {np.mean(sel)*100:6.0f}%  ({len(sel):>4}) " if sel else "|      -   (   0) ")
        print(f"  {name:<16}" + "".join(cells))
    print("\nLogistic regression on standardised features (positive = more likely to reschedule)")
    for t in tags:
        rows = data[t]
        X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
        y = np.array([r["action"] for r in rows], dtype=float)
        w, acc = logistic(X, y)
        order = np.argsort(-np.abs(w))
        print(f"  {t} (fit accuracy {acc*100:.0f}%): " + ", ".join(f"{FEATURES[i]} {w[i]:+.2f}" for i in order))


if __name__ == "__main__":
    main()
