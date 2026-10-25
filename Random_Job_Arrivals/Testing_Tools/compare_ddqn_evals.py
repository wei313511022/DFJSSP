"""Compare test_performance.py results of several DDQN evaluation folders.

usage (from Testing_Tools):
  python compare_ddqn_evals.py GNN=../models_pth/gnn_ddqn_eval_r4 GA=../models_pth/ga_ddqn_eval_r4
Each folder holds <training>_ep*.csv (+ .log) with training in {demo, r4}.
"""
import csv
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

ROW = re.compile(r"^(\d+)\s+\|")


def load(folder, tag):
    rows = {}
    for p in glob.glob(os.path.join(folder, f"{tag}_ep*.csv")):
        if p.endswith("_summary.csv"):
            continue
        for r in csv.DictReader(open(p)):
            rows[int(r["Episode"])] = r
    resched = defaultdict(int)
    for p in glob.glob(os.path.join(folder, f"{tag}_ep*.log")):
        n = 0
        for line in open(p):
            if "[AI-DECIDE]" in line:
                n += 1
            elif m := ROW.match(line):
                resched[int(m.group(1))] = n
                n = 0
    return rows, resched


def main():
    sets = [a.split("=", 1) for a in sys.argv[1:]]
    data = {(name, tag): load(folder, tag) for name, folder in sets for tag in ("demo", "r4")}
    common = sorted(set.intersection(*[set(rows) for rows, _ in data.values()]))
    print(f"episodes compared: {common}\n")
    f = lambda rows, col: st.mean(float(rows[e][col]) for e in common)
    print(f"{'scheduler':<9} {'trained':<7} | {'done% DDQN':>10} {'fixed':>6} | {'flow DDQN':>9} {'fixed':>6} | "
          f"{'sched s DDQN':>12} {'fixed':>6} | {'resched/ep':>10} | DDQN wins")
    for name, _ in sets:
        for tag in ("demo", "r4"):
            rows, rs = data[(name, tag)]
            wins = sum(rows[e]["Winner"] == "AI" for e in common)
            print(f"{name:<9} {tag:<7} | {f(rows, 'AI_Done(%)'):10.1f} {f(rows, 'Fix_Done(%)'):6.1f} | "
                  f"{f(rows, 'AI_Flow'):9.1f} {f(rows, 'Fix_Flow'):6.1f} | {f(rows, 'AI_GA_Time'):12.1f} {f(rows, 'Fix_GA_Time'):6.1f} | "
                  f"{st.mean(rs[e] for e in common):10.1f} | {wins}/{len(common)}")
    print("\nper episode, jobs done % (DDQN):")
    print("  ep  " + "".join(f"{n}-{t:<5} " for n, _ in sets for t in ("demo", "r4")) + " | fixed: " + " ".join(n for n, _ in sets))
    for e in common:
        cells = "".join(f"{float(data[(n, t)][0][e]['AI_Done(%)']):9.1f} " for n, _ in sets for t in ("demo", "r4"))
        fixed = " ".join(f"{float(data[(n, 'r4')][0][e]['Fix_Done(%)']):5.1f}" for n, _ in sets)
        print(f"  {e:<3} {cells} | {fixed}")


if __name__ == "__main__":
    main()
