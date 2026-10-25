"""Static comparison: SchedulerGNN vs GA.py vs Taguchi-GA on the dispatch_inbox test sets.

Every method ends with the same local search (1000 routing + 100 collision-checked
iterations; 100 is Taguchi-GA.py's native value, GA.py is raised from 1 to 100 for
this run). Makespan = collision-checked decode averaged over 5 fixed seeds, so the
random dodge in the decoder does not decide the comparison.

usage (from Static_alogorithm):
  python compare_schedulers.py --n_jobs 60 --first 950 --count 5 --out comparison_results/n60.jsonl
  python compare_schedulers.py --summarize comparison_results
"""
import argparse
import glob
import importlib.util
import json
import os
import random
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLLISION_ITERS = 100
EVAL_SEEDS = 5


def makespan(GA, ind, jobs):
    vals = []
    for r in range(EVAL_SEEDS):
        random.seed(10_000 + r)
        vals.append(max(GA.decode_schedule_tick_by_tick(ind, jobs, check_collision=True)[0].values()))
    return sum(vals) / len(vals)


def counted_evolve(mod, jobs, seed):
    calls = [0]
    orig = mod.fitness

    def fit(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    mod.fitness = fit
    random.seed(seed)
    t = time.perf_counter()
    best, _ = mod.evolve(jobs)
    elapsed = time.perf_counter() - t
    mod.fitness = orig
    return best, elapsed, calls[0]


def run(args):
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "GNN"))
    import torch
    torch.set_num_threads(1)
    import GA.GA as GA
    import GNN
    spec = importlib.util.spec_from_file_location("TaguchiGA", HERE / "Taguchi" / "Taguchi-GA.py")
    TG = importlib.util.module_from_spec(spec); spec.loader.exec_module(TG)

    GA.collision_routing_iters = COLLISION_ITERS
    TG.collision_routing_iters = COLLISION_ITERS
    model = GNN.SchedulerGNN()
    assert GNN.load_gnn_weights(model, GNN.DEFAULT_WEIGHTS), "train the GNN first"
    model.eval()

    inbox = HERE.parent / "test_case" / "static" / f"dispatch_inbox_{args.n_jobs}.jsonl"
    events = {int(e["index"]): e for e in GA.load_dispatch_events(inbox)}
    tevents = {int(e["index"]): e for e in TG.load_dispatch_events(inbox)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for idx in range(args.first, args.first + args.count):
            jobs = events[idx]["jobs"]
            row = {"n_jobs": args.n_jobs, "event": idx}

            best, row["GA_s"], row["GA_calls"] = counted_evolve(GA, jobs, idx)
            row["GA"] = makespan(GA, best, jobs)

            best, row["Taguchi_s"], row["Taguchi_calls"] = counted_evolve(TG, tevents[idx]["jobs"], idx)
            row["Taguchi"] = makespan(GA, best, jobs)

            t = time.perf_counter()
            raw, _, _ = GNN.solve_with_gnn(jobs, model, deterministic=True)
            row["GNN_build_s"] = time.perf_counter() - t
            random.seed(idx)
            t = time.perf_counter()
            ind = GA.local_improve(raw, jobs, max_iters=GA.routing_iters)
            ind = GA.local_improve(ind, jobs, max_iters=COLLISION_ITERS, check_collision=True)
            row["GNN_s"] = row["GNN_build_s"] + time.perf_counter() - t   # build + local search
            row["GNN_raw"] = makespan(GA, raw, jobs)
            row["GNN"] = makespan(GA, ind, jobs)

            f.write(json.dumps(row) + "\n"); f.flush()
            print(json.dumps(row), flush=True)


def summarize(folder):
    rows = [json.loads(l) for p in sorted(glob.glob(os.path.join(folder, "*.jsonl"))) for l in open(p)]
    by = defaultdict(list)
    for r in rows:
        by[r["n_jobs"]].append(r)
    header = ["n_jobs", "cases", "GA", "GA_s", "Taguchi", "Taguchi_s", "GNN_raw", "GNN", "GNN_s",
              "best_count_GA", "best_count_Taguchi", "best_count_GNN"]
    lines = [",".join(header)]
    print(f"{'jobs':>4} {'n':>2} | {'GA':>7} {'s':>6} | {'Taguchi':>7} {'s':>6} | {'GNN raw':>7} {'GNN':>7} {'s':>5} | best GA/Tag/GNN")
    for n in sorted(by):
        rs = by[n]
        m = {k: st.mean(r[k] for r in rs) for k in ("GA", "GA_s", "Taguchi", "Taguchi_s", "GNN_raw", "GNN", "GNN_s")}
        best = {k: 0 for k in ("GA", "Taguchi", "GNN")}
        for r in rs:
            lo = min(r["GA"], r["Taguchi"], r["GNN"])
            for k in best:
                best[k] += r[k] == lo
        print(f"{n:>4} {len(rs):>2} | {m['GA']:7.1f} {m['GA_s']:6.2f} | {m['Taguchi']:7.1f} {m['Taguchi_s']:6.2f} | "
              f"{m['GNN_raw']:7.1f} {m['GNN']:7.1f} {m['GNN_s']:5.2f} | {best['GA']}/{best['Taguchi']}/{best['GNN']}")
        lines.append(",".join(str(x) for x in [n, len(rs)] + [f"{m[k]:.2f}" for k in
                     ("GA", "GA_s", "Taguchi", "Taguchi_s", "GNN_raw", "GNN", "GNN_s")] + [best["GA"], best["Taguchi"], best["GNN"]]))
    out = os.path.join(folder, "summary.csv")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"summary -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--out", type=str)
    ap.add_argument("--summarize", type=str, default="")
    a = ap.parse_args()
    summarize(a.summarize) if a.summarize else run(a)
