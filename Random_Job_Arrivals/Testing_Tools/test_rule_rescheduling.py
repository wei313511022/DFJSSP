"""Event-driven rescheduling with a dispatching rule instead of GNN/GA.

Rule: FIFO + least-loaded AMR. Unstarted jobs are taken in arrival order; each goes to
the AMR with the least load = remaining time of its current job + processing time of the
jobs already given to it in this plan (ties -> lower AMR index). No local search.
Policy: reschedule every time the environment allows it (a job arrived or finished since
the last reschedule and some job is unstarted).

Runs in GNN_DDQN_V7's environment with its scheduler call swapped for the rule, so the
simulator, metrics and seeds match test_performance.py.

usage (from Testing_Tools):
  python test_rule_rescheduling.py --dataset ../../test_case/dynamic/test_dataset_r4.jsonl \
      --sim_time 500 --episode 0 --out ../models_pth/rule_fifo_eval_r4/ep0.json
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

GLOBAL_SEED = 42


def fifo_least_loaded(jobs, model=None, deterministic=True, init_state=None):
    from GA.GA import AMR_KEYS, Individual
    start = time.perf_counter()
    now = init_state["time"] if init_state else 0.0
    load = {a: max(0.0, init_state["availability"].get(a, now) - now) if init_state else 0.0 for a in AMR_KEYS}
    assignment = {}
    for job in jobs:                                    # env passes unstarted jobs in arrival order
        amr = min(AMR_KEYS, key=lambda a: (load[a], AMR_KEYS.index(a)))
        assignment[job.idx] = amr
        load[amr] += job.duration
    ind = Individual(order=[j.idx for j in jobs], amr_assignment=[assignment[i] for i in range(len(jobs))])
    return ind, 0.0, time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sim_time", type=float, default=500.0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, root)
    import GNN_models.GNN_DDQN_V7 as M
    M.solve_with_gnn = fifo_least_loaded                 # scheduler call inside GridEnv.step
    M.local_improve = lambda ind, *a, **k: ind           # no routing / collision local search
    CONFIG = M.CONFIG
    CONFIG["DATASET_PATH"] = args.dataset
    CONFIG["SIM_TIME"] = args.sim_time
    CONFIG["DEVICE"] = "cpu"

    random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); torch.manual_seed(GLOBAL_SEED)
    env = M.GridEnv()
    ep = args.episode
    random.seed(GLOBAL_SEED + ep * 1000); np.random.seed(GLOBAL_SEED + ep * 1000); torch.manual_seed(GLOBAL_SEED + ep * 1000)
    env.ep_idx = ep
    env.reset()
    resched, compute = 0, 0.0
    while True:
        action = 1 if env.get_action_mask()[1] > 0.5 else 0
        _, _, done, _ = env.step(action)
        if action == 1:
            resched += 1
            compute += env.last_ga_compute_time
        if done:
            break

    # same metrics as test_performance.episode_metrics
    total = len(env.completed_jobs) + len(env.active_jobs) + len(env.queue)
    finished_in = sum(1 for j in env.completed_jobs if j.finish_ts <= CONFIG["SIM_TIME"])
    flows = [j.finish_ts - j.arrival_ts for j in env.completed_jobs if j.finish_ts >= 0]
    res = {"episode": ep, "done_pct": finished_in / total * 100 if total else 0.0,
           "flow": float(np.mean(flows)) if flows else 1000.0,
           "reschedules": resched, "compute_s": compute, "completed": len(env.completed_jobs), "total": total}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
