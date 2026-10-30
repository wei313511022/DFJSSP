"""Step-by-step trace of one episode under a DDQN or a fixed-period rescheduling policy.

usage (from Testing_Tools):
  python trace_rescheduling.py --module Taguchi_models.Taguchi_DDQN --policy ddqn --model <ckpt.pth> \
      --dataset ../../test_case/dynamic/test_dataset_r4.jsonl --sim_time 500 --episode 0 --out trace.jsonl
  python trace_rescheduling.py --module Taguchi_models.Taguchi_DDQN --policy fixed --period 10 ...
Seeds match test_performance.py (DDQN: 42 + ep*1000, fixed: 42 + ep*1000 + 500).
Each line = one env step: time before/after, action, scheduler compute (s), jobs waiting, etc.

--replay <trace.jsonl>  re-runs a recorded trace deterministically: each reschedule is charged
                        exactly the recorded simulation time instead of measured wall-clock time
                        (actions are checked against the recording).
--ticks_out <file>      also writes one JSON line per simulation tick (AMR positions, modes, jobs,
                        queues, inventory, stations, jobs waiting for assignment, completed count,
                        and whether the AMRs are waiting for a reschedule computation).
"""
import argparse
import importlib
import json
import os
import random
import sys

import numpy as np
import time
import torch

GLOBAL_SEED = 42


class _RecordedDuration:
    """Stand-in for time.perf_counter during one reschedule: first call 0, later calls = duration."""
    def __init__(self, duration):
        self.duration, self.calls = float(duration), 0

    def __call__(self):
        self.calls += 1
        return 0.0 if self.calls == 1 else self.duration


def tick_snapshot(env, computing):
    sim = env.sim
    queued_ids, amrs = set(), {}
    for a, st in sim.amr_states.items():
        job = st["job"]
        q = [[j.idx, j.type_, j.station] for j in sim.amr_queues[a]]
        queued_ids.update(j[0] for j in q)
        if job is not None:
            queued_ids.add(job.idx)
        amrs[a] = {"pos": list(sim.positions[a]), "mode": st["mode"],
                   "job": [job.idx, job.type_, job.station] if job is not None else None,
                   "queue": q, "inv": dict(sim.inventory[a])}
    waiting = [[j.jid, j.material] for j in env.active_jobs if j.status == 1 and j.jid not in queued_ids]
    waiting += [[j.jid, j.material] for j in env.queue if j.arrival_ts <= sim.t]   # arrived, not yet seen
    return {"t": sim.t, "computing": computing, "amrs": amrs, "waiting": waiting,
            "completed": len(env.completed_jobs) + len(sim.completed_jobs_jids),
            "stations": dict(sim.station_occupied)}


def counts(env):
    queued_ids = set()
    for a in env.sim.amr_states:
        if env.sim.amr_states[a]["job"] is not None:
            queued_ids.add(env.sim.amr_states[a]["job"].idx)
        for qj in env.sim.amr_queues[a]:
            queued_ids.add(qj.idx)
    unstarted = [j for j in env.active_jobs if j.status == 1]
    return {
        "unstarted": len(unstarted),                                       # size of a reschedule now
        "unscheduled": sum(1 for j in unstarted if j.jid not in queued_ids),  # arrived, in no plan
        "queued": sum(len(env.sim.amr_queues[a]) for a in env.sim.amr_queues),
        "completed": len(env.completed_jobs),
        "arrived": len(env.active_jobs) + len(env.completed_jobs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--policy", choices=["ddqn", "fixed"], required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--period", type=float, default=10.0)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sim_time", type=float, default=500.0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay", default="", help="recorded trace to reproduce exactly")
    ap.add_argument("--ticks_out", default="", help="per-tick state output (JSON lines)")
    args = ap.parse_args()
    replay = [json.loads(l) for l in open(args.replay)][1:] if args.replay else None

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
    mod = importlib.import_module(args.module)
    mod.CONFIG["DATASET_PATH"] = args.dataset
    mod.CONFIG["SIM_TIME"] = args.sim_time
    mod.CONFIG["DEVICE"] = "cpu"

    random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); torch.manual_seed(GLOBAL_SEED)
    env = mod.GridEnv()
    agent = None
    if args.policy == "ddqn":
        agent = mod.SchedulerAgent()
        agent.load_state_dict(torch.load(args.model, map_location="cpu"))
        agent.eval()

    ep = args.episode
    seed = GLOBAL_SEED + ep * 1000 + (500 if args.policy == "fixed" else 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env.ep_idx = ep
    state = env.reset()
    last_fix = -1e9
    rows = []
    ticks, cur = [], {"computing": False}
    if args.ticks_out:
        sim_step = env.sim.step
        def stepped(dt):                       # record every tick inside env.step
            for _ in range(int(dt)):
                sim_step(1)
                ticks.append(tick_snapshot(env, cur["computing"]))
        env.sim.step = stepped
        ticks.append(tick_snapshot(env, False))
    while True:
        allowed = env.get_action_mask()[1] > 0.5
        if args.policy == "ddqn":
            a_feat, j_feat, q_feat = state
            s_job = torch.tensor([j_feat], dtype=torch.float32)
            with torch.no_grad():
                q = agent(torch.tensor([a_feat], dtype=torch.float32), s_job,
                          torch.tensor([q_feat], dtype=torch.float32), (s_job[:, :, 0:1] > 0.0).float())[0]
            action = 1 if (allowed and q[1] > q[0]) else 0
        else:
            action = 1 if (allowed and env.sim_time - last_fix >= args.period) else 0
            if action:
                last_fix = env.sim_time
        before = counts(env)
        t0 = env.sim_time
        i = len(rows)
        if replay is not None:
            rec = replay[i] if i < len(replay) else None
            if rec is None or rec["action"] != action or abs(rec["t"] - t0) > 1e-9:
                print(f"WARNING: replay diverged at step {i} (t={t0}); continuing with measured time")
                replay = None
        cur["computing"] = bool(action)
        real_clock = time.perf_counter
        if replay is not None and action:
            time.perf_counter = _RecordedDuration(replay[i]["t_after"] - replay[i]["t"])
        try:
            state, _, done, dt = env.step(action)
        finally:
            time.perf_counter = real_clock
        rows.append({"t": t0, "t_after": env.sim_time, "action": action, "allowed": allowed,
                     "compute_s": env.last_ga_compute_time if action else 0.0, **before,
                     "completed_after": len(env.completed_jobs)})
        if done:
            break

    total = len(env.completed_jobs) + len(env.active_jobs) + len(env.queue)
    finished_in = sum(1 for j in env.completed_jobs if j.finish_ts <= mod.CONFIG["SIM_TIME"])
    summary = {"episode": ep, "policy": args.policy, "total_jobs": total, "done_pct": finished_in / total * 100,
               "reschedules": sum(r["action"] for r in rows), "compute_s": sum(r["compute_s"] for r in rows)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if args.ticks_out:
        with open(args.ticks_out, "w") as f:
            for t in ticks:
                f.write(json.dumps(t) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
