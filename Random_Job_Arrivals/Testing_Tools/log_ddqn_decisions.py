"""Replay a trained DDQN greedily and log the state + Q-values at every step.

usage (from Testing_Tools):
  python log_ddqn_decisions.py --module GNN_models.GNN_DDQN_V7 --model <ckpt.pth> \
      --dataset ../../test_case/dynamic/test_dataset_r4.jsonl --sim_time 500 --episode 0 --out decisions.jsonl
Seeds match test_performance.py (GLOBAL_SEED + ep * 1000), so the run mirrors its AI episode.
"""
import argparse
import importlib
import json
import os
import random
import sys

import numpy as np
import torch

GLOBAL_SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sim_time", type=float, default=None)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
    mod = importlib.import_module(args.module)
    CONFIG = mod.CONFIG
    CONFIG["DATASET_PATH"] = args.dataset
    if args.sim_time is not None:
        CONFIG["SIM_TIME"] = args.sim_time
    CONFIG["DEVICE"] = "cpu"

    random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); torch.manual_seed(GLOBAL_SEED)
    env = mod.GridEnv()
    agent = mod.SchedulerAgent()
    agent.load_state_dict(torch.load(args.model, map_location="cpu"))
    agent.eval()

    ep = args.episode
    random.seed(GLOBAL_SEED + ep * 1000); np.random.seed(GLOBAL_SEED + ep * 1000); torch.manual_seed(GLOBAL_SEED + ep * 1000)
    env.ep_idx = ep
    state = env.reset()
    rows = []
    while True:
        a_feat, j_feat, q_feat = state
        s_amr = torch.tensor([a_feat], dtype=torch.float32)
        s_job = torch.tensor([j_feat], dtype=torch.float32)
        mask_job = (s_job[:, :, 0:1] > 0.0).float()
        s_q = torch.tensor([q_feat], dtype=torch.float32)
        amask = env.get_action_mask()
        with torch.no_grad():
            q = agent(s_amr, s_job, s_q, mask_job)[0].tolist()
        action = 1 if (amask[1] > 0.5 and q[1] > q[0]) else 0

        modes = [env.sim.amr_states[a]["mode"] for a in env.sim.amr_states]
        queues = [len(env.sim.amr_queues[a]) for a in env.sim.amr_queues]
        rows.append({
            "t": env.sim_time,
            "allowed": amask[1] > 0.5,
            "action": action,
            "q_wait": q[0], "q_resched": q[1],
            "unscheduled": int(q_feat[0]),                 # active, not in any AMR queue/slot
            "active": len(env.active_jobs),
            "queued": sum(queues),                         # jobs waiting in AMR queues
            "idle_amrs": sum(m == "idle" for m in modes),
            "idle_empty_amrs": sum(m == "idle" and n == 0 for m, n in zip(modes, queues)),
            "since_resched": env.sim_time - env.last_resched_t,
            "new_arrival": env.arrival_version != env.last_resched_version,
            "new_completion": len(env.completed_jobs) != env.last_resched_completion_count,
            "completed": len(env.completed_jobs),
            "progress": q_feat[3],
            "est_makespan": q_feat[6],
        })
        state, _, done, _ = env.step(action)
        if done:
            break
    with open(args.out, "w") as f:
        for r in rows:
            r["episode"] = ep
            f.write(json.dumps(r) + "\n")
    print(f"episode {ep}: {len(rows)} steps, {sum(r['action'] for r in rows)} reschedules, "
          f"{len(env.completed_jobs)}/{env.total_jobs} jobs done")


if __name__ == "__main__":
    main()
