"""GNN vs GA on real mid-episode reschedule problems (same snapshot, same local search).

Episodes are rolled out with the FIFO + least-loaded rule (rescheduling whenever allowed) so the
AMRs have realistic queues / jobs in progress; at each snapshot time the exact reschedule problem
(unstarted jobs + init_state, built like GridEnv.step) is solved by both schedulers with the
DDQN environments' settings: GA = evolve (pop 200, gen 150) + 1000 routing + 400 collision LS;
GNN = solve_with_gnn + 1000 routing + 400 collision LS.
Makespan (relative to the snapshot time) = collision-checked decode averaged over 5 fixed seeds.

usage (from Testing_Tools):
  python compare_dynamic_snapshots.py --episodes 0 1 2 3 4 --times 50 150 250 350 --out snapshots.jsonl
"""
import argparse
import json
import os
import random
import sys
import time

import torch

ROUTING, COLLISION = 1000, 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../../test_case/dynamic/test_dataset_r4.jsonl")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--times", type=float, nargs="+", default=[50, 150, 250, 350])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, root); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    torch.set_num_threads(1)
    import GNN_models.GNN_DDQN_V7 as M
    from test_rule_rescheduling import fifo_least_loaded
    import GA.GA as GA
    gnn = M.SchedulerGNN(); M.load_gnn_weights(gnn, M.DEFAULT_WEIGHTS); gnn.eval()
    true_solve, true_li = M.solve_with_gnn, M.local_improve
    M.solve_with_gnn, M.local_improve = fifo_least_loaded, (lambda ind, *a, **k: ind)   # rollout policy
    M.CONFIG["DATASET_PATH"] = args.dataset; M.CONFIG["SIM_TIME"] = 500.0; M.CONFIG["DEVICE"] = "cpu"
    GA.routing_iters, GA.collision_routing_iters = ROUTING, COLLISION

    def ms(ind, jobs, init):
        vals = []
        for r in range(5):
            random.seed(10_000 + r)
            vals.append(max(GA.decode_schedule_tick_by_tick(ind, jobs, check_collision=True, init_state=init)[0].values()) - init["time"])
        return sum(vals) / len(vals)

    env = M.GridEnv()
    with open(args.out, "w") as f:
        for ep in args.episodes:
            random.seed(42 + ep * 1000)
            env.ep_idx = ep; env.reset()
            for t_snap in sorted(args.times):
                while env.sim_time < t_snap:
                    env.step(1 if env.get_action_mask()[1] > 0.5 else 0)
                # build the reschedule problem exactly like GridEnv.step does
                p2s = {v: k for k, v in GA.STATIONS.items()}
                unstarted = [j for j in env.active_jobs if j.status == 1]
                if len(unstarted) < 2:
                    continue
                jobs = [GA.Job(idx=i, type_=j.material, station=p2s.get((int(j.dest_pos[0]), int(j.dest_pos[1])), "M1_1"),
                               duration=j.proc_time) for i, j in enumerate(unstarted)]
                init = env.sim.get_gnn_init_state()
                row = {"episode": ep, "t": env.sim_time, "n_jobs": len(jobs)}

                random.seed(ep * 100 + int(t_snap))
                t0 = time.perf_counter(); best, _ = GA.evolve(jobs, init_state=init); row["GA_s"] = time.perf_counter() - t0
                row["GA"] = ms(best, jobs, init)

                t0 = time.perf_counter()
                raw, _, _ = true_solve(jobs, gnn, deterministic=True, init_state=init)
                row["GNN_build_s"] = time.perf_counter() - t0
                random.seed(ep * 100 + int(t_snap))
                ind = true_li(raw, jobs, max_iters=ROUTING, init_state=init)
                ind = true_li(ind, jobs, max_iters=COLLISION, check_collision=True, init_state=init)
                row["GNN_s"] = time.perf_counter() - t0
                row["GNN_raw"] = ms(raw, jobs, init)
                row["GNN"] = ms(ind, jobs, init)
                row["FIFO"] = ms(fifo_least_loaded(jobs, init_state=init)[0], jobs, init)
                f.write(json.dumps(row) + "\n"); f.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
