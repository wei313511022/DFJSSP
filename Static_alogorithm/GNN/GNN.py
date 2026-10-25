import csv
import json
import os
import random
import time
from collections import deque
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import all constants and simulation logic from GA
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GA.GA import (
    Job, Individual, AMR_STARTS, AMR_KEYS, STATIONS, OBSTACLES, _GRID_POINTS,
    GRID_MIN_X, GRID_MAX_X, GRID_MIN_Y, GRID_MAX_Y, BASES, TYPE_DURATION,
    SUPPLY_LOCATIONS, SCHEDULE_OUTBOX, DISPATCH_INBOX, DISPATCH_EVENT_INDEX_ENV,
    JOB_COUNT, MAX_DEPTH, routing_iters, collision_routing_iters,
    _is_within_bounds, _DELTAS, _adjacent_points, _build_path, _manhattan_path,
    heuristic, shortest_path, find_dynamic_path, _extend_path_log, grid_distance,
    nearest_base_to_station, _diagnose_and_print_failure, decode_schedule, decode_schedule_tick_by_tick, fitness, local_improve,
    plot_gantt, station_key_from_value, load_dispatch_events, make_jobs,
    _earliest_fit, _book,
)

# Overwrite describe_solution locally to pass save_img
def describe_solution_gnn(individual: Individual, jobs: List[Job], solve_time: float = None, show_gantt: bool = False, save_img: str = None) -> Tuple[float, float]:
    availability, decoded_timeline, queue_infos, path_logs, invalid_count = decode_schedule_tick_by_tick(individual, jobs, need_log=True, check_collision=True)
    makespan = max(availability.values())
    print(f"Optimal Makespan Found: {makespan:.2f}s")
    print(f"Invalid Jobs Count: {invalid_count}")
    if solve_time is not None:
        print(f"Computation Time: {solve_time:.4f}s")
    if show_gantt or save_img:
        # Generate plot
        plot_gantt(decoded_timeline, queue_infos, jobs, solve_time=solve_time, invalid_count=invalid_count, show_gantt=show_gantt, save_img=save_img)

    return makespan, solve_time

# ==========================================
# Graph layout (architecture follows AMR-DFJSP extend_GNN, adapted to one
# operation per job: assign job -> AMR; refills happen implicitly in the decoder)
# ==========================================
AMR_IN_DIM = 11
JOB_IN_DIM = 11
DOCK_IN_DIM = 8
ACTION_IN_DIM = 4
DISTANCE_SCALE = 20.0
TIME_SCALE = 100.0
WORKLOAD_SCALE = 500.0
REFILL_AMOUNT = 3  # decode_schedule refills an empty material to 3
# The tensors here are tiny; with the default 24 CPU threads every op pays thread-sync
# overhead (60-job build: ~3 s with 24 threads vs 0.08 s with 1-4 threads).
GNN_NUM_THREADS = 4

MATERIALS = list(TYPE_DURATION.keys())
STATION_KEYS = list(STATIONS.keys())
SUPPLY_KEYS = [f"supply_{m}" for m in MATERIALS]
# "Dock" nodes = service points: the stations plus one supply point per material.
DOCK_KEYS = STATION_KEYS + SUPPLY_KEYS
DOCK_INDEX = {k: i for i, k in enumerate(DOCK_KEYS)}
DOCK_POSITIONS = {**STATIONS, **{f"supply_{m}": SUPPLY_LOCATIONS[m] for m in MATERIALS}}

# Weights for this architecture. Checkpoints of the old attention-only
# SchedulerGNN (gnn_scheduler_best.pth / gnn_scheduler_retrained.pth) do not fit it.
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "gnn_scheduler_extend.pth"


def travel_time(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Grid travel time, the same A* path length decode_schedule uses."""
    return len(shortest_path(a, b)) - 1


def load_gnn_weights(model: nn.Module, path=DEFAULT_WEIGHTS, map_location='cpu') -> bool:
    """Load weights into `model`; warn and keep it untrained if missing or incompatible."""
    path = Path(path)
    if not path.exists():
        print(f"WARNING: GNN weights {path} not found; using an untrained SchedulerGNN.")
        return False
    try:
        model.load_state_dict(torch.load(path, map_location=map_location))
    except RuntimeError:
        print(f"WARNING: {path} does not match the current SchedulerGNN architecture "
              f"(old checkpoint?); using an untrained SchedulerGNN.")
        return False
    return True


# ==========================================
# Construction state (fast model == decode_schedule without collisions)
# ==========================================
class ConstructionState:
    """Partial schedule while the GNN assigns jobs one at a time.

    Times follow decode_schedule exactly (A* travel, refill to 3 at the supply point when
    the material is empty, station booking calendars with _earliest_fit), so the prefix
    the GNN sees is what the non-collision decoder would produce for it.
    """

    def __init__(self, jobs: Sequence[Job], init_state: dict = None):
        self.jobs = list(jobs)
        self.n = len(self.jobs)
        if init_state:
            self.t0 = float(init_state.get("time", 0.0))
            self.avail = {a: float(init_state["availability"].get(a, 0.0)) for a in AMR_KEYS}
            self.pos = {a: tuple(init_state["positions"].get(a, AMR_STARTS[a])) for a in AMR_KEYS}
            self.inv = {a: dict(init_state["inventory"].get(a, {m: 0 for m in MATERIALS})) for a in AMR_KEYS}
        else:
            self.t0 = 0.0
            self.avail = {a: 0.0 for a in AMR_KEYS}
            self.pos = {a: AMR_STARTS[a] for a in AMR_KEYS}
            self.inv = {a: {m: 0 for m in MATERIALS} for a in AMR_KEYS}
            if "AMR1" in self.inv: self.inv["AMR1"]["A"] = 3
            if "AMR2" in self.inv: self.inv["AMR2"]["B"] = 3
            if "AMR3" in self.inv: self.inv["AMR3"]["C"] = 3
        self.calendar = {s: [] for s in STATION_KEYS}
        self.assigned = [False] * self.n
        self.carrier = [None] * self.n          # AMR index per job position
        self.order: List[int] = []
        self.assignment: Dict[int, str] = {}
        self.amr_load = {a: 0.0 for a in AMR_KEYS}
        self.amr_count = {a: 0 for a in AMR_KEYS}
        self.last_on_amr: Dict[str, int] = {}
        self.last_on_station: Dict[str, int] = {}
        self.adj = torch.zeros(self.n, self.n)
        self.station_idx = torch.tensor([DOCK_INDEX[j.station] for j in self.jobs], dtype=torch.long)
        self.supply_idx = torch.tensor([DOCK_INDEX[f"supply_{j.type_}"] for j in self.jobs], dtype=torch.long)

    def done(self) -> bool:
        return len(self.order) == self.n

    def outcome(self, amr: str, k: int):
        """(needs_refill, travel, station_wait, end_time) if job position k went to `amr` next."""
        job = self.jobs[k]
        p, travel = self.pos[amr], 0
        refill = self.inv[amr][job.type_] == 0
        if refill:
            travel += travel_time(p, SUPPLY_LOCATIONS[job.type_])
            p = SUPPLY_LOCATIONS[job.type_]
        travel += travel_time(p, STATIONS[job.station])
        arrive = self.avail[amr] + travel
        start = _earliest_fit(self.calendar[job.station], arrive, job.duration)
        return refill, travel, start - arrive, start + job.duration

    def apply(self, amr_idx: int, k: int) -> None:
        amr, job = AMR_KEYS[amr_idx], self.jobs[k]
        refill, _, _, end = self.outcome(amr, k)
        if refill:
            self.inv[amr][job.type_] = REFILL_AMOUNT
        self.inv[amr][job.type_] -= 1
        _book(self.calendar[job.station], end - job.duration, end)
        self.avail[amr] = end
        self.pos[amr] = STATIONS[job.station]
        self.assigned[k] = True
        self.carrier[k] = amr_idx
        self.order.append(job.idx)
        self.assignment[job.idx] = amr
        self.amr_load[amr] += job.duration
        self.amr_count[amr] += 1
        # Sequence arcs only (consecutive jobs on the same AMR / same station), as in
        # extend_GNN: dense shared-dock cliques collapse job embeddings.
        for prev in (self.last_on_amr.get(amr), self.last_on_station.get(job.station)):
            if prev is not None and prev != k:
                self.adj[prev, k] = 1.0
                self.adj[k, prev] = 1.0
        self.last_on_amr[amr] = k
        self.last_on_station[job.station] = k

    def to_individual(self) -> Individual:
        # amr_assignment is indexed by job id; jobs use ids 0..N-1
        return Individual(order=list(self.order), amr_assignment=[self.assignment[i] for i in range(self.n)])

    def features(self):
        t_now = min(self.avail.values())
        n, A = self.n, len(AMR_KEYS)

        # Action edges: what each (AMR, job) assignment would cost right now
        action = torch.zeros(A, n, ACTION_IN_DIM)
        best_end = [0.0] * n
        for k in range(n):
            if self.assigned[k]:
                continue
            ends = []
            for a_idx, amr in enumerate(AMR_KEYS):
                refill, travel, wait, end = self.outcome(amr, k)
                action[a_idx, k] = torch.tensor([
                    1.0 if refill else 0.0,
                    travel / DISTANCE_SCALE,
                    wait / TIME_SCALE,
                    (end - t_now) / TIME_SCALE,
                ])
                ends.append(end)
            best_end[k] = min(ends)

        amr_feat = []
        for amr in AMR_KEYS:
            p, av, inv = self.pos[amr], self.avail[amr], self.inv[amr]
            density = sum(1 for o in AMR_KEYS if o != amr and heuristic(p, self.pos[o]) <= 2) / max(A - 1, 1)
            amr_feat.append([
                1.0 if av > t_now else 0.0,
                (av - t_now) / TIME_SCALE,
                inv.get("A", 0) / REFILL_AMOUNT,
                inv.get("B", 0) / REFILL_AMOUNT,
                inv.get("C", 0) / REFILL_AMOUNT,
                self.amr_count[amr] / max(n, 1),
                self.amr_load[amr] / WORKLOAD_SCALE,
                min(heuristic(p, DOCK_POSITIONS[d]) for d in DOCK_KEYS) / DISTANCE_SCALE,
                density,
                p[0] / max(GRID_MAX_X, 1),
                p[1] / max(GRID_MAX_Y, 1),
            ])

        unassigned = [k for k in range(n) if not self.assigned[k]]
        dock_feat = []
        for d in DOCK_KEYS:
            dp = DOCK_POSITIONS[d]
            if d in STATIONS:
                live = [(s, e) for s, e in self.calendar[d] if e > t_now]
                demand = sum(self.jobs[k].duration for k in unassigned if self.jobs[k].station == d) / WORKLOAD_SCALE
                free_in = max((e for _, e in live), default=t_now) - t_now
                dock_feat.append([
                    1.0, 0.0,
                    free_in / TIME_SCALE,
                    len(live) / A,
                    sum(e - max(s, t_now) for s, e in live) / WORKLOAD_SCALE,
                    demand,
                    dp[0] / max(GRID_MAX_X, 1),
                    dp[1] / max(GRID_MAX_Y, 1),
                ])
            else:
                mat = d[len("supply_"):]
                dock_feat.append([
                    0.0, 1.0, 0.0, 0.0, 0.0,
                    sum(1 for k in unassigned if self.jobs[k].type_ == mat) / max(n, 1),
                    dp[0] / max(GRID_MAX_X, 1),
                    dp[1] / max(GRID_MAX_Y, 1),
                ])

        job_feat = []
        for k, job in enumerate(self.jobs):
            sp = STATIONS[job.station]
            job_feat.append([
                1.0,
                job.duration / 25.0,
                1.0 if job.type_ == "A" else 0.0,
                1.0 if job.type_ == "B" else 0.0,
                1.0 if job.type_ == "C" else 0.0,
                1.0 if self.assigned[k] else 0.0,
                (self.carrier[k] + 1) / A if self.carrier[k] is not None else 0.0,
                (STATION_KEYS.index(job.station) + 1) / len(STATION_KEYS),
                0.0 if self.assigned[k] else (best_end[k] - t_now) / TIME_SCALE,
                sp[0] / max(GRID_MAX_X, 1),
                sp[1] / max(GRID_MAX_Y, 1),
            ])

        assigned = torch.tensor(self.assigned, dtype=torch.bool)
        return {
            "amr": torch.tensor(amr_feat, dtype=torch.float32),
            "job": torch.tensor(job_feat, dtype=torch.float32),
            "dock": torch.tensor(dock_feat, dtype=torch.float32),
            "action": action,
            "actor_mask": assigned.unsqueeze(0).expand(A, n).clone(),  # True = illegal
            "job_mask": assigned,
            "adj": self.adj.clone(),
            "station_idx": self.station_idx,
            "supply_idx": self.supply_idx,
        }


def stack_features(feats: List[dict], device) -> dict:
    return {k: torch.stack([f[k] for f in feats]).to(device) for k in feats[0]}


# ==========================================
# GNN Architecture (extend_GNN style)
# ==========================================
class GINConv(nn.Module):
    def __init__(self, hidden_dim: int, eps_init: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps_init))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj):
        # Mean aggregation keeps message scale stable as the sequence-arc graph grows.
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        agg = torch.bmm(adj, x) / degree
        return self.norm(self.mlp((1.0 + self.eps) * x + agg))


class SchedulerGNN(nn.Module):
    """AMR / job / service-point graph scorer for (AMR, job) assignments.

    1. AMR and dock (station + supply point) embeddings, one pre-LN residual attention
       round between them (zero-initialised so it starts as identity).
    2. Jobs fuse their station and supply-point embeddings residually, then pass through
       GIN layers over the sequence-arc job graph.
    3. Actor scores every (AMR, job) pair from [AMR, job, station, supply, action edge].
    The critic head is only used by actor-critic training.
    """

    def __init__(self, amr_in_dim=AMR_IN_DIM, job_in_dim=JOB_IN_DIM, dock_in_dim=DOCK_IN_DIM,
                 action_in_dim=ACTION_IN_DIM, hidden_dim=128, gin_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.amr_emb = nn.Sequential(nn.Linear(amr_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.job_emb = nn.Linear(job_in_dim, hidden_dim)
        self.dock_emb = nn.Sequential(nn.Linear(dock_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.job_dock_fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        self.dock_from_amr_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.dock_from_amr_norm_q = nn.LayerNorm(hidden_dim)
        self.dock_from_amr_norm_kv = nn.LayerNorm(hidden_dim)
        self.amr_self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.amr_self_norm = nn.LayerNorm(hidden_dim)
        self.amr_from_dock_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.amr_from_dock_norm_q = nn.LayerNorm(hidden_dim)
        self.amr_from_dock_norm_kv = nn.LayerNorm(hidden_dim)
        for attn in (self.dock_from_amr_attn, self.amr_self_attn, self.amr_from_dock_attn):
            nn.init.zeros_(attn.out_proj.weight)
            nn.init.zeros_(attn.out_proj.bias)

        self.gin_layers = nn.ModuleList([GINConv(hidden_dim) for _ in range(gin_layers)])
        self.actor = nn.Sequential(nn.Linear(hidden_dim * 4 + action_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    @staticmethod
    def _gather(dock_emb, idx):
        return torch.gather(dock_emb, 1, idx.unsqueeze(-1).expand(-1, -1, dock_emb.size(-1)))

    def encode(self, f):
        amr = self.amr_emb(f["amr"])
        dock = self.dock_emb(f["dock"])

        q, kv = self.dock_from_amr_norm_q(dock), self.dock_from_amr_norm_kv(amr)
        dock = dock + self.dock_from_amr_attn(q, kv, kv)[0]
        q = self.amr_self_norm(amr)
        amr = amr + self.amr_self_attn(q, q, q)[0]
        q, kv = self.amr_from_dock_norm_q(amr), self.amr_from_dock_norm_kv(dock)
        amr = amr + self.amr_from_dock_attn(q, kv, kv)[0]

        job_base = self.job_emb(f["job"])
        station = self._gather(dock, f["station_idx"])
        supply = self._gather(dock, f["supply_idx"])
        job = job_base + self.job_dock_fusion(torch.cat([job_base, station, supply], dim=-1))
        for gin in self.gin_layers:
            job = job + gin(job, f["adj"])
        return amr, job, dock, station, supply

    def forward(self, f):
        """f: dict from stack_features. Returns logits (batch, num_amrs, num_jobs), illegal = -inf."""
        amr, job, _, station, supply = self.encode(f)
        A, N = amr.size(1), job.size(1)
        pairs = torch.cat([
            amr.unsqueeze(2).expand(-1, -1, N, -1),
            job.unsqueeze(1).expand(-1, A, -1, -1),
            station.unsqueeze(1).expand(-1, A, -1, -1),
            supply.unsqueeze(1).expand(-1, A, -1, -1),
            f["action"],
        ], dim=-1)
        logits = self.actor(pairs).squeeze(-1)
        return logits.masked_fill(f["actor_mask"], float('-inf'))

    def forward_critic(self, f):
        amr, job, dock, _, _ = self.encode(f)
        keep = (~f["job_mask"]).float().unsqueeze(-1)
        job_pool = (job * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.critic(torch.cat([job_pool, amr.mean(dim=1), dock.mean(dim=1)], dim=-1)).squeeze(-1)


# ===== GNN Scheduling Heuristic Loop =====
def build_schedules(job_lists: List[Sequence[Job]], model: SchedulerGNN, deterministic: bool,
                    init_states: Optional[List[dict]] = None):
    """Build one schedule per entry of `job_lists` in a single batched pass.

    All instances must have the same number of jobs. Returns (individuals, log_prob [B]).
    """
    device = next(model.parameters()).device
    states = [ConstructionState(jobs, init_states[b] if init_states else None) for b, jobs in enumerate(job_lists)]
    B, n = len(states), states[0].n
    log_prob = torch.zeros(B, device=device)
    model.eval() if deterministic else model.train()
    prev_threads = torch.get_num_threads()
    if prev_threads > GNN_NUM_THREADS:
        torch.set_num_threads(GNN_NUM_THREADS)
    try:
        for _ in range(n):
            logits = model(stack_features([s.features() for s in states], device)).reshape(B, -1)
            if deterministic:
                actions = logits.argmax(dim=1)
            else:
                lp = F.log_softmax(logits, dim=1)
                actions = torch.distributions.Categorical(logits=lp).sample()
                log_prob = log_prob + lp.gather(1, actions.unsqueeze(1)).squeeze(1)
            for s, act in zip(states, actions.tolist()):
                s.apply(act // n, act % n)
    finally:
        torch.set_num_threads(prev_threads)
    return [s.to_individual() for s in states], log_prob


def solve_with_gnn(jobs, model, deterministic=True, init_state: dict = None):
    """
    Uses the GNN to autoregressively build a schedule.
    Returns: The final Individual, the total log probabilities of the sequence, and the total execution time of the building process.
    """
    start_time = time.perf_counter()
    if deterministic:
        with torch.no_grad():
            inds, log_prob = build_schedules([jobs], model, True, [init_state] if init_state else None)
    else:
        inds, log_prob = build_schedules([jobs], model, False, [init_state] if init_state else None)
    solve_dur = time.perf_counter() - start_time
    return inds[0], (0.0 if deterministic else log_prob[0]), solve_dur


if __name__ == "__main__":
    import numpy as np # Local import for seeds
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gantt", action="store_true", help="Plot Gantt Chart")
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file")
    parser.add_argument("--save_img", type=str, default="", help="Save the schedule Gantt chart to this file (e.g., schedule.png)")
    parser.add_argument("--collision_iters", type=int, default=collision_routing_iters, help="Number of collision routing iterations")
    parser.add_argument("--output_csv", type=str, default="GNN_summary_results.csv", help="Output CSV filename")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="SchedulerGNN weights")
    args = parser.parse_args()

    # 1. Setup environment and seed
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. Create Model
    gnn = SchedulerGNN()
    if load_gnn_weights(gnn, args.weights):
        print(f"Loaded trained weights from {args.weights}")

    # 3. Load Jobs
    if args.inbox:
        dispatch_events = load_dispatch_events(Path(args.inbox))
    else:
        dispatch_events = load_dispatch_events()
    target_index = os.environ.get(DISPATCH_EVENT_INDEX_ENV)

    output_filename = args.output_csv
    results_data = []

    print("=== Using GNN Logic ===")

    if dispatch_events:
        if target_index is not None:
             dispatch_events = [e for e in dispatch_events if str(e["index"]) == str(target_index)]

        for event in dispatch_events:
            print(f"\n=== Processing Dispatch Event {event['index']} (Jobs: {len(event['jobs'])}) ===")

            # a. Run GNN Inference
            best_ind, _, solve_dur_ns = solve_with_gnn(event["jobs"], gnn)

            # b. Apply Local Improve for routing/collision adjustment exactly identically to GA.py
            improve_start = time.perf_counter()
            best_ind = local_improve(best_ind, event["jobs"], max_iters=routing_iters)
            if args.collision_iters > 0:
                best_ind = local_improve(best_ind, event["jobs"], max_iters=args.collision_iters, check_collision=True)
            solve_dur_ns += (time.perf_counter() - improve_start)

            # c. Evaluate with Exact GA routing logic
            img_path = f"{args.save_img.split('.')[0]}_{event['index']}.png" if args.save_img else None
            makespan, computation_time = describe_solution_gnn(best_ind, event["jobs"], solve_time=solve_dur_ns, show_gantt=args.gantt, save_img=img_path)

            results_data.append([event['index'], f"{makespan:.2f}", f"{computation_time:.4f}"])
    else:
        print("No dispatch file found. Generating random jobs...")
        jobs = make_jobs()

        # a. Run GNN Inference
        best_ind, _, solve_dur_ns = solve_with_gnn(jobs, gnn)

        # b. Apply Local Improve for routing/collision adjustment exactly identically to GA.py
        improve_start = time.perf_counter()
        best_ind = local_improve(best_ind, jobs, max_iters=routing_iters)
        if args.collision_iters > 0:
            best_ind = local_improve(best_ind, jobs, max_iters=args.collision_iters, check_collision=True)
        solve_dur_ns += (time.perf_counter() - improve_start)

        # c. Evaluate with Exact GA routing logic
        makespan, computation_time = describe_solution_gnn(best_ind, jobs, solve_time=solve_dur_ns, show_gantt=args.gantt, save_img=args.save_img)

        results_data.append(["random", f"{makespan:.2f}", f"{computation_time:.4f}"])

    # 4. Save metrics
    if results_data:
        with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Event_Index", "Makespan", "Computation_Time"])
            writer.writerows(results_data)
        print(f"\nSummary results saved to {output_filename}")
