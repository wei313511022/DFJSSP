#!/usr/bin/env python3
import os
import json
import time as pytime
from dataclasses import dataclass
from typing import List, Dict
 

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# Import visualization module
import visualization
from visualization import (
    JobVisual, update_title, rebuild_top_lane, draw_on_amr,
    setup_figure, setup_interactions, reset_state,
    amr_rects, amr_texts, amr_load, current_makespan, total_solve_time, waiting,
    UPDATE_INTERVAL_MS
)

# ===================== MILP map / LOGIC =====================
from map import (
    TIME_LIMIT,
    GRID_SIZE,
    TYPE_TO_MATERIAL_NODE,
    MATERIAL_PICK_QTY,
    P_NODES,
    JSON_STATION_MAPPING,
    M_SET,
    S_m,
    BARRIER_NODES,
    INBOX,
    SCHEDULE_OUTBOX,
)
from calcu_dist import make_calculate_distance


calculate_distance = make_calculate_distance(GRID_SIZE, BARRIER_NODES)


def solve_vrp_from_jobs(
    jobs,
    agv_available_time=None,
    agv_current_node=None,
    station_available_time=None,
    agv_inventory=None, 
    time_limit=TIME_LIMIT,
):
    if not jobs:
        return None

    # Global-state inputs
    if agv_available_time is None:
        agv_available_time = {m: 0.0 for m in M_SET}
    if agv_current_node is None:
        agv_current_node = {m: int(S_m[m]) for m in M_SET}
    if station_available_time is None:
        station_available_time = {int(n): 0.0 for n in JSON_STATION_MAPPING.values()}

    # Global inventory state across events:
    # - Each AMR can carry multiple material types concurrently.
    # - For each type, onboard inventory is capped at MATERIAL_PICK_QTY.
    # - Replenish that type to MATERIAL_PICK_QTY only when its onboard qty is 0.
    # Inventory is tracked as remaining units AFTER completing each job, so domain is [0, MATERIAL_PICK_QTY-1].
    TYPES = [str(k).upper() for k in TYPE_TO_MATERIAL_NODE.keys()]
    if agv_inventory is None:
        agv_inventory = {m: {} for m in M_SET}
    # Normalize inventory dict
    agv_inventory = {
        int(m): {str(t).upper(): int(q) for t, q in dict(agv_inventory.get(int(m), {})).items()}
        for m in M_SET
    }

    L_REAL = [j["jid"] for j in jobs]
    n_tasks = len(L_REAL)
    L_SET = range(1, n_tasks + 1)
    VIRTUAL_END = n_tasks + 1
    L_PRIME = range(0, VIRTUAL_END + 1)

    L_REAL_MAP = {idx: L_REAL[idx - 1] for idx in L_SET}
    TASK_DATA = {}
    for idx in L_SET:
        job = next(j for j in jobs if j["jid"] == L_REAL_MAP[idx])
        jtype = str(job.get("type", "?")).upper()
        g_l = int(job["g_l"])
        if g_l in BARRIER_NODES:
            raise ValueError(f"Job delivery node is a barrier: jid={job.get('jid')} g_l={g_l}")
        TASK_DATA[idx] = {
            "E_l": float(job["proc_time"]),
            "g_l": g_l,
            "type": jtype,
            "arrival_time": float(job.get("arrival_time", 0.0)),
        }

    # event-specific Big-M
    delivery_nodes = [TASK_DATA[i]["g_l"] for i in L_SET]
    D_max = max(calculate_distance(p, g) for p in P_NODES for g in delivery_nodes)
    S_max = max(calculate_distance(s, p) for s in S_m.values() for p in P_NODES)
    E_sum = sum(TASK_DATA[i]["E_l"] for i in L_SET)
    M_local = max(1.0, float(S_max + n_tasks * D_max + E_sum + 5.0))

    model = gp.Model("MILP_Scheduler")
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = 0.0
    model.Params.OutputFlag = 0
    # Helpful when you care about proving optimality (may take longer to find first feasible).
    try:
        model.Params.MIPFocus = 2
        model.Params.Presolve = 2
        model.Params.Cuts = 2
    except Exception:
        pass

    Y = model.addVars(L_SET, M_SET, vtype=GRB.BINARY, name="Y")
    W = model.addVars(L_PRIME, L_PRIME, M_SET, vtype=GRB.BINARY, name="W")

    # Material replenishment & inventory
    # - Each job consumes 1 unit of its type.
    # - Before executing a job, an AMR may optionally visit ONE material pickup node (any type) to replenish that type.
    #   This enables "順路補貨" even when the next job's type is different.
    # - Inventory is tracked as remaining units AFTER completing each job.
    #   After-job inventory can be MATERIAL_PICK_QTY for a type that was refilled but not consumed by the job.
    R_pick = model.addVars(L_SET, TYPES, vtype=GRB.BINARY, name="R_pick")
    Q_after = model.addVars(
        L_SET,
        TYPES,
        vtype=GRB.INTEGER,
        lb=0,
        ub=MATERIAL_PICK_QTY,
        name="Q_after",
    )
    T_Pick = model.addVars(L_SET, lb=0.0, name="T_pick")
    T_Del = model.addVars(L_SET, lb=0.0, name="T_del")
    T_End = model.addVars(L_SET, lb=0.0, name="T_end")
    # Explicit waiting (slack) between pick and delivery beyond pure travel.
    # Useful as a tie-break: prefer departing later over arriving early and queueing.
    T_Wait = model.addVars(L_SET, lb=0.0, name="T_wait")
    T_makespan = model.addVar(lb=0.0, name="T_makespan")
    
    # Station occupancy constraints - AGVs cannot be at same station simultaneously
    T_Station_Start = model.addVars(L_SET, lb=0.0, name="T_station_start")
    T_Station_End = model.addVars(L_SET, lb=0.0, name="T_station_end")
    Z_Station = model.addVars(L_SET, L_SET, vtype=GRB.BINARY, name="Z_station")

    model.addConstrs((T_makespan >= T_End[l] for l in L_SET), name="makespan_link")

    # ===================== Multi-stop pickup patterns =====================
    # Allow visiting multiple material pickup nodes (A/B/C) in an arbitrary order between tasks.
    # We enumerate all ordered pickup sequences with no repetition.
    PATTERNS = [()] \
        + [(t,) for t in TYPES] \
        + [(a, b) for a in TYPES for b in TYPES if b != a] \
        + [(a, b, c) for a in TYPES for b in TYPES for c in TYPES if (b != a and c != a and c != b)]
    P_SET = range(len(PATTERNS))

    _pat_seq = {p: tuple(PATTERNS[p]) for p in P_SET}
    _pat_inc = {(p, t): (1 if str(t).upper() in set(str(x).upper() for x in _pat_seq[p]) else 0) for p in P_SET for t in TYPES}

    # Choose exactly one pattern for each active incoming arc (start->job or job->job).
    U_start = model.addVars(L_SET, M_SET, P_SET, vtype=GRB.BINARY, name="U_start")
    U_arc = model.addVars(L_SET, L_SET, M_SET, P_SET, vtype=GRB.BINARY, name="U_arc")

    # Mutli-Objective:
    # 1) minimize makespan
    # 2) among optimal-makespan solutions, minimize total completion time
    try:
        model.setObjectiveN(T_makespan, index=0, priority=2, name="min_makespan")
        model.setObjectiveN(
            gp.quicksum(T_End[l] for l in L_SET),
            index=1,
            priority=1,
            name="min_total_completion",
        )
        # 3) tie-break: minimize waiting; add a tiny pickup-count term to avoid gratuitous pickups
        # when waiting is identical.
        model.setObjectiveN(
            gp.quicksum(T_Wait[l] for l in L_SET)
            + 1e-3 * gp.quicksum(R_pick[l, t] for l in L_SET for t in TYPES),
            index=2,
            priority=0,
            name="min_waiting_then_pickups",
        )
    except Exception:
        # if multi-objective is unavailable
        model.setObjective(
            T_makespan
            + 1e-4 * gp.quicksum(T_End[l] for l in L_SET)
            + 1e-6 * gp.quicksum(T_Wait[l] for l in L_SET)
            + 1e-9 * gp.quicksum(R_pick[l, t] for l in L_SET for t in TYPES),
            GRB.MINIMIZE,
        )

    # Multi-pick enabled: up to 3 distinct pickup types (A/B/C) may be visited before a job.
    model.addConstrs(
        (gp.quicksum(R_pick[l, t] for t in TYPES) <= len(TYPES) for l in L_SET),
        name="multi_pick_per_job",
    )

    # Pattern selection must match arc usage.
    model.addConstrs(
        (gp.quicksum(U_start[l, m, p] for p in P_SET) == W[0, l, m] for l in L_SET for m in M_SET),
        name="choose_pattern_start",
    )
    model.addConstrs(
        (
            gp.quicksum(U_arc[lp, l, m, p] for p in P_SET) == W[lp, l, m]
            for lp in L_SET
            for l in L_SET
            if lp != l
            for m in M_SET
        ),
        name="choose_pattern_arc",
    )

    # Link R_pick to the chosen incoming pattern.
    # R_pick[l,t] = 1 iff pickup type t appears in the chosen pattern on l's incoming arc.
    for l in L_SET:
        for t in TYPES:
            model.addConstr(
                R_pick[l, t]
                == gp.quicksum(_pat_inc[(p, t)] * U_start[l, m, p] for m in M_SET for p in P_SET)
                + gp.quicksum(
                    _pat_inc[(p, t)] * U_arc[lp, l, m, p]
                    for lp in L_SET
                    if lp != l
                    for m in M_SET
                    for p in P_SET
                ),
                name=f"link_R_pick_{l}_{t}",
            )

    model.addConstrs(
        (gp.quicksum(Y[l, m] for m in M_SET) == 1 for l in L_SET),
        name="assign",
    )

    # Validate job types and precompute each job's material node (fixed by type)
    material_node_of_l = {}
    for l in L_SET:
        t = str(TASK_DATA[l].get("type", "?")).upper()
        if t not in TYPE_TO_MATERIAL_NODE:
            raise ValueError(f"Unsupported job type '{t}' for jid={L_REAL_MAP[l]} (expected A/B/C)")
        material_node_of_l[l] = int(TYPE_TO_MATERIAL_NODE[t])

    for l in L_SET:
        model.addConstr(
            gp.quicksum(W[lp, l, m] for lp in L_PRIME if lp != l for m in M_SET) == 1,
            name=f"pred_{l}",
        )
        model.addConstr(
            gp.quicksum(W[l, ln, m] for ln in L_PRIME if ln != l for m in M_SET) == 1,
            name=f"succ_{l}",
        )
        model.addConstrs(
            (W[lp, l, m] <= Y[l, m] for lp in L_PRIME if lp != l for m in M_SET),
            name=f"link_in_{l}",
        )
        model.addConstrs(
            (W[l, ln, m] <= Y[l, m] for ln in L_PRIME if ln != l for m in M_SET),
            name=f"link_out_{l}",
        )

    # CRITICAL: an arc W[i,j,m] can only be used if BOTH endpoints are assigned to AMR m.
    # Without this, tasks on the same AMR may fail to get chained, allowing time overlap.
    for i in L_SET:
        for j in L_SET:
            if i == j:
                continue
            for m in M_SET:
                model.addConstr(W[i, j, m] <= Y[i, m], name=f"arc_src_{i}_{j}_{m}")
                model.addConstr(W[i, j, m] <= Y[j, m], name=f"arc_dst_{i}_{j}_{m}")

    # Virtual start/end arcs must also respect assignment of the real endpoint.
    for l in L_SET:
        for m in M_SET:
            model.addConstr(W[0, l, m] <= Y[l, m], name=f"arc_start_{l}_{m}")
            model.addConstr(W[l, VIRTUAL_END, m] <= Y[l, m], name=f"arc_end_{l}_{m}")

    for m in M_SET:
        # allow AMR m to be idle if no job is assigned to it
        model.addConstr(
            gp.quicksum(W[0, l, m] for l in L_SET) <= 1, name=f"start_{m}"
        )
        model.addConstr(
            gp.quicksum(W[l, VIRTUAL_END, m] for l in L_SET) <= 1, name=f"end_{m}"
        )


    model.addConstrs((W[l, 0, m] == 0 for l in L_SET for m in M_SET), name="no_ret0")

    # Virtual nodes constraints:
    # - Nothing can come *out* of VIRTUAL_END (it must be a sink).
    # - Nothing can go *into* 0 (already enforced for real l; also forbid VIRTUAL_END->0).
    model.addConstrs(
        (W[VIRTUAL_END, l, m] == 0 for l in L_PRIME for m in M_SET),
        name="no_out_of_virtual_end",
    )
    model.addConstrs(
        (W[VIRTUAL_END, 0, m] == 0 for m in M_SET),
        name="no_virtual_end_to_start",
    )
    model.addConstrs((W[0, 0, m] == 0 for m in M_SET), name="no_start_self")
    model.addConstrs((W[0, VIRTUAL_END, m] == 0 for m in M_SET), name="no_empty_arc")
    model.addConstrs((W[VIRTUAL_END, VIRTUAL_END, m] == 0 for m in M_SET), name="no_end_self")

    model.addConstrs(
        (T_End[l] == T_Del[l] + TASK_DATA[l]["E_l"] for l in L_SET),
        name="exec",
    )
    model.addConstrs(
        (T_Pick[l] >= TASK_DATA[l]["arrival_time"] for l in L_SET),
        name="arrival",
    )

    # Material constraints
    # 1) First-job refill requirement (multi-material): if an AMR starts with 0 qty for the job's type,
    #    and this job is the first on that AMR, then the AMR must pick up THAT type before the job.
    start_qty = {
        int(m): {t: int(agv_inventory.get(int(m), {}).get(t, 0) or 0) for t in TYPES}
        for m in M_SET
    }
    for l in L_SET:
        t_l = str(TASK_DATA[l]["type"]).upper()
        for m in M_SET:
            q0 = int(start_qty[int(m)].get(t_l, 0))
            if q0 <= 0:
                model.addConstr(R_pick[l, t_l] >= W[0, l, m], name=f"first_refill_required_{l}_{m}")

    # 2) Inventory transition + refill-only-when-empty rule (per material type)
    M_inv = MATERIAL_PICK_QTY

    # Start-arc inventory initialization (W[0,l,m]=1)
    for l in L_SET:
        t_l = str(TASK_DATA[l]["type"]).upper()
        for m in M_SET:
            for t in TYPES:
                q0 = int(start_qty[int(m)].get(t, 0))

                consume = 1 if t == t_l else 0
                target_if_pick = MATERIAL_PICK_QTY - consume

                # If we pick up type t before job l (and this is the first job on AMR m):
                # Q_after[l,t] = MATERIAL_PICK_QTY - consume
                model.addConstr(
                    Q_after[l, t] >= target_if_pick - M_inv * (2 - W[0, l, m] - R_pick[l, t]),
                    name=f"start_refill_lb_{l}_{m}_{t}",
                )
                model.addConstr(
                    Q_after[l, t] <= target_if_pick + M_inv * (2 - W[0, l, m] - R_pick[l, t]),
                    name=f"start_refill_ub_{l}_{m}_{t}",
                )

                # If we do NOT pick up type t before job l:
                # Q_after[l,t] = q0 - consume
                model.addConstr(
                    Q_after[l, t] >= (q0 - consume) - M_inv * (1 - W[0, l, m] + R_pick[l, t]),
                    name=f"start_consume_lb_{l}_{m}_{t}",
                )
                model.addConstr(
                    Q_after[l, t] <= (q0 - consume) + M_inv * (1 - W[0, l, m] + R_pick[l, t]),
                    name=f"start_consume_ub_{l}_{m}_{t}",
                )

    # Transition along a predecessor arc (W[lp,l,m]=1 with lp in L_SET)
    for lp in L_SET:
        for l in L_SET:
            if lp == l:
                continue
            t_l = str(TASK_DATA[l]["type"]).upper()
            for m in M_SET:
                for t in TYPES:
                    consume = 1 if t == t_l else 0
                    target_if_pick = MATERIAL_PICK_QTY - consume

                    # If we pick up type t before job l on this AMR: Q_after[l,t] = MATERIAL_PICK_QTY - consume
                    model.addConstr(
                        Q_after[l, t] >= target_if_pick - M_inv * (2 - W[lp, l, m] - R_pick[l, t]),
                        name=f"inv_refill_lb_{lp}_{l}_{m}_{t}",
                    )
                    model.addConstr(
                        Q_after[l, t] <= target_if_pick + M_inv * (2 - W[lp, l, m] - R_pick[l, t]),
                        name=f"inv_refill_ub_{lp}_{l}_{m}_{t}",
                    )

                    # If we do NOT pick up type t before job l: Q_after[l,t] = Q_after[lp,t] - consume
                    model.addConstr(
                        Q_after[l, t]
                        >= Q_after[lp, t] - consume - M_inv * (1 - W[lp, l, m]) - M_inv * R_pick[l, t],
                        name=f"inv_consume_lb_{lp}_{l}_{m}_{t}",
                    )
                    model.addConstr(
                        Q_after[l, t]
                        <= Q_after[lp, t] - consume + M_inv * (1 - W[lp, l, m]) + M_inv * R_pick[l, t],
                        name=f"inv_consume_ub_{lp}_{l}_{m}_{t}",
                    )
    
    # Station occupancy time constraints
    model.addConstrs(
        (T_Station_Start[l] == T_Del[l] for l in L_SET),
        name="station_start",
    )
    model.addConstrs(
        (T_Station_End[l] == T_End[l] for l in L_SET),
        name="station_end",
    )
    
    # AGVs cannot be at the same station simultaneously (within current event)
    for l1 in L_SET:
        for l2 in L_SET:
            if l1 >= l2:
                continue
            # Only constrain if both tasks are at the same delivery station
            if TASK_DATA[l1]["g_l"] == TASK_DATA[l2]["g_l"]:
                # Either task l1 completely finishes before l2 arrives (Z=1), 
                # or l2 completely finishes before l1 arrives (Z=0)
                # When Z=1: l2 arrives after l1 finishes (T_Del[l2] >= T_End[l1])
                # When Z=0: l1 arrives after l2 finishes (T_Del[l1] >= T_End[l2])
                model.addConstr(
                    T_Del[l2] >= T_End[l1] - M_local * (1 - Z_Station[l1, l2]),
                    name=f"no_overlap_1_{l1}_{l2}",
                )
                model.addConstr(
                    T_Del[l1] >= T_End[l2] - M_local * Z_Station[l1, l2],
                    name=f"no_overlap_2_{l1}_{l2}",
                )
    
    # Prevent overlap with already-scheduled work at stations (across events)
    for l in L_SET:
        station_node = int(TASK_DATA[l]["g_l"])
        prev_end = float(station_available_time.get(station_node, 0.0))
        if prev_end > 0.0:
            model.addConstr(T_Del[l] >= prev_end, name=f"after_prev_station_{l}")

    # Travel-time constraints with multi-stop pickup sequences
    # NOTE: Here we interpret T_Pick[l] as the DEPARTURE time from predecessor (or AMR start)
    # for the route to job l.

    for l in L_SET:
        g_l = int(TASK_DATA[l]["g_l"])

        # Depart after predecessor finishes (if that predecessor arc is chosen)
        for lp in L_SET:
            if lp == l:
                continue
            for m in M_SET:
                model.addConstr(
                    T_Pick[l] >= T_End[lp] - M_local * (1 - W[lp, l, m]),
                    name=f"depart_after_pred_{lp}_{l}_{m}",
                )

        # Depart after AMR is available (if start arc is chosen)
        for m in M_SET:
            t0 = float(agv_available_time.get(m, 0.0))
            model.addConstr(
                T_Pick[l] >= t0 - M_local * (1 - W[0, l, m]),
                name=f"depart_after_start_{l}_{m}",
            )

        # Start arcs: enforce arrival time using the chosen pickup pattern
        for m in M_SET:
            Snode = int(agv_current_node.get(m, S_m[m]))
            for p in P_SET:
                seq = _pat_seq[p]
                nodes = [Snode] + [int(TYPE_TO_MATERIAL_NODE[str(tt).upper()]) for tt in seq] + [g_l]
                travel = 0
                for a, b in zip(nodes, nodes[1:]):
                    travel += calculate_distance(int(a), int(b))
                model.addConstr(
                    T_Del[l] >= T_Pick[l] + travel - M_local * (1 - U_start[l, m, p]),
                    name=f"travel_start_{l}_{m}_{p}",
                )
                model.addConstr(
                    T_Wait[l] >= T_Del[l] - T_Pick[l] - travel - M_local * (1 - U_start[l, m, p]),
                    name=f"wait_start_{l}_{m}_{p}",
                )

        # Predecessor arcs: enforce arrival time using the chosen pickup pattern
        for lp in L_SET:
            if lp == l:
                continue
            g_lp = int(TASK_DATA[lp]["g_l"])
            for m in M_SET:
                for p in P_SET:
                    seq = _pat_seq[p]
                    nodes = [g_lp] + [int(TYPE_TO_MATERIAL_NODE[str(tt).upper()]) for tt in seq] + [g_l]
                    travel = 0
                    for a, b in zip(nodes, nodes[1:]):
                        travel += calculate_distance(int(a), int(b))
                    model.addConstr(
                        T_Del[l] >= T_Pick[l] + travel - M_local * (1 - U_arc[lp, l, m, p]),
                        name=f"travel_arc_{lp}_{l}_{m}_{p}",
                    )
                    model.addConstr(
                        T_Wait[l] >= T_Del[l] - T_Pick[l] - travel - M_local * (1 - U_arc[lp, l, m, p]),
                        name=f"wait_arc_{lp}_{l}_{m}_{p}",
                    )

    _t_wall_start = pytime.perf_counter()
    model.optimize()
    _t_wall = pytime.perf_counter() - _t_wall_start
    # Prefer Gurobi-reported solve time; fall back to wall time.
    try:
        solve_time = float(model.Runtime)
    except Exception:
        solve_time = float(_t_wall)

    if model.SolCount == 0:
        return None

    # pull solution values
    Y_vals       = model.getAttr("X", Y)
    T_pick_vals  = model.getAttr("X", T_Pick)
    T_del_vals   = model.getAttr("X", T_Del)
    T_end_vals   = model.getAttr("X", T_End)
    W_vals       = model.getAttr("X", W)
    R_vals       = model.getAttr("X", R_pick)
    U_start_vals = model.getAttr("X", U_start)
    U_arc_vals   = model.getAttr("X", U_arc)
    Q_after_vals  = model.getAttr("X", Q_after)

    # build sequences per AMR directly from Y + times
    seq_map = {m: [] for m in M_SET}

    for l in L_SET:
        # which AGV?
        assigned_m = None
        for m in M_SET:
            if Y_vals.get((l, m), 0.0) >= 0.5:
                assigned_m = m
                break
        if assigned_m is None:
            # shouldn't happen, but be safe
            continue

        # times
        pick_t = float(T_pick_vals.get(l, 0.0))
        del_t  = float(T_del_vals.get(l, 0.0))
        end_t  = float(T_end_vals.get(l, 0.0))

        # Resolve predecessor node for route reconstruction
        pred_lp = None
        pred_node = None
        for lp in L_PRIME:
            if lp == l:
                continue
            found = False
            for m in M_SET:
                if W_vals.get((lp, l, m), 0.0) >= 0.5:
                    pred_lp = int(lp)
                    if lp == 0:
                        pred_node = int(agv_current_node.get(int(m), S_m[int(m)]))
                    else:
                        pred_node = int(TASK_DATA[int(lp)]["g_l"])
                    found = True
                    break
            if found:
                break
        if pred_node is None:
            pred_node = int(material_node_of_l[l])

        # Resolve chosen pickup-sequence pattern for this incoming arc
        chosen_p = 0
        if pred_lp == 0:
            for m in M_SET:
                if W_vals.get((0, l, m), 0.0) >= 0.5:
                    for p in P_SET:
                        if U_start_vals.get((l, m, p), 0.0) >= 0.5:
                            chosen_p = int(p)
                            break
                if chosen_p != 0:
                    break
        elif pred_lp is not None and pred_lp != VIRTUAL_END:
            for m in M_SET:
                if W_vals.get((pred_lp, l, m), 0.0) >= 0.5:
                    for p in P_SET:
                        if U_arc_vals.get((pred_lp, l, m, p), 0.0) >= 0.5:
                            chosen_p = int(p)
                            break
                if chosen_p != 0:
                    break

        pickup_types = list(_pat_seq[int(chosen_p)])
        pickup_nodes = [int(TYPE_TO_MATERIAL_NODE[str(tt).upper()]) for tt in pickup_types]
        refill_now = len(pickup_nodes) > 0

        delivery_node = TASK_DATA[l]["g_l"]

        route_nodes = [int(pred_node)] + [int(n) for n in pickup_nodes] + [int(delivery_node)]
        route_legs = [
            int(calculate_distance(int(a), int(b)))
            for a, b in zip(route_nodes, route_nodes[1:])
        ]
        transport_time = float(sum(route_legs))

        # Waiting time at station queue = (departure->arrival) - (pure travel)
        wait_time = max(0.0, float(del_t - pick_t) - float(transport_time))
        
        # Remaining qty of this job's type after completion
        try:
            q_after_this = int(round(float(Q_after_vals.get((l, str(TASK_DATA[l].get("type", "?")).upper()), 0.0))))
        except Exception:
            q_after_this = None

        seq_map[assigned_m].append(
            {
                "idx": l,
                "jid": L_REAL_MAP[l],
                "assigned_agv": int(assigned_m),
                "type": TASK_DATA[l].get("type", "?"),
                "refill": bool(refill_now),
                "q_after": q_after_this,
                "pickup_node": int(pickup_nodes[0]) if pickup_nodes else int(pred_node),
                "pickup_nodes": [int(n) for n in pickup_nodes],
                "route_nodes": [int(n) for n in route_nodes],
                "route_legs": [int(x) for x in route_legs],
                "delivery_node": int(delivery_node),
                "pick_time": pick_t,
                "del_time": del_t,
                "end_time": end_t,
                "proc_time": float(TASK_DATA[l]["E_l"]),
                "transport_time": transport_time,
                "wait_time": wait_time,
            }
        )

    # sort tasks on each AMR by pick_time, then jid (for stable output)
    for m in M_SET:
        seq_map[m].sort(key=lambda j: (j["pick_time"], j["jid"]))

    # Add per-job context for time-aligned rendering: travel-to-pick and idle-before-pick.
    # This makes the time-aligned Gantt truly share one global clock across AMRs.
    for m in M_SET:
        prev_end = float(agv_available_time.get(m, 0.0))
        prev_node = int(agv_current_node.get(m, S_m[m]))
        for job in seq_map[m]:
            pick_t = float(job.get("pick_time", 0.0))

            # With multi-stop routes, pick_time is treated as departure time.
            # Route legs (prev->...->delivery) are stored in job["route_nodes"]/job["route_legs"] and drawn accordingly.
            to_pick = 0.0
            idle_before_pick = max(0.0, pick_t - prev_end)

            job["prev_end_time"] = prev_end
            job["prev_node"] = int(prev_node)
            job["to_pick_travel"] = to_pick
            job["idle_before_pick"] = idle_before_pick

            prev_end = float(job.get("end_time", prev_end))
            if job.get("delivery_node") is not None:
                prev_node = int(job.get("delivery_node"))

    # Persist per-AMR, per-type inventory across events (multi-material)
    end_inventory: Dict[int, Dict[str, int]] = {int(m): dict(agv_inventory.get(int(m), {})) for m in M_SET}
    for m in M_SET:
        if not seq_map.get(m):
            continue
        last_job = max(seq_map[m], key=lambda j: float(j.get("end_time", 0.0)))
        last_idx = int(last_job.get("idx"))
        inv_m: Dict[str, int] = {}
        for t in TYPES:
            try:
                q = int(round(float(Q_after_vals.get((last_idx, t), 0.0))))
            except Exception:
                q = 0
            q = max(0, min(int(MATERIAL_PICK_QTY), int(q)))
            if q > 0:
                inv_m[str(t).upper()] = int(q)
        end_inventory[int(m)] = inv_m

    makespan = max(float(v) for v in T_end_vals.values()) if T_end_vals else 0.0
    return {
        "sequence_map": seq_map,
        "makespan": float(makespan),
        "solve_time": float(solve_time),
        "end_inventory": end_inventory,
    }



# ===================== GLOBAL STATE (Multi-Material) =====================

# Global resource state across events (time is in the same units as MILP T_*)
agv_available_time: Dict[int, float] = {m: 0.0 for m in M_SET}
agv_current_node: Dict[int, int] = {m: int(S_m[m]) for m in M_SET}
station_available_time: Dict[int, float] = {int(n): 0.0 for n in JSON_STATION_MAPPING.values()}

# Global inventory state across events (MILP version: multi-material)
agv_inventory: Dict[int, Dict[str, int]] = {m: {} for m in M_SET}

# File tail progress tracking
_lines_consumed: int = 0


# ===================== INBOX INGEST + MILP + VISUAL =====================

def process_event_line_visual(line: str, ax, out_f):
    """Parse one dispatch event line, show top lane, run MILP, draw assignments, write schedule_outbox."""
    import visualization
    
    try:
        data = json.loads(line)
    except Exception:
        return

    dispatch_time = float(data.get("dispatch_time", 0.0))
    jobs_raw = data.get("jobs", [])
    if not jobs_raw:
        return

    # Build waiting list for top lane (show the event's jobs)
    visualization.waiting = []
    for j in jobs_raw:
        visualization.waiting.append(
            visualization.JobVisual(
                jid=int(j.get("jid")),
                jtype=str(j.get("type", "A")),
                proc_time=float(j.get("proc_time", 0.0)),
            )
        )
    visualization.rebuild_top_lane(ax)

    # Prepare jobs for MILP solver
    jobs_for_milp = []
    for j in jobs_raw:
        st = j.get("station")
        g_l = JSON_STATION_MAPPING.get(st)
        if g_l is None:
            continue
        jobs_for_milp.append(
            {
                "jid": int(j.get("jid")),
                "type": j.get("type"),
                "proc_time": float(j.get("proc_time", 0.0)),
                "station": int(st),
                "g_l": int(g_l),
                "arrival_time": 0.0,
            }
        )
    if not jobs_for_milp:
        return

    res = solve_vrp_from_jobs(
        jobs_for_milp,
        agv_available_time=agv_available_time,
        agv_current_node=agv_current_node,
        station_available_time=station_available_time,
        agv_inventory=agv_inventory,
    )
    if res is None:
        return
    
    # Store makespan for display
    visualization.current_makespan = res.get("makespan", 0.0)
    visualization.total_solve_time += float(res.get("solve_time", 0.0))

    # Flatten MILP result jobs by pick_time then jid
    all_jobs = []
    for m in M_SET:
        all_jobs.extend(res["sequence_map"].get(m, []))
    all_jobs.sort(key=lambda j: (j.get("pick_time", 0.0), j.get("jid", 0)))

    # Validate: detect any station-time overlaps in the solution intervals
    # (arrival at station = del_time, leave station = end_time)
    station_intervals: Dict[int, List[tuple]] = {}
    for job in all_jobs:
        try:
            node = int(job.get("delivery_node"))
            a = float(job.get("del_time", 0.0))
            b = float(job.get("end_time", 0.0))
            jid = int(job.get("jid"))
        except Exception:
            continue
        station_intervals.setdefault(node, []).append((a, b, jid))

    for node, intervals in station_intervals.items():
        intervals.sort(key=lambda t: (t[0], t[1], t[2]))
        for i in range(1, len(intervals)):
            prev_a, prev_b, prev_j = intervals[i - 1]
            cur_a, cur_b, cur_j = intervals[i]
            # overlap if current arrives before previous finished
            if cur_a < prev_b - 1e-6:
                print(
                    f"[StationOverlap] node={node} J{prev_j}({prev_a:.2f}-{prev_b:.2f}) overlaps J{cur_j}({cur_a:.2f}-{cur_b:.2f})"
                )
    
    # Update station availability (global timeline)
    for job in all_jobs:
        dn = int(job["delivery_node"])
        station_available_time[dn] = max(station_available_time.get(dn, 0.0), float(job["end_time"]))

    # Update AGV availability + current node (global timeline)
    for m in M_SET:
        seq = res["sequence_map"].get(m, [])
        if not seq:
            continue
        last = max(seq, key=lambda j: float(j.get("end_time", 0.0)))
        agv_available_time[m] = max(agv_available_time.get(m, 0.0), float(last.get("end_time", 0.0)))
        if last.get("delivery_node") is not None:
            agv_current_node[m] = int(last.get("delivery_node"))

    # Update AMR inventory across events (multi-material)
    try:
        end_inv = res.get("end_inventory")
        if isinstance(end_inv, dict):
            for m in M_SET:
                inv_m = end_inv.get(int(m), {})
                if isinstance(inv_m, dict):
                    agv_inventory[int(m)] = {str(t).upper(): int(q) for t, q in inv_m.items()}
    except Exception:
        pass

    # Draw assignments + write schedule_outbox.jsonl
    # Validate: within each AMR, processing intervals must not overlap in time.
    try:
        eps = 1e-6
        per_amr = {m: [] for m in M_SET}
        for job in all_jobs:
            m = int(job.get("assigned_agv"))
            per_amr[m].append(
                (
                    float(job.get("del_time", 0.0)),
                    float(job.get("end_time", 0.0)),
                    int(job.get("jid")),
                    int(job.get("delivery_node")) if job.get("delivery_node") is not None else None,
                )
            )
        for m, items in per_amr.items():
            items.sort(key=lambda x: (x[0], x[1], x[2]))
            for (s1, e1, jid1, dn1), (s2, e2, jid2, dn2) in zip(items, items[1:]):
                if s2 < e1 - eps:
                    print(
                        f"[AMROverlap] AMR {m}: J_{jid1}(D:{dn1}) [{s1:.3f},{e1:.3f}] overlaps J_{jid2}(D:{dn2}) [{s2:.3f},{e2:.3f}]"
                    )
    except Exception:
        pass

    for job in all_jobs:
        amr = job.get("assigned_agv")
        jid = job.get("jid")
        jtype = job.get("type") or "?"
        proc_time = float(job.get("proc_time", 0.0))
        transport_time = float(job.get("transport_time", 0.0))
        wait_time = float(job.get("wait_time", 0.0))
        pick_time = float(job.get("pick_time", 0.0))
        del_time = float(job.get("del_time", 0.0))
        end_time = float(job.get("end_time", 0.0))
        pickup_node = job.get("pickup_node")
        delivery_node = job.get("delivery_node")

        # recover station from delivery node
        station = None
        for st, del_node in JSON_STATION_MAPPING.items():
            if del_node == delivery_node:
                station = st
                break

        # draw on AMR lane (graphic) with transportation time and actual times
        vjob = visualization.JobVisual(
            jid=int(jid),
            jtype=jtype,
            proc_time=proc_time,
        )
        # attach time-aligned rendering context (if present)
        try:
            vjob.prev_end_time = float(job.get("prev_end_time", 0.0))
            vjob.prev_node = job.get("prev_node")
            vjob.to_pick_travel = float(job.get("to_pick_travel", 0.0))
            vjob.idle_before_pick = float(job.get("idle_before_pick", 0.0))
        except Exception:
            pass
        # attach multi-leg route info (if present)
        try:
            vjob.route_nodes = job.get("route_nodes")
            vjob.route_legs = job.get("route_legs")
        except Exception:
            pass
        visualization.draw_on_amr(ax, int(amr), vjob, transport_time=transport_time,
                   pick_time=pick_time, del_time=del_time, end_time=end_time,
                   pickup_node=pickup_node, delivery_node=delivery_node, wait_time=wait_time)

        # write schedule record (MILP-based dispatching result)
        rec = {
            "generated_at": dispatch_time,
            "amr": int(amr),
            "jid": int(jid),
            "type": jtype,
            "proc_time": proc_time,
            "transport_time": transport_time,
            "station": str(station) if station is not None else "?",
            "pickup_node": job.get("pickup_node"),
            "delivery_node": delivery_node,
            "refill": bool(job.get("refill", False)),
            "q_after": job.get("q_after"),
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()


# ===================== MAIN (timer + tail loop with graphics) =====================

def main():
    global _lines_consumed

    # Clear old schedule outbox
    os.makedirs(os.path.dirname(SCHEDULE_OUTBOX), exist_ok=True)
    open(SCHEDULE_OUTBOX, "w", encoding="utf-8").close()

    fig, ax = visualization.setup_figure()
    visualization.draw_static_panels(ax)
    visualization.update_title(ax)
    
    visualization.setup_interactions(fig, ax)

    timer = fig.canvas.new_timer(interval=visualization.UPDATE_INTERVAL_MS)

    def tick():
        global _lines_consumed
        changed = False

        try:
            with open(INBOX, "r", encoding="utf-8") as f_in, \
                 open(SCHEDULE_OUTBOX, "a", encoding="utf-8") as f_out:
                lines = f_in.readlines()
                if _lines_consumed < len(lines):
                    new_lines = lines[_lines_consumed:]
                    _lines_consumed = len(lines)
                    for line in new_lines:
                        if line.strip():
                            process_event_line_visual(line, ax, f_out)
                            changed = True
                            # Draw after each solved dispatch line
                            visualization.update_title(ax)
                            fig.canvas.draw_idle()
                            fig.canvas.flush_events()
        except FileNotFoundError:
            pass

        if changed:
            visualization.update_title(ax)
            fig.canvas.draw_idle()

        timer.start()

    timer.add_callback(tick)
    timer.start()
    plt.show()


if __name__ == "__main__":
    main()