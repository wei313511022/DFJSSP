#!/usr/bin/env python3
import os
import json
import time as pytime
from dataclasses import dataclass
from typing import List, Dict
from collections import deque
from functools import lru_cache

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# ===================== MILP CONFIG / LOGIC =====================
TIME_LIMIT = None
GRID_SIZE = 10
TYPE_TO_MATERIAL_NODE = {"A": 8, "B": 5, "C": 2}
MATERIAL_PICK_QTY = 3  # each pickup replenishes 3 units of the same material
P_NODES = set(TYPE_TO_MATERIAL_NODE.values())  # material pickup nodes
JSON_STATION_MAPPING = {5: 91, 4: 93, 3: 95, 2: 97, 1: 99}  # delivery nodes
M_SET = range(1, 4)   # 3 AGVs
S_m = {1: 2, 2: 5, 3: 8}  # AGV start nodes 

# Barrier nodes (AGVs cannot traverse these grid nodes)
BARRIER_NODES = {61, 62, 63, 65, 66, 67, 69, 70}

# Basic validation: key nodes must not be barriers
_fixed_nodes_to_check = set(S_m.values()) | set(TYPE_TO_MATERIAL_NODE.values()) | set(JSON_STATION_MAPPING.values())
_bad_fixed = sorted(int(n) for n in _fixed_nodes_to_check if int(n) in BARRIER_NODES)
if _bad_fixed:
    raise ValueError(f"Barrier nodes overlap with fixed start/pickup/delivery nodes: {_bad_fixed}")

# Paths
INBOX = "dispatch_inbox.jsonl"
SCHEDULE_OUTBOX = "Random_Job_Arrivals/schedule_outbox.jsonl"


def calculate_distance(node1, node2, grid_size=GRID_SIZE, barriers=None):
    """Shortest 4-neighbor grid distance avoiding barrier nodes.

    Uses cached distances with default barriers for speed.
    """

    n1 = int(node1)
    n2 = int(node2)
    if barriers is None or barriers == BARRIER_NODES:
        return _calculate_distance_cached(n1, n2)
    return _calculate_distance_uncached(n1, n2, grid_size=grid_size, barriers=barriers)


def _neighbors(node: int, grid_size: int):
    r, c = (node - 1) // grid_size, (node - 1) % grid_size
    if r > 0:
        yield node - grid_size
    if r < grid_size - 1:
        yield node + grid_size
    if c > 0:
        yield node - 1
    if c < grid_size - 1:
        yield node + 1


def _calculate_distance_uncached(n1: int, n2: int, *, grid_size: int, barriers):
    if n1 == n2:
        return 0

    barriers_set = set(int(b) for b in barriers)
    if n1 in barriers_set or n2 in barriers_set:
        raise ValueError(f"No path: start/end is a barrier (start={n1}, end={n2})")

    max_node = grid_size * grid_size
    if not (1 <= n1 <= max_node and 1 <= n2 <= max_node):
        raise ValueError(f"Node out of bounds for grid {grid_size}x{grid_size}: {n1}, {n2}")

    q = deque([n1])
    dist = {n1: 0}
    while q:
        cur = q.popleft()
        d = dist[cur]
        for nb in _neighbors(cur, grid_size):
            if nb in barriers_set:
                continue
            if nb in dist:
                continue
            nd = d + 1
            if nb == n2:
                return nd
            dist[nb] = nd
            q.append(nb)

    raise ValueError(f"No path found from {n1} to {n2} with barriers={sorted(barriers_set)}")


@lru_cache(maxsize=None)
def _calculate_distance_cached(n1: int, n2: int):
    # BARRIER_NODES and GRID_SIZE are treated as constants for caching.
    return _calculate_distance_uncached(n1, n2, grid_size=GRID_SIZE, barriers=BARRIER_NODES)


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

    # Global inventory state across events (optional; multi-material like DPR-1):
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

    model = gp.Model("Reschedule_event")
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = 0.0
    model.Params.OutputFlag = 1
    # Helpful when you care about proving optimality (may take longer to find first feasible).
    try:
        model.Params.MIPFocus = 2
        model.Params.Presolve = 2
        model.Params.Cuts = 2
    except Exception:
        pass

    Y = model.addVars(L_SET, M_SET, vtype=GRB.BINARY, name="Y")
    W = model.addVars(L_PRIME, L_PRIME, M_SET, vtype=GRB.BINARY, name="W")

    # Material replenishment & inventory (multi-material; matches DPR-1)
    # - Each job consumes 1 unit of its type.
    # - For each type, onboard qty is in [0, MATERIAL_PICK_QTY-1] after completing a job.
    # - If the job's type qty is 0 before service, AMR must visit that type's pickup and replenish to MATERIAL_PICK_QTY.
    # - AMR cannot replenish early (only when that type qty is 0).
    R_refill = model.addVars(L_SET, vtype=GRB.BINARY, name="R_refill")
    Q_after = model.addVars(
        L_SET,
        TYPES,
        vtype=GRB.INTEGER,
        lb=0,
        ub=MATERIAL_PICK_QTY - 1,
        name="Q_after",
    )
    T_Pick = model.addVars(L_SET, lb=0.0, name="T_pick")
    T_Del = model.addVars(L_SET, lb=0.0, name="T_del")
    T_End = model.addVars(L_SET, lb=0.0, name="T_end")
    T_makespan = model.addVar(lb=0.0, name="T_makespan")
    
    # Station occupancy constraints - AGVs cannot be at same station simultaneously
    T_Station_Start = model.addVars(L_SET, lb=0.0, name="T_station_start")
    T_Station_End = model.addVars(L_SET, lb=0.0, name="T_station_end")
    Z_Station = model.addVars(L_SET, L_SET, vtype=GRB.BINARY, name="Z_station")

    model.addConstrs((T_makespan >= T_End[l] for l in L_SET), name="makespan_link")

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
    except Exception:
        # if multi-objective is unavailable
        model.setObjective(T_makespan + 1e-4 * gp.quicksum(T_End[l] for l in L_SET), GRB.MINIMIZE)

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
    # 1) First-job refill logic (multi-material): based on this AMR's starting qty for the job's type.
    #    - If start qty is 0 -> must refill.
    #    - If start qty > 0 -> cannot refill.
    start_qty = {
        int(m): {t: int(agv_inventory.get(int(m), {}).get(t, 0) or 0) for t in TYPES}
        for m in M_SET
    }
    for l in L_SET:
        t_l = str(TASK_DATA[l]["type"]).upper()
        for m in M_SET:
            q0 = int(start_qty[int(m)].get(t_l, 0))
            if q0 <= 0:
                model.addConstr(R_refill[l] >= W[0, l, m], name=f"first_refill_required_{l}_{m}")
            else:
                model.addConstr(R_refill[l] <= 1 - W[0, l, m], name=f"first_refill_forbidden_{l}_{m}")

    # 2) Inventory transition + refill-only-when-empty rule (per material type)
    M_inv = MATERIAL_PICK_QTY

    # Start-arc inventory initialization (W[0,l,m]=1)
    for l in L_SET:
        t_l = str(TASK_DATA[l]["type"]).upper()
        for m in M_SET:
            # Other types unchanged from start inventory
            for t in TYPES:
                q0 = int(start_qty[int(m)].get(t, 0))
                if t != t_l:
                    model.addConstr(
                        Q_after[l, t] >= q0 - M_inv * (1 - W[0, l, m]),
                        name=f"start_keep_lb_{l}_{m}_{t}",
                    )
                    model.addConstr(
                        Q_after[l, t] <= q0 + M_inv * (1 - W[0, l, m]),
                        name=f"start_keep_ub_{l}_{m}_{t}",
                    )

            # This job's type transition depends on whether refill happens
            q0_t = int(start_qty[int(m)].get(t_l, 0))

            # If refill: Q_after = MATERIAL_PICK_QTY-1
            model.addConstr(
                Q_after[l, t_l] >= (MATERIAL_PICK_QTY - 1) - M_inv * (2 - W[0, l, m] - R_refill[l]),
                name=f"start_refill_lb_{l}_{m}",
            )
            model.addConstr(
                Q_after[l, t_l] <= (MATERIAL_PICK_QTY - 1) + M_inv * (2 - W[0, l, m] - R_refill[l]),
                name=f"start_refill_ub_{l}_{m}",
            )

            # If no refill: Q_after = q0_t - 1
            model.addConstr(
                Q_after[l, t_l] >= (q0_t - 1) - M_inv * (1 - W[0, l, m] + R_refill[l]),
                name=f"start_consume_lb_{l}_{m}",
            )
            model.addConstr(
                Q_after[l, t_l] <= (q0_t - 1) + M_inv * (1 - W[0, l, m] + R_refill[l]),
                name=f"start_consume_ub_{l}_{m}",
            )

    # Transition along a predecessor arc (W[lp,l,m]=1 with lp in L_SET)
    for lp in L_SET:
        for l in L_SET:
            if lp == l:
                continue
            t_l = str(TASK_DATA[l]["type"]).upper()
            for m in M_SET:
                for t in TYPES:
                    if t != t_l:
                        # Other types unchanged
                        model.addConstr(
                            Q_after[l, t] - Q_after[lp, t] <= M_inv * (1 - W[lp, l, m]),
                            name=f"keep_type_ub_{lp}_{l}_{m}_{t}",
                        )
                        model.addConstr(
                            Q_after[lp, t] - Q_after[l, t] <= M_inv * (1 - W[lp, l, m]),
                            name=f"keep_type_lb_{lp}_{l}_{m}_{t}",
                        )
                    else:
                        # If refill before l, then predecessor inventory of this type must be empty
                        model.addConstr(
                            Q_after[lp, t_l] <= M_inv * (2 - W[lp, l, m] - R_refill[l]),
                            name=f"refill_only_when_empty_ub_{lp}_{l}_{m}",
                        )

                        # If we do NOT refill before l, predecessor inventory must be >= 1
                        model.addConstr(
                            Q_after[lp, t_l] >= 1 - M_inv * (1 - W[lp, l, m]) - M_inv * R_refill[l],
                            name=f"no_refill_requires_stock_lb_{lp}_{l}_{m}",
                        )

                        # Refill transition: if W=1 and R=1 -> Q_after[l,t_l] = MATERIAL_PICK_QTY-1
                        model.addConstr(
                            Q_after[l, t_l] >= (MATERIAL_PICK_QTY - 1) - M_inv * (2 - W[lp, l, m] - R_refill[l]),
                            name=f"inv_refill_lb_{lp}_{l}_{m}",
                        )
                        model.addConstr(
                            Q_after[l, t_l] <= (MATERIAL_PICK_QTY - 1) + M_inv * (2 - W[lp, l, m] - R_refill[l]),
                            name=f"inv_refill_ub_{lp}_{l}_{m}",
                        )

                        # No-refill transition: if W=1 and R=0 -> Q_after[l,t_l] = Q_after[lp,t_l] - 1
                        model.addConstr(
                            Q_after[l, t_l]
                            >= Q_after[lp, t_l]
                            - 1
                            - M_inv * (1 - W[lp, l, m])
                            - M_inv * R_refill[l],
                            name=f"inv_consume_lb_{lp}_{l}_{m}",
                        )
                        model.addConstr(
                            Q_after[l, t_l]
                            <= Q_after[lp, t_l]
                            - 1
                            + M_inv * (1 - W[lp, l, m])
                            + M_inv * R_refill[l],
                            name=f"inv_consume_ub_{lp}_{l}_{m}",
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

    # Travel-time constraints with optional refill/pickup
    # Semantics:
    # - If R_refill[l]=1: AMR must go to the material node for type(l) before delivering job l.
    # - If R_refill[l]=0: AMR goes directly from predecessor delivery to this delivery (no pickup).

    for l in L_SET:
        g_l = int(TASK_DATA[l]["g_l"])
        p_l = int(material_node_of_l[l])
        d_pick_to_del = calculate_distance(p_l, g_l)

        # If we refill, delivery happens after reaching pickup node then traveling to delivery
        model.addConstr(
            T_Del[l] >= T_Pick[l] + d_pick_to_del - M_local * (1 - R_refill[l]),
            name=f"refill_pick_to_del_{l}",
        )

        for lp in L_SET:
            if lp == l:
                continue
            g_lp = int(TASK_DATA[lp]["g_l"])
            d_del_to_pick = calculate_distance(g_lp, p_l)
            d_direct = calculate_distance(g_lp, g_l)
            for m in M_SET:
                # If we refill before l (W=1,R=1), reach pickup after predecessor ends + travel
                model.addConstr(
                    T_Pick[l] >= T_End[lp] + d_del_to_pick - M_local * (2 - W[lp, l, m] - R_refill[l]),
                    name=f"seq_refill_to_pick_{lp}_{l}_{m}",
                )

                # If we do NOT refill before l (W=1,R=0), 'pickup time' is just the departure time from predecessor delivery
                model.addConstr(
                    T_Pick[l] >= T_End[lp] - M_local * ((1 - W[lp, l, m]) + R_refill[l]),
                    name=f"seq_norefill_depart_{lp}_{l}_{m}",
                )
                model.addConstr(
                    T_Del[l] >= T_Pick[l] + d_direct - M_local * ((1 - W[lp, l, m]) + R_refill[l]),
                    name=f"seq_norefill_direct_{lp}_{l}_{m}",
                )

        # Start constraints (from each AMR's current node/time)
        for m in M_SET:
            Snode = int(agv_current_node.get(m, S_m[m]))
            t0 = float(agv_available_time.get(m, 0.0))
            d_sp = calculate_distance(Snode, p_l)
            model.addConstr(
                T_Pick[l] >= t0 + d_sp - M_local * (2 - W[0, l, m] - R_refill[l]),
                name=f"start_refill_to_pick_{l}_{m}",
            )

            # If we do NOT refill and l is the first job on AMR m, depart from the AMR's current node.
            d_start_direct = calculate_distance(Snode, g_l)
            model.addConstr(
                T_Pick[l] >= t0 - M_local * ((1 - W[0, l, m]) + R_refill[l]),
                name=f"start_norefill_depart_{l}_{m}",
            )
            model.addConstr(
                T_Del[l] >= T_Pick[l] + d_start_direct - M_local * ((1 - W[0, l, m]) + R_refill[l]),
                name=f"start_norefill_direct_{l}_{m}",
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
    R_vals       = model.getAttr("X", R_refill)
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

        # Determine the effective "pickup_node" for the delivery leg.
        # - If refill happens for this job: pickup_node is the material node of the job type (A@7,B@4,C@1).
        # - If no refill: pickup_node is the predecessor delivery node (direct travel).
        chosen_pick = None
        refill_now = bool(R_vals.get(l, 0.0) >= 0.5)
        if refill_now:
            chosen_pick = int(material_node_of_l[l])
        else:
            pred_delivery = None
            for lp in L_PRIME:
                if lp == l:
                    continue
                for m in M_SET:
                    if W_vals.get((lp, l, m), 0.0) >= 0.5:
                        if lp == 0:
                            # first job on this AMR: depart from its current node
                            pred_delivery = int(agv_current_node.get(int(m), S_m[int(m)]))
                        elif lp == VIRTUAL_END:
                            pred_delivery = None
                        else:
                            pred_delivery = int(TASK_DATA[int(lp)]["g_l"])
                        break
                if pred_delivery is not None or any(
                    W_vals.get((lp, l, m), 0.0) >= 0.5 for m in M_SET
                ):
                    break
            # If predecessor couldn't be resolved (shouldn't happen), fall back to type material node.
            chosen_pick = int(pred_delivery) if pred_delivery is not None else int(material_node_of_l[l])

        delivery_node = TASK_DATA[l]["g_l"]

        # Calculate transportation time as true travel distance.
        # Note: del_t - pick_t may include waiting (e.g., station queue), which should not be shown as "transport".
        if chosen_pick is not None:
            transport_time = float(calculate_distance(int(chosen_pick), int(delivery_node)))
        else:
            transport_time = max(0.0, float(del_t - pick_t))

        # Waiting time = (time from pick to arrival) - (true travel time)
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
                "pickup_node": int(chosen_pick) if chosen_pick is not None else None,
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
            pnode = job.get("pickup_node")
            if pnode is None:
                to_pick = 0.0
            else:
                to_pick = float(calculate_distance(prev_node, int(pnode)))
            pick_t = float(job.get("pick_time", 0.0))
            idle_before_pick = max(0.0, pick_t - prev_end - to_pick)

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
            q = max(0, min(int(MATERIAL_PICK_QTY - 1), int(q)))
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



# ===================== VISUAL CONFIG / STATE  =====================

# Visual config
AMR_COUNT          = 3
UPDATE_INTERVAL_MS = 250

DISPATCHING_LEFT_LABEL_PAD = 7.25
QUEUE_LEFT_LABEL_PAD = 5.25
VIEW_WIDTH     = 60.0

AX_Y_MIN, AX_Y_MAX = 0.0, 2.0
TOP_Y_CENTER  = 1.25
TOP_LANE_H    = 0.5

BOTTOM_MIN    = 0.0
BOTTOM_HEIGHT = (AX_Y_MAX - AX_Y_MIN) / 2.0
AMR_Y_CENTERS = [BOTTOM_MIN + (i + 0.5) * (BOTTOM_HEIGHT / AMR_COUNT)
                 for i in range(AMR_COUNT)]
AMR_LANE_H    = BOTTOM_HEIGHT / AMR_COUNT * 0.7

_cycle = plt.rcParams.get("axes.prop_cycle", None)
_cycle_list = _cycle.by_key()["color"] if _cycle else ["C0", "C1", "C2", "C3"]
TYPE_COLORS = {"A": _cycle_list[0], "B": _cycle_list[1], "C": _cycle_list[2]}
TRANSPORT_COLOR = "lightgray"  # Color for transportation time segments
WAIT_COLOR = TRANSPORT_COLOR    # Reuse same color; differentiate by hatch

@dataclass
class JobVisual:
    jid: int
    jtype: str
    proc_time: float

# Top lane (dispatching queue) state
waiting: List[JobVisual] = []
rects_top: List[Rectangle] = []
texts_top: List = []

# Bottom AMR lanes
amr_rects: Dict[int, List[Rectangle]] = {i: [] for i in range(1, AMR_COUNT+1)}
amr_texts: Dict[int, List]           = {i: [] for i in range(1, AMR_COUNT+1)}
amr_load:  Dict[int, float] = {i: 0.0 for i in range(1, AMR_COUNT+1)}

# File tail progress
_lines_consumed = 0
current_makespan = 0.0  # Store the latest makespan
total_solve_time = 0.0  # Sum of solver runtimes across processed events (seconds)

# Global resource state across events (time is in the same units as MILP T_*)
agv_available_time: Dict[int, float] = {m: 0.0 for m in M_SET}
agv_current_node: Dict[int, int] = {m: int(S_m[m]) for m in M_SET}
station_available_time: Dict[int, float] = {int(n): 0.0 for n in JSON_STATION_MAPPING.values()}

# Global inventory state across events (MILP version: multi-material like DPR-1)
agv_inventory: Dict[int, Dict[str, int]] = {m: {} for m in M_SET}


def remove_artists(rects, texts):
    for a in rects:
        try:
            a.remove()
        except Exception:
            pass
    for t in texts:
        try:
            t.remove()
        except Exception:
            pass
    rects.clear()
    texts.clear()


def rebuild_top_lane(ax):
    """Show the current event's jobs in the top dispatching lane."""
    remove_artists(rects_top, texts_top)
    x = QUEUE_LEFT_LABEL_PAD
    for j in waiting:
        r = Rectangle(
            (x + 2.0, TOP_Y_CENTER),
            j.proc_time / 2.0,
            TOP_LANE_H,
            linewidth=1.2,
            edgecolor="black",
            facecolor=TYPE_COLORS.get(j.jtype, "C3"),
            clip_on=True,
        )
        ax.add_patch(r)
        rects_top.append(r)
        t = ax.text(
            x + j.proc_time / 4.0 + 2.0,
            TOP_Y_CENTER + TOP_LANE_H / 2.0,
            f"J_{j.jid}",
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            clip_on=True,
        )
        texts_top.append(t)
        x += j.proc_time / 2.0


def draw_on_amr(ax, amr_id: int, j: JobVisual, transport_time: float = 0.0, 
                pick_time: float = 0.0, del_time: float = 0.0, end_time: float = 0.0,
                pickup_node: int = None, delivery_node: int = None, wait_time: float = 0.0):
    """Draw one assigned job on an AMR lane (time-aligned only)."""
    y = AMR_Y_CENTERS[amr_id - 1]

    base_x = DISPATCHING_LEFT_LABEL_PAD
    scale = 1.0 / 2.5

    prev_end_time = float(getattr(j, "prev_end_time", 0.0))
    prev_node = getattr(j, "prev_node", None)
    to_pick_travel = float(getattr(j, "to_pick_travel", 0.0))
    idle_before_pick = float(getattr(j, "idle_before_pick", 0.0))

    pick_time = float(pick_time)
    del_time = float(del_time)
    end_time = float(end_time)
    transport_time = float(transport_time)
    wait_time = float(wait_time)

    # Build a clean, non-overlapping timeline on this AMR lane.
    t0 = prev_end_time
    t1 = t0 + max(0.0, to_pick_travel)  # arrive pickup
    t2 = pick_time  # start pick
    t3 = t2 + max(0.0, transport_time)  # finish travel to delivery
    t4 = del_time  # arrive/enter station
    t5 = end_time  # finish processing

    # Guard against small numerical inversions
    t1 = min(t1, t2)
    t3 = min(t3, t4)

    def _rect(t_start: float, t_end: float, *, face, hatch, alpha, z, label=None):
        w = max(0.0, (t_end - t_start)) * scale
        if w <= 0:
            return
        x0 = base_x + t_start * scale
        rseg = Rectangle(
            (x0, y - AMR_LANE_H / 2.0),
            w,
            AMR_LANE_H,
            linewidth=0.0,
            # Hatch color follows edgecolor; keep linewidth=0 to avoid border artifacts.
            edgecolor="gray",
            facecolor=face,
            hatch=hatch,
            clip_on=True,
        )
        # Opaque segments + no stroke eliminates visual artifacts at boundaries.
        rseg.set_alpha(1.0)
        rseg.set_zorder(z)
        ax.add_patch(rseg)
        amr_rects[amr_id].append(rseg)
        if label and w > 1.0:
            t = ax.text(
                x0 + w / 2.0,
                y,
                label,
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                weight="bold",
                clip_on=True,
            )
            t.set_zorder(4)
            amr_texts[amr_id].append(t)

    # Pre-pick travel + idle
    _rect(
        t0,
        t1,
        face=TRANSPORT_COLOR,
        hatch="///",
        alpha=1.0,
        z=1,
        label=(
            (
                f"{int(prev_node)}→{int(pickup_node)}\n({(t1 - t0):.0f})"
                if (prev_node is not None and pickup_node is not None)
                else f"({(t1 - t0):.0f})"
            )
            if (t1 - t0) > 0
            else None
        ),
    )
    _rect(
        t1,
        t2,
        face=WAIT_COLOR,
        hatch="..",
        alpha=1.0,
        z=1,
        label=f"W\n({(t2 - t1):.0f})" if (t2 - t1) > 0 else None,
    )

    # Pick->delivery travel + station-wait
    _rect(
        t2,
        t3,
        face=TRANSPORT_COLOR,
        hatch="///",
        alpha=1.0,
        z=2,
        label=(
            (
                f"{int(pickup_node)}→{int(delivery_node)}\n({(t3 - t2):.0f})"
                if (pickup_node is not None and delivery_node is not None)
                else f"({(t3 - t2):.0f})"
            )
            if (t3 - t2) > 0
            else None
        ),
    )
    _rect(
        t3,
        t4,
        face=WAIT_COLOR,
        hatch="..",
        alpha=1.0,
        z=2,
        label=f"W\n({(t4 - t3):.0f})" if (t4 - t3) > 0 else None,
    )

    # Processing block (on top)
    proc_w = max(0.0, (t5 - t4)) * scale
    x_proc = base_x + t4 * scale
    r = Rectangle(
        (x_proc, y - AMR_LANE_H / 2.0),
        proc_w,
        AMR_LANE_H,
        linewidth=1.8,
        edgecolor="black",
        facecolor=TYPE_COLORS.get(j.jtype, "C3"),
        clip_on=True,
    )
    r.set_zorder(5)
    ax.add_patch(r)
    amr_rects[amr_id].append(r)

    node_label = f"D:{delivery_node}" if delivery_node else ""
    label_text = (
        f"J_{j.jid}\n({float(j.proc_time):.0f})\n{node_label}"
        if node_label
        else f"J_{j.jid}\n({float(j.proc_time):.0f})"
    )
    t = ax.text(
        x_proc + proc_w / 2.0,
        y,
        label_text,
        ha="center",
        va="center",
        fontsize=6,
        weight="bold",
        color="white",
        clip_on=True,
    )
    t.set_zorder(6)
    amr_texts[amr_id].append(t)

    amr_load[amr_id] += (
        float(j.proc_time)
        + transport_time
        + wait_time
        + max(0.0, to_pick_travel)
        + max(0.0, idle_before_pick)
    )
    return


def draw_static_panels(ax):
    band_frac = 0.12
    # Top panel
    top_panel = Rectangle(
        (0.0, 0.5),
        band_frac,
        0.5,
        transform=ax.transAxes,
        fill=False,
        linewidth=1.8,
        clip_on=False,
        zorder=3,
    )
    ax.add_patch(top_panel)
    tp = ax.text(
        band_frac * 0.5,
        0.75,
        "Dispatching\nQueue",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color="gray",
        clip_on=True,
        zorder=4,
    )
    tp.set_clip_path(top_panel)

    # Bottom panel (AMRs)
    bot_panel = Rectangle(
        (0.0, 0.0),
        band_frac,
        0.5,
        transform=ax.transAxes,
        fill=False,
        linewidth=1.8,
        clip_on=False,
        zorder=3,
    )
    ax.add_patch(bot_panel)

    for i in range(AMR_COUNT):
        y_frac = (i + 0.5) / AMR_COUNT * 0.5
        txt = ax.text(
            band_frac * 0.5,
            y_frac,
            f"AMR {i+1}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            color="gray",
            clip_on=True,
            zorder=4,
        )
        txt.set_clip_path(bot_panel)

    handles = [
        Patch(facecolor=TYPE_COLORS[k], edgecolor="black", label=f"Type {k}")
        for k in TYPE_COLORS
    ]
    handles.append(Patch(facecolor=TRANSPORT_COLOR, edgecolor="gray", hatch="///", label="Transportation"))
    handles.append(Patch(facecolor=WAIT_COLOR, edgecolor="gray", hatch="..", label="Waiting"))
    ax.legend(handles=handles, loc="upper right", frameon=True)


def update_title(ax):
    ax.set_title(
        f"AMR Scheduler with Gurobi | Makespan: {current_makespan:.1f} | Solve: {total_solve_time:.2f}s"
    )


# ===================== INBOX INGEST + MILP + VISUAL =====================

def process_event_line_visual(line: str, ax, out_f):
    """Parse one dispatch event line, show top lane, run MILP, draw assignments, write schedule_outbox."""
    global waiting, current_makespan, total_solve_time

    try:
        data = json.loads(line)
    except Exception:
        return

    dispatch_time = float(data.get("dispatch_time", 0.0))
    jobs_raw = data.get("jobs", [])
    if not jobs_raw:
        return

    # Build waiting list for top lane (show the event's jobs)
    waiting = []
    for j in jobs_raw:
        waiting.append(
            JobVisual(
                jid=int(j.get("jid")),
                jtype=str(j.get("type", "A")),
                proc_time=float(j.get("proc_time", 0.0)),
            )
        )
    rebuild_top_lane(ax)

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
    current_makespan = res.get("makespan", 0.0)
    total_solve_time += float(res.get("solve_time", 0.0))

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
        vjob = JobVisual(
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
        draw_on_amr(ax, int(amr), vjob, transport_time=transport_time,
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

    fig, ax = plt.subplots(figsize=(16, 5.2))
    ax.set_ylim(AX_Y_MIN, AX_Y_MAX)
    ax.set_xlim(0.0, VIEW_WIDTH)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)

    draw_static_panels(ax)
    update_title(ax)
    
    # Simple scrolling with mouse wheel and keyboard
    def on_scroll(event):
        if event.inaxes != ax:
            return
        xlim = ax.get_xlim()
        xrange = xlim[1] - xlim[0]
        # Scroll to move horizontally
        if event.button == 'up':
            shift = -xrange * 0.1  # Move left
        elif event.button == 'down':
            shift = xrange * 0.1   # Move right
        else:
            return
        ax.set_xlim([xlim[0] + shift, xlim[1] + shift])
        fig.canvas.draw_idle()
    
    def on_key(event):
        xlim = ax.get_xlim()
        xrange = xlim[1] - xlim[0]
        shift = 0
        
        if event.key == 'left':
            shift = -xrange * 0.15  # Move left
        elif event.key == 'right':
            shift = xrange * 0.15   # Move right
        elif event.key == 'home':
            # Go to beginning
            ax.set_xlim([0, xrange])
            fig.canvas.draw_idle()
            return
        else:
            return
        
        ax.set_xlim([xlim[0] + shift, xlim[1] + shift])
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Enable panning (dragging to scroll)
    pan_data = {'pressed': False, 'xpress': None, 'xlim': None}
    
    def on_press(event):
        if event.inaxes != ax:
            return
        pan_data['pressed'] = True
        pan_data['xpress'] = event.xdata
        pan_data['xlim'] = list(ax.get_xlim())
    
    def on_release(event):
        pan_data['pressed'] = False
        pan_data['xpress'] = None
    
    def on_motion(event):
        if not pan_data['pressed'] or event.inaxes != ax or pan_data['xpress'] is None:
            return
        dx = event.xdata - pan_data['xpress']
        cur_xlim = ax.get_xlim()
        new_xlim = [pan_data['xlim'][0] - dx, pan_data['xlim'][1] - dx]
        ax.set_xlim(new_xlim)
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    timer = fig.canvas.new_timer(interval=UPDATE_INTERVAL_MS)

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
                            update_title(ax)
                            fig.canvas.draw_idle()
                            fig.canvas.flush_events()
        except FileNotFoundError:
            pass

        if changed:
            update_title(ax)
            fig.canvas.draw_idle()

        timer.start()

    timer.add_callback(tick)
    timer.start()
    plt.show()


if __name__ == "__main__":
    main()