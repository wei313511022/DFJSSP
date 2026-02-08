#!/usr/bin/env python3
import os
import json
import time
from dataclasses import dataclass
from typing import List, Dict

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

from calcu_dist import make_calculate_distance

# ===================== MILP CONFIG / LOGIC (from MILP.py) =====================

TIME_LIMIT = 3600
GRID_SIZE = 10
P_NODES = {1, 6, 10}  # pickup candidate nodes 
JSON_STATION_MAPPING = {1: 93, 2: 96, 3: 98}  # delivery nodes
M_SET = range(1, 4)   # 3 AGVs
S_m = {1: 1, 2: 6, 3: 10}  # AGV start nodes

# Material management config
MAT_TYPE_TO_PICKUP = {"A": 1, "B": 6, "C": 10}  # Material type -> pickup node
MAT_CAPACITY = 10  # Each pickup action gives 10 units
MAT_INVENTORY_CAPACITY = 30  # Max material each AMR can carry (3 types * 10 units)

# Paths
INBOX = "dispatch_inbox.jsonl"
SCHEDULE_OUTBOX = "Random_Job_Arrivals/schedule_outbox.jsonl"


calculate_distance = make_calculate_distance(GRID_SIZE, ())


def solve_vrp_from_jobs(
    # Global variables
    jobs,
    agv_available_time=None,
    agv_current_node=None,
    station_available_time=None, 
    material_inventory=None,
    time_limit=TIME_LIMIT,
):
    if not jobs:
        return None

    # Global-state inputs initialization
    if agv_available_time is None:
        agv_available_time = {m: 0.0 for m in M_SET}
    if agv_current_node is None:
        agv_current_node = {m: int(S_m[m]) for m in M_SET}
    if station_available_time is None:
        station_available_time = {int(n): 0.0 for n in JSON_STATION_MAPPING.values()}
    if material_inventory is None:
        material_inventory = {m: {"A": 0, "B": 0, "C": 0} for m in M_SET}

    L_REAL = [j["jid"] for j in jobs]
    n_tasks = len(L_REAL)
    L_SET = range(1, n_tasks + 1)
    VIRTUAL_END = n_tasks + 1
    L_PRIME = range(0, VIRTUAL_END + 1)

    L_REAL_MAP = {idx: L_REAL[idx - 1] for idx in L_SET}
    TASK_DATA = {}
    for idx in L_SET:
        job = next(j for j in jobs if j["jid"] == L_REAL_MAP[idx])
        job_type = str(job.get("type", "?"))
        TASK_DATA[idx] = {
            "E_l": float(job["proc_time"]),
            "g_l": int(job["g_l"]),
            "type": job_type,
            "arrival_time": float(job.get("arrival_time", 0.0)),
            "mat_pickup_node": MAT_TYPE_TO_PICKUP.get(job_type, None),
        }

    # event-specific Big-M
    delivery_nodes = [TASK_DATA[i]["g_l"] for i in L_SET]
    # Allow "no pickup" by letting the pickup-node decision choose the predecessor location.
    # This enables direct station->station travel when material is already on-board.
    pick_nodes = set(int(x) for x in P_NODES)
    pick_nodes.update(int(x) for x in delivery_nodes)
    pick_nodes.update(int(x) for x in S_m.values())
    pick_nodes.update(int(x) for x in agv_current_node.values())
    PICK_NODES = sorted(pick_nodes)

    D_max = max(calculate_distance(p, g) for p in PICK_NODES for g in delivery_nodes)
    S_max = max(calculate_distance(s, p) for s in S_m.values() for p in PICK_NODES)
    E_sum = sum(TASK_DATA[i]["E_l"] for i in L_SET)
    M_local = max(1.0, float(S_max + n_tasks * D_max + E_sum + 5.0))

    model = gp.Model("Reschedule_event")
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0.0
    model.Params.OutputFlag = 1

    Y = model.addVars(L_SET, M_SET, vtype=GRB.BINARY, name="Y")
    W = model.addVars(L_PRIME, L_PRIME, M_SET, vtype=GRB.BINARY, name="W")
    A_P = model.addVars(L_SET, PICK_NODES, vtype=GRB.BINARY, name="A")
    T_Pick = model.addVars(L_SET, lb=0.0, name="T_pick")
    T_Del = model.addVars(L_SET, lb=0.0, name="T_del")
    T_End = model.addVars(L_SET, lb=0.0, name="T_end")
    T_makespan = model.addVar(lb=0.0, name="T_makespan")

    # Material refill decision: R[l,m]=1 means AMR m refills the job's material type before job l.
    R = model.addVars(L_SET, M_SET, vtype=GRB.BINARY, name="R_refill")

    # Per-type inventory AFTER completing job l on AMR m.
    MAT_TYPES = ("A", "B", "C")
    Inv = model.addVars(
        L_SET,
        M_SET,
        MAT_TYPES,
        vtype=GRB.INTEGER,
        lb=0,
        ub=int(MAT_CAPACITY),
        name="Inv",
    )
    
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
        (gp.quicksum(A_P[l, p] for p in PICK_NODES) == 1 for l in L_SET),
        name="one_pick",
    )
    model.addConstrs(
        (gp.quicksum(Y[l, m] for m in M_SET) == 1 for l in L_SET),
        name="assign",
    )

    # Refill only makes sense on the assigned AMR
    model.addConstrs((R[l, m] <= Y[l, m] for l in L_SET for m in M_SET), name="refill_link")

    # If refilling for job l, force its pickup-node to be the material pickup node for that type.
    # If not refilling, forbid choosing that material pickup node (prevents unnecessary detours).
    for l in L_SET:
        job_type = TASK_DATA[l]["type"]
        mat_node = MAT_TYPE_TO_PICKUP.get(job_type)
        if mat_node is None:
            continue
        rsum = gp.quicksum(R[l, m] for m in M_SET)
        model.addConstr(A_P[l, int(mat_node)] == rsum, name=f"refill_pick_node_{l}")

    # If NOT refilling, enforce that the chosen "pickup node" equals the current AMR location:
    # - If predecessor is job lp, the current location is lp's delivery node.
    # - If predecessor is virtual start 0, the current location is the AMR's current node.
    # This guarantees direct station->station travel between consecutive jobs of the same AMR.
    for l in L_SET:
        for lp in L_SET:
            if lp == l:
                continue
            prev_node = int(TASK_DATA[lp]["g_l"])
            if prev_node not in PICK_NODES:
                continue
            for m in M_SET:
                model.addConstr(
                    A_P[l, prev_node] >= W[lp, l, m] - R[l, m],
                    name=f"direct_pick_prev_{lp}_{l}_{m}",
                )
        for m in M_SET:
            start_node = int(agv_current_node.get(m, S_m[m]))
            if start_node in PICK_NODES:
                model.addConstr(
                    A_P[l, start_node] >= W[0, l, m] - R[l, m],
                    name=f"direct_pick_start_{l}_{m}",
                )

    # Inventory constraints: material is consumed by each job (1 unit of that job's type).
    # A refill sets that type's inventory to MAT_CAPACITY before consuming (so after job it is MAT_CAPACITY-1).
    BIG_INV = int(max(20, MAT_CAPACITY + 10))
    init_inv = {
        m: {t: int(material_inventory.get(m, {}).get(t, 0)) for t in MAT_TYPES}
        for m in M_SET
    }

    for l in L_SET:
        t_l = str(TASK_DATA[l]["type"])
        for m in M_SET:
            for t in MAT_TYPES:
                if t != t_l:
                    # Other material types carry over unchanged along the chosen predecessor arc.
                    for lp in L_SET:
                        if lp == l:
                            continue
                        model.addConstr(
                            Inv[l, m, t] <= Inv[lp, m, t] + BIG_INV * (1 - W[lp, l, m]),
                            name=f"inv_carry_ub_{t}_{lp}_{l}_{m}",
                        )
                        model.addConstr(
                            Inv[l, m, t] >= Inv[lp, m, t] - BIG_INV * (1 - W[lp, l, m]),
                            name=f"inv_carry_lb_{t}_{lp}_{l}_{m}",
                        )
                    model.addConstr(
                        Inv[l, m, t] <= init_inv[m][t] + BIG_INV * (1 - W[0, l, m]),
                        name=f"inv_carry0_ub_{t}_{l}_{m}",
                    )
                    model.addConstr(
                        Inv[l, m, t] >= init_inv[m][t] - BIG_INV * (1 - W[0, l, m]),
                        name=f"inv_carry0_lb_{t}_{l}_{m}",
                    )
                else:
                    # The job's own type: either refill, or consume from previous inventory.
                    # Refill case fixes inventory after the job.
                    model.addConstr(
                        Inv[l, m, t] <= (MAT_CAPACITY - 1) + BIG_INV * (1 - R[l, m]),
                        name=f"inv_refill_ub_{t}_{l}_{m}",
                    )
                    model.addConstr(
                        Inv[l, m, t] >= (MAT_CAPACITY - 1) - BIG_INV * (1 - R[l, m]),
                        name=f"inv_refill_lb_{t}_{l}_{m}",
                    )

                    # No-refill case: Inv = Inv_prev - 1 on the chosen predecessor arc.
                    for lp in L_SET:
                        if lp == l:
                            continue
                        model.addConstr(
                            Inv[l, m, t]
                            <= Inv[lp, m, t]
                            - 1
                            + BIG_INV * (1 - W[lp, l, m])
                            + BIG_INV * R[l, m],
                            name=f"inv_noref_ub_{t}_{lp}_{l}_{m}",
                        )
                        model.addConstr(
                            Inv[l, m, t]
                            >= Inv[lp, m, t]
                            - 1
                            - BIG_INV * (1 - W[lp, l, m])
                            - BIG_INV * R[l, m],
                            name=f"inv_noref_lb_{t}_{lp}_{l}_{m}",
                        )
                        # If no refill and this arc is chosen, predecessor inventory must be >= 1.
                        model.addConstr(
                            Inv[lp, m, t] >= 1 - BIG_INV * (1 - W[lp, l, m]) - BIG_INV * R[l, m],
                            name=f"inv_need1_{t}_{lp}_{l}_{m}",
                        )

                    # Start predecessor (lp=0): use initial inventory.
                    model.addConstr(
                        Inv[l, m, t]
                        <= init_inv[m][t]
                        - 1
                        + BIG_INV * (1 - W[0, l, m])
                        + BIG_INV * R[l, m],
                        name=f"inv0_noref_ub_{t}_{l}_{m}",
                    )
                    model.addConstr(
                        Inv[l, m, t]
                        >= init_inv[m][t]
                        - 1
                        - BIG_INV * (1 - W[0, l, m])
                        - BIG_INV * R[l, m],
                        name=f"inv0_noref_lb_{t}_{l}_{m}",
                    )
                    # If no refill and job starts from lp=0 on this AMR, initial inventory must be >= 1.
                    model.addConstr(
                        init_inv[m][t] >= 1 - BIG_INV * (1 - W[0, l, m]) - BIG_INV * R[l, m],
                        name=f"inv0_need1_{t}_{l}_{m}",
                    )

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

    for l in L_SET:
        g_l = TASK_DATA[l]["g_l"]
        min_d = min(calculate_distance(pp, g_l) for pp in PICK_NODES)
        for p in PICK_NODES:
            d_pg = calculate_distance(p, g_l)
            model.addConstr(
                T_Del[l] >= T_Pick[l] + d_pg - M_local * (1 - A_P[l, p]),
                name=f"trans_{l}_{p}",
            )
        model.addConstr(T_Del[l] >= T_Pick[l] + min_d, name=f"trans_lb_{l}")

    for l in L_SET:
        for lp in L_SET:
            if lp == l:
                continue
            g_lp = TASK_DATA[lp]["g_l"]
            for p in PICK_NODES:
                d_dp = calculate_distance(g_lp, p)
                model.addConstrs(
                    (
                        T_Pick[l]
                        >= T_End[lp]
                        + d_dp
                        - M_local * (2 - W[lp, l, m] - A_P[l, p])
                        for m in M_SET
                    ),
                    name=f"seq_{lp}_{l}_{p}",
                )
        for m in M_SET:
            # This event starts from the AGV's current node and current available time (global timeline)
            Snode = int(agv_current_node.get(m, S_m[m]))
            t0 = float(agv_available_time.get(m, 0.0))
            for p in PICK_NODES:
                d_sp = calculate_distance(Snode, p)
                model.addConstr(
                    T_Pick[l]
                    >= t0 + d_sp - M_local * (2 - W[0, l, m] - A_P[l, p]),
                    name=f"start_seq_{l}_{m}_{p}",
                )

    t_opt_start = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - t_opt_start
    if model.SolCount == 0:
        return None

    # pull solution values
    A_vals       = model.getAttr("X", A_P)
    Y_vals       = model.getAttr("X", Y)
    R_vals       = model.getAttr("X", R)
    T_pick_vals  = model.getAttr("X", T_Pick)
    T_del_vals   = model.getAttr("X", T_Del)
    T_end_vals   = model.getAttr("X", T_End)

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

        # chosen pickup node
        chosen_pick = None
        for p in PICK_NODES:
            if A_vals.get((l, p), 0.0) >= 0.5:
                chosen_pick = p
                break

        delivery_node = TASK_DATA[l]["g_l"]

        # Calculate transportation time as true travel distance.
        # Note: del_t - pick_t may include waiting (e.g., station queue), which should not be shown as "transport".
        if chosen_pick is not None:
            transport_time = float(calculate_distance(int(chosen_pick), int(delivery_node)))
        else:
            transport_time = max(0.0, float(del_t - pick_t))

        # Waiting time = (time from pick to arrival) - (true travel time)
        wait_time = max(0.0, float(del_t - pick_t) - float(transport_time))
        
        # refill decision for this (l, m)
        refilled = bool(R_vals.get((l, assigned_m), 0.0) >= 0.5)

        seq_map[assigned_m].append(
            {
                "idx": l,
                "jid": L_REAL_MAP[l],
                "assigned_agv": int(assigned_m),
                "type": TASK_DATA[l].get("type", "?"),
                "pickup_node": int(chosen_pick) if chosen_pick is not None else None,
                "delivery_node": int(delivery_node),
                "pick_time": pick_t,
                "del_time": del_t,
                "end_time": end_t,
                "proc_time": float(TASK_DATA[l]["E_l"]),
                "transport_time": transport_time,
                "wait_time": wait_time,
                "material_refill_required": bool(refilled),
                "refill_pickup_node": int(MAT_TYPE_TO_PICKUP.get(TASK_DATA[l].get("type", "?")))
                if refilled and MAT_TYPE_TO_PICKUP.get(TASK_DATA[l].get("type", "?")) is not None
                else None,
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

    # Material inventory update (consistent with the model's refill decisions)
    material_refill_events = {m: {} for m in M_SET}  # {m: {"A": refill_count, ...}}
    for m in M_SET:
        seq = seq_map.get(m, [])
        for job in seq:
            job_type = job.get("type", "?")
            if job.get("material_refill_required", False):
                material_refill_events[m][job_type] = material_refill_events[m].get(job_type, 0) + 1
                material_inventory[m][job_type] = int(MAT_CAPACITY)
            # Consume exactly 1 unit of this job's material type
            material_inventory[m][job_type] = int(material_inventory[m].get(job_type, 0)) - 1
            job["material_after_consume"] = int(material_inventory[m].get(job_type, 0))

    makespan = max(float(v) for v in T_end_vals.values()) if T_end_vals else 0.0
    return {
        "sequence_map": seq_map,
        "makespan": float(makespan),
        "solve_time": float(solve_time),
        "material_refill_events": material_refill_events,
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
current_solve_time = 0.0  # Store the latest Gurobi solve time (seconds)
total_solve_time = 0.0  # Cumulative Gurobi solve time across all events

# Global resource state across events (time is in the same units as MILP T_*)
agv_available_time: Dict[int, float] = {m: 0.0 for m in M_SET}
agv_current_node: Dict[int, int] = {m: int(S_m[m]) for m in M_SET}
station_available_time: Dict[int, float] = {int(n): 0.0 for n in JSON_STATION_MAPPING.values()}

# AMR material inventory: {amr_id: {"A": count, "B": count, "C": count}}
amr_material_inventory: Dict[int, Dict[str, int]] = {
    m: {"A": 0, "B": 0, "C": 0} for m in M_SET
}


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
    scale = 1.0 / 3.0

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
        f"AMR Scheduler with Gurobi | Makespan: {current_makespan:.1f} | Solve(sum): {current_solve_time:.2f}s | Gantt: Time"
    )


# ===================== INBOX INGEST + MILP + VISUAL =====================

def process_event_line_visual(line: str, ax, out_f):
    """Parse one dispatch event line, show top lane, run MILP, draw assignments, write schedule_outbox."""
    global waiting, current_makespan, current_solve_time, total_solve_time

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
        material_inventory=amr_material_inventory,
    )
    if res is None:
        return
    
    # Store makespan and solve time for display
    current_makespan = res.get("makespan", 0.0)
    total_solve_time += float(res.get("solve_time", 0.0))
    current_solve_time = total_solve_time

    # Flatten MILP result jobs by pick_time then jid
    all_jobs = []
    for m in M_SET:
        all_jobs.extend(res["sequence_map"].get(m, []))
    all_jobs.sort(key=lambda j: (j.get("pick_time", 0.0), j.get("jid", 0)))

    # Print material refill events
    material_refills = res.get("material_refill_events", {})
    for m in M_SET:
        refills = material_refills.get(m, {})
        for mat_type, count in refills.items():
            if count > 0:
                pickup_node = MAT_TYPE_TO_PICKUP.get(mat_type)
                print(
                    f"[MaterialRefill] Event {dispatch_time:.1f}: AMR {m} visits node {pickup_node} to refill {count}x Type {mat_type} material (补充10个)"
                )

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
        material_after = job.get("material_after_consume", 0)
        refill_required = job.get("material_refill_required", False)
        refill_node = job.get("refill_pickup_node", None)
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
            "material_refill_before_job": refill_required,
            "material_refill_node": refill_node,
            "material_consumed": jtype,
            "material_inventory_after": material_after,
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