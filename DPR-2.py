#!/usr/bin/env python3
"""DPR-2 (SPT)

- Select the feasible operation with the smallest processing time (ties random).
- Assign AMR by earliest completion time; tie-break by refill, pickup distance, AMR id.
- Other behavior matches DPR-1.
"""

from __future__ import annotations

import json
import os
import random
import time as pytime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
 

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

# ===================== FIELD CONFIG =====================

from config import (
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


# ===================== DPR(a1) DISPATCHING LOGIC (SPT) =====================


@dataclass
class Operation:
    jtype: str
    proc_time: float
    station: int
    delivery_node: int
    eligible_amrs: Optional[List[int]] = None  # if None, all AMRs eligible


@dataclass
class TaskState:
    jid: int
    release_time: float
    op: Operation


@dataclass
class AMRState:
    amr_id: int
    available_time: float
    current_node: int
    inv: Dict[str, int]


def _build_job_states_from_event(jobs_raw: List[dict], dispatch_time: float) -> List[TaskState]:
    # Backward-compatible name: build task list (each operation is an independent task).
    tasks: List[TaskState] = []
    for j in jobs_raw:
        jid = int(j.get("jid"))
        jtype_default = str(j.get("type", "A")).upper()
        proc_time_default = float(j.get("proc_time", 0.0))

        # NOTE: We ignore dispatch_time (all tasks are available immediately).
        release_time = 0.0

        if isinstance(j.get("operations"), list):
            for op in j["operations"]:
                st = int(op.get("station"))
                g_l = JSON_STATION_MAPPING.get(st)
                if g_l is None:
                    raise ValueError(f"Unknown station {st} for jid={jid}")
                tasks.append(
                    TaskState(
                        jid=jid,
                        release_time=float(release_time),
                        op=Operation(
                            jtype=str(op.get("type", jtype_default)).upper(),
                            proc_time=float(op.get("proc_time", proc_time_default)),
                            station=st,
                            delivery_node=int(g_l),
                            eligible_amrs=op.get("eligible_amrs"),
                        ),
                    ),
                )
        else:
            st = int(j.get("station"))
            g_l = JSON_STATION_MAPPING.get(st)
            if g_l is None:
                raise ValueError(f"Unknown station {st} for jid={jid}")
            tasks.append(
                TaskState(
                    jid=jid,
                    release_time=float(release_time),
                    op=Operation(
                        jtype=jtype_default,
                        proc_time=proc_time_default,
                        station=st,
                        delivery_node=int(g_l),
                    ),
                ),
            )
    return tasks


def _eligible_amrs_for_op(op: Operation) -> List[int]:
    if op.eligible_amrs is None:
        return [int(m) for m in M_SET]
    return [int(m) for m in op.eligible_amrs]


def _amr_is_material_compatible(amr: AMRState, op: Operation) -> bool:
    # With multi-type inventory, any AMR can serve any type; refill happens only when
    # that type's onboard qty is 0.
    return True


def _estimate_required_station_wait(
    tsk: TaskState,
    amr: AMRState,
    station_available_time: Dict[int, float],
) -> float:
    """Estimate this task's station-queue wait time if executed next on this AMR.

    Matches the wait_time definition used in _schedule_one_operation:
    wait_time = station_start - arrive_delivery (clipped at 0).
    """
    op = tsk.op
    t = str(op.jtype).upper()

    prev_end_time = float(amr.available_time)
    prev_node = int(amr.current_node)
    op_ready_time = float(tsk.release_time)

    onboard_qty = int(amr.inv.get(t, 0))
    needs_refill = onboard_qty <= 0
    if needs_refill:
        pickup_node = int(TYPE_TO_MATERIAL_NODE[t])
        to_pick_travel = float(calculate_distance(prev_node, pickup_node))
        pick_time = prev_end_time + to_pick_travel
    else:
        pickup_node = prev_node
        pick_time = prev_end_time

    delivery_node = int(op.delivery_node)
    transport_time = float(calculate_distance(int(pickup_node), int(delivery_node)))
    arrive_delivery = pick_time + transport_time

    station_prev_end = float(station_available_time.get(delivery_node, 0.0))
    station_start = max(arrive_delivery, station_prev_end, op_ready_time)
    return float(max(0.0, station_start - arrive_delivery))


def _estimate_completion_time(
    tsk: TaskState,
    amr: AMRState,
    station_available_time: Dict[int, float],
) -> Tuple[float, float, float]:
    """Return (end_time, needs_refill_flag, dist_to_pickup) for this AMR on this task."""
    op = tsk.op
    t = str(op.jtype).upper()

    prev_end_time = float(amr.available_time)
    prev_node = int(amr.current_node)
    op_ready_time = float(tsk.release_time)

    onboard_qty = int(amr.inv.get(t, 0))
    needs_refill = onboard_qty <= 0
    if needs_refill:
        pickup_node = int(TYPE_TO_MATERIAL_NODE[t])
        to_pick_travel = float(calculate_distance(prev_node, pickup_node))
        pick_time = prev_end_time + to_pick_travel
        pick_dist = to_pick_travel
    else:
        pickup_node = prev_node
        pick_time = prev_end_time
        pick_dist = 0.0

    delivery_node = int(op.delivery_node)
    transport_time = float(calculate_distance(int(pickup_node), int(delivery_node)))
    arrive_delivery = pick_time + transport_time

    station_prev_end = float(station_available_time.get(delivery_node, 0.0))
    station_start = max(arrive_delivery, station_prev_end, op_ready_time)
    end_time = station_start + float(op.proc_time)

    return (float(end_time), float(needs_refill), float(pick_dist))


def _choose_task_by_spt(
    tasks: List[TaskState],
    amrs: Dict[int, AMRState],
    station_available_time: Dict[int, float],
) -> Optional[TaskState]:
    """Pick the task with the smallest proc_time (ties random) among feasible tasks."""
    feasible: List[TaskState] = []
    for tsk in tasks:
        eligible = _eligible_amrs_for_op(tsk.op)
        if any(_amr_is_material_compatible(amrs[m], tsk.op) for m in eligible):
            feasible.append(tsk)

    if not feasible:
        return None

    min_proc = min(float(tsk.op.proc_time) for tsk in feasible)
    candidates = [tsk for tsk in feasible if abs(float(tsk.op.proc_time) - min_proc) <= 1e-12]
    return random.choice(candidates)


def _choose_amr_for_op_a1(
    js: TaskState,
    op: Operation,
    amrs: Dict[int, AMRState],
    station_available_time: Dict[int, float],
) -> int:
    eligible = _eligible_amrs_for_op(op)
    compatible = [m for m in eligible if _amr_is_material_compatible(amrs[m], op)]
    if not compatible:
        raise RuntimeError(f"No eligible AMR for jid={js.jid}, type={op.jtype}.")

    vals: List[Tuple[int, float, float, float]] = []
    t = str(op.jtype).upper()
    pickup_node = int(TYPE_TO_MATERIAL_NODE.get(t, 0))

    for m in compatible:
        end_time, needs_refill, dist_to_pickup = _estimate_completion_time(
            js, amrs[m], station_available_time
        )
        vals.append((int(m), float(end_time), needs_refill, dist_to_pickup))

    # Sort by earliest completion; then prefer no refill; then nearer pickup; then smaller id.
    vals.sort(key=lambda x: (x[1], x[2], x[3], x[0]))
    return int(vals[0][0])


def _schedule_one_operation(
    js: TaskState,
    op: Operation,
    amr: AMRState,
    station_available_time: Dict[int, float],
) -> dict:
    """Schedule js.next op on a specific AMR and update states.

    Returns a record compatible with MILP_0108 plotting + outbox.
    """

    t = str(op.jtype).upper()
    if t not in TYPE_TO_MATERIAL_NODE:
        raise ValueError(f"Unknown job type: {t}")

    prev_end_time = float(amr.available_time)
    prev_node = int(amr.current_node)

    # No precedence among operations inside a job.
    op_ready_time = float(js.release_time)

    # Inventory is tracked per type; each type max 3; refill only when that type is 0.
    onboard_qty = int(amr.inv.get(t, 0))
    needs_refill = onboard_qty <= 0
    if needs_refill:
        pickup_node = int(TYPE_TO_MATERIAL_NODE[t])
        to_pick_travel = float(calculate_distance(prev_node, pickup_node))
        arrive_pick = prev_end_time + to_pick_travel
        # Pick immediately upon arrival.
        pick_time = arrive_pick
    else:
        pickup_node = prev_node
        to_pick_travel = 0.0
        # If no refill, treat "pick" as the departure moment from the previous node.
        pick_time = prev_end_time

    idle_before_pick = max(0.0, pick_time - (prev_end_time + to_pick_travel))

    delivery_node = int(op.delivery_node)
    transport_time = float(calculate_distance(int(pickup_node), int(delivery_node)))
    arrive_delivery = pick_time + transport_time

    station_prev_end = float(station_available_time.get(delivery_node, 0.0))
    # Station processing start must respect:
    # - AMR arriving at the station
    # - station availability (no overlap)
    # - job precedence / op-ready time
    station_start = max(arrive_delivery, station_prev_end, op_ready_time)
    wait_time = max(0.0, station_start - arrive_delivery)

    end_time = station_start + float(op.proc_time)

    # Update station and AMR states
    station_available_time[delivery_node] = end_time
    amr.available_time = end_time
    amr.current_node = delivery_node

    # Update inventory
    if needs_refill:
        # Refill to the per-type cap
        amr.inv[t] = MATERIAL_PICK_QTY
    # Consume one unit of this type
    amr.inv[t] = max(0, int(amr.inv.get(t, 0)) - 1)

    return {
        "jid": int(js.jid),
        "type": t,
        "proc_time": float(op.proc_time),
        "assigned_agv": int(amr.amr_id),
        "pickup_node": int(pickup_node),
        "delivery_node": int(delivery_node),
        "pick_time": float(pick_time),
        "del_time": float(station_start),
        "end_time": float(end_time),
        "transport_time": float(transport_time),
        "wait_time": float(wait_time),
        # extra context for time-aligned rendering
        "prev_end_time": float(prev_end_time),
        "prev_node": int(prev_node),
        "to_pick_travel": float(to_pick_travel),
        "idle_before_pick": float(idle_before_pick),
        # metadata
        "policy": "spt_all_tasks",
        "needs_refill": bool(needs_refill),
    }


def dispatch_a1_event(
    jobs_raw: List[dict],
    dispatch_time: float,
    amr_available_time: Dict[int, float],
    amr_current_node: Dict[int, int],
    station_available_time: Dict[int, float],
    amr_inventory: Dict[int, Dict[str, int]],
    *,
    seed: Optional[int] = None,
) -> List[dict]:
    """Run dispatch for one batch; returns scheduled operation records.

    NOTE: AMR inventory persists across events (each task consumes 1 unit of its type;
    replenish that type to 3 only when its onboard qty reaches 0).
    """
    if seed is not None:
        random.seed(int(seed))

    tasks = _build_job_states_from_event(jobs_raw, dispatch_time)

    amrs: Dict[int, AMRState] = {}
    for m in M_SET:
        amrs[int(m)] = AMRState(
            amr_id=int(m),
            available_time=float(amr_available_time.get(int(m), 0.0)),
            current_node=int(amr_current_node.get(int(m), S_m[int(m)])),
            # Inventory persists across events; copy to avoid aliasing globals during scheduling.
            inv={str(k).upper(): int(v) for k, v in dict(amr_inventory.get(int(m), {})).items()},
        )

    records: List[dict] = []
    while tasks:
        tsk = _choose_task_by_spt(tasks, amrs, station_available_time)
        if tsk is None:
            break
        op = tsk.op
        m = _choose_amr_for_op_a1(tsk, op, amrs, station_available_time)
        rec = _schedule_one_operation(tsk, op, amrs[m], station_available_time)
        records.append(rec)
        tasks.remove(tsk)

    if tasks:
        raise RuntimeError(
            "Dispatch could not schedule all tasks under the current constraints. "
            "This usually indicates an infeasible type/AMR/inventory configuration."
        )

    # Push AMR states back to globals (across events)
    for m in M_SET:
        amr_available_time[int(m)] = float(amrs[int(m)].available_time)
        amr_current_node[int(m)] = int(amrs[int(m)].current_node)
        # Persist inventory across events
        amr_inventory[int(m)] = dict(amrs[int(m)].inv)

    return records


# ===================== GLOBAL STATE =====================

# Global resource state across events
agv_available_time: Dict[int, float] = {m: 0.0 for m in M_SET}
agv_current_node: Dict[int, int] = {m: int(S_m[m]) for m in M_SET}
station_available_time: Dict[int, float] = {int(n): 0.0 for n in JSON_STATION_MAPPING.values()}

# Material inventory across events per AMR (multi-type, capped per type)
amr_inventory: Dict[int, Dict[str, int]] = {m: {} for m in M_SET}

# File tail progress tracking
_lines_consumed: int = 0


def process_event_line_visual(line: str, ax, out_f):

    try:
        data = json.loads(line)
    except Exception:
        return

    dispatch_time = float(data.get("dispatch_time", 0.0))
    jobs_raw = data.get("jobs", [])
    if not jobs_raw:
        return

    visualization.waiting = []
    for j in jobs_raw:
        visualization.waiting.append(
            visualization.JobVisual(
                jid=int(j.get("jid")),
                jtype=str(j.get("type", "A")).upper(),
                proc_time=float(j.get("proc_time", 0.0)),
            ),
        )
    visualization.rebuild_top_lane(ax)

    # Dispatch using DRP-2 (SPT)
    _t_wall_start = pytime.perf_counter()
    records = dispatch_a1_event(
        jobs_raw,
        dispatch_time=dispatch_time,
        amr_available_time=agv_available_time,
        amr_current_node=agv_current_node,
        station_available_time=station_available_time,
        amr_inventory=amr_inventory,
    )
    _t_wall = pytime.perf_counter() - _t_wall_start
    visualization.total_solve_time += float(_t_wall)
    if not records:
        return

    # Makespan over global timeline
    visualization.current_makespan = max(
        float(visualization.current_makespan),
        max(float(r.get("end_time", 0.0)) for r in records),
    )

    # Sort by time only (do not bias by jid); randomize exact-ties for visualization/output.
    records.sort(key=lambda r: float(r.get("pick_time", 0.0)))
    # random tie-break among identical pick_time
    _eps = 1e-12
    _i = 0
    while _i < len(records):
        _j = _i + 1
        t0 = float(records[_i].get("pick_time", 0.0))
        while _j < len(records) and abs(float(records[_j].get("pick_time", 0.0)) - t0) <= _eps:
            _j += 1
        if _j - _i > 1:
            sub = records[_i:_j]
            random.shuffle(sub)
            records[_i:_j] = sub
        _i = _j

    for rec in records:
        amr = int(rec.get("assigned_agv"))
        jid = int(rec.get("jid"))
        jtype = str(rec.get("type") or "?")
        proc_time = float(rec.get("proc_time", 0.0))
        transport_time = float(rec.get("transport_time", 0.0))
        wait_time = float(rec.get("wait_time", 0.0))
        pick_time = float(rec.get("pick_time", 0.0))
        del_time = float(rec.get("del_time", 0.0))
        end_time = float(rec.get("end_time", 0.0))
        pickup_node = rec.get("pickup_node")
        delivery_node = rec.get("delivery_node")

        station = None
        for st, del_node in JSON_STATION_MAPPING.items():
            if int(del_node) == int(delivery_node):
                station = st
                break

        vjob = visualization.JobVisual(jid=jid, jtype=jtype, proc_time=proc_time)
        try:
            vjob.prev_end_time = float(rec.get("prev_end_time", 0.0))
            vjob.prev_node = rec.get("prev_node")
            vjob.to_pick_travel = float(rec.get("to_pick_travel", 0.0))
            vjob.idle_before_pick = float(rec.get("idle_before_pick", 0.0))
        except Exception:
            pass

        # optional: multi-leg route rendering (supported by updated visualization.py)
        try:
            vjob.route_nodes = rec.get("route_nodes")
            vjob.route_legs = rec.get("route_legs")
        except Exception:
            pass

        visualization.draw_on_amr(
            ax,
            amr,
            vjob,
            transport_time=transport_time,
            pick_time=pick_time,
            del_time=del_time,
            end_time=end_time,
            pickup_node=pickup_node,
            delivery_node=delivery_node,
            wait_time=wait_time,
        )

        rec_out = {
            "generated_at": dispatch_time,
            "policy": "drp-2-spt",
            "amr": amr,
            "jid": jid,
            "type": jtype,
            "proc_time": proc_time,
            "transport_time": transport_time,
            "station": str(station) if station is not None else "?",
            "pickup_node": pickup_node,
            "delivery_node": delivery_node,
        }
        out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
        out_f.flush()


def main():
    global _lines_consumed

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
            with open(INBOX, "r", encoding="utf-8") as f_in, open(
                SCHEDULE_OUTBOX, "a", encoding="utf-8"
            ) as f_out:
                lines = f_in.readlines()
                if _lines_consumed < len(lines):
                    new_lines = lines[_lines_consumed :]
                    _lines_consumed = len(lines)
                    for ln in new_lines:
                        if ln.strip():
                            process_event_line_visual(ln, ax, f_out)
                            changed = True
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
