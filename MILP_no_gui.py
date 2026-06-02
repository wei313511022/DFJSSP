#!/usr/bin/env python3
"""Headless (no-plot) runner for MILP_inhance.

- Does NOT import matplotlib / visualization.
- Reads inbox JSONL, solves each dispatch event with MILP, prints solve_time & makespan,
  and writes schedule outbox JSONL.

Usage:
  python MILP_inhance_no_gui.py
  python MILP_inhance_no_gui.py --inbox path/to/inbox.jsonl --outbox path/to/outbox.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from MILP import (
    solve_vrp_from_jobs,
    INBOX,
    SCHEDULE_OUTBOX,
    JSON_STATION_MAPPING,
    M_SET,
    agv_available_time,
    agv_current_node,
    station_available_time,
    agv_inventory,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--inbox", default=INBOX, help="Path to dispatch inbox jsonl")
    p.add_argument("--outbox", default=SCHEDULE_OUTBOX, help="Path to schedule outbox jsonl")
    return p.parse_args()


def _process_event_line_text(line: str, *, out_f, cumulative_solve_time: float) -> float:
    """Solve one dispatch line; print solve_time+makespan; write outbox; return updated cumulative time."""

    try:
        data = json.loads(line)
    except Exception:
        return cumulative_solve_time

    dispatch_time = float(data.get("dispatch_time", 0.0))
    jobs_raw = data.get("jobs", [])
    if not jobs_raw:
        return cumulative_solve_time

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
        return cumulative_solve_time

    res = solve_vrp_from_jobs(
        jobs_for_milp,
        agv_available_time=agv_available_time,
        agv_current_node=agv_current_node,
        station_available_time=station_available_time,
        agv_inventory=agv_inventory,
    )
    if res is None:
        return cumulative_solve_time

    solve_time = float(res.get("solve_time", 0.0))
    makespan = float(res.get("makespan", 0.0))
    cumulative_solve_time += solve_time

    print(
        f"[MILP] dispatch_time={dispatch_time:.3f} "
        f"solve_time={solve_time:.3f}s makespan={makespan:.3f} "
        f"cumulative_solve_time={cumulative_solve_time:.3f}s"
    )

    # Flatten MILP result jobs by pick_time then jid
    all_jobs = []
    for m in M_SET:
        all_jobs.extend(res["sequence_map"].get(m, []))
    all_jobs.sort(key=lambda j: (j.get("pick_time", 0.0), j.get("jid", 0)))

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

    # Write schedule record (MILP-based dispatching result)
    for job in all_jobs:
        amr = int(job.get("assigned_agv"))
        jid = int(job.get("jid"))
        jtype = str(job.get("type") or "?")
        proc_time = float(job.get("proc_time", 0.0))
        transport_time = float(job.get("transport_time", 0.0))
        delivery_node = job.get("delivery_node")

        # recover station from delivery node
        station: Optional[int] = None
        for st, del_node in JSON_STATION_MAPPING.items():
            if del_node == delivery_node:
                station = int(st)
                break

        rec = {
            "generated_at": dispatch_time,
            "amr": amr,
            "jid": jid,
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

    return cumulative_solve_time


def main() -> None:
    args = _parse_args()

    os.makedirs(os.path.dirname(str(args.outbox)), exist_ok=True)
    open(str(args.outbox), "w", encoding="utf-8").close()

    cumulative_solve_time = 0.0

    try:
        with open(str(args.inbox), "r", encoding="utf-8") as f_in, open(
            str(args.outbox), "a", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                if line.strip():
                    cumulative_solve_time = _process_event_line_text(
                        line, out_f=f_out, cumulative_solve_time=cumulative_solve_time
                    )
    except FileNotFoundError:
        print(f"[MILP] inbox not found: {args.inbox}")


if __name__ == "__main__":
    main()
