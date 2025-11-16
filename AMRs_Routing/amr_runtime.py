#!/usr/bin/env python3
# Simple AMR Runtime (move-only version)
# - Reads assignments from schedule_outbox.jsonl
# - Each line: {"generated_at", "amr", "jid", "type", "proc_time", "station"}
# - Each AMR keeps a queue of jobs.
# - For now, AMRs ONLY move to the station cell for each job (no work, no inventory).
# - SPACE to pause/resume.

import os
import json
import math
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# ---------- Files ----------
SCHEDULE_INBOX = "../Random_Job_Arrivals/schedule_outbox.jsonl"   # this is your file

# ---------- Grid / Layout ----------
GRID_W, GRID_H = 10, 10

# Production stations (right side). You can adjust coordinates as needed.
STATION_POS: Dict[int, Tuple[int, int]] = {
    1: (9, 1),
    2: (9, 4),
    3: (9, 7),
}

# Optional: obstacles, empty for now
OBSTACLES: Set[Tuple[int, int]] = set()
AMR_LOCATIONS: Set[Tuple[int, int]] = set()

# ---------- Simulation timing / motion ----------
UPDATE_INTERVAL_MS = 100         # timer tick in ms
SIM_SPEED_MULT     = 5.0          # speed-up factor
CELLS_PER_SEC      = 1.0          # grid cells per simulated second

AMR_COUNT = 3                     # number of AMRs

Coord = Tuple[int, int]

# Global axis for drawing dashed paths
_AX = None

# Track how many lines we already consumed from SCHEDULE_INBOX
_lines_consumed = 0


# ---------- Data models ----------

@dataclass
class Job:
    jid: int
    station: int      # station index (1, 2, 3)
    jtype: str        # just for label, not used in logic
    proc_time: float  # not used yet, just kept for future


@dataclass
class AMRState:
    amr_id: int
    posx: float
    posy: float
    state: str = "idle"           # "idle" | "move"
    queue: List[Job] = field(default_factory=list)

    # path following
    path: List[Coord] = field(default_factory=list)
    waypoint_idx: int = 0
    move_budget: float = 0.0

    # visuals
    marker: Optional[Circle] = None
    label: Optional[plt.Text] = None
    route_artist: Optional[plt.Line2D] = None  # dashed path line


# ---------- Helpers ----------

def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < GRID_W and 0 <= y < GRID_H


def cur_cell(a: AMRState) -> Coord:
    return (round(a.posx), round(a.posy))


def octile(a: Coord, b: Coord) -> float:
    """Heuristic distance for 8-direction movement."""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    D, D2 = 1.0, math.sqrt(2.0)
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


def reconstruct_path(came_from: Dict[Coord, Coord],
                     start: Coord,
                     goal: Coord) -> List[Coord]:
    if goal not in came_from and goal != start:
        return []
    path = [goal]
    cur = goal
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


# ---------- A* (8-direction, static obstacles) ----------

def astar_8dir(start: Coord,
               goal: Coord,
               blocked: Set[Coord]) -> List[Coord]:
    if start == goal:
        return [start]
    if start in blocked or goal in blocked:
        return []

    STEPS = [
        (1, 0, 1.0), (-1, 0, 1.0),
        (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)),  (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]

    def can_move(x: int, y: int, dx: int, dy: int) -> bool:
        nx, ny = x + dx, y + dy
        if not in_bounds(nx, ny) or (nx, ny) in blocked:
            return False
        # prevent diagonal corner-cutting
        if dx != 0 and dy != 0 and ((x + dx, y) in blocked or (x, y + dy) in blocked):
            return False
        return True

    openh = []
    heapq.heappush(openh, (octile(start, goal), 0.0, start))
    came: Dict[Coord, Coord] = {}
    gscore = {start: 0.0}
    closed: Set[Coord] = set()

    while openh:
        _, gs, cur = heapq.heappop(openh)
        if cur in closed:
            continue
        if cur == goal:
            return reconstruct_path(came, start, goal)
        closed.add(cur)

        x, y = cur
        for dx, dy, base in STEPS:
            if not can_move(x, y, dx, dy):
                continue
            nxt = (x + dx, y + dy)
            ns = gs + base
            if ns < gscore.get(nxt, math.inf):
                came[nxt] = cur
                gscore[nxt] = ns
                heapq.heappush(openh, (ns + octile(nxt, goal), ns, nxt))
    return []


# ---------- Route drawing ----------

def _update_route_artist(amr: AMRState):
    """Draw/refresh dashed route line for the AMR's current path."""
    global _AX
    if _AX is None:
        return
    # Remove previous route
    if amr.route_artist is not None:
        try:
            amr.route_artist.remove()
        except Exception:
            pass
        amr.route_artist = None

    if not amr.path or len(amr.path) <= 1:
        return

    xs = [p[0] for p in amr.path]
    ys = [p[1] for p in amr.path]
    (line,) = _AX.plot(
        xs, ys,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        zorder=4,
    )
    amr.route_artist = line


def plan_path_to_station(amr: AMRState,
                         station_id: int,
                         amrs: Dict[int, AMRState]):
    """Plan a path from current AMR position to the given station."""
    start = cur_cell(amr)
    if station_id not in STATION_POS:
        print(f"[warn] unknown station {station_id} for AMR{amr.amr_id}")
        amr.path = []
        amr.waypoint_idx = 0
        _update_route_artist(amr)
        return
    goal = STATION_POS[station_id]

    blocked = set(OBSTACLES)
    # (Optional) you could add other AMRs as soft obstacles here if you want later.

    amr.path = astar_8dir(start, goal, blocked)
    amr.waypoint_idx = 1  # 0 is current cell
    _update_route_artist(amr)


# ---------- Assignment ingestion ----------

def ingest_new_assignments(amrs: Dict[int, AMRState]) -> bool:
    """Read any new JSON lines from SCHEDULE_INBOX and append jobs to AMR queues."""
    global _lines_consumed
    if not os.path.exists(SCHEDULE_INBOX):
        return False

    with open(SCHEDULE_INBOX, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if _lines_consumed >= len(lines):
        return False

    new_lines = lines[_lines_consumed:]
    _lines_consumed = len(lines)

    added = 0
    for raw in new_lines:
        ln = raw.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
            amr_id = int(rec["amr"])
            jid    = int(rec["jid"])
            jtype  = str(rec.get("type", "A"))
            proc_time = float(rec.get("proc_time", 0.0))
            station = int(rec["station"])

            if amr_id not in amrs:
                print(f"[warn] assignment for unknown AMR {amr_id}: {ln}")
                continue

            job = Job(jid=jid, station=station, jtype=jtype, proc_time=proc_time)
            amrs[amr_id].queue.append(job)
            added += 1

        except Exception as e:
            print(f"[warn] bad assignment JSON: {e} | line={ln!r}")

    # If any new jobs arrived, try to start movement for idle AMRs
    if added > 0:
        for amr in amrs.values():
            if amr.state == "idle":
                try_start_next_job(amr, amrs)

    return added > 0


def try_start_next_job(amr: AMRState, amrs: Dict[int, AMRState]) -> bool:
    """If AMR is idle and has a job queue, plan path to the next station."""
    if amr.state != "idle" or not amr.queue:
        return False

    job = amr.queue[0]  # do NOT pop yet; pop when arrived
    plan_path_to_station(amr, job.station, amrs)
    if len(amr.path) > 1:
        amr.state = "move"
    else:
        # Already on station; just mark job done and move to next
        amr.queue.pop(0)
        amr.state = "idle"
        # Immediately try next job
        try_start_next_job(amr, amrs)
    return True


def on_arrival(amr: AMRState, amrs: Dict[int, AMRState]):
    """Called when an AMR has finished its path to a station."""
    # Current job is the first in queue
    if amr.queue:
        job = amr.queue.pop(0)
        print(f"[info] AMR{amr.amr_id} arrived at station {job.station} for job {job.jid}")
    amr.state = "idle"
    amr.path = []
    amr.waypoint_idx = 0
    _update_route_artist(amr)
    # Immediately start next job if any
    try_start_next_job(amr, amrs)


# ---------- Motion ----------

def simple_step(amrs: Dict[int, AMRState], dt: float):
    budget = CELLS_PER_SEC * dt
    # accumulate budget
    for st in amrs.values():
        if st.state == "move" and st.path:
            st.move_budget += budget

    # move along path
    for st in amrs.values():
        while st.state == "move" and st.move_budget >= 1.0 and st.waypoint_idx < len(st.path):
            nxt = st.path[st.waypoint_idx]
            st.posx, st.posy = float(nxt[0]), float(nxt[1])
            st.waypoint_idx += 1
            st.move_budget -= 1.0
            if st.waypoint_idx >= len(st.path):
                # reached destination
                on_arrival(st, amrs)
                break


# ---------- Drawing ----------

def draw_static(ax):
    ax.set_xlim(-0.5, GRID_W - 0.5)
    ax.set_ylim(-0.5, GRID_H - 0.5)
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.grid(True, which="both", linewidth=0.4, color="black", alpha=0.4)

    # obstacles
    for (x, y) in OBSTACLES:
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor=(0.8, 0.8, 0.8),
                edgecolor="none",
                zorder=1,
            )
        )

    # stations
    for sid, (sx, sy) in STATION_POS.items():
        ax.add_patch(
            Rectangle(
                (sx - 0.5, sy - 0.5),
                1,
                1,
                facecolor="none",
                edgecolor="tab:red",
                linewidth=2.0,
                zorder=2,
            )
        )
        ax.text(
            sx + 0.6,
            sy + 0.2,
            f"S{sid}",
            fontsize=11,
            color="tab:red",
            weight="bold",
            zorder=3,
        )


def create_amrs(ax) -> Dict[int, AMRState]:
    """Create AMRs at some initial positions."""
    amrs: Dict[int, AMRState] = {}
    # simple starting positions; adjust as you like
    starts = {
        1: (2.0, 1.0),
        2: (2.0, 4.0),
        3: (2.0, 7.0),
    }
    for i in range(1, AMR_COUNT + 1):
        x, y = starts.get(i, (1.0, 1.0))
        st = AMRState(amr_id=i, posx=x, posy=y)
        mk = Circle((x, y), radius=0.35,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.8,
                    zorder=5)
        ax.add_patch(mk)
        st.marker = mk
        st.label = ax.text(
            x,
            y + 0.55,
            f"AMR{i} (idle)",
            fontsize=9,
            ha="center",
            va="bottom",
            zorder=6,
        )
        amrs[i] = st
        # AMR_LOCATIONS.add((amrs[i].posx, amrs[i].posy))
    return amrs


# ---------- Main ----------

def main():
    global _AX
    fig, ax = plt.subplots(figsize=(10, 6))
    _AX = ax

    draw_static(ax)
    amrs = create_amrs(ax)
    amrs[1].queue.append(Job(jid=0, station=1, jtype="A", proc_time=10.0))

    is_running = True
    sim_t = 0.0

    timer = fig.canvas.new_timer(interval=UPDATE_INTERVAL_MS)

    def tick():
        # print(AMR_LOCATIONS)
        nonlocal is_running, sim_t
        changed = False

        # 1) ingest new assignments
        if ingest_new_assignments(amrs):
            changed = True

        # 2) simulate movement if running
        if is_running:
            dt = (UPDATE_INTERVAL_MS / 1000.0) * SIM_SPEED_MULT
            sim_t += dt
            simple_step(amrs, dt)

            # update visuals
            for k, st in amrs.items():
                if st.marker:
                    st.marker.center = (st.posx, st.posy)
                if st.label:
                    status = "idle" if st.state == "idle" else "move"
                    st.label.set_text(f"AMR{k} ({status})")
                    st.label.set_position((st.posx, st.posy + 0.55))

            changed = True

        if changed:
            fig.canvas.draw_idle()
        timer.start()

    def on_key(e):
        nonlocal is_running
        if e.key == " ":
            is_running = not is_running

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer.add_callback(tick)
    timer.start()
    plt.show()


if __name__ == "__main__":
    main()
