#!/usr/bin/env python3
# Simple AMR Runtime (move-only version, priority-aware, with material stations)
# - Reads assignments from schedule_outbox.jsonl
# - Each line: {"generated_at", "amr", "jid", "type", "proc_time", "station"}
# - Each AMR keeps a queue of jobs.
# - For each job:
#   1) Go to material supply station on the left (based on job type A/B/C)
#   2) Then go to the production station on the right for delivery.
# - Priority:
#   AMR1: highest, ignores other AMRs.
#   AMR2: sees AMR1 as obstacle, ignores AMR3.
#   AMR3: sees AMR1 and AMR2 as obstacles.
# - Replans routes every tick.
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
SCHEDULE_INBOX = "../Random_Job_Arrivals/schedule_outbox.jsonl"

# ---------- Grid / Layout ----------
GRID_W, GRID_H = 10, 10

# Production stations (right side).
STATION_POS: Dict[int, Tuple[int, int]] = {
    1: (9, 1),
    2: (9, 4),
    3: (9, 7),
}

# Material stations (left side) by job type
MAT_POS: Dict[str, Tuple[int, int]] = {
    "A": (0, 1),
    "B": (0, 4),
    "C": (0, 7),
}

ALL_STATIONS: Set[Tuple[int, int]] = {
    (9, 1),
    (9, 4),
    (9, 7),
    (0, 1),
    (0, 4),
    (0, 7),
}
# Optional: static obstacles
OBSTACLES: Set[Tuple[int, int]] = set()

# Dynamic AMR occupancy: holds (x, y) int grid cells where AMRs currently are
AMR_LOCATIONS: Set[Tuple[int, int]] = set()

# ---------- Simulation timing / motion ----------
UPDATE_INTERVAL_MS = 100         # timer tick in ms
SIM_SPEED_MULT     = 1.0         # speed-up factor
CELLS_PER_SEC      = 1.0         # grid cells per simulated second

AMR_COUNT = 3                    # number of AMRs

Coord = Tuple[int, int]

# Global axis for drawing dashed paths
_AX = None

# Track how many lines we already consumed from SCHEDULE_INBOX
_lines_consumed = 0


# ---------- Data models ----------

@dataclass
class Job:
    jid: int
    station: int      # production station index (1, 2, 3)
    jtype: str        # A/B/C, used to pick material station
    proc_time: float  # not used yet, just kept for future


@dataclass
class AMRState:
    amr_id: int
    posx: float
    posy: float
    nxt_posx: float
    nxt_posy: float
    state: str = "idle"           # "idle" | "move"
    blocked: bool = False          # true if cannot move due to obstacles
    queue: List[Job] = field(default_factory=list)

    # current job + phase
    job: Optional[Job] = None     # job currently being served
    phase: Optional[str] = None   # None | "supply" | "deliver"

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


# ---------- Priority helper ----------

def amr_priority(amr_id: int) -> int:
    """
    Larger value = higher priority.
    With AMR IDs 1,2,3:
      AMR1 -> priority -1  (highest)
      AMR2 -> -2
      AMR3 -> -3  (lowest)
    We will treat other AMRs as obstacles IFF their priority is higher.
    """
    return -amr_id


# ---------- A* (8-direction, static + dynamic obstacles) ----------

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
        if dx != 0 and dy != 0 and (
            (x + dx, y) in blocked or (x, y + dy) in blocked
        ):
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


# ---------- Path planning (material + station, priority-aware) ----------

def _build_blocked_for(amr: AMRState, amrs: Dict[int, AMRState]) -> Set[Coord]:
    """Build blocked set for this AMR, considering static obstacles + higher-priority AMRs."""
    blocked = set(OBSTACLES)
    my_pri = amr_priority(amr.amr_id)

    for other_id, other in amrs.items():
        if other_id == amr.amr_id:
            continue
        ocell = cur_cell(other)
        if ocell in ALL_STATIONS or other.blocked:
            blocked.add(ocell)
        
        if amr_priority(other_id) > my_pri:
            # Block their current cell (idle or moving).
            # ocell = cur_cell(other)
            if in_bounds(*ocell):
                blocked.add(ocell)

            # Optionally block their next cell if they are moving and path valid.
            if (
                other.state == "move"
                and other.path
                and 0 <= other.waypoint_idx < len(other.path)
            ):
                nxt = other.path[other.waypoint_idx]
                if in_bounds(*nxt):
                    blocked.add(nxt)
    print(f"[debug] AMR{amr.amr_id} blocked cells: {blocked}")
    return blocked


def plan_path_to_material(amr: AMRState, amrs: Dict[int, AMRState]):
    """Plan a path from current AMR position to its material station (based on job type)."""
    if amr.job is None:
        amr.path = []
        amr.waypoint_idx = 0
        _update_route_artist(amr)
        return

    jtype = amr.job.jtype.upper()
    if jtype not in MAT_POS:
        print(f"[warn] no material station for job type {jtype} (AMR{amr.amr_id})")
        amr.path = []
        amr.waypoint_idx = 0
        _update_route_artist(amr)
        return

    start = cur_cell(amr)
    goal = MAT_POS[jtype]
    blocked = _build_blocked_for(amr, amrs)

    amr.path = astar_8dir(start, goal, blocked)
    if len(amr.path) < 1:
        amr.blocked = True
    else:
        amr.blocked = False 
    amr.waypoint_idx = 1  # index 0 is current cell
    _update_route_artist(amr)


def plan_path_to_station(amr: AMRState,
                         station_id: int,
                         amrs: Dict[int, AMRState]):
    """Plan a path from current AMR position to the given production station, with priority-aware obstacles."""
    start = cur_cell(amr)
    if station_id not in STATION_POS:
        print(f"[warn] unknown station {station_id} for AMR{amr.amr_id}")
        amr.path = []
        amr.waypoint_idx = 0
        _update_route_artist(amr)
        return
    goal = STATION_POS[station_id]

    blocked = _build_blocked_for(amr, amrs)

    amr.path = astar_8dir(start, goal, blocked)
    if len(amr.path) < 1:
        amr.blocked = True
    else:
        amr.blocked = False     
    amr.waypoint_idx = 1
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
            amr_id    = int(rec["amr"])
            jid       = int(rec["jid"])
            jtype     = str(rec.get("type", "A")).upper()
            proc_time = float(rec.get("proc_time", 0.0))
            station   = int(rec["station"])

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
            if amr.state == "idle" and amr.job is None:
                try_start_next_job(amr, amrs)

    return added > 0


def try_start_next_job(amr: AMRState, amrs: Dict[int, AMRState]) -> bool:
    """
    If AMR is idle and has a job queue:
      - Pop next job into amr.job
      - First leg: go to material station ("supply" phase)
    """
    if amr.state != "idle" or amr.job is not None or not amr.queue:
        return False

    amr.job = amr.queue.pop(0)
    amr.phase = "supply"
    plan_path_to_material(amr, amrs)

    if len(amr.path) > 1:
        amr.state = "move"
    else:
        # Already at material station -> treat as immediate arrival
        on_arrival(amr, amrs)
    return True


def on_arrival(amr: AMRState, amrs: Dict[int, AMRState]):
    """
    Called when an AMR has finished its path.
    Two-leg logic:
      - phase == "supply": we just arrived at material station -> plan path to production station.
      - phase == "deliver": we just arrived at production station -> job finished.
    """
    if amr.job is None:
        amr.state = "idle"
        amr.phase = None
        amr.path = []
        amr.waypoint_idx = 0
        _update_route_artist(amr)
        return

    if amr.phase == "supply":
        # arrived at material station, now go deliver job
        amr.phase = "deliver"
        plan_path_to_station(amr, amr.job.station, amrs)
        print(f"[info] AMR{amr.amr_id} picked up job {amr.job.jid} of type {amr.job.jtype} for station {amr.job.station} and path length {len(amr.path)}")
        if cur_cell(amr) != STATION_POS.get(amr.job.station, (-1, -1)):
            amr.state = "move"
        else:
            # already at station?? then job is effectively done
            print(f"[info] AMR{amr.amr_id} instantly finished job {amr.job.jid} at station {amr.job.station}")
            amr.job = None
            amr.phase = None
            amr.state = "idle"
            amr.path = []
            amr.waypoint_idx = 0
            _update_route_artist(amr)
            try_start_next_job(amr, amrs)
        return

    if amr.phase == "deliver":
        # Arrived at production station -> job finished
        print(f"[info] AMR{amr.amr_id} delivered job {amr.job.jid} to station {amr.job.station}")
        amr.job = None
        amr.phase = None
        amr.state = "idle"
        amr.path = []
        amr.waypoint_idx = 0
        _update_route_artist(amr)
        # Start next job if any
        try_start_next_job(amr, amrs)
        return

    # Fallback: unknown phase
    amr.state = "idle"
    amr.phase = None
    amr.path = []
    amr.waypoint_idx = 0
    _update_route_artist(amr)


# ---------- Motion ----------

def simple_step(amrs: Dict[int, AMRState], dt: float):
    # Replan routes for moving AMRs each tick (priority-aware obstacles)
    for st in amrs.values():
        if st.state == "move" and st.job is not None:
            if st.phase == "supply":
                plan_path_to_material(st, amrs)
            elif st.phase == "deliver":
                plan_path_to_station(st, st.job.station, amrs)

    budget = CELLS_PER_SEC * dt
    # accumulate budget
    for st in amrs.values():
        if st.state == "move" and st.path:
            st.move_budget += budget

    # move along path
    for st in amrs.values():
        while (
            st.state == "move"
            and st.move_budget >= 1.0
            and st.waypoint_idx < len(st.path)
        ):
            nxt = st.path[st.waypoint_idx]

            # update AMR_LOCATIONS: remove old cell, add new cell
            AMR_LOCATIONS.discard((round(st.posx), round(st.posy)))
            st.posx, st.posy = float(nxt[0]), float(nxt[1])
            AMR_LOCATIONS.add((round(st.posx), round(st.posy)))

            st.waypoint_idx += 1
            st.move_budget -= 1.0
            if st.waypoint_idx >= len(st.path):
                # reached target of this leg
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

    # production stations (red)
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

    # material stations (blue)
    for jt, (mx, my) in MAT_POS.items():
        ax.add_patch(
            Rectangle(
                (mx - 0.5, my - 0.5),
                1,
                1,
                facecolor="none",
                edgecolor="tab:blue",
                linewidth=2.0,
                zorder=2,
            )
        )
        ax.text(
            mx - 0.4,
            my + 0.2,
            f"M{jt}",
            fontsize=11,
            color="tab:blue",
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
        st = AMRState(amr_id=i, posx=x, posy=y, nxt_posx=x + 1, nxt_posy=y)
        mk = Circle(
            (x, y),
            radius=0.35,
            facecolor="white",
            edgecolor="black",
            linewidth=1.8,
            zorder=5,
        )
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

        # record initial location as occupied
        AMR_LOCATIONS.add((round(x), round(y)))

    return amrs


# ---------- Main ----------

def main():
    global _AX
    fig, ax = plt.subplots(figsize=(10, 6))
    _AX = ax

    draw_static(ax)
    amrs = create_amrs(ax)

    # Optional: inject a debug job so you see movement even without the file
    # amrs[1].queue.append(Job(jid=0, station=1, jtype="A", proc_time=10.0))
    # try_start_next_job(amrs[1], amrs)

    is_running = True
    sim_t = 0.0

    timer = fig.canvas.new_timer(interval=UPDATE_INTERVAL_MS)

    def tick():
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
                    if st.state == "idle":
                        phase_str = "" if st.job is None else f", {st.phase}"
                        status = f"idle{phase_str}"
                    else:
                        phase_str = "" if st.phase is None else f", {st.phase}"
                        status = f"move{phase_str}"
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
