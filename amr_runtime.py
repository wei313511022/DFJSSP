#!/usr/bin/env python3
# Program 3 — AMR Runtime (Shortest-Path, Station-Exclusive, NO 'hold')
# - Reads per-job assignments from schedule_outbox.jsonl (Program 2 output)
# - A* (8-dir) over static obstacles only (no AMR-AMR collision modeling)
# - Station cell is exclusive: only one AMR can be on a station tile
# - After finishing, AMR moves to the station's right-side slot (2 cells)
# - If a station is busy, AMRs wait at that right-side slot and auto-enter when free
# - States: 'idle' | 'move' | 'work'  (no 'hold')
# - Draw dashed routing lines for current planned paths
# - SPACE to pause/resume

import os, json, math, heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# ---------- Files ----------
SCHEDULE_INBOX = "schedule_outbox.jsonl"   # produced by Program 2

# ---------- Flags ----------
ENFORCE_STATION_EXCLUSIVE = True   # station cell cannot be shared

# ---------- Grid / Layout ----------
GRID_W, GRID_H = 20, 12

# Production stations (right)
STATION_POS = {1: (14, 1), 2: (14, 5), 3: (14, 9)}

# Material stations (left)
MAT_POS = {'A': (1, 3), 'B': (1, 6), 'C': (1, 9)}

# Demo obstacles: a vertical wall with two openings
OBSTACLES: Set[Tuple[int, int]] = set()
for y in range(1, GRID_H - 1):
    if y not in (4, 8):
        OBSTACLES.add((10, y))

# ---------- Sim timing / motion ----------
UPDATE_INTERVAL_MS = 120
SIM_SPEED_MULT = 2.0
CELLS_PER_SEC = 6.0

AMR_COUNT = 3
TYPE_TO_STATION = {"A": 1, "B": 2, "C": 3}

# ---------- Colors ----------
_cycle = plt.rcParams.get("axes.prop_cycle", None)
_cycle_list = _cycle.by_key()["color"] if _cycle else ["C0", "C1", "C2", "C3"]
TYPE_COLORS = {"A": _cycle_list[0], "B": _cycle_list[1], "C": _cycle_list[2]}

Coord = Tuple[int, int]

# ---------- (No crowd cost) constants ----------
STATION_PENALTY = 10.0  # used only to avoid resting on station tiles (not for routing)

# (module-level axis holder so we can draw path lines from helpers)
_AX = None

# ---------- Data models ----------
@dataclass
class Job:
    jid: int
    jtype: str
    proc_time: float
    station: int

@dataclass
class AMRState:
    amr_id: int
    posx: float
    posy: float
    state: str = "idle"        # "idle"|"move"|"work"  (no 'hold')
    job: Optional[Job] = None
    queue: List[Job] = field(default_factory=list)
    # path following
    path: List[Coord] = field(default_factory=list)
    waypoint_idx: int = 0
    move_budget: float = 0.0
    # work timing
    work_left: float = 0.0
    # inventory
    inv: Dict[str, int] = field(default_factory=lambda: {"A": 10, "B": 10, "C": 10})
    # "supply" -> refill; "prod" -> produce; "egress" -> leave to right-side slot
    phase: Optional[str] = None
    # visuals
    marker: Optional[Circle] = None
    label: Optional[plt.Text] = None
    hud: Optional[plt.Text] = None
    route_artist: Optional[plt.Line2D] = None  # <--- NEW: dashed route line

# ---------- helpers ----------
def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < GRID_W and 0 <= y < GRID_H

def cur_cell(a: AMRState) -> Coord:
    return (round(a.posx), round(a.posy))

def octile(a: Coord, b: Coord) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    D, D2 = 1.0, math.sqrt(2.0)
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def reconstruct_path(came_from: Dict[Coord, Coord], start: Coord, goal: Coord) -> List[Coord]:
    if goal not in came_from and goal != start:
        return []
    path = [goal]
    cur = goal
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path

# ---------- shortest-path A* (8-dir, static obstacles only) ----------
def terrain_cost(cell: Coord, amrs: Dict[int, AMRState], ignore_id: int = -1) -> float:
    # No AMR-AMR crowd cost; keep signature to match callers
    return 0.0

def astar_8dir(start: Coord, goal: Coord, blocked: Set[Coord],
               amrs: Dict[int, AMRState], ignore_id: int) -> List[Coord]:
    if start == goal:
        return [start]
    if start in blocked or goal in blocked:
        return []

    STEPS = [
        (1,0,1.0), (-1,0,1.0), (0,1,1.0), (0,-1,1.0),
        (1,1,math.sqrt(2)), (-1,1,math.sqrt(2)),
        (1,-1,math.sqrt(2)), (-1,-1,math.sqrt(2))
    ]
    def can_move(x:int,y:int,dx:int,dy:int)->bool:
        nx, ny = x+dx, y+dy
        if not in_bounds(nx,ny) or (nx,ny) in blocked:
            return False
        if dx!=0 and dy!=0 and ((x+dx,y) in blocked or (x,y+dy) in blocked):
            return False
        return True

    openh=[]; heapq.heappush(openh,(octile(start,goal), 0.0, start))
    came:Dict[Coord,Coord]={}; gscore={start:0.0}; closed=set()

    while openh:
        _, gs, cur = heapq.heappop(openh)
        if cur in closed:
            continue
        if cur == goal:
            return reconstruct_path(came, start, goal)
        closed.add(cur)
        x,y = cur
        for dx,dy,base in STEPS:
            if not can_move(x,y,dx,dy):
                continue
            nxt = (x+dx, y+dy)
            step = base + terrain_cost(nxt, amrs, ignore_id=ignore_id)  # = base
            ns   = gs + step
            if ns < gscore.get(nxt, math.inf):
                came[nxt] = cur
                gscore[nxt] = ns
                heapq.heappush(openh, (ns + octile(nxt,goal), ns, nxt))
    return []

def is_station_occupied(stid: int, amrs: Dict[int, AMRState], ignore_id: Optional[int] = None) -> bool:
    cell = STATION_POS[stid]
    for k, a in amrs.items():
        if ignore_id is not None and k == ignore_id:
            continue
        if cur_cell(a) == cell:
            return True
    return False

def right_side_slot_for_station(stid: int, amrs: Dict[int, AMRState]) -> Coord:
    """Prefer station's right side (2 cells). If blocked, pick a nearby good slot."""
    sx, sy = STATION_POS[stid]
    candidates: List[Coord] = [
        (sx + 2, sy), (sx + 1, sy), (sx + 3, sy),
        (sx + 2, sy + 1), (sx + 2, sy - 1),
        (sx + 1, sy + 1), (sx + 1, sy - 1),
        (sx + 3, sy + 1), (sx + 3, sy - 1),
    ]
    cands = [(x, y) for (x, y) in candidates if in_bounds(x, y) and (x, y) not in OBSTACLES]
    if not cands:
        return (min(GRID_W - 1, sx + 2), sy)
    for c in cands:
        if all(cur_cell(a) != c for a in amrs.values()):
            return c
    return cands[0]

# ---------- route line helper ----------
def _update_route_artist(amr: AMRState):
    """Create/update dashed route polyline for AMR's current path."""
    global _AX
    if _AX is None:
        return
    # Remove old route if exists
    if amr.route_artist is not None:
        try:
            amr.route_artist.remove()
        except Exception:
            pass
        amr.route_artist = None
    # Nothing to draw if no path or single node
    if not amr.path or len(amr.path) <= 1:
        return
    xs = [p[0] for p in amr.path]
    ys = [p[1] for p in amr.path]
    # draw dashed line, slightly under markers
    (line,) = _AX.plot(xs, ys, linestyle="--", linewidth=1.6, alpha=0.7, zorder=4)
    amr.route_artist = line

def plan_path(amr: AMRState, target: Coord, amrs: Dict[int, AMRState]):
    start = cur_cell(amr)
    blocked = set(OBSTACLES)
    if ENFORCE_STATION_EXCLUSIVE:
        for sid, sc in STATION_POS.items():
            if is_station_occupied(sid, amrs, ignore_id=amr.amr_id):
                blocked.add(sc)
    amr.path = astar_8dir(start, target, blocked, amrs, ignore_id=amr.amr_id)
    amr.waypoint_idx = 1
    _update_route_artist(amr)  # <-- draw/update path line

# ---------- ingest assignments ----------
_lines_consumed = 0

def _coerce_int(x, name):
    if x is None: raise ValueError(f"{name} is None")
    if isinstance(x, bool): raise ValueError(f"{name} is bool")
    if isinstance(x, int): return int(x)
    if isinstance(x, float): return int(x)
    s = str(x).strip()
    if s == "": raise ValueError(f"{name} empty")
    try: return int(s)
    except ValueError: return int(float(s))

def _coerce_float(x, name):
    if x is None: raise ValueError(f"{name} is None")
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    if s == "": raise ValueError(f"{name} empty")
    return float(s)

def ingest_new_assignments(ax, amrs: Dict[int, AMRState], assigned_ui: List[Job]) -> bool:
    global _lines_consumed
    if not os.path.exists(SCHEDULE_INBOX):
        return False
    with open(SCHEDULE_INBOX, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    if _lines_consumed >= len(lines):
        return False

    new_lines = lines[_lines_consumed:]
    _lines_consumed = len(lines)
    added = 0

    for raw in new_lines:
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        try:
            rec = json.loads(ln)
            amr_id = _coerce_int(rec.get("amr"), "amr")
            jid    = _coerce_int(rec.get("jid"), "jid")
            jtype  = str(rec.get("type", "A")).strip().upper() or "A"
            dur    = _coerce_float(rec.get("proc_time"), "proc_time")
            st_raw = rec.get("station", None)
            station = TYPE_TO_STATION.get(jtype, 1) if st_raw is None else _coerce_int(st_raw, "station")

            if amr_id not in amrs:
                raise ValueError(f"unknown amr {amr_id}")
            if dur <= 0:
                raise ValueError("proc_time must be > 0")

            j = Job(jid=jid, jtype=jtype, proc_time=float(dur), station=int(station))
            amrs[amr_id].queue.append(j)
            assigned_ui.append(j)

            if amrs[amr_id].state == "idle":
                amr_try_start_next(ax, amrs[amr_id], amrs)

            added += 1
        except Exception as e:
            print(f"[warn] bad assignment JSON: {e} | line={ln!r}")
    return added > 0

# ---------- job start / arrival / finish ----------
def amr_try_start_next(ax, amr: AMRState, amrs: Dict[int, AMRState]):
    if amr.state != "idle" or not amr.queue:
        return False
    amr.job = amr.queue.pop(0)
    jt = amr.job.jtype

    # Need material first?
    if amr.inv.get(jt, 0) <= 0:
        amr.phase = "supply"
        plan_path(amr, MAT_POS[jt], amrs)
        amr.state = "move" if len(amr.path) > 1 else "work"
        return True

    # Production phase
    amr.phase = "prod"
    stid = amr.job.station
    st_cell = STATION_POS[stid]

    if ENFORCE_STATION_EXCLUSIVE and is_station_occupied(stid, amrs, ignore_id=amr.amr_id):
        # Station busy -> go wait at right-side slot, we'll be 'idle' on arrival
        wait_pos = right_side_slot_for_station(stid, amrs)
        plan_path(amr, wait_pos, amrs)
        amr.state = "move" if len(amr.path) > 1 else "idle"
    else:
        # Station free -> go straight in
        plan_path(amr, st_cell, amrs)
        amr.state = "move" if len(amr.path) > 1 else "work"
    return True

def grant_arrival_action(amr: AMRState, amrs: Dict[int, AMRState]):
    """Called when an AMR finishes a path. Decide the next action immediately (no 'hold')."""
    if amr.phase == "egress":
        amr.phase = None
        amr.state = "idle"
        amr_try_start_next(None, amr, amrs)
        return

    if amr.phase == "supply":
        # Refilled on arrival
        if amr.job:
            amr.inv[amr.job.jtype] = 10
        amr.phase = "prod"
        amr.state = "idle"
        amr_try_start_next(None, amr, amrs)
        return

    if amr.phase == "prod" and amr.job:
        stid = amr.job.station
        st_cell = STATION_POS[stid]
        if cur_cell(amr) == st_cell:
            # On station tile -> start work immediately
            amr.state = "work"
            amr.work_left = amr.job.proc_time
            if amr.inv.get(amr.job.jtype, 0) > 0:
                amr.inv[amr.job.jtype] -= 1
            # path consumed -> clear route line
            amr.path = []
            amr.waypoint_idx = 0
            _update_route_artist(amr)
        else:
            # Arrived at waiting slot -> stay idle; promotion loop will send us in when free
            amr.state = "idle"

def do_work_and_maybe_finish(amr: AMRState, dt: float, ui_panel: List[Job], amrs: Dict[int, AMRState]):
    if amr.state != "work":
        return
    if amr.work_left > 0:
        amr.work_left -= dt
    if amr.work_left > 1e-6:
        return

    # Finished work now
    if amr.job:
        for i, w in enumerate(ui_panel):
            if w.jid == amr.job.jid:
                del ui_panel[i]; break

    stid = amr.job.station if amr.job else None
    amr.phase = "egress"
    if stid is not None:
        target = right_side_slot_for_station(stid, amrs)
    else:
        c = cur_cell(amr); target = (min(GRID_W-1, c[0]+2), c[1])

    plan_path(amr, target, amrs)
    amr.state = "move" if len(amr.path) > 1 else "idle"
    amr.job = None

def force_enter_if_on_station(amr: AMRState):
    """If an AMR is on its station tile while in prod-phase, begin work immediately."""
    if amr.phase == "prod" and amr.job and amr.state in ("idle", "move"):
        st_cell = STATION_POS[amr.job.station]
        if cur_cell(amr) == st_cell:
            amr.state = "work"
            amr.work_left = amr.job.proc_time
            if amr.inv.get(amr.job.jtype, 0) > 0:
                amr.inv[amr.job.jtype] -= 1
            # clear route line when we start working
            amr.path = []
            amr.waypoint_idx = 0
            _update_route_artist(amr)

# ---------- simple motion (no collision checks) ----------
def simple_step(amrs: Dict[int, AMRState], dt: float):
    budget = CELLS_PER_SEC * dt
    for st in amrs.values():
        if st.state == "move" and st.path:
            st.move_budget += budget

    for st in amrs.values():
        while st.state == "move" and st.move_budget >= 1.0 and st.waypoint_idx < len(st.path):
            nxt = st.path[st.waypoint_idx]
            st.posx, st.posy = float(nxt[0]), float(nxt[1])
            st.waypoint_idx += 1
            st.move_budget -= 1.0
            if st.waypoint_idx >= len(st.path):
                # arrived at target -> decide next action immediately
                grant_arrival_action(st, amrs)
                break
    # keep route line visible as the remaining path; when consumed, grant_arrival_action cleared it

# ---------- Drawing ----------
_wait_rect: Optional[Rectangle] = None
_wait_texts: List[plt.Text] = []

def draw_static(ax):
    ax.set_xlim(-0.5, GRID_W - 0.5); ax.set_ylim(-0.5, GRID_H - 0.5)
    ax.set_xticks(range(GRID_W)); ax.set_yticks(range(GRID_H))
    ax.grid(True, which='both', linewidth=0.4, color="black", alpha=0.4)
    # obstacles
    for (x, y) in OBSTACLES:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=(0.8, 0.8, 0.8), edgecolor='none', zorder=1))
    # production stations
    for sid, (sx, sy) in STATION_POS.items():
        ax.add_patch(Rectangle((sx - 0.5, sy - 0.5), 1, 1, facecolor="none", edgecolor="tab:red", linewidth=2.0, zorder=2))
        ax.text(sx + 0.6, sy + 0.2, f"S{sid}", fontsize=11, color="tab:red", weight="bold", zorder=3)
    # material stations
    for mt, (mx, my) in MAT_POS.items():
        ax.add_patch(Rectangle((mx - 0.5, my - 0.5), 1, 1, facecolor="none", edgecolor="tab:blue", linewidth=2.0, zorder=2))
        ax.text(mx - 0.4, my - 0.4, f"M{mt}", fontsize=11, color="tab:blue", weight="bold", zorder=3)

def update_assigned_panel(ax, assigned: List[Job]):
    global _wait_rect, _wait_texts
    if _wait_rect is not None:
        _wait_rect.remove(); _wait_rect = None
    for t in _wait_texts:
        try: t.remove()
        except Exception:
            pass
    _wait_texts.clear()

    x0, y0, w, h = -0.5, GRID_H - 0.5, 9.3, 3.5
    _wait_rect = Rectangle((x0, y0), w, h, facecolor="white", edgecolor="black", linewidth=1.0, zorder=10)
    ax.add_patch(_wait_rect)
    _wait_texts.append(ax.text(x0 + 0.3, y0 + h - 0.6, "Assigned (from Scheduler)",
                               fontsize=10, weight="bold", color="gray", zorder=11))
    if not assigned:
        _wait_texts.append(ax.text(x0 + 0.3, y0 + h - 1.3, "(empty)", fontsize=9, color="gray", zorder=11))
        return

def ensure_amrs(ax) -> Dict[int, AMRState]:
    amrs: Dict[int, AMRState] = {}
    starts = {1: (2.0, 3.0), 2: (2.0, 6.0), 3: (2.0, 9.0)}
    for i in range(1, AMR_COUNT + 1):
        x, y = starts[i]
        st = AMRState(amr_id=i, posx=x, posy=y)
        mk = Circle((x, y), radius=0.35, facecolor="white", edgecolor="black", linewidth=1.8, zorder=5)
        ax.add_patch(mk); st.marker = mk
        st.label = ax.text(x, y + 0.55, f"AMR{i} (idle)", fontsize=9, ha="center", va="bottom", zorder=6)
        st.hud = ax.text(x, y - 0.75, "A10 B10 C10", fontsize=8, ha="center", va="top", color="gray", zorder=6)
        amrs[i] = st
    return amrs

# ---------- Main ----------
def main():
    global _AX
    fig, ax = plt.subplots(figsize=(12.8, 6))
    _AX = ax  # expose axis to helpers for drawing route lines

    draw_static(ax)
    amrs = ensure_amrs(ax)
    assigned_panel: List[Job] = []  # for UI list

    is_running = True
    sim_t = 0.0

    update_assigned_panel(ax, assigned_panel)
    timer = fig.canvas.new_timer(interval=UPDATE_INTERVAL_MS)

    def tick():
        nonlocal is_running, sim_t
        changed = False

        # ingest
        if ingest_new_assignments(ax, amrs, assigned_panel):
            update_assigned_panel(ax, assigned_panel)
            changed = True

        if is_running:
            dt = (UPDATE_INTERVAL_MS / 1000.0) * SIM_SPEED_MULT
            sim_t += dt

            # work / finish → egress (right-side slot)
            for st in amrs.values():
                do_work_and_maybe_finish(st, dt, assigned_panel, amrs)

            # move along paths (no collision logic)
            simple_step(amrs, dt)

            # if we happen to be on the station tile, start work now
            for st in amrs.values():
                force_enter_if_on_station(st)

            # promote idles into station as soon as free
            if ENFORCE_STATION_EXCLUSIVE:
                for st in amrs.values():
                    if st.phase == "prod" and st.job and st.state == "idle":
                        stid = st.job.station
                        if not is_station_occupied(stid, amrs, ignore_id=st.amr_id):
                            plan_path(st, STATION_POS[stid], amrs)
                            if len(st.path) > 1:
                                st.state = "move"
                            else:
                                st.state = "work"
                                st.work_left = st.job.proc_time
                                if st.inv.get(st.job.jtype, 0) > 0:
                                    st.inv[st.job.jtype] -= 1

            # visuals
            for k, st in amrs.items():
                if st.marker: st.marker.center = (st.posx, st.posy)
                if st.label:
                    status = (
                        "idle" if st.state == "idle" else
                        ("mv→sup" if (st.state == "move" and st.phase == "supply") else
                         ("mv→prod" if st.state == "move" and st.phase == "prod" else
                          ("egress" if st.phase == "egress" else f"work {max(0.0, st.work_left):.1f}"))))
                    st.label.set_text(f"AMR{k} ({status})")
                    st.label.set_position((st.posx, st.posy + 0.55))
                if st.hud:
                    st.hud.set_text(f"A{st.inv['A']} B{st.inv['B']} C{st.inv['C']}")
                    st.hud.set_position((st.posx, st.posy - 0.75))

            update_assigned_panel(ax, assigned_panel)
            changed = True

        if changed:
            fig.canvas.draw_idle()
        timer.start()

    def on_key(e):
        nonlocal is_running
        if e.key == " ":
            is_running = not is_running

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer.add_callback(tick); timer.start()
    plt.show()

if __name__ == "__main__":
    main()
