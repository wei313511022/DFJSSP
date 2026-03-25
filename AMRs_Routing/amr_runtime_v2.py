import math
import heapq
from typing import Dict, List, Tuple, Set, Optional
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# Import essential maps from GA
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Static_alogorithm/GA_code')))
from GA import STATIONS, OBSTACLES, AMR_STARTS, AMR_KEYS, TYPE_DURATION

GRID_W, GRID_H = 10, 10
MATERIAL_CAPACITY = 3 # Aligned with RL config
CELLS_PER_SEC = 1.0

# Material stations (left side) by job type
MAT_POS: Dict[str, Tuple[float, float]] = {
    "C": (0.0, 1.0),
    "B": (0.0, 4.0),
    "A": (0.0, 7.0),
}
STATION_POS = {
    1: (9.0, 8.0), 
    2: (9.0, 6.0), 
    3: (9.0, 4.0), 
    4: (9.0, 2.0), 
    5: (9.0, 0.0)
}

Coord = Tuple[int, int]

class RuntimeJob:
    def __init__(self, jid: int, jtype: str, proc_time: float, station: int):
        self.jid = jid
        self.jtype = jtype
        self.proc_time = proc_time
        self.station = station # 1-5
        
class AMRState:
    def __init__(self, amr_id: int, start_x: float, start_y: float):
        self.amr_id = amr_id
        self.posx = start_x
        self.posy = start_y
        self.state = "idle" # "idle", "move", "process"
        self.blocked = False
        self.queue: List[RuntimeJob] = []
        
        self.job: Optional[RuntimeJob] = None
        self.phase: Optional[str] = None # None, "supply", "deliver"
        self.inventory = {"A": MATERIAL_CAPACITY, "B": MATERIAL_CAPACITY, "C": MATERIAL_CAPACITY}
        
        self.path: List[Coord] = []
        self.waypoint_idx = 0
        self.move_budget = 0.0
        self.proc_timer = 0.0
        
        self.tot_number_of_jobs = 0
        
    def cur_cell(self) -> Coord:
        return (int(round(self.posx)), int(round(self.posy)))

def manhattan(a: Coord, b: Coord) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

def astar_4dir(start: Coord, goal: Coord, blocked: Set[Coord]) -> List[Coord]:
    if start == goal: return [start]
    if start in blocked: return []
    
    STEPS = [(1,0), (-1,0), (0,1), (0,-1)]
    openh = []
    heapq.heappush(openh, (manhattan(start, goal), 0.0, start))
    came = {}
    gscore = {start: 0.0}
    closed = set()
    
    while openh:
        _, gs, cur = heapq.heappop(openh)
        if cur in closed: continue
        if cur == goal:
            path = [goal]
            c = goal
            while c != start:
                c = came[c]
                path.append(c)
            path.reverse()
            return path
        closed.add(cur)
        
        x, y = cur
        for dx, dy in STEPS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in blocked:
                ns = gs + 1.0
                if ns < gscore.get((nx, ny), math.inf):
                    came[(nx, ny)] = cur
                    gscore[(nx, ny)] = ns
                    heapq.heappush(openh, (ns + manhattan((nx, ny), goal), ns, (nx, ny)))
    return []

class FactorySimulator:
    def __init__(self):
        self.time = 0.0
        self.amrs = {
            1: AMRState(1, AMR_STARTS["AMR1"][0], AMR_STARTS["AMR1"][1]),
            2: AMRState(2, AMR_STARTS["AMR2"][0], AMR_STARTS["AMR2"][1]),
            3: AMRState(3, AMR_STARTS["AMR3"][0], AMR_STARTS["AMR3"][1])
        }
        # Clear initial inventory to match GNN_DDQN initialized stat
        for amr in self.amrs.values():
            amr.inventory = {"A": 0, "B": 0, "C": 0}
        
        self.completed_jobs_jids = []
        self._ax = None
        self._fig = None

    def assign_schedules(self, order: List[int], assignments: List[str], job_map: Dict[int, dict]):
        """
        Receives schedule pairing from GNN_DDQN_V6.py.
        order: list of job idxs
        assignments: list of AMR strings e.g. "AMR1"
        job_map: idx -> {'jid': int, 'jtype': 'A', 'time': float, 'station': int}
        """
        for amr in self.amrs.values():
            amr.queue.clear()
            
        for job_idx, amr_str in zip(order, assignments):
            jdata = job_map[job_idx]
            aid = int(amr_str.replace("AMR", ""))
            job = RuntimeJob(jdata['jid'], jdata['jtype'], jdata['time'], jdata['station'])
            self.amrs[aid].queue.append(job)

    def get_amr_priority(self, amr: AMRState) -> int:
        if amr.state == "process": return 100
        if amr.state == "idle": return 10
        if amr.phase == "deliver": return 50
        if amr.phase == "supply": return 40
        return 30

    def _build_blocked(self, amr) -> Set[Coord]:
        blocked = set(OBSTACLES)
        
        # Base priority (x1000) + ID tie-breaker (AMR 1 is technically -1, beating AMR 2 which is -2)
        my_pri = self.get_amr_priority(amr) * 1000 - amr.amr_id
        
        for oid, other in self.amrs.items():
            if oid == amr.amr_id: continue
            ocell = other.cur_cell()
            
            # Always block parking spaces if they are occupied or if an AMR is definitively blocked
            if ocell in STATION_POS.values() or ocell in MAT_POS.values() or other.blocked:
                blocked.add(ocell)
                
            o_pri = self.get_amr_priority(other) * 1000 - other.amr_id
            
            if o_pri > my_pri: # Only yielding to AMRs with strictly higher phase priority!
                blocked.add(ocell)
                if other.state == "move" and other.path and 0 <= other.waypoint_idx < len(other.path):
                    nxt = other.path[other.waypoint_idx]
                    blocked.add(nxt)
                    
        c = amr.cur_cell()
        if c in blocked: blocked.remove(c)
        return blocked

    def try_start_next(self, amr: AMRState):
        if amr.state != "idle" or amr.job is not None or not amr.queue:
            return
        amr.job = amr.queue.pop(0)
        jtype = amr.job.jtype
        if amr.inventory.get(jtype, 0) <= 0:
            amr.phase = "supply"
        else:
            amr.phase = "deliver"
        amr.state = "move"

    def _plan(self, amr: AMRState, goal: Coord):
        start = amr.cur_cell()
        blocked = self._build_blocked(amr)
        
        # Check if another AMR is sitting ON our exact target goal.
        # We don't care about priorities for the end destination parking spot.
        occupied = any(o.cur_cell() == goal and o.amr_id != amr.amr_id for o in self.amrs.values())
        
        if goal in blocked or occupied:
            cands = [(goal[0]+1, goal[1]), (goal[0]-1, goal[1]), (goal[0], goal[1]+1), (goal[0], goal[1]-1)]
            best = None
            for cx, cy in cands:
                if 0 <= cx < GRID_W and 0 <= cy < GRID_H and (cx, cy) not in blocked:
                    p = astar_4dir(start, (cx, cy), blocked)
                    if p and len(p) > 1:
                        if not best or len(p) < len(best):
                            best = p
            if best:
                amr.path = best; amr.blocked = False; amr.waypoint_idx = 1
                return
            amr.path = []; amr.blocked = True; amr.waypoint_idx = 0
            return
            
        path = astar_4dir(start, goal, blocked)
        if not path or len(path) <= 1:
            amr.path = []; amr.blocked = True; amr.waypoint_idx = 0
            return
        amr.path = path; amr.blocked = False; amr.waypoint_idx = 1

    def step(self, dt: float) -> int:
        """Advance time by dt seconds. Returns number of completed jobs in this step."""
        # Simulate second by second perfectly align with integer job times
        tick_sz = 1.0
        steps = int(dt / tick_sz)
        rem = dt - (steps * tick_sz)
        completed = 0
        
        for _ in range(steps):
            completed += self._micro_step(tick_sz)
        if rem > 0:
            completed += self._micro_step(rem)
            
        self.time += dt
        return completed

    def _micro_step(self, dt: float) -> int:
        comp = 0
        for amr in self.amrs.values():
            if amr.state == "idle" and amr.job is None:
                self.try_start_next(amr)
            
            if amr.state == "move":
                if amr.phase == "supply": goal = MAT_POS[amr.job.jtype]
                else: goal = STATION_POS[amr.job.station]
                self._plan(amr, (int(goal[0]), int(goal[1])))
                
            elif amr.state == "process":
                amr.proc_timer -= dt
                if amr.proc_timer <= 0:
                    amr.state = "idle"
                    self.completed_jobs_jids.append(amr.job.jid)
                    amr.tot_number_of_jobs += 1
                    comp += 1
                    amr.job = None
                    amr.phase = None
                    self.try_start_next(amr)
                
        # accumulate budget
        for amr in self.amrs.values():
            if amr.state == "move" and amr.path:
                amr.move_budget += CELLS_PER_SEC * dt

        # Move
        for amr in self.amrs.values():
            while amr.state == "move" and amr.move_budget >= CELLS_PER_SEC * 1.0 and amr.waypoint_idx < len(amr.path):
                # We move carefully. Only commit to cell entry if not colliding.
                nxt = amr.path[amr.waypoint_idx]
                if any(o.cur_cell() == nxt for o in self.amrs.values() if o != amr):
                    amr.blocked = True # wait
                    break
                    
                amr.posx, amr.posy = float(nxt[0]), float(nxt[1])
                amr.waypoint_idx += 1
                amr.move_budget -= CELLS_PER_SEC * 1.0 # Cost to move 1 cell
                amr.blocked = False
                
                if amr.waypoint_idx == len(amr.path):
                    # Arrived at path end
                    dest = nxt
                    station_goal = STATION_POS.get(amr.job.station, None) if amr.job else None
                    mat_goal = MAT_POS.get(amr.job.jtype, None) if amr.job else None
                    
                    if amr.phase == "supply" and dest == mat_goal:
                        amr.inventory[amr.job.jtype] = MATERIAL_CAPACITY
                        amr.phase = "deliver"
                    elif amr.phase == "deliver" and dest == station_goal:
                        before = amr.inventory.get(amr.job.jtype, 0)
                        amr.inventory[amr.job.jtype] = max(0, before - 1)
                        amr.state = "process"
                        amr.proc_timer = amr.job.proc_time
                        amr.path = []
                    else:
                        amr.path = []
                        amr.waypoint_idx = 0
                        amr.move_budget = 0.0
                    break
        return comp

    def get_state_snapshot(self) -> dict:
        """Returns snapshot for RL environment."""
        state = {
            "time": self.time,
            "positions": {f"AMR{amr.amr_id}": amr.cur_cell() for amr in self.amrs.values()},
            # Calculate total time to completion for availability metric
            "availability": {f"AMR{amr.amr_id}": self.time + (amr.proc_timer if amr.state == 'process' else 0.0) for amr in self.amrs.values()},
            "inventory": {f"AMR{amr.amr_id}": amr.inventory.copy() for amr in self.amrs.values()},
            "status": {f"AMR{amr.amr_id}": 1 if amr.state != "idle" else 0 for amr in self.amrs.values()},
            "remaining_time": {f"AMR{amr.amr_id}": amr.proc_timer if amr.state == 'process' else 0.0 for amr in self.amrs.values()},
            "tot_number_of_jobs": {f"AMR{amr.amr_id}": amr.tot_number_of_jobs for amr in self.amrs.values()},
            "queues": {f"AMR{amr.amr_id}": [j.jid for j in amr.queue] for amr in self.amrs.values()}
        }
        return state

    def _draw_static(self, ax):
        ax.set_xlim(-0.5, GRID_W - 0.5)
        ax.set_ylim(-0.5, GRID_H - 0.5)
        ax.set_xticks(range(GRID_W))
        ax.set_yticks(range(GRID_H))
        ax.grid(True, which="both", linewidth=0.4, color="black", alpha=0.4)

        for (x, y) in OBSTACLES:
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=(0.8, 0.8, 0.8), edgecolor="none", zorder=1))

        for sid, (sx, sy) in STATION_POS.items():
            ax.add_patch(Rectangle((sx - 0.5, sy - 0.5), 1, 1, facecolor="none", edgecolor="tab:red", linewidth=2.0, zorder=2))
            ax.text(sx - 0.4, sy + 0.15, f"S{sid}", fontsize=11, color="tab:red", weight="bold", zorder=3)

        for jt, (mx, my) in MAT_POS.items():
            ax.add_patch(Rectangle((mx - 0.5, my - 0.5), 1, 1, facecolor="none", edgecolor="tab:blue", linewidth=2.0, zorder=2))
            ax.text(mx - 0.4, my + 0.15, f"{jt}", fontsize=11, color="tab:blue", weight="bold", zorder=3)

    def _setup_amrs_plot(self, ax):
        for k, st in self.amrs.items():
            st.marker = Circle((st.posx, st.posy), radius=0.35, facecolor="white", edgecolor="black", linewidth=1.8, zorder=5)
            ax.add_patch(st.marker)
            st.label = ax.text(st.posx, st.posy + 0.55, f"AMR{k}", fontsize=9, ha="center", va="bottom", zorder=6)

    def render(self):
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(8, 8))
            self._fig.subplots_adjust(bottom=0.2)
            self._draw_static(self._ax)
            self._setup_amrs_plot(self._ax)
            
            self._timer_text = self._fig.text(0.5, 0.9, "Route Map", ha="center", va="bottom", fontsize=16, weight="bold", transform=self._fig.transFigure)
            self._status_texts = {}
            for k in sorted(self.amrs.keys()):
                self._status_texts[k] = self._fig.text(0.5, 0.03 + (4-k)*0.03, "", fontsize=14, ha="center", va="bottom", transform=self._fig.transFigure)
            
            plt.ion() # Interactive mode on
            plt.show()

        for k, st in self.amrs.items():
            if hasattr(st, 'marker') and st.marker:
                st.marker.center = (st.posx, st.posy)
            
            inv = st.inventory
            inv_str = f"A{inv.get('A',0)} B{inv.get('B',0)} C{inv.get('C',0)}"
            status_str = f"AMR{k}: {st.state}, {st.phase}, {inv_str}, (x:{st.posx:.1f} y:{st.posy:.1f})"
            if hasattr(st, 'label') and st.label:
                st.label.set_position((st.posx, st.posy + 0.55))
            if k in self._status_texts:
                self._status_texts[k].set_text(status_str)
        
        self._timer_text.set_text(f"Simulation time: {self.time:6.1f} s")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

