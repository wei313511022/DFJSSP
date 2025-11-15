# A* demo (8-direction moves) on a 10x10 grid with 3 AMRs and 3 Stops
# - Compute A* costs for all AMR→Stop pairs
# - Add 3 obstacles around each Stop (deterministic)
# - Solve min-sum assignment (unique Stop per AMR)
# - Plot only the chosen 3 routes (thick lines), with obstacles shown

import random, math, heapq
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from matplotlib.patches import Rectangle

# ----------------------- Config -----------------------
SEED = 42
GRID_W = GRID_H = 10
N_AMR, N_STOP = 3, 3

random.seed(SEED)
Coord = Tuple[int, int]

# ----------------------- Geometry helpers -----------------------
def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < GRID_W and 0 <= y < GRID_H

def neighbors8(p: Coord):
    x, y = p
    cand = [
        (x+1, y, 1.0), (x-1, y, 1.0), (x, y+1, 1.0), (x, y-1, 1.0),
        (x+1, y+1, math.sqrt(2)), (x-1, y+1, math.sqrt(2)),
        (x+1, y-1, math.sqrt(2)), (x-1, y-1, math.sqrt(2))
    ]
    return [(nx, ny, c) for nx, ny, c in cand if in_bounds(nx, ny)]

# Octile heuristic (orth=1, diag=√2)
def octile(a: Coord, b: Coord) -> float:
    dx = abs(a[0] - b[0]); dy = abs(a[1] - b[1])
    D, D2 = 1.0, math.sqrt(2.0)
    return D*(dx+dy) + (D2 - 2*D)*min(dx, dy)

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

def astar_8dir(start: Coord, goal: Coord, blocked: Optional[Set[Coord]] = None) -> List[Coord]:
    if blocked is None:
        blocked = set()
    if start == goal:
        return [start]
    if start in blocked or goal in blocked:
        return []

    # 8-neighbor deltas with costs
    STEPS = [
        ( 1,  0, 1.0), (-1,  0, 1.0), ( 0,  1, 1.0), ( 0, -1, 1.0),
        ( 1,  1, math.sqrt(2)), (-1,  1, math.sqrt(2)),
        ( 1, -1, math.sqrt(2)), (-1, -1, math.sqrt(2))
    ]

    def can_move(x: int, y: int, dx: int, dy: int) -> bool:
        nx, ny = x + dx, y + dy
        if not in_bounds(nx, ny): return False
        if (nx, ny) in blocked:   return False
        # ---- NO CORNER CUTTING for diagonal moves ----
        if dx != 0 and dy != 0:
            side1 = (x + dx, y)   # step horizontally
            side2 = (x, y + dy)   # step vertically
            if side1 in blocked or side2 in blocked:
                return False
        return True

    open_heap = []
    heapq.heappush(open_heap, (octile(start, goal), 0.0, start))
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}
    closed = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return reconstruct_path(came_from, start, goal)
        closed.add(current)

        x, y = current
        for dx, dy, step_cost in STEPS:
            if not can_move(x, y, dx, dy):
                continue
            nxt = (x + dx, y + dy)
            tentative = g + step_cost
            if tentative < g_score.get(nxt, math.inf):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + octile(nxt, goal)
                heapq.heappush(open_heap, (f, tentative, nxt))
    return []


# ----------------------- Fixed placement (your pattern) -----------------------
def sample_unique_coords_amrs(k: int, forbidden: set, x: int, y: int) -> List[Coord]:
    coords = []
    while len(coords) < k:
        c = (x, y)
        if c not in forbidden:
            forbidden.add(c); coords.append(c)
        y += 1
    return coords

def sample_unique_coords_stops(k: int, forbidden: set, x: int, y: int) -> List[Coord]:
    coords = []
    while len(coords) < k:
        c = (x, y)
        if c not in forbidden:
            forbidden.add(c); coords.append(c)
        y += 3
    return coords

used = set()
amrs  = sample_unique_coords_amrs(N_AMR, used, 1, 4)
stops = sample_unique_coords_stops(N_STOP, used, 7, 2)

# ----------------------- Build obstacles: 3 around each stop -----------------------
def obstacles_around_stops(stops: List[Coord], amrs: List[Coord], k_each: int = 3) -> Set[Coord]:
    blocked: Set[Coord] = set()
    occupied = set(stops) | set(amrs)

    # priority order: up, left, right, down, then diagonals (keeps at least one side open usually)
    deltas = [
        (0, 1), (-1, 0), (-1, 1), (0, -1),
        (-1, 1), (1, 1), (-1, -1), (1, -1)
    ]

    for sx, sy in stops:
        placed = 0
        for dx, dy in deltas:
            if placed >= k_each:
                break
            nx, ny = sx + dx, sy + dy
            c = (nx, ny)
            if in_bounds(nx, ny) and c not in occupied and c not in blocked:
                blocked.add(c)
                placed += 1
        # If fewer than k_each available (edge cases), it's fine—place as many as fit.
    return blocked

blocked = obstacles_around_stops(stops, amrs, k_each=3)

# ----------------------- Compute all pairs -----------------------
paths: Dict[Tuple[int,int], List[Coord]] = {}
dist_mat = np.zeros((N_AMR, N_STOP), dtype=float)

for i, a in enumerate(amrs):
    for j, s in enumerate(stops):
        p = astar_8dir(a, s, blocked=blocked)
        paths[(i, j)] = p
        if len(p) <= 1:
            dist_mat[i, j] = math.inf  # unreachable due to obstacles
        else:
            total = 0.0
            for u, v in zip(p[:-1], p[1:]):
                dx = abs(u[0] - v[0]); dy = abs(u[1] - v[1])
                total += math.sqrt(2.0) if dx == 1 and dy == 1 else 1.0
            dist_mat[i, j] = round(total, 3)

# ----------------------- Solve assignment: minimize total cost -----------------------
best_perm = None
best_total = math.inf
for perm in permutations(range(N_STOP)):  # perm[i] = stop index for AMR i
    total = sum(dist_mat[i, perm[i]] for i in range(N_AMR))
    if total < best_total:
        best_total = total
        best_perm = perm

chosen_pairs = []
chosen_paths = {}
if best_perm is not None and math.isfinite(best_total):
    for i in range(N_AMR):
        j = best_perm[i]
        if math.isfinite(dist_mat[i, j]) and len(paths[(i, j)]) > 1:
            chosen_pairs.append((i, j))
            chosen_paths[(i, j)] = paths[(i, j)]

# ----------------------- Print results -----------------------
print("AMR positions:", amrs)
print("Stop positions:", stops)
print("Blocked cells (count):", len(blocked))
print("\nCost matrix (inf = unreachable):")
header = ["Stop{}".format(j) for j in range(N_STOP)]
print("        " + "  ".join(f"{h:>8s}" for h in header))
for i in range(N_AMR):
    row = "AMR{}   ".format(i) + "  ".join(
        f"{dist_mat[i,j]:8.3f}" if math.isfinite(dist_mat[i,j]) else f"{'inf':>8s}"
        for j in range(N_STOP)
    )
    print(row)

if chosen_pairs:
    print("\nBest assignment (AMR -> Stop):")
    for i, j in chosen_pairs:
        print(f"  AMR{i} @ {amrs[i]}  ->  Stop{j} @ {stops[j]}   cost = {dist_mat[i,j]:.3f}")
    print(f"\nMinimum total cost = {best_total:.3f}")
else:
    print("\nNo feasible full assignment (some paths unreachable with these obstacles).")

# ----------------------- Plot -----------------------
fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
ax.set_facecolor("white")

ax.set_xlim(0, GRID_W)
ax.set_ylim(0, GRID_H)
ax.set_xticks(range(GRID_W))
ax.set_yticks(range(GRID_H))
ax.grid(True, which='both', linewidth=0.3, color="black")

# Draw obstacles as light gray squares
for (x, y) in blocked:
    ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, facecolor=(0.8,0.8,0.8), edgecolor='none', zorder=1))

# Markers
ax.scatter([x for x, y in amrs],  [y for x, y in amrs],
           marker='s', s=120, label='AMRs', facecolors='tab:blue', edgecolors='black', linewidths=0.8, zorder=3)
ax.scatter([x for x, y in stops], [y for x, y in stops],
           marker='^', s=120, label='Stops', facecolors='tab:red',  edgecolors='black', linewidths=0.8, zorder=3)

# # Draw chosen paths
# for (i, j), p in chosen_paths.items():
#     xs = [x for x, y in p]; ys = [y for x, y in p]
#     ax.plot(xs, ys, linewidth=3, zorder=2)

# Labels
for idx, (x, y) in enumerate(amrs):
    ax.text(x + 0.2, y + 0.2, f"AMR_{idx}", fontsize=14)
for idx, (x, y) in enumerate(stops):
    ax.text(x + 0.2, y + 0.2, f"Stop_{idx}", fontsize=14)

# ax.set_title("Min-Cost Assignment with 3 Obstacles Around Each Stop (10×10, 8-dir A*)")
ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
