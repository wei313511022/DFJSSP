# priority_astar_two_cars.py
# A（藍，高優先）直行不讓；B（紅，低優先）接近時才 A* 轉向
# 前方成本＝平滑漸層；通過 A 後，B 立即回到原車道再前進
# 輸出：priority_astar_two_cars.gif

import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.patches import Rectangle, FancyArrow

# ---------------- Config ----------------
GRID_W, GRID_H = 30, 14
CELL = 26
FPS = 60
PROXIMITY_TRIGGER = 10
MAX_STEPS = 200
REPLAN_MODE = "once"           # "once" or "limited"
STEPS_BETWEEN_REPLAN = 3
TURN_PENALTY = 0.15            # 垂直移動小懲罰，讓 B 不會太早偏航

LANE_Y = GRID_H // 2           # 原本要走的車道（中線）

# A: 左->右（高優先）
A_start = (2, LANE_Y)
A_goal  = (GRID_W - 3, LANE_Y)

# B: 右->左（低優先）
B_start = (GRID_W - 3, LANE_Y)
B_goal  = (2, LANE_Y)

# 邊框當障礙
obstacles = set()
for y in range(GRID_H):
    obstacles.add((0, y)); obstacles.add((GRID_W - 1, y))
for x in range(GRID_W):
    obstacles.add((x, 0)); obstacles.add((x, GRID_H - 1))

def in_bounds(p):
    x, y = p
    return 0 <= x < GRID_W and 0 <= y < GRID_H

def neighbors4(p):
    x, y = p
    cand = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    return [q for q in cand if in_bounds(q) and q not in obstacles]

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# ---------------- A* ----------------
def astar(start, goal, costmap, turn_penalty=0.0):
    open_set = [(manhattan(start, goal), 0.0, start, None)]
    gbest = {start: 0.0}
    parent = {}

    while open_set:
        f, g, u, par = heapq.heappop(open_set)
        if u in parent:   # finalized
            continue
        parent[u] = par
        if u == goal:
            break
        for v in neighbors4(u):
            step = 1.0 + float(costmap[v[1], v[0]])
            if v[1] != u[1]:
                step += turn_penalty
            ng = g + step
            if v not in gbest or ng < gbest[v]:
                gbest[v] = ng
                nf = ng + manhattan(v, goal)
                heapq.heappush(open_set, (nf, ng, v, u))

    if goal not in parent:
        return [start]
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def straight_line_path(p0, p1):
    """先水平、後垂直（目標和起點同 y 時，只會水平）。"""
    (x0, y0), (x1, y1) = p0, p1
    path = []
    step = 1 if x1 >= x0 else -1
    for x in range(x0, x1 + step, step):
        path.append((x, y0))
    if y1 != y0:
        ystep = 1 if y1 > y0 else -1
        for y in range(y0 + ystep, y1 + ystep, ystep):
            path.append((x1, y))
    return path

def merge_back_path(current, goal_x, lane_y):
    """通過 A 後：先垂直回到 lane_y，再水平到 goal_x（穩定、不抖動）。"""
    x0, y0 = current
    path = [(x0, y0)]
    if y0 != lane_y:
        ystep = 1 if lane_y > y0 else -1
        for y in range(y0 + ystep, lane_y + ystep, ystep):
            path.append((x0, y))
    if x0 != goal_x:
        step = -1 if goal_x < x0 else 1
        for x in range(x0 + step, goal_x + step, step):
            path.append((x, lane_y))
    return path

# ---------------- 平滑漸層成本地圖 ----------------
def build_costmap(a_pos, a_dir=(1,0),
                  global_amp=3.0, global_side_decay=2.5, global_front_scale=6.0,
                  bump_amp=6.0, sigma_front=2.0, sigma_side=1.2,
                  behind_cost=0.05, base_bias=0.2):
    """
    1) 全域前方走廊（平滑）：global_amp * exp(-|side|/global_side_decay) * 1/(1 + front/global_front_scale)
    2) 當下位置前方「平滑高斯 bump」：bump_amp * exp(-(front^2/sigma_front^2) - (side^2/sigma_side^2)), front>0
    3) 後方小成本 behind_cost；無硬塊
    """
    base = np.zeros((GRID_H, GRID_W), dtype=float)
    ax, ay = a_pos
    for y in range(GRID_H):
        for x in range(GRID_W):
            if (x, y) in obstacles:
                base[y, x] = 9.0
                continue
            vx, vy = x-ax, y-ay
            front = vx*a_dir[0] + vy*a_dir[1]
            side  = -vx*a_dir[1] + vy*a_dir[0]

            cost = behind_cost
            if front > 0:
                corridor = global_amp * math.exp(-abs(side)/global_side_decay) * (1.0/(1.0 + front/global_front_scale))
                bump = bump_amp * math.exp(-(front**2)/(sigma_front**2) - (side**2)/(sigma_side**2))
                cost = base_bias + corridor + bump
            base[y, x] = cost

    for (x, y) in obstacles:
        base[y, x] = 9.0
    return base

# ---------------- A 走直線 ----------------
A_path = straight_line_path(A_start, A_goal)

# ---------------- 模擬 ----------------
A_pos = list(A_start)
B_pos = list(B_start)

B_has_triggered = False
B_has_passed = False          # <- 新增：是否已經通過 A（x 已在 A 左側）
tick = 0
steps = 0

B_path = straight_line_path(B_start, B_goal)   # 觸發前不 A*
B_path_idx = 0
A_path_idx = 0

frames_positions = []
frames_costmap = []

while steps < MAX_STEPS and (tuple(A_pos) != A_goal or tuple(B_pos) != B_goal):
    steps += 1

    # A 前進（直線）
    if A_path_idx < len(A_path) - 1:
        A_path_idx += 1
        A_pos = list(A_path[A_path_idx])

    # 觸發再規劃（尚未通過 A 才會觸發）
    if (not B_has_triggered) and manhattan(tuple(A_pos), tuple(B_pos)) <= PROXIMITY_TRIGGER:
        B_has_triggered = True
        tick = 0

    # 依照 A 當前位置建立「平滑漸層」成本圖
    cur_costmap = build_costmap(tuple(A_pos), a_dir=(1,0))

    # --- B 的行為 ---
    if not B_has_passed:
        # 還沒通過 A：可能需要 A* 轉向
        if B_has_triggered:
            if REPLAN_MODE == "once" and tick == 0:
                B_path = astar(tuple(B_pos), B_goal, cur_costmap, turn_penalty=TURN_PENALTY)
                B_path_idx = 0
            elif REPLAN_MODE == "limited":
                if tick % STEPS_BETWEEN_REPLAN == 0 or B_path_idx >= len(B_path) - 1:
                    B_path = astar(tuple(B_pos), B_goal, cur_costmap, turn_penalty=TURN_PENALTY)
                    B_path_idx = 0
            tick += 1

        # 沿目前路徑走一步
        if B_path_idx < len(B_path) - 1:
            B_path_idx += 1
            B_pos = list(B_path[B_path_idx])

        # 檢查是否「通過」A（B 往左、A 往右：通過條件為 B.x <= A.x）
        if B_pos[0] <= A_pos[0]:
            B_has_passed = True
            # 立刻規劃「回到原車道再到終點」的合併路徑（停止後續 A*）
            merge_path = merge_back_path(tuple(B_pos), B_goal[0], LANE_Y)
            B_path = merge_path
            B_path_idx = 0

    else:
        # 已通過 A：只跟隨合併路徑，不再 A*
        if B_path_idx < len(B_path) - 1:
            B_path_idx += 1
            B_pos = list(B_path[B_path_idx])

    # 記錄影格
    frames_positions.append((tuple(A_pos), tuple(B_pos), list(A_path), list(B_path)))
    frames_costmap.append(cur_costmap.copy())

    if tuple(A_pos) == A_goal and tuple(B_pos) == B_goal:
        break

# ---------------- 繪製動畫 ----------------
fig_w = GRID_W * CELL / 100
fig_h = GRID_H * CELL / 100
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
ax.set_xlim(0, GRID_W); ax.set_ylim(0, GRID_H)
ax.set_aspect('equal'); ax.axis('off')

# 平滑色帶 + 固定尺度
vmin = 0.0
vmax = 1.0 + 3.0 + 6.0
img = ax.imshow(frames_costmap[0], origin='lower', extent=(0,GRID_W,0,GRID_H),
                interpolation='bilinear', vmin=vmin, vmax=vmax, cmap='viridis')

# 淡格線
for x in range(GRID_W):
    ax.plot([x,x],[0,GRID_H], linewidth=0.2, alpha=0.25, color='black')
for y in range(GRID_H):
    ax.plot([0,GRID_W],[y,y], linewidth=0.2, alpha=0.25, color='black')

# 車子
carA = Rectangle((A_start[0]+0.1, A_start[1]+0.1), 0.8, 0.8, facecolor="#3b82f6", edgecolor="black", linewidth=0.8)
carB = Rectangle((B_start[0]+0.1, B_start[1]+0.1), 0.8, 0.8, facecolor="#ef4444", edgecolor="black", linewidth=0.8)
ax.add_patch(carA); ax.add_patch(carB)

# 指向箭頭
arrA = FancyArrow(A_start[0]+0.5, A_start[1]+0.9, 1.2, 0.0, width=0.05, head_width=0.25, head_length=0.35, length_includes_head=True)
arrB = FancyArrow(B_start[0]+0.5, B_start[1]+0.9, -1.2, 0.0, width=0.05, head_width=0.25, head_length=0.35, length_includes_head=True)
ax.add_patch(arrA); ax.add_patch(arrB)

# 路徑
lineA, = ax.plot([], [], linewidth=2.0, color="#1f77b4")
lineB, = ax.plot([], [], linewidth=2.0, color="#d62728", linestyle="--")

title = ax.text(
    0.02, 0.98,
    "A (blue) keeps straight.  B (red) goes straight, replans near A with A*,\nthen merges back to the lane right after passing A.",
    transform=ax.transAxes, ha="left", va="top", fontsize=12,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9)
)

def update(frame_idx):
    a_pos, b_pos, a_path, b_path = frames_positions[frame_idx]
    img.set_data(frames_costmap[frame_idx])

    carA.set_xy((a_pos[0]+0.1, a_pos[1]+0.1))
    carB.set_xy((b_pos[0]+0.1, b_pos[1]+0.1))

    aN = min(frame_idx+1, len(a_path))
    axA = [p[0]+0.5 for p in a_path[:aN]]
    ayA = [p[1]+0.5 for p in a_path[:aN]]
    lineA.set_data(axA, ayA)

    axB = [p[0]+0.5 for p in b_path]
    ayB = [p[1]+0.5 for p in b_path]
    lineB.set_data(axB, ayB)

    return (img, carA, carB, lineA, lineB, title)

anim = FuncAnimation(fig, update, frames=len(frames_positions), interval=1000 / FPS, blit=False)
anim.save("priority_astar_two_cars.gif", writer=PillowWriter(fps=FPS))
print("Saved: priority_astar_two_cars.gif")
