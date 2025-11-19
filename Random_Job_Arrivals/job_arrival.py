#!/usr/bin/env python3
# Two Queues (Move) but only show Job Queue visually
# - Every 10 sim-sec: MOVE all jobs from top (Job Queue) and write dispatch_inbox.jsonl.
# - Each job is assigned a random station (S1/S2/S3) at arrival time.
# - Job types are NEVER changed; they stay as generated on arrival.
# - Bottom "dispatching queue" is not drawn at all, but its data is used for JSONL.

import math, random
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import json, time, os

# reset outboxes each run
open("dispatch_inbox.jsonl", "w").close()
open("schedule_outbox.jsonl", "w").close()

DISPATCH_INBOX = "dispatch_inbox.jsonl"

def append_dispatch_inbox(jobs, dispatch_time):
    rec = {
        "generated_at": time.time(),
        "dispatch_time": float(dispatch_time),
        "jobs": [
            {
                "jid": int(j.jid),
                "type": str(j.jtype),
                "proc_time": float(j.proc_time),
                "station": int(j.station),
            }
            for j in jobs
        ],
    }
    with open(DISPATCH_INBOX, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[inbox] appended {len(jobs)} jobs -> {os.path.abspath(DISPATCH_INBOX)}")


# --------------------------- Config ---------------------------
AVG_INTERARRIVAL_TIME = 5.0
SIM_SPEED_MULTIPLIER   = 1.0
UPDATE_INTERVAL_MS     = 200

LEFT_LABEL_PAD = 5.0
VIEW_WIDTH     = 40.0

AX_Y_MIN, AX_Y_MAX = 0.0, 2.0

# Top lane (Job Queue)
TOP_Y_CENTER   = 0.6
TOP_LANE_H     = 0.75

# We still keep station_count for station assignment logic (JSONL),
# but we won't draw the bottom lanes.
STATION_COUNT  = 3

# Dispatch timings
DISPATCH_PERIOD_S = 10.0
CLEAR_DELAY_S     = 10.0

# Job types (block widths)
JOB_TYPES: Dict[str, Dict[str, float]] = {
    "A": {"time": 10},
    "B": {"time": 15},
    "C": {"time": 20},
}
JOB_TYPE_KEYS = list(JOB_TYPES.keys())

# Colors
_cycle = plt.rcParams.get("axes.prop_cycle", None)
_cycle_list = _cycle.by_key()["color"] if _cycle else ["C0", "C1", "C2", "C3", "C4"]
COLORS = {k: _cycle_list[i % len(_cycle_list)] for i, k in enumerate(JOB_TYPE_KEYS)}


# --------------------------- State ---------------------------
@dataclass
class Job:
    jid: int
    jtype: str
    proc_time: float
    arrival_ts: float
    station: Optional[int] = None  # final station (decided at arrival)

simulation_time: float = 0.0
job_counter: int = 0
next_arrival_time: float = 0.0
is_running: bool = False

# Top queue (Job Queue)
jobs_top: List[Job] = []
rects_top: List[Rectangle] = []
texts_top: List = []

# Bottom queue (Dispatching Queue) — logic only, no drawing
jobs_bot: List[Job] = []

next_dispatch_time: float = 0.0
next_clear_time: Optional[float] = None


# --------------------------- Scheduling helpers ---------------------------
def exp_wait(mean: float) -> float:
    u = random.random()
    return -math.log(1.0 - u) * mean

def schedule_next_arrival():
    global next_arrival_time
    next_arrival_time = simulation_time + exp_wait(AVG_INTERARRIVAL_TIME)

def schedule_next_dispatch():
    global next_dispatch_time
    next_dispatch_time = simulation_time + DISPATCH_PERIOD_S

def schedule_next_clear():
    global next_clear_time
    next_clear_time = simulation_time + CLEAR_DELAY_S


# --------------------------- Artist helpers ---------------------------
def remove_all_artists(rects_list: List[Rectangle], texts_list: List):
    for a in rects_list:
        try:
            a.remove()
        except Exception:
            pass
    for t in texts_list:
        try:
            t.remove()
        except Exception:
            pass
    rects_list.clear()
    texts_list.clear()

def add_job_artist_top(ax, j: Job):
    # Lay out blocks left-to-right in the job queue
    x_start = LEFT_LABEL_PAD + sum(job.proc_time / 2 for job in jobs_top[:-1])
    r = Rectangle(
        (x_start, TOP_Y_CENTER),
        j.proc_time / 2,
        TOP_LANE_H,
        linewidth=2.0,
        edgecolor="black",
        facecolor=COLORS[j.jtype],
        clip_on=True,
    )
    ax.add_patch(r)
    rects_top.append(r)
    t = ax.text(
        x_start + j.proc_time / 4.0,
        TOP_Y_CENTER + TOP_LANE_H / 2,
        f"J_{j.jid}: S_{j.station}",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        clip_on=True,
    )
    texts_top.append(t)


# --------------------------- Queue ops ---------------------------
def spawn_job_top(ax):
    global job_counter
    jtype = random.choice(JOB_TYPE_KEYS)
    proc_time = JOB_TYPES[jtype]["time"]
    jid = job_counter
    job_counter += 1

    # Assign random station immediately on arrival
    station = random.randint(1, STATION_COUNT)

    j = Job(
        jid=jid,
        jtype=jtype,
        proc_time=proc_time,
        arrival_ts=simulation_time,
        station=station,
    )
    jobs_top.append(j)
    add_job_artist_top(ax, j)

def dispatch_move(ax):
    """MOVE all current jobs from top (Job Queue),
       and write dispatch_inbox.jsonl.
       Stations are already assigned at arrival; no change here.
       No bottom visualization.
    """
    global jobs_top, jobs_bot
    if not jobs_top:
        jobs_bot = []
        return

    # Remove top visuals, move data, clear top
    remove_all_artists(rects_top, texts_top)
    moving = jobs_top
    jobs_top = []

    # DO NOT change j.station here; it was set at spawn time.
    jobs_bot = list(moving)
    append_dispatch_inbox(jobs_bot, simulation_time)

def clear_dispatching_lane():
    """Logically clear the dispatching queue (for timing semantics)."""
    global jobs_bot
    jobs_bot.clear()


# --------------------------- Decorations ---------------------------
def draw_static_decor(ax):
    band_frac = 0.10  # narrow band on the left for "Job Queue" label

    # Left panel for Job Queue label (axes coordinates)
    job_panel = Rectangle(
        (0.0, 0.0),
        band_frac,
        1.0,
        transform=ax.transAxes,
        fill=False,
        linewidth=1.8,
        clip_on=False,
        zorder=3,
    )
    ax.add_patch(job_panel)
    job_txt = ax.text(
        band_frac * 0.5,
        0.5,
        "Job Queue",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color="gray",
        clip_on=True,
        zorder=4,
    )
    job_txt.set_clip_path(job_panel)

    # Legend for job types
    handles = [
        Patch(facecolor=COLORS[k], edgecolor="black", label=f"Type {k}")
        for k in JOB_TYPE_KEYS
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)

def update_title(ax):
    status = "RUNNING" if is_running else "PAUSED"
    ax.set_title(
        f"t={simulation_time:.2f}s — {status}"
    )


# --------------------------- Main ---------------------------
def main():
    global is_running, simulation_time, next_arrival_time, next_dispatch_time, next_clear_time

    fig, ax = plt.subplots(figsize=(13, 3))
    ax.set_ylim(AX_Y_MIN, AX_Y_MAX)
    ax.set_xlim(0.0, VIEW_WIDTH)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)

    draw_static_decor(ax)
    update_title(ax)

    # init schedules
    schedule_next_arrival()
    schedule_next_dispatch()
    next_clear_time = None

    # timer-driven update
    timer = fig.canvas.new_timer(interval=UPDATE_INTERVAL_MS)

    def tick():
        global simulation_time, next_arrival_time, next_dispatch_time, next_clear_time

        if is_running:
            dt = (UPDATE_INTERVAL_MS / 1000.0) * SIM_SPEED_MULTIPLIER
            simulation_time += dt

            # arrivals
            if simulation_time >= next_arrival_time:
                spawn_job_top(ax)
                schedule_next_arrival()

            # dispatch move top -> logical bottom + write JSONL
            if simulation_time >= next_dispatch_time:
                dispatch_move(ax)
                schedule_next_dispatch()
                schedule_next_clear()

            # clear logical dispatching lane after delay
            if next_clear_time is not None and simulation_time >= next_clear_time:
                clear_dispatching_lane()
                next_clear_time = None

            update_title(ax)
            fig.canvas.draw_idle()

        timer.start()

    timer.add_callback(tick)
    timer.start()

    def on_key(event):
        nonlocal ax
        global is_running
        if event.key == " ":
            is_running = not is_running
            update_title(ax)
            fig.canvas.draw_idle()
        elif event.key and event.key.lower() == "n":
            # Allow manual spawn; schedule next arrival if not already
            if simulation_time == 0.0 and next_arrival_time == 0.0:
                schedule_next_arrival()
            spawn_job_top(ax)
            update_title(ax)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
