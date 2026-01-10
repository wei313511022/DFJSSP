#!/usr/bin/env python3
# Program 2: Live AMR FIFO Scheduler (visual)
# - Tails 'dispatch_inbox.jsonl' (one JSON per line from Program 1).
# - Ingests batches into the top Dispatching Queue (waiting area).
# - Keeps scheduling while queue not empty:
#   FIFO jobs -> earliest-available AMR (ties -> lower index).
# - Renders like Program 1: fixed viewport, left label band; blocks overflow right.

import os, json, math, time, random
from dataclasses import dataclass
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# Output file for Program 3
SCHEDULE_OUTBOX = "schedule_outbox.jsonl"

def emit_assignment(amr_id, job):
    """Append a single assigned job to the schedule outbox (tailed by Program 3)."""
    rec = {
        "generated_at": time.time(),
        "amr": int(amr_id),
        "jid": int(job.jid),
        "type": str(job.jtype),
        "proc_time": float(job.proc_time),
        "station": str(job.station),
    }
    with open(SCHEDULE_OUTBOX, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ------------ Config ------------
INBOX_PATH         = "dispatch_inbox.jsonl"
AMR_COUNT          = 3
UPDATE_INTERVAL_MS = 250

# --- MODIFICATION: Isolate Job Types & Colors ---
# Edit this list to change the number of job types (e.g., 5 types)
JOB_TYPES = ["A", "B", "C"]

LEFT_LABEL_PAD = 5.5
VIEW_WIDTH     = 40.0 

AX_Y_MIN, AX_Y_MAX = 0.0, 2.0
TOP_Y_CENTER  = 1.25              
TOP_LANE_H    = 0.5

BOTTOM_MIN    = 0.0               
BOTTOM_HEIGHT = (AX_Y_MAX - AX_Y_MIN) / 2.0
AMR_Y_CENTERS = [BOTTOM_MIN + (i + 0.5) * (BOTTOM_HEIGHT / AMR_COUNT)
                 for i in range(AMR_COUNT)]
AMR_LANE_H    = BOTTOM_HEIGHT / AMR_COUNT * 0.7

# Dynamic color generation based on JOB_TYPES
_cycle = plt.rcParams.get("axes.prop_cycle", None)
# Ensure we have enough colors; if not, use a default list
_cycle_list = _cycle.by_key()["color"] if _cycle else ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]

# Map each job type to a color (cycling if types > colors)
TYPE_COLORS = {
    jtype: _cycle_list[i % len(_cycle_list)]
    for i, jtype in enumerate(JOB_TYPES)
}

# ------------ Data models ------------
@dataclass
class Job:
    jid: int
    jtype: str
    proc_time: float
    station: str

# ------------ Scheduler state ------------
waiting: List[Job] = []         # FIFO waiting (top lane)
rects_top: List[Rectangle] = [] # top lane artists
texts_top: List            = []

# Per-AMR assigned jobs (visual + cursors)
amr_cursor: Dict[int, float] = {i: LEFT_LABEL_PAD for i in range(1, AMR_COUNT+1)}
amr_rects: Dict[int, List[Rectangle]] = {i: [] for i in range(1, AMR_COUNT+1)}
amr_texts: Dict[int, List]            = {i: [] for i in range(1, AMR_COUNT+1)}
amr_load:  Dict[int, float] = {i: 0.0 for i in range(1, AMR_COUNT+1)}

# Tail progress
_lines_consumed = 0

# ------------ Helpers ------------
def remove_artists(rects, texts):
    for a in rects:
        try: a.remove()
        except Exception: pass
    for t in texts:
        try: t.remove()
        except Exception: pass
    rects.clear(); texts.clear()

def rebuild_top_lane(ax):
    """Lay out waiting FIFO jobs left-aligned, non-overlapping."""
    remove_artists(rects_top, texts_top)
    x = LEFT_LABEL_PAD
    for j in waiting:
        r = Rectangle((x, 1.25),
                      j.proc_time/2.0, TOP_LANE_H,
                      linewidth=1.2, edgecolor="black",
                      facecolor=TYPE_COLORS.get(j.jtype, "gray"),
                      clip_on=True)
        ax.add_patch(r); rects_top.append(r)
        t = ax.text(x + j.proc_time/4.0, 1.5,
                    f"J_{j.jid} ({j.jtype})",
                    ha="center", va="center",
                    fontsize=9, weight="bold",
                    clip_on=True)
        texts_top.append(t)
        x += j.proc_time/2.0

def draw_on_amr(ax, amr_id: int, j: Job):
    """Draw one assigned job on an AMR lane at that lane's cursor."""
    x = amr_cursor[amr_id]
    y = AMR_Y_CENTERS[amr_id-1]
    r = Rectangle((x, y - AMR_LANE_H/2.0),
                  j.proc_time/4.0, AMR_LANE_H,
                  linewidth=1.2, edgecolor="black",
                  facecolor=TYPE_COLORS.get(j.jtype, "gray"),
                  clip_on=True)
    ax.add_patch(r); amr_rects[amr_id].append(r)
    t = ax.text(x + j.proc_time/8.0, y,
                f"J_{j.jid}",
                ha="center", va="center",
                fontsize=9, weight="bold",
                clip_on=True)
    amr_texts[amr_id].append(t)
    amr_cursor[amr_id] += j.proc_time/4.0 

def fifo_dispatch(ax):
    """While we have waiting jobs, pop FIFO and assign to earliest-available AMR."""
    global waiting
    changed = False
    while waiting:
        # Earliest available; tie -> lower id
        amr = min(amr_load, key=lambda k: (amr_load[k], k)) 
        j = waiting.pop(0)
        
        draw_on_amr(ax, amr, j)
        amr_load[amr] += j.proc_time
        emit_assignment(amr, j) 
        changed = True
    return changed

def ingest_new_batches(ax):
    """Read any new JSON lines from INBOX_PATH and enqueue jobs to waiting."""
    global _lines_consumed
    if not os.path.exists(INBOX_PATH):
        return False
    with open(INBOX_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if _lines_consumed >= len(lines):
        return False

    new_lines = lines[_lines_consumed:]
    _lines_consumed = len(lines)

    added = 0
    for ln in new_lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            data = json.loads(ln)
            for j in data.get("jobs", []):
                waiting.append(Job(
                    jid=int(j["jid"]),
                    # Use the first configured type as fallback if "type" missing
                    jtype=str(j.get("type", JOB_TYPES[0])), 
                    proc_time=float(j["proc_time"]),
                    station=str(j["station"]),
                ))
                added += 1
        except Exception as e:
            print(f"[warn] bad JSON line: {e}")
    if added:
        rebuild_top_lane(ax)
    return added > 0

def draw_static_panels(ax):
    band_frac = 0.12 
    # Top panel (Dispatching Queue)
    top_panel = Rectangle((0.0, 0.5), band_frac, 0.5,
                          transform=ax.transAxes, fill=False,
                          linewidth=1.8, clip_on=False, zorder=3)
    ax.add_patch(top_panel)
    tp = ax.text(band_frac*0.5, 0.75, "Dispatching\nQueue",
                 transform=ax.transAxes, ha="center", va="center",
                 fontsize=10, weight="bold", color="gray",
                 clip_on=True, zorder=4)
    tp.set_clip_path(top_panel)

    # Bottom panel (AMRs)
    bot_panel = Rectangle((0.0, 0.0), band_frac, 0.5,
                          transform=ax.transAxes, fill=False,
                          linewidth=1.8, clip_on=False, zorder=3)
    ax.add_patch(bot_panel)

    # Lane labels inside bottom panel
    for i in range(AMR_COUNT):
        y_frac = (i + 0.5) / AMR_COUNT * 0.5 
        txt = ax.text(band_frac*0.5, y_frac, f"AMR {i+1}",
                      transform=ax.transAxes,
                      ha="center", va="center",
                      fontsize=10, weight="bold", color="gray",
                      clip_on=True, zorder=4)
        txt.set_clip_path(bot_panel)

    # Legend for types (Dynamically generated from TYPE_COLORS)
    handles = [Patch(facecolor=TYPE_COLORS[k], edgecolor="black", label=f"Type {k}") for k in TYPE_COLORS]
    ax.legend(handles=handles, loc="upper right", frameon=True)

def update_title(ax):
    # Optional title update (commented out in original, kept consistent)
    # total_wait = sum(j.proc_time for j in waiting)
    # loads = ", ".join(f"{k}:{amr_load[k]:.0f}" for k in sorted(amr_load))
    pass

def main():
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_ylim(AX_Y_MIN, AX_Y_MAX)
    ax.set_xlim(0.0, VIEW_WIDTH)
    ax.set_yticks([]); ax.set_xticks([])
    for side in ('top','right','bottom','left'):
        ax.spines[side].set_visible(True)

    draw_static_panels(ax)
    update_title(ax)

    timer = fig.canvas.new_timer(interval=UPDATE_INTERVAL_MS)

    def tick():
        changed = False
        # 1) Ingest any new batches from Program 1
        if ingest_new_batches(ax):
            changed = True
        # 2) Keep dispatching until waiting is empty
        if fifo_dispatch(ax):
            changed = True
        if changed:
            update_title(ax)
            fig.canvas.draw_idle()
        timer.start()

    timer.add_callback(tick)
    timer.start()
    plt.show()

if __name__ == "__main__":
    main()