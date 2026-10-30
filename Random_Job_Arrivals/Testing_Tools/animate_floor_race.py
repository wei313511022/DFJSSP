"""Floor-view companion to animate_trace_race.py (Traditional Chinese).

Left: iSched, right: fixed every 10 s. Each half shows the AMRs moving on the floor map,
each AMR's current job and task queue, and the jobs still waiting to be assigned; while
Taguchi-GA is computing the map is covered by a "計算中" overlay (the AMRs only finish
their current job). Same timeline as animate_trace_race.py (500 sim s over --seconds,
then --hold), so the two videos can be played side by side in sync.

Needs per-tick files from trace_rescheduling.py --ticks_out (best with --replay so they
match the recorded trace exactly):
  python animate_floor_race.py ../models_pth/taguchi_trace_r4/replay --episode 3
  python animate_floor_race.py ../models_pth/taguchi_trace_r4/replay --episode 3 --preview 250
"""
import argparse
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from plot_trace_comparison import GRID, H, POL, SURF, TEXT, TEXT2, cjk_font_family

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Static_alogorithm")))
from GA.GA import AMR_KEYS, AMR_STARTS, OBSTACLES, STATIONS, SUPPLY_LOCATIONS  # noqa: E402

NAMES = {"ddqn": "iSched", "fixed": "固定每 10 秒"}
PCOLOR = {pol: c for pol, _, c in POL}
AMR_COLOR = {"AMR1": "#008300", "AMR2": "#6250d6", "AMR3": "#c2185b"}      # never the policy blue/orange
JOB_FILL = {"A": "#eceae4", "B": "#c9c7bd", "C": "#8a8984"}
JOB_INK = {"A": TEXT, "B": TEXT, "C": "white"}
OBSTACLE = "#cfcdc6"


def load_ticks(folder, pol, ep):
    ticks = [json.loads(l) for l in open(os.path.join(folder, f"{pol}_ep{ep}_ticks.jsonl"))]
    return {t["t"]: t for t in ticks}, max(t["t"] for t in ticks)


def load_steps(folder, pol, ep):
    """Completed-job events (step end time, count) - the same counting as animate_trace_race.py
    and all reported results (a job counts when the env step that finishes it ends)."""
    rows = [json.loads(l) for l in open(os.path.join(folder, f"{pol}_ep{ep}.jsonl"))][1:]
    events, last = [], 0
    for r in rows:
        if r["t_after"] <= H and r["completed_after"] != last:
            events.append((r["t_after"], r["completed_after"]))
            last = r["completed_after"]
    return events


def completed_at(events, T):
    c = 0
    for t, n in events:
        if t <= T:
            c = n
    return c


def at(ticks, last, T):
    """Snapshot at integer tick floor(T) plus the next one (for smooth motion)."""
    k = min(int(math.floor(T)), last)
    return ticks[k], ticks[min(k + 1, last)], T - math.floor(T)


def station_label(name):
    return "S" + name.replace("station", "")


def mode_text(amr, computing):
    m, job = amr["mode"], amr["job"]
    if m == "moving_supply" and job:
        return f"前往物料 {job[1]}"
    if m == "moving_station" and job:
        return f"前往 {station_label(job[2])}"
    if m in ("processing", "processing_old") and job:
        return f"加工中 {station_label(job[2])}"
    if m == "moving_base":
        return "返回基地"
    return "等待排程" if computing else "閒置"


def chip(ax, x, y, w, h, jtype, edge=None, lw=0.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=JOB_FILL[jtype], edgecolor=edge or "none", linewidth=lw,
                                transform=ax.transAxes, clip_on=False))
    ax.text(x + w / 2, y + h / 2, jtype, ha="center", va="center", fontsize=14, color=JOB_INK[jtype],
            transform=ax.transAxes, fontweight="bold")


class Half:
    def __init__(self, fig, pol, x0, ticks, last, events):
        self.pol, self.ticks, self.last, self.events = pol, ticks, last, events
        color = PCOLOR[pol]
        fig.patches.append(Rectangle((x0 + 0.025, 0.858), 0.013, 0.024, transform=fig.transFigure,
                                     facecolor=color, edgecolor="none"))
        fig.text(x0 + 0.045, 0.87, NAMES[pol], fontsize=24, color=TEXT, va="center")
        self.done = fig.text(x0 + 0.205, 0.87, "", fontsize=24, color=TEXT, va="center", fontweight="bold")
        self.status = fig.text(x0 + 0.33, 0.87, "", fontsize=17, va="center")

        # floor map
        m = fig.add_axes([x0 + 0.012, 0.215, 0.305, 0.60])
        m.set_xlim(-0.5, 9.5); m.set_ylim(0.5, 9.5); m.set_aspect("equal")
        m.set_xticks([]); m.set_yticks([])
        for sp in m.spines.values():
            sp.set_color(GRID)
        for x in range(11):
            m.axvline(x - 0.5, color=GRID, lw=0.8, zorder=0)
        for y in range(1, 11):
            m.axhline(y - 0.5, color=GRID, lw=0.8, zorder=0)
        for (x, y) in OBSTACLES:
            m.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=OBSTACLE, edgecolor="none", zorder=1))
        for name, (x, y) in STATIONS.items():
            m.add_patch(Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, facecolor="none", edgecolor=TEXT, lw=2, zorder=2))
            m.text(x, y, station_label(name), ha="center", va="center", fontsize=13, color=TEXT, zorder=2)
        for mat, (x, y) in SUPPLY_LOCATIONS.items():
            m.add_patch(Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, facecolor=JOB_FILL[mat], edgecolor=TEXT2, lw=1.6, zorder=2))
            m.text(x, y, mat, ha="center", va="center", fontsize=14, color=JOB_INK[mat], fontweight="bold", zorder=2)
        for amr, (x, y) in AMR_STARTS.items():
            m.add_patch(Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, facecolor="none", edgecolor=AMR_COLOR[amr],
                                  lw=1.2, ls=(0, (3, 2)), zorder=1))
        self.dots, self.nums = {}, {}
        for amr in AMR_KEYS:
            self.dots[amr] = Circle((0, 0), 0.38, facecolor=AMR_COLOR[amr], edgecolor="white", lw=2, zorder=5)
            m.add_patch(self.dots[amr])
            self.nums[amr] = m.text(0, 0, amr[-1], ha="center", va="center", fontsize=14, color="white",
                                    fontweight="bold", zorder=6)
        self.veil = Rectangle((-0.5, 0.5), 10, 9, facecolor=SURF, alpha=0.55, edgecolor="none", zorder=7)
        m.add_patch(self.veil)
        self.veil_text = m.text(4.5, 6.9, "Taguchi-GA 計算中…\nAMR 完成手上工作後等待", ha="center", va="center",
                                fontsize=19, color=color, fontweight="bold", zorder=8, linespacing=1.5)
        m.text(4.5, 0.05, "左：物料補給點 A/B/C　右：工作站 S1–S5　虛框：AMR 基地", ha="center", va="top",
               fontsize=12.5, color=TEXT2)
        self.map = m

        self.q = fig.add_axes([x0 + 0.325, 0.25, 0.165, 0.56]); self.q.axis("off")
        self.w = fig.add_axes([x0 + 0.02, 0.03, 0.465, 0.13]); self.w.axis("off")

    def draw(self, T):
        a, b, frac = at(self.ticks, self.last, T)
        computing = a["computing"]
        self.done.set_text(f"已完成 {completed_at(self.events, T)} 件")
        self.status.set_text("● 計算中，AMR 等待" if computing else "● AMR 作業中")
        self.status.set_color(PCOLOR[self.pol] if computing else TEXT2)
        for amr in AMR_KEYS:
            (x0, y0), (x1, y1) = a["amrs"][amr]["pos"], b["amrs"][amr]["pos"]
            x, y = x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac
            self.dots[amr].center = (x, y)
            self.nums[amr].set_position((x, y))
        self.veil.set_visible(computing)
        self.veil_text.set_visible(computing)

        q = self.q
        q.cla(); q.axis("off"); q.set_xlim(0, 1); q.set_ylim(0, 1)
        q.text(0, 0.985, "AMR 任務佇列", fontsize=17, color=TEXT, va="top")
        cw, ch, gap = 0.135, 0.065, 0.025
        for i, amr in enumerate(AMR_KEYS):
            st = a["amrs"][amr]
            y = 0.87 - i * 0.29
            q.add_patch(Circle((0.035, y), 0.03, facecolor=AMR_COLOR[amr], edgecolor="none", transform=q.transAxes))
            q.text(0.035, y, amr[-1], ha="center", va="center", fontsize=11, color="white", fontweight="bold")
            q.text(0.09, y, f"{amr}  {mode_text(st, computing)}", fontsize=14.5, color=TEXT, va="center")
            cur = st["job"][0] if st["job"] else None
            jobs = ([st["job"]] if st["job"] else []) + [j for j in st["queue"] if j[0] != cur]
            per_row, rows = 6, 2
            for k, j in enumerate(jobs[: per_row * rows]):
                r, c = divmod(k, per_row)
                cx, cy = c * (cw + gap) + 0.0, y - 0.11 - r * (ch + 0.025)
                chip(q, cx, cy, cw, ch, j[1], edge=AMR_COLOR[amr] if k == 0 and cur is not None else None,
                     lw=3 if k == 0 and cur is not None else 0)
            if len(jobs) > per_row * rows:
                q.text(1.0, y - 0.11 - (rows - 1) * (ch + 0.025) + ch / 2, f"+{len(jobs) - per_row * rows}",
                       fontsize=13, color=TEXT2, va="center", ha="right")
            if not jobs:
                q.text(0.0, y - 0.11 + ch / 2, "（無工件）", fontsize=13, color=TEXT2, va="center")

        w = self.w
        w.cla(); w.axis("off"); w.set_xlim(0, 1); w.set_ylim(0, 1)
        waiting = a["waiting"]
        w.text(0, 0.95, f"待分配工件：{len(waiting)} 件", fontsize=17, color=TEXT, va="top")
        cw2, ch2, gap2, n_max = 0.034, 0.30, 0.008, 22
        for k, j in enumerate(waiting[:n_max]):
            chip(w, k * (cw2 + gap2), 0.18, cw2, ch2, j[1])
        if len(waiting) > n_max:
            w.text(n_max * (cw2 + gap2) + 0.005, 0.18 + ch2 / 2, f"+{len(waiting) - n_max}", fontsize=14,
                   color=TEXT2, va="center")


def build(folder, ep):
    plt.rcParams.update({"font.family": cjk_font_family(), "axes.unicode_minus": False})
    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    fig.text(0.025, 0.955, "AMR 作業與工件分配：iSched vs. 固定每 10 秒（Taguchi-GA）", fontsize=26, color=TEXT, va="center")
    clock = fig.text(0.975, 0.955, "", fontsize=24, color=TEXT, va="center", ha="right", fontweight="bold")
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.05, 0.9], transform=fig.transFigure, color=GRID, lw=1.5))
    halves = [Half(fig, pol, x0, *load_ticks(folder, pol, ep), load_steps(folder, pol, ep))
              for pol, x0 in (("ddqn", 0.0), ("fixed", 0.5))]

    def draw(T):
        clock.set_text(f"模擬時間 {T:.0f} 秒")
        for h in halves:
            h.draw(T)

    return fig, draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder with <policy>_ep<N>_ticks.jsonl")
    ap.add_argument("--episode", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--preview", type=float, default=None)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--ffmpeg", default=os.path.expanduser("~/.cache/dfjssp_tools/imageio_ffmpeg/binaries"))
    a = ap.parse_args()
    base = os.path.join(a.folder, f"floor_episode{a.episode}_zh")

    if a.preview is not None:
        fig, draw = build(a.folder, a.episode)
        draw(a.preview)
        fig.savefig(f"{base}_preview.png", dpi=120, facecolor=SURF)
        print(f"{base}_preview.png")
        return

    exe = a.ffmpeg
    if os.path.isdir(exe):
        found = sorted(glob.glob(os.path.join(exe, "ffmpeg*")))
        exe = found[0] if found else "ffmpeg"
    plt.rcParams["animation.ffmpeg_path"] = exe

    def render(out, writer, fps, dpi):
        fig, draw = build(a.folder, a.episode)
        n_run, n_hold = int(a.seconds * fps), int(a.hold * fps)
        times = [H * i / (n_run - 1) for i in range(n_run)] + [H] * n_hold   # same timeline as the race video
        FuncAnimation(fig, lambda i: draw(times[i]), frames=len(times)).save(
            out, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": SURF})
        plt.close(fig)
        print(out)

    render(f"{base}.mp4", FFMpegWriter(fps=25, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]), 25, 120)
    if not a.no_gif:
        render(f"{base}.gif", PillowWriter(fps=10), 10, 60)


if __name__ == "__main__":
    sys.exit(main())
