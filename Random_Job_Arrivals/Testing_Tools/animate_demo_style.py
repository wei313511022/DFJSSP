"""Recorded version of run_all_demos' layout: iSched (動態, top) vs fixed every 10 s (週期, bottom).

Each row has the scheduling window of Dynamic/Periodic_Pairing_Demo.py (New Arrival Jobs lane +
AMR 3/2/1 queue lanes, blocks sized by processing time, Type A/B/C colours) and the route map of
AMRs_Routing/Routing_Demo.py (stations, material points, obstacles, AMRs, dashed A* routes and the
per-AMR status lines). Built from trace_rescheduling.py --replay --ticks_out files, so it matches the
recorded test episode exactly. Default speed: 500 sim s in 31.25 s = 16x.

usage (from Testing_Tools):
  python animate_demo_style.py ../models_pth/taguchi_trace_r4/replay --episode 3
  python animate_demo_style.py ../models_pth/taguchi_trace_r4/replay --episode 3 --preview 63
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
from matplotlib.patches import Circle, Patch, Rectangle

from plot_trace_comparison import H, cjk_font_family

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Static_alogorithm")))
from GA.GA import AMR_KEYS, AMR_STARTS, OBSTACLES, STATIONS, SUPPLY_LOCATIONS, TYPE_DURATION, shortest_path  # noqa: E402

# ---- same geometry / colours as the original demos ----
LEFT_LABEL_PAD, VIEW_WIDTH = 5.5, 40.0
TOP_Y, TOP_H = 1.25, 0.5
AMR_Y = [(i + 0.5) * (1.0 / 3) for i in range(3)]          # AMR 1 bottom, AMR 3 top
AMR_H = 1.0 / 3 * 0.7
TYPE_COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
ROUTE_COLORS = {"AMR1": "tab:red", "AMR2": "tab:green", "AMR3": "tab:purple"}
GRID_W, GRID_H = 10, 11

ROWS = [  # policy, scheduling-window title, route-map title, overlay label, sub label, frame colour
    ("ddqn", "Dynamic Rescheduling (iSched)", "Dynamic Route Map", "動態", "iSched", "#e01b1b"),
    ("fixed", "Periodic Rescheduling (Period=10.0s)", "Periodic Route Map", "週期", "每 10 秒", "#1e5fd6"),
]


def load(folder, pol, ep):
    ticks = [json.loads(l) for l in open(os.path.join(folder, f"{pol}_ep{ep}_ticks.jsonl"))]
    by_t = {t["t"]: t for t in ticks}
    last = max(by_t)
    # processing time left (ticks until the current processing ends)
    left = {}
    for t in range(last, -1, -1):
        for amr in AMR_KEYS:
            a = by_t[t]["amrs"][amr]
            if a["mode"] in ("processing", "processing_old") and a["job"]:
                nxt = by_t.get(t + 1, {}).get("amrs", {}).get(amr)
                same = nxt and nxt["mode"] == a["mode"] and nxt["job"] and nxt["job"][0] == a["job"][0]
                left[(t, amr)] = (left.get((t + 1, amr), 0) if same else 0) + 1
    rows = [json.loads(l) for l in open(os.path.join(folder, f"{pol}_ep{ep}.jsonl"))][1:]
    events, c = [], 0          # Done counter: same counting as the race video / reported results
    for r in rows:
        if r["t_after"] <= H and r["completed_after"] != c:
            events.append((r["t_after"], r["completed_after"]))
            c = r["completed_after"]
    return by_t, last, left, events


def done_at(events, T):
    c = 0
    for t, n in events:
        if t <= T:
            c = n
    return c


def goal_of(amr, a):
    m, job = a["mode"], a["job"]
    if m == "moving_supply" and job:
        return SUPPLY_LOCATIONS[job[1]]
    if m == "moving_station" and job:
        return STATIONS[job[2]]
    if m == "moving_base":
        return AMR_STARTS[amr]
    return None


def status_line(amr, a, left):
    inv = a["inv"]
    inv_s = f"A{inv.get('A', 0)} B{inv.get('B', 0)} C{inv.get('C', 0)}"
    x, y = a["pos"]
    job_s = f"Job{a['job'][0]}" if a["job"] else "-"
    phase = "supply" if a["mode"] == "moving_supply" else "deliver"
    if a["mode"] in ("processing", "processing_old"):
        return f"{amr}: processing ({left}s left), {job_s}, {inv_s}, (x:{x} y:{y})"
    if a["mode"] in ("moving_supply", "moving_station", "moving_base"):
        return f"{amr}: {a['mode']}, {phase}, {job_s if a['mode'] != 'moving_base' else '-'}, {inv_s}, (x:{x} y:{y})"
    return f"{amr}: {a['mode']}, -, {inv_s}, (x:{x} y:{y})"


class Row:
    def __init__(self, fig, y0, spec, data, serif, stations):
        self.pol, self.title, self.map_title, label, sub, color = spec
        self.stations = stations
        self.by_t, self.last, self.left, self.events = data
        h = 0.5
        fig.patches.append(Rectangle((0.004, y0 + 0.006), 0.992, h - 0.012, transform=fig.transFigure,
                                     fill=False, edgecolor=color, linewidth=7))
        fig.text(0.014, y0 + h - 0.05, label, fontsize=42, color=color, fontfamily=serif, va="top")
        fig.text(0.016, y0 + h - 0.155, sub, fontsize=17, color=color, fontfamily=serif, va="top")
        fig.text(0.014, y0 + 0.035, "16x", fontsize=40, color=color, fontfamily=serif, va="bottom")

        # scheduling window (Dynamic/Periodic_Pairing_Demo.py)
        g = fig.add_axes([0.09, y0 + 0.035, 0.57, 0.36])
        g.set_xlim(0, VIEW_WIDTH); g.set_ylim(0, 2)
        g.set_xticks([]); g.set_yticks([])
        band = 0.12
        for yb in (0.0, 0.5):
            g.add_patch(Rectangle((0, yb), band, 0.5, transform=g.transAxes, fill=False, linewidth=2, clip_on=False, zorder=3))
        g.text(band / 2, 0.75, "New Arrival\nJobs", transform=g.transAxes, ha="center", va="center",
               fontsize=10.5, weight="bold", color="gray")
        for i in range(3):
            g.text(band / 2, (i + 0.5) / 3 * 0.5, f"AMR {i + 1}", transform=g.transAxes, ha="center", va="center",
                   fontsize=12, weight="bold", color="gray")
        g.legend(handles=[Patch(facecolor=TYPE_COLORS[k], edgecolor="black", label=f"Type {k}") for k in "ABC"],
                 loc="upper right", frameon=True, fontsize=11)
        self.g_title = g.set_title("", fontsize=14)
        self.g, self.dyn = g, []

        # route map (Routing_Demo.py)
        # size the axes to the grid's own proportions so cells stay square and AMRs stay round
        mh = 0.315
        mw = mh * fig.get_figheight() / fig.get_figwidth() * GRID_W / GRID_H
        m = fig.add_axes([0.82 - mw / 2, y0 + 0.11, mw, mh])
        m.set_xlim(-0.5, GRID_W - 0.5); m.set_ylim(-0.5, GRID_H - 0.5)
        m.set_aspect("equal")
        m.set_xticks(range(GRID_W)); m.set_yticks(range(GRID_H))
        m.tick_params(labelsize=10)
        m.grid(True, linewidth=0.4, color="black", alpha=0.4)
        for (x, y) in OBSTACLES:
            m.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=(0.8, 0.8, 0.8), edgecolor="none", zorder=1))
        for name, (sx, sy) in STATIONS.items():
            m.add_patch(Rectangle((sx - 0.5, sy - 0.5), 1, 1, facecolor="none", edgecolor="tab:red", linewidth=2, zorder=2))
            m.text(sx, sy, "S" + name.replace("station", ""), fontsize=9, color="tab:red", weight="bold", ha="center", va="center", zorder=3)
        for jt, (mx, my) in SUPPLY_LOCATIONS.items():
            m.add_patch(Rectangle((mx - 0.5, my - 0.5), 1, 1, facecolor="none", edgecolor="tab:blue", linewidth=2, zorder=2))
            m.text(mx, my, f"M{jt}", fontsize=9, color="tab:blue", weight="bold", ha="center", va="center", zorder=3)
        self.marks, self.labels, self.routes = {}, {}, {}
        for amr in AMR_KEYS:
            self.routes[amr], = m.plot([], [], linestyle="--", linewidth=1.8, alpha=0.75, zorder=4, color=ROUTE_COLORS[amr])
            self.marks[amr] = Circle((0, 0), 0.35, facecolor="white", edgecolor="black", linewidth=2, zorder=5)
            m.add_patch(self.marks[amr])
            self.labels[amr] = m.text(0, 0, amr, fontsize=10, ha="center", va="bottom", zorder=6)
        self.m_title = fig.text(0.82, y0 + 0.445, "", ha="center", va="center", fontsize=17, weight="bold")
        self.status = [fig.text(0.82, y0 + 0.066 - k * 0.022, "", ha="center", va="center", fontsize=10.5)
                       for k in range(3)]

    def draw(self, T):
        k = min(int(math.floor(T)), self.last)
        a, b = self.by_t[k], self.by_t[min(k + 1, self.last)]
        frac = T - math.floor(T) if k < self.last else 0.0
        computing = a["computing"]

        # scheduling window
        g = self.g
        for art in self.dyn:
            art.remove()
        self.dyn = []
        x = LEFT_LABEL_PAD
        for jid, jt in a["waiting"]:
            w = TYPE_DURATION[jt] / 2.0
            if x < VIEW_WIDTH:
                self.dyn.append(g.add_patch(Rectangle((x, TOP_Y), w, TOP_H, linewidth=1.2, edgecolor="black",
                                                      facecolor=TYPE_COLORS[jt], clip_on=True)))
                label = f"J_{jid}: S_{self.stations[jid]}" if w >= 4 else f"J_{jid}"   # narrow blocks: id only
                self.dyn.append(g.text(x + w / 2, TOP_Y + TOP_H / 2, label,
                                       ha="center", va="center", fontsize=10 if w >= 4 else 8.5,
                                       weight="bold", clip_on=True))
            x += w
        queued = 0
        for i, amr in enumerate(AMR_KEYS):
            st = a["amrs"][amr]
            cur = st["job"][0] if st["job"] else None
            jobs = ([st["job"]] if st["job"] else []) + [j for j in st["queue"] if j[0] != cur]
            queued += len(jobs)
            x, y = LEFT_LABEL_PAD, AMR_Y[i]
            for jid, jt, _ in jobs:
                w = TYPE_DURATION[jt] / 4.0
                if x < VIEW_WIDTH:
                    self.dyn.append(g.add_patch(Rectangle((x, y - AMR_H / 2), w, AMR_H, linewidth=1.2, edgecolor="black",
                                                          facecolor=TYPE_COLORS[jt], clip_on=True)))
                    self.dyn.append(g.text(x + w / 2, y, f"J_{jid}", ha="center", va="center", fontsize=10,
                                           weight="bold", clip_on=True))
                x += w
        status = "COMPUTING (Taguchi-GA)" if computing else "RUNNING"
        self.g_title.set_text(f"{self.title}\n t={T:.1f}s — {status} | "
                              f"New={len(a['waiting'])}  Active={len(a['waiting']) + queued}  Done={done_at(self.events, T)}")
        self.g_title.set_color("#c62828" if computing else "black")

        # route map
        for amr in AMR_KEYS:
            (x0, y0), (x1, y1) = a["amrs"][amr]["pos"], b["amrs"][amr]["pos"]
            x, y = x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac
            self.marks[amr].center = (x, y)
            self.labels[amr].set_position((x, y + 0.45))
            goal = goal_of(amr, a["amrs"][amr])
            if goal and tuple(a["amrs"][amr]["pos"]) != tuple(goal):
                path = shortest_path(tuple(a["amrs"][amr]["pos"]), tuple(goal))
                self.routes[amr].set_data([p[0] for p in path], [p[1] for p in path])
            else:
                self.routes[amr].set_data([], [])
        self.m_title.set_text(f"{self.map_title}  —  Sim Time: {T:.0f}s")
        for line, amr in zip(self.status, AMR_KEYS):
            line.set_text(status_line(amr, a["amrs"][amr], self.left.get((k, amr), 0)))


def build(folder, ep, dataset):
    episode = [json.loads(l) for l in open(dataset)][ep]
    stations = {j["id"]: j["dest_station_id"] for j in episode["jobs"]}
    serif = cjk_font_family("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", "Noto Serif CJK TC")
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    rows = [Row(fig, y0, spec, load(folder, spec[0], ep), serif, stations) for y0, spec in ((0.5, ROWS[0]), (0.0, ROWS[1]))]

    def draw(T):
        for r in rows:
            r.draw(T)
    return fig, draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--episode", type=int, default=3)
    ap.add_argument("--dataset", default="../../test_case/dynamic/test_dataset_r4.jsonl")
    ap.add_argument("--seconds", type=float, default=H / 16.0, help="video length of the 500 s episode (default 16x)")
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--preview", type=float, default=None)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--ffmpeg", default=os.path.expanduser("~/.cache/dfjssp_tools/imageio_ffmpeg/binaries"))
    a = ap.parse_args()
    base = os.path.join(a.folder, f"demo_episode{a.episode}")

    if a.preview is not None:
        fig, draw = build(a.folder, a.episode, a.dataset)
        draw(a.preview)
        fig.savefig(f"{base}_preview.png", dpi=120, facecolor="white")
        print(f"{base}_preview.png")
        return

    exe = a.ffmpeg
    if os.path.isdir(exe):
        found = sorted(glob.glob(os.path.join(exe, "ffmpeg*")))
        exe = found[0] if found else "ffmpeg"
    plt.rcParams["animation.ffmpeg_path"] = exe

    def render(out, writer, fps, dpi):
        fig, draw = build(a.folder, a.episode, a.dataset)
        n_run, n_hold = int(round(a.seconds * fps)), int(a.hold * fps)
        times = [H * i / (n_run - 1) for i in range(n_run)] + [H] * n_hold
        FuncAnimation(fig, lambda i: draw(times[i]), frames=len(times)).save(
            out, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": "white"})
        plt.close(fig)
        print(out)

    render(f"{base}.mp4", FFMpegWriter(fps=25, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]), 25, 120)
    if not a.no_gif:
        render(f"{base}.gif", PillowWriter(fps=10), 10, 60)


if __name__ == "__main__":
    sys.exit(main())
