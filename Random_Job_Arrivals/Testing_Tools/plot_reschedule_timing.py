"""Plot when GNN_DDQN chose to reschedule, from test_performance.py logs.

usage (from Testing_Tools):
  python plot_reschedule_timing.py ../models_pth/gnn_ddqn_eval_r4 ../../test_case/dynamic/test_dataset_r4.jsonl
Writes reschedule_timeline.png and reschedule_intervals.png into the eval folder.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = [("demo", "Trained on demo (1 episode)", "#2a78d6"),
          ("r4", "Trained on r4 (1000 episodes)", "#eb6834")]
ARRIVAL = "#a3a29c"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
FIXED_PERIOD = 10.0

DECIDE = re.compile(r"\[AI-DECIDE\] t=\s*([\d.]+).*?unstarted=(\d+)")
AFTER = re.compile(r"\[AI-AFTER \] t=\s*([\d.]+).*?GA=\s*([\d.]+) ms")
ROW = re.compile(r"^(\d+)\s+\|")


def parse_logs(folder, tag):
    out = defaultdict(list)
    for path in glob.glob(os.path.join(folder, f"{tag}_ep*.log")):
        buf, pending = [], None
        for line in open(path):
            if m := DECIDE.search(line):
                pending = [float(m.group(1)), int(m.group(2)), 0.0]
            elif (m := AFTER.search(line)) and pending is not None:
                pending[2] = float(m.group(2)) / 1000.0
                buf.append(tuple(pending)); pending = None
            elif m := ROW.match(line):        # summary row = that episode (AI then Fix) is finished
                out[int(m.group(1))] = buf; buf = []
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    folder, dataset = sys.argv[1], sys.argv[2]
    episodes = [json.loads(l) for l in open(dataset)]
    data = {tag: parse_logs(folder, tag) for tag, _, _ in MODELS}
    eps = sorted(set(data["demo"]) & set(data["r4"]))
    sim_end = 500

    # ---- Figure 1: timeline per test episode ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    for ax, (tag, label, color) in zip(axes, MODELS):
        for row, ep in enumerate(eps):
            arr = [j["arrival_time"] for j in episodes[ep]["jobs"]]
            ax.scatter(arr, [row - 0.22] * len(arr), marker="|", s=40, color=ARRIVAL, linewidths=1, zorder=2)
            dec = data[tag][ep]
            if dec:
                t = [d[0] for d in dec]
                ax.scatter(t, [row + 0.12] * len(t), s=30, color=color, edgecolors="white", linewidths=0.8, zorder=3)
            ax.text(sim_end + 6, row + 0.12, f"{len(dec)}", va="center", fontsize=8.5, color=TEXT2)
        ax.set_title(label, loc="left", fontsize=11, color=TEXT, pad=8)
        ax.set_xlim(0, sim_end + 22)
        ax.set_xlabel("Simulation time (s)", color=TEXT2, fontsize=9.5)
        style(ax)
    axes[0].set_yticks(range(len(eps)), [f"ep {e}" for e in eps])
    axes[0].set_ylim(len(eps) - 0.4, -0.6)
    axes[1].text(sim_end + 6, -0.55, "count", fontsize=8, color=TEXT2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=MODELS[0][2], markeredgecolor="white", markersize=6, label="DDQN reschedule (demo-trained)"),
               plt.Line2D([], [], marker="o", ls="", color=MODELS[1][2], markeredgecolor="white", markersize=6, label="DDQN reschedule (r4-trained)"),
               plt.Line2D([], [], marker="|", ls="", color=ARRIVAL, markersize=9, label="Job arrival")]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=TEXT2)
    fig.suptitle("When GNN_DDQN chose to reschedule - test_dataset_r4, episodes 0-9", x=0.01, ha="left",
                 fontsize=13, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p1 = os.path.join(folder, "reschedule_timeline.png")
    fig.savefig(p1, dpi=150, facecolor="#fcfcfb"); plt.close(fig)

    # ---- Figure 2: gaps between reschedules, backlog at decision --------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    for r, (tag, label, color) in enumerate(MODELS):
        gaps, backlog = [], []
        for ep in eps:
            t = [d[0] for d in data[tag][ep]]
            gaps += list(np.diff(t))
            backlog += [d[1] for d in data[tag][ep]]
        ax = axes[r][0]
        ax.hist(gaps, bins=np.arange(0, 62, 2), color=color, edgecolor="white", linewidth=1)
        ax.axvline(FIXED_PERIOD, color=TEXT2, ls="--", lw=1.2)
        ax.text(FIXED_PERIOD + 0.8, 0.95, "fixed policy: every 10 s", fontsize=8.5, color=TEXT2,
                transform=ax.get_xaxis_transform(), va="top",
                bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1.5), zorder=4)
        med = float(np.median(gaps))
        ax.set_title(f"{label} - time between reschedules (median {med:.0f} s, n={len(gaps)})", loc="left", fontsize=10, color=TEXT)
        ax.set_ylabel("Count", color=TEXT2, fontsize=9)
        style(ax)
        ax = axes[r][1]
        ax.hist(backlog, bins=np.arange(0, max(backlog) + 3, 1), color=color, edgecolor="white", linewidth=1)
        ax.set_title(f"Unstarted jobs when it rescheduled (median {np.median(backlog):.0f})", loc="left", fontsize=10, color=TEXT)
        style(ax)
    axes[1][0].set_xlabel("Seconds since previous reschedule (simulation time; gaps > 60 s not shown)", color=TEXT2, fontsize=9)
    axes[1][1].set_xlabel("Unstarted jobs at the decision", color=TEXT2, fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(folder, "reschedule_intervals.png")
    fig.savefig(p2, dpi=150, facecolor="#fcfcfb"); plt.close(fig)

    for tag, label, _ in MODELS:
        all_gaps = [g for ep in eps for g in np.diff([d[0] for d in data[tag][ep]])]
        n = sum(len(data[tag][ep]) for ep in eps)
        comp = [d[2] for ep in eps for d in data[tag][ep]]
        print(f"{label}: {n} reschedules in {len(eps)} episodes ({n/len(eps):.1f}/episode), "
              f"median gap {np.median(all_gaps):.1f} s, gaps > 60 s: {sum(g > 60 for g in all_gaps)}, "
              f"median compute {np.median(comp):.2f} s")
    print(p1); print(p2)


if __name__ == "__main__":
    main()
