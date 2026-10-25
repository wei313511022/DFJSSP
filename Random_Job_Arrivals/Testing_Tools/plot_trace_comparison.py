"""Plot DDQN vs fixed-period rescheduling from trace_rescheduling.py output.

usage (from Testing_Tools):
  python plot_trace_comparison.py ../models_pth/taguchi_trace_r4 --title "Taguchi-GA scheduler" [--episode N]
Writes trace_episode<N>.png and trace_summary.png into the trace folder.
"""
import argparse
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POL = [("ddqn", "DDQN (r4-trained)", "#2a78d6"), ("fixed", "Fixed every 10 s", "#eb6834")]
TEXT, TEXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
H = 500.0   # episode horizon (SIM_TIME); the last step may overshoot it, so everything is clipped here


def lost_time(rows):
    return sum(min(r["t_after"], H) - r["t"] for r in rows if r["action"] and r["t"] < H)


def load(folder):
    data = {}
    for p in glob.glob(os.path.join(folder, "*_ep*.jsonl")):
        lines = [json.loads(l) for l in open(p)]
        s = lines[0]["summary"]
        data[(s["policy"], s["episode"])] = (s, lines[1:])
    return data


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def episode_figure(data, ep, title, out):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.6, 1.6]})
    # (a) reschedules as bars spanning the simulation time they consumed
    ax = axes[0]
    ylabels = []
    for lane, (pol, label, color) in enumerate(POL):
        s, rows = data[(pol, ep)]
        bars = [(r["t"], min(r["t_after"], H) - r["t"]) for r in rows if r["action"] and r["t"] < H]
        ax.broken_barh(bars, (1 - lane - 0.3, 0.6), facecolors=color, edgecolors=SURF, linewidth=0.6)
        ylabels.append(f"{label}\n{len(bars)} reschedules, {lost_time(rows):.0f} s computing")
    ax.set_yticks([1, 0], ylabels)
    ax.set_ylim(-0.6, 1.6)
    ax.set_title("(a) When it rescheduled - bar length = simulation time spent computing (AMR queues frozen)",
                 loc="left", fontsize=10, color=TEXT)
    style(ax)
    # (b) completed jobs, (c) jobs waiting to start
    for ax, key, sub in ((axes[1], "completed_after", "(b) Jobs completed"),
                         (axes[2], "unstarted", "(c) Jobs waiting to start (arrived, not started)")):
        for pol, label, color in POL:
            s, rows = data[(pol, ep)]
            pts = [((r["t_after"] if key == "completed_after" else r["t"]), r[key]) for r in rows]
            pts = [(0.0, 0)] + [(x, y) for x, y in pts if x <= H]
            pts.append((H, pts[-1][1]))
            xs, ys = zip(*pts)
            ax.step(xs, ys, where="post", color=color, linewidth=2, label=label)
            ax.text(H + 3, ys[-1], f"{ys[-1]}", va="center", fontsize=9, color=TEXT2)
        if key == "unstarted":
            s, rows = data[("fixed", ep)]
            ax.text(3, ax.get_ylim()[1] * 0.92, "jumps = jobs that arrived during a long computation become visible when it ends",
                    fontsize=8.5, color=TEXT2)
        ax.set_title(sub, loc="left", fontsize=10, color=TEXT)
        ax.set_ylabel("Jobs", color=TEXT2, fontsize=9)
        style(ax, "both")
    axes[2].set_xlabel("Simulation time (s)", color=TEXT2, fontsize=9.5)
    axes[2].set_xlim(0, H + 25)
    handles = [plt.Line2D([], [], color=c, linewidth=3, label=l) for _, l, c in POL]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=TEXT2)
    tot = data[("fixed", ep)][0]["total_jobs"]
    fig.suptitle(f"{title}: DDQN vs fixed 10 s - test episode {ep} ({tot} jobs)", x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)


def summary_figure(data, eps, title, out):
    metrics = [
        ("Reschedules started before t = 500 s", lambda s, rows: sum(1 for r in rows if r["action"] and r["t"] < H), "{:.1f}"),
        ("Simulation time spent computing, within 500 s (s)", lambda s, rows: lost_time(rows), "{:.0f}"),
        ("Jobs done by t = 500 s (%)", lambda s, rows: s["done_pct"], "{:.1f}"),
        ("Median reschedule size (jobs re-planned)", lambda s, rows: st.median([r["unstarted"] for r in rows if r["action"]] or [0]), "{:.0f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (name, fn, fmt) in zip(axes.flat, metrics):
        vals = {pol: [fn(*data[(pol, e)]) for e in eps] for pol, _, _ in POL}
        for i, e in enumerate(eps):
            ax.plot([i, i], [vals["ddqn"][i], vals["fixed"][i]], color=GRID, linewidth=2, zorder=1)
        for pol, label, color in POL:
            ax.scatter(range(len(eps)), vals[pol], s=46, color=color, edgecolors=SURF, linewidths=1.2, zorder=3, label=label)
        means = {pol: st.mean(vals[pol]) for pol, _, _ in POL}
        ax.set_title(f"{name}\nmean: DDQN {fmt.format(means['ddqn'])}  |  fixed {fmt.format(means['fixed'])}",
                     loc="left", fontsize=10, color=TEXT)
        ax.set_xticks(range(len(eps)), [f"ep {e}" for e in eps], fontsize=8)
        style(ax, "y")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=7, label=l) for _, l, c in POL]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=TEXT2)
    fig.suptitle(f"{title}: DDQN vs fixed 10 s - 10 test episodes (test_dataset_r4)", x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    return {name: {pol: st.mean(fn(*data[(pol, e)]) for e in eps) for pol, _, _ in POL} for name, fn, _ in metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--title", default="Taguchi-GA scheduler")
    ap.add_argument("--episode", type=int, default=None, help="episode for the detail figure (default: median gap)")
    a = ap.parse_args()
    data = load(a.folder)
    eps = sorted(e for (p, e) in data if p == "ddqn" and ("fixed", e) in data)
    if a.episode is None:   # episode whose jobs-done gap is the median one: a typical case, not the best
        gaps = sorted(eps, key=lambda e: data[("ddqn", e)][0]["done_pct"] - data[("fixed", e)][0]["done_pct"])
        a.episode = gaps[len(gaps) // 2]
    p1 = os.path.join(a.folder, f"trace_episode{a.episode}.png")
    episode_figure(data, a.episode, a.title, p1)
    p2 = os.path.join(a.folder, "trace_summary.png")
    means = summary_figure(data, eps, a.title, p2)
    for k, v in means.items():
        print(f"{k}: DDQN {v['ddqn']:.1f} | fixed {v['fixed']:.1f}")
    print(p1); print(p2)


if __name__ == "__main__":
    main()
