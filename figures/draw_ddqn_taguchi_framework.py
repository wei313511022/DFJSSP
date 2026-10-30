"""Framework diagram: DDQN decides WHEN to reschedule, Taguchi-GA decides HOW.

usage:  python draw_ddqn_taguchi_framework.py               -> ddqn_taguchi_framework.{png,svg,pdf}
        python draw_ddqn_taguchi_framework.py --breakdown   -> ddqn_taguchi_framework_breakdown.{png,svg,pdf}
                (adds a machine-breakdown event that forces a Taguchi-GA reschedule, bypassing the DDQN)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})

INK = "#1f2328"
MUTED = "#57606a"
PHASE = {  # tag / frame colour, light fill for the key box
    "I": ("#e0282e", None),
    "II": ("#1f6fe0", "#eaf1fd"),
    "III": ("#22893a", "#e8f5ea"),
    "IV": ("#e8b40c", None),
}
ALERT, ALERT_FILL = "#6b3fa0", "#f1ebfa"   # machine breakdown / forced reschedule (purple: not a phase colour)
FS = 13.5        # box text
LW = 1.6         # box outline


def box(ax, cx, cy, w, h, lines, edge=INK, fill="white", lw=LW, rounding=0.6, fs=FS):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3))
    n = len(lines)
    for i, (txt, kw) in enumerate(lines):
        y = cy + (n - 1) / 2 * 3.0 - i * 3.0
        ax.text(cx, y, txt, ha="center", va="center", fontsize=kw.get("fs", fs), color=kw.get("color", INK),
                fontweight=kw.get("weight", "normal"), style=kw.get("style", "normal"), zorder=4)


def frame(ax, x0, y0, x1, y1, key, tag):
    color = PHASE[key][0]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                linewidth=2.2, edgecolor=color, facecolor="none", zorder=1))
    t = ax.text(x0 + 0.6, y1 + 2.6, tag, ha="left", va="center", fontsize=14.5, color="white",
                fontweight="bold", zorder=5)
    t.set_bbox(dict(boxstyle="round,pad=0.35,rounding_size=0.5", facecolor=color, edgecolor="none"))


def path(ax, pts, color=INK, lw=2.0, ls="-", head=True, ms=16):
    """Polyline through pts; arrow head on the last segment."""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=2)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    if head:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=ms, shrinkA=0, shrinkB=0))
    else:
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=2)


def main(breakdown=False):
    fig = plt.figure(figsize=(10, 12.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 122)
    ax.axis("off")

    # ---- arrivals ----------------------------------------------------------------
    box(ax, 22, 116, 26, 5.2, [(r"New Job Arrival: $\mathbf{j}$", {"weight": "bold"})], rounding=2.6)
    box(ax, 77, 116, 30, 5.2, [(r"New Job Queue: $\mathbf{Q_{new}}$", {"weight": "bold"})], rounding=1.4)
    path(ax, [(35, 116), (62, 116)])

    # ---- I. perception -------------------------------------------------------------
    frame(ax, 3, 92, 97, 106, "I", "I. PERCEPTION PHASE")
    box(ax, 25, 99, 34, 9, [(r"Extract_GNN ($A$, $Q_{sch}$)", {}),
                            (r"AMR State: $\mathbf{S_{amr}}$", {"weight": "bold"})])
    box(ax, 75, 99, 34, 9, [(r"Extract_Jobs ($Q_{new}$)", {}),
                            (r"$\mathbf{Q_{new}}$ State: $\mathbf{S_{new}}$", {"weight": "bold"})])
    path(ax, [(77, 113.4), (77, 103.5)], lw=1.4)

    # ---- II. decision (WHEN) and III. execution (HOW), side by side ------------------
    frame(ax, 3, 46, 45, 82, "II", "II. DECISION PHASE")
    frame(ax, 55, 46, 97, 82, "III", "III. EXECUTION PHASE")
    ax.text(5, 78.6, "When to reschedule?", ha="left", va="center", fontsize=14, style="italic", color=PHASE["II"][0])
    ax.text(57, 78.6, "How to reschedule?", ha="left", va="center", fontsize=14, style="italic", color=PHASE["III"][0])

    # state vectors into the DDQN
    path(ax, [(25, 94.5), (25, 74.6)])
    path(ax, [(75, 94.5), (75, 88.5), (25, 88.5)], head=False)
    ax.text(23.8, 90.3, r"$\mathbf{S_{amr}}$", ha="right", va="center", fontsize=FS)
    ax.text(52, 89.4, r"$\mathbf{S_{new}}$", ha="center", va="bottom", fontsize=FS)

    # DDQN diamond
    cx, cy, hw, hh = 25, 65, 15.5, 9.6
    ax.add_patch(Polygon([(cx, cy + hh), (cx + hw, cy), (cx, cy - hh), (cx - hw, cy)], closed=True,
                         facecolor=PHASE["II"][1], edgecolor=PHASE["II"][0], linewidth=2.4, zorder=3))
    ax.text(cx, cy + 3.0, "iSched", ha="center", va="center", fontsize=FS + 0.5, color=INK, zorder=4)
    ax.text(cx, cy, r"($S_{amr}$, $S_{new}$)", ha="center", va="center", fontsize=FS, color=INK, zorder=4)
    ax.text(cx, cy - 3.2, "Action", ha="center", va="center", fontsize=FS, fontweight="bold", color=INK, zorder=4)

    # Taguchi-GA box
    tx, ty, tw, th = 76, 65, 36, 12.5
    box(ax, tx, ty, tw, th, [(r"Taguchi_GA ($Q_{sch}$, $Q_{new}$, $A$)", {"fs": FS + 0.5}),
                              (r"Scheduled Queue: $\mathbf{Q_{sch}}$", {"weight": "bold"})],
        edge=PHASE["III"][0], fill=PHASE["III"][1], lw=2.4, rounding=1.0)

    # Yes: DDQN triggers Taguchi-GA (the key link)
    path(ax, [(cx + hw, cy), (tx - tw / 2, cy)], lw=3.2, ms=22)
    ax.text(49.5, cy + 1.4, "Yes", ha="center", va="bottom", fontsize=FS + 1, fontweight="bold", color=INK)

    # feedback: Taguchi-GA's gain and computing cost come back to the DDQN as reward
    fy = cy - 4.5
    x_edge = cx + hw * (1 - (cy - fy) / hh)
    path(ax, [(tx - tw / 2, fy), (x_edge + 0.6, fy)], color=MUTED, lw=2.0, ls=(0, (4, 3)))
    ax.text(49.5, fy - 1.1, "Reward", ha="center", va="top", fontsize=FS + 1, fontweight="bold", color=MUTED)

    # No: keep the current plan
    path(ax, [(cx - 6, cy - hh * (1 - 6 / hw) + 0.2), (cx - 6, 54.6)])
    ax.text(cx - 7.2, 56.6, "No", ha="right", va="center", fontsize=FS + 1, fontweight="bold", color=INK)
    box(ax, cx - 8, 50.6, 24, 7, [(r"Maintain $\mathbf{Q_{sch}}$", {"weight": "bold"})])

    # ---- IV. physical ---------------------------------------------------------------
    frame(ax, 3, 13, 97, 34, "IV", "IV. PHYSICAL PHASE")
    out_x = 66 if breakdown else 91          # Taguchi-GA output; leave room for the breakdown arrow
    path(ax, [(cx - 8, 47.1), (cx - 8, 41), (50, 41)], head=False)
    path(ax, [(out_x, ty - th / 2), (out_x, 41), (50, 41)], head=False)
    path(ax, [(50, 41), (50, 31.2)])
    sim_w = 36 if breakdown else 40
    if breakdown:
        ax.text(48.8, 37.8, r"Update $\mathbf{Q_{sch}}$ commands", ha="right", va="center", fontsize=FS, fontweight="bold", color=INK)
    else:
        ax.text(51.2, 37.8, r"Update $\mathbf{Q_{sch}}$ commands", ha="left", va="center", fontsize=FS, fontweight="bold", color=INK)
    box(ax, 50, 23.4, sim_w, 15.6, [(r"SimulationStep ($Q_{sch}$)", {}),
                                     (r"AMRs' Attribute: $\mathbf{A}$", {"weight": "bold"}),
                                     (r"Scheduled Queue: $\mathbf{Q_{sch}}$", {"weight": "bold"}),
                                     (r"Finished Jobs: $\mathbf{FIN}$", {"weight": "bold"})])
    path(ax, [(50, 15.6), (50, 8.2)])
    box(ax, 50, 5.4, 26, 5.2, [(r"Finished Jobs: $\mathbf{FIN}$", {"weight": "bold"})], rounding=2.6)

    if breakdown:
        # machine breakdown: the current plan becomes infeasible -> forced reschedule, bypassing the DDQN
        bx, bw = 84.5, 21
        bh = 6.5
        box(ax, bx, 23.4, bw, bh, [("Machine Breakdown", {"weight": "bold"})],
            edge=ALERT, fill=ALERT_FILL, lw=2.4, rounding=1.0)
        path(ax, [(50 + sim_w / 2, 23.4), (bx - bw / 2, 23.4)], color=ALERT, lw=2.2)
        path(ax, [(bx, 23.4 + bh / 2), (bx, ty - th / 2)], color=ALERT, lw=3.2, ms=22)
        ax.text(bx - 1.6, 52, "Forced reschedule", ha="right", va="center", fontsize=FS + 1, fontweight="bold", color=ALERT)

    # loop back: new AMR attributes and scheduled queue feed the next perception step
    path(ax, [(50 - sim_w / 2, 23.4), (1.2, 23.4), (1.2, 99), (8, 99)])
    t = ax.text(1.2, 60, r"$\mathbf{A}$ & $\mathbf{Q_{sch}}$", ha="center", va="center", fontsize=FS, rotation=90, color=INK)
    t.set_bbox(dict(facecolor="white", edgecolor="none", pad=2))

    here = os.path.dirname(os.path.abspath(__file__))
    name = "ddqn_taguchi_framework_breakdown" if breakdown else "ddqn_taguchi_framework"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(here, f"{name}.{ext}"), dpi=300, facecolor="white")
    print(os.path.join(here, f"{name}.png"))


if __name__ == "__main__":
    import sys
    main(breakdown="--breakdown" in sys.argv[1:])
