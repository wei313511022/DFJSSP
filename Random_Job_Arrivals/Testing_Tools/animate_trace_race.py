"""Race animation (Traditional Chinese): iSched vs fixed-period rescheduling, from trace_rescheduling.py output.

The simulated 500 s play out over ~30 s of video: panel (a) reschedule bars grow while the
scheduler computes, panel (b) completed-job curves climb, and a scoreboard shows completed
jobs, whether the AMRs are waiting for a plan, and the total waiting time. The final frame is held.

usage (from Testing_Tools):
  python animate_trace_race.py ../models_pth/taguchi_trace_r4 --episode 3              # MP4 + GIF
  python animate_trace_race.py ../models_pth/taguchi_trace_r4 --episode 3 --preview 250  # one PNG frame
MP4 needs an ffmpeg with libx264; by default the one bundled by imageio-ffmpeg in
~/.cache/dfjssp_tools is used (pip install --target ~/.cache/dfjssp_tools imageio-ffmpeg).
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from plot_trace_comparison import GRID, H, POL, SURF, TEXT, TEXT2, cjk_font_family, load, style

NAMES = {"ddqn": "iSched", "fixed": "固定每 10 秒"}
COLOR = {pol: c for pol, _, c in POL}


def series(rows):
    """Bars (start, end) clipped to H, and completed-job step events (t, count)."""
    bars = [(r["t"], min(r["t_after"], H)) for r in rows if r["action"] and r["t"] < H]
    events, last = [], 0
    for r in rows:
        if r["t_after"] <= H and r["completed_after"] != last:
            events.append((r["t_after"], r["completed_after"]))
            last = r["completed_after"]
    return bars, events


def completed_at(events, T):
    c = 0
    for t, n in events:
        if t <= T:
            c = n
    return c


def step_path(events, T):
    xs, ys, cur = [0.0], [0], 0
    for t, n in events:
        if t > T:
            break
        xs += [t, t]; ys += [cur, n]; cur = n
    xs.append(T); ys.append(cur)
    return xs, ys


def build(data, ep, size=(16, 9)):
    plt.rcParams.update({"font.family": cjk_font_family(), "axes.unicode_minus": False})
    fig = plt.figure(figsize=size, facecolor=SURF)
    tot = data[("fixed", ep)][0]["total_jobs"]
    fig.text(0.03, 0.945, "Taguchi-GA 重排程：iSched vs. 固定每 10 秒", fontsize=30, color=TEXT, va="center")
    fig.text(0.03, 0.885, f"測試回合 {ep}｜{tot} 個工件｜模擬時長 {H:.0f} 秒", fontsize=19, color=TEXT2, va="center")

    ax_a = fig.add_axes([0.14, 0.59, 0.57, 0.21])
    ax_b = fig.add_axes([0.14, 0.175, 0.57, 0.30])
    S = {}
    # (a) reschedule bars, one Rectangle per reschedule; width grows with time
    for lane, (pol, _, color) in enumerate(POL):
        bars, events = series(data[(pol, ep)][1])
        rects = []
        for s, e in bars:
            r = Rectangle((s, 1 - lane - 0.32), 0, 0.64, facecolor=color, edgecolor=SURF, linewidth=0.8)
            ax_a.add_patch(r)
            rects.append((r, s, e))
        S[pol] = {"bars": bars, "events": events, "rects": rects}
    ax_a.set_xlim(0, H); ax_a.set_ylim(-0.6, 1.6)
    ax_a.set_yticks([1, 0], [NAMES["ddqn"], NAMES["fixed"]])
    style(ax_a)
    ax_a.tick_params(axis="y", length=0, labelsize=22, labelcolor=TEXT)
    ax_a.tick_params(axis="x", labelsize=17)
    ax_a.set_xlabel("模擬時間（秒）", color=TEXT2, fontsize=18)
    ax_a.set_title("(a) 重排程時機（長條＝計算時間）", loc="left", fontsize=22, color=TEXT, pad=10)

    # (b) completed jobs
    final = {pol: completed_at(S[pol]["events"], H) for pol in S}
    for pol, _, color in POL:
        S[pol]["line"], = ax_b.plot([], [], color=color, linewidth=3.4, label=NAMES[pol], solid_capstyle="butt")
        S[pol]["head"] = ax_b.text(0, 0, "", fontsize=18, color=TEXT, va="center", ha="left", clip_on=False)
    ax_b.set_xlim(0, H); ax_b.set_ylim(0, max(final.values()) * 1.12 + 1)
    style(ax_b, "both")
    ax_b.tick_params(labelsize=17)
    ax_b.set_xlabel("模擬時間（秒）", color=TEXT2, fontsize=18)
    ax_b.set_ylabel("工件數", color=TEXT2, fontsize=18)
    ax_b.set_title("(b) 累計完成工件數", loc="left", fontsize=22, color=TEXT, pad=10)
    ax_b.legend(loc="upper left", frameon=False, fontsize=19, labelcolor=TEXT2, handlelength=1.6)
    cursors = [ax.axvline(0, color=TEXT2, lw=1.4, alpha=0.6, zorder=0) for ax in (ax_a, ax_b)]

    # scoreboard: clock centred on panel (a), counts level with panel (b);
    # name and count share one row, at the same size as the panel titles
    x0, x1 = 0.765, 0.97
    ya = 0.59 + 0.21 / 2
    fig.text(x0, ya, "模擬時間", fontsize=18, color=TEXT2, va="center")
    clock = fig.text(x1, ya, "", fontsize=22, color=TEXT, fontweight="bold", ha="right", va="center")
    for i, (pol, _, color) in enumerate(POL):
        y = 0.415 - i * 0.15
        fig.patches.append(Rectangle((x0, y - 0.012), 0.014, 0.025, transform=fig.transFigure, facecolor=color, edgecolor="none"))
        fig.text(x0 + 0.022, y, NAMES[pol], fontsize=22, color=TEXT, va="center")
        S[pol]["count"] = fig.text(x1, y, "", fontsize=22, color=TEXT, fontweight="bold", ha="right", va="center")
        S[pol]["status"] = fig.text(x0, y - 0.05, "", fontsize=17, color=TEXT2, va="center")
        S[pol]["wait"] = fig.text(x0, y - 0.09, "", fontsize=17, color=TEXT2, va="center")
    banner = fig.text(0.5, 0.017, "", fontsize=22, color=TEXT, fontweight="bold", ha="center", va="bottom")

    def draw(T, show_banner=False):
        for pol in S:
            for r, s, e in S[pol]["rects"]:
                w = max(0.0, min(e, T) - s)
                r.set_width(w)
                r.set_visible(w > 0)          # no stray outline before the bar starts
            xs, ys = step_path(S[pol]["events"], T)
            S[pol]["line"].set_data(xs, ys)
            n = ys[-1]
            S[pol]["head"].set_position((T + 6, n)); S[pol]["head"].set_text(f"{n}")
            S[pol]["count"].set_text(f"{n} 件")
            computing = any(s <= T < e for s, e in S[pol]["bars"])
            S[pol]["status"].set_text("● 計算中，AMR 等待" if computing else "● AMR 作業中")
            S[pol]["status"].set_color(COLOR[pol] if computing else TEXT2)
            waited = sum(max(0.0, min(e, T) - s) for s, e in S[pol]["bars"])
            S[pol]["wait"].set_text(f"累計等待計算 {waited:.0f} 秒")
        for c in cursors:
            c.set_xdata([T, T])
        clock.set_text(f"{T:.0f} 秒")
        if show_banner and final["fixed"] > 0:
            banner.set_text(f"同樣的 Taguchi-GA、同一批工單：iSched 完成 {final['ddqn']} 件，"
                            f"為固定每 10 秒（{final['fixed']} 件）的 {final['ddqn'] / final['fixed']:.1f} 倍")
        else:
            banner.set_text("")

    return fig, draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--episode", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=H / 16, help="video length for the 500 s episode (default 16x, same as animate_demo_style.py)")
    ap.add_argument("--hold", type=float, default=5.0, help="seconds to hold the final frame")
    ap.add_argument("--preview", type=float, default=None, help="only save one PNG frame at this sim time")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--ffmpeg", default=os.path.expanduser("~/.cache/dfjssp_tools/imageio_ffmpeg/binaries"))
    a = ap.parse_args()
    data = load(a.folder)
    base = os.path.join(a.folder, f"race_episode{a.episode}_zh")

    if a.preview is not None:
        fig, draw = build(data, a.episode)
        draw(a.preview, show_banner=a.preview >= H)
        fig.savefig(f"{base}_preview.png", dpi=120, facecolor=SURF)
        print(f"{base}_preview.png")
        return

    def render(out, writer, fps, dpi, size):
        fig, draw = build(data, a.episode, size)
        n_run, n_hold = int(a.seconds * fps), int(a.hold * fps)
        times = [H * i / (n_run - 1) for i in range(n_run)] + [H] * n_hold
        anim = FuncAnimation(fig, lambda i: draw(times[i], show_banner=i >= n_run), frames=len(times))
        anim.save(out, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": SURF})
        plt.close(fig)
        print(out)

    exe = a.ffmpeg
    if os.path.isdir(exe):
        found = sorted(glob.glob(os.path.join(exe, "ffmpeg*")))
        exe = found[0] if found else "ffmpeg"
    plt.rcParams["animation.ffmpeg_path"] = exe
    render(f"{base}.mp4", FFMpegWriter(fps=25, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]),
           fps=25, dpi=120, size=(16, 9))                       # 1920 x 1080
    if not a.no_gif:
        render(f"{base}.gif", PillowWriter(fps=10), fps=10, dpi=60, size=(16, 9))   # 960 x 540


if __name__ == "__main__":
    sys.exit(main())
