#!/usr/bin/env python3
"""
plot_scaling.py - draw the array-size scaling chart used in the README.

    python tools/plot_scaling.py

Reads  docs/results/scaling.csv
Writes docs/scaling_light.png and docs/scaling_dark.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import PathPatch              # noqa: E402
from matplotlib.path import Path                      # noqa: E402
from matplotlib.ticker import NullLocator             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "docs", "results", "scaling.csv")

THEMES = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7", "bar": "#2a78d6"},
    "dark":  {"surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7",
              "muted": "#898781", "grid": "#2c2c2a", "baseline": "#383835", "bar": "#3987e5"},
}
DPI = 200
BAR_PX = 24          # bar thickness in screen pixels (at 100 % zoom)
RADIUS_PX = 4        # rounded data end


def rounded_column(ax, x, y0, y1, color):
    """Column from the baseline y0 to y1, square at the base, 4 px round top."""
    to_px = ax.transData.transform
    scale = DPI / 100.0
    (xc, yb), = to_px([(x, y0)])
    (_, yt), = to_px([(x, y1)])
    half = BAR_PX * scale / 2
    r = min(RADIUS_PX * scale, half, max(yt - yb, 0))
    left, right = xc - half, xc + half
    verts = [(left, yb), (left, yt - r), (left, yt), (left + r, yt),
             (right - r, yt), (right, yt), (right, yt - r), (right, yb), (left, yb)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none",
                           transform=None, zorder=3))


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["baseline"])
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", colors=t["muted"], labelsize=9, length=0)
    ax.tick_params(axis="x", labelcolor=t["secondary"], pad=6)
    ax.grid(axis="y", color=t["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def draw(theme):
    t = THEMES[theme]
    with open(CSV) as f:
        rows = list(csv.DictReader(f))
    labels = ["{0}×{0}".format(r["N"]) for r in rows]
    xs = list(range(len(rows)))
    speedup = [float(r["speedup"]) for r in rows]
    util = [float(r["utilization_pct"]) for r in rows]
    macs = [int(r["macs"]) for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=DPI,
                                 gridspec_kw={"wspace": 0.28})
    fig.patch.set_facecolor(t["surface"])

    # ---- speed-up over a single MAC (log scale) ----
    style_axes(a1, t)
    a1.set_yscale("log")
    a1.set_ylim(0.8, 600)
    a1.set_xlim(-0.6, len(rows) - 0.4)
    a1.set_yticks([1, 10, 100])
    a1.set_yticklabels(["1×", "10×", "100×"])
    a1.yaxis.set_minor_locator(NullLocator())
    a1.tick_params(axis="y", which="both", length=0)
    a1.set_xticks(xs)
    a1.set_xticklabels(labels)
    fig.canvas.draw()
    for x, s in zip(xs, speedup):
        rounded_column(a1, x, 0.8, s, t["bar"])
        a1.annotate("{:.1f}×".format(s), (x, s), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=t["primary"])
    for x, m in zip(xs, macs):
        a1.annotate("{} MAC{}".format(m, "" if m == 1 else "s"), (x, 0.8), xytext=(0, -26),
                    textcoords="offset points", ha="center", va="top", fontsize=8,
                    color=t["muted"], annotation_clip=False)
    a1.set_title("Speed-up over a single MAC unit", loc="left", fontsize=11,
                 color=t["primary"], pad=22)
    a1.text(0, 1.02, "cycles per image, 1×1 = 109,226; ideal = number of MACs",
            transform=a1.transAxes, fontsize=8.5, color=t["secondary"])

    # ---- array utilization ----
    style_axes(a2, t)
    a2.set_ylim(0, 112)
    a2.set_xlim(-0.6, len(rows) - 0.4)
    a2.set_yticks([0, 25, 50, 75, 100])
    a2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    a2.set_xticks(xs)
    a2.set_xticklabels(labels)
    fig.canvas.draw()
    for x, u in zip(xs, util):
        rounded_column(a2, x, 0, u, t["bar"])
        a2.annotate("{:.2f}%".format(u), (x, u), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=t["primary"])
    a2.set_title("Array utilization", loc="left", fontsize=11, color=t["primary"], pad=22)
    a2.text(0, 1.02, "useful MACs ÷ (cycles × array MACs), 10,000 images",
            transform=a2.transAxes, fontsize=8.5, color=t["secondary"])

    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.17)
    fig.canvas.draw()
    # re-draw the columns now that the layout is final (they are placed in pixels)
    for ax in (a1, a2):
        for p in list(ax.patches):
            p.remove()
    for x, s in zip(xs, speedup):
        rounded_column(a1, x, 0.8, s, t["bar"])
    for x, u in zip(xs, util):
        rounded_column(a2, x, 0, u, t["bar"])

    out = os.path.join(ROOT, "docs", "scaling_{}.png".format(theme))
    fig.savefig(out, dpi=DPI, facecolor=t["surface"])
    plt.close(fig)
    print("wrote " + out)


if __name__ == "__main__":
    for name in THEMES:
        draw(name)
