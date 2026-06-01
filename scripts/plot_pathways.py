#!/usr/bin/env python3
"""Render ligand escape trajectories out of a buried pocket (multiple pathways).

    cargo run --release --example render_data paths > paths.csv
    python scripts/plot_pathways.py paths.csv [r_b]

Writes paths.png next to the CSV. usetex, no gridlines.
"""
import sys
import csv
from collections import defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "axes.grid": False, "font.size": 13}
)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "paths.csv"
    r_b = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    traj = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            traj[row["path"]].append((float(row["x"]), float(row["y"]), float(row["z"])))

    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    colors = cm.viridis(np.linspace(0.0, 1.0, len(traj)))
    for (_, pts), c in zip(traj.items(), colors):
        a = np.array(pts)
        ax.plot(a[:, 0], a[:, 1], a[:, 2], color=c, lw=1.4, alpha=0.85)
        ax.scatter(a[-1, 0], a[-1, 1], a[-1, 2], color=c, s=34, edgecolors="k", linewidths=0.4)
    ax.scatter([0], [0], [0], color="k", s=90, marker="*", label="pocket centre")

    # bottleneck sphere of radius r_b
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = r_b * np.outer(np.cos(u), np.sin(v))
    ys = r_b * np.outer(np.sin(u), np.sin(v))
    zs = r_b * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.18, linewidth=0.5)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title(r"Ligand escape by multiple egress pathways")
    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
