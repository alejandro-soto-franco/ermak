#!/usr/bin/env python3
"""Render a tracer diffusing through a crowder matrix.

    cargo run --release --example render_data crowd > crowd.csv
    python scripts/plot_crowd.py crowd.csv [box_l] [sigma]

Writes crowd.png next to the CSV. usetex, no gridlines.
"""
import sys
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "axes.grid": False, "font.size": 13}
)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "crowd.csv"
    crowders, tracer = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            p = (float(row["x"]), float(row["y"]), float(row["z"]))
            (crowders if row["kind"] == "crowder" else tracer).append(p)
    crowders = np.array(crowders)
    tracer = np.array(tracer)

    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.scatter(
        crowders[:, 0], crowders[:, 1], crowders[:, 2],
        s=260, color="#9bb7d4", alpha=0.45, edgecolors="#3a5a80", linewidths=0.5,
    )
    ax.plot(tracer[:, 0], tracer[:, 1], tracer[:, 2], color="#e0467c", lw=1.6)
    ax.scatter([tracer[0, 0]], [tracer[0, 1]], [tracer[0, 2]], color="k", s=50, marker="*")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title(r"Tracer diffusion through a crowder matrix")
    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
