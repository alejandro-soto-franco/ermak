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

    box_l = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    pad = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

    # The system is periodic, so plot the tracer at its minimum-image position
    # inside the (origin-centred) crowder cell: it then threads through the
    # matrix by construction, independent of how far the unwrapped walk drifts.
    # Break the line wherever it crosses a face (a jump of more than half a box)
    # so wrapped re-entries are not drawn as spurious chords across the cell.
    wrapped = tracer - box_l * np.round(tracer / box_l)
    segments = [wrapped[0]]
    for i in range(1, len(wrapped)):
        if np.any(np.abs(wrapped[i] - wrapped[i - 1]) > box_l / 2):
            segments.append(np.full(3, np.nan))
        segments.append(wrapped[i])
    track = np.array(segments)
    start = wrapped[0]

    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.scatter(
        crowders[:, 0], crowders[:, 1], crowders[:, 2],
        s=240, color="#9bb7d4", alpha=0.40, edgecolors="#3a5a80", linewidths=0.5,
    )
    ax.plot(track[:, 0], track[:, 1], track[:, 2], color="#e0467c", lw=1.7)
    ax.scatter([start[0]], [start[1]], [start[2]], color="k", s=60, marker="*", zorder=5)

    lo = crowders.min(axis=0) - pad
    hi = crowders.max(axis=0) + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi - lo))
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
