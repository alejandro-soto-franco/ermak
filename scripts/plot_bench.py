#!/usr/bin/env python3
"""Bar chart of CPU vs GPU ensemble throughput.

    cargo run --release --features gpu --example bench_throughput > bench.csv
    python scripts/plot_bench.py bench.csv

Writes bench.png next to the CSV. usetex, no gridlines.
"""
import sys
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "axes.grid": False, "font.size": 15}
)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "bench.csv"
    label, wps = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            label.append(row["backend"])
            wps.append(float(row["walkers_per_sec"]))

    palette = {"CPU": "#1a1a2e", "GPU (f64)": "#9aa7c7", "GPU (f32)": "#2563eb"}
    colors = [palette.get(l, "#888888") for l in label]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    bars = ax.bar(label, wps, color=colors, width=0.6)
    ax.set_ylabel(r"throughput (walkers / s)")
    ax.set_title(r"Ensemble throughput, crowded workload")
    for b, w in zip(bars, wps):
        ax.text(b.get_x() + b.get_width() / 2, w, f"{w:,.0f}", ha="center", va="bottom")
    if "CPU" in label and "GPU (f32)" in label:
        sp = wps[label.index("GPU (f32)")] / wps[label.index("CPU")]
        ax.text(
            0.5, 0.93, rf"GPU (f32) speedup $= {sp:.1f}\times$ vs CPU",
            transform=ax.transAxes, ha="center", fontsize=14,
        )
    ax.margins(y=0.2)
    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
