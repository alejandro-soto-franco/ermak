#!/usr/bin/env python3
"""Plot ligand-escape kinetics from a barrier sweep.

    cargo run --release --example ligand_escape > escape.csv
    python scripts/plot_escape.py escape.csv

Left: residence time vs bottleneck barrier (log scale, the Arrhenius law).
Right: tauRAMD egress time vs true residence time (it ranks them).
Needs LaTeX + dvipng. Writes escape.png next to the CSV.
"""
import sys
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "axes.grid": False, "font.size": 12}
)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "escape.csv"
    barrier, residence, tau = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            barrier.append(float(row["barrier"]))
            residence.append(float(row["residence_time"]))
            tau.append(float(row["tauramd_time"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.5))

    ax1.semilogy(barrier, residence, "o-", color="#1a1a2e", markerfacecolor="#2563eb", lw=1.6)
    ax1.set_xlabel(r"bottleneck barrier $\Delta U / k_\mathrm{B}T$")
    ax1.set_ylabel(r"residence time $\tau_\mathrm{res}$")
    ax1.set_title(r"Residence time vs barrier (Arrhenius)")

    ax2.plot(residence, tau, "o", color="#1a1a2e", markerfacecolor="#e0467c", markersize=7)
    ax2.set_xlabel(r"true residence time $\tau_\mathrm{res}$")
    ax2.set_ylabel(r"tauRAMD egress time")
    ax2.set_title(r"tauRAMD ranks residence times")

    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
