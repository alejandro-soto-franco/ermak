#!/usr/bin/env python3
"""Plot ligand-escape kinetics from a barrier sweep, as two large figures.

    cargo run --release --example ligand_escape > escape.csv
    python scripts/plot_escape.py escape.csv

Writes escape_residence.png (residence time vs barrier, log scale) and
escape_tauramd.png (tauRAMD egress time vs true residence time) next to the CSV.
Large single panels, usetex, no gridlines.
"""
import sys
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "axes.grid": False, "font.size": 16}
)

INK = "#1a1a2e"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "escape.csv"
    barrier, residence, tau = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            barrier.append(float(row["barrier"]))
            residence.append(float(row["residence_time"]))
            tau.append(float(row["tauramd_time"]))

    stem = path.rsplit(".", 1)[0]

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.semilogy(barrier, residence, "o-", color=INK, markerfacecolor="#2563eb", lw=2.0, markersize=11)
    ax.set_xlabel(r"bottleneck barrier $\Delta U / k_\mathrm{B}T$")
    ax.set_ylabel(r"residence time $\tau_\mathrm{res}$")
    ax.set_title(r"Residence time grows with the barrier")
    fig.tight_layout()
    fig.savefig(f"{stem}_residence.png", dpi=200)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.plot(residence, tau, "o-", color=INK, markerfacecolor="#e0467c", lw=2.0, markersize=12)
    ax.set_xlabel(r"true residence time $\tau_\mathrm{res}$ (plain BD)")
    ax.set_ylabel(r"tauRAMD egress time")
    ax.set_title(r"tauRAMD egress times rank residence times")
    fig.tight_layout()
    fig.savefig(f"{stem}_tauramd.png", dpi=200)

    print(f"wrote {stem}_residence.png and {stem}_tauramd.png")


if __name__ == "__main__":
    main()
