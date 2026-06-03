#!/usr/bin/env python3
"""Plot the binding free energy from the pocket well, one large figure.

    cargo run --release --example binding_free_energy > binding.csv
    python scripts/plot_binding.py binding.csv

Writes binding_free_energy.png next to the CSV: dG vs well depth for three
bottleneck radii, showing that deeper or wider wells bind more tightly.
Large single panel, usetex, no gridlines.
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
SERIES = [
    ("rb_1.0", r"$r_b = 1.0$", "#2563eb"),
    ("rb_1.5", r"$r_b = 1.5$", "#7c3aed"),
    ("rb_2.0", r"$r_b = 2.0$", "#e0467c"),
]


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "binding.csv"
    depth = []
    cols = {key: [] for key, _, _ in SERIES}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            depth.append(float(row["well_depth"]))
            for key, _, _ in SERIES:
                cols[key].append(float(row[key]))

    stem = path.rsplit(".", 1)[0]

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for key, label, colour in SERIES:
        ax.plot(depth, cols[key], "o-", color=INK, markerfacecolor=colour, lw=2.0, markersize=10, label=label)
    ax.set_xlabel(r"well depth $\epsilon$ ($k_\mathrm{B}T$ units of integrand)")
    ax.set_ylabel(r"binding free energy $\Delta G$ (kcal/mol)")
    ax.set_title(r"Deeper, wider wells bind more tightly")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{stem}_free_energy.png", dpi=200)
    print(f"wrote {stem}_free_energy.png")


if __name__ == "__main__":
    main()
