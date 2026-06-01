#!/usr/bin/env python3
"""Plot effective diffusion vs crowder volume fraction from a sweep CSV.

    cargo run --release --example crowding_sweep > crowding.csv
    python scripts/plot_crowding.py crowding.csv

Needs LaTeX + dvipng (text.usetex). Writes crowding.png next to the CSV.
"""
import sys
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.grid": False,
        "font.size": 12,
    }
)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "crowding.csv"
    phi, ratio = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            phi.append(float(row["phi"]))
            ratio.append(float(row["d_eff_over_d0"]))

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(phi, ratio, "o-", color="#1a1a2e", markerfacecolor="#2563eb", lw=1.6)
    ax.set_xlabel(r"crowder volume fraction $\phi$")
    ax.set_ylabel(r"$D_\mathrm{eff} / D_0$")
    ax.set_title(r"Crowders slow tracer diffusion")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(left=0.0)
    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
