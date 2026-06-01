#!/usr/bin/env python3
"""Parity plot for the random-forest k_off prediction.

    cargo run --release --example koff_prediction > koff_parity.csv
    python scripts/plot_koff.py koff_parity.csv

Large single panel, usetex, no gridlines. Writes koff_parity.png next to the CSV.
"""
import sys
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "axes.grid": False, "font.size": 16}
)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "koff_parity.csv"
    true, pred = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            true.append(float(row["true_log_koff"]))
            pred.append(float(row["pred_log_koff"]))

    lo = min(min(true), min(pred))
    hi = max(max(true), max(pred))
    pad = 0.05 * (hi - lo)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", lw=1.4)
    ax.plot(true, pred, "o", color="#1a1a2e", markerfacecolor="#2563eb", markersize=10)
    ax.set_xlabel(r"true $\log k_\mathrm{off}$ (Brownian dynamics)")
    ax.set_ylabel(r"predicted $\log k_\mathrm{off}$ (random forest)")
    ax.set_title(r"Held-out $k_\mathrm{off}$ prediction")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
