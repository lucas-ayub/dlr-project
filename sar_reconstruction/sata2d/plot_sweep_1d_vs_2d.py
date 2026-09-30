"""
SATA 1D vs SATA 2D on the ramp sweeps: heatmaps (N targets x alpha) of the improvement over
no SATA for the along-track ramps and the cross-track ramps, 1D and 2D side by side, plus the
worst ambiguity itself (no SATA / SATA / oracle) for the cross-track ramps.

Usage: python -m sata2d.plot_sweep_1d_vs_2d --csv1d <1D csv ...> --csv2d <2D csv ...> [--out DIR]
"""
import argparse
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_sweep_heatmaps_2d import CMAP, VMIN, VMAX, draw, fmt


def load(paths):
    return [r for p in paths for r in csv.DictReader(open(p)) if float(r["bxt_max"]) > 0]


def alpha(slug):
    return float(re.search(r"_a([\d.]+)", slug).group(1))


def grid(rows, a, b):
    Ns = sorted({int(r["N"]) for r in rows})
    Ps = sorted({alpha(r["slug"]) for r in rows})
    M = np.full((len(Ns), len(Ps)), np.nan)
    for r in rows:
        M[Ns.index(int(r["N"])), Ps.index(alpha(r["slug"]))] = float(r[a]) - (float(r[b]) if b else 0.0)
    return Ns, Ps, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv1d", nargs="+", required=True)
    ap.add_argument("--csv2d", nargs="+", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    r1, r2 = load(a.csv1d), load(a.csv2d)
    os.makedirs(a.out, exist_ok=True)

    # ---- improvement, 1D vs 2D, both ramp families ----------------------
    fig, axs = plt.subplots(2, 2, figsize=(17, 9.5), gridspec_kw=dict(hspace=0.45, wspace=0.15))
    for i, (fam, title, xl) in enumerate((("ramp", "along-track ramps", r"ramp inclination $\alpha$ [deg]"),
                                          ("topo_ramp", "cross-track ramps", r"ramp angle $\alpha$ [deg]"))):
        for j, (rows, lab) in enumerate(((r1, "SATA 1D"), (r2, "SATA 2D"))):
            rr = [x for x in rows if x["family"] == fam]
            Ns, Ps, M = grid(rr, "amb_db_none", "amb_db_whole")
            im = draw(axs[i, j], Ns, Ps, M, xl, fmt, fs=8)
            axs[i, j].set_title(f"{lab}, {title}: median {np.nanmedian(M):.1f} dB, "
                                f"min {np.nanmin(M):.1f}, max {np.nanmax(M):.1f}", fontsize=10)
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01)
    cb.set_label("improvement over no SATA [dB]")
    fig.suptitle(r"SATA 1D vs SATA 2D (whole band): improvement = worst ambiguity (no SATA) $-$ worst ambiguity (SATA)"
                 r"  [dB, > 0: SATA better] | $b_{xt}\sim U(0,100)$ m", fontsize=11.5, y=0.96)
    fig.savefig(os.path.join(a.out, "improvement_1d_vs_2d_ramps.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- cross-track ramps: the ambiguity itself -------------------------
    fig, axs = plt.subplots(2, 3, figsize=(20, 9), gridspec_kw=dict(hspace=0.45, wspace=0.15))
    panels = ((r1, "amb_db_none", "SATA 1D: no SATA"), (r1, "amb_db_whole", "SATA 1D: whole band"),
              (r1, "amb_db_oracle", "SATA 1D: oracle"),
              (r2, "amb_db_none", "SATA 2D: no SATA"), (r2, "amb_db_whole", "SATA 2D: whole band"),
              (r2, "amb_db_oracle", "SATA 2D: oracle"))
    for ax, (rows, key, lab) in zip(axs.ravel(), panels):
        rr = [x for x in rows if x["family"] == "topo_ramp"]
        Ns, Ps, M = grid(rr, key, None)
        im = ax.imshow(M, cmap="viridis_r", vmin=-40, vmax=-5, aspect="auto")
        ax.set_xticks(range(len(Ps)), [fmt(p) for p in Ps], fontsize=8); ax.set_yticks(range(len(Ns)), Ns, fontsize=8)
        ax.set_xlabel(r"ramp angle $\alpha$ [deg]", fontsize=9); ax.set_ylabel("N targets", fontsize=9)
        for i_ in range(len(Ns)):
            for j_ in range(len(Ps)):
                ax.text(j_, i_, f"{M[i_, j_]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i_, j_] < -22 else "#222222")
        ax.set_title(f"{lab}: median {np.nanmedian(M):.1f} dB", fontsize=10)
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01)
    cb.set_label("worst ambiguity [dB]")
    fig.suptitle(r"Cross-track ramps: worst ambiguity (reference $\approx -36\ldots-39$ dB) | "
                 r"$\Delta h=\Delta y\tan\alpha$, targets at azimuth 0, $b_{xt}\sim U(0,100)$ m", fontsize=11.5, y=0.96)
    fig.savefig(os.path.join(a.out, "crosstrack_ramps_1d_vs_2d.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", a.out)


if __name__ == "__main__":
    main()
