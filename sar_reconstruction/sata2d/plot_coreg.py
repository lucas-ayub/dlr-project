# -*- coding: utf-8 -*-
"""
Figures for the range co-registration study, from the caches written by
``run_coreg.py``.  Plotting only -- no reconstruction is run here.

Produces
    plots/coreg_channels.png      the four channel echoes, before and after
    plots/coreg_equivalence.png   the 2x2 matrix and the resulting IRFs
    plots/coreg_pipeline.png      no SATA / standard / explicit co-registration

Run from ``sar_reconstruction/``::

    python -m sata2d.plot_coreg
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)

PLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
COLS = ["#C62828", "#EF6C00", "#1565C0", "#2E7D32"]


def _fine(line, Z=16):
    """Zero-padding interpolation of one range line, for a readable peak."""
    L = len(line)
    S = np.fft.fftshift(np.fft.fft(line))
    P = np.zeros(L * Z, complex)
    P[L * Z // 2 - L // 2: L * Z // 2 + L // 2] = S
    return np.abs(np.fft.ifft(np.fft.ifftshift(P)) * Z)


def fig_channels(d, out):
    """The concept: the same target, four channels, four range positions."""
    rho = float(d["rho_r"])
    prof, prof_co = d["prof"], d["prof_co"]
    Z = 16
    x = float(d["r_scan0"]) + np.arange(len(prof[0]) * Z) / Z * rho - float(d["r_target"])
    nrm = max(_fine(p).max() for p in prof)

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.3), dpi=170, sharey=True)
    for a, arr, ttl in ((ax[0], prof, "as recorded: four channels, four range bins"),
                        (ax[1], prof_co, "after co-registration: all four aligned")):
        a.axvspan(-rho / 2, rho / 2, color="0.85", alpha=.6, zorder=0)
        for i in range(len(arr)):
            a.plot(x, _fine(arr[i]) / nrm, color=COLS[i], lw=1.3,
                   label=f"ch{i}   $b_{{xt}}$={d['bxt'][i]:+.0f} m" if a is ax[0] else None)
        a.axvline(0, color="k", ls="--", lw=1.0)
        a.set_xlim(-90, 90)
        a.set_xlabel("slant-range offset from the true target range [m]")
        a.grid(alpha=.25)
        a.set_title(ttl)
    ax[0].set_ylabel("|range-compressed echo|")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].text(0.5, -0.30, f"grey band = one range cell ({rho:.0f} m)",
               transform=ax[0].transAxes, ha="center", fontsize=8, color="0.35")
    fig.suptitle("What the reconstruction sees, one range bin at a time",
                 fontsize="medium")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("written", out)


def fig_equivalence(d, out):
    """The 2x2 matrix: the two routes are the same correction."""
    labels = [str(s) for s in d["labels"]]
    pct, esr, irfs = d["pct"], d["esr"], d["irfs"]
    mono = d["irf_mono"]
    nrm = np.abs(mono).max()
    Na, prf = int(d["Na"]), float(d["prf"])
    t = (np.arange(Na) - Na // 2) / prf
    short = ["$\\lambda(f_r)$\nno co-reg\n(standard)",
             "$\\lambda(f_r)$\n+ co-reg\n(double count)",
             "$\\lambda_0$\nno co-reg\n(uncorrected)",
             "$\\lambda_0$\n+ co-reg\n(new pipeline)"]
    cols = ["#1565C0", "#C62828", "#C62828", "#2E7D32"]

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.6), dpi=170,
                           gridspec_kw={"width_ratios": [1.0, 1.35]})
    bars = ax[0].bar(range(4), pct, color=cols, width=.62)
    for i, (b, q, e) in enumerate(zip(bars, pct, esr)):
        ax[0].text(b.get_x() + b.get_width() / 2, q + 2.5,
                   f"{q:.1f}%\n{e:+.2f} dB", ha="center", fontsize=8)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(short, fontsize=7.5)
    ax[0].set_ylim(0, 118); ax[0].set_ylabel("focused peak, % of monostatic")
    ax[0].axhline(100, color="0.5", ls=":", lw=1.0)
    ax[0].grid(alpha=.25, axis="y")
    ax[0].set_title("STEP 1 filter $\\times$ explicit co-registration")

    db = lambda v: 20 * np.log10(np.abs(v) / nrm + 1e-20)
    ax[1].plot(t, db(mono), color="0.45", lw=1.0, label="monostatic reference")
    for i in (0, 2, 3):
        ax[1].plot(t, db(irfs[i]), color=cols[i], lw=0.9,
                   ls="--" if i == 3 else "-",
                   label=short[i].replace("\n", " "))
    ax[1].set_xlabel("Azimuth time $t_a$ [s]")
    ax[1].set_ylabel("Impulse Response [dB]")
    ax[1].set_ylim(-90, 3); ax[1].grid(alpha=.25)
    ax[1].legend(fontsize=7.5, loc="upper right")
    ax[1].set_title("azimuth IRF (the green dashes sit on the blue line)")

    fig.suptitle("The range-frequency filter and the explicit co-registration "
                 "are the same correction", fontsize="medium")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("written", out)


def fig_pipeline(d, out):
    """The full pipeline, standard vs explicit co-registration."""
    labels = [str(s) for s in d["labels"]]
    pct, esr, irfs = d["pct"], d["esr"], d["irfs"]
    mono = d["irf_mono"]; nrm = np.abs(mono).max()
    Na, prf = int(d["Na"]), float(d["prf"])
    t = (np.arange(Na) - Na // 2) / prf
    cols = ["#C62828", "#1565C0", "#2E7D32"]
    db = lambda v: 20 * np.log10(np.abs(v) / nrm + 1e-20)

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.4), dpi=170,
                           gridspec_kw={"width_ratios": [1.4, 1.0]})
    ax[0].plot(t, db(mono), color="0.45", lw=1.0, label="monostatic reference")
    for i, lab in enumerate(labels):
        ax[0].plot(t, db(irfs[i]), color=cols[i], lw=0.9,
                   ls="--" if i == 2 else "-",
                   label=f"{lab}  ({pct[i]:.1f}%)")
    ax[0].set_xlabel("Azimuth time $t_a$ [s]"); ax[0].set_ylabel("Impulse Response [dB]")
    ax[0].set_ylim(-90, 3); ax[0].grid(alpha=.25); ax[0].legend(fontsize=7.5)
    ax[0].set_title("azimuth IRF")

    w = 260
    sl = slice(Na // 2 - w, Na // 2 + w)
    ax[1].plot(t[sl] * 1e3, db(mono)[sl], color="0.45", lw=1.2)
    for i in range(len(labels)):
        ax[1].plot(t[sl] * 1e3, db(irfs[i])[sl], color=cols[i], lw=1.2,
                   ls="--" if i == 2 else "-")
    ax[1].set_xlabel("Azimuth time $t_a$ [ms]"); ax[1].set_ylabel("[dB]")
    ax[1].set_ylim(-40, 3); ax[1].grid(alpha=.25)
    ax[1].set_title("main lobe -- zoom")

    fig.suptitle("SATA per sub-band: standard pipeline vs explicit "
                 "co-registration (they coincide)", fontsize="medium")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("written", out)


def main():
    eq = os.path.join(PLOTS, "cache_coreg_equiv.npz")
    pl = os.path.join(PLOTS, "cache_coreg_pipeline.npz")
    if os.path.exists(pl):
        d = np.load(pl, allow_pickle=False)
        fig_channels(d, os.path.join(PLOTS, "coreg_channels.png"))
        fig_pipeline(d, os.path.join(PLOTS, "coreg_pipeline.png"))
    else:
        print("missing", pl, "-- run: python -m sata2d.run_coreg --stage pipeline")
    if os.path.exists(eq):
        fig_equivalence(np.load(eq, allow_pickle=False),
                        os.path.join(PLOTS, "coreg_equivalence.png"))
    else:
        print("missing", eq, "-- run: python -m sata2d.run_coreg --stage equiv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
