# -*- coding: utf-8 -*-
"""
Summary figures over the whole case matrix produced by ``run_cases.py``.

Reads ``plots/cases/cases.json`` and writes

``cases_summary_single.png``  the single-target family: focused peak of the
                              three reconstructions per case, and the in-band
                              error-to-signal ratio next to it.
``cases_predictor.png``       every target of every case, focused peak against
                              the channel-to-channel spread of its own
                              topographic residual -- the single number that
                              orders the whole matrix.
``cases_summary_ramp.png``    the iso-range ramp family: peak recovery against
                              the ramp slope alpha, the spread of the recovery
                              ALONG the ramp (the quantity a single whole-band
                              pass cannot follow), and the ESR.

Run from ``sar_reconstruction/``::

    python -m sata2d.plot_cases
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots", "cases")

METHODS = ("no", "whole", "sub")
COLOR = {"no": "#C62828", "whole": "#EF6C00", "sub": "#2E7D32"}
LABEL = {"no": "no SATA", "whole": "SATA whole band", "sub": "SATA per sub-band"}


def load(path=None):
    with open(path or os.path.join(PLOTS, "cases.json")) as fh:
        return json.load(fh)


def short_label(case):
    """Two-line tick label: the id, then whatever this case varies."""
    d = []
    if case["kind"] == "single":
        d.append(f"$h$={case['h']:.0f} m")
        if case["x_az"]:
            d.append(f"$x$={case['x_az']:.0f} m")
    else:
        d.append(f"$\\alpha$={case['alpha']:.0f}$^\\circ$")
        d.append(f"$h_{{ref}}$={case['h_ref']:.0f}")
    if case["bxt_max"] != 100.0:
        d.append(f"$b^{{max}}_{{xt}}$={case['bxt_max']:.0f}")
    if case["Nrx"] != 4:
        d.append(f"$N_{{rx}}$={case['Nrx']}")
    return case["id"] + "\n" + ", ".join(d)


def _mean_pct(rec, m):
    return float(np.mean([t[m]["pct"] for t in rec["targets"]]))


def _worst_pct(rec, m):
    return float(np.min([t[m]["pct"] for t in rec["targets"]]))


def _spread_pct(rec, m):
    v = [t[m]["pct"] for t in rec["targets"]]
    return float(np.max(v) - np.min(v))


# ---------------------------------------------------------------------------
def fig_single(recs, out):
    recs = [r for r in recs if r["case"]["kind"] == "single"]
    if not recs:
        return
    ids = [r["case"]["id"] for r in recs]
    x = np.arange(len(recs))
    w = 0.26

    fig, ax = plt.subplots(2, 1, figsize=(11.0, 6.4), dpi=150,
                           gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.38))

    for i, m in enumerate(METHODS):
        v = [r["targets"][0][m]["pct"] for r in recs]
        b = ax[0].bar(x + (i - 1) * w, v, w, color=COLOR[m], label=LABEL[m])
        ax[0].bar_label(b, fmt="%.0f", fontsize=7, padding=1)
    ax[0].axhline(100.0, color="0.45", ls="--", lw=1.0)
    ax[0].set_ylim(0, 108)
    ax[0].set_ylabel("focused peak [% of monostatic]")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([short_label(r["case"]) for r in recs], fontsize=7.5)
    ax[0].legend(fontsize=8, ncol=3, loc="upper center",
                 bbox_to_anchor=(0.5, 1.16), frameon=False)
    ax[0].grid(alpha=.3, axis="y")
    ax[0].set_title("Single point target -- peak recovery", fontsize="medium")

    for m, lw, ms in (("no", 1.4, 5), ("whole", 3.4, 8), ("sub", 1.4, 4)):
        ax[1].plot(x, [r["esr"][m] for r in recs], marker="o", ms=ms, lw=lw,
                   alpha=0.75 if m == "whole" else 1.0,
                   color=COLOR[m], label=LABEL[m])
    ax[1].set_ylabel("in-band error/signal [dB]")
    ax[1].set_xticks(x); ax[1].set_xticklabels(ids, fontsize=8)
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8, ncol=3)
    ax[1].set_title("whole-image error against the monostatic reference "
                    "(lower is better; the two SATA traces coincide)",
                    fontsize="medium")

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
def fig_ramp(recs, out):
    recs = [r for r in recs if r["case"]["kind"] == "ramp"]
    if not recs:
        return
    # the alpha sweep at the nominal geometry, in alpha order
    sweep = sorted([r for r in recs
                    if r["case"]["bxt_max"] == 100.0 and r["case"]["Nrx"] == 4
                    and r["case"]["h_ref"] == 240.0],
                   key=lambda r: r["case"]["alpha"])
    others = [r for r in recs if r not in sweep]

    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.3), dpi=150)
    fig.subplots_adjust(wspace=0.30)

    a = [r["case"]["alpha"] for r in sweep]
    for m, lw, ms in (("no", 1.4, 5), ("whole", 3.4, 8), ("sub", 1.4, 4)):
        ax[0].plot(a, [_worst_pct(r, m) for r in sweep], marker="o", ms=ms,
                   lw=lw, alpha=0.75 if m == "whole" else 1.0,
                   color=COLOR[m], label=LABEL[m])
    ax[0].axhline(100.0, color="0.45", ls="--", lw=1.0)
    ax[0].set_xlabel(r"ramp slope $\alpha$ [deg]")
    ax[0].set_ylabel("worst target [% of monostatic]")
    ax[0].set_title("peak recovery vs. ramp slope\n"
                    r"(the two SATA traces coincide up to $\alpha \approx 3^\circ$)",
                    fontsize="medium")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

    for m, lw, ms in (("no", 1.4, 5), ("whole", 3.4, 8), ("sub", 1.4, 4)):
        ax[1].plot(a, [_spread_pct(r, m) for r in sweep], marker="o", ms=ms,
                   lw=lw, alpha=0.75 if m == "whole" else 1.0,
                   color=COLOR[m], label=LABEL[m])
    ax[1].set_xlabel(r"ramp slope $\alpha$ [deg]")
    ax[1].set_ylabel("spread along the ramp\n[percentage points]")
    ax[1].set_title("how uneven the recovery is along the ramp", fontsize="medium")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)

    ids = [r["case"]["id"] for r in recs]
    x = np.arange(len(recs))
    for m, lw, ms in (("no", 1.4, 5), ("whole", 3.4, 8), ("sub", 1.4, 4)):
        ax[2].plot(x, [r["esr"][m] for r in recs], marker="o", ms=ms, lw=lw,
                   alpha=0.75 if m == "whole" else 1.0,
                   color=COLOR[m], label=LABEL[m])
    ax[2].set_xticks(x); ax[2].set_xticklabels(ids, fontsize=8)
    ax[2].set_ylabel("in-band error/signal [dB]")
    ax[2].set_title("whole-image error, all ramp cases", fontsize="medium")
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
def fig_predictor(recs, out):
    """Focused peak against the residual spread, one point per target.

    Eq. (3) of the report says the un-modelled residual of channel i is
    ``dC0_i = -b_xt,i h / (r0 tan theta)``.  A residual COMMON to all channels
    is a global image phase and costs nothing; what the reconstruction cannot
    absorb is the disagreement BETWEEN channels.  Plotting every target of
    every case against ``max_i dC0_i - min_i dC0_i`` in wavelengths therefore
    ought to collapse the whole matrix onto one curve -- and it does.
    """
    if not recs or "residual" not in recs[0]:
        print("  (no residual field in cases.json -- skipping the predictor figure)")
        return
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=150)
    for m, ms, z in (("no", 34, 3), ("whole", 46, 1), ("sub", 20, 4)):
        x = [r["residual"][j]["spread_cycles"]
             for r in recs for j in range(len(r["targets"]))]
        y = [t[m]["pct"] for r in recs for t in r["targets"]]
        ax.scatter(x, y, s=ms, color=COLOR[m], label=LABEL[m], zorder=z,
                   alpha=0.85 if m == "whole" else 1.0,
                   edgecolors="none")
    # label the extremes of the no-SATA cloud
    pts = sorted(((r["residual"][j]["spread_cycles"], r["targets"][j]["no"]["pct"],
                   r["case"]["id"]) for r in recs for j in range(len(r["targets"]))))
    for xx, yy, cid in (pts[0], pts[len(pts) // 2], pts[-1]):
        ax.annotate(cid, (xx, yy), textcoords="offset points", xytext=(6, -3),
                    fontsize=7.5, color="0.3")
    ax.axhline(100.0, color="0.45", ls="--", lw=1.0)
    ax.set_xlabel(r"channel-to-channel spread of the residual, "
                  r"$(\max_i - \min_i)\,\delta C_0^{(i)} / \lambda$   [cycles]")
    ax.set_ylabel("focused peak [% of monostatic]")
    ax.set_title("every target of every case, against the one number that "
                 "predicts it", fontsize="medium")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
def main(argv=None):
    recs = load(argv[0] if argv else None)
    fig_single(recs, os.path.join(PLOTS, "cases_summary_single.png"))
    fig_ramp(recs, os.path.join(PLOTS, "cases_summary_ramp.png"))
    fig_predictor(recs, os.path.join(PLOTS, "cases_predictor.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
