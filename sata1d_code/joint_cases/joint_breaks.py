"""
Every case of the joint sweep where the joint reconstruction (occupied parcels) breaks, over all the baselines of
run_joint_sweep.py (BASELINES + EXTRA_BASELINES). Break: highest ambiguity of the joint at least CRIT dB above
SATA whole band or above no SATA, and at least GUARD dB above the reference. First pages: the difference joint - SATA whole band for every case x baseline,
breaks marked; then the sweep page of every break, case by case.
Run from sata1d_code after
    python joint_cases/run_joint_sweep.py --workers 4 --baselines <tags of EXTRA_BASELINES>
    python joint_cases/joint_breaks.py                      # -> joint_cases/joint_breaks.pdf
"""
import argparse
import glob
import json
import os
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import run_joint_sweep as W
import joint_sweep_summary as JS

CRIT = 3.0
GUARD = 10.0          # and at least GUARD dB above the reference (cases already near the floor are not breaks)
ALL = W.BASELINES + W.EXTRA_BASELINES
# columns grouped by kind of baseline
ORDER = ["rnd100", "rnd500",
         "s10", "s20", "s30", "s50", "s75", "s100", "s150",
         "perm50", "dec50", "uneven",
         "c50", "s110", "c200", "s300",
         "pm25", "pm50", "pm100", "pm60", "pm120", "pm150"]
COLGROUPS = [("random", 2), ("equal steps", 7), ("other orders", 3), ("close together", 4), ("negative and positive", 6)]


def breaks_of(r):
    return ((r["joint_occ"] - r["whole"] >= CRIT) or (r["joint_occ"] > r["none"])) and (r["joint_occ"] - r["ref"] >= GUARD)


def overview(cases, res, out_png):
    nb = len(ORDER)
    M = np.full((len(cases), nb), np.nan)
    B = np.zeros_like(M, bool)
    for i, c in enumerate(cases):
        for j, b in enumerate(ORDER):
            r = res.get((c, b))
            if r:
                M[i, j] = r["joint_occ"] - r["whole"]
                B[i, j] = breaks_of(r)
    labs = [f"{int(c[1:])}. " + res[(c, ORDER[0])]["label"].replace("$\\pi$", "π") for c in cases]
    fig = plt.figure(figsize=(26, 0.36 * len(cases) + 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[nb, 0.3], wspace=0.02, left=0.2, right=0.95, top=0.76, bottom=0.04)
    ax = fig.add_subplot(gs[0, 0]); cb = fig.add_subplot(gs[0, 1])
    lim = 20.0
    im = ax.imshow(np.clip(M, -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=7.5,
                        color="0.1" if abs(M[i, j]) < 0.6 * lim else "white", fontweight="bold" if B[i, j] else "normal")
                if B[i, j]:
                    ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec="k", lw=1.6))
    ax.set_yticks(range(len(cases))); ax.set_yticklabels(labs, fontsize=8.5)
    ax.set_xticks(range(nb))
    ax.set_xticklabels([W.bxt_text(W.BXT[b]) + "  " + W.BKIND[b] for b in ORDER], fontsize=8.5, rotation=90)
    ax.xaxis.tick_top()
    k0 = 0
    for name, n in COLGROUPS:
        if k0:
            ax.axvline(k0 - 0.5, color="k", lw=1.6)
        ax.text(k0 + (n - 1) / 2.0, len(cases) - 0.5 + 0.4, name, ha="center", va="top", fontsize=10, fontweight="bold")
        k0 += n
    k0 = 0
    for g, n in [(g, sum(W.CASES[c][0] == g for c in cases)) for g, _ in W.GROUPS]:
        if k0:
            ax.axhline(k0 - 0.5, color="k", lw=1.6)
        ax.text(nb - 0.3, k0 + (n - 1) / 2.0, dict(W.GROUPS)[g], rotation=270, ha="left", va="center", fontsize=10, fontweight="bold")
        k0 += n
    cbar = fig.colorbar(im, cax=cb)
    cbar.set_label("joint (occupied parcels) minus SATA whole band [dB]   (blue: joint better, red: joint worse)")
    cb.set_position([cb.get_position().x0 + 0.02, cb.get_position().y0, 0.008, cb.get_position().height])
    nbrk = int(B.sum())
    fig.suptitle(f"Where the joint reconstruction breaks: {nbrk} of {int(np.isfinite(M).sum())} runs (black boxes)\n"
                 f"break = joint at least {CRIT:g} dB above SATA whole band or above no SATA, and at least {GUARD:g} dB above the reference | cross-track baselines "
                 "$b_{xt}$ [m] of channels 0, 1, 2, 3, channel 0 is the transmitter ($b_{at}$ = $b_{xt}$ = 0)", fontsize=13, y=0.99)
    fig.savefig(out_png, dpi=85); plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=W.OUT)
    ap.add_argument("--pdf", default=os.path.join(HERE, "joint_breaks.pdf"))
    a = ap.parse_args()
    res = {}
    for p in glob.glob(os.path.join(a.out, "*", "C*.json")):
        r = json.load(open(p))
        res[(r["case"], r["baseline"])] = r
    cases = [c for c in W.CASES if (c, ORDER[0]) in res]
    missing = [(c, b) for c in cases for b in ORDER if (c, b) not in res]
    if missing:
        print(len(missing), "runs missing, e.g.", missing[:5])
    hm = os.path.join(a.out, "heatmaps"); os.makedirs(hm, exist_ok=True)
    pngs = [overview(cases, res, os.path.join(hm, "breaks_overview.png"))]
    outline = [("Overview: where the joint breaks", 0, 0)]
    g_prev = None
    for c in cases:
        brk = [b for b in ORDER if (c, b) in res and breaks_of(res[(c, b)])]
        if not brk:
            continue
        g = W.CASES[c][0]
        if g != g_prev:
            outline.append((dict(W.GROUPS)[g], len(pngs), 0)); g_prev = g
        outline.append((f"{int(c[1:])}. {res[(c, brk[0])]['label']}".replace("$\\pi$", "π") + f" ({len(brk)} baselines)", len(pngs), 1))
        pngs += [os.path.join(a.out, b, f"{c}.png") for b in brk]
    print("wrote", JS.bundle(pngs, a.pdf, outline), len(pngs), "pages")


if __name__ == "__main__":
    main()
