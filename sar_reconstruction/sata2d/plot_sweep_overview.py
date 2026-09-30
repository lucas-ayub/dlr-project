"""
Overview figures of a geometry sweep (SATA 1D or SATA 2D, same CSV columns): for every
family the height profiles (N = 9) in 2D, the same cases in 3D, and the heatmaps
(N targets x parameter) of the improvement over no SATA, whole band and per sub-band:
    improvement [dB] = worst ambiguity (no SATA) - worst ambiguity (SATA);  > 0: SATA better.

  overview_<tag>_alongtrack_ramps.png
  overview_<tag>_crosstrack_ramps.png
  overview_<tag>_families_A.png   zig-zag, steepening, flattening, piecewise
  overview_<tag>_families_B.png   cliff, spike, mesa, random heights

Usage: python -m sata2d.plot_sweep_overview sweep_results.csv --label "SATA 2D" --tag 2d --out DIR
"""
import argparse
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    from .sweep_geometry_plots import iso_range_dy, family_3d
except ImportError:                                   # run as a plain script next to the helper
    from sweep_geometry_plots import iso_range_dy, family_3d

CMAP = LinearSegmentedColormap.from_list("improv", ["#B2182B", "#E8A48C", "#E4E4E4", "#8FB8DB", "#1F5FA8", "#0B2E5C"])
VMIN, VMAX = -40.0, 40.0
VARIANTS = [("amb_db_whole", "SATA whole band"), ("amb_db_sub", "SATA per sub-band")]
FAMILIES = {"ramp": ("along-track ramp", r"ramp inclination $\alpha$ [deg]", "a", "\N{DEGREE SIGN}"),
            "topo_ramp": ("cross-track ramp", r"ramp angle $\alpha$ [deg]", "a", "\N{DEGREE SIGN}"),
            "zigzag": ("zig-zag ramp", r"slope $\pm\alpha$ [deg]", "a", "\N{DEGREE SIGN}"),
            "steepening": ("steepening ramp", r"$\alpha_{max}$ [deg]", "amax", "\N{DEGREE SIGN}"),
            "flattening": ("flattening ramp", r"$\alpha_{max}$ [deg]", "amax", "\N{DEGREE SIGN}"),
            "piecewise": ("piecewise ramp", "seed", "seed", ""),
            "cliff": ("cliff", r"jump $\Delta h$ [m]", "dh", " m"),
            "spike": ("spike", r"$\Delta h$ [m]", "dh", " m"),
            "mesa": ("mesa", r"plateau $\Delta h$ [m]", "dh", " m"),
            "random_heights": ("random heights", r"$h_{max}$ [m] / seed", "hmax", "")}


def load(paths):
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    return [r for r in rows if float(r["bxt_max"]) > 0]


def param(slug, key):
    if key == "hmax":
        m = re.search(r"hmax([\d.]+)_seed(\d+)", slug)
        return (float(m.group(1)), int(m.group(2)))
    return float(re.search(rf"_{key}([\d.]+)", slug).group(1))


def plabel(p, key, unit):
    if key == "hmax":
        return f"{p[0]:g} m / seed {p[1]}"
    return ("seed " if key == "seed" else "") + f"{p:g}{unit}"


def coords(row):
    h = np.array([float(v) for v in row["heights"].split()])
    if row["family"] == "topo_ramp":
        dy = np.linspace(0.1, 1.0, len(h)) * 2000.0
        return np.zeros(len(h)), dy, h
    xs = (np.arange(len(h)) - (len(h) - 1) / 2.0) * 200.0
    return xs, iso_range_dy(h), h


def grid(rows, key, metric):
    Ns = sorted({int(r["N"]) for r in rows})
    Ps = sorted({param(r["slug"], key) for r in rows})
    M = np.full((len(Ns), len(Ps)), np.nan)
    for r in rows:
        M[Ns.index(int(r["N"])), Ps.index(param(r["slug"], key))] = float(r["amb_db_none"]) - float(r[metric])
    return Ns, Ps, M


def draw(ax, Ns, Ps, M, xlabel, key, unit, fs=7):
    im = ax.imshow(M, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect="auto")
    ax.set_xticks(range(len(Ps)), [(f"{p[0]:g}/{p[1]}" if key == "hmax" else f"{p:g}") for p in Ps], fontsize=8)
    ax.set_yticks(range(len(Ns)), Ns, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("N targets", fontsize=9)
    ax.set_xticks(np.arange(-.5, len(Ps)), minor=True); ax.set_yticks(np.arange(-.5, len(Ns)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    for i in range(len(Ns)):
        for j in range(len(Ps)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}".replace("-0.0", "0.0"), ha="center", va="center", fontsize=fs,
                        color="white" if abs(M[i, j]) > 17 else "#222222")
    if key == "hmax":
        ax.tick_params(axis="x", labelrotation=45)
    return im


def profiles(ax, rows, fam, N):
    title, xlabel, key, unit = FAMILIES[fam]
    r = sorted([x for x in rows if int(x["N"]) == N], key=lambda x: param(x["slug"], key))
    seq = key not in ("seed", "hmax")
    cols = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(r))) if seq else plt.get_cmap("tab10")(np.arange(len(r)) % 10)
    topo = fam == "topo_ramp"
    for x, c in zip(r, cols):
        xs, dys, hs = coords(x)
        ax.plot(dys if topo else xs, hs, "-o", color=c, ms=3, lw=1.3, label=plabel(param(x["slug"], key), key, unit))
    ax.axhline(0, color="#888888", lw=.7, ls="--")
    ax.set_title(f"{title}: height profiles, N = {N} (one curve per heatmap column)", fontsize=9.5)
    ax.set_xlabel("ground-range offset $\\Delta y$ [m] (all at azimuth 0)" if topo else "azimuth position [m]", fontsize=9)
    ax.set_ylabel("height [m]", fontsize=9); ax.tick_params(labelsize=8); ax.grid(alpha=.3)
    ax.legend(fontsize=6.5, ncol=2 if len(r) > 5 else 1, loc="best", handlelength=1.2, borderpad=.3, labelspacing=.2)
    return r, key, unit


def family_row(fig, gs_row, rows, fam, label, N=9):
    title, xlabel, key, unit = FAMILIES[fam]
    fr = [x for x in rows if x["family"] == fam]
    if not fr:
        return None
    r, key, unit = profiles(fig.add_subplot(gs_row[0]), fr, fam, N)
    a3 = fig.add_subplot(gs_row[1], projection="3d")
    family_3d(a3, [coords(x) for x in r], f"{title} in 3D, N = {N}",
              [plabel(param(x["slug"], key), key, unit) for x in r], topo=(fam == "topo_ramp"))
    im = None
    for j, (metric, lab) in enumerate(VARIANTS):
        ax = fig.add_subplot(gs_row[2 + j])
        Ns, Ps, M = grid(fr, key, metric)
        im = draw(ax, Ns, Ps, M, xlabel, key, unit, fs=7 if len(Ps) < 8 else 6.5)
        ax.set_title((f"{label}, {lab}: median {np.nanmedian(M):.1f} dB, min {np.nanmin(M):.1f}, max {np.nanmax(M):.1f}")
                     .replace("-0.0", "0.0"), fontsize=9)
    return im


def figure(rows, fams, label, out_png, title):
    fig = plt.figure(figsize=(24, 5.4 * len(fams)))
    gs = fig.add_gridspec(len(fams), 4, width_ratios=[1.0, 1.15, 1.25, 1.25], hspace=0.5, wspace=0.36)
    im = None
    for i, fam in enumerate(fams):
        im_ = family_row(fig, [gs[i, j] for j in range(4)], rows, fam, label)
        im = im_ or im
    if im is not None:
        cb = fig.colorbar(im, ax=fig.axes, fraction=0.012, pad=0.01)
        cb.set_label("improvement over no SATA [dB]")
    fig.suptitle(title, fontsize=12, y=1.0 if len(fams) > 1 else 1.04)
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)


def family_compare(rows1d, rows2d, fam, out_png, N=9):
    """One family, both algorithms: profiles (2D) and 3D view of the N = 9 cases on top,
    the improvement heatmaps of SATA 1D and of SATA 2D (whole band) below."""
    title, xlabel, key, unit = FAMILIES[fam]
    r1 = [x for x in rows1d if x["family"] == fam]
    r2 = [x for x in rows2d if x["family"] == fam]
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.1], height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.22)
    r, key, unit = profiles(fig.add_subplot(gs[0, 0]), r1, fam, N)
    a3 = fig.add_subplot(gs[0, 1], projection="3d")
    family_3d(a3, [coords(x) for x in r], f"{title} in 3D, N = {N}",
              [plabel(param(x["slug"], key), key, unit) for x in r], topo=(fam == "topo_ramp"))
    im = None
    for j, (rows, lab) in enumerate(((r1, "SATA 1D"), (r2, "SATA 2D"))):
        if not rows:
            continue
        ax = fig.add_subplot(gs[1, j])
        Ns, Ps, M = grid(rows, key, "amb_db_whole")
        im = draw(ax, Ns, Ps, M, xlabel, key, unit, fs=8 if len(Ps) < 8 else 7.5)
        ax.set_title((f"{lab}: improvement over no SATA, median {np.nanmedian(M):.1f} dB "
                      f"(min {np.nanmin(M):.1f}, max {np.nanmax(M):.1f})").replace("-0.0", "0.0"), fontsize=9.5)
    if im is not None:
        cb = fig.colorbar(im, ax=fig.axes[2:], fraction=0.02, pad=0.02)
        cb.set_label("improvement over no SATA [dB]", fontsize=9)
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--label", default="SATA 2D")
    ap.add_argument("--tag", default="2d")
    ap.add_argument("--out", default=".")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="CSV(s) of the 1D sweep: writes one figure per family with both heatmaps")
    a = ap.parse_args()
    rows = load(a.csv)
    os.makedirs(a.out, exist_ok=True)
    if a.compare:
        rows1d = load(a.compare)
        for fam in FAMILIES:
            family_compare(rows1d, rows, fam, os.path.join(a.out, f"family_{fam}.png"))
        print("saved", a.out)
        return
    base = (rf"{a.label}: improvement = worst ambiguity (no SATA) $-$ worst ambiguity (SATA)  [dB, > 0: SATA better] | "
            r"$b_{xt}\sim U(0,100)$ m, $N_{rx}=4$, DPCA")
    figure(rows, ["ramp"], a.label, os.path.join(a.out, f"overview_{a.tag}_alongtrack_ramps.png"),
           base + " | along-track ramps: iso-range targets 200 m apart, $h_j = j\\,200\\tan\\alpha$")
    figure(rows, ["topo_ramp"], a.label, os.path.join(a.out, f"overview_{a.tag}_crosstrack_ramps.png"),
           base + " | cross-track ramps: targets at azimuth 0, $\\Delta h = \\Delta y\\tan\\alpha$, $\\Delta y$ = 200..2000 m")
    figure(rows, ["zigzag", "steepening", "flattening", "piecewise"], a.label,
           os.path.join(a.out, f"overview_{a.tag}_families_A.png"), base + " | iso-range targets 200 m apart")
    figure(rows, ["cliff", "spike", "mesa", "random_heights"], a.label,
           os.path.join(a.out, f"overview_{a.tag}_families_B.png"), base + " | iso-range targets 200 m apart")
    print("saved", a.out)


if __name__ == "__main__":
    main()
