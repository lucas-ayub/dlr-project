"""
Heatmaps of the improvement given by SATA over no correction, from the geometry sweep:
    improvement [dB] = worst ambiguity (no SATA) - worst ambiguity (SATA);  > 0: SATA better.

  improvement_heatmap_ramps.png       uniform ramps: profiles (N = 9) + heatmaps, N targets x inclination
  improvement_heatmap_geometries.png  other families: profiles (N = 9) above heatmaps, N targets x parameter
  improvement_heatmap_topo_ramps.png  topographic ramp (targets at one azimuth, ramp in ground range)

Usage: python plot_sweep_improvement_heatmaps.py [sweep_results.csv ...] [--out DIR]
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

CMAP = LinearSegmentedColormap.from_list("improv", ["#B2182B", "#E8A48C", "#E4E4E4", "#8FB8DB", "#1F5FA8", "#0B2E5C"])
VMIN, VMAX = -40.0, 40.0
VARIANTS = [("amb_db_whole", "SATA whole band"), ("amb_db_sub", "SATA per sub-band")]
FAMILIES = [("zigzag", "zigzag", "slope $\\pm\\alpha$ [deg]", "a"),
            ("steepening", "steepening", "$\\alpha_{max}$ [deg]", "amax"),
            ("flattening", "flattening", "$\\alpha_{max}$ [deg]", "amax"),
            ("piecewise", "piecewise ramp", "seed", "seed"),
            ("cliff", "cliff", "jump $\\Delta h$ [m]", "dh"),
            ("spike", "spike", "$\\Delta h$ [m]", "dh"),
            ("mesa", "mesa", "plateau $\\Delta h$ [m]", "dh"),
            ("random_heights", "random heights", "$h_{max}$ [m] / seed", "hmax")]


def load(paths):
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    return [r for r in rows if float(r["bxt_max"]) > 0]


def param(slug, key):
    if key == "hmax":
        m = re.search(r"hmax([\d.]+)_seed(\d+)", slug)
        return (float(m.group(1)), int(m.group(2)))
    return float(re.search(rf"_{key}([\d.]+)", slug).group(1))


def grid(rows, key, metric):
    Ns = sorted({int(r["N"]) for r in rows})
    Ps = sorted({param(r["slug"], key) for r in rows})
    M = np.full((len(Ns), len(Ps)), np.nan)
    for r in rows:
        M[Ns.index(int(r["N"])), Ps.index(param(r["slug"], key))] = \
            float(r["amb_db_none"]) - float(r[metric])
    return Ns, Ps, M


def draw(ax, Ns, Ps, M, xlabel, plabel, fs=7):
    im = ax.imshow(M, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect="auto")
    ax.set_xticks(range(len(Ps)), [plabel(p) for p in Ps], fontsize=8)
    ax.set_yticks(range(len(Ns)), Ns, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("N targets", fontsize=9)
    ax.set_xticks(np.arange(-.5, len(Ps)), minor=True); ax.set_yticks(np.arange(-.5, len(Ns)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    for i in range(len(Ns)):
        for j in range(len(Ps)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{round(M[i, j], 1) + 0.0:.1f}", ha="center", va="center", fontsize=fs,
                        color="white" if abs(M[i, j]) > 17 else "#222222")
    return im


def fmt(p):
    return f"{p:g}"


def profiles(ax, rows, key, N, title, plabel, sequential=True, loc="best", topo=False):
    """Height profiles of one family (N targets), one curve per parameter value."""
    r = sorted([x for x in rows if int(x["N"]) == N], key=lambda x: param(x["slug"], key))
    if sequential:
        cols = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(r)))
    else:
        cols = plt.get_cmap("tab10")(np.arange(len(r)) % 10)
    for x, c in zip(r, cols):
        h = np.array([float(v) for v in x["heights"].split()])
        xs = np.linspace(0.1, 1.0, len(h)) * 2000.0 if topo else (np.arange(len(h)) - (len(h) - 1) / 2.0) * 200.0
        ax.plot(xs, h, "-o", color=c, ms=3, lw=1.3, label=plabel(param(x["slug"], key)))
    ax.axhline(0, color="#888888", lw=.7, ls="--")
    ax.set_title(f"{title}: profiles, N = {N}", fontsize=10)
    ax.set_xlabel("ground-range offset $\\Delta y$ [m] (all at azimuth 0)" if topo else "azimuth position [m]", fontsize=9); ax.set_ylabel("height [m]", fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=.3)
    ax.legend(fontsize=6.5, ncol=2 if len(r) > 5 else 1, loc=loc, handlelength=1.2, borderpad=.3, labelspacing=.2)


def ramps(rows, out_png):
    r = [x for x in rows if x["family"] == "ramp"]
    fig = plt.figure(figsize=(19, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.35, wspace=0.18)
    top = gs[0, :].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=0.05)
    profiles(fig.add_subplot(top[0, 0]), r, "a", 9, "uniform ramp", lambda p: f"{p:g}\N{DEGREE SIGN}")
    azramp3d(fig.add_subplot(top[0, 1], projection="3d"), r)
    axs = [fig.add_subplot(gs[1, j]) for j in range(2)]
    for ax, (metric, lab) in zip(axs, VARIANTS):
        Ns, Ps, M = grid(r, "a", metric)
        im = draw(ax, Ns, Ps, M, r"ramp inclination $\alpha$ [deg]", fmt)
        ax.set_title(f"{lab}: median {np.nanmedian(M):.1f} dB, min {np.nanmin(M):.1f}, max {np.nanmax(M):.1f}",
                     fontsize=10)
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01)
    cb.set_label("improvement over no SATA [dB]")
    fig.suptitle(r"Uniform ramps: improvement = worst ambiguity (no SATA) $-$ worst ambiguity (SATA)  "
                 r"[dB, > 0: SATA better] | spacing 200 m, $b_{xt}\sim U(0,100)$ m", fontsize=11, y=0.97)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def geometries(rows, out_png, metric="amb_db_whole", label="SATA whole band"):
    fig, axs = plt.subplots(4, 4, figsize=(21, 15.5),
                            gridspec_kw=dict(height_ratios=[1, 0.9, 1, 0.9], hspace=0.62, wspace=0.28))
    for k, (fam, title, xlabel, key) in enumerate(FAMILIES):
        ag, ah = axs[2 * (k // 4), k % 4], axs[2 * (k // 4) + 1, k % 4]
        r = [x for x in rows if x["family"] == fam]
        plabel = (lambda p: f"{p[0]:g}/{p[1]}") if key == "hmax" else fmt
        unit = {"a": "\N{DEGREE SIGN}", "amax": "\N{DEGREE SIGN}", "dh": " m"}.get(key, "")
        profiles(ag, r, key, 9, title,
                 (lambda p: ("seed " if key == "seed" else "") + plabel(p) + unit),
                 sequential=key not in ("seed", "hmax"), loc="center" if fam == "mesa" else "best")
        Ns, Ps, M = grid(r, key, metric)
        im = draw(ah, Ns, Ps, M, xlabel, plabel, fs=7 if len(Ps) < 8 else 6.5)
        ah.set_title(f"{title}: improvement, median {np.nanmedian(M):.1f} dB", fontsize=10)
        if key == "hmax":
            ah.tick_params(axis="x", labelrotation=45)
    fig.subplots_adjust(right=0.9)
    cb = fig.colorbar(im, cax=fig.add_axes([0.92, 0.3, 0.012, 0.4]))
    cb.set_label("improvement over no SATA [dB]")
    fig.suptitle(rf"Varied geometries, {label}: improvement = worst ambiguity (no SATA) $-$ worst ambiguity (SATA)  "
                 r"[dB, > 0: SATA better] | spacing 200 m, $b_{xt}\sim U(0,100)$ m" + "\n"
                 "above each heatmap: the height profiles of that family for N = 9 (one curve per heatmap column)",
                 fontsize=11, y=0.935)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def ramp3d(ax, rows, N=9):
    """Topographic ramps in 3D: one surface per angle (extended in azimuth to show the slope), targets as dots."""
    r = sorted([x for x in rows if int(x["N"]) == N], key=lambda x: param(x["slug"], "a"))
    cols = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(r)))
    az = np.linspace(-1000, 1000, 2)
    for x, c in zip(r, cols):
        h = np.array([float(v) for v in x["heights"].split()])
        dy = np.linspace(0.1, 1.0, len(h)) * 2000.0
        A_, Y_ = np.meshgrid(az, np.r_[0.0, dy])
        Z_ = np.tile(np.r_[0.0, h][:, None], (1, len(az)))
        ax.plot_surface(A_, Y_, Z_, color=c, alpha=0.18, linewidth=0, shade=False)
        ax.plot(az, [dy[-1]] * 2, [h[-1]] * 2, color=c, lw=1.2)
        ax.plot([0, 0], [0, dy[-1]], [0, h[-1]], color=c, lw=1.0, ls="--")
        ax.scatter(np.zeros_like(dy), dy, h, color=c, s=14, depthshade=False,
                   label=f"{param(x['slug'], 'a'):g}\N{DEGREE SIGN}")
    A_, Y_ = np.meshgrid(az, [0, 2000])
    ax.plot_surface(A_, Y_, np.zeros_like(A_), color="0.6", alpha=0.15, linewidth=0, shade=False)
    ax.set_xlabel("azimuth [m]", fontsize=8, labelpad=2); ax.set_ylabel("ground range $\\Delta y$ [m]", fontsize=8, labelpad=4)
    ax.set_zlabel("height [m]", fontsize=8, labelpad=2); ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect((1.5, 1.3, 0.8), zoom=1.35)
    ax.locator_params(axis="x", nbins=5); ax.locator_params(axis="y", nbins=5); ax.locator_params(axis="z", nbins=5)
    ax.set_title(f"cross-track ramps in 3D, N = {N}", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", handlelength=1)


def azramp3d(ax, rows, N=9):
    """Uniform ramps in 3D: targets along azimuth (200 m apart) on the iso-range surface, so higher targets
    sit further out in ground range; each ramp is drawn as a ribbon along its targets."""
    import sata1d as S
    base = S.make_config([])
    r0, H, y0 = base.scene.r0, base.scene.H, base.scene.y0
    r = sorted([x for x in rows if int(x["N"]) == N], key=lambda x: param(x["slug"], "a"))
    cols = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(r)))
    for x, c in zip(r, cols):
        h = np.array([float(v) for v in x["heights"].split()])
        xs = (np.arange(len(h)) - (len(h) - 1) / 2.0) * 200.0
        dy = np.sqrt(r0 ** 2 - (H - h) ** 2) - y0            # iso-range: higher targets sit further out in range
        X_ = np.tile(xs, (2, 1)); YY = np.vstack([dy - 400.0, dy + 400.0])     # ribbon along the ramp
        ax.plot_surface(X_, YY, np.tile(h, (2, 1)), color=c, alpha=0.22, linewidth=0, shade=False)
        ax.plot(xs, dy, h, "-", color=c, lw=1.2)
        ax.scatter(xs, dy, h, color=c, s=12, depthshade=False, label=f"{param(x['slug'], 'a'):g}\N{DEGREE SIGN}")
    X_, YY = np.meshgrid([-1000, 1000], [-600, 5800])
    ax.plot_surface(X_, YY, np.zeros_like(X_, dtype=float), color="0.6", alpha=0.10, linewidth=0, shade=False)
    ax.set_xlabel("azimuth [m]", fontsize=8, labelpad=4); ax.set_ylabel("ground range $\\Delta y$ [m]", fontsize=8, labelpad=4)
    ax.set_zlabel("height [m]", fontsize=8, labelpad=2); ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=20, azim=-35)
    ax.set_box_aspect((1.5, 1.3, 0.8), zoom=1.35)
    ax.locator_params(axis="x", nbins=5); ax.locator_params(axis="y", nbins=5); ax.locator_params(axis="z", nbins=5)
    ax.set_title(f"uniform ramps in 3D, N = {N}", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", handlelength=1)


def topo_ramps(rows, out_png):
    r = [x for x in rows if x["family"] == "topo_ramp"]
    if not r:
        return
    fig = plt.figure(figsize=(19, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.35, wspace=0.18)
    top = gs[0, :].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=0.05)
    profiles(fig.add_subplot(top[0, 0]), r, "a", 9, "cross-track ramp", lambda p: f"{p:g}\N{DEGREE SIGN}", topo=True)
    ramp3d(fig.add_subplot(top[0, 1], projection="3d"), r)
    axs = [fig.add_subplot(gs[1, j]) for j in range(2)]
    for ax, (metric, lab) in zip(axs, VARIANTS):
        Ns, Ps, M = grid(r, "a", metric)
        im = draw(ax, Ns, Ps, M, r"ramp angle $\alpha$ [deg]", fmt, fs=8)
        ax.set_title(f"{lab}: median {np.nanmedian(M):.1f} dB, min {np.nanmin(M):.1f}, max {np.nanmax(M):.1f}",
                     fontsize=10)
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01)
    cb.set_label("improvement over no SATA [dB]")
    fig.suptitle(r"Cross-track ramps: improvement = worst ambiguity (no SATA) $-$ worst ambiguity (SATA)  "
                 r"[dB, > 0: SATA better] | $\Delta h = \Delta y\tan\alpha$, targets at azimuth 0, "
                 r"$b_{xt}\sim U(0,100)$ m", fontsize=11, y=0.97)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*", default=["geometry_sweep/sweep_results.csv"])
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    rows = load(a.csv)
    os.makedirs(a.out, exist_ok=True)
    topo_ramps(rows, os.path.join(a.out, "improvement_heatmap_topo_ramps.png"))
    if any(r["family"] == "ramp" for r in rows):
        ramps(rows, os.path.join(a.out, "improvement_heatmap_ramps.png"))
    if any(r["family"] == "zigzag" for r in rows):
        geometries(rows, os.path.join(a.out, "improvement_heatmap_geometries.png"))
        geometries(rows, os.path.join(a.out, "improvement_heatmap_geometries_subband.png"),
                   "amb_db_sub", "SATA per sub-band")
    print(f"saved heatmaps to {a.out}")


if __name__ == "__main__":
    main()
