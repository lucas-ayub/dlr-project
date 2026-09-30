"""
Heatmaps and IRF pages of sweep_v4.
Usage: python plot_v4.py [--cases]
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import plot_family_report as P
import sweep_v4 as V
from sweep_geometry_plots import geometry_2d, geometry_3d, coordinates_text

OUT = "report_v4"
DIST_TXT = {"dS": "targets 200 m apart", "dN": "targets 2.6 km apart", "dL": "targets 10 km apart"}
H_TXT = {"hS": "small heights", "hN": "normal heights", "hL": "large heights"}
STEP = {"hS": 20, "hN": 200, "hL": 2600}
REF = "Reference"
NONE = "no SATA"
WHOLE = "SATA whole band"
SUB = "SATA per sub-band"
FOOT = ("Highest ambiguity of the image, in dB relative to the reference peak. "
        "$N_{rx}$ = 4, PRF = 2000 Hz, $B_a$ = 1235 Hz, $b_{xt}$ = [63.7, 27.0, 4.1, 1.7] m.")
_a = lambda sl: float(sl.split("_a")[1])
_amax = lambda sl: int(sl.split("amax")[1])
_dh = lambda sl: int(sl.split("_dh")[1])
P.FAM.update({
    "zigzag": dict(name="zig-zag ramp", plural="zig-zag ramps", col=_a, fmt=lambda p: f"{p:g}"),
    "steepening": dict(name="steepening ramp", plural="steepening ramps", col=_amax, fmt=lambda p: f"{p}"),
    "flattening": dict(name="flattening ramp", plural="flattening ramps", col=_amax, fmt=lambda p: f"{p}"),
    "cliff": dict(name="cliff", plural="cliffs", col=_dh, fmt=lambda p: f"{p}"),
    "spike": dict(name="spike", plural="spikes", col=_dh, fmt=lambda p: f"{p}"),
    "mesa": dict(name="mesa", plural="mesas", col=_dh, fmt=lambda p: f"{p}"),
})
NAME = {"ramp": "Along-track ramp", "random_heights": "Random heights", "piecewise": "Piecewise ramp",
        "zigzag": "Zig-zag ramp", "steepening": "Steepening ramp", "flattening": "Flattening ramp",
        "cliff": "Cliff", "spike": "Spike", "mesa": "Mesa"}


ALPHAS = [0.1, 1, 1.5, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]


def column(r, fam, hk):
    if fam == "random_heights":
        return int(r["slug"].split("hmax")[1].split("_")[0])
    if fam == "piecewise":
        h = np.array([float(x) for x in r["heights"].split()])
        a = np.degrees(np.arctan(np.abs(np.diff(h)).max() / (200.0 * V.HSCALE[hk])))
        return min(ALPHAS, key=lambda x: abs(x - a))
    return P.FAM[fam]["col"](r["slug"])


def grid2(rows, fam, hk, fn, worst):
    Ns = sorted({int(r["N"]) for r in rows}); Ps = sorted({column(r, fam, hk) for r in rows})
    cells = {}
    for r in rows:
        cells.setdefault((Ns.index(int(r["N"])), Ps.index(column(r, fam, hk))), []).append(fn(r))
    M = np.full((len(Ns), len(Ps)), np.nan)
    for (i, j), v in cells.items():
        M[i, j] = worst(v)
    return Ns, Ps, M, max(len(v) for v in cells.values())


def split(tag):
    pre, dk, hk = tag.split("_")
    return {v: k for k, v in V.FAMS.items()}[pre], dk, hk


def height_rule(fam, hk):
    s = V.HSCALE[hk]
    if fam == "ramp":
        return f"target j at height j$\\cdot${STEP[hk]} m$\\cdot$tan$\\alpha$"
    if fam == "random_heights":
        return f"heights drawn uniformly between 0 and {100 * s:g}, {500 * s:g} or {1000 * s:g} m"
    if fam in ("piecewise", "zigzag", "steepening", "flattening"):
        return f"height changes over steps of {STEP[hk]} m"
    return f"height jumps x{s:g}"


def rows_of(tag):
    return [r for r in csv.DictReader(open(os.path.join(V.OUT, "results.csv"))) if r["config"] == tag]


def setup_fam(fam, hk):
    s = V.HSCALE[hk]
    if fam == "random_heights":
        P.FAM[fam]["fmt"] = lambda p: f"{p * s:g}"
        return "maximum height $h_{max}$ [m] (heights drawn uniformly between 0 and $h_{max}$)"
    if fam == "piecewise":
        P.FAM[fam]["fmt"] = lambda p: f"{p:g}"
        return "largest slope between neighbouring targets [deg]"
    if fam == "zigzag":
        return "slope $\\pm\\alpha$ [deg], alternating"
    if fam in ("steepening", "flattening"):
        return "largest slope $\\alpha_{max}$ [deg]"
    if fam in ("cliff", "spike", "mesa"):
        P.FAM[fam]["fmt"] = lambda p: f"{p * s:g}"
        return "height jump $\\Delta h$ [m]"
    return f"ramp slope $\\alpha$ [deg]   (target j at height j$\\cdot${STEP[hk]} m$\\cdot$tan$\\alpha$)"


def head(tag):
    fam, dk, hk = split(tag)
    return f"{NAME[fam]}, {DIST_TXT[dk]}, {H_TXT[hk]} ({height_rule(fam, hk)})"


def _foot(fig, y=-0.02):
    fig.text(0.5, y, FOOT, ha="center", va="top", fontsize=9, color="0.35")


def heatmap_improvement(tag, out_png):
    fam, dk, hk = split(tag); rows = rows_of(tag); xl = setup_fam(fam, hk)
    f = lambda r, k: float(r[f"ambpk_db_{k}"])
    fig, axs = P._fig_stack(2, len({r["N"] for r in rows}))
    for ax, (k, lab) in zip(axs, (("whole", "SATA whole band"), ("sub", "SATA per sub-band"))):
        Ns, Ps, M, nmax = grid2(rows, fam, hk, lambda r: f(r, "none") - f(r, k), min)
        im = P.draw(ax, fam, Ns, Ps, M, P.CMAP, -40, 40,
                    f"{lab}: improvement over no SATA [dB]  (median {np.nanmedian(M):.1f}, min {np.nanmin(M):.1f}, "
                    f"max {np.nanmax(M):.1f})".replace("-0.0", "0.0"), lambda v: abs(v) > 17, xl)
        ax.set_ylabel("number of targets N")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("improvement [dB]")
    several = " | cells with more than one case: smallest improvement" if nmax > 1 else ""
    fig.suptitle(f"{head(tag)}\nimprovement = highest ambiguity (no SATA) $-$ highest ambiguity (SATA){several}",
                 fontsize=12, y=1 - 0.05 / fig.get_figheight())
    _foot(fig)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def heatmap_level(tag, out_png):
    fam, dk, hk = split(tag); rows = rows_of(tag); xl = setup_fam(fam, hk)
    f = lambda r, k: float(r[f"ambpk_db_{k}"])
    fig, axs = P._fig_stack(3, len({r["N"] for r in rows}))
    for ax, (k, lab) in zip(axs, (("none", "no SATA"), ("whole", "SATA whole band"),
                                  ("sub", "SATA per sub-band"))):
        Ns, Ps, M, nmax = grid2(rows, fam, hk, lambda r: f(r, k), max)
        im = P.draw(ax, fam, Ns, Ps, M, "viridis_r", -65, -5,
                    f"{lab}: highest ambiguity [dB]  (median {np.nanmedian(M):.1f})", lambda v: v > -32, xl)
        ax.set_ylabel("number of targets N")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("highest ambiguity [dB]")
    several = " | cells with more than one case: highest value" if nmax > 1 else ""
    fig.suptitle(f"{head(tag)}\nhighest ambiguity [dB re. reference peak]{several}",
                 fontsize=12, y=1 - 0.05 / fig.get_figheight())
    _foot(fig)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def case_desc(r, fam, hk):
    slug = r["slug"]
    if fam == "ramp":
        return f"slope $\\alpha$ = {slug.split('_a')[1]}$^\\circ$ (target j at height j$\\cdot${STEP[hk]} m$\\cdot$tan$\\alpha$)"
    if fam == "random_heights":
        hm, sd = slug.split("hmax")[1].split("_seed")
        return f"heights drawn uniformly between 0 and {float(hm) * V.HSCALE[hk]:g} m (draw {sd})"
    if fam == "piecewise":
        return f"random slopes between neighbours, steps of {STEP[hk]} m (draw {slug.split('seed')[1]})"
    if fam == "zigzag":
        return f"slopes alternating +/-{slug.split('_a')[1]}$^\\circ$, steps of {STEP[hk]} m"
    if fam in ("steepening", "flattening"):
        am = slug.split("amax")[1]
        rng = f"0.1$^\\circ$ to {am}$^\\circ$" if fam == "steepening" else f"{am}$^\\circ$ to 0.1$^\\circ$"
        return f"slope changing from {rng}, steps of {STEP[hk]} m"
    dh = _dh(slug) * V.HSCALE[hk]
    return {"cliff": "second half raised by", "spike": "middle target raised by",
            "mesa": "all but the outer targets raised by"}[fam] + f" {dh:g} m"


def case_page(r, out_png):
    fam, dk, hk = split(r["config"])
    z = np.load(r["npz"])
    t = z["t"]; db = {k: 20 * np.log10(z[f"env_{k}"] + 1e-12) for k in ("ref", "none", "whole", "sub")}
    a = {k: float(r[f"ambpk_db_{k}"]) for k in ("ref", "none", "whole", "sub")}
    xs, dys, hs = z["xs"], z["dys"], z["hs"]
    fig = plt.figure(figsize=(17, 9.6))
    gs = GridSpec(2, 2, width_ratios=[1.75, 1.0], hspace=0.36, wspace=0.14, figure=fig)
    fig.suptitle(f"{r['case_id']}: {NAME[fam]}, {r['N']} targets {DIST_TXT[dk][8:]}, {H_TXT[hk]}\n{case_desc(r, fam, hk)}",
                 fontsize=12, y=0.995)
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label=REF)
    ax.plot(t, db["whole"], color="tab:blue", lw=1.6, ls=(0, (5, 3)), label=WHOLE)
    ax.plot(t, db["sub"], color="tab:orange", lw=1.8, ls=(0, (1, 2)), label=SUB)
    ax.set_title(f"highest ambiguity: reference {a['ref']:.1f} dB | SATA whole band {a['whole']:.1f} dB | "
                 f"SATA per sub-band {a['sub']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label=REF)
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label=NONE)
    az.set_title(f"highest ambiguity: reference {a['ref']:.1f} dB | no SATA {a['none']:.1f} dB | improvement: "
                 f"whole band {a['none'] - a['whole']:.1f} dB, per sub-band {a['none'] - a['sub']:.1f} dB", fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]")
        a_.legend(fontsize=9, loc="upper left")
    geometry_2d(fig.add_subplot(gs[0, 1]), xs, dys, hs, False,
                title=f"Target geometry | $\\Delta h_{{max}}$ = {np.ptp(hs):.0f} m")
    a3 = fig.add_subplot(gs[1, 1], projection="3d")
    geometry_3d(a3, xs, dys, hs, False, title="Target geometry in 3D (x, $\\Delta y$, h)")
    a3.text2D(0.99, 0.02, coordinates_text(xs, dys, hs), transform=a3.transAxes, fontsize=5.8, family="monospace",
              ha="right", va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig.text(0.5, 0.015, FOOT, ha="center", va="bottom", fontsize=9, color="0.35")
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(out_png, dpi=72, bbox_inches="tight"); plt.close(fig)


def tags(pres=None):
    return [f"{pre}_{dk}_{hk}" for pre in V.FAMS.values() if not pres or pre in pres for dk in V.DIST for hk in V.HSCALE]


def _cases(tag):
    d = os.path.join(OUT, "cases", tag); os.makedirs(d, exist_ok=True)
    for r in rows_of(tag):
        case_page(r, os.path.join(d, os.path.basename(r["npz"])[:-4] + ".png"))
    return tag


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", action="store_true"); ap.add_argument("--pre", nargs="*")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for tag in tags(a.pre):
        heatmap_improvement(tag, os.path.join(OUT, f"heatmap_improvement_{tag}.png"))
        heatmap_level(tag, os.path.join(OUT, f"heatmap_level_{tag}.png"))
    print("heatmaps done", flush=True)
    if a.cases:
        from multiprocessing import Pool
        with Pool(2) as pool:
            for t in pool.imap_unordered(_cases, tags(a.pre)):
                print(t, flush=True)
