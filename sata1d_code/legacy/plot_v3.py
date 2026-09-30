"""
Heatmaps and IRF pages of sweep_v3 (topographic ambiguity, reference = flat-terrain reconstruction).
Usage: python plot_v3.py [--cases CONFIG ...]
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

import plot_family_report as P
from sweep_geometry_plots import geometry_2d, geometry_3d, coordinates_text

OUT = "report_v3"
SRC = "sweep_v3"
HN = r"$h_j = j\cdot200\,\mathrm{m}\cdot\tan\alpha$"
HL = r"$h_j = j\cdot2600\,\mathrm{m}\cdot\tan\alpha$"
LABEL = {"A": ("ramp", "targets 200 m apart, normal heights", HN), "B": ("ramp", "targets 2.6 km apart, normal heights", HN),
         "C": ("ramp", "targets 200 m apart, large heights", HL), "D": ("ramp", "targets 2.6 km apart, large heights", HL),
         "E": ("ramp", "targets 5.2 km apart, normal heights", HN), "F": ("ramp", "targets 5.2 km apart, large heights", HL),
         "G": ("ramp", "targets 10 km apart, normal heights", HN), "H": ("ramp", "targets 10 km apart, large heights", HL)}
for sp in (200, 2600, 5200, 10000):
    txt = f"targets {sp} m apart" if sp == 200 else f"targets {sp / 1000:g} km apart (same heights as at 200 m)"
    LABEL[f"RH{sp}"] = ("random_heights", txt, None)
    LABEL[f"PW{sp}"] = ("piecewise", txt, None)


def rows_of(tag):
    return [r for r in csv.DictReader(open(os.path.join(SRC, "results.csv"))) if r["config"] == tag]


def xlabel(tag):
    fam, _, h = LABEL[tag]
    return rf"ramp inclination $\alpha$ [deg] (heights {h})" if fam == "ramp" else None


def title_head(tag):
    fam, txt, _ = LABEL[tag]
    return f"SATA 1D, {P.FAM[fam]['plural']}, configuration {tag}: {txt}" if fam == "ramp" else \
        f"SATA 1D, {P.FAM[fam]['plural']}, {txt}"


def heatmap_improvement(tag, out_png):
    fam = LABEL[tag][0]; rows = rows_of(tag)
    f = lambda r, k: float(r[k])
    rowsN = len({r["N"] for r in rows})
    fig, axs = P._fig_stack(2, rowsN)
    below = lambda r: f(r, "ambt_db_none") < f(r, "floor_db")
    for ax, (k, lab) in zip(axs, (("whole", "SATA whole band"), ("sub", "SATA per sub-band"))):
        Ns, Ps, M = P.grid(rows, fam, lambda r: f(r, "ambt_db_none") - f(r, f"ambt_db_{k}"))
        _, _, B = P.grid(rows, fam, lambda r: float(below(r)))
        im = P.draw(ax, fam, Ns, Ps, M, P.CMAP, -60, 60,
                    f"{lab}: improvement = no SATA $-$ SATA [dB] | median {np.nanmedian(M):.1f}, min {np.nanmin(M):.1f}, "
                    f"max {np.nanmax(M):.1f}", lambda v: abs(v) > 25, xlabel(tag))
        for i in range(len(Ns)):
            for j in range(len(Ps)):
                if B[i, j] > 0:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", ec="0.35", lw=0))
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("improvement [dB]")
    fig.suptitle(f"{title_head(tag)}\nimprovement of the topographic ambiguity (|s $-$ s$_{{flat}}$| at $x_j/v \\pm m\\,0.81$ s, "
                 f"$m$ = 1, 2) | hatched: without SATA already below the floor of the flat reconstruction",
                 fontsize=12, y=1 - 0.05 / fig.get_figheight())
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def heatmap_level(tag, out_png):
    fam = LABEL[tag][0]; rows = rows_of(tag)
    f = lambda r, k: float(r[k])
    rowsN = len({r["N"] for r in rows})
    fig, axs = P._fig_stack(3, rowsN)
    for ax, (k, lab) in zip(axs, (("none", "no SATA"), ("whole", "SATA whole band"), ("sub", "SATA per sub-band"))):
        Ns, Ps, M = P.grid(rows, fam, lambda r: f(r, f"ambt_db_{k}"))
        im = P.draw(ax, fam, Ns, Ps, M, "viridis_r", -90, -5,
                    f"{lab}: topographic ambiguity [dB re. flat peak] | median {np.nanmedian(M):.1f}",
                    lambda v: v > -45, xlabel(tag))
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("topographic ambiguity [dB]")
    fl = [f(r, "floor_db") for r in rows]
    fig.suptitle(f"{title_head(tag)}\ntopographic ambiguity |s $-$ s$_{{flat}}$| at $x_j/v \\pm m\\,0.81$ s ($m$ = 1, 2) | "
                 f"floor of the flat reconstruction in the same windows: {min(fl):.1f} to {max(fl):.1f} dB",
                 fontsize=12, y=1 - 0.05 / fig.get_figheight())
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def case_page(r, out_png):
    tag = r["config"]; fam, txt, _ = LABEL[tag]
    z = np.load(r["npz"])
    t = z["t"]; db = {k: 20 * np.log10(z[k] + 1e-12) for k in z.files if k.startswith(("env_", "err_"))}
    xs, dys, hs = z["xs"], z["dys"], z["hs"]
    a = {k: float(r[f"ambt_db_{k}"]) for k in ("none", "whole", "sub")}
    step = 2600 if tag in ("C", "D", "F", "H") else 200
    desc = P.FAM[fam]["desc"](r, step) if fam == "ramp" else P.FAM[fam]["desc"](r)
    fig = plt.figure(figsize=(17, 9.2))
    gs = GridSpec(2, 2, width_ratios=[1.75, 1.0], hspace=0.36, wspace=0.14, figure=fig)
    fig.suptitle(f"{r['case_id']} | SATA 1D | {P.FAM[fam]['name']}: N = {r['N']}, {desc}, {txt}\n"
                 f"$b_{{xt}}$ = {P.BXT} m | $N_{{rx}}$ = 4 | PRF = 2000 Hz | $B_a$ = 1235 Hz | $\\Delta b_{{at}}$ = 7.69 m (DPCA)",
                 fontsize=11.5, y=0.995)
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["env_flat"], color="tab:green", lw=2.0, label="flat-terrain reconstruction (reference)")
    ax.plot(t, db["env_whole"], color="tab:blue", lw=1.5, ls=(0, (5, 3)), label="SATA whole band")
    ax.plot(t, db["env_sub"], color="tab:orange", lw=1.6, ls=(0, (1, 2)), label="SATA per sub-band")
    ax.plot(t, db["err_whole"], color="k", lw=0.6, label="SATA whole band $-$ reference")
    ax.set_title(f"topographic ambiguity: SATA whole {a['whole']:.1f} dB | SATA sub {a['sub']:.1f} dB | "
                 f"floor of the reference {float(r['floor_db']):.1f} dB", fontsize=9.5)
    az.plot(t, db["env_flat"], color="tab:green", lw=2.0, label="flat-terrain reconstruction (reference)")
    az.plot(t, db["env_none"], color="tab:red", lw=1.5, ls=(0, (5, 3)), label="no SATA")
    az.plot(t, db["err_none"], color="k", lw=0.6, label="no SATA $-$ reference")
    az.set_title(f"topographic ambiguity: no SATA {a['none']:.1f} dB | improvement whole {a['none'] - a['whole']:.1f} dB, "
                 f"sub {a['none'] - a['sub']:.1f} dB", fontsize=9.5)
    for a_ in (ax, az):
        for x in xs:
            for m in (-2, -1, 1, 2):
                a_.axvspan(x / P.VS + m * P.DT_AMB - 0.02, x / P.VS + m * P.DT_AMB + 0.02, color="0.8", alpha=0.5, lw=0, zorder=0)
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-100, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB re. flat peak]"); a_.legend(fontsize=8.5, loc="upper right")
    geometry_2d(fig.add_subplot(gs[0, 1]), xs, dys, hs, False, title=f"Target geometry | $\\Delta h_{{max}}$ = {np.ptp(hs):.0f} m")
    a3 = fig.add_subplot(gs[1, 1], projection="3d")
    geometry_3d(a3, xs, dys, hs, False, title="Target geometry in 3D (x, $\\Delta y$, h)")
    a3.text2D(0.99, 0.02, coordinates_text(xs, dys, hs), transform=a3.transAxes, fontsize=5.8, family="monospace",
              ha="right", va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig.savefig(out_png, dpi=90, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", nargs="*", default=[]); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for tag in LABEL:
        heatmap_improvement(tag, os.path.join(OUT, f"heatmap_improvement_{tag}.png"))
        heatmap_level(tag, os.path.join(OUT, f"heatmap_level_{tag}.png"))
    print("heatmaps done", flush=True)
    for tag in a.cases:
        d = os.path.join(OUT, "cases", tag); os.makedirs(d, exist_ok=True)
        rows = sorted(rows_of(tag), key=lambda r: int(r["case_id"].split("-")[-1]))
        for r in rows:
            case_page(r, os.path.join(d, os.path.basename(r["npz"])[:-4] + ".png"))
        print(tag, len(rows), flush=True)
