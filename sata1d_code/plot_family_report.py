"""
Figures of the SATA 1D family reports (close vs separated targets), from the sweep CSVs and the
saved envelopes (.npz):
  heatmaps_<family>_<tag>.png     improvement over no SATA (whole band, per sub-band) and the ambiguity peak
  metric_explained.png            where the old and the new ambiguity metric read the IRF
  cases/<family>/<tag>/<case>.png one page per case: IRFs with the ambiguity windows, geometry 2D + 3D
Metric: ambiguity peak = largest |IRF| within +-20 ms of x_j/vs + m dt_amb (m = +-1, +-2) over all
targets j, outside +-20 ms of every main lobe, in dB re. the reference peak; dt_amb = 0.810 s.
Usage: python plot_family_report.py <out_dir> [family ...]
"""
import csv
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from sweep_geometry_plots import geometry_2d, geometry_3d, coordinates_text

CMAP = LinearSegmentedColormap.from_list("improv", ["#B2182B", "#E8A48C", "#E4E4E4", "#8FB8DB", "#1F5FA8", "#0B2E5C"])
DT_AMB = 0.8101
VS = 7688.53706432
BXT = "[63.7, 27.0, 4.1, 1.7]"
H200 = r"(heights $h_j = j\cdot 200\,\mathrm{m}\cdot\tan\alpha$)"
H2600 = r"(heights $h_j = j\cdot 2600\,\mathrm{m}\cdot\tan\alpha$)"
CONFIGS = {
    "ramp": [("close", "geometry_sweep_v2", "targets 200 m apart, normal heights", 200),
             ("separated", "geometry_sweep_separated", "targets 2.6 km apart, normal heights", 200),
             ("largeh_close", "geometry_sweep_ramp_largeh_close", "targets 200 m apart, large heights", 2600),
             ("largeh_separated", "geometry_sweep_ramp_largeh_separated", "targets 2.6 km apart, large heights", 2600)],
    "random_heights": [("close", "geometry_sweep_v2", "targets 200 m apart", None),
                       ("separated", "geometry_sweep_separated", "targets 2.6 km apart (same heights)", None)],
    "piecewise": [("close", "geometry_sweep_v2", "targets 200 m apart", None),
                  ("separated", "geometry_sweep_separated", "targets 2.6 km apart (same heights)", None)],
}


def ramp_xlabel(step):
    return (r"ramp inclination $\alpha$ [deg] " + (H200 if step == 200 else H2600)) if step else None
FAM = {
    "ramp": dict(name="along-track ramp", plural="along-track ramps", col=lambda s: float(s.split("_a")[1]), fmt=lambda p: f"{p:g}",
                 xlabel=r"ramp inclination $\alpha$ [deg] (heights $h_j = j\cdot 200\,\mathrm{m}\cdot\tan\alpha$)",
                 desc=lambda r, step=200: f"$\\alpha$ = {r['slug'].split('_a')[1]}$^\\circ$ (heights $j\\cdot{step}\\tan\\alpha$)"),
    "random_heights": dict(name="random heights", plural="random heights",
                           col=lambda s: tuple(int(v) for v in re.search(r"hmax(\d+)_seed(\d+)", s).groups()),
                           fmt=lambda p: f"{p[0]}/{p[1]}",
                           xlabel=r"$h_{max}$ [m] / seed (heights $\sim U(0, h_{max})$, first target at 0)",
                           desc=lambda r, step=None: "heights $\\sim U(0, %s)$ m, seed %s" % re.search(r"hmax(\d+)_seed(\d+)", r["slug"]).groups()),
    "piecewise": dict(name="piecewise ramp", plural="piecewise ramps", col=lambda s: int(s.split("seed")[1]), fmt=lambda p: f"{p}",
                      xlabel="seed (slopes drawn from {0.1 ... 50} deg with random sign, 200-m steps)",
                      desc=lambda r, step=None: f"random slopes, seed {r['slug'].split('seed')[1]}"),
}


def load(sweep_dir, fam):
    return [r for r in csv.DictReader(open(os.path.join(sweep_dir, "sweep_results.csv")))
            if r["family"] == fam and float(r["bxt_max"]) > 0]


def grid(rows, fam, fn):
    c = FAM[fam]["col"]
    Ns = sorted({int(r["N"]) for r in rows}); Ps = sorted({c(r["slug"]) for r in rows})
    M = np.full((len(Ns), len(Ps)), np.nan)
    for r in rows:
        M[Ns.index(int(r["N"])), Ps.index(c(r["slug"]))] = fn(r)
    return Ns, Ps, M


def draw(ax, fam, Ns, Ps, M, cmap, vmin, vmax, title, dark, xlabel=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(Ps)), [FAM[fam]["fmt"](p) for p in Ps], fontsize=9)
    ax.set_yticks(range(len(Ns)), Ns, fontsize=9)
    ax.set_xlabel(xlabel or FAM[fam]["xlabel"], fontsize=10); ax.set_ylabel("N targets", fontsize=10)
    ax.set_xticks(np.arange(-.5, len(Ps)), minor=True); ax.set_yticks(np.arange(-.5, len(Ns)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    for i in range(len(Ns)):
        for j in range(len(Ps)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}".replace("-0.0", "0.0"), ha="center", va="center", fontsize=9,
                        color="white" if dark(M[i, j]) else "#222222")
    ax.set_title(title, fontsize=11)
    return im


def _fig_stack(n, rowsN):
    fig, axs = plt.subplots(n, 1, figsize=(15, n * (1.3 + 0.62 * rowsN) + 0.5),
                            gridspec_kw=dict(hspace=0.45 if rowsN > 1 else 0.95))
    H = fig.get_figheight()
    fig.subplots_adjust(top=1 - 0.95 / H, bottom=0.55 / H)
    return fig, np.atleast_1d(axs)


def heatmap_improvement(rows, fam, label, out_png, xlabel=None):
    f = lambda r, k: float(r[f"ambpk_db_{k}"])
    rowsN = len({r["N"] for r in rows})
    fig, axs = _fig_stack(2, rowsN)
    for ax, (k, lab) in zip(axs, (("whole", "SATA whole band"), ("sub", "SATA per sub-band"))):
        Ns, Ps, M = grid(rows, fam, lambda r: f(r, "none") - f(r, k))
        im = draw(ax, fam, Ns, Ps, M, CMAP, -40, 40,
                  f"{lab}: improvement = no SATA $-$ SATA [dB] | median {np.nanmedian(M):.1f}, min {np.nanmin(M):.1f}, "
                  f"max {np.nanmax(M):.1f}".replace("-0.0", "0.0"), lambda v: abs(v) > 17, xlabel)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("improvement [dB]")
    fig.suptitle(f"SATA 1D, {FAM[fam]['plural']}, {label}\nimprovement over no SATA, ambiguity peak at $x_j/v \\pm m\\,0.81$ s "
                 f"($m$ = 1, 2) | $b_{{xt}}$ = {BXT} m", fontsize=12, y=1 - 0.05 / fig.get_figheight())
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def heatmap_peak(rows, fam, label, out_png, xlabel=None):
    f = lambda r, k: float(r[f"ambpk_db_{k}"])
    rowsN = len({r["N"] for r in rows})
    fig, axs = _fig_stack(3, rowsN)
    for ax, (k, lab) in zip(axs, (("none", "no SATA"), ("whole", "SATA whole band"), ("sub", "SATA per sub-band"))):
        Ns, Ps, M = grid(rows, fam, lambda r: f(r, k))
        im = draw(ax, fam, Ns, Ps, M, "viridis_r", -60, -5,
                  f"{lab}: ambiguity peak [dB re. reference peak] | median {np.nanmedian(M):.1f}", lambda v: v > -30, xlabel)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("ambiguity peak [dB]")
    fig.suptitle(f"SATA 1D, {FAM[fam]['plural']}, {label}\nambiguity peak at $x_j/v \\pm m\\,0.81$ s ($m$ = 1, 2) | "
                 f"$b_{{xt}}$ = {BXT} m", fontsize=12, y=1 - 0.05 / fig.get_figheight())
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def case_page(npz, row, fam, extra, out_png, step=None):
    z = np.load(npz, allow_pickle=True)
    t = z["t"]; pk = z["env_ref"].max()
    db = {k: 20 * np.log10(z[f"env_{k}"] / pk + 1e-12) for k in ("ref", "none", "whole", "sub")}
    a = {k: float(row[f"ambpk_db_{k}"]) for k in ("ref", "none", "whole", "sub")}
    xs, dys, hs = z["xs"], z["dys"], z["hs"]
    sp = np.diff(xs).min() if len(xs) > 1 else 0
    fig = plt.figure(figsize=(17, 9.2))
    gs = GridSpec(2, 2, width_ratios=[1.75, 1.0], hspace=0.36, wspace=0.14, figure=fig)
    fig.suptitle(f"{row['case_id']} | SATA 1D | {FAM[fam]['name']}: N = {row['N']}, {FAM[fam]['desc'](row, step) if step else FAM[fam]['desc'](row)}, "
                 f"targets {sp:.0f} m apart{extra}\n"
                 f"$b_{{xt}}$ = {BXT} m | $N_{{rx}}$ = 4 | PRF = 2000 Hz | $B_a$ = 1235 Hz | $\\Delta b_{{at}}$ = 7.69 m (DPCA)",
                 fontsize=11.5, y=0.995)
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["whole"], color="tab:blue", lw=1.6, ls=(0, (5, 3)), label="SATA whole band")
    ax.plot(t, db["sub"], color="tab:orange", lw=1.8, ls=(0, (1, 2)), label="SATA per sub-band")
    ax.set_title(f"ambiguity peak: ref {a['ref']:.1f} dB | SATA whole {a['whole']:.1f} dB | SATA sub {a['sub']:.1f} dB",
                 fontsize=9.5)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"ambiguity peak: ref {a['ref']:.1f} dB | no SATA {a['none']:.1f} dB | "
                 f"improvement whole {a['none'] - a['whole']:.1f} dB, sub {a['none'] - a['sub']:.1f} dB", fontsize=9.5)
    for a_ in (ax, az):
        for x in xs:
            for m in (-2, -1, 1, 2):
                a_.axvspan(x / VS + m * DT_AMB - 0.02, x / VS + m * DT_AMB + 0.02, color="0.8", alpha=0.5, lw=0, zorder=0)
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=10, loc="upper right")
    az.text(0.01, 0.03, "grey: ambiguity windows $x_j/v \\pm m\\,0.81$ s, $m$ = 1, 2", transform=az.transAxes, fontsize=8)
    geometry_2d(fig.add_subplot(gs[0, 1]), xs, dys, hs, False,
                title=f"Target geometry | $\\Delta h_{{max}}$ = {np.ptp(hs):.0f} m")
    a3 = fig.add_subplot(gs[1, 1], projection="3d")
    geometry_3d(a3, xs, dys, hs, False, title="Target geometry in 3D (x, $\\Delta y$, h)")
    a3.text2D(0.99, 0.02, coordinates_text(xs, dys, hs), transform=a3.transAxes, fontsize=5.8, family="monospace",
              ha="right", va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig.savefig(out_png, dpi=100, bbox_inches="tight"); plt.close(fig)


def metric_explained(npz, row, out_png):
    z = np.load(npz, allow_pickle=True)
    t = z["t"]; pk = z["env_ref"].max()
    db = {k: 20 * np.log10(z[f"env_{k}"] / pk + 1e-12) for k in ("ref", "none", "whole")}
    xs = z["xs"]; Na = len(t); ds = VS / 2000.0
    mask = np.ones(Na, bool)
    for x in xs:
        i = Na // 2 + int(round(x / ds)); mask[max(0, i - 40):i + 41] = False
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for ax, xl in ((axs[0], (t[0], t[-1])), (axs[1], (-1.05, 1.05))):
        ax.plot(t, db["ref"], color="tab:green", lw=1.8, label="Reference")
        ax.plot(t, db["none"], color="tab:red", lw=1.2, ls=(0, (5, 3)), label="no SATA")
        ax.plot(t, db["whole"], color="tab:blue", lw=1.2, label="SATA whole band")
        for k, col in (("none", "tab:red"), ("whole", "tab:blue")):
            i = np.flatnonzero(mask)[np.argmax(db[k][mask])]
            ax.plot(t[i], db[k][i], "v", ms=11, color=col, mec="k", zorder=5)
        for x in xs:
            for m in (-2, -1, 1, 2):
                ax.axvspan(x / VS + m * DT_AMB - 0.02, x / VS + m * DT_AMB + 0.02, color="0.75", alpha=0.6, lw=0, zorder=0)
        ax.set_xlim(*xl); ax.set_ylim(-90, 5); ax.grid(alpha=0.3); ax.set_xlabel("Azimuth time [s]"); ax.set_ylabel("[dB]")
    axs[0].legend(fontsize=9, loc="upper right")
    axs[0].set_title(f"{row['case_id']} (N = 2, $\\alpha$ = 5$^\\circ$, 200 m apart). Triangles: where the OLD metric reads "
                     f"(max outside $\\pm$40 samples of each target).\nGrey: where the NEW metric reads (ambiguity windows).", fontsize=9.5)
    axs[1].set_title(f"zoom | old: no SATA {float(row['amb_db_none']):.1f}, SATA {float(row['amb_db_whole']):.1f} dB "
                     f"(improvement {float(row['amb_db_none']) - float(row['amb_db_whole']):.1f})\n"
                     f"new: no SATA {float(row['ambpk_db_none']):.1f}, SATA {float(row['ambpk_db_whole']):.1f} dB "
                     f"(improvement {float(row['ambpk_db_none']) - float(row['ambpk_db_whole']):.1f})", fontsize=9.5)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main(out, fams, cases=True):
    os.makedirs(out, exist_ok=True)
    r7 = next(r for r in load("geometry_sweep_v2", "ramp") if r["slug"] == "N2_a5")
    metric_explained(glob.glob(f"geometry_sweep_v2/bxt100/ramp/{r7['case_id']}_*.npz")[0], r7,
                     os.path.join(out, "metric_explained.png"))
    key = lambda f: int(re.search(r"-(\d+)_", os.path.basename(f)).group(1))
    for fam in fams:
        for tag, d, label, step in CONFIGS[fam]:
            rows = load(d, fam)
            xl = ramp_xlabel(step)
            heatmap_improvement(rows, fam, label, os.path.join(out, f"heatmap_improvement_{fam}_{tag}.png"), xl)
            heatmap_peak(rows, fam, label, os.path.join(out, f"heatmap_peak_{fam}_{tag}.png"), xl)
            if not cases:
                continue
            od = os.path.join(out, "cases", fam, tag); os.makedirs(od, exist_ok=True)
            byid = {r["case_id"]: r for r in rows}
            extra = "" if tag == "close" else (" (same heights as the 200-m case)" if tag == "separated" and fam != "ramp" else "")
            for npz in sorted(glob.glob(os.path.join(d, "bxt100", fam, "*.npz")), key=key):
                cid = os.path.basename(npz).split("_")[0]
                case_page(npz, byid[cid], fam, extra, os.path.join(od, os.path.basename(npz)[:-4] + ".png"), step)
            print(fam, tag, len(rows), "cases", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "family_report", [f for f in sys.argv[2:] if f != "--no-cases"] or list(FAM), "--no-cases" not in sys.argv)
