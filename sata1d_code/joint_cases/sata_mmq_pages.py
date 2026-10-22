"""
PDF of the test of sata_mmq.py (no file of the pipeline nor sata_mmq.py is changed): SATA whole band and SATA per
sub-band followed by the 4x4 inversion of the standard reconstruction, against the same SATA followed by the solution
that the joint uses (regularised least squares, mu = 0.03 sigma_max, two passes).

    page 1      the 86 cases of sweep_final (numbers of sata_mmq/numbers): inversion against regularised least squares,
                for no SATA, SATA whole band and SATA per sub-band, and every method against the reference
    pages 2-11  the cases of sata_mmq.PAGES, with the scene (2D zoom on the targets, 3D over the whole line) and the DEM

    python joint_cases/sata_mmq_pages.py          # from sata1d_code
Output: joint_cases/sata_mmq/pages_v2/*.png, joint_cases/sata_mmq/sata_mmq_v2.pdf
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import sata_mmq as MM
import audit_dem as AD
import run_joint_sweep as W
from sata1d.joint_line import line_positions
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy
from sweep_joint_all import bundle

OUT = os.path.join(MM.OUT, "pages_v2")
CH = AD.CH
NAMES = {"ref": "Reference",
         "none": "no SATA, matrix inversion", "none_mmq": "no SATA, regularised least squares",
         "whole": "SATA whole band, matrix inversion", "whole_mmq": "SATA whole band, regularised least squares (as the joint)",
         "sub": "SATA per sub-band, matrix inversion", "sub_mmq": "SATA per sub-band, regularised least squares (as the joint)",
         "joint_line": "joint, final version"}


def summary(out_png):
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(MM.OUT, "numbers", "*.json")))]
    fig = plt.figure(figsize=(20, 9.5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.25, left=0.05, right=0.985, top=0.86, bottom=0.15, figure=fig)
    lo, hi = -75, 0
    for k, (a, b, col, title) in enumerate((("whole", "whole_mmq", "tab:blue", "SATA whole band (DEM)"),
                                            ("sub", "sub_mmq", "tab:orange", "SATA per sub-band (DEM)"))):
        ax = fig.add_subplot(gs[0, k])
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=1, ls="--", label="same value")
        ax.scatter([r["none"] for r in R], [r["none_mmq"] for r in R], s=18, color="tab:red", alpha=0.55, label="no SATA")
        ax.scatter([r[a] for r in R], [r[b] for r in R], s=22, color=col, edgecolor="k", lw=0.4, label=title)
        d = np.array([r[b] - r[a] for r in R])
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("highest ambiguity, after the matrix inversion of the standard reconstruction [dB]")
        ax.set_ylabel("highest ambiguity, after regularised least squares as in the joint [dB]")
        ax.set_title(f"{title}: inversion vs. regularised least squares\ndifference: median {np.median(d):+.2f} dB, "
                     f"from {d.min():+.2f} to {d.max():+.2f} dB ({len(R)} cases)", fontsize=10.5)
        ax.legend(fontsize=9, loc="upper left")
    ax = fig.add_subplot(gs[0, 2])
    keys = ["none", "none_mmq", "whole", "whole_mmq", "sub", "sub_mmq", "joint_line", "ref"]
    cols = ["tab:red", "tab:red", "tab:blue", "tab:cyan", "tab:orange", "tab:brown", "tab:purple", "tab:green"]
    rng = np.random.default_rng(0)
    for i, (k, c) in enumerate(zip(keys, cols)):
        v = np.array([r[k] for r in R])
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.18, 0.18, len(v)), v, s=12, color=c, alpha=0.6)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="k", lw=2)
        ax.text(i, 3, f"{np.median(v):.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(keys)))
    short = {"none": "no SATA\ninversion", "none_mmq": "no SATA\nreg. LS", "whole": "SATA whole\nband\ninversion",
             "whole_mmq": "SATA whole\nband\nreg. LS", "sub": "SATA per\nsub-band\ninversion", "sub_mmq": "SATA per\nsub-band\nreg. LS",
             "joint_line": "joint,\nfinal\nversion", "ref": "Reference"}
    ax.set_xticklabels([short[k] for k in keys], fontsize=9)
    ax.set_ylim(-80, 7); ax.grid(alpha=0.3, axis="y"); ax.set_ylabel("highest ambiguity [dB]")
    ax.set_title("Every case, every method (black line and number: median)\ninversion = matrix inversion of the standard reconstruction\n"
                 "reg. LS = regularised least squares as in the joint", fontsize=10.5)
    fig.suptitle("Does the regularisation of the joint help the SATA? The same SATA-corrected channels, inverted as in the standard "
                 "reconstruction or solved by regularised least squares\n($\\mu$ = 0.03 $\\sigma_{max}$, two passes: the solver of the joint). "
                 "86 cases: ramps, sines / cosines and realistic terrains, 1 to 49 targets, 2 to 8 channels; every method reads only the DEM",
                 fontsize=12)
    fig.text(0.5, 0.015, "With no SATA, the regularised least squares give the joint itself with a flat DEM (checked to the iteration tolerance).\n"
             "So the only difference between 'SATA + regularised least squares' and the joint is where the phase of the DEM is applied: "
             "to each channel before the reconstruction (SATA) or inside the model, at every position of the line (joint).",
             ha="center", fontsize=9, color="0.35")
    fig.savefig(out_png, dpi=85); plt.close(fig)


def case_page(sc, tag, out_png):
    res, d = MM.compute(sc, tag)
    c, cfg, tr, ch, Mch, Mline, kind = MM.setup(sc, tag)
    env, t, pk, amb = d["env"], d["t"], d["pk"], d["amb"]
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    fig = plt.figure(figsize=(20, 12.5))
    gs = GridSpec(3, 2, width_ratios=[1.55, 1.0], hspace=0.42, wspace=0.13, left=0.045, right=0.985, top=0.9, bottom=0.07, figure=fig)
    lab = c["label"] if c["label"].startswith("h =") else c["label"][0].upper() + c["label"][1:]
    fig.suptitle(f"{lab}\n$N_{{rx}}$ = {cfg.Nrx}, $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)} ({kind}) | the same SATA-corrected channels, "
                 "matrix inversion vs. regularised least squares as in the joint", fontsize=12.5)
    sty = {"ref": dict(color="tab:green", lw=2.0), "whole": dict(color="tab:blue", lw=1.3, ls=(0, (5, 3))),
           "whole_mmq": dict(color="tab:cyan", lw=1.0), "sub": dict(color="tab:orange", lw=1.3, ls=(0, (5, 3))),
           "sub_mmq": dict(color="tab:brown", lw=1.0), "joint_line": dict(color="tab:purple", lw=1.0)}
    for row, keys in enumerate((("ref", "whole", "whole_mmq"), ("ref", "sub", "sub_mmq"), ("ref", "joint_line"))):
        ax = fig.add_subplot(gs[row, 0])
        for k in keys:
            ax.plot(t, db[k], label=NAMES[k], **sty[k])
        ax.set_title("highest ambiguity: " + " | ".join(f"{NAMES[k]} {amb[k]:.1f} dB" for k in keys), fontsize=9.5)
        ax.set_ylim(-90, 5); ax.set_xlim(t[0], t[-1]); ax.grid(alpha=0.3); ax.set_ylabel("[dB]"); ax.legend(fontsize=8.5, loc="upper left")
        ax.set_xlabel("Azimuth time [s]")
    # the scene: 2D zoom on the targets and the reference point, 3D over the whole line
    xs_, dys, hs_ = scene_coordinates(xs, hs)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    g2 = fig.add_subplot(gs[0, 1])
    lo2, hi2 = min(xs_.min(), 0.0), max(xs_.max(), 0.0); pad2 = max(0.12 * (hi2 - lo2), 500.0)
    xg = np.linspace(-half, half, 9601); hg = terrain(xg)
    hz = hg[(xg >= lo2 - pad2) & (xg <= hi2 + pad2)]
    geometry_2d(g2, xs_, dys, hs_, False, title=None, connect=False, annotate=len(xs_) <= 13)
    g2.fill_between(xg, min(hg.min(), 0) - 0.05 * max(np.ptp(hg), 1.0), hg, color="#D7C9A7", alpha=0.45, lw=0)
    g2.plot(xg, hg, color="#6D4C2F", lw=1.2, zorder=2, label="terrain (the scene)")
    ylo, yhi = min(hz.min(), 0.0), max(hz.max(), hs_.max(), 1.0)
    g2.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))
    g2.plot(0.0, 0.0, marker="*", ms=13, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $h$ = 0)")
    g2.set_xlim(lo2 - pad2, hi2 + pad2)
    g2.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, borderaxespad=0.1)
    g2.set_title("Zoom on the point targets and the reference point", fontsize=9.5, pad=30)
    g2.xaxis.set_major_formatter(km); g2.set_xlabel("azimuth position $x$ [km]")
    g3 = fig.add_subplot(gs[1, 1], projection="3d")
    geometry_3d(g3, xs_, dys, hs_, False, connect=False, title=None)
    g3.plot(xg, iso_range_dy(hg), hg, color="#6D4C2F", lw=1.0)
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=6)
    g3.set_xlim(-half, half); g3.xaxis.set_major_formatter(km); g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4)
    g3.set_zlim(min(0.0, hg.min()), max(hg.max(), 50.0) * 1.15)
    g3.set_title(f"Whole azimuth line ({2 * half / 1e3:.0f} km) in 3D (x, $\\Delta y$, h)", fontsize=9.5)
    for t_ in list(g3.texts):
        t_.remove()
    am = fig.add_subplot(gs[2, 1])
    xl = line_positions(cfg)
    for i in range(cfg.Nrx):
        am.plot(xl, 1e3 * Mline[i], color=CH[i], lw=1.2, label=f"channel {i} ($b_{{xt}}$ = {cfg.array.bxt[i]:g} m)")
    if len(xs) <= 60:
        for x in xs:
            am.axvline(x, color="0.75", lw=0.6, zorder=0)
    am.set_xlim(xl[0], xl[-1]); am.xaxis.set_major_formatter(km); am.grid(alpha=0.3)
    am.set_xlabel("azimuth position $x$ [km]"); am.set_ylabel("$\\Delta C_0$ from the DEM [mm]")
    lo_, hi_ = am.get_ylim(); am.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    am.set_title("$\\Delta C_0$ of every channel from the DEM (the same for every method; grey lines: the targets)", fontsize=9.5)
    am.legend(fontsize=7.5 if cfg.Nrx <= 4 else 6.5, loc="upper center", ncol=2 if cfg.Nrx <= 4 else 3)
    fig.text(0.5, 0.012, f"Highest ambiguity in dB relative to the reference peak (same measure as the sweeps). No SATA: matrix inversion "
             f"{amb['none']:.1f} dB, regularised least squares {amb['none_mmq']:.1f} dB. Every method reads only the DEM. "
             f"Channel 0 is the transmitter; PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz.", ha="center", fontsize=9, color="0.35")
    fig.savefig(out_png, dpi=80); plt.close(fig)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    pngs = [os.path.join(OUT, "00_summary.png")]
    summary(pngs[0])
    outline = [("The 86 cases", 0, 0)]
    for k, (sc, tag) in enumerate(MM.PAGES):
        p = os.path.join(OUT, f"{k + 1:02d}_{sc}__{tag}.png")
        if not os.path.exists(p):
            r = case_page(sc, tag, p)
            print(sc, tag, f"whole {r['whole']:.1f} -> {r['whole_mmq']:.1f}  sub {r['sub']:.1f} -> {r['sub_mmq']:.1f}  joint {r['joint_line']:.1f}", flush=True)
        outline.append((f"{sc} | {tag}", len(pngs), 0))
        pngs.append(p)
    print("wrote", bundle(pngs, os.path.join(MM.OUT, "sata_mmq_v2.pdf"), outline))


if __name__ == "__main__":
    main()
