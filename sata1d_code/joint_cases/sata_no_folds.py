"""
The same pipeline as the sweep pages (Nrx channels -> SATA on every channel -> standard reconstruction -> azimuth
compression), run twice on the same scene and the same cross-track baselines:

    PRF_op >= B_a   total PRF = Nrx x 2000 Hz, so every channel alone samples the Doppler band (no folds)
    PRF_op <  B_a   total PRF = 2000 Hz, the normal case of the sweeps (PRF_op = 2000 / Nrx, folds)

Nothing else changes: same scene, same b_xt, along-track baselines on the DPCA condition of each PRF, same DEM table
(channel_maps), same SATA calls (audit_dem.sata_whole / sata_sub), same reconstruction (S.reconstruct). Per page:
    left top      azimuth IRF without folds: reference, no SATA, SATA whole band, SATA per sub-band
    left bottom   azimuth IRF with folds (as in the sweeps): the same four
    middle top    scene, zoom on the targets | right top  scene, whole line in 3D
    middle bottom channels seen from above    | right bottom zoom on the peaks, without folds

    python joint_cases/sata_no_folds.py --budget 150 --workers 4      # from sata1d_code (resumable)
    python joint_cases/sata_no_folds.py --pdf
Output: joint_cases/sata_no_folds/cache/*.npz, pages/*.png, sata_no_folds.pdf
"""
import argparse
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
from sata1d.config import ArrayGeometry
from sata1d.joint_line import height_table, channel_maps
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy
import audit_dem as AD
import sweep_final as SF
import run_joint_sweep as W
from sweep_joint_all import bundle

OUT = os.path.join(HERE, "sata_no_folds")
PRF_BASE = 2000.0
JOBS = [  # gentle terrains first, then steep terrains (slopes / steps inside one SATA cell of ~310 m)
        ("one_sine", "N4b300"), ("few_cos5km_7", "N4b300"), ("one_ramp8", "N4b300"), ("few_plateau_5", "N4b500"),
        ("one_peak", "N4b300"), ("one_flank", "N4b100"), ("many_ramp8_49", "N4b300"), ("few_hills_5", "N4b300"),
        ("few_mountains_5", "N3b300"), ("few_mountains_5", "N2b300"), ("few_peak_5", "N6b300"), ("many_mountains_49", "N4b300"),
        ("few_ramp20_7", "N4b300"), ("few_escarpment_5", "N4b300"), ("one_escarpment", "N4b500")]
KEYS = ("ref", "none", "whole", "sub")
STY = {"ref": ("Reference", dict(color="tab:green", lw=2.2)),
       "none": ("no SATA", dict(color="tab:red", lw=1.6, ls=(0, (5, 3)))),
       "whole": ("SATA whole band (DEM)", dict(color="tab:blue", lw=1.2, ls=(0, (1, 1.5)))),
       "sub": ("SATA per sub-band (DEM)", dict(color="tab:orange", lw=1.2, ls=(0, (3, 1.5))))}


def run_one(c, bxt, prf):
    """Reference, no SATA, SATA whole band, SATA per sub-band (|IRF|) at total PRF prf. Same calls as audit_dem.compute."""
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    cfg = S.make_config(list(zip(xs, hs)), nrx=len(bxt), prf=prf, bxt_max=0.0, amplitudes=c["amps"], tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy()))
    tr = S.build_platform_tracks(cfg)
    ch = S.generate_channels(cfg, tr)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    hgrid = terrain(np.linspace(-half - 100.0, half + 100.0, 8001))
    Mch = channel_maps(cfg, tr, terrain, height_table(cfg, tr, hgrid.min(), hgrid.max()))
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2)).astype(np.float32)
    env = {"ref": F(S.generate_reference(cfg, tr)), "none": F(S.reconstruct(cfg, tr, ch)),
           "whole": F(AD.sata_whole(cfg, tr, ch, Mch)), "sub": F(AD.sata_sub(cfg, tr, ch, Mch))}
    return cfg, env


def cache_path(sc, tag):
    return os.path.join(OUT, "cache", f"{sc}__{tag}.npz")


def job(a):
    sc, tag = a
    t0 = time.time()
    c = AD.SCENES[sc][1]()
    bxt, kind = AD.baseline(tag)
    out = {}
    for name, prf in (("nf", PRF_BASE * len(bxt)), ("f", PRF_BASE)):
        cfg, env = run_one(c, bxt, prf)
        out.update({f"{name}_{k}": v for k, v in env.items()})
    os.makedirs(os.path.dirname(cache_path(sc, tag)), exist_ok=True)
    np.savez(cache_path(sc, tag), **out)
    return sc, tag, time.time() - t0


def cfg_of(c, tag, prf):
    bxt, kind = AD.baseline(tag)
    cfg = S.make_config(list(zip(c["xs"], c["hs"])), nrx=len(bxt), prf=prf, bxt_max=0.0, amplitudes=c["amps"], tx_rx_channel0=True)
    return replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy())), kind


def worst(v):
    """The value of largest magnitude (signed)."""
    return float(v[np.argmax(np.abs(v))])


def target_peaks(env, t, xs, vs, half_win=0.003):
    """Peak of |IRF| next to every target [dB re. the peak of the reference at that target]."""
    out = {k: [] for k in KEYS}
    for x in xs:
        w = np.abs(t - x / vs) <= half_win
        r = env["ref"][w].max()
        for k in KEYS:
            out[k].append(20 * np.log10(env[k][w].max() / r))
    return {k: np.array(v) for k, v in out.items()}


def page(sc, tag, out_png):
    """Only the case without folds (PRF_op >= B_a): SATA corrects the height, the reconstruction has no folds to undo."""
    c = AD.SCENES[sc][1]()
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    z = np.load(cache_path(sc, tag))
    nrx = len(AD.baseline(tag)[0])
    cfg, kind = cfg_of(c, tag, PRF_BASE * nrx)
    cfg_f, _ = cfg_of(c, tag, PRF_BASE)                     # only for the time span of the sweep pages
    vs = cfg.system.vs
    fig = plt.figure(figsize=(22, 10.6))
    gs = GridSpec(2, 3, width_ratios=[1.32, 0.86, 1.0], hspace=0.38, wspace=0.2, figure=fig,
                  left=0.04, right=0.985, top=0.86, bottom=0.1)
    lab = c["label"] if c["label"].startswith("h =") else c["label"][0].upper() + c["label"][1:]
    fig.suptitle(f"{lab} | $N_{{rx}}$ = {nrx}, $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)} ({kind})\n"
                 f"no undersampling: PRF$_{{op}}$ = {cfg.PRF_op:.0f} Hz $\\geq$ $B_a$ = {cfg.abw:.0f} Hz (PRF = {cfg.prf:.0f} Hz); "
                 "channels -> SATA (DEM) on every channel -> standard reconstruction", fontsize=12.5, y=0.985)
    env = {k: z[f"nf_{k}"].astype(float) for k in KEYS}
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    pk = env["ref"].max()
    p = target_peaks(env, t, xs, vs)
    a_ = fig.add_subplot(gs[0, 0])
    for k in KEYS:
        n_, st = STY[k]
        a_.plot(t, 20 * np.log10(env[k] / pk + 1e-12), label=n_, **st)
    tf = (cfg_f.Na // 2) / cfg_f.prf
    a_.set_xlim(-tf, tf); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
    a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")
    a_.set_title(f"PRF$_{{op}}$ = {cfg.PRF_op:.0f} Hz $\\geq$ $B_a$ = {cfg.abw:.0f} Hz: no folds (PRF = {cfg.prf:.0f} Hz)\n"
                 f"target peak re. the reference (worst target): no SATA {worst(p['none']):+.1f} dB | SATA whole band {worst(p['whole']):+.1f} dB | "
                 f"SATA per sub-band {worst(p['sub']):+.1f} dB", fontsize=10)
    # scene, zoom on the targets
    xs_, dys, hs_ = scene_coordinates(xs, hs)
    half = 0.5 * cfg_f.Na / cfg_f.prf * vs
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    g2 = fig.add_subplot(gs[0, 1])
    lo2, hi2 = min(xs_.min(), 0.0), max(xs_.max(), 0.0); pad2 = max(0.12 * (hi2 - lo2), 500.0)
    xg = np.linspace(-half, half, 4801); hg = terrain(xg)
    hz = hg[(xg >= lo2 - pad2) & (xg <= hi2 + pad2)]
    geometry_2d(g2, xs_, dys, hs_, False, title=None, connect=False, annotate=len(xs_) <= 13)
    g2.fill_between(xg, min(hg.min(), 0) - 0.05 * max(np.ptp(hg), 1.0), hg, color="#D7C9A7", alpha=0.45, lw=0)
    g2.plot(xg, hg, color="#6D4C2F", lw=1.2, zorder=2, label="terrain (the scene)")
    ylo, yhi = min(hz.min(), 0.0), max(hz.max(), hs_.max(), 1.0)
    g2.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))
    g2.plot(0.0, 0.0, marker="*", ms=13, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $h$ = 0)")
    g2.set_xlim(lo2 - pad2, hi2 + pad2)
    g2.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.1)
    dh = float(np.ptp(terrain(np.linspace(lo2 - pad2, hi2 + pad2, 2001))))
    g2.set_title(f"Scene: zoom on the targets | $\\Delta h_{{max}}$ = {dh:.0f} m | max slope {W.max_slope_deg(terrain, lo2 - pad2, hi2 + pad2):.0f}$^\\circ$", fontsize=10, pad=34)
    g2.xaxis.set_major_formatter(km); g2.set_xlabel("azimuth position $x$ [km]")
    g3 = fig.add_subplot(gs[0, 2], projection="3d")
    geometry_3d(g3, xs_, dys, hs_, False, connect=False, title=None)
    g3.plot(xg, iso_range_dy(hg), hg, color="#6D4C2F", lw=1.0)
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=6)
    g3.set_xlim(-half, half); g3.xaxis.set_major_formatter(km); g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4)
    g3.set_zlim(min(0.0, hg.min()), max(hg.max(), 50.0) * 1.15)
    g3.set_title(f"Scene: whole azimuth line ({2 * half / 1e3:.0f} km) in 3D (x, $\\Delta y$, h)", fontsize=10)
    for t_ in list(g3.texts):
        t_.remove()
    AD.channel_geometry(fig.add_subplot(gs[1, 1]), cfg)
    # zoom on the peaks: the (up to) three targets where no SATA is most wrong
    sel = np.sort(np.argsort(-np.abs(p["none"]))[:3])
    sub = GridSpecFromSubplotSpec(1, len(sel), subplot_spec=gs[1, 2], wspace=0.08)
    for j, i in enumerate(sel):
        a_ = fig.add_subplot(sub[0, j])
        w = np.abs(t - xs[i] / vs) <= 0.012
        for k in KEYS:
            n_, st = STY[k]
            a_.plot(1e3 * (t[w] - xs[i] / vs), 20 * np.log10(env[k][w] / pk + 1e-12), **dict(st, lw=st["lw"] + 0.4))
        a_.set_ylim(-40, 8); a_.grid(alpha=0.3); a_.tick_params(labelsize=8)
        a_.set_xlabel("[ms] from the target", fontsize=8)
        a_.set_title(f"x = {xs[i] / 1e3:g} km, h = {hs[i]:.0f} m\nno SATA {p['none'][i]:+.1f} dB | SATA {p['whole'][i]:+.1f} dB", fontsize=8.5)
        if j:
            a_.set_yticklabels([])
        else:
            a_.set_ylabel("[dB]")
    fig.text(0.79, 0.475, f"Zoom on the peaks, PRF$_{{op}}$ = {cfg.PRF_op:.0f} Hz (no folds)", ha="center", fontsize=10)
    fig.text(0.5, 0.012, f"Along-track baselines on the DPCA condition of the PRF ($b_{{at}}$ steps of 2 v / PRF = {2 * vs / cfg.prf:.2f} m). "
             "All methods use the same look-up table from the terrain (DEM); none uses the positions of the targets.\n"
             "Target peak: largest |IRF| within +-3 ms of the target, relative to the reference there (worst target: the largest difference).",
             ha="center", fontsize=8.5, color="0.35")
    fig.savefig(out_png, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.0); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--pdf", action="store_true"); ap.add_argument("--one", default="")
    a = ap.parse_args()
    t0 = time.time()
    todo = [j for j in JOBS if not os.path.exists(cache_path(*j))]
    if a.one:
        todo = [tuple(a.one.split(","))]
    if a.budget > 0 or a.one:
        with Pool(a.workers) as pool:
            it = pool.imap_unordered(job, todo)
            for r in it:
                print("done", *r, f"total {time.time() - t0:.0f} s", flush=True)
                if a.budget and time.time() - t0 > a.budget:
                    pool.terminate(); break
        print("missing:", sum(not os.path.exists(cache_path(*j)) for j in JOBS))
    if a.pdf:
        os.makedirs(os.path.join(OUT, "pages"), exist_ok=True)
        pngs, outline = [], []
        for sc, tag in JOBS:
            p = os.path.join(OUT, "pages", f"{sc}__{tag}.png")
            page(sc, tag, p)
            outline.append((f"{sc} | {tag}", len(pngs), 0)); pngs.append(p)
            print(p, flush=True)
        print("wrote", bundle(pngs, os.path.join(OUT, "sata_no_folds.pdf"), outline))


if __name__ == "__main__":
    main()
