"""
SATA before the reconstruction (whole band and per sub-band) with the DEM look-up table (terrain_lookup_table),
on the terrain cases of the joint sweep: one page per (case, baseline) in the 4-panel layout of the sweep.
For comparison, the footer gives the same SATA with the block map with copies (built from the point targets).
    SATA_LUT_CACHE=/tmp/lutcache python joint_cases/sata_dem_pages.py --budget 140      # from sata1d_code
Pages: joint_cases/sata_dem/<baseline>/<case>.png, PDF: joint_cases/sata_dem/sata_dem.pdf
"""
import argparse
import json
import os
import sys
import time
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
import sata1d.sata as SS
from sata1d.subband import sata_channels_subband, channel_spectrum
from sata1d.reconstruction import reconstruct_from_spectra
from run_one_case import ambiguity_level
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy
import run_geometry_sweep as G
import run_joint_sweep as W
import joint_sweep_summary as JS

OUT = os.path.join(HERE, "sata_dem")
CASES = ["C15", "C24", "C29", "C31", "C32", "C33"]
BASELINES = ["rnd500"]


def run(cid, btag, out=OUT):
    t0 = time.time()
    c = W.case_info(cid)
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    cfg = W.make_cfg(c, btag)
    tr = S.build_platform_tracks(cfg)
    dC0 = S.delta_C0_per_target(cfg, tr)
    ch = S.generate_channels(cfg, tr)
    tmaps = np.array([W._lut(cfg, tr, i, terrain, cid) for i in range(cfg.Nrx)])          # DEM look-up table
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))

    def whole(M):
        o = ch.copy()
        for i in range(cfg.Nrx):
            o[i] = SS.sata_1d(o[i], M[i], rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=cfg.system.wl,
                              r=cfg.scene.r0, squint=cfg.sq_tx, inverse=True, sata_osf=W.SATA_OSF)
        return S.reconstruct(cfg, tr, o)

    def sub(M):
        spec = []
        for k in range(cfg.Nsb):
            corr = sata_channels_subband(cfg, tr, ch, k, sata_osf=W.SATA_OSF, maps=list(M))
            spec.append([channel_spectrum(cfg, corr[j]) for j in range(cfg.Nrx)])
        return reconstruct_from_spectra(cfg, tr, spec, True)

    env = {"ref": F(S.generate_reference(cfg, tr)),
           "none": F(S.reconstruct(cfg, tr, ch)),
           "whole_dem": F(whole(tmaps)),
           "sub_dem": F(sub(tmaps)),
           "whole_blk": F(S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=W.SATA_OSF, mode="footprint", dC0=dC0))),
           "sub_blk": F(S.reconstruct_subband(cfg, tr, ch, sata_osf=W.SATA_OSF, mode="footprint", dC0=dC0))}
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    amb = {k: float(ambiguity_level(t, v, pk, xs, cfg.system.vs, G.ambiguity_times(cfg))) for k, v in env.items()}
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}

    fig = plt.figure(figsize=(18, 9.6))
    gs = GridSpec(2, 2, width_ratios=[1.45, 1.25], hspace=0.36, wspace=0.14, figure=fig)
    fig.suptitle(f"Case {c['number']}: {c['label']} | $b_{{xt}}$ = {W.bxt_text(W.BXT[btag])} ({W.BKIND[btag]}) | "
                 f"SATA before the reconstruction (map: look-up table from the terrain, DEM)", fontsize=12, y=0.995)
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["whole_dem"], color="tab:blue", lw=1.4, ls=(0, (5, 3)), label="SATA whole band (DEM)")
    ax.plot(t, db["sub_dem"], color="tab:orange", lw=1.3, label="SATA per sub-band (DEM)")
    ax.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | SATA whole band (DEM) {amb['whole_dem']:.1f} dB | "
                 f"SATA per sub-band (DEM) {amb['sub_dem']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | improvement: "
                 f"whole band (DEM) {amb['none'] - amb['whole_dem']:.1f} dB, per sub-band (DEM) {amb['none'] - amb['sub_dem']:.1f} dB",
                 fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")

    # geometry (as in the sweep pages)
    xs_, dys, hs_ = scene_coordinates(xs, hs)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    gg = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.18, width_ratios=[1.0, 1.1])
    g2 = fig.add_subplot(gg[0, 0]); g3 = fig.add_subplot(gg[0, 1], projection="3d")
    lo2, hi2 = min(xs_.min(), 0.0), max(xs_.max(), 0.0); pad2 = max(0.12 * (hi2 - lo2), 500.0)
    xg = np.linspace(-half, half, 4801); hg = terrain(xg)
    hz = hg[(xg >= lo2 - pad2) & (xg <= hi2 + pad2)]; dh = np.ptp(hz)
    slope = f" | max slope {W.max_slope_deg(terrain, lo2 - pad2, hi2 + pad2):.0f}$^\\circ$"
    geometry_2d(g2, xs_, dys, hs_, False, title=None, connect=False, annotate=len(xs_) <= 13)
    g2.fill_between(xg, min(hg.min(), 0) - 0.05 * np.ptp(hg), hg, color="#D7C9A7", alpha=0.45, lw=0)
    g2.plot(xg, hg, color="#6D4C2F", lw=1.2, zorder=2, label="terrain (the scene)")
    ylo, yhi = min(hz.min(), 0.0), max(hz.max(), hs_.max(), 1.0)
    g2.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))
    geometry_3d(g3, xs_, dys, hs_, False, connect=False, title=None)
    g3.plot(xg, iso_range_dy(hg), hg, color="#6D4C2F", lw=1.0)
    g2.plot(0.0, 0.0, marker="*", ms=13, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $h$ = 0)")
    g2.set_xlim(lo2 - pad2, hi2 + pad2)
    g2.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.1, handlelength=1.6, columnspacing=1.2)
    g2.set_title(f"Zoom on the point targets and the reference point | $\\Delta h_{{max}}$ = {dh:.0f} m{slope}", fontsize=9.5, pad=36)
    g2.xaxis.set_major_formatter(km); g2.set_xlabel("azimuth position $x$ [km]")
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=6)
    g3.set_xlim(-half, half); g3.xaxis.set_major_formatter(km); g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4)
    g3.set_title(f"Whole azimuth line ({2 * half / 1e3:.0f} km) in 3D (x, $\\Delta y$, h)", fontsize=9.5)
    for t_ in list(g3.texts):
        t_.remove()

    # the map read by the SATA
    am = fig.add_subplot(gs[1, 1])
    n = np.arange(cfg.Na_ch)
    for i in range(cfg.Nrx):
        am.step(n, 1e3 * tmaps[i], where="mid", color=W.CH_COLORS[i], lw=1.3, label=f"channel {i} ($b_{{xt}}$ = {cfg.array.bxt[i]:g} m)")
    cells1 = np.array([SS.az_pixel_of_scatterer(cfg, x, 1) for x in xs])
    for cc in cells1:
        am.axvline(cc, color="0.75", lw=0.6, zorder=0)
    am.plot(cells1, 1e3 * dC0[:, 1], "o", ms=5, color=W.CH_COLORS[1], mec="k", zorder=5, label="point targets: own $\\Delta C_0$ (channel 1)")
    am.set_xlim(0, cfg.Na_ch - 1); am.grid(alpha=0.3)
    am.set_xlabel("azimuth cell of the channel line"); am.set_ylabel("$\\Delta C_0$ in the map [mm]")
    lo_, hi_ = am.get_ylim(); am.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    am.set_title("$\\Delta C_0$ per cell read by the SATA (look-up table from the terrain, DEM)", fontsize=10)
    am.legend(fontsize=8, loc="upper center", ncol=3)
    fig.text(0.5, 0.012, f"Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions of every "
             f"target. For comparison (block map with copies, built from the targets): SATA whole band {amb['whole_blk']:.1f} dB, "
             f"SATA per sub-band {amb['sub_blk']:.1f} dB. $N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz, "
             f"channel 0 is the transmitter ($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)}.",
             ha="center", fontsize=8.5, color="0.35")
    fig.subplots_adjust(bottom=0.1)
    d = os.path.join(out, btag); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"{cid}.png"), dpi=90, bbox_inches="tight"); plt.close(fig)
    res = dict(amb, case=cid, baseline=btag, label=c["label"], seconds=time.time() - t0)
    json.dump(res, open(os.path.join(d, f"{cid}.json"), "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--baselines", nargs="+", default=BASELINES)
    ap.add_argument("--budget", type=float, default=1e9, help="seconds after which no new run is started")
    a = ap.parse_args()
    t0 = time.time()
    for b in a.baselines:
        for cid in a.cases:
            if os.path.exists(os.path.join(OUT, b, f"{cid}.json")):
                continue
            if time.time() - t0 > a.budget:
                print("budget reached, run again", flush=True)
                return
            r = run(cid, b)
            print(cid, b, {k: round(r[k], 1) for k in ("ref", "none", "whole_dem", "sub_dem", "whole_blk", "sub_blk")},
                  f"{r['seconds']:.0f} s", flush=True)
    pngs = [os.path.join(OUT, b, f"{c}.png") for b in a.baselines for c in a.cases]
    outline = [(f"{int(c[1:])}. " + json.load(open(os.path.join(OUT, b, f"{c}.json")))["label"].replace("$\\pi$", "π")
                + f" | b_xt = {W.bxt_text(W.BXT[b])}", k, 0) for k, (b, c) in enumerate((b, c) for b in a.baselines for c in a.cases)]
    print("wrote", JS.bundle(pngs, os.path.join(OUT, "sata_dem.pdf"), outline), flush=True)


if __name__ == "__main__":
    main()
