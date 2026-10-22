"""
Option 1 (truncated pinv) in the layout of the sweep pages (run_joint_sweep.py): reference + joint with occupied parcels,
pinv as now (dashed) and pinv with rcond = 0.1 (solid) | geometry | reference + no SATA | map of the joint. For the
cases of test_joint_rcond.py, every baseline of the sweep (--baselines to choose). sata1d/joint.py is used unchanged.
Output: joint_cases/rcond_test/sweep_layout/<baseline>/<case>.png and joint_cases/joint_rcond_irf.pdf
    python joint_cases/rcond_sweep_pages.py --workers 4          # from sata1d_code
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
from sata1d.joint import joint_maps
import run_geometry_sweep as G
import test_joint_rcond as TR
from run_joint_sweep import case_info, make_cfg, bxt_text, max_slope_deg, BXT, BKIND, SATA_OSF, CH_COLORS
from run_one_case import ambiguity_level
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy

OUT = os.path.join(TR.OUT, "sweep_layout")
RC = 0.1
from run_joint_sweep import BASELINES
ALL_BASES = [b for b, _, _ in BASELINES]          # order of the sweep


def run(cid, btag, out=OUT):
    t0 = time.time()
    c = case_info(cid)
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    cfg = make_cfg(c, btag)
    tr = S.build_platform_tracks(cfg)
    dC0 = S.delta_C0_per_target(cfg, tr)
    ch = S.generate_channels(cfg, tr)
    if terrain is None:
        jmaps, jlabel = np.array(joint_maps(cfg, tr, dC0)), "nearest target (joint_maps)"
    else:
        jmaps, jlabel = np.array([S.terrain_lookup_table(cfg, tr, i, terrain, step=4) for i in range(cfg.Nrx)]), "look-up table from the terrain"
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    tj = {}
    env = {"ref": F(S.generate_reference(cfg, tr)),
           "none": F(S.reconstruct(cfg, tr, ch)),
           "whole": F(S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=SATA_OSF, mode="footprint", dC0=dC0))),
           "sub": F(S.reconstruct_subband(cfg, tr, ch, sata_osf=SATA_OSF, mode="footprint", dC0=dC0))}
    for key, rc in (("joint_occ", 0.0), ("joint_rc", RC)):
        t1 = time.time(); env[key] = F(TR.joint_rcond(cfg, tr, ch, jmaps, rc)); tj[key] = time.time() - t1
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    amb = {k: float(ambiguity_level(t, v, pk, xs, cfg.system.vs, G.ambiguity_times(cfg))) for k, v in env.items()}
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}

    fig = plt.figure(figsize=(18, 9.6))
    gs = GridSpec(2, 2, width_ratios=[1.45, 1.25], hspace=0.36, wspace=0.14, figure=fig)
    fig.suptitle(f"Case {c['number']}: {c['label']} | $b_{{xt}}$ = {bxt_text(BXT[btag])} ({BKIND[btag]}) | "
                 f"joint reconstruction, occupied parcels (map: {jlabel}): pinv as now vs. pinv with rcond = {RC:g}", fontsize=12, y=0.995)
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["joint_occ"], color="tab:purple", lw=1.4, ls=(0, (5, 3)), label="joint, occupied parcels, pinv as now")
    ax.plot(t, db["joint_rc"], color="tab:blue", lw=1.3, label=f"joint, occupied parcels, pinv with rcond = {RC:g}")
    ax.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | joint, pinv as now {amb['joint_occ']:.1f} dB | "
                 f"joint, rcond = {RC:g}: {amb['joint_rc']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | improvement: "
                 f"joint, pinv as now {amb['none'] - amb['joint_occ']:.1f} dB, joint, rcond = {RC:g}: {amb['none'] - amb['joint_rc']:.1f} dB", fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, max(5.0, max(amb.values()) + 5)); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")

    # geometry: 2D zoom on the targets and the reference point, 3D over the whole azimuth line
    xs_, dys, hs_ = scene_coordinates(xs, hs)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    gg = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.18, width_ratios=[1.0, 1.1])
    g2 = fig.add_subplot(gg[0, 0]); g3 = fig.add_subplot(gg[0, 1], projection="3d")
    lo2, hi2 = min(xs_.min(), 0.0), max(xs_.max(), 0.0); pad2 = max(0.12 * (hi2 - lo2), 500.0)
    slope = ""
    if terrain is None:
        geometry_2d(g2, xs_, dys, hs_, False, title=None, annotate=len(xs_) <= 13)
        geometry_3d(g3, xs_, dys, hs_, False, title=None)
        dh = np.ptp(hs_)
    else:
        xg = np.linspace(-half, half, 4801); hg = terrain(xg)
        hz = hg[(xg >= lo2 - pad2) & (xg <= hi2 + pad2)]; dh = np.ptp(hz)
        slope = f" | max slope {max_slope_deg(terrain, lo2 - pad2, hi2 + pad2):.0f}$^\\circ$"
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

    # the map read by the joint reconstruction
    am = fig.add_subplot(gs[1, 1])
    n = np.arange(cfg.Na_ch)
    for i in range(cfg.Nrx):
        am.step(n, 1e3 * jmaps[i], where="mid", color=CH_COLORS[i], lw=1.3, label=f"channel {i} ($b_{{xt}}$ = {cfg.array.bxt[i]:g} m)")
    cells1 = np.array([SS.az_pixel_of_scatterer(cfg, x, 1) for x in xs])
    if len(cells1) <= 60:
        for cc in cells1:
            am.axvline(cc, color="0.75", lw=0.6, zorder=0)
    am.plot(cells1, 1e3 * dC0[:, 1], "o", ms=5, color=CH_COLORS[1], mec="k", zorder=5, label="point targets: own $\\Delta C_0$ (channel 1)")
    am.set_xlim(0, cfg.Na_ch - 1); am.grid(alpha=0.3)
    am.set_xlabel("azimuth cell of the channel line"); am.set_ylabel("$\\Delta C_0$ in the map [mm]")
    lo_, hi_ = am.get_ylim(); am.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    am.set_title(f"$\\Delta C_0$ per cell read by the joint reconstruction ({jlabel})", fontsize=10)
    am.legend(fontsize=8, loc="upper center", ncol=3)
    fig.text(0.5, 0.012, f"Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions of every "
             f"target. For comparison (block map with copies): SATA whole band {amb['whole']:.1f} dB, SATA per sub-band {amb['sub']:.1f} dB. "
             f"Joint: {tj['joint_occ']:.2f} s (pinv as now) and {tj['joint_rc']:.2f} s (rcond = {RC:g}) per line. "
             f"$N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz, channel 0 is the transmitter "
             f"($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {bxt_text(cfg.array.bxt)}.", ha="center", fontsize=8.5, color="0.35")
    fig.subplots_adjust(bottom=0.1)
    d = os.path.join(out, btag); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"{cid}.png"), dpi=90, bbox_inches="tight"); plt.close(fig)
    res = dict(amb, case=cid, baseline=btag, bxt=list(map(float, cfg.array.bxt)), label=c["label"], group=c["group"],
               old=c["old"], seconds=time.time() - t0)
    json.dump(res, open(os.path.join(d, f"{cid}.json"), "w"), indent=1)
    return res



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cases", nargs="+", default=TR.CASES)
    ap.add_argument("--baselines", nargs="+", default=ALL_BASES, choices=ALL_BASES)
    ap.add_argument("--pdf", default=os.path.join(HERE, "joint_rcond_irf.pdf"))
    a = ap.parse_args()
    todo = [(c, b) for c in a.cases for b in a.baselines if not os.path.exists(os.path.join(OUT, b, f"{c}.json"))]
    from multiprocessing import Pool
    with Pool(a.workers) as p:
        for r in p.starmap(run, todo):
            print(r["case"], r["baseline"], f"now {r['joint_occ']:.1f} rcond {r['joint_rc']:.1f} none {r['none']:.1f} whole {r['whole']:.1f}", flush=True)
    # PDF: every case of test_joint_rcond, every baseline already computed, in the order of the sweep
    jobs = [(c, b) for c in TR.CASES for b in ALL_BASES if os.path.exists(os.path.join(OUT, b, f"{c}.png"))]
    import joint_sweep_summary as JS
    outline = []
    for k, (c, b) in enumerate(jobs):
        if k == 0 or jobs[k - 1][0] != c:
            outline.append((f"case {int(c[1:])}: {case_info(c)['label']}", k, 0))
        outline.append((f"bxt {bxt_text(BXT[b])}", k, 1))
    print("wrote", JS.bundle([os.path.join(OUT, b, f"{c}.png") for c, b in jobs], a.pdf, outline), len(jobs), "pages")


if __name__ == "__main__":
    main()
