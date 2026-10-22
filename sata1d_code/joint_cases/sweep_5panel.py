"""
Sweep page with 5 panels: the 4 panels of run_joint_sweep.py plus the geometry of the channels (transmitter,
receivers seen from above). Layout test on one case.
    SATA_LUT_CACHE=/tmp/lutcache python joint_cases/sweep_5panel.py --case C24 --baseline rnd500      # from sata1d_code
Output: joint_cases/sweep_5panel/<baseline>/<case>.png
"""
import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
import sata1d.sata as SS
from sata1d.joint import reconstruct_joint, joint_maps
import run_geometry_sweep as G
from run_one_case import ambiguity_level
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy
import run_joint_sweep as W

OUT = os.path.join(HERE, "sweep_5panel")
CH = W.CH_COLORS + ["tab:brown", "tab:pink", "tab:gray", "tab:olive"]


def channel_geometry(ax, cfg):
    """Transmitter and receivers seen from above, in the frame of the transmitter."""
    bat, bxt, n = cfg.array.bat, cfg.array.bxt, cfg.Nrx
    xr, yr = -bat, bxt                                         # receiver i: bat behind the Tx, bxt towards the scene
    span_y = max(np.ptp(np.r_[0.0, yr]), 1.0)
    span_x = max(np.ptp(np.r_[0.0, xr]), 1.0)
    for i in range(n):
        ax.plot([0.0, xr[i]], [0.0, yr[i]], color=CH[i], lw=1.0, ls=":", zorder=1)
    for i in range(n):
        ax.plot(xr[i], yr[i], "o", ms=13, color=CH[i], mec="k", zorder=4,
                label=f"Rx {i}: $b_{{at}}$ = {bat[i]:.2f} m, $b_{{xt}}$ = {bxt[i]:g} m")
        at_tx = np.hypot(xr[i], yr[i]) < 0.02 * max(span_x, span_y)
        ax.annotate(f"Rx {i}", (xr[i], yr[i]), xytext=(-12, 10) if at_tx else (9, 5), textcoords="offset points",
                    ha="right" if at_tx else "left", color=CH[i], fontsize=10, fontweight="bold")
    ax.plot(0.0, 0.0, marker="*", ms=17, color="gold", mec="k", ls="none", zorder=5, label="Tx")
    ax.annotate("Tx", (0.0, 0.0), xytext=(9, -14), textcoords="offset points", fontsize=10, fontweight="bold")
    # directions
    ax.annotate("", xy=(0.10 * span_x, 1.13 * span_y), xytext=(-0.28 * span_x, 1.13 * span_y),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.4))
    ax.text(-0.29 * span_x, 1.13 * span_y, "flight direction ", ha="right", va="center", fontsize=9, color="0.25")
    ax.annotate("", xy=(0.12 * span_x, 0.95 * span_y), xytext=(0.12 * span_x, 0.65 * span_y),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.4))
    ax.text(0.135 * span_x, 0.80 * span_y, "towards\nthe scene", fontsize=8.5, color="0.25", va="center")
    many = n > 4                                               # legend to the right, so it does not hide receivers
    ax.set_xlim(-1.5 * span_x, (2.3 if many else 0.62) * span_x)
    ax.set_ylim(-0.15 * span_y, 1.22 * span_y)
    ax.grid(alpha=0.3)
    ax.set_xlabel("along-track position relative to the Tx [m]")
    ax.set_ylabel("cross-track position (horizontal) [m]")
    ax.set_title("Channels seen from above (all at the same altitude; axes not to scale)", fontsize=10)
    if many:
        ax.legend(fontsize=7.2, loc="upper right", framealpha=0.92)
    else:
        ax.legend(fontsize=7.6, loc="upper left", bbox_to_anchor=(0.0, 0.92), framealpha=0.92)


def sata_whole_table(cfg, tr, ch, M):
    """SATA whole band with a ready table per channel (e.g. the DEM), then the standard reconstruction."""
    o = np.array(ch, complex).copy()
    for i in range(cfg.Nrx):
        o[i] = SS.sata_1d(o[i], M[i], rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=cfg.system.wl,
                          r=cfg.scene.r0, squint=cfg.sq_tx, inverse=True, sata_osf=W.SATA_OSF)
    return S.reconstruct(cfg, tr, o)


def sata_sub_table(cfg, tr, ch, M):
    """SATA per sub-band with a ready table per channel (the same table for every sub-band)."""
    from sata1d.subband import sata_channels_subband, channel_spectrum
    from sata1d.reconstruction import reconstruct_from_spectra
    spec = []
    for k in range(cfg.Nsb):
        corr = sata_channels_subband(cfg, tr, ch, k, sata_osf=W.SATA_OSF, maps=list(M))
        spec.append([channel_spectrum(cfg, corr[j]) for j in range(cfg.Nrx)])
    return reconstruct_from_spectra(cfg, tr, spec, True)


def run(cid, btag, out=OUT, cfg=None, kind=None, lut_fn=None, case=None, with_support=False, name=None, realistic=False):
    """One page. cfg / kind / lut_fn: a ready configuration (e.g. another number of channels), the text after the
    baselines in the title, and the function that gives the terrain table of channel i (default: the sweep cache).
    case: a scene dict (xs, hs, amps, terrain, label) instead of the sweep case cid. with_support: the first panel shows
    the joint with occupied parcels and with the empty positions left out (support=scene_support) instead of the full
    system. name: file name of the page (default cid). realistic: every method uses only the DEM table (no
    information about where the targets are): SATA whole band and per sub-band read the same table as the joint,
    and the second panel shows them."""
    t0 = time.time()
    c = W.case_info(cid) if case is None else case
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    cfg = W.make_cfg(c, btag) if cfg is None else cfg
    tr = S.build_platform_tracks(cfg)
    dC0 = S.delta_C0_per_target(cfg, tr)
    ch = S.generate_channels(cfg, tr)
    if terrain is None:
        jmaps, jlabel = np.array(joint_maps(cfg, tr, dC0)), "nearest target (joint_maps)"
    else:
        jmaps, jlabel = np.array([(lut_fn or W._lut)(cfg, tr, i, terrain, cid) for i in range(cfg.Nrx)]), "look-up table from the terrain"
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    env = {"ref": F(S.generate_reference(cfg, tr)), "none": F(S.reconstruct(cfg, tr, ch))}
    if realistic:
        env["whole"] = F(sata_whole_table(cfg, tr, ch, jmaps))
        env["sub"] = F(sata_sub_table(cfg, tr, ch, jmaps))
    else:
        env["whole"] = F(S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=W.SATA_OSF, mode="footprint", dC0=dC0)))
        env["sub"] = F(S.reconstruct_subband(cfg, tr, ch, sata_osf=W.SATA_OSF, mode="footprint", dC0=dC0))
    for key, occ in (("joint_full", False), ("joint_occ", True)):
        env[key] = F(reconstruct_joint(cfg, tr, ch, maps=jmaps, sata_osf=W.SATA_OSF, occupied=occ))
    if with_support:
        env["joint_sup"] = F(reconstruct_joint(cfg, tr, ch, maps=jmaps, sata_osf=W.SATA_OSF, occupied=True,
                                               support=S.scene_support(cfg)))
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    amb = {k: float(ambiguity_level(t, v, pk, xs, cfg.system.vs, G.ambiguity_times(cfg))) for k, v in env.items()}
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}

    fig = plt.figure(figsize=(22, 10.6))
    gs = GridSpec(2, 3, width_ratios=[1.32, 0.86, 1.0], hspace=0.38, wspace=0.2, figure=fig,
                  left=0.04, right=0.985, top=0.86, bottom=0.1)
    head = f"Case {c['number']}: {c['label']}" if "number" in c else c["label"]
    fig.suptitle(f"{head} | $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)} ({kind or W.BKIND[btag]}) | "
                 f"joint reconstruction (map: {jlabel})", fontsize=13, y=0.975)
    # 1, 2: IRFs
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    if with_support:
        ax.plot(t, db["joint_occ"], color="tab:pink", lw=1.3, ls=(0, (5, 3)), label="joint, occupied parcels (occupied=True)")
        ax.plot(t, db["joint_sup"], color="tab:purple", lw=1.3,
                label="joint, occupied parcels, positions without targets left out (support)")
        ax.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | joint, occupied {amb['joint_occ']:.1f} dB | "
                     f"joint, occupied, positions without targets left out {amb['joint_sup']:.1f} dB", fontsize=10)
    else:
        ax.plot(t, db["joint_full"], color="tab:pink", lw=1.4, ls=(0, (5, 3)), label="joint, full Nrx x Nsb system (occupied=False)")
        ax.plot(t, db["joint_occ"], color="tab:purple", lw=1.3, label="joint, occupied parcels only (occupied=True)")
        ax.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | joint, occupied {amb['joint_occ']:.1f} dB | "
                     f"joint, full system {amb['joint_full']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    if realistic:
        az.plot(t, db["whole"], color="tab:blue", lw=1.2, ls=(0, (1, 1.5)), label="SATA whole band (DEM)")
        az.plot(t, db["sub"], color="tab:orange", lw=1.2, ls=(0, (3, 1.5)), label="SATA per sub-band (DEM)")
        az.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | SATA whole band "
                     f"{amb['whole']:.1f} dB | SATA per sub-band {amb['sub']:.1f} dB", fontsize=10)
    elif with_support:
        az.plot(t, db["whole"], color="tab:blue", lw=1.2, ls=(0, (1, 1.5)), label="SATA whole band (block map)")
        az.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | SATA whole band "
                     f"{amb['whole']:.1f} dB | SATA per sub-band {amb['sub']:.1f} dB", fontsize=10)
    else:
        az.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | improvement: "
                     f"joint, occupied {amb['none'] - amb['joint_occ']:.1f} dB, full system {amb['none'] - amb['joint_full']:.1f} dB", fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")
    # 3: scene, zoom on the targets
    xs_, dys, hs_ = scene_coordinates(xs, hs)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    g2 = fig.add_subplot(gs[0, 1])
    lo2, hi2 = min(xs_.min(), 0.0), max(xs_.max(), 0.0); pad2 = max(0.12 * (hi2 - lo2), 500.0)
    if terrain is None:
        geometry_2d(g2, xs_, dys, hs_, False, title=None, annotate=len(xs_) <= 13)
        dh, slope = np.ptp(hs_), ""
    else:
        xg = np.linspace(-half, half, 4801); hg = terrain(xg)
        hz = hg[(xg >= lo2 - pad2) & (xg <= hi2 + pad2)]; dh = np.ptp(hz)
        slope = f" | max slope {W.max_slope_deg(terrain, lo2 - pad2, hi2 + pad2):.0f}$^\\circ$"
        geometry_2d(g2, xs_, dys, hs_, False, title=None, connect=False, annotate=len(xs_) <= 13)
        g2.fill_between(xg, min(hg.min(), 0) - 0.05 * np.ptp(hg), hg, color="#D7C9A7", alpha=0.45, lw=0)
        g2.plot(xg, hg, color="#6D4C2F", lw=1.2, zorder=2, label="terrain (the scene)")
        ylo, yhi = min(hz.min(), 0.0), max(hz.max(), hs_.max(), 1.0)
        g2.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))
    g2.plot(0.0, 0.0, marker="*", ms=13, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $h$ = 0)")
    g2.set_xlim(lo2 - pad2, hi2 + pad2)
    g2.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.1)
    g2.set_title(f"Scene: zoom on the targets | $\\Delta h_{{max}}$ = {dh:.0f} m{slope}", fontsize=10, pad=34)
    g2.xaxis.set_major_formatter(km); g2.set_xlabel("azimuth position $x$ [km]")
    # 4: scene, whole line in 3D
    g3 = fig.add_subplot(gs[0, 2], projection="3d")
    geometry_3d(g3, xs_, dys, hs_, False, connect=terrain is None, title=None)
    if terrain is not None:
        g3.plot(xg, iso_range_dy(hg), hg, color="#6D4C2F", lw=1.0)
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=6)
    g3.set_xlim(-half, half); g3.xaxis.set_major_formatter(km); g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4)
    g3.set_title(f"Scene: whole azimuth line ({2 * half / 1e3:.0f} km) in 3D (x, $\\Delta y$, h)", fontsize=10)
    for t_ in list(g3.texts):
        t_.remove()
    # 5: channels
    cg = fig.add_subplot(gs[1, 1])
    channel_geometry(cg, cfg)
    # 6: map read by the joint
    am = fig.add_subplot(gs[1, 2])
    n = np.arange(cfg.Na_ch)
    for i in range(cfg.Nrx):
        am.step(n, 1e3 * jmaps[i], where="mid", color=CH[i], lw=1.3, label=f"channel {i} ($b_{{xt}}$ = {cfg.array.bxt[i]:g} m)")
    cells1 = np.array([SS.az_pixel_of_scatterer(cfg, x, 1) for x in xs])
    if len(cells1) <= 60:
        for cc in cells1:
            am.axvline(cc, color="0.75", lw=0.6, zorder=0)
    am.plot(cells1, 1e3 * dC0[:, 1], "o", ms=5, color=CH[1], mec="k", zorder=5, label="point targets: own $\\Delta C_0$ (channel 1)")
    am.set_xlim(0, cfg.Na_ch - 1); am.grid(alpha=0.3)
    am.set_xlabel("azimuth cell of the channel line"); am.set_ylabel("$\\Delta C_0$ in the map [mm]")
    lo_, hi_ = am.get_ylim(); am.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    am.set_title(f"$\\Delta C_0$ per cell read by the joint ({jlabel})", fontsize=10)
    am.legend(fontsize=7.8, loc="upper center", ncol=2)
    cmp_ = ("All methods use the same look-up table from the terrain (DEM); none uses the positions of the targets. " if realistic else
            f"For comparison (block map with copies): SATA whole band {amb['whole']:.1f} dB, SATA per sub-band {amb['sub']:.1f} dB. ")
    fig.text(0.5, 0.012, f"Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions of every "
             f"target. " + cmp_ +
             f"$N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz, channel 0 is the transmitter "
             f"($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)}.", ha="center", fontsize=8.5, color="0.35")
    d = os.path.join(out, btag); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{name or cid}.png")
    fig.savefig(p, dpi=90); plt.close(fig)
    print(cid, btag, {k: round(v, 1) for k, v in amb.items()}, f"{time.time() - t0:.0f} s", flush=True)
    return p, amb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="C24")
    ap.add_argument("--baseline", default="rnd500")
    a = ap.parse_args()
    print(run(a.case, a.baseline))
