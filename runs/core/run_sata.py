# -*- coding: utf-8 -*-
r"""
Driver / test for the 1D SATA topography correction (sar_recon.sata).

It does three things:

  1. KERNEL SELF-TEST
     Feeds sata_1d a synthetic azimuth line and checks that (a) with
     delta_C0 = 0 it reconstructs the identity to machine precision, and
     (b) with a constant delta_C0 it applies exactly the phase
     -2*pi/lambda * delta_C0.

  2. INTEGRATION SWEEP  (the quantitative result)
     For a single elevated target on the iso-range surface (height dh, cross-
     track baseline dxt), it reconstructs three ways and compares the focused
     peak amplitude:
         no-SATA : flat-earth reconstruction (ignores the height)
         SATA    : each channel pre-conditioned with sata_channels()
         ideal   : reconstruction whose filter is told the true height
     SATA should recover ~100 % of the ideal peak, i.e. it removes the
     per-channel residual C0 (propto dh * dxt) the flat filter left behind.

  3. PLOT on an azimuth-varying topography scene (needs matplotlib).

Console output uses ASCII tokens instead of unicode so it is copy-paste-safe.
Plots use matplotlib mathtext ($...$), which renders without a LaTeX install.

Run:
    cd sar_reconstruction
    PYTHONPATH=. python ../runs/core/run_sata.py
(or place this file under runs/core/ in the repo and run it directly.)

Figures are written to  runs/core/plots/run_sata/  (png + pdf).
"""
import os
import numpy as np

import sar_recon as sar
from sar_recon.config import (SystemParams, Scene, ArrayGeometry,
                              prf_from_fixed, integration_time, build_time_axis)
from sar_recon.geometry import build_platform_tracks
from sar_recon.reconstruction import ReconstructSignalNumeri
from sar_recon.signal_model import getRawData1D
from sar_recon.analysis import matched_filter
from sar_recon.sata import sata_1d, sata_channels, sata_2d, build_delta_C0_2d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Mirror the repo convention: figures live under plots/<script_name>/ .
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots", "run_sata")


def _save_fig(fig, name, vector=True):
    """Save a figure to PLOTS_DIR as png (and pdf, for LaTeX inclusion)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    png = os.path.join(PLOTS_DIR, name + ".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    if vector:
        fig.savefig(os.path.join(PLOTS_DIR, name + ".pdf"), bbox_inches="tight")
    return png


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _build_single_target_cfg(Nrx, dxt, dh, rDelay=0.0051115753, dx=100.0):
    """Config with the reconstruction centre at h0=0 and one elevated target
    (height dh) placed on the iso-range surface (same slant range as centre)."""
    system = SystemParams()
    base = Scene(rDelay=rDelay, c0=system.c0, h0=0.0)
    r0, H = base.r0, base.H
    y0 = base.y0
    y_t = np.sqrt(r0 ** 2 - (H - dh) ** 2)
    off = (0.0, float(y_t - y0), float(dh))          # dx=0, iso-range, elevated

    scene = Scene(rDelay=rDelay, c0=system.c0, h0=0.0, extra_offsets=(off,))
    array = ArrayGeometry.linear(Nrx, dx, dxt)
    prf, PRF_op = prf_from_fixed(2000.0, Nrx)
    Na, Na_ch, ta = build_time_axis(prf, Nrx, 2.0 * integration_time(system, scene))
    cfg = sar.ExperimentConfig(name=f"sata_Nrx{Nrx}_dxt{int(dxt)}", system=system,
                               scene=scene, array=array, prf=prf, PRF_op=PRF_op,
                               Na=Na, Na_ch=Na_ch, ta=ta, plots_dir=None)
    return cfg, build_platform_tracks(cfg), np.array(off)


def _channels_and_ref(cfg, tracks, ptg_true):
    """Reference (monostatic) line and the Nrx bistatic channels for one target."""
    ptgs = ptg_true[None, :]
    s = cfg.system
    sref = getRawData1D(ptgs, tracks.ptx, tracks.ptx, tracks.vtx, tracks.vtx,
                        cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx,
                        s.wl, cfg.prf)
    s_ch = np.zeros([cfg.Nrx, cfg.Na_ch], complex)
    for ii in range(cfg.Nrx):
        s_ch[ii] = getRawData1D(ptgs, tracks.ptx, tracks.prx[ii], tracks.vtx,
                                tracks.vrx[ii], cfg.ta, cfg.sq_tx, cfg.sq_tx,
                                cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)[::cfg.Nrx]
    return sref, s_ch


def _peak_amp(srec, sref):
    f = matched_filter(srec, sref)
    return float(np.abs(f[np.argmax(np.abs(f))]))


# ---------------------------------------------------------------------------
# 1. kernel self-test
# ---------------------------------------------------------------------------
def kernel_selftest():
    print("\n[1] SATA kernel self-test")
    wl, prf, v, r0, naz = 0.25, 1000.0, 7688.0, 766206.0, 5120
    t = np.arange(naz)
    x = np.exp(2j * np.pi * 0.13 * t) + 0.5 * np.exp(-2j * np.pi * 0.07 * t)

    y0 = sata_1d(x, np.zeros(naz), rref=r0, prf=prf, v=v, wl=wl, r=r0, verbose=False)
    id_err = np.max(np.abs(y0[200:-200] - x[200:-200]))

    D = 0.01
    yC = sata_1d(x, np.full(naz, D), rref=r0, prf=prf, v=v, wl=wl, r=r0, verbose=False)
    applied = np.angle(np.mean((yC / x)[200:-200]))
    expected = np.angle(np.exp(-1j * 2 * np.pi / wl * D))

    print(f"    identity reconstruction error (dC0=0): {id_err:.2e}")
    print(f"    constant-dC0 phase  applied={applied:+.6f}  expected={expected:+.6f}")
    assert id_err < 1e-9 and abs(applied - expected) < 1e-6
    print("    OK")


# ---------------------------------------------------------------------------
# 2. integration sweep
# ---------------------------------------------------------------------------
def integration_sweep(cases=((2, 100, 120), (3, 100, 120), (4, 100, 120),
                             (5, 100, 120), (4, 150, 200))):
    print("\n[2] Integration sweep (single iso-range elevated target)")
    print(f"    {'Nrx':>3} {'dxt':>5} {'dh':>4} | {'no-SATA':>9} {'SATA':>9} "
          f"{'ideal':>9} | {'SATA/ideal':>10}")
    for Nrx, dxt, dh in cases:
        cfg, tracks, off = _build_single_target_cfg(Nrx, dxt, dh)
        ptg_true = cfg.scene.ptg + off
        sref, s_ch = _channels_and_ref(cfg, tracks, ptg_true)

        p_no = _peak_amp(sar.reconstruct(cfg, tracks, s_ch.copy()), sref)
        p_sa = _peak_amp(sar.reconstruct(cfg, tracks,
                         sata_channels(cfg, tracks, s_ch.copy())), sref)
        srecI = ReconstructSignalNumeri(
            s_ch.copy().reshape([Nrx, cfg.Na_ch, 1]), cfg.PRF_op, cfg.system.wl,
            ptg_true.reshape([3, 1]), cfg.ta, tracks.ptx, tracks.prx, tracks.vtx,
            tracks.vrx, tracks.ptx, tracks.vtx, cfg.sq_tx, cfg.sq_rx, cfg.theta_tx,
            cfg.theta_rx, cfg.array.bat, cfg.system.ve * np.ones(cfg.Na_ch),
            cfg.abw, zeroOutBw=True).flatten()
        p_id = _peak_amp(srecI, sref)
        print(f"    {Nrx:3d} {dxt:5.0f} {dh:4.0f} | {p_no:9.1f} {p_sa:9.1f} "
              f"{p_id:9.1f} | {100*p_sa/p_id:9.1f}%")


# ---------------------------------------------------------------------------
# 2b. plot: focused IRF of a single target -- the clearest "SATA works" view
# ---------------------------------------------------------------------------
def plot_single_target_irf(Nrx=4, dxt=150.0, dh=200.0, save=True):
    """
    Focused impulse response (IRF) of ONE elevated target, in dB, zoomed to the
    peak. Shows three curves: reference/ideal (sharp), no-SATA (crushed by the
    unaccounted height), and +SATA (back on the reference). This is the most
    direct visualisation that SATA restores the focus.
    """
    print("\n[2b] Single-target focused IRF (no-SATA vs +SATA)")
    cfg, tracks, off = _build_single_target_cfg(Nrx, dxt, dh)
    ptg = cfg.scene.ptg + off
    sref, s_ch = _channels_and_ref(cfg, tracks, ptg)
    srec_no = sar.reconstruct(cfg, tracks, s_ch.copy())
    srec_sa = sar.reconstruct(cfg, tracks, sata_channels(cfg, tracks, s_ch.copy()))
    p_ref = np.abs(matched_filter(sref, sref)).max()
    print(f"    peak: no-SATA {100*_peak_amp(srec_no,sref)/p_ref:5.1f}%   "
          f"+SATA {100*_peak_amp(srec_sa,sref)/p_ref:5.1f}%   of reference")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception as e:                        # pragma: no cover
        print("    (matplotlib unavailable, skipping plot)", e); return

    def irf_db(s):
        f = np.abs(matched_filter(s, sref))
        n = len(f)
        return 20 * np.log10(f[n // 2 - 120:n // 2 + 120] / p_ref + 1e-12)
    x = np.arange(240) - 120
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, irf_db(sref), "k", lw=1.6, label="reference / ideal")
    ax.plot(x, irf_db(srec_no), "C3", lw=1.4, label="no-SATA")
    ax.plot(x, irf_db(srec_sa), "C0--", lw=1.5, label="+ SATA")
    ax.set_xlim(-120, 120); ax.set_ylim(-45, 3)
    ax.set_title(rf"Focused target IRF  ($N_\mathrm{{rx}}={Nrx}$, "
                 rf"$d_\mathrm{{xt}}={dxt:.0f}$ m, $\Delta h={dh:.0f}$ m)")
    ax.set_xlabel("azimuth sample (around peak)"); ax.set_ylabel("amplitude [dB]")
    ax.legend(); ax.grid(alpha=0.3)
    if save:
        out = _save_fig(fig, "sata_single_irf")
        print(f"    IRF plot saved -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. plot: azimuth-varying topography (SATA's position-dependent correction)
# ---------------------------------------------------------------------------
def _build_azimuth_topo_cfg(Nrx=2, dxt=100.0,
                            specs=((-400, 40), (-200, 90), (0, 140),
                                   (200, 190), (400, 240)),
                            rDelay=0.0051115753):
    """Several targets at DIFFERENT azimuth positions dx, each at a different
    iso-range height dh. This is where SATA's per-position correction matters:
    every target carries its own residual C0 and gets corrected at its own
    azimuth pixel."""
    system = SystemParams()
    base = Scene(rDelay=rDelay, c0=system.c0, h0=0.0)
    r0, H, y0 = base.r0, base.H, base.y0
    extra = []
    for dxm, dh in specs:
        y_t = np.sqrt(r0 ** 2 - (H - dh) ** 2)
        extra.append((float(dxm), float(y_t - y0), float(dh)))
    scene = Scene(rDelay=rDelay, c0=system.c0, h0=0.0, extra_offsets=tuple(extra))
    array = ArrayGeometry.linear(Nrx, 100.0, dxt)
    prf, PRF_op = prf_from_fixed(2000.0, Nrx)
    Na, Na_ch, ta = build_time_axis(prf, Nrx, 2.0 * integration_time(system, scene))
    cfg = sar.ExperimentConfig(name="azimuth_topo", system=system, scene=scene,
                               array=array, prf=prf, PRF_op=PRF_op, Na=Na,
                               Na_ch=Na_ch, ta=ta, plots_dir=None)
    return cfg, build_platform_tracks(cfg)


def plot_azimuth_topo(Nrx=2, dxt=100.0, save=True):
    print("\n[3] Azimuth-varying topography (position-dependent SATA)")
    cfg, tracks = _build_azimuth_topo_cfg(Nrx, dxt)
    # generate only the elevated targets (exclude the flat reconstruction centre)
    ptgs = cfg.scene.points[1:]
    s = cfg.system
    sref = getRawData1D(ptgs, tracks.ptx, tracks.ptx, tracks.vtx, tracks.vtx,
                        cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx,
                        s.wl, cfg.prf)
    s_ch = np.zeros([cfg.Nrx, cfg.Na_ch], complex)
    for ii in range(cfg.Nrx):
        s_ch[ii] = getRawData1D(ptgs, tracks.ptx, tracks.prx[ii], tracks.vtx,
                                tracks.vrx[ii], cfg.ta, cfg.sq_tx, cfg.sq_tx,
                                cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)[::cfg.Nrx]

    srec_no = sar.reconstruct(cfg, tracks, s_ch.copy())
    srec_sa = sar.reconstruct(cfg, tracks, sata_channels(cfg, tracks, s_ch.copy()))

    def stats(sig):
        f = np.abs(matched_filter(sig, sref))
        return f, f.max(), float(np.sum(f ** 2))
    f_ref, pk_ref, en_ref = stats(sref)
    f_no, pk_no, en_no = stats(srec_no)
    f_sa, pk_sa, en_sa = stats(srec_sa)
    print(f"    peak   : no-SATA {100*pk_no/pk_ref:5.1f}%   SATA {100*pk_sa/pk_ref:5.1f}%  of reference")
    print(f"    energy : no-SATA {100*en_no/en_ref:5.1f}%   SATA {100*en_sa/en_ref:5.1f}%  of reference")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # mathtext (renders $...$ without a LaTeX install)
        matplotlib.rcParams["text.usetex"] = False
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception as e:                       # pragma: no cover
        print("    (matplotlib unavailable, skipping plot)", e)
        return

    n = len(f_ref)
    w = slice(n // 2 - 260, n // 2 + 260)
    db = lambda z: 20 * np.log10(z / f_ref.max() + 1e-12)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(db(f_ref)[w], "k", lw=1.0, label="reference")
    ax.plot(db(f_no)[w], "C3", lw=1.0, label="reconstruction (no SATA)")
    ax.plot(db(f_sa)[w], "C0", lw=1.0, label="reconstruction + SATA")
    ax.set_title(rf"Azimuth-varying topography  $N_\mathrm{{rx}}={Nrx}$,  "
                 rf"$d_\mathrm{{xt}}={dxt:.0f}$ m")
    ax.set_xlabel("azimuth sample")
    ax.set_ylabel(r"amplitude [dB]")
    ax.set_ylim(-50, 2)
    ax.legend()
    ax.grid(alpha=0.3)
    if save:
        out = _save_fig(fig, "sata_azimuth_topo")
        print(f"    IRF plot saved -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. SATA 2D: a topographic ramp across range, driven by a (simulated) DEM
# ---------------------------------------------------------------------------
def _ramp_curves(Nrx, dxt, heights):
    """Peak (% of ideal) along a range topographic ramp, for a given Nrx.
    Returns (pct_no, pct_sa): flat-earth vs SATA-2D reconstruction."""
    system = SystemParams()
    base = Scene(rDelay=0.0051115753, c0=system.c0, h0=0.0)
    r0, H = base.r0, base.H
    array = ArrayGeometry.linear(Nrx, 100.0, dxt)
    nrg = len(heights)

    refs, chans, ptgs = [], [], []
    tracks = Na_ch = ta = base_cfg = None
    for h in heights:
        cfg, tracks, off = _build_single_target_cfg(Nrx, dxt, h)
        ptg = cfg.scene.ptg + off
        sref, s_ch = _channels_and_ref(cfg, tracks, ptg)
        refs.append(sref); chans.append(s_ch); ptgs.append(ptg)
        Na_ch, ta, base_cfg = cfg.Na_ch, cfg.ta, cfg

    # DEM (ramp in range) and SATA-2D correction, channel by channel
    r_vec = np.full(nrg, r0)
    dem_true = np.tile(heights, (Na_ch, 1))
    dem_assumed = np.zeros((Na_ch, nrg))
    corr = [np.empty_like(chans[0]) for _ in range(nrg)]
    for k in range(Nrx):
        block = np.stack([chans[j][k] for j in range(nrg)], axis=1)
        dC0 = build_delta_C0_2d(dem_true, dem_assumed, r_vec, H, array.bxt[k])
        cb = sata_2d(block, dC0, r_vec, base_cfg.PRF_op, system.vs, system.wl,
                     inverse=True, sata_osf=4)
        for j in range(nrg):
            corr[j][k] = cb[:, j]

    pct_no, pct_sa = [], []
    for j in range(nrg):
        sref = refs[j]
        p_no = _peak_amp(sar.reconstruct(base_cfg, tracks, chans[j].copy()), sref)
        p_sa = _peak_amp(sar.reconstruct(base_cfg, tracks, corr[j].copy()), sref)
        srecI = ReconstructSignalNumeri(
            chans[j].copy().reshape([Nrx, Na_ch, 1]), base_cfg.PRF_op, system.wl,
            ptgs[j].reshape([3, 1]), ta, tracks.ptx, tracks.prx, tracks.vtx,
            tracks.vrx, tracks.ptx, tracks.vtx, base_cfg.sq_tx, base_cfg.sq_rx,
            base_cfg.theta_tx, base_cfg.theta_rx, array.bat,
            system.ve * np.ones(Na_ch), base_cfg.abw, zeroOutBw=True).flatten()
        p_id = _peak_amp(srecI, sref)
        pct_no.append(100 * p_no / p_id); pct_sa.append(100 * p_sa / p_id)
    return pct_no, pct_sa


def test_2d_range_ramp(Nrx_list=(2, 3, 4, 5), dxt=100.0, nrg=9, hmax=240.0, save=True):
    """
    SATA 2D exercised on a topographic RAMP along the range dimension, for
    SEVERAL channel counts.

    Each range bin holds a target at a height following a ramp (0..hmax). The
    correction is built from a simulated DEM (dem_true = the ramp, dem_assumed
    = 0) with `build_delta_C0_2d`, applied per channel with `sata_2d`, and the
    reconstruction is done per range bin. For every Nrx we report the focused
    peak (% of the ideal) with and without SATA. SATA-2D recovers ~100 % of the
    ideal at every range bin, regardless of Nrx; the flat-earth (no-SATA)
    degradation depends on Nrx and the baseline layout.
    """
    print("\n[4] SATA 2D -- topographic ramp across range (DEM-driven)")
    heights = np.linspace(0.0, hmax, nrg)
    curves = {Nrx: _ramp_curves(Nrx, dxt, heights) for Nrx in Nrx_list}

    # console table: no-SATA % per Nrx (SATA-2D is ~100 everywhere)
    hdr = "    h[m] | " + " ".join(f"Nrx={n:>2}" for n in Nrx_list) + "   | SATA-2D"
    print(hdr + "  (no-SATA %, then SATA-2D min%)")
    for j in range(nrg):
        cells = " ".join(f"{curves[n][0][j]:6.1f}" for n in Nrx_list)
        print(f"    {heights[j]:4.0f} | {cells}   | ~100")
    sata_min = min(min(curves[n][1]) for n in Nrx_list)
    print(f"    SATA-2D min over all Nrx / all bins: {sata_min:.1f}% of ideal")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception as e:                        # pragma: no cover
        print("    (matplotlib unavailable, skipping plot)", e)
        return

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.axhline(100, color="k", lw=1.4, ls="--", alpha=0.85, label="ideal", zorder=1)
    colors = ["C3", "C1", "C4", "C2", "C5", "C6"]
    for i, n in enumerate(Nrx_list):
        ax.plot(heights, curves[n][0], "-o", color=colors[i % len(colors)],
                lw=1.2, ms=5, label=rf"no-SATA, $N_\mathrm{{rx}}={n}$", zorder=2)
    # SATA-2D: every Nrx lands on the ideal line (100%), so one marker set stands
    # for all of them -> open squares so the ideal line shows through.
    ax.plot(heights, curves[Nrx_list[0]][1], linestyle="none", marker="s", ms=9,
            markerfacecolor="none", markeredgecolor="C0", markeredgewidth=1.6,
            label=r"+ SATA-2D $\approx$ ideal (any $N_\mathrm{rx}$)", zorder=3)
    ax.set_title(rf"SATA 2D on a range topographic ramp  ($d_\mathrm{{xt}}={dxt:.0f}$ m)")
    ax.set_xlabel("terrain height along the ramp [m]")
    ax.set_ylabel(r"focused peak [% of ideal]")
    ax.set_ylim(0, 112); ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    if save:
        out = _save_fig(fig, "sata_2d_range_ramp")
        print(f"    ramp plot saved -> {out}")
    plt.close(fig)


def main():
    kernel_selftest()
    integration_sweep()
    plot_single_target_irf()
    plot_azimuth_topo()
    test_2d_range_ramp()
    print("\ndone")


if __name__ == "__main__":
    main()