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
from sar_recon.sata import sata_1d, sata_channels

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Figures live under plots/run_sata/<category>/ to keep the tree tidy.
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots", "run_sata")

# Apply a Hamming taper when focusing the IRF plots. The antenna pattern in the
# model is a hard rectangular window, which gives ~-13 dB sinc sidelobes; the
# taper suppresses them to ~-40 dB (slightly wider mainlobe). Set False to see
# the raw rectangular-aperture sidelobes.
#
# NOTE: the sar_recon reconstruction/analysis pipeline does NOT apply any
# amplitude taper (matched_filter is a pure correlation). To keep every figure
# faithful to the real pipeline, TAPER is False: all plots show the raw
# rectangular-aperture sinc IRF (-13 dB sidelobes), exactly what the code
# produces.
TAPER = False


def _save_fig(fig, name, subdir="", vector=True):
    """Save a figure to PLOTS_DIR/<subdir>/ as png (and pdf)."""
    outdir = os.path.join(PLOTS_DIR, subdir) if subdir else PLOTS_DIR
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, name + ".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    if vector:
        fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
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


def _band_hamming(Na, prf, abw):
    """Hamming window over the processed Doppler band |f| <= abw/2 (zeros
    elsewhere), in FFT order. Used to taper the aperture and suppress the
    rectangular-aperture sinc sidelobes (~-13 dB -> ~-40 dB)."""
    fa = np.fft.fftshift(np.fft.fftfreq(Na, d=1.0 / prf))
    w = np.zeros(Na)
    band = np.abs(fa) <= abw / 2.0
    w[band] = np.hamming(int(band.sum()))
    return np.fft.ifftshift(w)


def _focus_mag(sig, ref, prf=None, abw=None, taper=False):
    """|azimuth-compressed signal|: matched filter of `sig` against the
    single-target reference `ref`, optionally with a Hamming taper over the
    processed band to suppress aperture sidelobes."""
    S = np.fft.fft(sig) * np.conj(np.fft.fft(ref))
    if taper and abw is not None:
        S = S * _band_hamming(len(S), prf, abw)
    return np.abs(np.roll(np.fft.ifft(S), len(ref) // 2))


def _recon_elevated(Nrx, dxt, dh, dx=100.0, bat_offset=0.0, array=None):
    """Reconstruct ONE elevated iso-range target three ways.
    `array` overrides the (linear) array geometry -- pass an ArrayGeometry with
    custom bat/bxt to test non-uniform / arbitrary baselines.
    Returns (cfg, ref, srec_no, srec_sa, srec_ideal)."""
    system = SystemParams()
    base = Scene(rDelay=0.0051115753, c0=system.c0, h0=0.0)
    r0, H, y0 = base.r0, base.H, base.y0
    y_t = np.sqrt(r0 ** 2 - (H - dh) ** 2)
    off = (0.0, float(y_t - y0), float(dh))
    scene = Scene(rDelay=0.0051115753, c0=system.c0, h0=0.0, extra_offsets=(off,))
    if array is None:
        array = ArrayGeometry.linear(Nrx, dx, dxt, bat_offset=bat_offset)
    Nrx = array.Nrx
    prf, PRF_op = prf_from_fixed(2000.0, Nrx)
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.0 * integration_time(system, scene))
    cfg = sar.ExperimentConfig(name="c", system=system, scene=scene, array=array,
                               prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Nc, ta=ta,
                               plots_dir=None)
    tr = build_platform_tracks(cfg)
    ptg = scene.ptg + np.array(off)
    ref = getRawData1D(ptg[None, :], tr.ptx, tr.ptx, tr.vtx, tr.vtx, ta,
                       cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx, system.wl, prf)
    s_ch = np.zeros([Nrx, Nc], complex)
    for i in range(Nrx):
        s_ch[i] = getRawData1D(ptg[None, :], tr.ptx, tr.prx[i], tr.vtx, tr.vrx[i],
                               ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx,
                               system.wl, prf)[::Nrx]
    no = sar.reconstruct(cfg, tr, s_ch.copy())
    sa = sar.reconstruct(cfg, tr, sata_channels(cfg, tr, s_ch.copy()))
    ideal = ReconstructSignalNumeri(
        s_ch.copy().reshape([Nrx, Nc, 1]), PRF_op, system.wl, ptg.reshape([3, 1]),
        ta, tr.ptx, tr.prx, tr.vtx, tr.vrx, tr.ptx, tr.vtx, cfg.sq_tx, cfg.sq_rx,
        cfg.theta_tx, cfg.theta_rx, array.bat, system.ve * np.ones(Nc), cfg.abw,
        zeroOutBw=True).flatten()
    return cfg, ref, no, sa, ideal


def _ambiguity_db(f, mask=100):
    """Worst replica level [dB below peak] of a focused signal |f|, ignoring the
    mainlobe (+/-mask samples). Azimuth ambiguities sit far from the mainlobe."""
    f = f / f.max()
    n = len(f)
    i0 = int(np.argmax(f))
    m = np.ones(n, bool)
    m[max(0, i0 - mask):i0 + mask] = False
    return 20.0 * np.log10(f[m].max())


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
    Focused impulse response (IRF) of ONE elevated target, in dB, as TWO panels:
      left  -- ZOOM: mainlobe + sinc sidelobes (the familiar IRF look);
      right -- WIDE: the whole aperture, where the azimuth AMBIGUITY replicas
               show up (no-SATA has strong ones; SATA suppresses them).
    Three curves each: reference/ideal, no-SATA (crushed + ambiguities), +SATA.
    No taper, so the real sinc structure and the ambiguities are both visible.
    """
    print("\n[2b] Single-target focused IRF (zoom + wide)")
    cfg, tracks, off = _build_single_target_cfg(Nrx, dxt, dh)
    ptg = cfg.scene.ptg + off
    sref, s_ch = _channels_and_ref(cfg, tracks, ptg)
    srec_no = sar.reconstruct(cfg, tracks, s_ch.copy())
    srec_sa = sar.reconstruct(cfg, tracks, sata_channels(cfg, tracks, s_ch.copy()))
    foc = lambda s: _focus_mag(s, sref, cfg.prf, cfg.abw, taper=False)  # raw sinc IRF
    f_ref, f_no, f_sa = foc(sref), foc(srec_no), foc(srec_sa)
    p_ref = f_ref.max()
    a_no, a_sa = _ambiguity_db(f_no), _ambiguity_db(f_sa)
    print(f"    peak: no-SATA {100*f_no.max()/p_ref:.0f}%  +SATA {100*f_sa.max()/p_ref:.0f}%"
          f"   | worst ambiguity: no-SATA {a_no:.0f} dB  +SATA {a_sa:.0f} dB")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception as e:                        # pragma: no cover
        print("    (matplotlib unavailable, skipping plot)", e); return

    n = len(f_ref); i0 = n // 2
    db = lambda f: 20 * np.log10(f / p_ref + 1e-12)
    # replica location (for the wide-panel annotation)
    mm = np.ones(n, bool); mm[i0 - 100:i0 + 100] = False
    rep = int(np.argmax(f_no * mm)) - i0

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for W, ax, ttl in ((120, a1, "Zoom: mainlobe + sinc sidelobes"),
                       (1800, a2, "Wide: azimuth ambiguities")):
        w = slice(i0 - W, i0 + W); x = np.arange(2 * W) - W
        ax.plot(x, db(f_ref)[w], "k", lw=1.2, label="reference / ideal")
        ax.plot(x, db(f_no)[w], "C3", lw=0.9, label="no-SATA")
        ax.plot(x, db(f_sa)[w], "C0--", lw=1.1, label="+ SATA")
        ax.set_ylim(-50, 6); ax.set_xlabel("azimuth sample (around target)")
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    a1.set_ylabel("amplitude [dB]")
    for xa in (rep, -rep):
        a2.annotate("ambiguity", xy=(xa, db(f_no)[i0 + xa]), xytext=(xa, 3),
                    ha="center", fontsize=8, color="C3",
                    arrowprops=dict(arrowstyle="->", color="C3"))
    fig.suptitle(rf"Single-target IRF  ($N_\mathrm{{rx}}={Nrx}$, "
                 rf"$d_\mathrm{{xt}}={dxt:.0f}$ m, $\Delta h={dh:.0f}$ m)  --  "
                 rf"ambiguity no-SATA {a_no:.0f} dB $\rightarrow$ +SATA {a_sa:.0f} dB",
                 fontsize=11)
    fig.tight_layout()
    if save:
        out = _save_fig(fig, "sata_single_irf", subdir="irf")
        print(f"    IRF plot saved -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2c. 4-panel reconstruction diagnostic (repo style: amp / dB / IRF / phase)
# ---------------------------------------------------------------------------
def _plot_4panel(cfg, sref, srec_no, srec_sa, fname, extra_title="", subdir="irf"):
    """Repo-style 2x2 diagnostic (amplitude / dB over aperture / zoomed IRF /
    spectral phase) comparing reference / no-SATA / +SATA. Shared by the
    single-target and topo_ramp 4-panel plots."""
    Nrx = cfg.Nrx
    res_no = sar.analyze(cfg, sref, srec_no)
    res_sa = sar.analyze(cfg, sref, srec_sa)
    ta = cfg.ta

    def dph(res):
        d = np.angle(np.fft.fft(res.srecNF) * np.conjugate(np.fft.fft(res.srefF)), deg=True)
        d[res.abw_idx] = 0.0
        return d
    ndb = lambda x: 20.0 * np.log10(np.abs(x) / np.max(np.abs(x)) + 1e-12)

    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception as e:                        # pragma: no cover
        print("    (matplotlib unavailable)", e); return

    dbat = cfg.array.bat[1] - cfg.array.bat[0] if Nrx > 1 else 0.0
    dbxt = cfg.array.bxt[1] - cfg.array.bxt[0] if Nrx > 1 else 0.0
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(rf"SATA reconstruction | Nrx={Nrx} | PRF={cfg.prf:.0f} Hz | "
                 rf"$B_a$={cfg.abw:.0f} Hz | $\Delta b_{{at}}$={dbat:.0f} m | "
                 rf"$\Delta b_{{xt}}$={dbxt:.0f} m" + extra_title)
    C = dict(ref="k", no="C3", sa="C0")

    ax[0, 0].plot(ta, np.abs(sref), C["ref"], lw=1.0, label="reference")
    ax[0, 0].plot(ta, np.abs(srec_no), C["no"], lw=0.8, label="no-SATA")
    ax[0, 0].plot(ta, np.abs(srec_sa), C["sa"], lw=0.8, ls="--", label="+SATA")
    ax[0, 0].set_xlabel("Time [s]"); ax[0, 0].set_ylabel("Amplitude"); ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend(fontsize="small")

    ax[0, 1].plot(ta, ndb(res_no.srefF), C["ref"], lw=0.7, label="reference")
    ax[0, 1].plot(ta, ndb(res_no.srecNF), C["no"], lw=0.7, label="no-SATA")
    ax[0, 1].plot(ta, ndb(res_sa.srecNF), C["sa"], lw=0.7, ls="--", label="+SATA")
    ax[0, 1].set_xlabel("Time [s]"); ax[0, 1].set_ylabel("[dB]")
    ax[0, 1].set_ylim([-100, 2]); ax[0, 1].grid(alpha=0.3); ax[0, 1].legend(fontsize="small")
    ax[0, 1].set_title("ambiguities (no-SATA) vs suppressed (+SATA)", fontsize=9)

    ax[1, 0].plot(res_no.taz * 1e3, ndb(res_no.u_refFocC), C["ref"], lw=1.2, label="reference")
    ax[1, 0].plot(res_no.taz * 1e3, ndb(res_no.u_interpFocCN), C["no"], lw=0.9, label="no-SATA")
    ax[1, 0].plot(res_sa.taz * 1e3, ndb(res_sa.u_interpFocCN), C["sa"], lw=1.1, ls="--", label="+SATA")
    ax[1, 0].set_xlabel("Time [ms]"); ax[1, 0].set_ylabel("[dB]")
    ax[1, 0].set_ylim([-45, 2]); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize="small")
    ax[1, 0].set_title("zoomed IRF", fontsize=9)

    ax[1, 1].plot(res_no.fa, dph(res_no), C["no"], lw=0.7, label="no-SATA")
    ax[1, 1].plot(res_sa.fa, dph(res_sa), C["sa"], lw=0.9, label="+SATA")
    ax[1, 1].axvline(cfg.abw / 2, color="r", ls="-."); ax[1, 1].axvline(-cfg.abw / 2, color="r", ls="-.")
    ax[1, 1].set_xlabel("Doppler freq [Hz]"); ax[1, 1].set_ylabel("[deg]")
    ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize="small")
    ax[1, 1].set_title("spectral phase error", fontsize=9)

    fig.tight_layout()
    print(f"    4-panel plot saved -> {_save_fig(fig, fname, subdir=subdir)}")
    plt.close(fig)


def plot_irf_4panel(Nrx=4, dxt=150.0, dh=200.0, save=True):
    """Single-target 4-panel diagnostic (repo style)."""
    print("\n[2c] Single-target 4-panel diagnostic (amp / dB / IRF / phase)")
    cfg, tracks, off = _build_single_target_cfg(Nrx, dxt, dh)
    ptg = cfg.scene.ptg + off
    sref, s_ch = _channels_and_ref(cfg, tracks, ptg)
    srec_no = sar.reconstruct(cfg, tracks, s_ch.copy())
    srec_sa = sar.reconstruct(cfg, tracks, sata_channels(cfg, tracks, s_ch.copy()))
    _plot_4panel(cfg, sref, srec_no, srec_sa, "sata_irf_4panel",
                 extra_title=rf" | $\Delta h$={dh:.0f} m (single target)")


def plot_toporamp_4panel(Nrx=4, dxt=100.0, save=True):
    """topo_ramp 4-panel diagnostic (the scene you used before), repo style."""
    print("\n[3c] topo_ramp 4-panel diagnostic (amp / dB / IRF / phase)")
    from sar_recon.config import SCENE_PRESETS
    system = SystemParams()
    scene = Scene(rDelay=0.0051115753, c0=system.c0, h0=0.0,
                  extra_offsets=SCENE_PRESETS["topo_ramp"])
    array = ArrayGeometry.linear(Nrx, 100.0, dxt)
    prf, PRF_op = prf_from_fixed(2000.0, Nrx)
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.0 * integration_time(system, scene))
    cfg = sar.ExperimentConfig(name="topo_ramp", system=system, scene=scene, array=array,
                               prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Nc, ta=ta, plots_dir=None)
    tracks = build_platform_tracks(cfg)
    sref = sar.generate_reference(cfg, tracks)
    s_ch = sar.generate_channels(cfg, tracks)
    srec_no = sar.reconstruct(cfg, tracks, s_ch.copy())
    srec_sa = sar.reconstruct(cfg, tracks, sata_channels(cfg, tracks, s_ch.copy()))
    _plot_4panel(cfg, sref, srec_no, srec_sa, "sata_toporamp_4panel",
                 extra_title=" | topo_ramp scene", subdir="topography")




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


def plot_azimuth_topo(Nrx=4, dxt=150.0, save=True):
    print("\n[3] Azimuth-varying topography (position-dependent SATA)")
    # Five targets at different azimuth positions and increasing heights (a ramp
    # along azimuth). Default: a strong case (Nrx=4, dxt=150) so the effect shows.
    specs = ((-400, 80), (-200, 160), (0, 240), (200, 320), (400, 400))
    cfg, tracks = _build_azimuth_topo_cfg(Nrx, dxt, specs=specs)
    s = cfg.system
    ptgs = cfg.scene.points[1:]                       # the 5 elevated targets

    # IMPORTANT: focus (azimuth-compress) with a SINGLE-point-target reference,
    # not the multi-target signal -- otherwise the matched filter returns the
    # autocorrelation of the scene (extra peaks at every target-to-target lag,
    # which look like ambiguities but are not). A single-target reference gives a
    # proper focused image: one clean peak per real target.
    sref1 = getRawData1D(cfg.scene.ptg[None, :], tracks.ptx, tracks.ptx, tracks.vtx,
                         tracks.vtx, cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx,
                         cfg.theta_tx, s.wl, cfg.prf)
    sig_true = getRawData1D(ptgs, tracks.ptx, tracks.ptx, tracks.vtx, tracks.vtx,
                            cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx,
                            s.wl, cfg.prf)             # ideal (monostatic) scene
    s_ch = np.zeros([cfg.Nrx, cfg.Na_ch], complex)
    for ii in range(cfg.Nrx):
        s_ch[ii] = getRawData1D(ptgs, tracks.ptx, tracks.prx[ii], tracks.vtx,
                                tracks.vrx[ii], cfg.ta, cfg.sq_tx, cfg.sq_tx,
                                cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)[::cfg.Nrx]

    srec_no = sar.reconstruct(cfg, tracks, s_ch.copy())
    srec_sa = sar.reconstruct(cfg, tracks, sata_channels(cfg, tracks, s_ch.copy()))

    focus = lambda sig: _focus_mag(sig, sref1, cfg.prf, cfg.abw, taper=TAPER)
    f_ref = focus(sig_true); pmax = f_ref.max()
    f_no, f_sa = focus(srec_no), focus(srec_sa)
    print(f"    tallest-target peak: no-SATA {100*f_no.max()/pmax:5.1f}%   "
          f"SATA {100*f_sa.max()/pmax:5.1f}%  of reference   (taper={TAPER})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["text.usetex"] = False
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception as e:                       # pragma: no cover
        print("    (matplotlib unavailable, skipping plot)", e)
        return

    n = len(f_ref)
    w = slice(n // 2 - 300, n // 2 + 300)
    x = np.arange(600) - 300
    db = lambda z: 20 * np.log10(z / pmax + 1e-12)
    fig, ax = plt.subplots(figsize=(8.7, 4.3))
    ax.plot(x, db(f_ref)[w], "k", lw=1.3, label="reference")
    ax.plot(x, db(f_no)[w], "C3", lw=1.0, label="reconstruction (no SATA)")
    ax.plot(x, db(f_sa)[w], "C0--", lw=1.3, label="reconstruction + SATA")
    ax.set_title(rf"Azimuth-varying topography  ($N_\mathrm{{rx}}={Nrx}$, "
                 rf"$d_\mathrm{{xt}}={dxt:.0f}$ m) -- focused image")
    ax.set_xlabel("azimuth sample (around scene centre)")
    ax.set_ylabel(r"amplitude [dB]")
    ax.set_ylim(-40, 3)
    ax.legend()
    ax.grid(alpha=0.3)
    if save:
        out = _save_fig(fig, "sata_azimuth_topo", subdir="topography")
        print(f"    focused-image plot saved -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3b. "SATA looks worse in a zoom" pitfall -- zoom vs wide view
# ---------------------------------------------------------------------------
def plot_zoom_vs_wide(Nrx=4, dxt=150.0, save=True):
    """Register a common pitfall: in a narrow zoom the multi-target SATA image
    can LOOK worse than no-SATA (higher pedestal), but that pedestal is RECOVERED
    energy. no-SATA only looks clean at the centre because its energy fled to the
    far azimuth ambiguities. SATA is better by every metric (peak, focused energy,
    scattered energy) -- the zoom just hides where the energy went."""
    print("\n[3b] Zoom vs wide view (why SATA can look worse in a zoom)")
    specs = ((-400, 80), (-200, 160), (0, 240), (200, 320), (400, 400))
    cfg, tracks = _build_azimuth_topo_cfg(Nrx, dxt, specs=specs)
    s = cfg.system
    ptgs = cfg.scene.points[1:]
    sref1 = getRawData1D(cfg.scene.ptg[None, :], tracks.ptx, tracks.ptx, tracks.vtx,
                         tracks.vtx, cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx,
                         cfg.theta_tx, s.wl, cfg.prf)
    sig = getRawData1D(ptgs, tracks.ptx, tracks.ptx, tracks.vtx, tracks.vtx, cfg.ta,
                       cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)
    s_ch = np.zeros([cfg.Nrx, cfg.Na_ch], complex)
    for ii in range(cfg.Nrx):
        s_ch[ii] = getRawData1D(ptgs, tracks.ptx, tracks.prx[ii], tracks.vtx,
                                tracks.vrx[ii], cfg.ta, cfg.sq_tx, cfg.sq_tx,
                                cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)[::cfg.Nrx]
    no = sar.reconstruct(cfg, tracks, s_ch.copy())
    sa = sar.reconstruct(cfg, tracks, sata_channels(cfg, tracks, s_ch.copy()))
    fr = _focus_mag(sig, sref1, cfg.prf, cfg.abw, taper=TAPER)
    fn = _focus_mag(no, sref1, cfg.prf, cfg.abw, taper=TAPER)
    fs = _focus_mag(sa, sref1, cfg.prf, cfg.abw, taper=TAPER)
    pref = fr.max(); n = len(fr); i0 = n // 2

    # metrics
    pk = [i for i in range(1, n - 1) if fr[i] > 0.5 * pref and fr[i] > fr[i - 1] and fr[i] > fr[i + 1]]
    mlobe = np.zeros(n, bool)
    for i in pk:
        mlobe[i - 8:i + 8] = True
    def focused(f):   return 100 * np.sum(f[mlobe] ** 2) / np.sum(f ** 2)
    def scattered(f): return 100 * (1 - np.sum(f[i0 - 600:i0 + 600] ** 2) / np.sum(f ** 2))
    m = (f"peak {100*fn.max()/pref:.0f}%->{100*fs.max()/pref:.0f}% | "
         f"focused {focused(fn):.0f}%->{focused(fs):.0f}% | "
         f"scattered {scattered(fn):.0f}%->{scattered(fs):.0f}%")
    print("    no-SATA -> +SATA:  " + m)

    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception as e:                        # pragma: no cover
        print("    (matplotlib unavailable)", e); return

    db = lambda f: 20 * np.log10(f / pref + 1e-12)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for W, ax, ttl in ((300, a1, "ZOOM (+/-300): SATA looks worse (higher pedestal)"),
                       (2000, a2, "WIDE (+/-2000): no-SATA energy went to the ambiguities")):
        w = slice(i0 - W, i0 + W); x = np.arange(2 * W) - W
        ax.plot(x, db(fn)[w], "C3", lw=0.7, label="no SATA")
        ax.plot(x, db(fs)[w], "C0", lw=0.7, label="+ SATA")
        ax.set_ylim(-55, 12); ax.set_xlabel("azimuth sample"); ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    a1.set_ylabel("amplitude [dB]")
    a2.annotate("ambiguities", xy=(-1620, -6), xytext=(-1500, 7), fontsize=8, color="C3",
                arrowprops=dict(arrowstyle="->", color="C3"))
    fig.suptitle("SATA is NOT worse -- " + m, fontsize=11)
    fig.tight_layout()
    if save:
        print(f"    zoom-vs-wide plot -> {_save_fig(fig, 'sata_zoom_vs_wide', subdir='topography')}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3d. RANGE ramp handled 1D, per range bin -- where SATA clearly improves
# ---------------------------------------------------------------------------
def plot_range_ramp_1d(Nrx=4, dxt=100.0, nrg=9, hmax=300.0, save=True):
    """A topographic ramp along RANGE, processed the 1D way: each range bin is an
    INDEPENDENT azimuth line with a single target at a height that grows with
    range. 1D SATA (sata_channels) is applied per bin. Unlike the topo_ramp scene
    (all heights stacked in ONE azimuth cell -> 1D cannot separate them), here
    each bin has a single dominant height, so SATA recovers each one. This is the
    honest 1D way to treat a range ramp and shows a clear improvement.
    Two panels: peak recovery and worst ambiguity vs terrain height."""
    print("\n[3d] Range ramp, 1D per range bin (peak %% / ambiguity vs height)")
    heights = np.linspace(0.0, hmax, nrg)
    pct_no, pct_sa, am_no, am_sa = [], [], [], []
    for h in heights:
        cfg, ref, no, sa, ideal = _recon_elevated(Nrx, dxt, float(h))
        fi = _focus_mag(ideal, ref, cfg.prf, cfg.abw, taper=TAPER).max()
        fn = _focus_mag(no, ref, cfg.prf, cfg.abw, taper=TAPER)
        fs = _focus_mag(sa, ref, cfg.prf, cfg.abw, taper=TAPER)
        pct_no.append(100 * fn.max() / fi); pct_sa.append(100 * fs.max() / fi)
        am_no.append(_ambiguity_db(fn)); am_sa.append(_ambiguity_db(fs))
        print(f"    h={h:5.0f} m | peak no/SATA {pct_no[-1]:5.0f}/{pct_sa[-1]:5.0f}%"
              f"   ambig no/SATA {am_no[-1]:6.1f}/{am_sa[-1]:6.1f} dB")

    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception as e:                        # pragma: no cover
        print("    (matplotlib unavailable)", e); return

    def _lab(ax, xs, ys, dy, color, fmt="{:.1f}"):
        for xi, yi in zip(xs, ys):
            ax.annotate(fmt.format(yi), (xi, yi), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=5.5, color=color)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.axhline(100, color="k", ls="--", lw=0.8, label="ideal")
    a1.plot(heights, pct_no, "C3o-", lw=1.3, label="no-SATA")
    a1.plot(heights, pct_sa, "C0s-", lw=1.3, label="+ SATA")
    _lab(a1, heights, pct_no, -9, "C3"); _lab(a1, heights, pct_sa, 5, "C0")
    a1.set_xlabel("terrain height at the range bin [m]"); a1.set_ylabel("focused peak [% of ideal]")
    a1.set_ylim(0, 115); a1.set_title("Peak recovery per range bin"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(heights, am_no, "C3o-", lw=1.3, label="no-SATA")
    a2.plot(heights, am_sa, "C0s-", lw=1.3, label="+ SATA")
    _lab(a2, heights, am_no, -9, "C3", fmt="{:.0f}"); _lab(a2, heights, am_sa, 6, "C0", fmt="{:.0f}")
    a2.set_xlabel("terrain height at the range bin [m]"); a2.set_ylabel("worst ambiguity [dB below peak]")
    # inverted axis (0 dB at bottom) with headroom above the -43 dB floor so the
    # +SATA value labels do not collide with the title.
    lo = min(min(am_no), min(am_sa))
    a2.set_ylim(3, lo - 7)
    a2.set_title("Ambiguity suppression per range bin"); a2.legend(loc="center right"); a2.grid(alpha=0.3)
    fig.suptitle(rf"Range ramp handled 1D per bin ($N_\mathrm{{rx}}={Nrx}$, $d_\mathrm{{xt}}={dxt:.0f}$ m)",
                 fontsize=12)
    fig.tight_layout()
    if save:
        print(f"    range-ramp-1D plot -> {_save_fig(fig, 'sata_range_ramp_1d', subdir='topography')}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1b. per-channel sub-Nyquist aliasing (the premise of the reconstruction)
# ---------------------------------------------------------------------------
def plot_channel_aliasing(Nrx=4, save=True):
    """Show that each receiver channel is sub-Nyquist: the azimuth Doppler band
    (abw) is wider than the per-channel PRF_op, so a single channel's spectrum
    is aliased (folded). The reconstruction is what unfolds the Nrx channels."""
    print("\n[1b] Per-channel sub-Nyquist aliasing")
    cfg, tracks, off = _build_single_target_cfg(Nrx, 0.0, 0.0)
    s = cfg.system
    ptg = cfg.scene.ptg
    sref = getRawData1D(ptg[None, :], tracks.ptx, tracks.ptx, tracks.vtx, tracks.vtx,
                        cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)
    ch = getRawData1D(ptg[None, :], tracks.ptx, tracks.prx[0], tracks.vtx, tracks.vrx[0],
                      cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)[::Nrx]
    print(f"    abw = {s.abw:.0f} Hz,  PRF_op = {cfg.PRF_op:.0f} Hz  ->  "
          f"aliasing factor abw/PRF_op = {s.abw/cfg.PRF_op:.2f}x")

    def spec(x, fs):
        X = np.abs(np.fft.fftshift(np.fft.fft(x)))
        fa = np.fft.fftshift(np.fft.fftfreq(len(x), d=1.0 / fs))
        return fa, X / X.max()

    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception as e:                        # pragma: no cover
        print("    (matplotlib unavailable)", e); return

    fa_r, S_r = spec(sref, cfg.prf)
    fa_c, S_c = spec(ch, cfg.PRF_op)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 5), sharex=False)
    a1.plot(fa_r, 20 * np.log10(S_r + 1e-6), "k", lw=0.8)
    a1.axvspan(-s.abw / 2, s.abw / 2, color="C2", alpha=0.15, label="Doppler band (abw)")
    a1.axvline(-cfg.prf / 2, color="gray", ls=":"); a1.axvline(cfg.prf / 2, color="gray", ls=":")
    a1.set_title(rf"Full PRF (reference): band fits inside $\pm$PRF/2  "
                 rf"(PRF={cfg.prf:.0f} Hz)")
    a1.set_ylim(-50, 3); a1.set_ylabel("dB"); a1.legend(fontsize=8, loc="upper right")
    a2.plot(fa_c, 20 * np.log10(S_c + 1e-6), "C3", lw=0.8)
    a2.axvline(-cfg.PRF_op / 2, color="gray", ls=":"); a2.axvline(cfg.PRF_op / 2, color="gray", ls=":")
    a2.set_title(rf"One channel at PRF_op={cfg.PRF_op:.0f} Hz: band ({s.abw:.0f} Hz) "
                 rf"> PRF_op $\Rightarrow$ ALIASED (spectrum folded/filled)")
    a2.set_ylim(-50, 3); a2.set_ylabel("dB"); a2.set_xlabel("Doppler frequency [Hz]")
    fig.tight_layout()
    if save:
        print(f"    aliasing plot saved -> {_save_fig(fig, 'sata_channel_aliasing', subdir='sub_nyquist')}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. azimuth ambiguities: appear WITHOUT SATA, suppressed WITH SATA
# ---------------------------------------------------------------------------
def plot_ambiguities(Nrx=4, dh=200.0, dxt_show=50.0,
                     dxt_sweep=(0, 10, 20, 50, 100, 150), save=True):
    """The topographic residual corrupts the multichannel unmixing, so the
    per-channel aliasing re-appears as azimuth AMBIGUITY replicas far from the
    target. SATA removes the residual and suppresses them.
    Two figures: (a) worst ambiguity vs cross-track baseline; (b) a wide IRF
    showing the replicas for one dxt."""
    print("\n[5] Azimuth ambiguities (no-SATA vs SATA)")

    # (a) ambiguity level vs dxt
    A_no, A_sa = [], []
    for dxt in dxt_sweep:
        cfg, ref, no, sa, _ = _recon_elevated(Nrx, float(dxt), dh)
        fn = _focus_mag(no, ref, cfg.prf, cfg.abw, taper=TAPER)
        fs = _focus_mag(sa, ref, cfg.prf, cfg.abw, taper=TAPER)
        A_no.append(_ambiguity_db(fn)); A_sa.append(_ambiguity_db(fs))
        print(f"    dxt={dxt:4d} m | worst ambiguity  no-SATA {A_no[-1]:6.1f} dB   "
              f"SATA {A_sa[-1]:6.1f} dB")

    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception as e:                        # pragma: no cover
        print("    (matplotlib unavailable)", e); return

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(dxt_sweep, A_no, "C3o-", lw=1.3, label="no-SATA")
    ax.plot(dxt_sweep, A_sa, "C0s-", lw=1.3, label="+ SATA")
    ax.set_xlabel(r"cross-track baseline $d_\mathrm{xt}$ [m]")
    ax.set_ylabel("worst azimuth ambiguity [dB below peak]")
    ax.set_title(rf"Ambiguity vs baseline ($N_\mathrm{{rx}}={Nrx}$, $\Delta h={dh:.0f}$ m)")
    ax.invert_yaxis(); ax.legend(); ax.grid(alpha=0.3)
    if save:
        print(f"    ambiguity-vs-dxt plot -> {_save_fig(fig, 'sata_ambiguity_vs_dxt', subdir='ambiguity')}")
    plt.close(fig)

    # (b) wide IRF showing the replicas for one dxt
    cfg, ref, no, sa, _ = _recon_elevated(Nrx, dxt_show, dh)
    fr = _focus_mag(ref, ref, cfg.prf, cfg.abw, taper=TAPER)
    fn = _focus_mag(no, ref, cfg.prf, cfg.abw, taper=TAPER)
    fs = _focus_mag(sa, ref, cfg.prf, cfg.abw, taper=TAPER)
    n = len(fr); i0 = n // 2
    # locate the strongest replica (for the annotation / window)
    m = np.ones(n, bool); m[i0 - 100:i0 + 100] = False
    rep = int(np.argmax(fn * m))
    W = int(min(2.2 * abs(rep - i0) + 300, n // 2 - 1))
    w = slice(i0 - W, i0 + W); x = np.arange(2 * W) - W
    db = lambda f: 20 * np.log10(f / fr.max() + 1e-12)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(x, db(fr)[w], "k", lw=0.9, label="reference")
    ax.plot(x, db(fn)[w], "C3", lw=0.9, label="no SATA")
    ax.plot(x, db(fs)[w], "C0--", lw=1.1, label="+ SATA")
    for xa in (rep - i0, i0 - rep):
        ax.annotate("ambiguity", xy=(xa, db(fn)[i0 + xa]), xytext=(xa, 12),
                    ha="center", fontsize=8, color="C3",
                    arrowprops=dict(arrowstyle="->", color="C3"))
    ax.set_ylim(-55, 18); ax.set_xlabel("azimuth sample (around target)")
    ax.set_ylabel("amplitude [dB]")
    ax.set_title(rf"Azimuth ambiguities ($N_\mathrm{{rx}}={Nrx}$, "
                 rf"$d_\mathrm{{xt}}={dxt_show:.0f}$ m, $\Delta h={dh:.0f}$ m)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    if save:
        print(f"    ambiguity IRF plot   -> {_save_fig(fig, 'sata_ambiguity_irf', subdir='ambiguity')}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. baseline robustness: SATA across many array configurations
# ---------------------------------------------------------------------------
def plot_baseline_robustness(save=True):
    """Run SATA over a range of baseline configurations and show it recovers the
    peak and suppresses ambiguities in all the realistic ones."""
    print("\n[6] Baseline robustness (peak %% of ideal, worst ambiguity dB)")
    # a deliberately non-uniform (irregular) array to stress the algorithm
    nonunif = ArrayGeometry(bat=np.array([0., 70., 190., 360.]),
                            bxt=np.array([-120., -30., 40., 130.]))
    # (label, Nrx, dx, dxt, bat_offset, array_override)
    cases = [
        ("dxt=50",       4, 100, 50,  0.0,    None),
        ("dxt=150",      4, 100, 150, 0.0,    None),
        ("dxt=300",      4, 100, 300, 0.0,    None),
        ("DPCA dx=11",   4, 11,  100, 0.0,    None),
        ("DPCA offset",  4, 11,  100, 11 / 2, None),
        ("large dx=200", 4, 200, 100, 0.0,    None),
        ("non-uniform",  4, 0,   0,   0.0,    nonunif),
        ("Nrx=2",        2, 100, 100, 0.0,    None),
        ("Nrx=6",        6, 100, 100, 0.0,    None),
    ]
    labels, pk_no, pk_sa, am_no, am_sa = [], [], [], [], []
    for lbl, Nrx, dx, dxt, boff, arr in cases:
        cfg, ref, no, sa, ideal = _recon_elevated(Nrx, dxt, 200.0, dx=dx,
                                                  bat_offset=boff, array=arr)
        fi = _focus_mag(ideal, ref, cfg.prf, cfg.abw, taper=TAPER)
        fn = _focus_mag(no, ref, cfg.prf, cfg.abw, taper=TAPER)
        fs = _focus_mag(sa, ref, cfg.prf, cfg.abw, taper=TAPER)
        labels.append(lbl)
        pk_no.append(100 * fn.max() / fi.max()); pk_sa.append(100 * fs.max() / fi.max())
        am_no.append(_ambiguity_db(fn)); am_sa.append(_ambiguity_db(fs))
        print(f"    {lbl:<12} | peak no/SATA {pk_no[-1]:5.0f}/{pk_sa[-1]:5.0f}%   "
              f"ambig no/SATA {am_no[-1]:6.1f}/{am_sa[-1]:6.1f} dB")

    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception as e:                        # pragma: no cover
        print("    (matplotlib unavailable)", e); return

    xpos = np.arange(len(labels)); wbar = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(xpos - wbar / 2, np.clip(pk_no, 0, 140), wbar, color="C3", label="no-SATA")
    a1.bar(xpos + wbar / 2, np.clip(pk_sa, 0, 140), wbar, color="C0", label="+ SATA")
    a1.axhline(100, color="k", ls="--", lw=0.8)
    a1.set_ylim(0, 145)
    a1.set_ylabel("focused peak [% of ideal]"); a1.set_title("Peak recovery")
    a1.set_xticks(xpos); a1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    a1.legend(fontsize=8); a1.grid(alpha=0.3, axis="y")
    # flag any bar that was clipped (e.g. ill-conditioned reconstruction)
    for k, (pn, ps) in enumerate(zip(pk_no, pk_sa)):
        if pn > 140 or ps > 140:
            a1.text(k, 141, "off-scale\n(recon.\nill-cond.)", ha="center", va="top",
                    fontsize=6, color="C3")
    a2.bar(xpos - wbar / 2, am_no, wbar, color="C3", label="no-SATA")
    a2.bar(xpos + wbar / 2, am_sa, wbar, color="C0", label="+ SATA")
    a2.set_ylabel("worst ambiguity [dB below peak]"); a2.set_title("Ambiguity suppression")
    a2.set_xticks(xpos); a2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    a2.legend(fontsize=8); a2.grid(alpha=0.3, axis="y")
    fig.suptitle("SATA robustness across baseline configurations", fontsize=12)
    fig.tight_layout()
    if save:
        print(f"    robustness plot -> {_save_fig(fig, 'sata_baseline_robustness', subdir='baselines')}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. exhaustive baseline grid: SATA over the whole (Nrx x dxt) space
# ---------------------------------------------------------------------------
def baseline_grid(Nrx_list=(2, 3, 4, 5, 6), dxt_list=(0, 20, 50, 100, 150, 200, 300),
                  dh=200.0, save=True):
    """Exhaustive sweep: reconstruct one elevated target for every (Nrx, dxt)
    pair and record the worst azimuth ambiguity with and without SATA. Two
    heatmaps: no-SATA (ambiguities grow with dxt and Nrx) vs +SATA (flat, deep).
    Proves the 1D algorithm across the whole baseline space."""
    print("\n[7] Exhaustive baseline grid (Nrx x dxt) -- worst ambiguity [dB]")
    A_no = np.full((len(Nrx_list), len(dxt_list)), np.nan)
    A_sa = np.full_like(A_no, np.nan)
    for i, Nrx in enumerate(Nrx_list):
        row = []
        for j, dxt in enumerate(dxt_list):
            cfg, ref, no, sa, _ = _recon_elevated(Nrx, float(dxt), dh)
            fn = _focus_mag(no, ref, cfg.prf, cfg.abw, taper=TAPER)
            fs = _focus_mag(sa, ref, cfg.prf, cfg.abw, taper=TAPER)
            A_no[i, j] = _ambiguity_db(fn); A_sa[i, j] = _ambiguity_db(fs)
            row.append(f"{A_no[i,j]:5.0f}/{A_sa[i,j]:4.0f}")
        print(f"    Nrx={Nrx}: " + " ".join(row))
    print("    (each cell: no-SATA / +SATA worst ambiguity in dB)")

    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception as e:                        # pragma: no cover
        print("    (matplotlib unavailable)", e); return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    vmin, vmax = -50, 0
    for ax, A, ttl in ((axes[0], A_no, "no-SATA"), (axes[1], A_sa, "+ SATA")):
        im = ax.imshow(A, aspect="auto", origin="lower", cmap="RdYlGn_r",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(dxt_list))); ax.set_xticklabels(dxt_list)
        ax.set_yticks(range(len(Nrx_list))); ax.set_yticklabels(Nrx_list)
        ax.set_xlabel(r"cross-track baseline $d_\mathrm{xt}$ [m]")
        ax.set_ylabel(r"$N_\mathrm{rx}$"); ax.set_title(ttl)
        for i in range(len(Nrx_list)):
            for j in range(len(dxt_list)):
                ax.text(j, i, f"{A[i,j]:.0f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="worst ambiguity [dB]")
    fig.suptitle("SATA over the baseline grid: no-SATA (bad, red) vs +SATA (good, green)",
                 fontsize=12)
    fig.tight_layout()
    if save:
        print(f"    baseline-grid plot -> {_save_fig(fig, 'sata_baseline_grid', subdir='baselines')}")
    plt.close(fig)


def main():
    kernel_selftest()
    plot_channel_aliasing()
    integration_sweep()
    plot_single_target_irf()
    plot_irf_4panel()
    plot_toporamp_4panel()
    plot_azimuth_topo()
    plot_range_ramp_1d()
    plot_zoom_vs_wide()
    plot_ambiguities()
    plot_baseline_robustness()
    baseline_grid()
    print("\ndone")


if __name__ == "__main__":
    main()