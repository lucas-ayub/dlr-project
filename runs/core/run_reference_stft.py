# -*- coding: utf-8 -*-
"""
runs/core/run_reference_stft.py

2 (scenes) x 2 (rate cases) x 2 (domains) = 8 plots:

  SCENES     : "single" (1 target) / "5targets" (5 targets)
  RATE CASES : "nyquist" -- the raw signal for channel CHANNEL's bistatic
                            geometry, built with getRawData1D() at the FULL
                            rate cfg.prf/cfg.ta (undecimated, satisfies
                            Nyquist: cfg.prf >= abw).
               "channel" -- that SAME channel, but as it's actually acquired
                            in this multichannel system: sar.generate_channels()
                            (imported, unmodified), decimated to the
                            per-channel operating rate PRF_op = cfg.prf/Nrx
                            (sub-Nyquist by design).
  DOMAINS    : time domain (|x| vs time) and after an STFT -- the ACTUAL
               sub-aperture spectrum SATA computes, obtained by calling the
               real sar_recon.sata.sata_1d() with a new opt-in
               `debug_center=True` flag (default False, so every other
               caller of sata_1d is unaffected) that captures
               `spec = np.fft.fft(buf)` -- SATA's own sizing (Tsubeff, Nzp)
               and its own triangular analysis window -- for the ONE
               sub-aperture window closest to the centre of the aperture.
               Nothing about SATA's math is reimplemented here.

Both rate cases use ONLY channel CHANNEL (no multichannel combination) and
the exact same imported functions (getRawData1D, sar.generate_channels,
sata_1d) -- nothing is reimplemented.

Why this matters for SATA: at a given instant, every target in the beam has
a DIFFERENT instantaneous Doppler frequency (depends on its along-track
offset). That's the property SATA exploits -- each Doppler bin maps to a
distinct azimuth position (`azpos` in sata_1d). The STFT plots make that
mapping visible: targets mixed together in time domain should show up as
separate peaks in frequency, resolution permitting.

Uses the real pipeline (sar_recon.generate_channels / build_platform_tracks
/ signal_model.getRawData1D) via the lightweight config builder in
runs/_common.py -- same helper the other runs/ scripts use for isolated
checks.

Run:
    cd sar_reconstruction
    PYTHONPATH=. python ../runs/core/run_reference_stft.py

Figures saved to runs/core/plots/run_reference_stft/<scene>/<rate_case>/
(png), matching the convention used by the other scripts here (e.g.
run_sata.py -> plots/run_sata/).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from _common import build_cfg, save_fig  # noqa: E402

import sar_recon as sar  # noqa: E402
from sar_recon.config import SystemParams  # noqa: E402
from sar_recon.signal_model import getRawData1D  # noqa: E402
from sar_recon.sata import sata_1d  # noqa: E402

# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------
NRX = 4          # number of receiver channels
CHANNEL = 0      # which channel to plot (0-indexed)
BXT = 100.0      # cross-track baseline of `CHANNEL` [m] (0 for the others)
WL = 0.031       # wavelength [m]
SATA_OSF = 1     # sata_1d's own sata_osf (zero-padding factor for its Nzp).
                 # 1 = minimal zero-padding (just up to the next power of 2),
                 # so the plotted spectrum has fewer points and the line
                 # segments between them stay visible (less "smooth"/
                 # interpolated-looking); raise it for a finer, smoother curve.

# The FULL PRF (cfg.prf) must satisfy Nyquist for the processed Doppler
# bandwidth abw = 2*ve/La (i.e. cfg.prf >= abw). `build_cfg`'s `prf_fixed`
# default (2000 Hz) only satisfies this for the SystemParams default
# wl=0.25; for a different WL, abw changes, so PRF_FIXED is derived from it
# here (10% margin, rounded up to the next 100 Hz).
_abw_probe = SystemParams(wl=WL).abw
PRF_FIXED = float(np.ceil(_abw_probe * 1.1 / 100.0) * 100.0)

# Scenes: (dx, dy, dh) offsets [m] relative to the central target.
SCENES = {
    "single":   (),
    "5targets": ((-1200.0, 0.0, 0.0), (-600.0, 0.0, 0.0),
                (600.0, 0.0, 0.0), (1200.0, 0.0, 0.0)),
}

# Plots go into runs/core/plots/run_reference_stft/<scene>/<rate_case>/,
# same convention as the other scripts in this folder (run_sata.py, ...).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
PLOTS_ROOT = os.path.join(SCRIPT_DIR, "plots", SCRIPT_NAME)


def plot_time_domain(x, t, save_dir, title, n_targets):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, np.abs(x), lw=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    save_fig(fig, save_dir, "time_domain", vector=False)
    print("    time domain ->", os.path.join(save_dir, "time_domain.png"))


def plot_after_stft(x, prf, rref, v, wl, r, save_dir, title, n_targets):
    """
    Runs the REAL sata_1d() (delta_C0_array all zeros -> no correction
    applied, we only want its internal sizing/spectrum) with
    debug_center=True, and plots the captured sub-aperture spectrum -- SATA's
    own Tsubeff sizing, nothing reimplemented. analysis_window="rectangular"
    and sata_osf=1 (SATA_OSF) keep the spectrum un-tapered and minimally
    zero-padded (narrow lobes, few points, visible straight segments between
    them) instead of the smoother, triangular-windowed/heavily-padded look
    sata_1d uses by default for its own reconstruction.
    """
    _, dbg = sata_1d(
        x, np.zeros(len(x)), rref=rref, prf=prf, v=v, wl=wl, r=r,
        squint=0.0, Nsb=1, inverse=True, sata_osf=SATA_OSF,
        verbose=False, debug_center=True, analysis_window="rectangular",
    )
    if dbg is None:
        print("    after STFT  -> skipped (Tsubeff <= 2, sata_1d returned no correction)")
        return

    spec = np.fft.fftshift(dbg["spec"])
    Nzp = dbg["Nzp"]
    f = np.fft.fftshift(np.fft.fftfreq(Nzp, d=1.0 / prf))
    db = 20 * np.log10(np.abs(spec) / np.abs(spec).max() + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(f, db, lw=1.0)
    ax.set_xlabel("Doppler frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(-40, 3)
    ax.set_title(title + f" (Tsubeff={dbg['Tsubeff']}, Nzp={Nzp})")
    ax.grid(alpha=0.3)
    save_fig(fig, save_dir, "after_stft", vector=False)
    print("    after STFT  ->", os.path.join(save_dir, "after_stft.png"),
          f" (Tsubeff={dbg['Tsubeff']}, Nzp={Nzp})")


def run_scene(scene_name, extra_offsets):
    cfg, tracks, scene = build_cfg(wl=WL, bxt=BXT, channel=CHANNEL, nrx=NRX,
                                   extra_offsets=extra_offsets, prf_fixed=PRF_FIXED)
    n_targets = len(scene.points)
    assert cfg.prf >= cfg.abw, (
        "full PRF still below abw -- raw signal would be aliased; "
        "increase the margin in PRF_FIXED above.")
    print(f"\n[{scene_name}] Nrx={cfg.Nrx}  abw={cfg.abw:.1f} Hz  "
          f"prf(full/Nyquist)={cfg.prf:.1f} Hz  PRF_op(channel)={cfg.PRF_op:.1f} Hz  "
          f"n_targets={n_targets}")

    # --- RATE CASE 1: "nyquist" -- channel CHANNEL's geometry, built at the
    # FULL rate cfg.prf via getRawData1D directly (undecimated, no aliasing).
    save_dir = os.path.join(PLOTS_ROOT, scene_name, "nyquist")
    os.makedirs(save_dir, exist_ok=True)
    x_nyq = getRawData1D(
        cfg.scene.points, tracks.ptx, tracks.prx[CHANNEL],
        tracks.vtx, tracks.vrx[CHANNEL], cfg.ta,
        cfg.sq_tx, cfg.sq_rx[CHANNEL], cfg.theta_tx, cfg.theta_rx[CHANNEL],
        cfg.system.wl, cfg.prf,
    )
    plot_time_domain(x_nyq, cfg.ta, save_dir,
                     f"Channel {CHANNEL}, Nyquist rate (prf={cfg.prf:.0f} Hz), "
                     f"{n_targets} target{'s' if n_targets != 1 else ''} "
                     f"-- time domain", n_targets)
    plot_after_stft(x_nyq, cfg.prf, cfg.scene.r0, cfg.system.vs, cfg.system.wl,
                    cfg.scene.r0, save_dir,
                    f"Channel {CHANNEL}, Nyquist rate, {n_targets} target"
                    f"{'s' if n_targets != 1 else ''} -- after STFT (sata_1d)",
                    n_targets)

    # --- RATE CASE 2: "channel" -- the SAME channel as it's actually
    # acquired: sar.generate_channels() (imported, unmodified), decimated to
    # PRF_op = cfg.prf / Nrx (sub-Nyquist by design of this system).
    save_dir = os.path.join(PLOTS_ROOT, scene_name, "channel")
    os.makedirs(save_dir, exist_ok=True)
    s_ch = sar.generate_channels(cfg, tracks)     # [Nrx, Na_ch]
    x_ch = s_ch[CHANNEL, :]
    ta_ch = cfg.ta[::cfg.Nrx][:len(x_ch)]
    plot_time_domain(x_ch, ta_ch, save_dir,
                     f"Channel {CHANNEL}, PRF_op={cfg.PRF_op:.0f} Hz "
                     f"(sub-Nyquist), {n_targets} target"
                     f"{'s' if n_targets != 1 else ''} -- time domain", n_targets)
    plot_after_stft(x_ch, cfg.PRF_op, cfg.scene.r0, cfg.system.vs, cfg.system.wl,
                    cfg.scene.r0, save_dir,
                    f"Channel {CHANNEL}, PRF_op={cfg.PRF_op:.0f} Hz, "
                    f"{n_targets} target{'s' if n_targets != 1 else ''} "
                    f"-- after STFT (sata_1d)", n_targets)


def main():
    for scene_name, extra_offsets in SCENES.items():
        run_scene(scene_name, extra_offsets)


if __name__ == "__main__":
    main()