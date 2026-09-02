# -*- coding: utf-8 -*-
"""
runs/core/run_case1_raw_stft.py

Raw-data + after-STFT plots for the Case 1 DPCA scene (run_case1_dpca_dxsweep.py's
5 iso-range targets on a height ramp), swept over ALL 5 target spacings
(close/few-samples to far/bunch-samples), alpha fixed, for ONE channel, at
BOTH rate cases (Nyquist-satisfying full PRF, and the sub-Nyquist per-channel
PRF_op) -- same structure as run_reference_stft.py.

SCENE (verbatim from run_case1_dpca_dxsweep.py's build()): 5 iso-range
targets at dx_k = k * S_m, k in {-2,-1,0,1,2}, S_m = S_samp * v/PRF (spacing
in focused samples). Height is a ramp tied to position: dh_k = (dx_k -
dx_min) * tan(alpha), left-most target on the ground. DPCA (uniform) array
spacing, Nrx=4, b_xt=20 m fixed.

SPACINGS swept: 50, 100, 300, 600, 1000 samples (close/few -> far/bunch).
ALPHA fixed at 1.5 deg.

RATE CASES (per spacing), for channel CHANNEL:
  "nyquist" -- getRawData1D() at the FULL total PRF (cfg.prf = 2000 Hz),
               undecimated. abw ~= 1235 Hz < 2000 Hz, so this rate already
               satisfies Nyquist with the default SystemParams (no PRF
               override needed, unlike the wl=0.031 case in
               run_reference_stft.py).
  "channel" -- getRawData1D() decimated by Nrx (i.e. sar.generate_channels()
               semantics), at PRF_op = 500 Hz -- sub-Nyquist by design.

DOMAINS (per rate case):
  time_domain.png -- |raw| vs time, no processing.
  after_stft.png  -- the REAL sar_recon.sata.sata_1d(debug_center=True,
                     analysis_window="rectangular", sata_osf=1), capturing
                     the un-tapered, minimally-padded sub-aperture spectrum
                     of the window closest to the aperture centre. Nothing
                     reimplemented; delta_C0_array is all zeros (no
                     correction applied, we only want the internal spectrum).

Uses ptgs = cfg.scene.points[1:] (the 5 explicit ramp targets only, same as
run_case1_dpca_dxsweep.py -- excludes the arbitrary h0=0 central point).

Run:
    cd sar_reconstruction
    PYTHONPATH=. python ../runs/core/run_case1_raw_stft.py

Figures saved to runs/core/plots/run_case1_raw_stft/spacing<S>/<rate_case>/
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from _common import save_fig  # noqa: E402

import sar_recon as sar  # noqa: E402
from sar_recon.config import (SystemParams, Scene, ArrayGeometry, prf_from_dpca,
                              integration_time, build_time_axis)  # noqa: E402
from sar_recon.geometry import build_platform_tracks  # noqa: E402
from sar_recon.signal_model import getRawData1D  # noqa: E402
from sar_recon.sata import sata_1d  # noqa: E402

# ---------------------------------------------------------------------------
# PARAMETERS (mirrors run_case1_dpca_dxsweep.py)
# ---------------------------------------------------------------------------
CHANNEL = 0
SATA_OSF = 1                # minimal zero-padding (see run_reference_stft.py)
ALPHA = 1.5                 # ramp inclination [deg], fixed
SPACINGS = [50, 100, 300, 600, 1000]   # target spacing [samples]: close -> far

RDELAY = 0.0051115753
sysp = SystemParams()
_base = Scene(rDelay=RDELAY, c0=sysp.c0, h0=0.0)
R0, H, Y0 = _base.r0, _base.H, _base.y0

PRF_OP = 500.0
SAMP = sysp.vs / 2000.0     # focused-sample spacing [m] at the total PRF (2000 Hz)
DPCA_DX = 2.0 * sysp.vs / (4 * PRF_OP)
BXT = 20.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
PLOTS_ROOT = os.path.join(SCRIPT_DIR, "plots", SCRIPT_NAME)


def build(alpha, S_samp, Nrx=4):
    """Verbatim from run_case1_dpca_dxsweep.py."""
    Sm = S_samp * SAMP
    pos = np.array([-2, -1, 0, 1, 2]) * Sm
    hts = (pos - pos.min()) * np.tan(np.deg2rad(alpha))
    extra = []
    for dxm, dh in zip(pos, hts):
        y_t = np.sqrt(R0 ** 2 - (H - dh) ** 2)
        extra.append((float(dxm), float(y_t - Y0), float(dh)))
    scene = Scene(rDelay=RDELAY, c0=sysp.c0, h0=0.0, extra_offsets=tuple(extra))
    array = ArrayGeometry.linear(Nrx, DPCA_DX, BXT)
    prf, PRFop = prf_from_dpca(sysp, Nrx, DPCA_DX)
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.0 * integration_time(sysp, scene))
    cfg = sar.ExperimentConfig(name="dpcadx", system=sysp, scene=scene, array=array,
                               prf=prf, PRF_op=PRFop, Na=Na, Na_ch=Nc, ta=ta, plots_dir=None)
    return cfg, build_platform_tracks(cfg), pos, hts


def plot_time_domain(x, t, save_dir, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, np.abs(x), lw=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    save_fig(fig, save_dir, "time_domain", vector=False)
    print("    time domain ->", os.path.join(save_dir, "time_domain.png"))


def plot_after_stft(x, prf, rref, v, wl, r, save_dir, title):
    _, dbg = sata_1d(
        x, np.zeros(len(x)), rref=rref, prf=prf, v=v, wl=wl, r=r,
        squint=0.0, Nsb=1, inverse=True, sata_osf=SATA_OSF,
        verbose=False, debug_center=True, analysis_window="rectangular",
    )
    if dbg is None:
        print("    after STFT  -> skipped (Tsubeff <= 2)")
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


def run_spacing(S_samp):
    cfg, tr, pos, hts = build(ALPHA, S_samp)
    s = cfg.system
    ptgs = cfg.scene.points[1:]     # the 5 explicit ramp targets only
    print(f"\n[spacing={S_samp}] Sm={S_samp * SAMP:.1f} m  dh_max={hts.max():.1f} m  "
          f"abw={cfg.abw:.1f} Hz  prf(total)={cfg.prf:.1f} Hz  PRF_op={cfg.PRF_op:.1f} Hz")

    label = f"spacing{S_samp}"

    # --- RATE CASE 1: "nyquist" -- full total PRF, undecimated ------------
    save_dir = os.path.join(PLOTS_ROOT, label, "nyquist")
    os.makedirs(save_dir, exist_ok=True)
    x_nyq = getRawData1D(
        ptgs, tr.ptx, tr.prx[CHANNEL], tr.vtx, tr.vrx[CHANNEL], cfg.ta,
        cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf,
    )
    plot_time_domain(x_nyq, cfg.ta, save_dir,
                     f"Case1 DPCA, spacing={S_samp} samp, alpha={ALPHA} deg, "
                     f"channel {CHANNEL}, prf(total)={cfg.prf:.0f} Hz -- time domain")
    plot_after_stft(x_nyq, cfg.prf, cfg.scene.r0, s.vs, s.wl, cfg.scene.r0, save_dir,
                    f"Case1 DPCA, spacing={S_samp} samp, channel {CHANNEL}, "
                    f"prf(total)={cfg.prf:.0f} Hz -- after STFT (sata_1d)")

    # --- RATE CASE 2: "channel" -- PRF_op, decimated -----------------------
    save_dir = os.path.join(PLOTS_ROOT, label, "channel")
    os.makedirs(save_dir, exist_ok=True)
    x_ch = getRawData1D(
        ptgs, tr.ptx, tr.prx[CHANNEL], tr.vtx, tr.vrx[CHANNEL], cfg.ta,
        cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf,
    )[::cfg.Nrx]
    ta_ch = cfg.ta[::cfg.Nrx][:len(x_ch)]
    plot_time_domain(x_ch, ta_ch, save_dir,
                     f"Case1 DPCA, spacing={S_samp} samp, alpha={ALPHA} deg, "
                     f"channel {CHANNEL}, PRF_op={cfg.PRF_op:.0f} Hz (sub-Nyquist) "
                     f"-- time domain")
    plot_after_stft(x_ch, cfg.PRF_op, cfg.scene.r0, s.vs, s.wl, cfg.scene.r0, save_dir,
                    f"Case1 DPCA, spacing={S_samp} samp, channel {CHANNEL}, "
                    f"PRF_op={cfg.PRF_op:.0f} Hz -- after STFT (sata_1d)")


def main():
    for S in SPACINGS:
        run_spacing(S)


if __name__ == "__main__":
    main()
