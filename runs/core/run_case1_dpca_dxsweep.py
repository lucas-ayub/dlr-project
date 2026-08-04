#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Case 1 --- DPCA target-spacing sweep with the height varying with dx.
Reproduces the figures in sata_case1_dpca_dxsweep.pdf.

Five iso-range targets at dx_k = k * S_m, k in {-2,-1,0,1,2}, where the spacing is
set in focused samples (S_m = S_samp * v/PRF). The height is a genuine ramp tied to
position, with the left-most target on the ground:
    dh_k = (dx_k - dx_min) * tan(alpha) = (k+2) * S_m * tan(alpha)
DPCA (uniform) sampling: along-track spacing db_at = 2v/(Nrx*PRF_op) ~ 7.69 m,
which fixes PRF_op = 500 Hz (total PRF = 2000 Hz). b_xt = 20 m fixed, Nrx = 4.

HOW TO RUN (place in runs/core/, next to run_sata.py):
    python runs/core/run_case1_dpca_dxsweep.py
Output PNGs (sata_c1dpcadx_a{alpha*10}_dx{spacing}.png) go to $PLOT_OUT
(default: runs/core/plots/dpca_dxsweep/). USE_LATEX=0 falls back to mathtext.
"""
import os
import sys
import numpy as np
import matplotlib

USE_LATEX = os.environ.get("USE_LATEX", "1") == "1"
if USE_LATEX:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex", "text.usetex": True, "font.family": "serif",
        "pgf.rcfonts": False, "pgf.preamble": r"\usepackage{amsmath,amssymb}"})
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- locate the sar_recon package (runs/core/ and sar_reconstruction/ are siblings) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, "..", "..", "sar_reconstruction"))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import sar_recon as sar
from sar_recon.config import (SystemParams, Scene, ArrayGeometry, prf_from_dpca,
                              integration_time, build_time_axis)
from sar_recon.geometry import build_platform_tracks
from sar_recon.signal_model import getRawData1D
from sar_recon.analysis import zoom1Dpeak
from sar_recon.sata import sata_channels

OUT = os.environ.get("PLOT_OUT", os.path.join(_HERE, "plots", "dpca_dxsweep"))
os.makedirs(OUT, exist_ok=True)

RDELAY = 0.0051115753
sysp = SystemParams()
_base = Scene(rDelay=RDELAY, c0=sysp.c0, h0=0.0)
R0, H, Y0 = _base.r0, _base.H, _base.y0

PRF = 2000.0                             # total (fixed) PRF [Hz]
PRF_OP = 500.0                           # per-channel operational PRF [Hz]
SAMP = sysp.vs / PRF                      # focused-sample spacing [m] = v / PRF
DPCA_DX = 2.0 * sysp.vs / (4 * PRF_OP)   # along-track DPCA spacing (uniform sampling)
BXT = 20.0                               # fixed cross-track baseline [m]

ALPHAS = [0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0]   # ramp inclinations [deg]
SPACINGS = [50, 100, 300, 600, 1000]           # target spacing [samples]


def build(alpha, S_samp, Nrx=4):
    Sm = S_samp * SAMP
    pos = np.array([-2, -1, 0, 1, 2]) * Sm
    hts = (pos - pos.min()) * np.tan(np.deg2rad(alpha))   # height varies WITH dx (ramp)
    extra = []
    for dxm, dh in zip(pos, hts):
        y_t = np.sqrt(R0 ** 2 - (H - dh) ** 2)
        extra.append((float(dxm), float(y_t - Y0), float(dh)))
    scene = Scene(rDelay=RDELAY, c0=sysp.c0, h0=0.0, extra_offsets=tuple(extra))
    array = ArrayGeometry.linear(Nrx, DPCA_DX, BXT)       # along-track = DPCA spacing
    prf, PRFop = prf_from_dpca(sysp, Nrx, DPCA_DX)
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.0 * integration_time(sysp, scene))
    cfg = sar.ExperimentConfig(name="dpcadx", system=sysp, scene=scene, array=array,
                               prf=prf, PRF_op=PRFop, Na=Na, Na_ch=Nc, ta=ta, plots_dir=None)
    return cfg, build_platform_tracks(cfg), pos, hts


def panel(alpha, S_samp):
    cfg, tr, pos, hts = build(alpha, S_samp)
    s = cfg.system; Na, prf, abw, ta = cfg.Na, cfg.prf, cfg.abw, cfg.ta
    ptgs = cfg.scene.points[1:]
    sref1 = getRawData1D(cfg.scene.ptg[None, :], tr.ptx, tr.ptx, tr.vtx, tr.vtx, cfg.ta,
                         cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)
    sig = getRawData1D(ptgs, tr.ptx, tr.ptx, tr.vtx, tr.vtx, cfg.ta, cfg.sq_tx, cfg.sq_tx,
                       cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)
    s_ch = np.zeros([cfg.Nrx, cfg.Na_ch], complex)
    for ii in range(cfg.Nrx):
        s_ch[ii] = getRawData1D(ptgs, tr.ptx, tr.prx[ii], tr.vtx, tr.vrx[ii], cfg.ta, cfg.sq_tx,
                                cfg.sq_tx, cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)[::cfg.Nrx]
    no = sar.reconstruct(cfg, tr, s_ch.copy()); sa = sar.reconstruct(cfg, tr, sata_channels(cfg, tr, s_ch.copy()))
    fc = lambda x: np.roll(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(sref1))), Na // 2)
    Fr, Fn, Fs = fc(sig), fc(no), fc(sa); pmax = np.abs(Fr).max()
    ndb = lambda z: 20 * np.log10(np.abs(z) / pmax + 1e-12)
    fa = np.roll((np.arange(Na) / Na - 0.5) * prf, Na // 2); band = np.abs(fa) <= 0.5 * abw
    def dph(x):
        d = np.angle(np.fft.fft(x) * np.conj(np.fft.fft(sig)), deg=True); d[~band] = np.nan; return d
    c = Na // 2; tsamp = [c + int(round(p / SAMP)) for p in pos]
    mask = np.ones(Na, bool)
    for i in tsamp: mask[max(0, i - 160):min(Na, i + 160)] = False
    amb_no = ndb(Fn)[mask].max(); amb_sa = ndb(Fs)[mask].max()
    sn, ss = np.nanstd(dph(no)[band]), np.nanstd(dph(sa)[band])
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(r"SATA (\textbf{DPCA}), $N_{rx}=4$, PRF=%.0f Hz, $B_a$=%.0f Hz, "
                 r"$\Delta b_{at}$=%.2f m, $\Delta b_{xt}$=%.0f m, $\alpha=%.1f^\circ$, "
                 r"spacing $=%d$ samp ($%.0f$ m), $\Delta h_{\max}$=%.0f m"
                 % (prf, abw, DPCA_DX, BXT, alpha, S_samp, S_samp * SAMP, hts.max()), fontsize=9)
    ax[0, 0].plot(ta, np.abs(sig), "k", lw=1.0, label="reference (5 targets)")
    ax[0, 0].plot(ta, np.abs(no), "C3", lw=0.8, label="no-SATA"); ax[0, 0].plot(ta, np.abs(sa), "C0", lw=0.8, ls="--", label="+SATA")
    ax[0, 0].set_xlabel("Time [s]"); ax[0, 0].set_ylabel("Amplitude"); ax[0, 0].grid(alpha=0.3); ax[0, 0].legend(fontsize="small")
    ax[0, 1].plot(ta, ndb(Fr), "k", lw=0.7, label="reference"); ax[0, 1].plot(ta, ndb(Fn), "C3", lw=0.7, label="no-SATA")
    ax[0, 1].plot(ta, ndb(Fs), "C0", lw=0.8, ls="--", label="+SATA")
    for i in tsamp: ax[0, 1].axvline(ta[i], color="C2", ls=":", lw=0.5)
    ax[0, 1].set_ylim([-60, 3]); ax[0, 1].set_xlabel("Time [s]"); ax[0, 1].set_ylabel("[dB]")
    ax[0, 1].grid(alpha=0.3); ax[0, 1].legend(fontsize="small")
    ax[0, 1].set_title(r"focused image (dotted = targets; worst amb: no-SATA %.1f dB, +SATA %.1f dB)" % (amb_no, amb_sa), fontsize=8)
    Nz = int(16 * prf / abw); zpf = 64; taz = (np.arange(2 * Nz * zpf) - Nz * zpf) / prf / zpf * 1e3
    zpk = np.abs(zoom1Dpeak(Fr, Nz, zpf)).max(); zdb = lambda F: 20 * np.log10(np.abs(zoom1Dpeak(F, Nz, zpf)) / zpk + 1e-12)
    ax[1, 0].plot(taz, zdb(Fr), "k", lw=1.2, label="reference"); ax[1, 0].plot(taz, zdb(Fn), "C3", lw=0.9, label="no-SATA")
    ax[1, 0].plot(taz, zdb(Fs), "C0", lw=1.1, ls="--", label="+SATA")
    ax[1, 0].set_xlim(-12, 12); ax[1, 0].set_ylim([-45, 2]); ax[1, 0].set_xlabel("Time [ms]"); ax[1, 0].set_ylabel("[dB]")
    ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize="small"); ax[1, 0].set_title("zoomed IRF (central target)", fontsize=9)
    ax[1, 1].plot(fa, dph(no), "C3", lw=0.7, label="no-SATA"); ax[1, 1].plot(fa, dph(sa), "C0", lw=0.9, label="+SATA")
    ax[1, 1].axvline(abw / 2, color="r", ls="-."); ax[1, 1].axvline(-abw / 2, color="r", ls="-.")
    ax[1, 1].set_xlabel("Doppler freq [Hz]"); ax[1, 1].set_ylabel("[deg]"); ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize="small")
    ax[1, 1].set_title(r"phase error (in-band std: no-SATA $%.1f^\circ$, +SATA $%.1f^\circ$)" % (sn, ss), fontsize=8)
    fig.tight_layout(); n = int(round(alpha * 10))
    fig.savefig(f"{OUT}/sata_c1dpcadx_a{n}_dx{S_samp}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    print("output ->", os.path.abspath(OUT))
    for a in ALPHAS:
        for S in SPACINGS:
            panel(a, S); print(f"  sata_c1dpcadx_a{int(round(a*10))}_dx{S}.png")
    print("done.")
