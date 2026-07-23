#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Case 1 (5 iso-range ramp targets) -- sweep of cross-track baseline b_xt and ramp
inclination alpha. Produces the 16 four-panels used in
sata_case1_bxt_alpha_sweep.pdf (b_xt in {5,10,15,20} m, alpha in {5,10,15,20} deg).

Each 4-panel: azimuth signal (time) | focused image (dB) with worst ambiguity |
zoomed IRF | spectral phase error with in-band std. Header carries the parameters.

PLACE THIS FILE IN runs/core/ (next to run_sata.py). It locates the sar_recon
package automatically (../../sar_reconstruction), so just run:
    python runs/core/run_case1_bxt_alpha.py
Output PNGs (sata_c1_a{alpha}_b{bxt}.png) go to $PLOT_OUT
(default: runs/core/plots/case1_sweep/).
Needs pdflatex (matplotlib pgf); set USE_LATEX=0 to fall back to mathtext.
"""
import os
import sys
import numpy as np
import matplotlib

# --- locate the sar_recon package (repo layout: runs/core/ and sar_reconstruction/ are siblings) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, "..", "..", "sar_reconstruction"))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

USE_LATEX = os.environ.get("USE_LATEX", "1") == "1"
if USE_LATEX:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex", "text.usetex": True, "font.family": "serif",
        "pgf.rcfonts": False, "pgf.preamble": r"\usepackage{amsmath,amssymb}"})
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sar_recon as sar
from sar_recon.config import (SystemParams, Scene, ArrayGeometry, prf_from_fixed,
                              integration_time, build_time_axis)
from sar_recon.geometry import build_platform_tracks
from sar_recon.signal_model import getRawData1D
from sar_recon.analysis import zoom1Dpeak
from sar_recon.sata import sata_channels

OUT = os.environ.get("PLOT_OUT", os.path.join(_HERE, "plots", "case1_sweep")); os.makedirs(OUT, exist_ok=True)
RDELAY = 0.0051115753
sysp = SystemParams()
_base = Scene(rDelay=RDELAY, c0=sysp.c0, h0=0.0)
R0, H, Y0 = _base.r0, _base.H, _base.y0
DXS = np.array([-400, -200, 0, 200, 400.0])   # azimuth positions [m]

BXT = [5, 10, 15, 20]        # cross-track baseline [m]
ALPHA = [5, 10, 15, 20]      # ramp inclination [deg]


def build(alpha, bxt, Nrx=4):
    hts = (DXS + 400) * np.tan(np.deg2rad(alpha))
    extra = []
    for dxm, dh in zip(DXS, hts):
        y_t = np.sqrt(R0 ** 2 - (H - dh) ** 2)
        extra.append((float(dxm), float(y_t - Y0), float(dh)))
    scene = Scene(rDelay=RDELAY, c0=sysp.c0, h0=0.0, extra_offsets=tuple(extra))
    array = ArrayGeometry.linear(Nrx, 100.0, float(bxt))
    prf, PRFop = prf_from_fixed(2000.0, Nrx)
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.0 * integration_time(sysp, scene))
    cfg = sar.ExperimentConfig(name="c1", system=sysp, scene=scene, array=array,
                               prf=prf, PRF_op=PRFop, Na=Na, Na_ch=Nc, ta=ta, plots_dir=None)
    return cfg, build_platform_tracks(cfg)


def panel(alpha, bxt):
    cfg, tr = build(alpha, bxt); s = cfg.system; Na, prf, abw, ta = cfg.Na, cfg.prf, cfg.abw, cfg.ta
    ptgs = cfg.scene.points[1:]
    sref1 = getRawData1D(cfg.scene.ptg[None, :], tr.ptx, tr.ptx, tr.vtx, tr.vtx, cfg.ta,
                         cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)
    sig = getRawData1D(ptgs, tr.ptx, tr.ptx, tr.vtx, tr.vtx, cfg.ta, cfg.sq_tx, cfg.sq_tx,
                       cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)
    s_ch = np.zeros([cfg.Nrx, cfg.Na_ch], complex)
    for ii in range(cfg.Nrx):
        s_ch[ii] = getRawData1D(ptgs, tr.ptx, tr.prx[ii], tr.vtx, tr.vrx[ii], cfg.ta, cfg.sq_tx,
                                cfg.sq_tx, cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)[::cfg.Nrx]
    no = sar.reconstruct(cfg, tr, s_ch.copy())
    sa = sar.reconstruct(cfg, tr, sata_channels(cfg, tr, s_ch.copy()))
    fc = lambda x: np.roll(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(sref1))), Na // 2)
    Fr, Fn, Fs = fc(sig), fc(no), fc(sa); pmax = np.abs(Fr).max()
    ndb = lambda z: 20 * np.log10(np.abs(z) / pmax + 1e-12)
    fa = np.roll((np.arange(Na) / Na - 0.5) * prf, Na // 2); band = np.abs(fa) <= 0.5 * abw
    def dph(x):
        d = np.angle(np.fft.fft(x) * np.conj(np.fft.fft(sig)), deg=True); d[~band] = np.nan; return d
    c = Na // 2; mask = np.ones(Na, bool); mask[c - 300:c + 300] = False
    amb_no = ndb(Fn)[mask].max(); amb_sa = ndb(Fs)[mask].max()
    sn, ss = np.nanstd(dph(no)[band]), np.nanstd(dph(sa)[band])
    dbat = cfg.array.bat[1] - cfg.array.bat[0]; dbxt = cfg.array.bxt[1] - cfg.array.bxt[0]
    dhmax = 800 * np.tan(np.deg2rad(alpha))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(r"SATA reconstruction, $N_{rx}=%d$, PRF=%.0f Hz, $B_a$=%.0f Hz, "
                 r"$\Delta b_{at}$=%.0f m, $\Delta b_{xt}$=%.0f m, Case 1 azimuth ramp: "
                 r"5 targets, $\alpha=%d^\circ$ ($\Delta h_{\max}$=%.0f m)"
                 % (cfg.Nrx, prf, abw, dbat, dbxt, alpha, dhmax), fontsize=11)
    ax[0, 0].plot(ta, np.abs(sig), "k", lw=1.0, label="reference (5 targets)")
    ax[0, 0].plot(ta, np.abs(no), "C3", lw=0.8, label="no-SATA")
    ax[0, 0].plot(ta, np.abs(sa), "C0", lw=0.8, ls="--", label="+SATA")
    ax[0, 0].set_xlabel("Time [s]"); ax[0, 0].set_ylabel("Amplitude"); ax[0, 0].grid(alpha=0.3); ax[0, 0].legend(fontsize="small")
    ax[0, 1].plot(ta, ndb(Fr), "k", lw=0.8, label="reference"); ax[0, 1].plot(ta, ndb(Fn), "C3", lw=0.8, label="no-SATA")
    ax[0, 1].plot(ta, ndb(Fs), "C0", lw=0.8, ls="--", label="+SATA")
    ax[0, 1].set_xlabel("Time [s]"); ax[0, 1].set_ylabel("[dB]"); ax[0, 1].set_ylim([-60, 3]); ax[0, 1].grid(alpha=0.3); ax[0, 1].legend(fontsize="small")
    ax[0, 1].set_title(r"focused image (worst ambiguity: no-SATA %.1f dB, +SATA %.1f dB)" % (amb_no, amb_sa), fontsize=8)
    Nz = int(16 * prf / abw); zpf = 64; taz = (np.arange(2 * Nz * zpf) - Nz * zpf) / prf / zpf * 1e3
    zpk = np.abs(zoom1Dpeak(Fr, Nz, zpf)).max(); zdb = lambda F: 20 * np.log10(np.abs(zoom1Dpeak(F, Nz, zpf)) / zpk + 1e-12)
    ax[1, 0].plot(taz, zdb(Fr), "k", lw=1.2, label="reference"); ax[1, 0].plot(taz, zdb(Fn), "C3", lw=0.9, label="no-SATA")
    ax[1, 0].plot(taz, zdb(Fs), "C0", lw=1.1, ls="--", label="+SATA")
    ax[1, 0].set_xlim(-12, 12); ax[1, 0].set_ylim([-45, 2]); ax[1, 0].set_xlabel("Time [ms]"); ax[1, 0].set_ylabel("[dB]")
    ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize="small"); ax[1, 0].set_title("zoomed IRF", fontsize=9)
    ax[1, 1].plot(fa, dph(no), "C3", lw=0.7, label="no-SATA"); ax[1, 1].plot(fa, dph(sa), "C0", lw=0.9, label="+SATA")
    ax[1, 1].axvline(abw / 2, color="r", ls="-."); ax[1, 1].axvline(-abw / 2, color="r", ls="-.")
    ax[1, 1].set_xlabel("Doppler freq [Hz]"); ax[1, 1].set_ylabel("[deg]"); ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize="small")
    ax[1, 1].set_title(r"phase error (in-band std: no-SATA $%.0f^\circ$, +SATA $%.0f^\circ$)" % (sn, ss), fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/sata_c1_a{alpha}_b{bxt}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    print("output ->", os.path.abspath(OUT))
    for a in ALPHA:
        for b in BXT:
            panel(a, b); print(f"  sata_c1_a{a}_b{b}.png")
    print("done.")
