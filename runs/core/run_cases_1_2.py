#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Regenerate the SATA multi-target CASE plots (Case 1 & Case 2), i.e. every figure
used in sata_cases_1_2.pdf.

Case 1 -- iso-range azimuth ramp with 5 targets.
Case 2 -- same ramp, but the non-reference targets sit exactly at the reference
          target's azimuth-ambiguity positions (+/-1620, +/-3240 samples).

PLACE THIS FILE IN runs/core/ (next to run_sata.py). It locates the sar_recon
package automatically (../../sar_reconstruction), so just run:
    python runs/core/run_cases_1_2.py
Output PNGs go to $PLOT_OUT (default: runs/core/plots/cases/).

Needs a LaTeX install (pdflatex) for the matplotlib pgf backend, matching the
reconstruction pipeline's typography. If you don't have LaTeX, set USE_LATEX=0.
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

OUT = os.environ.get("PLOT_OUT", os.path.join(_HERE, "plots", "cases"))
os.makedirs(OUT, exist_ok=True)

RDELAY = 0.0051115753
sysp = SystemParams(); V = sysp.vs; WL = sysp.wl
_base = Scene(rDelay=RDELAY, c0=sysp.c0, h0=0.0)
R0, H, Y0 = _base.r0, _base.H, _base.y0
XAMB = WL * R0 / (2 * V) * (2000.0 / 4)      # azimuth ambiguity position [m]
SAMP = V / 2000.0                            # focused-sample spacing [m]
RED, GREEN, BLUE, GRAY = "#d1372e", "#2ca02c", "#1f5fa0", "#555555"


# --------------------------------------------------------------------------- #
# scene builders / reconstruction
# --------------------------------------------------------------------------- #
def build(specs, Nrx=4, dxt=150.0):
    """specs = iterable of (dx_metres, dh_metres)."""
    extra = []
    for dxm, dh in specs:
        y_t = np.sqrt(R0 ** 2 - (H - dh) ** 2)
        extra.append((float(dxm), float(y_t - Y0), float(dh)))
    scene = Scene(rDelay=RDELAY, c0=sysp.c0, h0=0.0, extra_offsets=tuple(extra))
    array = ArrayGeometry.linear(Nrx, 100.0, float(dxt))
    prf, PRFop = prf_from_fixed(2000.0, Nrx)
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.0 * integration_time(sysp, scene))
    cfg = sar.ExperimentConfig(name="case", system=sysp, scene=scene, array=array,
                               prf=prf, PRF_op=PRFop, Na=Na, Na_ch=Nc, ta=ta,
                               plots_dir=None)
    return cfg, build_platform_tracks(cfg)


def recon(cfg, tr):
    """Return complex focused (vs single-point ref) reference/no-SATA/+SATA."""
    s = cfg.system; ptgs = cfg.scene.points[1:]
    sref1 = getRawData1D(cfg.scene.ptg[None, :], tr.ptx, tr.ptx, tr.vtx, tr.vtx,
                         cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx,
                         s.wl, cfg.prf)
    sig = getRawData1D(ptgs, tr.ptx, tr.ptx, tr.vtx, tr.vtx, cfg.ta, cfg.sq_tx,
                       cfg.sq_tx, cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)
    s_ch = np.zeros([cfg.Nrx, cfg.Na_ch], complex)
    for ii in range(cfg.Nrx):
        s_ch[ii] = getRawData1D(ptgs, tr.ptx, tr.prx[ii], tr.vtx, tr.vrx[ii], cfg.ta,
                                cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx,
                                s.wl, cfg.prf)[::cfg.Nrx]
    no = sar.reconstruct(cfg, tr, s_ch.copy())
    sa = sar.reconstruct(cfg, tr, sata_channels(cfg, tr, s_ch.copy()))
    Na = cfg.Na
    fc = lambda x: np.roll(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(sref1))), Na // 2)
    return sig, no, sa, fc(sig), fc(no), fc(sa)


C1_SPECS = tuple((dx, (dx + 400) * np.tan(np.deg2rad(20.0)))
                 for dx in (-400, -200, 0, 200, 400))   # ramp at alpha=20 deg
C2_GRID = np.array([-2, -1, 0, 1, 2]) * XAMB
C2_STEPS = np.array([0, 200, 400, 600, 800])
def c2_specs(alpha):
    return tuple((float(dx), float(st * np.tan(np.deg2rad(alpha))))
                 for dx, st in zip(C2_GRID, C2_STEPS))


# --------------------------------------------------------------------------- #
# geometry diagrams (pure matplotlib)
# --------------------------------------------------------------------------- #
def geom_case1():
    dx = np.array([-400, -200, 0, 200, 400.0]); h = (dx + 400) * np.tan(np.deg2rad(20.0))
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    aL.plot([-520, 520], [0, 0], color=GRAY, lw=1.0, ls=(0, (4, 3)))
    aL.text(520, 6, r"iso-range ground ($r_0$)", color=GRAY, ha="right", va="bottom", fontsize=8.5)
    aL.plot(dx, h, color=GRAY, lw=1.2, zorder=1)
    for x, hh in zip(dx, h):
        aL.plot([x, x], [0, hh], color=GRAY, lw=0.8, ls=":", zorder=1)
        aL.plot(x, hh, "o", color=BLUE, ms=10, zorder=3)
        aL.annotate(r"$\Delta h=%.0f$" % hh, (x, hh), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8, color=BLUE)
        aL.annotate(r"$%d$" % x, (x, 0), textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=8, color=GRAY)
    aL.set_xlim(-560, 560); aL.set_ylim(-40, h.max() * 1.35)
    aL.set_xlabel(r"azimuth position $dx$ [samples]"); aL.set_ylabel(r"height $\Delta h$ [m]")
    aL.set_title(r"(a) azimuth ramp ($\alpha=20^\circ$)"); aL.grid(True, alpha=0.25)
    Rx, Ry, rho = 0.0, 3.0, 3.4
    aR.plot([-0.2, 3.8], [0, 0], color=GRAY, lw=1.0, ls=(0, (4, 3)))
    aR.text(3.8, 0.07, "ground", color=GRAY, ha="right", va="bottom", fontsize=8.5)
    aR.plot([Rx - 0.5, Rx + 0.5], [Ry, Ry], color="k", lw=2.2, solid_capstyle="round")
    aR.plot(Rx, Ry, marker="s", color="k", ms=9)
    aR.annotate("radar (platform)", (Rx, Ry), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
    aR.plot([Rx, Rx], [Ry, 0], color=GRAY, lw=0.7, ls=":")
    th = np.linspace(np.deg2rad(28.1), np.deg2rad(56), 200)
    aR.plot(Rx + rho * np.sin(th), Ry - rho * np.cos(th), color=RED, lw=1.8)
    aR.annotate(r"iso-range circle ($r_0$)", (Rx + rho * np.sin(th[-1]), Ry - rho * np.cos(th[-1])),
                textcoords="offset points", xytext=(8, 2), ha="left", fontsize=9, color=RED)
    hts = np.array([0, 73, 146, 218, 291]) / 291.0 * 0.85
    tth = np.arccos(np.clip((Ry - hts) / rho, -1, 1))
    tx = Rx + rho * np.sin(tth); ty = Ry - rho * np.cos(tth)
    aR.plot(tx, ty, "o", color=BLUE, ms=9, zorder=4)
    aR.plot([Rx, tx[2]], [Ry, ty[2]], color=BLUE, lw=0.9, ls="--")
    aR.annotate(r"$r_0$", ((Rx + tx[2]) / 2, (Ry + ty[2]) / 2), textcoords="offset points",
                xytext=(-13, 2), ha="center", fontsize=10, color=BLUE)
    aR.text(0.97, 0.60, r"5 targets:" + "\n" + r"same $r_0$, diff.\ $\Delta h$",
            transform=aR.transAxes, ha="right", va="top", fontsize=8.5, color=BLUE)
    aR.text(0.03, 0.05, "schematic, not to scale", transform=aR.transAxes, fontsize=8, color=GRAY, style="italic")
    aR.set_xlim(-0.3, 4.0); aR.set_ylim(-0.35, 3.7)
    aR.set_xlabel(r"cross-track / ground range $\rightarrow$"); aR.set_ylabel(r"height $\rightarrow$")
    aR.set_title(r"(b) range cross-section: radar \& iso-range circle")
    aR.set_aspect("equal", adjustable="box"); aR.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(f"{OUT}/sata_case1_geom.png", dpi=200); plt.close(fig)


def geom_case2():
    pos = C2_GRID / SAMP  # samples
    h = C2_STEPS * 0.123 * np.tan(np.deg2rad(20.0)) * (800 / 800.0)  # display heights
    h = (np.array([0, 73, 146, 218, 291.0]))  # match alpha=20 heights for display
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.plot([-3600, 3600], [0, 0], color=GRAY, lw=1.0, ls=(0, (4, 3)))
    ax.text(3600, 8, r"iso-range ground ($r_0$)", color=GRAY, ha="right", va="bottom", fontsize=8.5)
    ax.plot(pos, h, color=GRAY, lw=1.2, zorder=1)
    labs = {int(pos[0]): r"$-2x_{\mathrm{amb}}$", int(pos[1]): r"$-x_{\mathrm{amb}}$",
            0: r"$dx=0$", int(pos[3]): r"$+x_{\mathrm{amb}}$", int(pos[4]): r"$+2x_{\mathrm{amb}}$"}
    for i, (x, hh) in enumerate(zip(pos, h)):
        ax.plot([x, x], [0, hh], color=GRAY, lw=0.8, ls=":", zorder=1)
        if i == 2:
            ax.plot(x, hh, "*", color=BLUE, ms=20, zorder=4)
            ax.annotate(r"\textbf{reference}", (x, hh), textcoords="offset points",
                        xytext=(0, 14), ha="center", fontsize=9.5, color=BLUE)
        else:
            ax.plot(x, hh, "o", color=GREEN, ms=11, zorder=4)
        ax.annotate(r"$\Delta h=%.0f$" % hh, (x, hh), textcoords="offset points",
                    xytext=(0, -14 if i == 2 else 12), ha="center",
                    va="top" if i == 2 else "bottom", fontsize=8,
                    color=(BLUE if i == 2 else GREEN))
        ax.annotate(labs[int(round(x))] + r" (%d)" % int(round(x)), (x, 0),
                    textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8, color=GRAY)
    for s in (-2, -1, 1, 2):
        ax.annotate("", xy=(s * XAMB / SAMP, 4), xytext=(0, 4),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.1, ls="--"))
    ax.text(0, h.max() * 0.62, r"non-reference targets sit at the reference's"
            + "\n" + r"ambiguity positions $\pm x_{\mathrm{amb}},\pm2x_{\mathrm{amb}}$",
            color=RED, ha="center", fontsize=8.7)
    ax.set_xlim(-3700, 3700); ax.set_ylim(-40, h.max() * 1.45)
    ax.set_xlabel(r"azimuth position $dx$ [samples]"); ax.set_ylabel(r"height $\Delta h$ [m]")
    ax.set_title(r"Case 2 --- ramp, with non-reference targets at the ambiguity positions ($\alpha=20^\circ$)")
    ax.grid(True, alpha=0.25); fig.tight_layout()
    fig.savefig(f"{OUT}/sata_case2_geom.png", dpi=200); plt.close(fig)


# --------------------------------------------------------------------------- #
# Case 1 focused image + Case 2 four-panels
# --------------------------------------------------------------------------- #
def case1_focused():
    cfg, tr = build(C1_SPECS); Na = cfg.Na
    sig, no, sa, Fr, Fn, Fs = recon(cfg, tr); pmax = np.abs(Fr).max()
    db = lambda z: 20 * np.log10(np.abs(z) / pmax + 1e-12)
    n = Na; w = slice(n // 2 - 300, n // 2 + 300); x = np.arange(600) - 300
    fig, ax = plt.subplots(figsize=(8.7, 4.3))
    ax.plot(x, db(Fr)[w], "k", lw=1.3, label="reference")
    ax.plot(x, db(Fn)[w], "C3", lw=1.0, label="reconstruction (no SATA)")
    ax.plot(x, db(Fs)[w], "C0--", lw=1.3, label="reconstruction + SATA")
    ax.set_title(r"Case 1 focused image (five targets, focus vs.\ one point)")
    ax.set_xlabel("azimuth sample (around scene centre)"); ax.set_ylabel(r"amplitude [dB]")
    ax.set_ylim(-40, 3); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/sata_azimuth_topo.png", dpi=150); plt.close(fig)


def case2_four_panel(mode):
    cfg, tr = build(c2_specs(20.0)); Na, prf, abw, ta = cfg.Na, cfg.prf, cfg.abw, cfg.ta
    sig, no, sa, Fr, Fn, Fs = recon(cfg, tr); pmax = np.abs(Fr).max()
    ndb = lambda z: 20 * np.log10(np.abs(z) / pmax + 1e-12)
    fa = np.roll((np.arange(Na) / Na - 0.5) * prf, Na // 2); band = np.abs(fa) <= 0.5 * abw
    def dph(x):
        d = np.angle(np.fft.fft(x) * np.conj(np.fft.fft(sig)), deg=True); d[~band] = np.nan; return d
    show_no = mode in ("nosata", "both"); show_sa = mode in ("sata", "both")
    ttl = {"nosata": "no SATA", "sata": "+SATA", "both": "no-SATA vs.\\ +SATA"}[mode]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(r"Case 2 (ramp, non-ref.\ targets at ambiguities) --- %s, $N_{rx}=4$, $\alpha=20^\circ$" % ttl)
    ax[0, 0].plot(ta, np.abs(sig), "k", lw=1.0, label="reference (5 targets)")
    if show_no: ax[0, 0].plot(ta, np.abs(no), "C3", lw=0.8, label="no-SATA")
    if show_sa: ax[0, 0].plot(ta, np.abs(sa), "C0", lw=0.8, ls="--", label="+SATA")
    ax[0, 0].set_xlabel("Time [s]"); ax[0, 0].set_ylabel("Amplitude"); ax[0, 0].grid(alpha=0.3); ax[0, 0].legend(fontsize="small")
    ax[0, 1].plot(ta, ndb(Fr), "k", lw=0.8, label="reference")
    if show_no: ax[0, 1].plot(ta, ndb(Fn), "C3", lw=0.8, label="no-SATA")
    if show_sa: ax[0, 1].plot(ta, ndb(Fs), "C0", lw=0.8, ls="--", label="+SATA")
    ax[0, 1].set_xlabel("Time [s]"); ax[0, 1].set_ylabel("[dB]"); ax[0, 1].set_ylim([-60, 3])
    ax[0, 1].grid(alpha=0.3); ax[0, 1].legend(fontsize="small"); ax[0, 1].set_title("focused image (focus vs.\\ 1 point)", fontsize=9)
    Nz = int(16 * prf / abw); zpf = 64; taz = (np.arange(2 * Nz * zpf) - Nz * zpf) / prf / zpf * 1e3
    zpk = np.abs(zoom1Dpeak(Fr, Nz, zpf)).max(); zdb = lambda F: 20 * np.log10(np.abs(zoom1Dpeak(F, Nz, zpf)) / zpk + 1e-12)
    ax[1, 0].plot(taz, zdb(Fr), "k", lw=1.2, label="reference")
    if show_no: ax[1, 0].plot(taz, zdb(Fn), "C3", lw=0.9, label="no-SATA")
    if show_sa: ax[1, 0].plot(taz, zdb(Fs), "C0", lw=1.1, ls="--", label="+SATA")
    ax[1, 0].set_xlim(-12, 12); ax[1, 0].set_ylim([-45, 2]); ax[1, 0].set_xlabel("Time [ms]"); ax[1, 0].set_ylabel("[dB]")
    ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize="small"); ax[1, 0].set_title("zoomed IRF (reference target)", fontsize=9)
    if show_no: ax[1, 1].plot(fa, dph(no), "C3", lw=0.7, label="no-SATA")
    if show_sa: ax[1, 1].plot(fa, dph(sa), "C0", lw=0.9, label="+SATA")
    ax[1, 1].axvline(abw / 2, color="r", ls="-."); ax[1, 1].axvline(-abw / 2, color="r", ls="-.")
    ax[1, 1].set_xlabel("Doppler freq [Hz]"); ax[1, 1].set_ylabel("[deg]"); ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend(fontsize="small"); ax[1, 1].set_title("spectral phase error", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/sata_case2_4panel_{mode}.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# per-target bars (Case 1 & Case 2) + Case 2 ambiguity suppression
# --------------------------------------------------------------------------- #
def _per_target(Fr, Fn, Fs, Na, dxs, w=25):
    c = Na // 2; an = []; as_ = []; pn = []; ps = []
    for dx in dxs:
        i0 = c + int(round(dx / SAMP)); seg = slice(i0 - w, i0 + w + 1)
        j = i0 - w + np.argmax(np.abs(Fr[seg]))
        an.append(100 * np.abs(Fn[j]) / np.abs(Fr[j])); as_.append(100 * np.abs(Fs[j]) / np.abs(Fr[j]))
        pn.append(np.angle(Fn[j] * np.conj(Fr[j]), deg=True)); ps.append(np.angle(Fs[j] * np.conj(Fr[j]), deg=True))
    return map(np.array, (an, as_, pn, ps))


def bars(specs, dxs, labels, title, fname):
    cfg, tr = build(specs); Na = cfg.Na
    _, _, _, Fr, Fn, Fs = recon(cfg, tr)
    an, as_, pn, ps = _per_target(Fr, Fn, Fs, Na, dxs)
    x = np.arange(len(dxs)); wdt = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2)); fig.suptitle(title)
    a1.bar(x - wdt / 2, an, wdt, color=RED, label="no-SATA"); a1.bar(x + wdt / 2, as_, wdt, color=GREEN, label="+SATA")
    a1.axhline(100, color="k", ls=":", lw=0.8); a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylabel(r"peak recovery [\% of ideal]"); a1.set_title("amplitude recovery per target")
    a1.grid(alpha=0.3, axis="y"); a1.legend()
    a2.bar(x - wdt / 2, np.abs(pn), wdt, color=RED, label="no-SATA"); a2.bar(x + wdt / 2, np.abs(ps), wdt, color=GREEN, label="+SATA")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8); a2.set_ylabel(r"$|$peak phase error$|$ [deg]")
    a2.set_title("phase error at each target peak"); a2.grid(alpha=0.3, axis="y"); a2.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=150); plt.close(fig)


def case2_ambiguity():
    cfg, tr = build(c2_specs(20.0)); Na = cfg.Na
    _, _, _, Fr, Fn, Fs = recon(cfg, tr); c = Na // 2; pmax = np.abs(Fr).max()
    db = lambda z: 20 * np.log10(np.abs(z) / pmax + 1e-12); ax0 = np.arange(Na) - c
    W = (ax0 >= -5600) & (ax0 <= 5600)
    fig, ax = plt.subplots(figsize=(11.4, 4.5))
    ax.plot(ax0[W], db(Fn)[W], color=RED, lw=0.7, label="no-SATA")
    ax.plot(ax0[W], db(Fs)[W], color=GREEN, lw=0.9, ls="--", label="+SATA")
    ax.plot(ax0[W], db(Fr)[W], "k", lw=0.7, label="reference", alpha=0.7)
    for g in [-2, -1, 0, 1, 2]: ax.axvline(g * XAMB / SAMP, color=BLUE, ls=":", lw=0.7)
    for a in [-3, 3]: ax.axvline(a * XAMB / SAMP, color=RED, ls="-.", lw=0.8)
    ax.annotate("real targets\n(on 1620 grid)", (0, 2.3), fontsize=8, color=BLUE, ha="center")
    ax.set_ylim(-45, 6); ax.set_xlabel("azimuth sample (around scene centre)"); ax.set_ylabel("amplitude [dB]")
    ax.set_title(r"Case 2 --- focused image (dB): ambiguity replicas suppressed by SATA")
    ax.grid(alpha=0.3); ax.legend(loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/sata_case2_ambiguity.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# alpha studies (Case 1 & Case 2)
# --------------------------------------------------------------------------- #
def _metrics(specs, dxs):
    cfg, tr = build(specs); Na = cfg.Na
    _, _, _, Fr, Fn, Fs = recon(cfg, tr)
    Fr, Fn, Fs = np.abs(Fr), np.abs(Fn), np.abs(Fs); pmax = Fr.max(); c = Na // 2
    ts = [c + int(round(dx / SAMP)) for dx in dxs]
    pk = lambda F, i, w=20: F[i - w:i + w + 1].max()
    rec_no = np.mean([100 * pk(Fn, i) / pk(Fr, i) for i in ts])
    rec_sa = np.mean([100 * pk(Fs, i) / pk(Fr, i) for i in ts])
    mask = np.ones(Na, bool)
    for i in ts: mask[max(0, i - 160):i + 160] = False
    amb_no = 20 * np.log10(Fn[mask].max() / pmax + 1e-12)
    amb_sa = 20 * np.log10(Fs[mask].max() / pmax + 1e-12)
    return rec_no, rec_sa, amb_no, amb_sa


def alpha_study(kind):
    alphas = [5, 10, 15, 20, 25, 30]; R = []
    for a in alphas:
        if kind == "case1":
            specs = tuple((dx, (dx + 400) * np.tan(np.deg2rad(a))) for dx in (-400, -200, 0, 200, 400))
            dxs = [-400, -200, 0, 200, 400]
            title = r"Case 1 (same range, azimuth ramp) --- study vs.\ ramp inclination $\alpha$"
            fn = "sata_case1_alpha.png"
        else:
            specs = c2_specs(a); dxs = list(C2_GRID)
            title = r"Case 2 (targets at ambiguity positions) --- study vs.\ ramp inclination $\alpha$"
            fn = "sata_case2_alpha.png"
        R.append(_metrics(specs, dxs))
    R = np.array(R)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3)); fig.suptitle(title)
    a1.plot(alphas, R[:, 0], "o--", color=RED, label="no-SATA"); a1.plot(alphas, R[:, 1], "s-", color=GREEN, label="+SATA")
    a1.axhline(100, color="k", ls=":", lw=0.8); a1.set_xlabel(r"ramp inclination $\alpha$ [deg]")
    a1.set_ylabel(r"mean peak recovery [\% of ideal]"); a1.set_title("peak recovery vs.\\ ramp angle"); a1.grid(alpha=0.3); a1.legend()
    a2.plot(alphas, R[:, 2], "o--", color=RED, label="no-SATA"); a2.plot(alphas, R[:, 3], "s-", color=GREEN, label="+SATA")
    a2.set_xlabel(r"ramp inclination $\alpha$ [deg]"); a2.set_ylabel(r"worst ambiguity [dB]")
    a2.set_title("ambiguity vs.\\ ramp angle"); a2.grid(alpha=0.3); a2.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/{fn}", dpi=150); plt.close(fig)


if __name__ == "__main__":
    print("output ->", os.path.abspath(OUT))
    geom_case1(); print("  sata_case1_geom.png")
    geom_case2(); print("  sata_case2_geom.png")
    case1_focused(); print("  sata_azimuth_topo.png")
    for m in ("nosata", "sata", "both"): case2_four_panel(m); print(f"  sata_case2_4panel_{m}.png")
    bars(C1_SPECS, [-400, -200, 0, 200, 400],
         [r"$-400$", r"$-200$", r"$0$", r"$200$", r"$400$"],
         r"Case 1 --- per-target analysis (azimuth ramp)", "sata_case1_bars.png"); print("  sata_case1_bars.png")
    bars(c2_specs(20.0), list(C2_GRID),
         [r"$-2x_{\mathrm{amb}}$", r"$-x_{\mathrm{amb}}$", r"$0$ (ref)", r"$+x_{\mathrm{amb}}$", r"$+2x_{\mathrm{amb}}$"],
         r"Case 2 --- per-target analysis (ramp, $\alpha=20^\circ$)", "sata_case2_bars.png"); print("  sata_case2_bars.png")
    case2_ambiguity(); print("  sata_case2_ambiguity.png")
    alpha_study("case1"); print("  sata_case1_alpha.png")
    alpha_study("case2"); print("  sata_case2_alpha.png")
    print("done.")
