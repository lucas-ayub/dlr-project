# -*- coding: utf-8 -*-
"""
run_model_validation_report.py

Publication-quality figure for the LaTeX report: validates this project's
numerical Delta_r (GetCoeffNu, C0 coefficient) against Natalia's analytic
Eq. (8) elevation term, across several baselines, on the iso-range surface
(the geometry Eq. 8 assumes: same slant range r0 as the reference, only
height differs).

Sign convention note (documented, applied explicitly, not left implicit):
    This project's phase convention is  phi = +2*pi*C0/wl.
    Eq. (9) of Natalia's paper is       Delta_Phi = -2*pi/wl * Delta_r.
    These are the SAME physical quantity up to an overall sign, so we plot
        Delta_r_paper_convention := -1 * (numerical C0 residual)
    to match Eq. (8)/(9) directly, and note this explicitly in the caption.

Baseline convention: Eq. (8)'s Bn is the PERPENDICULAR baseline,
    B_perp = bxt * cos(theta0),
not the raw horizontal cross-track offset bxt.

Outputs:
    plots/report/model_validation_natalia.pdf  (and .png for quick preview)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import sar_recon as sar
from sar_recon.config import (SystemParams, Scene, ArrayGeometry,
                              prf_from_fixed, integration_time, build_time_axis)
from sar_recon.geometry import build_platform_tracks
from sar_recon.reconstruction import GetCoeffNu

# =============================================================================
# Report-quality plotting style
# =============================================================================
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "figure.dpi": 150,
})
# If a LaTeX installation is available and you want exact font matching with
# the report, uncomment the two lines below. Left off by default since it
# requires a working LaTeX toolchain and slows down rendering considerably.
# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.serif"] = ["Computer Modern Roman"]

# =============================================================================
# Parameters
# =============================================================================
WL         = 0.240                     # L-band
BXT_VALUES = [20.0, 50.0, 100.0]       # cross-track baselines to validate against
CHANNEL    = 1
DX         = 100.0                     # along-track receiver spacing [m]
NRX        = 2

DH_MAX = 200.0
N_DH   = 21
DH_VALUES = np.linspace(0.0, DH_MAX, N_DH)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "plots", "report")
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================
def _build_cfg(bxt, extra_offsets):
    system = SystemParams(wl=WL)
    scene = Scene(rDelay=0.0051115753, c0=system.c0, h0=0.0,
                  extra_offsets=extra_offsets)

    bat = DX * np.arange(NRX)
    bxt_arr = np.zeros(NRX)
    bxt_arr[CHANNEL] = bxt
    array = ArrayGeometry(bat=bat, bxt=bxt_arr)

    prf, PRF_op = prf_from_fixed(2000.0, NRX)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, NRX, acq_time)

    cfg = sar.ExperimentConfig(
        name="model_validation", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta, plots_dir=SAVE_DIR,
    )
    return cfg, build_platform_tracks(cfg), scene


def _C0(cfg, tracks, ptg):
    kk = CHANNEL
    C0, C1, C2, Dt = GetCoeffNu(
        ptg, tracks.ptx, tracks.prx[kk], tracks.vtx, tracks.vrx[kk],
        tracks.ptx, tracks.vtx,
        cfg.prf, cfg.system.wl, cfg.ta,
        cfg.sq_tx, cfg.sq_rx[kk], cfg.theta_tx, cfg.theta_rx[kk],
    )
    return C0


# Reference geometry (fixed, independent of bxt and dh)
cfg0, tracks0, scene0 = _build_cfg(BXT_VALUES[0], ())
r0 = scene0.r0
y0 = scene0.y0
H  = scene0.H
sin_t0 = y0 / r0
cos_t0 = H / r0
theta0_deg = np.degrees(np.arcsin(sin_t0))

print(f"Reference geometry: r0={r0:.2f} m | theta0={theta0_deg:.3f} deg | "
      f"sin={sin_t0:.5f} | cos={cos_t0:.5f}")

# =============================================================================
# Sweep: iso-range target, per baseline
# =============================================================================
results = {}   # bxt -> dict(dr_num, dr_an, rel_err_pct)

for bxt in BXT_VALUES:
    dr_num = np.zeros_like(DH_VALUES)
    dr_an  = np.zeros_like(DH_VALUES)

    for i, dh in enumerate(DH_VALUES):
        if dh == 0.0:
            extra = ()
        else:
            y_target = np.sqrt(r0**2 - (H - dh)**2)
            dy = y_target - y0
            extra = ((0.0, float(dy), float(dh)),)

        cfg, tracks, scene = _build_cfg(bxt, extra)
        ptg_real = np.asarray(scene.points[-1], dtype=float)
        ptg_ref  = np.asarray(scene.ptg,        dtype=float)

        # Apply the sign convention flip here, explicitly, so the numerical
        # curve is directly comparable to Eq. (8)/(9) as published.
        dr_num[i] = -1.0 * (_C0(cfg, tracks, ptg_real) - _C0(cfg, tracks, ptg_ref))

        b_perp = bxt * cos_t0
        dr_an[i] = (b_perp * dh) / (r0 * sin_t0)

    rel_err_pct = np.zeros_like(DH_VALUES)
    nonzero = dr_an != 0.0
    rel_err_pct[nonzero] = 100.0 * (dr_num[nonzero] - dr_an[nonzero]) / dr_an[nonzero]

    results[bxt] = dict(dr_num=dr_num, dr_an=dr_an, rel_err_pct=rel_err_pct)

    max_err = np.max(np.abs(rel_err_pct[nonzero])) if np.any(nonzero) else 0.0
    print(f"bxt={bxt:5.0f} m | max relative error = {max_err:.4f} %")

# =============================================================================
# Figure: two panels -- (top) overlay, (bottom) relative error
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 6.0), sharex=True,
                                gridspec_kw=dict(height_ratios=[2.2, 1.0]))

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(BXT_VALUES)))

for bxt, color in zip(BXT_VALUES, colors):
    r = results[bxt]
    ax1.plot(DH_VALUES, r["dr_an"], color=color, linestyle="-",
             label=f"Eq. (8), $b_{{xt}}$={bxt:.0f} m")
    ax1.plot(DH_VALUES, r["dr_num"], color=color, linestyle="none",
             marker="o", markersize=4, markerfacecolor="none",
             label=f"Numerical, $b_{{xt}}$={bxt:.0f} m")

ax1.set_ylabel(r"$\Delta r$ [m]")
ax1.set_title(r"Validation against Eq.~(8): $B_\perp = b_{xt}\cos\theta_0$, "
             r"iso-range target placement")
ax1.grid(alpha=0.4)
ax1.legend(fontsize=8, ncol=2, loc="upper left")

for bxt, color in zip(BXT_VALUES, colors):
    r = results[bxt]
    ax2.plot(DH_VALUES, r["rel_err_pct"], color=color, marker="o", markersize=3,
             label=f"$b_{{xt}}$={bxt:.0f} m")

ax2.axhline(0.0, color="gray", linewidth=0.8)
ax2.set_xlabel(r"Height offset $\Delta h$ [m] (target kept at slant range $r_0$)")
ax2.set_ylabel("Relative error [\\%]" if plt.rcParams.get("text.usetex") else "Relative error [%]")
ax2.grid(alpha=0.4)
ax2.legend(fontsize=8, ncol=3, loc="best")

fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "model_validation_natalia.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(SAVE_DIR, "model_validation_natalia.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {os.path.join(SAVE_DIR, 'model_validation_natalia.pdf')}")