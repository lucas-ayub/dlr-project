# -*- coding: utf-8 -*-
"""
runs/validation/run_model_validation_report_combined.py

Two publication-quality validation figures, completing the model-validation
family started in run_model_validation_report.py (elevation term only):

  Figure 1 (model_validation_flatearth.pdf):
    Validates the FIRST (flat-earth) term of Eq. (8) in isolation, across
    several baselines. Mirrors run_model_validation_report.py's style and
    methodology, but for the flat-earth term instead of elevation.

  Figure 2 (model_validation_full_effect.pdf):
    Validates the SUM of BOTH terms of Eq. (8) against the actual,
    single-point numerical residual res(C0) on the REAL topographic ramp
    scene (target displaced in both ground position AND height at once --
    no iso-range placement, no three-point decomposition). This tests not
    just each term individually but that they linearly superpose to
    reproduce the true combined numerical effect, which is the physically
    complete quantity relevant to the reconstruction pipeline.

Corrections applied (see run_height_sweep_isorange.py /
run_range_sweep_flatearth.py for the individual derivations):
  (1) Sign convention: phi=+2*pi*C0/wl vs. Eq.(9)'s -2*pi/wl*Delta_r.
  (2) Baseline projection: B_perp = bxt*cos(theta0), not raw bxt.
  (3) For the flat-earth term, "r" is the EXACT slant-range delta induced
      by the displacement, sqrt((y0+dy)^2+(H-dh)^2) - r0, not the raw
      ground-range offset dy.

Outputs (in plots/report/):
    model_validation_flatearth.pdf/.png
    model_validation_full_effect.pdf/.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import sar_recon as sar
from sar_recon.config import (SystemParams, Scene, ArrayGeometry,
                              _make_topo_ramp, prf_from_fixed,
                              integration_time, build_time_axis)
from sar_recon.geometry import build_platform_tracks
from sar_recon.reconstruction import GetCoeffNu

# =============================================================================
# Report-quality plotting style (matches run_model_validation_report.py)
# =============================================================================
plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 9.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "lines.linewidth": 1.6, "lines.markersize": 5, "figure.dpi": 150,
})

# =============================================================================
# Parameters
# =============================================================================
WL = 0.240                       # L-band
BXT_VALUES = [20.0, 50.0, 100.0]
CHANNEL = 1
DX = 100.0
NRX = 2

DY_MAX, N_DY = 2000.0, 21
DY_VALUES = np.linspace(0.0, DY_MAX, N_DY)

ALPHA_VALUES = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
DY_RAMP = 2000.0   # fixed ground-range offset of the ramp's top point

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
        name="model_validation_combined", system=system, scene=scene, array=array,
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


# Reference geometry (fixed, independent of bxt/dy/dh)
cfg0, tracks0, scene0 = _build_cfg(BXT_VALUES[0], ())
r0, y0, H = scene0.r0, scene0.y0, scene0.H
sin_t0 = y0 / r0
cos_t0 = H / r0
tan_t0 = sin_t0 / cos_t0
theta0_deg = np.degrees(np.arcsin(sin_t0))
print(f"Reference geometry: r0={r0:.2f} m | theta0={theta0_deg:.3f} deg")


def _two_panel_figure(x_vals, x_label, results, title, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 6.0), sharex=True,
                                    gridspec_kw=dict(height_ratios=[2.2, 1.0]))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(BXT_VALUES)))

    for bxt, color in zip(BXT_VALUES, colors):
        r = results[bxt]
        ax1.plot(x_vals, r["an"], color=color, linestyle="-",
                 label=f"Eq. (8), $b_{{xt}}$={bxt:.0f} m")
        ax1.plot(x_vals, r["num"], color=color, linestyle="none",
                 marker="o", markersize=4, markerfacecolor="none",
                 label=f"Numerical, $b_{{xt}}$={bxt:.0f} m")
    ax1.set_ylabel(r"$\Delta r$ [m]")
    ax1.set_title(title)
    ax1.grid(alpha=0.4)
    ax1.legend(fontsize=8, ncol=2, loc="upper left")

    for bxt, color in zip(BXT_VALUES, colors):
        r = results[bxt]
        ax2.plot(x_vals, r["rel_err_pct"], color=color, marker="o", markersize=3,
                 label=f"$b_{{xt}}$={bxt:.0f} m")
    ax2.axhline(0.0, color="gray", linewidth=0.8)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Relative error [%]")
    ax2.grid(alpha=0.4)
    ax2.legend(fontsize=8, ncol=3, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, fname + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(SAVE_DIR, fname + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Figure 1 -- flat-earth term, isolated (h=0 fixed, dy swept)
# =============================================================================
results_flat = {}
for bxt in BXT_VALUES:
    dr_num = np.zeros_like(DY_VALUES)
    dr_an = np.zeros_like(DY_VALUES)

    for i, dy in enumerate(DY_VALUES):
        extra = ((0.0, float(dy), 0.0),) if dy != 0.0 else ()
        cfg, tracks, scene = _build_cfg(bxt, extra)
        ptg_real = np.asarray(scene.points[-1], dtype=float)
        ptg_ref = np.asarray(scene.ptg, dtype=float)

        dr_num[i] = -1.0 * (_C0(cfg, tracks, ptg_real) - _C0(cfg, tracks, ptg_ref))

        delta_r_exact = np.sqrt((y0 + dy) ** 2 + H ** 2) - r0
        b_perp = bxt * cos_t0
        dr_an[i] = (b_perp * delta_r_exact) / (r0 * tan_t0)

    rel_err_pct = np.zeros_like(DY_VALUES)
    nz = dr_an != 0.0
    rel_err_pct[nz] = 100.0 * (dr_num[nz] - dr_an[nz]) / dr_an[nz]
    results_flat[bxt] = dict(num=dr_num, an=dr_an, rel_err_pct=rel_err_pct)

    max_err = np.max(np.abs(rel_err_pct[nz])) if np.any(nz) else 0.0
    print(f"[flat-earth] bxt={bxt:5.0f} m | max relative error = {max_err:.4f} %")

_two_panel_figure(
    DY_VALUES, r"Ground-range offset $dy$ [m] (height fixed at $h=0$)",
    results_flat,
    r"Validation against Eq.~(8), flat-earth term: $B_\perp$, exact slant-range delta",
    "model_validation_flatearth",
)

# =============================================================================
# Figure 2 -- full effect, both terms, on the real topographic ramp
# =============================================================================
results_full = {}
for bxt in BXT_VALUES:
    res_num = np.zeros_like(ALPHA_VALUES)
    res_an = np.zeros_like(ALPHA_VALUES)

    for i, alpha in enumerate(ALPHA_VALUES):
        ramp = _make_topo_ramp(alpha) if alpha > 0 else ()
        cfg, tracks, scene = _build_cfg(bxt, ramp)
        ptg_real = np.asarray(scene.points[-1], dtype=float)
        ptg_ref = np.asarray(scene.ptg, dtype=float)

        # single-point evaluation -- the physically complete residual,
        # no decomposition into flat/elevation parts
        res_num[i] = -1.0 * (_C0(cfg, tracks, ptg_real) - _C0(cfg, tracks, ptg_ref))

        dy = DY_RAMP if alpha > 0 else 0.0
        dh = float(ptg_real[2] - ptg_ref[2])

        # exact slant-range delta for the COMBINED (dy, dh) displacement
        delta_r_exact = np.sqrt((y0 + dy) ** 2 + (H - dh) ** 2) - r0
        b_perp = bxt * cos_t0

        flat_term = (b_perp * delta_r_exact) / (r0 * tan_t0)
        elev_term = (b_perp * dh) / (r0 * sin_t0)
        res_an[i] = flat_term + elev_term

    rel_err_pct = np.zeros_like(ALPHA_VALUES)
    nz = res_an != 0.0
    rel_err_pct[nz] = 100.0 * (res_num[nz] - res_an[nz]) / res_an[nz]
    results_full[bxt] = dict(num=res_num, an=res_an, rel_err_pct=rel_err_pct)

    max_err = np.max(np.abs(rel_err_pct[nz])) if np.any(nz) else 0.0
    print(f"[full effect] bxt={bxt:5.0f} m | max relative error = {max_err:.4f} %")

_two_panel_figure(
    ALPHA_VALUES, r"Ramp elevation angle $\alpha$ [deg]",
    results_full,
    r"Validation against Eq.~(8), full effect: flat-earth + elevation terms combined",
    "model_validation_full_effect",
)

print(f"\nAll done. Output folder: {SAVE_DIR}")