# -*- coding: utf-8 -*-
"""
runs/validation/run_range_sweep_flatearth.py

RANGE (flat-earth) sweep: validates the FIRST term of Natalia's Eq. (8) --
the flat-earth, range-dependent term -- against this project's numerical
GetCoeffNu residual. This completes the validation started in
run_height_sweep_isorange.py (which validated only the SECOND term,
elevation), closing the loop on the full Eq. (8) formula.

    Delta_r_m(x, r) ~= (Bn * r) / (r0 * tan(theta0))   <- validated HERE
                     + (Bn * q) / (r0 * sin(theta0))    <- validated in
                                                            run_height_sweep_isorange.py

WHY THIS EXPERIMENT NEEDS NO DECOMPOSITION:
    The target is displaced ONLY in cross-range (dy), with height fixed at
    h=0 for every point in the sweep. There is no elevation term at all in
    this configuration, so the raw residual
        res_full = C0(displaced target) - C0(reference center)
    IS the flat-earth term in isolation -- no need for the three-point
    decomposition used on the topographic ramp scene.

WHY "r" IS NOT SIMPLY "dy" (the key correction this script identifies):
    The project's original mapping (r <-> dy, i.e. the raw ground-range
    offset) is only a first-order approximation. The exact quantity Eq.(8)
    needs is the SLANT-RANGE difference induced by that ground-range
    offset:
        delta_r_exact = r(dy) - r0 = sqrt((y0+dy)^2 + H^2) - r0
    which differs from the naive dy*sin(theta0) approximation once dy is
    not small. Both are computed here so the two can be compared directly.

Analogous to the elevation-term script, this also applies:
  (1) SIGN CONVENTION: phi=+2*pi*C0/wl (this project) vs.
      Delta_Phi=-2*pi/wl*Delta_r (Eq. 9) -> numerical residual sign-flipped.
  (2) BASELINE PROJECTION: Bn = B_perp = bxt*cos(theta0), not raw bxt.

Outputs (in plots/flatearth_sweep/):
    dr_flatearth_overlay.png/.pdf     -- numeric vs analytic candidates
    dr_flatearth_difference.png/.pdf  -- explicit differences
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
# Parameters
# =============================================================================
WL        = 0.240         # L-band
BXT_FIXED = 100.0         # cross-track baseline of the analysed channel [m]
CHANNEL   = 1
DX        = 100.0         # along-track receiver spacing [m] (irrelevant for C0 check)
NRX       = 2

DY_MAX    = 2000.0        # max ground-range offset [m] -- matches the topo ramp's dy
N_DY      = 21
DY_VALUES = np.linspace(0.0, DY_MAX, N_DY)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "plots", "flatearth_sweep")
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_VECTOR = True


# =============================================================================
# Helpers
# =============================================================================
def _build_cfg(extra_offsets):
    system = SystemParams(wl=WL)
    scene = Scene(rDelay=0.0051115753, c0=system.c0, h0=0.0,
                  extra_offsets=extra_offsets)

    bat = DX * np.arange(NRX)
    bxt_arr = np.zeros(NRX)
    bxt_arr[CHANNEL] = BXT_FIXED
    array = ArrayGeometry(bat=bat, bxt=bxt_arr)

    prf, PRF_op = prf_from_fixed(2000.0, NRX)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, NRX, acq_time)

    cfg = sar.ExperimentConfig(
        name="flatearth_sweep", system=system, scene=scene, array=array,
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


def _save(fig, name):
    fig.savefig(os.path.join(SAVE_DIR, name + ".png"), dpi=150, bbox_inches="tight")
    if SAVE_VECTOR:
        fig.savefig(os.path.join(SAVE_DIR, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Reference geometry (fixed for all dy)
# =============================================================================
cfg0, tracks0, scene0 = _build_cfg(())
r0 = scene0.r0
y0 = scene0.y0
H  = scene0.H
sin_t0 = y0 / r0
cos_t0 = H / r0
tan_t0 = sin_t0 / cos_t0
theta0_deg = np.degrees(np.arcsin(sin_t0))

print(f"Reference geometry: r0={r0:.2f} m | y0={y0:.2f} m | H={H:.0f} m")
print(f"theta0={theta0_deg:.3f} deg | sin={sin_t0:.5f} | cos={cos_t0:.5f} | tan={tan_t0:.5f}")

# =============================================================================
# Sweep: target displaced ONLY in cross-range (dy), height fixed at h=0
# =============================================================================
dr_numeric        = np.zeros_like(DY_VALUES)
delta_r_exact_arr = np.zeros_like(DY_VALUES)
an_raw            = np.zeros_like(DY_VALUES)   # old mapping: r <-> dy, no B_perp
an_perp_approx    = np.zeros_like(DY_VALUES)   # B_perp applied, r <-> dy (still approximate)
an_perp_exact     = np.zeros_like(DY_VALUES)   # B_perp applied, r <-> exact slant-range delta
an_predicted      = np.zeros_like(DY_VALUES)   # an_perp_exact with the sign convention applied

print(f"\n{'dy [m]':>8} | {'delta_r_exact':>13} | {'dy*sin(t0)':>11} | "
      f"{'dr_numeric':>12} | {'an_perp_exact':>13}")

for i, dy in enumerate(DY_VALUES):
    extra = ((0.0, float(dy), 0.0),) if dy != 0.0 else ()

    cfg, tracks, scene = _build_cfg(extra)
    ptg_real = np.asarray(scene.points[-1], dtype=float)
    ptg_ref  = np.asarray(scene.ptg,        dtype=float)

    dr_numeric[i] = _C0(cfg, tracks, ptg_real) - _C0(cfg, tracks, ptg_ref)

    r_real = np.sqrt((y0 + dy) ** 2 + H ** 2)
    delta_r_exact = r_real - r0
    delta_r_exact_arr[i] = delta_r_exact

    b_perp = BXT_FIXED * cos_t0
    an_raw[i]         = (BXT_FIXED * dy)          / (r0 * tan_t0)
    an_perp_approx[i] = (b_perp    * dy)          / (r0 * tan_t0)
    an_perp_exact[i]  = (b_perp    * delta_r_exact) / (r0 * tan_t0)
    an_predicted[i]   = -an_perp_exact[i]

    print(f"{dy:8.1f} | {delta_r_exact:13.5f} | {dy*sin_t0:11.5f} | "
          f"{dr_numeric[i]:12.6f} | {an_perp_exact[i]:13.6f}")

# =============================================================================
# Best-fit k against all three analytic candidates
# =============================================================================
def _fit_k(an):
    m = an != 0.0
    k = np.sum(an[m] * dr_numeric[m]) / np.sum(an[m] ** 2)
    resid = dr_numeric - k * an
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((dr_numeric - np.mean(dr_numeric)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return k, r2, resid

k_raw,   r2_raw,   _         = _fit_k(an_raw)
k_approx, r2_approx, _       = _fit_k(an_perp_approx)
k_exact, r2_exact, resid_exc = _fit_k(an_perp_exact)

print(f"\nBest-fit vs raw mapping   (no B_perp,  r=dy):            k = {k_raw:+.4f} | R^2 = {r2_raw:.6f}")
print(f"Best-fit vs B_perp only   (B_perp,     r=dy):            k = {k_approx:+.4f} | R^2 = {r2_approx:.6f}")
print(f"Best-fit vs FULL fix      (B_perp, r=exact slant delta): k = {k_exact:+.4f} | R^2 = {r2_exact:.6f}")
print("Expected: k_exact = -1.0000 -> only Eq.(9)'s sign convention remains.")

# =============================================================================
# Plots
# =============================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(DY_VALUES, dr_numeric, marker="o", color="k", linewidth=2,
        label="Numerical $\\Delta r$ (GetCoeffNu res $C_0$, $h=0$ fixed)")
ax.plot(DY_VALUES, an_predicted, marker="s", linestyle="--",
        label="Prediction: $-B_\\perp\\cdot\\Delta r_{exact}/(r_0\\tan\\theta_0)$")
ax.plot(DY_VALUES, an_perp_exact, marker="^", linestyle=":",
        label="Eq.(8) flat term, $B_\\perp$ + exact slant-range delta (unsigned)")
ax.plot(DY_VALUES, an_perp_approx, marker="v", linestyle=":",
        label="Eq.(8) flat term, $B_\\perp$ + $r=dy$ approx.")
ax.plot(DY_VALUES, an_raw, marker="d", linestyle=":",
        label="Old mapping: raw $b_{xt}$, $r=dy$ (no corrections)")
ax.set_xlabel("Ground-range offset $dy$ [m] (height fixed at $h=0$)")
ax.set_ylabel("$\\Delta r$ [m]")
ax.set_title(f"Flat-earth term sweep | L-band, $b_{{xt}}$={BXT_FIXED:.0f} m, "
            f"$\\theta_0$={theta0_deg:.1f}°")
ax.grid()
ax.legend(fontsize="small")
fig.tight_layout()
_save(fig, "dr_flatearth_overlay")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(DY_VALUES, dr_numeric - an_predicted, marker="o",
        label="$\\Delta r_{num} - \\Delta r_{predicted}$ (should be ~0)")
ax.plot(DY_VALUES, resid_exc, marker="^", linestyle=":",
        label=f"$\\Delta r_{{num}} - ({k_exact:+.4f})\\,\\Delta r_{{an,exact}}$ (best fit)")
ax.axhline(0.0, color="gray", linewidth=0.8)
ax.set_xlabel("Ground-range offset $dy$ [m]")
ax.set_ylabel("Difference in $\\Delta r$ [m]")
ax.set_title(f"Flat-earth sweep: residual vs. prediction | L-band, "
            f"$b_{{xt}}$={BXT_FIXED:.0f} m")
ax.grid()
ax.legend(fontsize="small")
fig.tight_layout()
_save(fig, "dr_flatearth_difference")

print(f"\nAll done. Output folder: {SAVE_DIR}")