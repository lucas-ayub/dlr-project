# -*- coding: utf-8 -*-
"""
run_height_sweep_isorange.py

ISO-RANGE height sweep: the definitive comparison of this project's numerical
Delta_r (GetCoeffNu residual C0) against Natalia's analytic elevation term.

WHY ISO-RANGE (the resolution of the k=-0.110 mystery):
    The previous sweep raised the target at CONSTANT GROUND POSITION
    (dx=dy=0, only h changed). That changes the target's slant range along
    with its height -- a different derivative than the one Eq. (8) describes.
    InSAR-style height-sensitivity formulas (Natalia's Eq. 8 included) are
    defined for targets AT THE SAME SLANT RANGE (same range bin) as the
    reference, i.e. the target moves along the iso-range circle as h grows.

    Exact geometry, small-dh limit:
      constant ground:  dDr/dh = -b*sin(t0)*cos(t0)/r0
      constant range :  dDr/dh = -b*cos(t0)/(r0*sin(t0))
      ratio (ground)/(code's old analytic +b/(r0 sin t0)) = -sin^2(t0)*cos(t0)
          = -0.1099 for t0=20 deg  ->  EXACTLY the measured k=-0.110.

    Additionally, Eq. (8)'s Bn is the PERPENDICULAR baseline (B_perp =
    bxt*cos(theta0)), not the raw horizontal bxt. Both fixes are applied here.

PREDICTION for this script:
    dr_numeric == -(bxt*cos(t0))*dh/(r0*sin(t0)) == -bxt*dh/(r0*tan(t0))
    -> best-fit k against the corrected analytic should be -1.000, the
    remaining minus sign being exactly Eq. (9)'s convention
    (Delta_Phi = -2*pi/lambda * Delta_r) vs. this project's phi = +2*pi*C0/wl.

Outputs (in plots/band_comparison/):
    dr_isorange_overlay.png/.pdf     -- numeric vs analytic candidates
    dr_isorange_difference.png/.pdf  -- explicit differences
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

DH_MAX    = 200.0         # max height offset [m] -- linear regime
N_DH      = 21
DH_VALUES = np.linspace(0.0, DH_MAX, N_DH)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "plots", "band_comparison")
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
        name="isorange_sweep", system=system, scene=scene, array=array,
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
# Reference geometry (fixed for all dh)
# =============================================================================
cfg0, tracks0, scene0 = _build_cfg(())
r0 = scene0.r0
y0 = scene0.y0
H  = scene0.H
sin_t0 = y0 / r0
cos_t0 = (H - 0.0) / r0          # h0 = 0
tan_t0 = sin_t0 / cos_t0
theta0_deg = np.degrees(np.arcsin(sin_t0))

print(f"Reference geometry: r0={r0:.2f} m | y0={y0:.2f} m | H={H:.0f} m")
print(f"theta0={theta0_deg:.3f} deg | sin={sin_t0:.5f} | cos={cos_t0:.5f} | tan={tan_t0:.5f}")
print(f"Predicted old-experiment k = -sin^2*cos = {-sin_t0**2 * cos_t0:+.4f} "
      f"(measured before: -0.110)")

# =============================================================================
# Sweep: target ON THE ISO-RANGE SURFACE (same r0, height dh)
#   y_target(dh) = sqrt(r0^2 - (H - dh)^2)   ->  |P_ref_track - target| = r0
# =============================================================================
dr_numeric   = np.zeros_like(DH_VALUES)
an_raw       = np.zeros_like(DH_VALUES)   # old analytic: +bxt*dh/(r0*sin)
an_perp      = np.zeros_like(DH_VALUES)   # corrected:    +bxt*cos*dh/(r0*sin) = +bxt*dh/(r0*tan)
an_predicted = np.zeros_like(DH_VALUES)   # full prediction incl. sign: -bxt*dh/(r0*tan)

print(f"\n{'dh [m]':>8} | {'y_tgt-y0':>10} | {'dr_numeric':>12} | "
      f"{'an_perp':>12} | {'num/an_perp':>12}")

for i, dh in enumerate(DH_VALUES):
    if dh == 0.0:
        extra = ()
    else:
        y_target = np.sqrt(r0**2 - (H - dh)**2)
        dy = y_target - y0            # iso-range: move outward in ground range as h rises
        extra = ((0.0, float(dy), float(dh)),)

    cfg, tracks, scene = _build_cfg(extra)
    ptg_real = np.asarray(scene.points[-1], dtype=float)
    ptg_ref  = np.asarray(scene.ptg,        dtype=float)

    # sanity: verify the real target is indeed at slant range r0 from the
    # reference track point (0, 0, H) at broadside
    r_check = np.linalg.norm(ptg_real - np.array([ptg_real[0], 0.0, H]))
    assert abs(r_check - r0) < 1e-6 * r0, f"iso-range violated: {r_check} vs {r0}"

    dr_numeric[i]   = _C0(cfg, tracks, ptg_real) - _C0(cfg, tracks, ptg_ref)
    an_raw[i]       = (BXT_FIXED * dh) / (r0 * sin_t0)
    an_perp[i]      = (BXT_FIXED * cos_t0 * dh) / (r0 * sin_t0)
    an_predicted[i] = -an_perp[i]

    ratio = dr_numeric[i] / an_perp[i] if an_perp[i] != 0 else float("nan")
    print(f"{dh:8.1f} | {(np.sqrt(r0**2-(H-dh)**2)-y0) if dh>0 else 0.0:10.4f} | "
          f"{dr_numeric[i]:12.6f} | {an_perp[i]:12.6f} | {ratio:12.6f}")

# Best-fit k against BOTH analytic candidates
def _fit_k(an):
    m = an != 0.0
    k = np.sum(an[m] * dr_numeric[m]) / np.sum(an[m] ** 2)
    resid = dr_numeric - k * an
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((dr_numeric - np.mean(dr_numeric)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return k, r2, resid

k_raw,  r2_raw,  _        = _fit_k(an_raw)
k_perp, r2_perp, resid_pp = _fit_k(an_perp)

print(f"\nBest-fit vs OLD analytic (raw bxt, 1/sin):        k = {k_raw:+.4f} | R^2 = {r2_raw:.6f}")
print(f"Best-fit vs CORRECTED analytic (B_perp = bxt*cos): k = {k_perp:+.4f} | R^2 = {r2_perp:.6f}")
print("Expected: k_perp = -1.0000 -> only Eq.(9)'s sign convention remains.")

# =============================================================================
# Plots
# =============================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(DH_VALUES, dr_numeric, marker="o", color="k", linewidth=2,
        label="Numerical $\\Delta r$ (GetCoeffNu res $C_0$, iso-range)")
ax.plot(DH_VALUES, an_predicted, marker="s", linestyle="--",
        label="Prediction: $-b_{xt}\\cos\\theta_0\\,\\Delta h/(r_0\\sin\\theta_0)$")
ax.plot(DH_VALUES, an_perp, marker="^", linestyle=":",
        label="Eq.(8) elev. term, $B_\\perp=b_{xt}\\cos\\theta_0$ (unsigned)")
ax.plot(DH_VALUES, an_raw, marker="v", linestyle=":",
        label="Old analytic (raw $b_{xt}$, no $\\cos\\theta_0$)")
ax.set_xlabel("Height offset $\\Delta h$ [m] (target kept at slant range $r_0$)")
ax.set_ylabel("$\\Delta r$ [m]")
ax.set_title(f"Iso-range height sweep | L-band, $b_{{xt}}$={BXT_FIXED:.0f} m, "
            f"$\\theta_0$={theta0_deg:.1f}°")
ax.grid()
ax.legend(fontsize="small")
fig.tight_layout()
_save(fig, "dr_isorange_overlay")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(DH_VALUES, dr_numeric - an_predicted, marker="o",
        label="$\\Delta r_{num} - \\Delta r_{predicted}$ (should be ~0)")
ax.plot(DH_VALUES, resid_pp, marker="^", linestyle=":",
        label=f"$\\Delta r_{{num}} - ({k_perp:+.4f})\\,\\Delta r_{{an,\\perp}}$ (best fit)")
ax.axhline(0.0, color="gray", linewidth=0.8)
ax.set_xlabel("Height offset $\\Delta h$ [m]")
ax.set_ylabel("Difference in $\\Delta r$ [m]")
ax.set_title(f"Iso-range sweep: residual vs. prediction | L-band, "
            f"$b_{{xt}}$={BXT_FIXED:.0f} m")
ax.grid()
ax.legend(fontsize="small")
fig.tight_layout()
_save(fig, "dr_isorange_difference")

print(f"\nAll done. Output folder: {SAVE_DIR}")