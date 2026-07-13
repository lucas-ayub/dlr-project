# -*- coding: utf-8 -*-
"""
runs/validation/run_band_comparison.py

Band comparison (X/C/L) of the residual phase-polynomial coefficients, and
comparison against the analytic topographic range-error model of Eq. (8)
in Natalia's paper.

INVESTIGATION HISTORY (kept as diagnostic figures A-D below, for the
report's methodology section -- the final, correct comparison is Figure E):

  Step 1 (Figures A, B): res0 = term0(real) - term0(assumed center) mixes a
  flat-earth term (driven by the ramp's fixed horizontal offset dx,dy) with
  an elevation term (driven by dh). Isolating them requires a THIRD
  evaluation point: same (x,y) as the real target, height forced back to
  the reference. Figure A confirms the geometric ramp mapping itself
  (q_analytic = dy*tan(alpha)) is correct; Figure B shows the flat-earth
  term is ~constant across alpha while the elevation term grows smoothly.

  Step 2 (Figure C, "BEFORE fix"): comparing the (still mixed) res_full
  directly against Eq.(8)'s elevation term shows a sharp jump + plateau
  that does not match Eq.(8)'s smooth curve -- the flat-earth contamination.

  Step 3 (Figure D, "ALL CANDIDATES"): even after isolating the elevation
  term, comparing against Eq.(8) with the RAW bxt (no baseline projection)
  and testing both signs still does not match in magnitude. This was later
  resolved (see run_height_sweep_isorange.py) by two further corrections:

    (a) SIGN CONVENTION: this project's phase is phi=+2*pi*C0/wl; Eq.(9) is
        Delta_Phi = -2*pi/wl*Delta_r. The numerical residual must be
        sign-flipped (*-1) to compare directly against Eq.(8)/(9).
    (b) BASELINE PROJECTION: Eq.(8)'s Bn is the PERPENDICULAR baseline,
        B_perp = bxt*cos(theta0), not the raw horizontal bxt.

  Step 4 (Figure E, "AFTER fix", corrected): applying both (a) and (b), the
  elevation-isolated numerical residual matches Eq.(8) closely in the
  linear regime (small dh). A vertical marker flags where the ramp's
  Delta_h = dy*tan(alpha) grows large enough (dy is fixed at ~2000 m) to
  leave that linear regime -- expected nonlinear deviation beyond that
  point, not a bug. The clean, in-regime, multi-baseline validation lives
  in run_model_validation_report.py / run_height_sweep_isorange.py.

Part 1 -- band sweep (unchanged, never had a bug):
    For X/C/L band, each with azimuth resolution = 15*lambda:
      - compute the azimuth (Doppler) bandwidth B_az = ve / resolution
      - re-run the residual coefficient sweep with SystemParams(wl=lambda)
      - convert to phase: phi0 = 2*pi*res0/lambda            (constant, any f)
                          phi1_edge = 2*pi*res1*(B_az/2)      (at band edge)
                          phi2_edge = 2*pi*res2*wl*(B_az/2)**2
    Uses res_full (real vs. assumed center), matching the definition used
    everywhere else in the project (run_coeff_sweep.py, Figures 1-24 of the
    report) -- the physically complete reconstruction error, flat-earth +
    elevation combined.
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

# ------------------- bands -------------------
BANDS = {
    "X": 0.031,   # m  (~9.65 GHz, e.g. TerraSAR-X)
    "C": 0.056,   # m  (~5.405 GHz, e.g. Sentinel-1)
    "L": 0.240,   # m  (~1.25 GHz, e.g. ALOS-2)
}
RES_FACTOR = 15.0     # azimuth resolution = RES_FACTOR * wavelength

# ------------------- sweep parameters -------------------
ALPHA_VALUES = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
BXT_FIXED = 100.0
CHANNEL = 1
DX = 100.0
NRX = 2

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "plots", "band_comparison")
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_VECTOR = True


def _build_cfg(wl, alpha_deg, dxt, nrx=NRX):
    system = SystemParams(wl=wl)
    ramp = _make_topo_ramp(alpha_deg) if alpha_deg > 0 else ()
    scene = Scene(rDelay=0.0051115753, c0=system.c0, h0=0.0, extra_offsets=ramp)

    bat = DX * np.arange(nrx)
    bxt_arr = np.zeros(nrx)
    bxt_arr[CHANNEL] = dxt
    array = ArrayGeometry(bat=bat, bxt=bxt_arr)

    prf, PRF_op = prf_from_fixed(2000.0, nrx)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, nrx, acq_time)

    cfg = sar.ExperimentConfig(
        name="band_cmp", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta, plots_dir=SAVE_DIR,
    )
    tracks = build_platform_tracks(cfg)
    return cfg, tracks, scene


def _call_coeffs(cfg, tracks, ptg):
    kk = CHANNEL
    C0, C1, C2, Dt = GetCoeffNu(
        ptg, tracks.ptx, tracks.prx[kk], tracks.vtx, tracks.vrx[kk],
        tracks.ptx, tracks.vtx,
        cfg.prf, cfg.system.wl, cfg.ta,
        cfg.sq_tx, cfg.sq_rx[kk], cfg.theta_tx, cfg.theta_rx[kk],
    )
    return C0, Dt - C1, C2


def fit_coeffs(wl, alpha_deg, dxt):
    """
    Returns:
        term_real      -- (C0, Dt-C1, C2) at the real ramp-top target
        res_full        -- term_real - term_center (mixes flat-earth + elevation)
        res_elev_only   -- term_real - term_same_xy_h0 (elevation isolated)
        res_flat_only    -- term_same_xy_h0 - term_center (flat-earth isolated)
        scene, dh_actual
    """
    cfg, tracks, scene = _build_cfg(wl, alpha_deg, dxt)

    ptg_real   = np.asarray(scene.points[-1], dtype=float)
    ptg_center = np.asarray(scene.ptg,        dtype=float)
    ptg_same_xy_h0 = ptg_real.copy()
    ptg_same_xy_h0[2] = ptg_center[2]

    term_real   = np.array(_call_coeffs(cfg, tracks, ptg_real))
    term_center = np.array(_call_coeffs(cfg, tracks, ptg_center))
    term_flat   = np.array(_call_coeffs(cfg, tracks, ptg_same_xy_h0))

    res_full      = term_real - term_center
    res_elev_only = term_real - term_flat
    res_flat_only = term_flat - term_center

    dh_actual = float(ptg_real[2] - ptg_center[2])
    return term_real, res_full, res_elev_only, res_flat_only, scene, dh_actual


def _save(fig, name):
    fig.savefig(os.path.join(SAVE_DIR, name + ".png"), dpi=150, bbox_inches="tight")
    if SAVE_VECTOR:
        fig.savefig(os.path.join(SAVE_DIR, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Part 1 -- band sweep (unchanged)
# =============================================================================
SYSTEM_VE = SystemParams().ve

band_results = {}
for band, wl in BANDS.items():
    resolution = RES_FACTOR * wl
    B_az = SYSTEM_VE / resolution

    phi0 = np.zeros_like(ALPHA_VALUES)
    phi1_edge = np.zeros_like(ALPHA_VALUES)
    phi2_edge = np.zeros_like(ALPHA_VALUES)

    for i, a in enumerate(ALPHA_VALUES):
        _, res_full, _, _, _, _ = fit_coeffs(wl, a, BXT_FIXED)
        res0, res1, res2 = res_full
        phi0[i] = 2 * np.pi * res0 / wl
        phi1_edge[i] = 2 * np.pi * res1 * (B_az / 2.0)
        phi2_edge[i] = 2 * np.pi * res2 * wl * (B_az / 2.0) ** 2

    band_results[band] = dict(wl=wl, resolution=resolution, B_az=B_az,
                              phi0=phi0, phi1_edge=phi1_edge, phi2_edge=phi2_edge)
    print(f"{band}-band: wl={wl:.4f} m | resolution={resolution:.3f} m | B_az={B_az:.1f} Hz")

phase_labels = ["$\\phi(\\mathrm{res}\\,C_0)$ [rad]",
               "$\\phi(\\mathrm{res}(\\Delta t-C_1))$ at band edge [rad]",
               "$\\phi(\\mathrm{res}\\,C_2)$ at band edge [rad]"]
phase_keys = ["phi0", "phi1_edge", "phi2_edge"]
phase_tags = ["phase_res0", "phase_res1_edge", "phase_res2_edge"]

for key, label, tag in zip(phase_keys, phase_labels, phase_tags):
    fig, ax = plt.subplots(figsize=(8, 5))
    for band in BANDS:
        ax.plot(ALPHA_VALUES, band_results[band][key], marker="o", label=f"{band}-band")
    ax.set_xlabel("Ramp elevation angle $\\alpha$ [deg]")
    ax.set_ylabel(label)
    ax.set_title(f"Max expected phase vs $\\alpha$ | $b_{{xt}}$={BXT_FIXED:.0f} m, resolution=15$\\lambda$")
    ax.grid()
    ax.legend(fontsize="small")
    fig.tight_layout()
    _save(fig, tag)

# =============================================================================
# Part 2 -- model comparison against Eq. (8): full investigation history
# =============================================================================
wl_ref = BANDS["L"]

res_full_arr   = np.zeros_like(ALPHA_VALUES)
res_elev_arr   = np.zeros_like(ALPHA_VALUES)
res_flat_arr   = np.zeros_like(ALPHA_VALUES)
q_analytic_arr = np.zeros_like(ALPHA_VALUES)
dh_actual_arr  = np.zeros_like(ALPHA_VALUES)
eq8_elev_raw_arr  = np.zeros_like(ALPHA_VALUES)   # Step 3: no B_perp (historical)
eq8_elev_perp_arr = np.zeros_like(ALPHA_VALUES)   # Step 4: WITH B_perp (correct)
r0_arr         = np.zeros_like(ALPHA_VALUES)
sin_theta0_arr = np.zeros_like(ALPHA_VALUES)
cos_theta0_arr = np.zeros_like(ALPHA_VALUES)

print("\n--- Part 2 diagnostics: q_analytic vs dh_actual, per alpha ---")
print(f"{'alpha':>6} | {'dy':>10} | {'q_analytic':>12} | {'dh_actual':>12} | "
      f"{'r0':>12} | {'sin(theta0)':>12} | {'res_full':>10} | "
      f"{'res_flat':>10} | {'res_elev':>10}")

for i, a in enumerate(ALPHA_VALUES):
    term_real, res_full, res_elev, res_flat, scene, dh_actual = fit_coeffs(wl_ref, a, BXT_FIXED)

    r0 = scene.r0
    y0 = scene.y0
    sin_theta0 = y0 / r0
    cos_theta0 = scene.H / r0
    tan_theta0 = sin_theta0 / cos_theta0
    dy = scene.extra_offsets[-1][1] if a > 0 else 0.0
    q = dy * np.tan(np.radians(a))

    q_analytic_arr[i] = q
    dh_actual_arr[i]  = dh_actual
    r0_arr[i]          = r0
    sin_theta0_arr[i]  = sin_theta0
    cos_theta0_arr[i]  = cos_theta0

    b_perp = BXT_FIXED * cos_theta0
    eq8_elev_raw_arr[i]  = (BXT_FIXED * dh_actual) / (r0 * sin_theta0)   # historical (Step 3)
    eq8_elev_perp_arr[i] = (b_perp   * dh_actual) / (r0 * sin_theta0)   # correct (Step 4)

    res_full_arr[i] = res_full[0]
    res_elev_arr[i] = res_elev[0]
    res_flat_arr[i] = res_flat[0]

    print(f"{a:6.1f} | {dy:10.3f} | {q:12.4f} | {dh_actual:12.4f} | "
          f"{r0:12.2f} | {sin_theta0:12.5f} | {res_full[0]:10.4f} | "
          f"{res_flat[0]:10.4f} | {res_elev[0]:10.4f}")

if not np.allclose(r0_arr, r0_arr[0]) or not np.allclose(sin_theta0_arr, sin_theta0_arr[0]):
    print("\n[WARNING] r0/theta0 are NOT constant across alpha.")
else:
    print(f"\n[OK] r0={r0_arr[0]:.2f} m, sin(theta0)={sin_theta0_arr[0]:.5f} "
          f"constant across all alpha, as expected.")

flat_std = np.std(res_flat_arr[1:])
flat_mean = np.mean(res_flat_arr[1:])
print(f"[CHECK] res_flat_only: mean={flat_mean:.5f} m, std={flat_std:.2e} m "
      f"({'looks constant, as expected' if flat_std < 0.01 * abs(flat_mean) else 'NOT constant'})")

recon_err = np.max(np.abs(res_full_arr - (res_flat_arr + res_elev_arr)))
print(f"[CHECK] max|res_full - (res_flat + res_elev)| = {recon_err:.2e} m (should be ~0)")

# Best-fit k of (-res_elev_arr) against eq8_elev_perp_arr, and linear-regime flag
res_elev_signed = -1.0 * res_elev_arr   # sign convention (a) applied
mask = eq8_elev_perp_arr != 0.0
k_fit = np.sum(eq8_elev_perp_arr[mask] * res_elev_signed[mask]) / np.sum(eq8_elev_perp_arr[mask] ** 2)
rel_err = np.full_like(ALPHA_VALUES, np.nan)
rel_err[mask] = np.abs((res_elev_signed[mask] - eq8_elev_perp_arr[mask]) / eq8_elev_perp_arr[mask]) * 100
beyond_5pct = np.where(rel_err > 5.0)[0]
alpha_linear_limit = ALPHA_VALUES[beyond_5pct[0]] if len(beyond_5pct) else None
print(f"[FINAL] best-fit k (with B_perp + sign applied) = {k_fit:+.4f} "
      f"(expect ~+1.0 within the linear regime)")

# --- Figure A: geometric mapping check ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ALPHA_VALUES, q_analytic_arr, marker="o", label="$q$ analytic ($dy\\cdot\\tan\\alpha$)")
ax.plot(ALPHA_VALUES, dh_actual_arr, marker="s", linestyle="--", label="$\\Delta h$ actual (from scene)")
ax.set_xlabel("Ramp elevation angle $\\alpha$ [deg]")
ax.set_ylabel("Elevation [m]")
ax.set_title("Step 1: geometric mapping check ($q$ analytic vs. actual scene $\\Delta h$)")
ax.grid(); ax.legend(fontsize="small")
fig.tight_layout()
_save(fig, "diag_q_vs_dh_actual")

# --- Figure B: numerical decomposition ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(ALPHA_VALUES, res_full_arr, marker="o", color="k", linewidth=2, label="res (total, real vs. assumed center)")
ax.plot(ALPHA_VALUES, res_flat_arr, marker="s", linestyle="--", label="res (flat-earth only, isolated)")
ax.plot(ALPHA_VALUES, res_elev_arr, marker="^", linestyle="--", label="res (elevation only, isolated)")
ax.set_xlabel("Ramp elevation angle $\\alpha$ [deg]")
ax.set_ylabel("res$(C_0)$ [m]")
ax.set_title("Step 1: numerical decomposition -- flat-earth vs. elevation")
ax.grid(); ax.legend(fontsize="small")
fig.tight_layout()
_save(fig, "diag_decomposition_flat_vs_elev")

# --- Figure C: mixed comparison (historical, before isolating elevation) ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ALPHA_VALUES, res_full_arr, marker="o", label="Numerical (res_full, mixed)")
ax.plot(ALPHA_VALUES, eq8_elev_raw_arr, marker="s", linestyle="--", label="Analytic (Eq. 8, elevation, raw $b_{xt}$)")
ax.set_xlabel("Ramp elevation angle $\\alpha$ [deg]")
ax.set_ylabel("res$(C_0)$ [m]")
ax.set_title(f"Step 2 (historical): mixed numerical vs. Eq.(8) | L-band, $b_{{xt}}$={BXT_FIXED:.0f} m")
ax.grid(); ax.legend(fontsize="small")
fig.tight_layout()
_save(fig, "model_comparison_res0_vs_alpha_BEFORE_FIX")

# --- Figure D: elevation isolated, raw bxt, sign candidates (historical) ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(ALPHA_VALUES, res_elev_arr, marker="o", linewidth=2, color="k", label="Numerical (elevation isolated)")
ax.plot(ALPHA_VALUES, eq8_elev_raw_arr, marker="s", linestyle="--", label="Analytic, Eq.(8), raw $b_{xt}$")
ax.plot(ALPHA_VALUES, -eq8_elev_raw_arr, marker="s", linestyle=":", label="Analytic, sign-flipped")
ax.set_xlabel("Ramp elevation angle $\\alpha$ [deg]")
ax.set_ylabel("res$(C_0)$ [m]")
ax.set_title(f"Step 3 (historical): sign candidates, still no $B_\\perp$ | L-band, $b_{{xt}}$={BXT_FIXED:.0f} m")
ax.grid(); ax.legend(fontsize="small")
fig.tight_layout()
_save(fig, "model_comparison_res0_vs_alpha_ALL_CANDIDATES")

# --- Figure E: FINAL, corrected comparison (B_perp + sign + linear-regime marker) ---
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(ALPHA_VALUES, res_elev_signed, marker="o", color="k",
        label="Numerical (elevation isolated, sign-corrected)")
ax.plot(ALPHA_VALUES, eq8_elev_perp_arr, marker="s", linestyle="--",
        label="Analytic, Eq. (8) elevation term ($B_\\perp=b_{xt}\\cos\\theta_0$)")
if alpha_linear_limit is not None:
    ax.axvline(alpha_linear_limit, color="gray", linestyle=":", linewidth=1)
    ax.text(alpha_linear_limit, ax.get_ylim()[1] * 0.9,
            f"  linear regime limit\n  (>5% dev., $\\alpha$={alpha_linear_limit:.0f}°)",
            fontsize=8, color="gray")
ax.set_xlabel("Ramp elevation angle $\\alpha$ [deg]")
ax.set_ylabel("$\\Delta r$ [m]")
ax.set_title(f"Step 4 (FINAL, corrected): numerical vs. Eq. (8) | L-band, $b_{{xt}}$={BXT_FIXED:.0f} m")
ax.grid(); ax.legend(fontsize="small")
fig.tight_layout()
_save(fig, "model_comparison_ELEVATION_ONLY_fixed")

print(f"\nAll done. Output folder: {SAVE_DIR}")
if alpha_linear_limit is not None:
    print(f"Note: deviates >5% beyond alpha={alpha_linear_limit:.0f} deg "
          f"(dh={dh_actual_arr[beyond_5pct[0]]:.0f} m) -- expected nonlinearity, not a bug. "
          f"See run_model_validation_report.py for the clean small-dh, multi-baseline validation.")