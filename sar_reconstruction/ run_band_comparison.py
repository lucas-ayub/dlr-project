# -*- coding: utf-8 -*-
"""
Band comparison (X/C/L) of the residual phase-polynomial coefficients, and
comparison against the analytic topographic range-error model of Eq. (8)
in Natalia's paper:

    Delta_r_m(x, r) ~= (Bn * r) / (r0 * tan(theta0))   <- flat-earth, range-dependent term
                     + (Bn * q) / (r0 * sin(theta0))    <- elevation-dependent (topographic) term

Mapping to this project's variables:
    Bn      <-> bxt      (cross-track baseline; assumed ~perpendicular to LOS)
    r       <-> dy        (fixed ground-range offset of the ramp's top point, 2000 m)
    q       <-> dh = dy*tan(alpha)   (elevation of the ramp's top point; alpha
                                     is this project's parametrisation of q,
                                     NOT the same quantity as q itself)
    r0, theta0  <-> scene incidence geometry, computed from Scene

Our own C0 (from GetCoeffNu) is a two-way path-length quantity in metres
(phase = 2*pi*C0/wl), directly comparable to Delta_r_m without conversion.
"term0" should therefore match the FULL Eq. (8) (both terms), and "res0"
(term0 at the real target minus term0 at the assumed center) should match
ONLY the elevation term of Eq. (8), since the range term r is common to
both GetCoeffNu evaluations and cancels in the subtraction. This script
checks that prediction directly.

Part 1 -- band sweep:
    For X/C/L band, each with azimuth resolution = 15*lambda:
      - compute the azimuth (Doppler) bandwidth B_az = ve / resolution
      - re-run the residual coefficient sweep with SystemParams(wl=lambda)
      - convert to phase: phi0 = 2*pi*res0/lambda            (constant, any f)
                          phi1_edge = 2*pi*res1*(B_az/2)      (at band edge)
                          phi2_edge = 2*pi*res2*lambda*(B_az/2)**2
      - plot phi0/phi1_edge/phi2_edge vs alpha, one curve per band

Part 2 -- model comparison:
    For the L-band system, plot res0 (numerical, from GetCoeffNu) against
    the analytic elevation-only term of Eq. (8), both vs alpha.
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
# Refined near alpha=0 to resolve the fast initial rise seen previously.
ALPHA_VALUES = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
BXT_FIXED = 100.0     # representative baseline for the band comparison
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
    scene = Scene(rDelay=0.0051115753, c0=system.c0, h0=0.0,
                  extra_offsets=ramp)

    bat = DX * np.arange(nrx)
    bxt_arr = np.zeros(nrx)
    bxt_arr[CHANNEL] = dxt
    array = ArrayGeometry(bat=bat, bxt=bxt_arr)

    prf, PRF_op = prf_from_fixed(2000.0, nrx)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, nrx, acq_time)

    cfg = sar.ExperimentConfig(
        name="band_cmp", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta,
        plots_dir=SAVE_DIR,
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
    cfg, tracks, scene = _build_cfg(wl, alpha_deg, dxt)
    ptg_real = scene.points[-1]
    ptg_ref = scene.ptg
    term = np.array(_call_coeffs(cfg, tracks, ptg_real))
    term_ref = np.array(_call_coeffs(cfg, tracks, ptg_ref))
    res = term - term_ref
    return term, res, scene


def _save(fig, name):
    fig.savefig(os.path.join(SAVE_DIR, name + ".png"), dpi=150, bbox_inches="tight")
    if SAVE_VECTOR:
        fig.savefig(os.path.join(SAVE_DIR, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------- Part 1: band sweep -------------------
SYSTEM_VE = SystemParams().ve   # platform velocity is band-independent

band_results = {}
for band, wl in BANDS.items():
    resolution = RES_FACTOR * wl
    B_az = SYSTEM_VE / resolution

    phi0 = np.zeros_like(ALPHA_VALUES)
    phi1_edge = np.zeros_like(ALPHA_VALUES)
    phi2_edge = np.zeros_like(ALPHA_VALUES)

    for i, a in enumerate(ALPHA_VALUES):
        _, res, _ = fit_coeffs(wl, a, BXT_FIXED)
        res0, res1, res2 = res
        phi0[i] = 2 * np.pi * res0 / wl
        phi1_edge[i] = 2 * np.pi * res1 * (B_az / 2.0)
        phi2_edge[i] = 2 * np.pi * res2 * wl * (B_az / 2.0) ** 2

    band_results[band] = dict(wl=wl, resolution=resolution, B_az=B_az,
                              phi0=phi0, phi1_edge=phi1_edge, phi2_edge=phi2_edge)

    print(f"{band}-band: wl={wl:.4f} m | resolution={resolution:.3f} m | "
          f"B_az={B_az:.1f} Hz")

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
    ax.set_title(f"Max expected phase vs $\\alpha$ | $b_{{xt}}$={BXT_FIXED:.0f} m, "
                f"resolution=15$\\lambda$")
    ax.grid()
    ax.legend(fontsize="small")
    fig.tight_layout()
    _save(fig, tag)

# ------------------- Part 2: comparison against Eq. (8) -------------------
# Only the elevation-dependent term of Eq. (8) is compared, since res0
# isolates exactly that contribution (the range term r is common to both
# GetCoeffNu evaluations and cancels in the subtraction).
wl_ref = BANDS["L"]
res0_numeric = np.zeros_like(ALPHA_VALUES)
res0_analytic = np.zeros_like(ALPHA_VALUES)

for i, a in enumerate(ALPHA_VALUES):
    _, res, scene = fit_coeffs(wl_ref, a, BXT_FIXED)
    res0_numeric[i] = res[0]

    r0 = scene.r0
    y0 = scene.y0
    sin_theta0 = y0 / r0
    dy = scene.extra_offsets[-1][1] if a > 0 else 0.0
    q = dy * np.tan(np.radians(a))     # elevation; alpha is how q is parametrised here
    res0_analytic[i] = (BXT_FIXED * q) / (r0 * sin_theta0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ALPHA_VALUES, res0_numeric, marker="o", label="Numerical (GetCoeffNu residual)")
ax.plot(ALPHA_VALUES, res0_analytic, marker="s", linestyle="--",
        label="Analytic (Eq. 8, elevation term)")
ax.set_xlabel("Ramp elevation angle $\\alpha$ [deg]")
ax.set_ylabel("res$(C_0)$ [m]")
ax.set_title(f"Model comparison: numerical vs. Eq. (8) | L-band, $b_{{xt}}$={BXT_FIXED:.0f} m")
ax.grid()
ax.legend(fontsize="small")
fig.tight_layout()
_save(fig, "model_comparison_res0_vs_alpha")

print(f"\nAll done. Output folder: {SAVE_DIR}")