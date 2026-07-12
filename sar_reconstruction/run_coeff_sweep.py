# -*- coding: utf-8 -*-
"""
Sweep of the reconstruction phase-polynomial coefficients as a function of
the ramp elevation angle (alpha), the cross-track baseline (bxt), and the
number of receive channels (Nrx).

GetCoeffNu is called EXACTLY as it is inside ReconstructSignalNumeri (same
signature, same track-building via build_platform_tracks). For each
(Nrx, alpha, bxt) it is evaluated twice:

  1) at the TOP of the ramp (the real target position)      -> "term_i"
  2) at the scene CENTER, h=0 (what the reconstruction       -> "term_i (ref)"
     actually assumes for every target in the range bin)

The reconstruction filter's phase is a second-order polynomial in Doppler
frequency f:

    phi(f) = 2*pi * [ C0/wl + (Dt - C1)*f + C2*wl*f^2 ]

so the three terms plotted are the three coefficients of that polynomial:

    term_0 = C0            -- constant phase offset      [m]
    term_1 = Dt - C1        -- coefficient of f (linear)  [s]
    term_2 = C2             -- coefficient of f^2         [m/s^2]

The RESIDUAL terms (res_0, res_1, res_2) are the difference

    res_i = term_i(real target) - term_i(assumed center, h=0)

This residual is the quantity that actually enters the reconstruction
error: GetCoeffNu is only ever evaluated at the assumed scene center, so
the mismatch between the coefficients a real, elevated target would need
and the coefficients the reconstruction actually uses is exactly res_i.
This isolates the topographic-mismatch effect from the constant offset
caused by the ramp's fixed 2000 m range displacement (which cancels out
in the subtraction).

CHANNEL is fixed at index 1. The array is built manually (not via
ArrayGeometry.linear) so that bxt[CHANNEL] == dxt exactly, regardless of
Nrx -- ArrayGeometry.linear() centres the array around the transmitter,
so the effective baseline of a fixed channel index would otherwise shift
with Nrx.

For each (Nrx, alpha, bxt):
  - build a topo_ramp scene with that alpha, referenced at h=0
  - build an array of size Nrx with bxt[CHANNEL] = dxt
  - call GetCoeffNu() for channel CHANNEL at the top-of-ramp target
  - call GetCoeffNu() for channel CHANNEL at the scene center (h=0)
  - store term_0/1/2 (real target) and res_0/1/2 (residual vs. assumed
    center)

Produces, per Nrx, in its own subfolder:
  - term_i vs alpha / vs bxt / surface           -> plots_2d/, plots_3d/
  - res_i  vs alpha / vs bxt / surface           -> plots_2d/, plots_3d/
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

# ------------------- sweep parameters -------------------
NRX_VALUES   = [2, 3, 4, 6]                                          # channel counts to sweep
ALPHA_VALUES = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0])   # [deg]
BXT_VALUES   = np.array([0.0, 10.0, 20.0, 50.0, 100.0])             # [m]
CHANNEL      = 1          # receiver index; bxt[CHANNEL] is forced to dxt for any Nrx
SAVE_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "plots", "coeff_sweep")
SAVE_VECTOR  = True


def _build_cfg(alpha_deg: float, dxt: float, nrx: int):
    """Build an ExperimentConfig + tracks for the given (alpha, dxt, nrx)."""
    system = SystemParams()

    ramp = _make_topo_ramp(alpha_deg) if alpha_deg > 0 else ()
    scene = Scene(rDelay=0.0051115753, c0=system.c0, h0=0.0,
                  extra_offsets=ramp)

    dx = 100.0
    bat = dx * np.arange(nrx)
    bxt_arr = np.zeros(nrx)
    bxt_arr[CHANNEL] = dxt      # fix the analysed channel's baseline to
                                # dxt regardless of nrx (see module docstring)
    array = ArrayGeometry(bat=bat, bxt=bxt_arr)

    prf, PRF_op = prf_from_fixed(2000.0, nrx)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, nrx, acq_time)

    cfg = sar.ExperimentConfig(
        name="coeff_sweep", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta,
        plots_dir=SAVE_DIR,
    )
    tracks = build_platform_tracks(cfg)
    return cfg, tracks, scene


def _call_coeffs(cfg, tracks, ptg):
    """Call GetCoeffNu for CHANNEL at the given target point, and combine
    into the (term_0, term_1, term_2) phase-polynomial coefficients."""
    kk = CHANNEL
    C0, C1, C2, Dt = GetCoeffNu(
        ptg, tracks.ptx, tracks.prx[kk], tracks.vtx, tracks.vrx[kk],
        tracks.ptx, tracks.vtx,
        cfg.prf, cfg.system.wl, cfg.ta,
        cfg.sq_tx, cfg.sq_rx[kk], cfg.theta_tx, cfg.theta_rx[kk],
    )
    # Matches the exp(...) phase term in ReconstructSignalNumeri:
    #     C0/wl + (-C1 + Dt)*f + C2*wl*f^2
    return C0, Dt - C1, C2


def fit_coeffs(alpha_deg: float, dxt: float, nrx: int):
    """
    Return (term_0, term_1, term_2, res_0, res_1, res_2):

      term_i -- phase-polynomial coefficients evaluated at the REAL
                top-of-ramp target.
      res_i  -- term_i(real target) - term_i(assumed scene center, h=0),
                i.e. the residual mismatch the reconstruction actually
                suffers, since GetCoeffNu is only ever evaluated at the
                assumed center in the real pipeline.
    """
    cfg, tracks, scene = _build_cfg(alpha_deg, dxt, nrx)

    ptg_real = scene.points[-1]     # top of the ramp: the real target
    ptg_ref = scene.ptg             # scene center, h=0: what the
                                    # reconstruction assumes

    term = np.array(_call_coeffs(cfg, tracks, ptg_real))
    term_ref = np.array(_call_coeffs(cfg, tracks, ptg_ref))
    res = term - term_ref

    return (*term, *res)


coeff_names = ["$C_0$ [m]", "$(\\Delta t - C_1)$ [s]", "$C_2$ [m/s$^2$]"]
coeff_tags  = ["term0", "term1", "term2"]

res_names = ["$\\mathrm{res}(C_0)$ [m]",
            "$\\mathrm{res}(\\Delta t - C_1)$ [s]",
            "$\\mathrm{res}(C_2)$ [m/s$^2$]"]
res_tags  = ["res0", "res1", "res2"]


def _save(fig, name, subdir):
    os.makedirs(subdir, exist_ok=True)
    fig.savefig(os.path.join(subdir, name + ".png"), dpi=150,
                bbox_inches="tight")
    if SAVE_VECTOR:
        fig.savefig(os.path.join(subdir, name + ".pdf"),
                    bbox_inches="tight")
    plt.close(fig)


def _plot_vs_alpha(values, names, tags, nrx, save_dir_2d, title_suffix=""):
    for c in range(3):
        fig, ax = plt.subplots(figsize=(8, 5))
        for j, b in enumerate(BXT_VALUES):
            ax.plot(ALPHA_VALUES, values[:, j, c], marker="o",
                    label=f"$b_{{xt}}$ = {b:.0f} m")
        ax.set_xlabel("Ramp elevation angle $\\alpha$ [deg]")
        ax.set_ylabel(names[c])
        ax.set_title(f"{tags[c]} vs $\\alpha$ ({title_suffix}Nrx={nrx}, CH{CHANNEL})")
        ax.grid()
        ax.legend(fontsize="small")
        fig.tight_layout()
        _save(fig, f"{tags[c]}_vs_alpha", save_dir_2d)


def _plot_vs_bxt(values, names, tags, nrx, save_dir_2d, title_suffix=""):
    for c in range(3):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, a in enumerate(ALPHA_VALUES):
            ax.plot(BXT_VALUES, values[i, :, c], marker="s",
                    label=f"$\\alpha$ = {a:.0f}$^\\circ$")
        ax.set_xlabel("Cross-track baseline $b_{xt}$ [m]")
        ax.set_ylabel(names[c])
        ax.set_title(f"{tags[c]} vs $b_{{xt}}$ ({title_suffix}Nrx={nrx}, CH{CHANNEL})")
        ax.grid()
        ax.legend(fontsize="small")
        fig.tight_layout()
        _save(fig, f"{tags[c]}_vs_bxt", save_dir_2d)


def _plot_surfaces(values, names, tags, nrx, save_dir_3d, title_suffix=""):
    A, B = np.meshgrid(ALPHA_VALUES, BXT_VALUES, indexing="ij")
    for c in range(3):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(A, B, values[:, :, c], cmap="viridis",
                               edgecolor="k", lw=0.3, alpha=0.9)
        ax.set_xlabel("$\\alpha$ [deg]", labelpad=8)
        ax.set_ylabel("$b_{xt}$ [m]", labelpad=8)
        ax.set_zlabel(names[c], labelpad=8)
        ax.set_title(f"{tags[c]}($\\alpha$, $b_{{xt}}$) | {title_suffix}Nrx={nrx}, CH{CHANNEL}")
        fig.colorbar(surf, shrink=0.6, pad=0.1)
        ax.view_init(elev=25, azim=-60)
        _save(fig, f"{tags[c]}_surface_3d", save_dir_3d)


# ------------------- run the full sweep -------------------
# coeffs[n, i, j, :] = (term_0, term_1, term_2, res_0, res_1, res_2)
coeffs = np.zeros((len(NRX_VALUES), len(ALPHA_VALUES), len(BXT_VALUES), 6))

for n, nrx in enumerate(NRX_VALUES):
    for i, a in enumerate(ALPHA_VALUES):
        for j, b in enumerate(BXT_VALUES):
            coeffs[n, i, j, :] = fit_coeffs(a, b, nrx)
            print(f"Nrx={nrx} | alpha={a:6.2f} deg | bxt={b:6.1f} m | "
                  f"term_0={coeffs[n,i,j,0]:+.4e}  "
                  f"term_1={coeffs[n,i,j,1]:+.4e}  "
                  f"term_2={coeffs[n,i,j,2]:+.4e}  |  "
                  f"res_0={coeffs[n,i,j,3]:+.4e}  "
                  f"res_1={coeffs[n,i,j,4]:+.4e}  "
                  f"res_2={coeffs[n,i,j,5]:+.4e}")

    save_dir_2d = os.path.join(SAVE_DIR, "plots_2d", f"Nrx{nrx}")
    save_dir_3d = os.path.join(SAVE_DIR, "plots_3d", f"Nrx{nrx}")

    term_vals = coeffs[n, :, :, 0:3]
    res_vals = coeffs[n, :, :, 3:6]

    _plot_vs_alpha(term_vals, coeff_names, coeff_tags, nrx, save_dir_2d)
    _plot_vs_bxt(term_vals, coeff_names, coeff_tags, nrx, save_dir_2d)
    _plot_surfaces(term_vals, coeff_names, coeff_tags, nrx, save_dir_3d)

    _plot_vs_alpha(res_vals, res_names, res_tags, nrx, save_dir_2d,
                   title_suffix="residual, ")
    _plot_vs_bxt(res_vals, res_names, res_tags, nrx, save_dir_2d,
                title_suffix="residual, ")
    _plot_surfaces(res_vals, res_names, res_tags, nrx, save_dir_3d,
                  title_suffix="residual, ")

    print(f"Nrx={nrx}: plots saved to {save_dir_2d} and {save_dir_3d}")

print(f"\nAll done. Root output folder: {SAVE_DIR}")