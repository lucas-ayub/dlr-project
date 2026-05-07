# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

from signal_model import getRawData1D
from geometry import build_geometry
from reconstruction import ReconstructSignalNumeri
from utils import zoom1Dpeak
from plotting import (
    plot_results,
    plot_delta_r_all_channels,
    plot_delta_phi_H_all_channels,
)


def run_case(
    case_name,
    case_params,
    scene_params,
    save_plots=True,
    show_plots=False,
):
    """
    Run one complete SAR reconstruction test case.

    Parameters
    ----------
    case_name : str
        Internal case name used for folder naming.
    case_params : dict
        Dictionary with:
            - "title"
            - "b_at"
            - "b_xt"
    scene_params : dict
        Dictionary with scene/system parameters.
    save_plots : bool, optional
        If True, save generated figures.
    show_plots : bool, optional
        If True, show figures with plt.show().
    """

    # ============================================================
    # Parameters from scene
    # ============================================================

    wl = scene_params["wl"]

    v = scene_params["v"]
    ve = v

    r0 = scene_params["r0"]
    H = scene_params["H"]
    h0 = scene_params.get("h0", 0.0)

    prf = scene_params["prf"]
    prf_margin = scene_params.get("prf_margin", 1.2)
    divfac = scene_params.get("divfac", 1024)
    antenna_length_factor = scene_params.get("antenna_length_factor", 24)

    # ============================================================
    # Case parameters
    # ============================================================

    case_title = case_params.get("title", case_name)

    b_at = np.asarray(case_params["b_at"], dtype=np.float64)
    b_xt = np.asarray(case_params["b_xt"], dtype=np.float64)

    if len(b_at) != len(b_xt):
        raise ValueError("b_at and b_xt must have the same length.")

    Nrx = len(b_at)

    # ============================================================
    # Derived parameters
    # ============================================================

    da = antenna_length_factor * wl
    Tint = (ve / da) / 2.0 / ve**2 * wl * r0

    abw = prf / prf_margin
    PRF_op = prf / Nrx

    acq_time = 2 * Tint

    Na = int(np.ceil(acq_time * prf / Nrx / divfac) * Nrx * divfac)
    Na_ch = int(Na / Nrx)

    ta = (np.arange(Na) - Na * 0.5) / prf

    sq_tx = scene_params.get("sq_tx", 0.0)

    La = 2 * da
    theta_tx = wl / La

    theta_rx = theta_tx * np.ones(Nrx)
    sq_rx = np.zeros(Nrx, dtype=np.float64)

    print("\n" + "=" * 80)
    print(f"Running case: {case_name}")
    print("=" * 80)
    print(f"title    = {case_title}")
    print(f"Nrx      = {Nrx}")
    print(f"wl       = {wl:.6e} m")
    print(f"v        = {v:.6f} m/s")
    print(f"r0       = {r0:.3f} m")
    print(f"H        = {H:.3f} m")
    print(f"h0       = {h0:.3f} m")
    print(f"b_at     = {b_at}")
    print(f"b_xt     = {b_xt}")
    print(f"Tint     = {Tint:.6f} s")
    print(f"acq_time = {acq_time:.6f} s")
    print(f"prf      = {prf:.3f} Hz")
    print(f"PRF_op   = {PRF_op:.3f} Hz")
    print(f"abw      = {abw:.3f} Hz")
    print(f"Na       = {Na}")
    print(f"Na_ch    = {Na_ch}")

    # ============================================================
    # Geometry
    # ============================================================

    sceneMid, ptx, prx, vtx, vrx = build_geometry(
        Na=Na,
        ta=ta,
        v=v,
        H=H,
        r0=r0,
        b_at=b_at,
        b_xt=b_xt,
        h0=h0,
    )

    pax = ptx.copy()
    vax = vtx.copy()

    # ============================================================
    # Reference signal
    # ============================================================

    sref = getRawData1D(
        sceneMid,
        ptx,
        ptx,
        vtx,
        vtx,
        ta,
        sq_tx,
        sq_tx,
        theta_tx,
        theta_tx,
        wl,
        prf,
    )

    # ============================================================
    # Multichannel signal
    # ============================================================

    s_channel = np.zeros((Nrx, Na_ch), dtype=np.complex128)

    for ii in range(Nrx):
        sig_i = getRawData1D(
            sceneMid,
            ptx,
            prx[ii, :, :],
            vtx,
            vrx[ii, :],
            ta,
            sq_tx,
            sq_rx[ii],
            theta_tx,
            theta_rx[ii],
            wl,
            prf,
        )

        # Same decimation convention as your current main
        s_channel[ii, :] = sig_i[::Nrx]

    # ============================================================
    # Numerical reconstruction
    # ============================================================

    srec, coeffs = ReconstructSignalNumeri(
        s_channel.reshape((Nrx, Na_ch, 1)),
        PRF_op,
        wl,
        sceneMid.reshape((3, 1)),
        ta,
        ptx,
        prx,
        vtx,
        vrx,
        pax,
        vax,
        sq_tx,
        sq_rx,
        theta_tx,
        theta_rx,
        ve * np.ones(Na_ch),
        abw,
        True,
    )

    srec = srec.flatten()

    # ============================================================
    # IRF / correlation
    # ============================================================

    fa = np.roll((np.arange(Na) / Na - 0.5) * prf, int(Na / 2))

    abw_idx = np.concatenate(
        (
            np.where(fa < -0.5 * abw)[0],
            np.where(fa > 0.5 * abw)[0],
        ),
        axis=0,
    )

    srefF = np.roll(
        np.fft.ifft(np.fft.fft(sref) * np.conjugate(np.fft.fft(sref))),
        int(Na / 2),
    )

    srecF = np.roll(
        np.fft.ifft(np.fft.fft(srec) * np.conjugate(np.fft.fft(sref))),
        int(Na / 2),
    )

    N = scene_params.get("zoom_N", 16)
    zpf = scene_params.get("zoom_zpf", 64)

    taz = (np.arange(2 * N * zpf) - N * zpf) / prf / zpf

    u_ref = zoom1Dpeak(srefF, N, zpf)
    u_rec = zoom1Dpeak(srecF, N, zpf)

    # ============================================================
    # Plots
    # ============================================================

    plot_results(
        ta=ta,
        taz=taz,
        sref=sref,
        srec=srec,
        srefF=srefF,
        srecF=srecF,
        u_ref=u_ref,
        u_rec=u_rec,
        fa=fa,
        abw=abw,
        abw_idx=abw_idx,
        case_title=case_title,
        b_at=b_at,
        b_xt=b_xt,
    )

    plot_delta_r_all_channels(
        ta=ta,
        ptx=ptx,
        prx=prx,
        sceneMid=sceneMid,
    )

    plot_delta_phi_H_all_channels(
        fa=fa,
        abw=abw,
        wl=wl,
        coeffs=coeffs,
    )

    # ============================================================
    # Save/show figures
    # ============================================================

    if save_plots:
        save_figures_by_type(case_name)

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    print("end")

    return {
        "case_name": case_name,
        "case_title": case_title,
        "scene_params": scene_params,
        "case_params": case_params,
        "b_at": b_at,
        "b_xt": b_xt,
        "sref": sref,
        "srec": srec,
        "srefF": srefF,
        "srecF": srecF,
        "u_ref": u_ref,
        "u_rec": u_rec,
        "coeffs": coeffs,
        "ta": ta,
        "taz": taz,
        "fa": fa,
        "abw": abw,
        "prf": prf,
        "PRF_op": PRF_op,
        "Na": Na,
        "Na_ch": Na_ch,
        "Nrx": Nrx,
        "sceneMid": sceneMid,
        "ptx": ptx,
        "prx": prx,
        "vtx": vtx,
        "vrx": vrx,
    }


def save_figures_by_type(case_name):
    """
    Save all open figures into case-specific folders grouped by plot type.
    """

    out_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "plots",
        case_name,
    )

    dirs = {
        "delta_r": os.path.join(out_root, "delta_r"),
        "delta_phi": os.path.join(out_root, "delta_phi"),
        "general": os.path.join(out_root, "general"),
    }

    for folder in dirs.values():
        os.makedirs(folder, exist_ok=True)

    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)

        title = ""
        if fig._suptitle is not None:
            title = fig._suptitle.get_text().lower()

        axes_titles = " ".join(
            ax.get_title().lower()
            for ax in fig.axes
            if ax.get_title()
        )

        all_titles = title + " " + axes_titles

        if (
            "delta r" in all_titles
            or "δr" in all_titles
            or "\\delta r" in all_titles
        ):
            folder = dirs["delta_r"]
            prefix = "delta_r"

        elif (
            "delta phi h" in all_titles
            or "phi_h" in all_titles
            or "\\delta \\phi_h" in all_titles
        ):
            folder = dirs["delta_phi"]
            prefix = "delta_phi_h"

        else:
            folder = dirs["general"]
            prefix = "general"

        fig.tight_layout()
        filename = os.path.join(folder, f"{prefix}_figure_{i + 1}.png")
        fig.savefig(filename, dpi=180)
        print(f"saved: {filename}")