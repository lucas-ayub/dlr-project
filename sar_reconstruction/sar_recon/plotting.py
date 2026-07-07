# -*- coding: utf-8 -*-
"""
Plotting / diagnostics. Each function is self-contained and takes the config,
tracks and results it needs, so plotting can be disabled or replaced without
affecting the numerical pipeline.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from .config import ExperimentConfig
from .geometry import PlatformTracks
from .analysis import ReconResult

_text_kwargs = dict(
    fontsize="small",
    verticalalignment="top",
    bbox=dict(boxstyle="round", fc="w", ec="0.5"),
)


def set_font_size(size: float = 14) -> None:
    """
    Bump every plot font together.
    """
    plt.rcParams.update({"font.size": size})


def enable_latex_fonts(font_family: str = "serif", preamble: str = "") -> None:
    """
    Opt-in: render all plot text through a real LaTeX install.
    """
    missing = [tool for tool in ("latex", "dvipng", "gs") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "enable_latex_fonts(): missing required executable(s) on PATH: "
            f"{', '.join(missing)}. Install a LaTeX distribution + dvipng + "
            "ghostscript before calling this."
        )

    rc = {"text.usetex": True, "font.family": font_family}
    if preamble:
        rc["text.latex.preamble"] = preamble
    plt.rcParams.update(rc)


def _auto_dpi(data_len, min_dpi=100, max_dpi=300, ref_len=4096):
    return int(np.clip(min_dpi * data_len / ref_len, min_dpi, max_dpi))


def _subdir(cfg: ExperimentConfig, name: str) -> str:
    """
    Return cfg.plots_dir/<name>, creating it if needed.
    """
    path = os.path.join(cfg.plots_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def _savefig(fig, path_without_ext: str, dpi: int = 150, vector: bool = False):
    """
    Always save PNG. If vector=True, also save PDF.

    Example:
        _savefig(fig, "plots/combined/plot_combined_Nrx2", vector=True)

    Saves:
        plots/combined/plot_combined_Nrx2.png
        plots/combined/plot_combined_Nrx2.pdf
    """
    fig.savefig(path_without_ext + ".png", dpi=dpi, bbox_inches="tight")
    if vector:
        fig.savefig(path_without_ext + ".pdf", bbox_inches="tight")


def plot_combined(cfg: ExperimentConfig, res: ReconResult, vector: bool = False):
    """2x2 summary: amplitude, spectrum, zoomed IRF, spectral phase."""
    Nrx, prf, abw = cfg.Nrx, cfg.prf, cfg.abw
    dx = cfg.array.bat[1] - cfg.array.bat[0] if Nrx > 1 else 0.0
    dxt = cfg.array.bxt[1] - cfg.array.bxt[0] if Nrx > 1 else 0.0
    ta = cfg.ta

    dph = np.angle(
        np.fft.fft(res.srecNF) * np.conjugate(np.fft.fft(res.srefF)),
        deg=True,
    )
    dph[res.abw_idx] = 0

    dpi_all = _auto_dpi(cfg.Na)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=dpi_all)
    fig.suptitle(
        f"Numerical Reconstruction | Nrx={Nrx} | $\\mathrm{{PRF}}={prf:.1f}\\,\\mathrm{{Hz}}$ | "
        f"$\\mathrm{{B_{{a}}}}={abw:.1f}\\,\\mathrm{{Hz}}$ | "
        f"$\\Delta b_{{at}}={dx:.1f}\\,\\mathrm{{m}}$ | "
        f"$\\Delta b_{{xt}}={dxt:.1f}\\,\\mathrm{{m}}$"
    )

    axes[0, 0].plot(ta, abs(res.sref), label="ref")
    axes[0, 0].plot(ta, abs(res.srecN), label="rec")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid()
    axes[0, 0].legend(fontsize="small", loc="best")
    axes[0, 0].text(0.02, 0.95, f"Nrx={Nrx}", transform=axes[0, 0].transAxes, **_text_kwargs)

    axes[0, 1].plot(
        ta,
        20.0 * np.log10(abs(res.srefF) / np.max(abs(res.srefF))),
        label="ref",
    )
    axes[0, 1].plot(
        ta,
        20.0 * np.log10(abs(res.srecNF) / np.max(abs(res.srecNF))),
        label="rec",
    )
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("[dB]")
    axes[0, 1].set_ylim([-100, 0])
    axes[0, 1].grid()
    axes[0, 1].legend(fontsize="small", loc="best")
    axes[0, 1].text(0.02, 0.95, f"Nrx={Nrx}", transform=axes[0, 1].transAxes, **_text_kwargs)

    axes[1, 0].plot(
        res.taz * 1e3,
        20.0 * np.log10(abs(res.u_refFocC) / np.max(abs(res.u_refFocC))),
        label="ref",
    )
    axes[1, 0].plot(
        res.taz * 1e3,
        20.0 * np.log10(abs(res.u_interpFocCN) / np.max(abs(res.u_interpFocCN))),
        label="rec",
    )
    axes[1, 0].set_xlabel("Time [ms]")
    axes[1, 0].set_ylabel("[dB]")
    axes[1, 0].grid()
    axes[1, 0].legend(fontsize="small", loc="best")
    axes[1, 0].text(0.02, 0.95, f"Nrx={Nrx}", transform=axes[1, 0].transAxes, **_text_kwargs)

    axes[1, 1].plot(res.fa, dph)
    axes[1, 1].axvline(x=abw / 2, color="r", linestyle="-.")
    axes[1, 1].axvline(x=-abw / 2, color="r", linestyle="-.")
    axes[1, 1].set_xlabel("Doppler freq [Hz]")
    axes[1, 1].set_ylabel("[deg]")
    axes[1, 1].grid()
    axes[1, 1].text(0.02, 0.95, f"Nrx={Nrx}", transform=axes[1, 1].transAxes, **_text_kwargs)

    fig.tight_layout()

    _savefig(
        fig,
        os.path.join(_subdir(cfg, "combined"), f"plot_combined_Nrx{Nrx}"),
        dpi=dpi_all,
        vector=vector,
    )

    plt.close(fig)


def plot_polyfit_diagnostic(
    cfg: ExperimentConfig,
    tracks: PlatformTracks,
    vector: bool = False,
):
    """Per-channel bistatic path-difference fit residual."""
    ptg = cfg.scene.ptg
    prfFinal = cfg.PRF_op * cfg.Nrx
    wl = cfg.system.wl
    ta = cfg.ta

    for kk in range(cfg.Nrx):
        rhT_ = np.sqrt(np.sum((tracks.ptx - ptg[np.newaxis, :]) ** 2, axis=1))
        rhA_ = rhT_.copy()
        rhR_ = np.sqrt(np.sum((tracks.prx[kk] - ptg[np.newaxis, :]) ** 2, axis=1))

        inst_sqT_ = np.arcsin(np.gradient(rhT_, 1 / prfFinal) / tracks.vtx)
        inst_sqR_ = np.arcsin(np.gradient(rhR_, 1 / prfFinal) / tracks.vrx[kk])

        valid_T_ = np.where(abs(inst_sqT_) <= (cfg.sq_tx + cfg.theta_tx / 2))[0]
        valid_R_ = np.where(abs(inst_sqR_) <= (cfg.sq_rx[kk] + cfg.theta_rx[kk] / 2))[0]

        taCommon_ = np.intersect1d(ta[valid_T_], ta[valid_R_])
        idx_com_ = np.nonzero(np.isin(ta, taCommon_))[0]

        if len(idx_com_) == 0:
            print(f"  CH{kk}: no common samples, skipping diagnostic plot")
            continue

        rhMS_ = 2 * rhT_[idx_com_]
        rhBS_ = (rhA_ + rhR_)[idx_com_]

        fi_ms_ = -1 / wl * np.diff(rhMS_) * prfFinal
        fi_bs_ = -1 / wl * np.diff(rhT_[idx_com_] + rhR_[idx_com_]) * prfFinal

        f_max_ = np.min([abs(np.max(fi_ms_)), abs(np.max(fi_bs_))])
        f_min_ = -(np.min([abs(np.min(fi_ms_)), abs(np.min(fi_bs_))]))

        vld_ms_ = np.where((fi_ms_ < f_max_) & (fi_ms_ > f_min_))[0]
        vld_bs_ = np.where((fi_bs_ < f_max_) & (fi_bs_ > f_min_))[0]

        rh_ms_s = rhMS_[vld_ms_]
        ta_ms_s = taCommon_[vld_ms_]
        rh_bs_s = rhBS_[vld_bs_]
        ta_bs_s = taCommon_[vld_bs_]

        im_ = np.argmin(rh_ms_s)
        ib_ = np.argmin(rh_bs_s)

        np_r_ = np.min([len(rh_ms_s[im_:]), len(rh_bs_s[ib_:])])
        np_l_ = np.min([len(rh_ms_s[:im_]), len(rh_bs_s[:ib_])])

        rh_ms_s = rh_ms_s[im_ - np_l_:im_ + np_r_]
        ta_ms_s = ta_ms_s[im_ - np_l_:im_ + np_r_]
        rh_bs_s = rh_bs_s[ib_ - np_l_:ib_ + np_r_]
        ta_bs_s = ta_bs_s[ib_ - np_l_:ib_ + np_r_]

        tbc_bs_ = ta_bs_s[np.argmin(rh_bs_s)]
        dc_time_ = np.polyfit(ta_bs_s - tbc_bs_, rh_bs_s - rh_ms_s, 2)[::-1]

        ta_fit_ = ta_bs_s - tbc_bs_
        diff_real_ = rh_bs_s - rh_ms_s
        diff_fit_ = np.polyval(dc_time_[::-1], ta_fit_)
        residual_ = diff_real_ - diff_fit_

        rmse_um = np.sqrt(np.mean(residual_ ** 2)) * 1e6
        rms_phase = np.sqrt(np.mean((residual_ / wl * 360.0) ** 2))

        dpi_d = _auto_dpi(len(ta_fit_))
        fig_d, axes_d = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi_d)

        fig_d.suptitle(
            f"Poly fit diagnostic | Nrx={cfg.Nrx} | CH{kk} | "
            f"bat={cfg.array.bat[kk]:.1f} m",
            fontsize="medium",
        )

        axes_d[0].plot(ta_fit_, diff_real_ * 1e3, label="real rh_bs - rh_ms")
        axes_d[0].plot(ta_fit_, diff_fit_ * 1e3, label="poly fit (order 2)", linestyle="--")
        axes_d[0].set_xlabel("ta - tbc [s]")
        axes_d[0].set_ylabel("[mm]")
        axes_d[0].set_title("Bistatic path difference")
        axes_d[0].legend(fontsize="small")
        axes_d[0].grid()

        axes_d[1].plot(ta_fit_, residual_ ** 2 * 1e12)
        axes_d[1].set_xlabel("ta - tbc [s]")
        axes_d[1].set_ylabel("[$\\mu$m$^2$]")
        axes_d[1].set_title(
            f"Squared residual | RMSE={rmse_um:.3f} $\\mu$m | "
            f"RMS phase err={rms_phase:.3f} deg"
        )
        axes_d[1].grid()

        fig_d.tight_layout()

        _savefig(
            fig_d,
            os.path.join(_subdir(cfg, "polyfit"), f"plot_polyfit_Nrx{cfg.Nrx}_CH{kk}"),
            dpi=dpi_d,
            vector=vector,
        )

        plt.close(fig_d)


def plot_geometry_3d(
    cfg: ExperimentConfig,
    n_plot: int = 400,
    vector: bool = False,
):
    """
    Two-panel geometry figure.
    """
    sk = 1e-3
    idx_vis = np.linspace(0, cfg.Na - 1, n_plot, dtype=int)
    ta_vis = cfg.ta[idx_vis]

    vs, H = cfg.system.vs, cfg.scene.H
    points = cfg.scene.points
    x0, y0, h0 = points[0]

    fig = plt.figure(figsize=(16, 7))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    fig.suptitle(f"SAR Acquisition Geometry  —  Nrx={cfg.Nrx}", fontsize="large")

    tx_x = vs * ta_vis * sk
    ax3d.plot(
        tx_x,
        np.zeros(n_plot),
        H * np.ones(n_plot) * sk,
        color="royalblue",
        lw=2,
        label="TX",
    )
    ax3d.text(tx_x[-1], 0, H * sk, "  TX", color="royalblue", fontsize="small")

    rx_colors = plt.cm.tab10(np.linspace(0, 0.9, cfg.Nrx))

    for jj in range(cfg.Nrx):
        rx_x = (vs * ta_vis - cfg.array.bat[jj]) * sk
        rx_y = cfg.array.bxt[jj] * np.ones(n_plot) * sk
        rx_z = H * np.ones(n_plot) * sk

        ax3d.plot(
            rx_x,
            rx_y,
            rx_z,
            color=rx_colors[jj],
            lw=1.2,
            linestyle="--",
            label=(
                f"RX{jj + 1} ($b_{{at}}$={cfg.array.bat[jj]:.1f} m, "
                f"$b_{{xt}}$={cfg.array.bxt[jj]:.1f} m)"
            ),
        )
        ax3d.text(rx_x[-1], rx_y[-1], rx_z[-1], f"  RX{jj + 1}", color=rx_colors[jj], fontsize="small")

    ax3d.scatter(
        [x0 * sk],
        [y0 * sk],
        [h0 * sk],
        color="red",
        s=80,
        zorder=5,
        label="Target (center)",
    )
    ax3d.text(
        x0 * sk,
        y0 * sk,
        h0 * sk + 2 * sk,
        f"  Center\n  ({x0:.0f}, {y0:.0f}, {h0:.0f}) m",
        color="red",
        fontsize="small",
    )
    ax3d.plot(
        [x0 * sk, x0 * sk],
        [y0 * sk, y0 * sk],
        [h0 * sk, H * sk],
        color="red",
        lw=0.8,
        linestyle=":",
        alpha=0.6,
    )

    if len(points) > 1:
        ax3d.scatter(
            points[1:, 0] * sk,
            points[1:, 1] * sk,
            points[1:, 2] * sk,
            color="darkorange",
            s=50,
            zorder=5,
            label=f"Extra scatterers ({len(points) - 1})",
        )

    ax3d.set_xlabel("Azimuth [km]")
    ax3d.set_ylabel("Range [km]")
    ax3d.set_zlabel("Altitude [km]")
    ax3d.set_title("Full geometry (km scale)", fontsize="medium")
    ax3d.legend(fontsize="small", loc="upper left", ncol=1)
    ax3d.view_init(elev=25, azim=-60)

    ax2d.scatter([0], [0], color="royalblue", s=120, zorder=5, label="TX")
    ax2d.annotate(
        "TX",
        (0, 0),
        textcoords="offset points",
        xytext=(6, 6),
        color="royalblue",
        fontsize="small",
    )

    for jj in range(cfg.Nrx):
        bat_j = -cfg.array.bat[jj]
        bxt_j = cfg.array.bxt[jj]

        ax2d.scatter([bat_j], [bxt_j], color=rx_colors[jj], s=80, zorder=5)
        ax2d.annotate(
            f"RX{jj + 1}",
            (bat_j, bxt_j),
            textcoords="offset points",
            xytext=(6, 4),
            color=rx_colors[jj],
            fontsize="small",
        )

    all_bat = np.concatenate([[0], -cfg.array.bat])
    all_bxt = np.concatenate([[0], cfg.array.bxt])

    margin = max(np.ptp(all_bat), np.ptp(all_bxt), 1.0) * 0.3

    ax2d.set_xlim(all_bat.min() - margin, all_bat.max() + margin)
    ax2d.set_ylim(all_bxt.min() - margin, all_bxt.max() + margin)

    ax2d.set_xlabel("Azimuth / $b_{at}$ (Along-Track baseline) [m]")
    ax2d.set_ylabel("Range / $b_{xt}$ (Across-Track baseline) [m]")
    ax2d.set_title("Receiver array (metres, $t=0$)", fontsize="medium")
    ax2d.axhline(0, color="grey", lw=0.6, linestyle=":")
    ax2d.axvline(0, color="grey", lw=0.6, linestyle=":")
    ax2d.set_aspect("equal")
    ax2d.grid()

    fig.tight_layout()

    _savefig(
        fig,
        os.path.join(_subdir(cfg, "geometry_3d"), f"plot_geometry_3d_Nrx{cfg.Nrx}"),
        dpi=150,
        vector=vector,
    )

    plt.close(fig)


def plot_scene_points(cfg: ExperimentConfig, vector: bool = False):
    points = cfg.scene.points
    center = points[0]
    rel = points - center[np.newaxis, :]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    fig.suptitle(
        f"Scene scatterers relative to reconstruction center "
        f'({len(points)} point{"s" if len(points) != 1 else ""})',
        fontsize=10,
    )

    panel_specs = [
        (0, 1, "Azimuth [m]", "Range [m]", "Top-down (Azimuth $\\times$ Range)"),
        (0, 2, "Azimuth [m]", "Altitude [m]", "Side (Azimuth $\\times$ Altitude)"),
        (1, 2, "Range [m]", "Altitude [m]", "Front (Range $\\times$ Altitude)"),
    ]

    for ax, (xi, yi, xlabel, ylabel, title) in zip(axes, panel_specs):
        ax.scatter([0.0], [0.0], color="red", s=90, zorder=5, label="Center")

        if len(points) > 1:
            ax.scatter(
                rel[1:, xi],
                rel[1:, yi],
                color="darkorange",
                s=60,
                zorder=5,
                label="Extra scatterers",
            )

            for ii in range(1, len(points)):
                ax.annotate(
                    f"P{ii}",
                    (rel[ii, xi], rel[ii, yi]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid()
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()

    _savefig(
        fig,
        os.path.join(_subdir(cfg, "scene_points"), f"plot_scene_points_Nrx{cfg.Nrx}"),
        dpi=150,
        vector=vector,
    )

    plt.close(fig)


def plot_scene_points_3d(cfg: ExperimentConfig, vector: bool = False):
    """
    3D scatter of the scene scatterers, relative to the central point.
    """
    points = cfg.scene.points
    center = points[0]
    rel = points - center[np.newaxis, :]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        color="red",
        s=110,
        zorder=5,
        label="Center (reconstruction pt)",
    )
    ax.text(0.0, 0.0, 0.0, "  Center", color="red", fontsize="small")

    if len(points) > 1:
        ax.scatter(
            rel[1:, 0],
            rel[1:, 1],
            rel[1:, 2],
            color="darkorange",
            s=70,
            zorder=5,
            label=f"Extra scatterers ({len(points) - 1})",
        )

        for ii in range(1, len(points)):
            ax.text(
                rel[ii, 0],
                rel[ii, 1],
                rel[ii, 2],
                f"  P{ii}",
                color="darkorange",
                fontsize="small",
            )
            ax.plot(
                [rel[ii, 0], rel[ii, 0]],
                [rel[ii, 1], rel[ii, 1]],
                [0.0, rel[ii, 2]],
                color="darkorange",
                lw=0.7,
                linestyle=":",
                alpha=0.6,
            )

    span = np.max(np.abs(rel)) if len(points) > 1 else 1.0
    span = max(span, 1.0) * 1.2

    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(-span, span)

    ax.set_xlabel("Azimuth [m]", labelpad=10)
    ax.set_ylabel("Range [m]", labelpad=10)
    ax.set_zlabel("")  # native zlabel disabled, drawn manually below

    ax.set_title(
        f"Scene scatterers (3D, zoomed) | Nrx={cfg.Nrx} | "
        f'{len(points)} point{"s" if len(points) != 1 else ""}',
        fontsize="large",
    )

    ax.legend(fontsize="small", loc="upper left")
    ax.view_init(elev=25, azim=-60)

    fig.subplots_adjust(left=0.05, right=0.82, top=0.92, bottom=0.08)

    # Manual altitude label, drawn as 2D figure text so it can't be
    # clipped by the (buggy) 3D label bounding-box calculation.
    fig.text(
        0.78, 0.5, "Altitude [m]",
        rotation=90,
        va="center",
        ha="center",
        fontsize=plt.rcParams["axes.labelsize"],
    )

    _savefig(
        fig,
        os.path.join(_subdir(cfg, "scene_points_3d"), f"plot_scene_points_3d_Nrx{cfg.Nrx}"),
        dpi=150,
        vector=vector,
    )

    plt.close(fig)