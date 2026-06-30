# -*- coding: utf-8 -*-
"""
Plotting / diagnostics. Each function is self-contained and takes the config,
tracks and results it needs, so plotting can be disabled or replaced without
affecting the numerical pipeline.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from .config import ExperimentConfig
from .geometry import PlatformTracks
from .analysis import ReconResult

_text_kwargs = dict(fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle="round", fc="w", ec="0.5"))


def _auto_dpi(data_len, min_dpi=100, max_dpi=300, ref_len=4096):
    return int(np.clip(min_dpi * data_len / ref_len, min_dpi, max_dpi))


def plot_combined(cfg: ExperimentConfig, res: ReconResult):
    """2x2 summary: amplitude, spectrum, zoomed IRF, spectral phase."""
    Nrx, prf, abw = cfg.Nrx, cfg.prf, cfg.abw
    dx = cfg.array.bat[1] - cfg.array.bat[0] if Nrx > 1 else 0.0
    dxt = cfg.array.bxt[1] - cfg.array.bxt[0] if Nrx > 1 else 0.0
    ta = cfg.ta

    dph = np.angle(np.fft.fft(res.srecNF) * np.conjugate(np.fft.fft(res.srefF)),
                   deg=True)
    dph[res.abw_idx] = 0

    dpi_all = _auto_dpi(cfg.Na)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=dpi_all)
    fig.suptitle(f'Numerical Reconstruction | Nrx={Nrx} | prf={prf:.1f} Hz | '
                 f'abw={abw:.1f} Hz | Δbat={dx:.1f} m | Δbxt={dxt:.1f} m')

    axes[0, 0].plot(ta, abs(res.sref), label='ref')
    axes[0, 0].plot(ta, abs(res.srecN), label='rec')
    axes[0, 0].set_xlabel('Time [s]'); axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(); axes[0, 0].legend(prop={'size': 7}, loc='best')
    axes[0, 0].text(0.02, 0.95, f'Nrx={Nrx}', transform=axes[0, 0].transAxes, **_text_kwargs)

    axes[0, 1].plot(ta, 20. * np.log10(abs(res.srefF) / np.max(abs(res.srefF))), label='ref')
    axes[0, 1].plot(ta, 20. * np.log10(abs(res.srecNF) / np.max(abs(res.srecNF))), label='rec')
    axes[0, 1].set_xlabel('Time [s]'); axes[0, 1].set_ylabel('[dB]')
    axes[0, 1].set_ylim([-100, 0]); axes[0, 1].grid()
    axes[0, 1].legend(prop={'size': 7}, loc='best')
    axes[0, 1].text(0.02, 0.95, f'Nrx={Nrx}', transform=axes[0, 1].transAxes, **_text_kwargs)

    axes[1, 0].plot(res.taz * 1e3, 20. * np.log10(abs(res.u_refFocC) / np.max(abs(res.u_refFocC))), label='ref')
    axes[1, 0].plot(res.taz * 1e3, 20. * np.log10(abs(res.u_interpFocCN) / np.max(abs(res.u_interpFocCN))), label='rec')
    axes[1, 0].set_xlabel('Time [ms]'); axes[1, 0].set_ylabel('[dB]')
    axes[1, 0].grid(); axes[1, 0].legend(prop={'size': 7}, loc='best')
    axes[1, 0].text(0.02, 0.95, f'Nrx={Nrx}', transform=axes[1, 0].transAxes, **_text_kwargs)

    axes[1, 1].plot(res.fa, dph)
    axes[1, 1].axvline(x=abw / 2, color='r', linestyle='-.')
    axes[1, 1].axvline(x=-abw / 2, color='r', linestyle='-.')
    axes[1, 1].set_xlabel('Doppler freq [Hz]'); axes[1, 1].set_ylabel('[deg]')
    axes[1, 1].grid()
    axes[1, 1].text(0.02, 0.95, f'Nrx={Nrx}', transform=axes[1, 1].transAxes, **_text_kwargs)

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.plots_dir, f'plot_combined_Nrx{Nrx}.png'),
                dpi=dpi_all, bbox_inches='tight')
    plt.close(fig)


def plot_polyfit_diagnostic(cfg: ExperimentConfig, tracks: PlatformTracks):
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
            print(f'  CH{kk}: no common samples, skipping diagnostic plot')
            continue

        rhMS_ = 2 * rhT_[idx_com_]
        rhBS_ = (rhA_ + rhR_)[idx_com_]

        fi_ms_ = -1 / wl * np.diff(rhMS_) * prfFinal
        fi_bs_ = -1 / wl * np.diff(rhT_[idx_com_] + rhR_[idx_com_]) * prfFinal
        f_max_ = np.min([abs(np.max(fi_ms_)), abs(np.max(fi_bs_))])
        f_min_ = -(np.min([abs(np.min(fi_ms_)), abs(np.min(fi_bs_))]))

        vld_ms_ = np.where((fi_ms_ < f_max_) & (fi_ms_ > f_min_))[0]
        vld_bs_ = np.where((fi_bs_ < f_max_) & (fi_bs_ > f_min_))[0]
        rh_ms_s = rhMS_[vld_ms_]; ta_ms_s = taCommon_[vld_ms_]
        rh_bs_s = rhBS_[vld_bs_]; ta_bs_s = taCommon_[vld_bs_]

        im_ = np.argmin(rh_ms_s); ib_ = np.argmin(rh_bs_s)
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
        rms_phase = np.sqrt(np.mean((residual_ / wl * 360.) ** 2))

        dpi_d = _auto_dpi(len(ta_fit_))
        fig_d, axes_d = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi_d)
        fig_d.suptitle(f'Poly fit diagnostic | Nrx={cfg.Nrx} | CH{kk} | '
                       f'bat={cfg.array.bat[kk]:.1f} m', fontsize=9)

        axes_d[0].plot(ta_fit_, diff_real_ * 1e3, label='real rh_bs - rh_ms')
        axes_d[0].plot(ta_fit_, diff_fit_ * 1e3, label='poly fit (order 2)', linestyle='--')
        axes_d[0].set_xlabel('ta - tbc [s]'); axes_d[0].set_ylabel('[mm]')
        axes_d[0].set_title('Bistatic path difference')
        axes_d[0].legend(fontsize=8); axes_d[0].grid()

        axes_d[1].plot(ta_fit_, residual_ ** 2 * 1e12)
        axes_d[1].set_xlabel('ta - tbc [s]'); axes_d[1].set_ylabel('[μm²]')
        axes_d[1].set_title(f'Squared residual | RMSE={rmse_um:.3f} μm | '
                            f'RMS phase err={rms_phase:.3f} deg')
        axes_d[1].grid()

        fig_d.tight_layout()
        fig_d.savefig(os.path.join(cfg.plots_dir, f'plot_polyfit_Nrx{cfg.Nrx}_CH{kk}.png'),
                      dpi=dpi_d, bbox_inches='tight')
        plt.close(fig_d)


def plot_geometry_3d(cfg: ExperimentConfig, n_plot: int = 400):
    """3D acquisition geometry (TX track, RX tracks, target)."""
    sk = 1e-3
    idx_vis = np.linspace(0, cfg.Na - 1, n_plot, dtype=int)
    ta_vis = cfg.ta[idx_vis]
    vs, H = cfg.system.vs, cfg.scene.H
    x0, y0, h0 = cfg.scene.ptg

    fig3d = plt.figure(figsize=(10, 7))
    ax3d = fig3d.add_subplot(111, projection='3d')

    tx_x = vs * ta_vis * sk
    ax3d.plot(tx_x, np.zeros(n_plot), H * np.ones(n_plot) * sk,
              color='royalblue', lw=2, label='TX')
    ax3d.text(tx_x[-1], 0, H * sk, '  TX', color='royalblue', fontsize=8)

    rx_colors = plt.cm.tab10(np.linspace(0, 0.9, cfg.Nrx))
    for jj in range(cfg.Nrx):
        rx_x = (vs * ta_vis - cfg.array.bat[jj]) * sk
        rx_y = cfg.array.bxt[jj] * np.ones(n_plot) * sk
        rx_z = H * np.ones(n_plot) * sk
        ax3d.plot(rx_x, rx_y, rx_z, color=rx_colors[jj], lw=1.2, linestyle='--',
                  label=f'RX{jj+1} (bat={cfg.array.bat[jj]:.1f} m, bxt={cfg.array.bxt[jj]:.1f} m)')
        ax3d.text(rx_x[-1], rx_y[-1], rx_z[-1], f'  RX{jj+1}', color=rx_colors[jj], fontsize=7)

    ax3d.scatter([x0 * sk], [y0 * sk], [h0 * sk], color='red', s=80, zorder=5, label='Target')
    ax3d.text(x0 * sk, y0 * sk, h0 * sk + 2 * sk,
              f'  Target\n  ({x0:.0f}, {y0:.0f}, {h0:.0f}) m', color='red', fontsize=8)
    ax3d.plot([x0 * sk, x0 * sk], [y0 * sk, y0 * sk], [h0 * sk, H * sk],
              color='red', lw=0.8, linestyle=':', alpha=0.6)

    ax3d.set_xlabel('Along-track [km]'); ax3d.set_ylabel('Cross-track [km]')
    ax3d.set_zlabel('Altitude [km]')
    ax3d.set_title(f'SAR Acquisition Geometry  —  Nrx={cfg.Nrx}', fontsize=11)
    ax3d.legend(fontsize=7, loc='upper left', ncol=2)
    ax3d.view_init(elev=25, azim=-60)

    fig3d.tight_layout()
    fig3d.savefig(os.path.join(cfg.plots_dir, f'plot_geometry_3d_Nrx{cfg.Nrx}.png'),
                  dpi=150, bbox_inches='tight')
    plt.close(fig3d)
