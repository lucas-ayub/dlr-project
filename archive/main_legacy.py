# -*- coding: utf-8 -*-
"""
Synthetic data version: replaces Orbit, rat, and SignalProcessingLibrary
with an analytical SAR geometry.

Geometry:
  Transmitter position:   ptx(t) = (vs*t,  0,   H)
  Receiver i position:    prx(t) = (vs*t - bat[i],  bxt[i],  H)
  Target:                 ptg    = (x0, y0, h0)
"""
import numpy as np
import os
os.system('cls' if os.name == 'nt' else 'clear')
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))


def getRawData1D(ptgs, ptx, prx, vtx, vrx, ta, sq_tx, sq_rx,
                 theta_tx, theta_rx, wl, prf):

    Na = len(ta)
    Np = len(ptgs[:, 0])

    inst_sq_tx = np.zeros(Na, np.float64)
    inst_sq_rx = np.zeros(Na, np.float64)
    wa_tx = np.zeros(Na)
    wa_rx = np.zeros(Na)
    datal = np.zeros(Na, np.complex128)

    for p_idx in range(Np):
        rh_ms = np.sqrt(np.sum((ptx - ptgs[p_idx, :][np.newaxis, :]) ** 2, axis=1))
        inst_sq_tx[0:Na - 1] = np.arcsin(np.diff(rh_ms) * prf / vtx[1:Na])
        inst_sq_tx[Na - 1] = 2 * inst_sq_tx[Na - 2] - inst_sq_tx[Na - 3]
        wa_tx[np.where(abs(inst_sq_tx) <= (sq_tx + theta_tx / 2))] = 1

        rh_bs = np.sqrt(np.sum((prx - ptgs[p_idx, :][np.newaxis, :]) ** 2, axis=1))
        inst_sq_rx[0:Na - 1] = np.arcsin(np.diff(rh_bs) * prf / vrx[1:Na])
        inst_sq_rx[Na - 1] = 2 * inst_sq_rx[Na - 2] - inst_sq_tx[Na - 3]
        wa_rx[np.where(abs(inst_sq_rx) <= (sq_rx + theta_rx / 2))] = 1

        aPattern = wa_tx * wa_rx
        rh = rh_ms + rh_bs
        datal = aPattern * np.exp(-2j * np.pi * rh / wl)
    return datal


def GetCoeffNu(ptg, ptx, prx, vtx, vrx, pax, vax,
               prf, wl, ta, sq_tx, sq_rx, theta_tx,
               theta_rx, N_time=2, dN_time=2):
    """Get coefficients for numerical azimuth reconstruction.
    PRF is the final PRF after reconstruction."""

    rhT = np.sqrt(np.sum((ptx - ptg[np.newaxis, :]) ** 2, axis=1))
    inst_sqT = np.arcsin(np.gradient(rhT, 1 / prf) / vtx)
    valid_idxT = np.where(abs(inst_sqT) <= (sq_tx + theta_tx / 2))[0]

    rhA = np.sqrt(np.sum((pax - ptg[np.newaxis, :]) ** 2, axis=1))
    inst_sqA = np.arcsin(np.gradient(rhA, 1 / prf) / vax)
    valid_idxA = np.where(abs(inst_sqA) <= (sq_tx + theta_tx / 2))[0]

    if valid_idxT.all() != valid_idxA.all():
        print('Transmitter and Receiver valid pixels do not match!')

    rhR = np.sqrt(np.sum((prx - ptg[np.newaxis, :]) ** 2, axis=1))
    inst_sqR = np.arcsin(np.gradient(rhR, 1 / prf) / vrx)
    valid_idxR = np.where(abs(inst_sqR) <= (sq_rx + theta_rx / 2))[0]

    taCommon = np.intersect1d(ta[valid_idxT], ta[valid_idxR])
    idx_com = np.nonzero(np.isin(ta, taCommon))[0]

    rhMS = 2 * rhT[idx_com]
    rhBS = (rhA + rhR)[idx_com]

    fi_ms = -1 / wl * np.diff(2 * rhT[idx_com]) * prf
    fi_bs = -1 / wl * np.diff(rhT[idx_com] + rhR[idx_com]) * prf
    f_max = np.min([abs(np.max(fi_ms)), abs(np.max(fi_bs))])
    f_min = -(np.min([abs(np.min(fi_ms)), abs(np.min(fi_bs))]))

    vld_ms = np.where((fi_ms < f_max) & (fi_ms > f_min))[0]
    vld_bs = np.where((fi_bs < f_max) & (fi_bs > f_min))[0]
    rh_ms = rhMS[vld_ms]
    rh_bs = rhBS[vld_bs]
    ta_ms = taCommon[vld_ms]
    ta_bs = taCommon[vld_bs]

    idx_ms = np.argmin(rh_ms)
    idx_bs = np.argmin(rh_bs)

    np_r = np.min([len(rh_ms[idx_ms:]), len(rh_bs[idx_bs:])])
    np_l = np.min([len(rh_ms[:idx_ms]), len(rh_bs[:idx_bs])])

    rh_ms = rh_ms[idx_ms - np_l:idx_ms + np_r]
    ta_ms = ta_ms[idx_ms - np_l:idx_ms + np_r]
    rh_bs = rh_bs[idx_bs - np_l:idx_bs + np_r]
    ta_bs = ta_bs[idx_bs - np_l:idx_bs + np_r]

    idx_ms = np.where(rh_ms == min(rh_ms))[0][0]
    idx_bs = np.where(rh_bs == min(rh_bs))[0][0]
    if idx_ms != idx_bs:
        print('Minimum indices are not same!')

    tbc_ms = ta_ms[idx_ms]
    tbc_bs = ta_bs[idx_bs]

    c_time  = np.polyfit(ta_ms - tbc_ms, rh_ms,        N_time)[::-1]
    dc_time = np.polyfit(ta_bs - tbc_bs, rh_bs - rh_ms, dN_time)[::-1]

    C0 = (dc_time[0]
          + c_time[1] ** 2 / 4 / c_time[2]
          - (c_time[1] + dc_time[1]) ** 2 / 4 / (c_time[2] + dc_time[2]))
    C1 = ((c_time[2] * dc_time[1] - c_time[1] * dc_time[2])
          / 2 / c_time[2] / (c_time[2] + dc_time[2]))
    C2 = dc_time[2] / 4 / c_time[2] / (c_time[2] + dc_time[2])
    Dt = tbc_bs - tbc_ms

    return C0, C1, C2, Dt


def GetInversionFilters(Hf):
    """Get inversion filter."""
    Na_ch = Hf.shape[0]
    Nsb   = Hf.shape[1]
    Nrx   = Hf.shape[2]
    iHf   = np.empty([Na_ch * Nsb, Nrx], np.complex64)

    id_mat = np.identity(Nrx) * Nrx
    Ti = np.repeat(id_mat[np.newaxis, :, :], Na_ch, axis=0)
    for jj in range(Nsb):
        iHf[jj * Na_ch:(jj + 1) * Na_ch, :] = np.linalg.solve(
            Hf, Ti[:, :, jj].reshape([Na_ch, Nrx]))
    return iHf


def ReconstructSignalNumeri(data_ch, prfCh, wl, sceneMid, ta,
                            ptx, prx, vtx, vrx, pax, vax,
                            sq_tx, sq_rx, theta_tx, theta_rx,
                            deltax, ve, abw,
                            zeroOutBw=False):
    """
    Numerical Azimuth Reconstruction

    data_ch  [Nrx, Na_ch, Nr]  : data matrix
    sceneMid [3, Nr]            : scene-centre coordinates per range bin
    ptx      [Na, 3]            : transmitter positions
    vtx      [Na]               : transmitter speed scalar
    prx      [Nrx, Na, 3]      : receiver positions
    vrx      [Nrx, Na]         : receiver speed scalar
    pax      [Na, 3]            : active sensor positions (= ptx here)
    vax      [Na]               : active sensor speed scalar
    """
    Nrx, Na_ch, Nr = np.shape(data_ch)
    Nsb     = Nrx
    prfFinal = prfCh * Nrx
    Na      = Na_ch * Nrx
    fsub    = -prfFinal / 2 + np.arange(Na_ch) * prfCh / Na_ch
    srec    = np.zeros([Na, Nr], np.complex64)
    Nsh     = int(Na_ch / 2)

    if Nrx % 2 == 0:
        Nsh = 0
    for kk in range(Nrx):
        data_ch[kk, :, :] = np.roll(np.fft.fft(data_ch[kk, :, :], axis=0), Nsh, axis=0)

    hf = np.zeros([Na_ch, Nsb, Nrx], np.complex64)
    C0, C1, C2, Dt = np.zeros(Nrx), np.zeros(Nrx), np.zeros(Nrx), np.zeros(Nrx)

    for mm in range(Nr):
        for kk in range(Nrx):
            C0[kk], C1[kk], C2[kk], Dt[kk] = \
                GetCoeffNu(sceneMid[:, mm], ptx, prx[kk, :, :], vtx, vrx[kk, :],
                           pax, vax, prfFinal, wl, ta,
                           sq_tx, sq_rx[kk], theta_tx, theta_rx[kk])
            print(C0[kk], -C1[kk] + Dt[kk], C2[kk])

        for jj in range(Nsb):
            for ii in range(Nrx):
                hf[:, jj, ii] = np.exp(-2j * np.pi * (
                    C0[ii] / wl
                    + (-C1[ii] + Dt[ii]) * (fsub + jj * prfCh)
                    + C2[ii] * (fsub + jj * prfCh) ** 2 * wl))

        iHf = GetInversionFilters(hf)

        for kk in range(Nsb):
            for jj in range(Nrx):
                srec[kk * Na_ch:(kk + 1) * Na_ch, mm] += (
                    data_ch[jj, :, mm] * iHf[kk * Na_ch:(kk + 1) * Na_ch, jj])

    if zeroOutBw:
        fa = -prfFinal / 2 + np.arange(Na) * prfFinal / Na
        abw_idx = np.concatenate(
            (np.where(fa < -abw / 2)[0], np.where(fa > abw / 2)[0]))
        srec[abw_idx, :] *= 0

    srec = np.fft.ifft(np.roll(srec, int(Na / 2), axis=0), axis=0)
    return srec


def zoom1Dpeak(s, N, zpf, Ndc=0):
    Na = len(s)
    ii = np.argmax(abs(s))
    sc = (np.roll(s, int(0.5 * Na - ii)))[int(0.5 * Na - N):int(0.5 * Na + N)]
    sc = np.roll(np.fft.fft(sc), int(-Ndc))
    s2 = np.zeros(N * 2 * zpf, dtype=np.complex128)
    s2[0:N] = sc[0:N]
    s2[N * 2 * zpf - N:] = sc[N:]
    s2 = np.fft.ifft(np.roll(s2, int(Ndc)))
    return s2


# =============================================================================
# Synthetic scene / orbit parameters
# =============================================================================

import matplotlib.pyplot as plt
plt.ion()

channel_numbers = [i for i in range(2, 10)]
for Nrx in channel_numbers:
    for case in ["diff"]:

        if case == "dpca":
            Nsb = Nrx
            wl = 0.25
            ve = 7408.5313923924796
            vs = 7688.53706432
            c0 = 299792458.0
            da = 24 * wl
            La = 2 * da
            dxt = 100.0

            rDelay = 0.0038659204080400003
            t0_raw = -0.7657200576115073
            r0 = c0 * rDelay / 2

            Tint = (ve / da) / 2. / ve**2 * wl * r0

            dx = 11.0  # spacing between adjacent receivers [m]

            PRF_op = 2 * vs / (Nrx * dx)
            prf    = PRF_op * Nrx
            abw    = 2 * ve / La

            bat    = dx * np.arange(Nrx)
            deltaX = bat
            deltaT = bat / (2 * vs)

            deltaXt = dxt * (np.arange(Nrx) - (Nrx - 1) / 2.0)
            bxt = deltaXt

            acq_time = Tint * 2
            c = 3e8
            theta_tx = wl / La
            theta_rx = wl / La * np.ones(Nrx)
            sq_tx = 0.
            sq_rx = np.zeros(Nrx, np.float64)

            PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots_dpca_prf')
            os.makedirs(PLOTS_DIR, exist_ok=True)

        else:
            Nsb = Nrx
            wl = 0.25
            ve = 7408.5313923924796
            vs = 7688.53706432
            c0 = 299792458.0
            da = 24 * wl
            La = 2 * da
            dxt = 200.0

            rDelay = 0.0038659204080400003
            t0_raw = -0.7657200576115073
            r0 = c0 * rDelay / 2

            Tint = (ve / da) / 2. / ve**2 * wl * r0

            alpha  = 0.15
            prf    = 1500
            PRF_op = prf / Nrx
            # abw    = prf / (1 + alpha)

            # d_dpca  = vs / prf
            # dx      = 2 * d_dpca * Nrx

            # bat_spacing = 50.0
            # bat    = bat_spacing * np.arange(Nrx)
            # deltaX = bat
            # deltaT = bat / (2 * vs)
            
            alpha      = 0.15
            prf        = 2000                  # ← change in the PRF
            PRF_op     = prf / Nrx
            abw        = 2 * ve / La           # = 1235 Hz

            bat_spacing = 100.0                # large baselines, works for Nrx=2..9
            bat        = bat_spacing * np.arange(Nrx)
            deltaX     = bat
            deltaT     = bat / (2 * vs)

            deltaXt = dxt * (np.arange(Nrx) - (Nrx - 1) / 2.0)
            bxt = deltaXt

            Dx = 2 * vs / PRF_op

            acq_time = Tint * 2
            c = 3e8
            theta_tx = wl / La
            theta_rx = wl / La * np.ones(Nrx)
            sq_tx = 0.
            sq_rx = np.zeros(Nrx, np.float64)

            PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
            os.makedirs(PLOTS_DIR, exist_ok=True)

        divfac = 1024
        Na     = int(np.ceil(acq_time * prf / Nrx / divfac) * Nrx * divfac)
        Na_ch  = int(Na / Nrx)
        ta     = (np.arange(Na) - Na * 0.5) / prf

        H  = 720e3
        x0 = 20.0
        h0 = 2.0
        y0 = np.sqrt(max(r0**2 - (H - h0)**2, 0.0))
        sceneMid = np.array([[x0, y0, h0]])

        ptxM_int = np.column_stack([vs * ta, np.zeros(Na), H * np.ones(Na)])
        vtxM_int = vs * np.ones(Na)

        ptxB_int = np.zeros([Nrx, Na, 3], np.float64)
        vtxB_int = np.zeros([Nrx, Na],    np.float64)

        for jj in range(Nrx):
            ptxB_int[jj, :, 0] = vs * ta - bat[jj]
            ptxB_int[jj, :, 1] = bxt[jj]
            ptxB_int[jj, :, 2] = H
            vtxB_int[jj, :]    = vs

        # =============================================================================
        # Signal generation & reconstruction
        # =============================================================================

        sref = getRawData1D(sceneMid, ptxM_int, ptxM_int,
                            vtxM_int, vtxM_int,
                            ta, sq_tx, sq_tx, theta_tx, theta_tx, wl, prf)

        fa      = np.roll((np.arange(Na) / Na - 0.5) * prf, int(Na / 2))
        abw_idx = np.concatenate((np.where(fa < -0.5 * abw)[0],
                                  np.where(fa >  0.5 * abw)[0]))

        s_channel = np.zeros([Nrx, Na_ch], dtype=np.complex128)
        for ii in range(Nrx):
            s_channel[ii, :] = getRawData1D(
                sceneMid,
                ptxM_int, ptxB_int[ii, :, :],
                vtxM_int, vtxB_int[ii, :],
                ta, sq_tx, sq_tx, theta_tx, theta_tx, wl, prf)[::Nrx]

        def _auto_dpi(data_len, min_dpi=100, max_dpi=300, ref_len=4096):
            return int(np.clip(min_dpi * data_len / ref_len, min_dpi, max_dpi))

        srefF  = np.roll(np.fft.ifft(np.fft.fft(sref) * np.conjugate(np.fft.fft(sref))),
                         int(Na / 2))
        N   = int(16 * prf / abw)
        zpf = 64
        taz = (np.arange(2 * N * zpf) - N * zpf) / prf / zpf
        u_refFocC = zoom1Dpeak(srefF, N, zpf)

        srecN = ReconstructSignalNumeri(
            s_channel.reshape([Nrx, Na_ch, 1]),
            PRF_op, wl,
            sceneMid.reshape([3, 1]),
            ta,
            ptxM_int, ptxB_int,
            vtxM_int, vtxB_int,
            ptxM_int, vtxM_int,
            sq_tx, sq_rx, theta_tx, theta_rx,
            deltaX, ve * np.ones(Na_ch), abw,
            zeroOutBw=True)

        srecN  = srecN.flatten()
        srecNF = np.roll(np.fft.ifft(np.fft.fft(srecN) * np.conjugate(np.fft.fft(sref))),
                         int(Na / 2))
        u_interpFocCN = zoom1Dpeak(srecNF, N, zpf)

        # =============================================================================
        # Plots
        # =============================================================================

        def save_individual(fig, filename, dpi):
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

        dph = np.angle(np.fft.fft(srecNF) * np.conjugate(np.fft.fft(srefF)), deg=True)
        dph[abw_idx] = 0

        _text_kwargs = dict(fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle="round", fc="w", ec="0.5"))

        # --- combined 2x2 figure -----------------------------------------------------
        dpi_all = _auto_dpi(Na)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=dpi_all)
        fig.suptitle('Numerical Reconstruction')

        axes[0, 0].plot(ta, abs(sref),  label='ref')
        axes[0, 0].plot(ta, abs(srecN), label='rec')
        axes[0, 0].set_xlabel('Time [s]'); axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(); axes[0, 0].legend(prop={'size': 7}, loc='best')
        axes[0, 0].text(0.02, 0.95, f'Nrx={Nrx}', transform=axes[0, 0].transAxes, **_text_kwargs)

        axes[0, 1].plot(ta, 20. * np.log10(abs(srefF)  / np.max(abs(srefF))),  label='ref')
        axes[0, 1].plot(ta, 20. * np.log10(abs(srecNF) / np.max(abs(srecNF))), label='rec')
        axes[0, 1].set_xlabel('Time [s]'); axes[0, 1].set_ylabel('[dB]')
        axes[0, 1].set_ylim([-100, 0]); axes[0, 1].grid()
        axes[0, 1].legend(prop={'size': 7}, loc='best')
        axes[0, 1].text(0.02, 0.95, f'Nrx={Nrx}', transform=axes[0, 1].transAxes, **_text_kwargs)

        axes[1, 0].plot(taz * 1e3, 20. * np.log10(abs(u_refFocC)     / np.max(abs(u_refFocC))),     label='ref')
        axes[1, 0].plot(taz * 1e3, 20. * np.log10(abs(u_interpFocCN) / np.max(abs(u_interpFocCN))), label='rec')
        axes[1, 0].set_xlabel('Time [ms]'); axes[1, 0].set_ylabel('[dB]')
        axes[1, 0].grid(); axes[1, 0].legend(prop={'size': 7}, loc='best')
        axes[1, 0].text(0.02, 0.95, f'Nrx={Nrx}', transform=axes[1, 0].transAxes, **_text_kwargs)

        axes[1, 1].plot(fa, dph)
        axes[1, 1].axvline(x= abw / 2, color='r', linestyle='-.')
        axes[1, 1].axvline(x=-abw / 2, color='r', linestyle='-.')
        axes[1, 1].set_xlabel('Doppler freq [Hz]'); axes[1, 1].set_ylabel('[deg]')
        axes[1, 1].grid()
        axes[1, 1].text(0.02, 0.95, f'Nrx={Nrx}', transform=axes[1, 1].transAxes, **_text_kwargs)

        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f'plot_combined_Nrx{Nrx}.png'),
                    dpi=dpi_all, bbox_inches='tight')
        plt.show()

        # =============================================================================
        # Polynomial fit diagnostic — one plot per channel
        # =============================================================================
        ptg_diag = sceneMid[0]
        prfFinal_diag = PRF_op * Nrx

        for kk in range(Nrx):
            rhT_ = np.sqrt(np.sum((ptxM_int - ptg_diag[np.newaxis, :]) ** 2, axis=1))
            rhA_ = np.sqrt(np.sum((ptxM_int - ptg_diag[np.newaxis, :]) ** 2, axis=1))
            rhR_ = np.sqrt(np.sum((ptxB_int[kk] - ptg_diag[np.newaxis, :]) ** 2, axis=1))

            inst_sqT_ = np.arcsin(np.gradient(rhT_, 1 / prfFinal_diag) / vtxM_int)
            inst_sqR_ = np.arcsin(np.gradient(rhR_, 1 / prfFinal_diag) / vtxB_int[kk])
            valid_T_  = np.where(abs(inst_sqT_) <= (sq_tx + theta_tx / 2))[0]
            valid_R_  = np.where(abs(inst_sqR_) <= (sq_rx[kk] + theta_rx[kk] / 2))[0]

            taCommon_ = np.intersect1d(ta[valid_T_], ta[valid_R_])
            idx_com_  = np.nonzero(np.isin(ta, taCommon_))[0]

            if len(idx_com_) == 0:
                print(f'  CH{kk}: no common samples, skipping diagnostic plot')
                continue

            rhMS_ = 2 * rhT_[idx_com_]
            rhBS_ = (rhA_ + rhR_)[idx_com_]

            fi_ms_ = -1/wl * np.diff(rhMS_) * prfFinal_diag
            fi_bs_ = -1/wl * np.diff(rhT_[idx_com_] + rhR_[idx_com_]) * prfFinal_diag
            f_max_ = np.min([abs(np.max(fi_ms_)), abs(np.max(fi_bs_))])
            f_min_ = -(np.min([abs(np.min(fi_ms_)), abs(np.min(fi_bs_))]))

            vld_ms_ = np.where((fi_ms_ < f_max_) & (fi_ms_ > f_min_))[0]
            vld_bs_ = np.where((fi_bs_ < f_max_) & (fi_bs_ > f_min_))[0]
            rh_ms_s = rhMS_[vld_ms_]; ta_ms_s = taCommon_[vld_ms_]
            rh_bs_s = rhBS_[vld_bs_]; ta_bs_s = taCommon_[vld_bs_]

            im_ = np.argmin(rh_ms_s); ib_ = np.argmin(rh_bs_s)
            np_r_ = np.min([len(rh_ms_s[im_:]), len(rh_bs_s[ib_:])])
            np_l_ = np.min([len(rh_ms_s[:im_]), len(rh_bs_s[:ib_])])
            rh_ms_s = rh_ms_s[im_-np_l_:im_+np_r_]
            ta_ms_s = ta_ms_s[im_-np_l_:im_+np_r_]
            rh_bs_s = rh_bs_s[ib_-np_l_:ib_+np_r_]
            ta_bs_s = ta_bs_s[ib_-np_l_:ib_+np_r_]

            tbc_ms_ = ta_ms_s[np.argmin(rh_ms_s)]
            tbc_bs_ = ta_bs_s[np.argmin(rh_bs_s)]

            dc_time_ = np.polyfit(ta_bs_s - tbc_bs_, rh_bs_s - rh_ms_s, 2)[::-1]

            ta_fit_   = ta_bs_s - tbc_bs_
            diff_real_ = rh_bs_s - rh_ms_s
            diff_fit_  = np.polyval(dc_time_[::-1], ta_fit_)
            residual_  = diff_real_ - diff_fit_

            max_res_um    = np.sqrt(np.mean(residual_**2)) * 1e6
            max_phase_err = np.sqrt(np.mean((residual_ / wl * 360.)**2))

            dpi_d = _auto_dpi(len(ta_fit_))
            fig_d, axes_d = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi_d)
            fig_d.suptitle(
                f'Poly fit diagnostic | Nrx={Nrx} | CH{kk} | bat={bat[kk]:.1f} m',
                fontsize=9)

            axes_d[0].plot(ta_fit_, diff_real_ * 1e3, label='real rh_bs - rh_ms')
            axes_d[0].plot(ta_fit_, diff_fit_  * 1e3, label='poly fit (order 2)',
                           linestyle='--')
            axes_d[0].set_xlabel('ta - tbc [s]')
            axes_d[0].set_ylabel('[mm]')
            axes_d[0].set_title('Bistatic path difference')
            axes_d[0].legend(fontsize=8)
            axes_d[0].grid()

            residual_sq_  = residual_ ** 2
            phase_err_sq_ = residual_sq_ / wl**2 * 360.**2

            axes_d[1].plot(ta_fit_, residual_sq_ * 1e12)
            axes_d[1].set_xlabel('ta - tbc [s]')
            axes_d[1].set_ylabel('[μm²]')
            axes_d[1].set_title(
                f'Squared residual | RMSE={max_res_um:.3f} μm'
                f' | RMS phase err={max_phase_err:.3f} deg')
            axes_d[1].grid()

            fig_d.tight_layout()
            fname_d = os.path.join(PLOTS_DIR, f'plot_polyfit_Nrx{Nrx}_CH{kk}.png')
            fig_d.savefig(fname_d, dpi=dpi_d, bbox_inches='tight')
            plt.show()

        # =============================================================================
        # 3D geometry plot
        # =============================================================================
        n_plot  = 400
        idx_vis = np.linspace(0, Na - 1, n_plot, dtype=int)
        ta_vis  = ta[idx_vis]
        sk = 1e-3

        fig3d = plt.figure(figsize=(10, 7))
        ax3d  = fig3d.add_subplot(111, projection='3d')

        tx_x = vs * ta_vis * sk
        tx_y = np.zeros(n_plot)
        tx_z = H * np.ones(n_plot) * sk
        ax3d.plot(tx_x, tx_y, tx_z, color='royalblue', lw=2, label='TX')
        ax3d.text(tx_x[-1], tx_y[-1], tx_z[-1], '  TX', color='royalblue', fontsize=8)

        rx_colors = plt.cm.tab10(np.linspace(0, 0.9, Nrx))
        for jj in range(Nrx):
            rx_x = (vs * ta_vis - bat[jj]) * sk
            rx_y = bxt[jj] * np.ones(n_plot) * sk
            rx_z = H * np.ones(n_plot) * sk
            lbl  = f'RX{jj+1} (bat={bat[jj]:.1f} m, bxt={bxt[jj]:.1f} m)'
            ax3d.plot(rx_x, rx_y, rx_z, color=rx_colors[jj], lw=1.2,
                      linestyle='--', label=lbl)
            ax3d.text(rx_x[-1], rx_y[-1], rx_z[-1], f'  RX{jj+1}',
                      color=rx_colors[jj], fontsize=7)

        ax3d.scatter([x0 * sk], [y0 * sk], [h0 * sk],
                     color='red', s=80, zorder=5, label='Target')
        ax3d.text(x0 * sk, y0 * sk, h0 * sk + 2 * sk,
                  f'  Target\n  ({x0:.0f}, {y0:.0f}, {h0:.0f}) m',
                  color='red', fontsize=8)
        ax3d.plot([x0 * sk, x0 * sk], [y0 * sk, y0 * sk], [h0 * sk, H * sk],
                  color='red', lw=0.8, linestyle=':', alpha=0.6)

        ax3d.set_xlabel('Along-track [km]')
        ax3d.set_ylabel('Cross-track [km]')
        ax3d.set_zlabel('Altitude [km]')
        ax3d.set_title(f'SAR Acquisition Geometry  —  Nrx={Nrx}', fontsize=11)
        ax3d.legend(fontsize=7, loc='upper left', ncol=2)
        ax3d.view_init(elev=25, azim=-60)

        fig3d.tight_layout()
        fig3d.savefig(os.path.join(PLOTS_DIR, f'plot_geometry_3d_Nrx{Nrx}.png'),
                      dpi=150, bbox_inches='tight')
        plt.show()

    print('end')
