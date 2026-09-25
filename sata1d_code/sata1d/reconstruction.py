"""Numerical multichannel azimuth reconstruction."""
from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .geometry import PlatformTracks


def GetCoeffNu(ptg, ptx, prx, vtx, vrx, pax, vax, prf, wl, ta, sq_tx, sq_rx, theta_tx, theta_rx,
               N_time=2, dN_time=2):
    """(C0, C1, C2, Dt) of the bistatic-minus-monostatic range history of point ptg.
    Parabolas are fitted over the common illuminated aperture; prf is the full PRF."""
    rhT = np.sqrt(np.sum((ptx - ptg[None, :]) ** 2, axis=1))
    inst_sqT = np.arcsin(np.gradient(rhT, 1 / prf) / vtx)
    valid_idxT = np.where(abs(inst_sqT) <= (sq_tx + theta_tx / 2))[0]
    rhA = np.sqrt(np.sum((pax - ptg[None, :]) ** 2, axis=1))
    rhR = np.sqrt(np.sum((prx - ptg[None, :]) ** 2, axis=1))
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
    rh_ms, ta_ms = rhMS[vld_ms], taCommon[vld_ms]
    rh_bs, ta_bs = rhBS[vld_bs], taCommon[vld_bs]

    # symmetric aperture around closest approach
    idx_ms, idx_bs = np.argmin(rh_ms), np.argmin(rh_bs)
    np_r = np.min([len(rh_ms[idx_ms:]), len(rh_bs[idx_bs:])])
    np_l = np.min([len(rh_ms[:idx_ms]), len(rh_bs[:idx_bs])])
    rh_ms, ta_ms = rh_ms[idx_ms - np_l:idx_ms + np_r], ta_ms[idx_ms - np_l:idx_ms + np_r]
    rh_bs, ta_bs = rh_bs[idx_bs - np_l:idx_bs + np_r], ta_bs[idx_bs - np_l:idx_bs + np_r]
    tbc_ms = ta_ms[np.where(rh_ms == min(rh_ms))[0][0]]
    tbc_bs = ta_bs[np.where(rh_bs == min(rh_bs))[0][0]]

    c = np.polyfit(ta_ms - tbc_ms, rh_ms, N_time)[::-1]
    dc = np.polyfit(ta_bs - tbc_bs, rh_bs - rh_ms, dN_time)[::-1]
    C0 = dc[0] + c[1] ** 2 / 4 / c[2] - (c[1] + dc[1]) ** 2 / 4 / (c[2] + dc[2])
    C1 = (c[2] * dc[1] - c[1] * dc[2]) / 2 / c[2] / (c[2] + dc[2])
    C2 = dc[2] / 4 / c[2] / (c[2] + dc[2])
    return C0, C1, C2, tbc_bs - tbc_ms


def GetInversionFilters(Hf):
    """Invert the reconstruction matrix Hf [Na_ch, Nsb, Nrx] per sub-band."""
    Na_ch, Nsb, Nrx = Hf.shape
    iHf = np.empty([Na_ch * Nsb, Nrx], np.complex64)
    Ti = np.repeat((np.identity(Nrx) * Nrx)[None, :, :], Na_ch, axis=0)
    for jj in range(Nsb):
        b = Ti[:, :, jj].reshape([Na_ch, Nrx])[..., None]
        iHf[jj * Na_ch:(jj + 1) * Na_ch, :] = np.linalg.solve(Hf, b)[..., 0]
    return iHf


def filter_matrix(cfg: ExperimentConfig, tracks: PlatformTracks, fsub):
    """Reconstruction filters hf [Na_ch, Nsb, Nrx] for the reference point."""
    Nrx, prfCh, wl = cfg.Nrx, cfg.PRF_op, cfg.system.wl
    C0, C1, C2, Dt = np.zeros(Nrx), np.zeros(Nrx), np.zeros(Nrx), np.zeros(Nrx)
    for kk in range(Nrx):
        C0[kk], C1[kk], C2[kk], Dt[kk] = GetCoeffNu(
            cfg.scene.ptg, tracks.ptx, tracks.prx[kk], tracks.vtx, tracks.vrx[kk],
            tracks.ptx, tracks.vtx, cfg.prf, wl, cfg.ta,
            cfg.sq_tx, cfg.sq_rx[kk], cfg.theta_tx, cfg.theta_rx[kk])
    hf = np.zeros([cfg.Na_ch, Nrx, Nrx], np.complex64)
    for jj in range(Nrx):
        f = fsub + jj * prfCh
        for ii in range(Nrx):
            hf[:, jj, ii] = np.exp(-2j * np.pi * (C0[ii] / wl + (-C1[ii] + Dt[ii]) * f + C2[ii] * f ** 2 * wl))
    return hf


def reconstruct_from_spectra(cfg: ExperimentConfig, tracks: PlatformTracks, spec, zeroOutBw=True):
    """spec[k][j]: spectrum of channel j used for output sub-band k (already rolled).
    Returns the reconstructed full-PRF azimuth line."""
    Nrx, Na_ch, prfCh = cfg.Nrx, cfg.Na_ch, cfg.PRF_op
    prfFinal, Na = prfCh * Nrx, Na_ch * Nrx
    fsub = -prfFinal / 2 + np.arange(Na_ch) * prfCh / Na_ch
    iHf = GetInversionFilters(filter_matrix(cfg, tracks, fsub))
    srec = np.zeros(Na, np.complex64)
    for kk in range(Nrx):
        for jj in range(Nrx):
            srec[kk * Na_ch:(kk + 1) * Na_ch] += spec[kk][jj] * iHf[kk * Na_ch:(kk + 1) * Na_ch, jj]
    if zeroOutBw:
        fa = -prfFinal / 2 + np.arange(Na) * prfFinal / Na
        srec[(fa < -cfg.abw / 2) | (fa > cfg.abw / 2)] = 0
    return np.fft.ifft(np.roll(srec, Na // 2))


def channel_spectrum(cfg: ExperimentConfig, line):
    """FFT of one channel line with the roll used by the reconstruction."""
    Nsh = 0 if cfg.Nrx % 2 == 0 else cfg.Na_ch // 2
    return np.roll(np.fft.fft(line), Nsh)


def reconstruct(cfg: ExperimentConfig, tracks: PlatformTracks, s_channel, zeroOutBw=True):
    """Standard reconstruction: every output sub-band uses the same channels."""
    specs = [channel_spectrum(cfg, s_channel[j]) for j in range(cfg.Nrx)]
    return reconstruct_from_spectra(cfg, tracks, [specs] * cfg.Nrx, zeroOutBw)
