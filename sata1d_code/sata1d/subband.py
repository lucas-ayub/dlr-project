"""
Per-sub-band SATA and reconstruction (C0 term).

Output sub-band k (k = 0..Nrx-1) is centred at f_k = (-Nrx/2 + k + 1/2) PRF_op.
For each k, every channel is SATA-corrected with
  - delta_C0 fitted only over the look angles of sub-band k,
  - a kernel whose bins carry the absolute Doppler in [f_k - PRF_op/2, f_k + PRF_op/2)
    (the channel's baseband frequency wrapped into that window), with positions on the
    broadside image grid, so the energy of sub-band k is looked up at its true position,
and sub-band k of the reconstruction uses only that corrected set.
"""
from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .geometry import PlatformTracks
from .reconstruction import channel_spectrum, reconstruct_from_spectra
from .sata import build_delta_C0_array, subaperture_sizing, triangular_window


def subband_frequency_beam(cfg: ExperimentConfig, k):
    """(f_k, beta_k, beta_lo, beta_hi) of output sub-band k."""
    vs, wl = cfg.system.vs, cfg.system.wl
    f_k = (-cfg.Nrx / 2.0 + k + 0.5) * cfg.PRF_op
    a = lambda f: float(np.arcsin(np.clip(wl * f / (2.0 * vs), -1.0, 1.0)))
    return f_k, a(f_k), a(f_k - cfg.PRF_op / 2.0), a(f_k + cfg.PRF_op / 2.0)


def getcoeff_beam(ptg, ptx, prx, vtx, vrx, prf, wl, ta, beta_lo, beta_hi, N_time=2, dN_time=2):
    """GetCoeffNu restricted to the samples whose instantaneous squint is in
    [beta_lo, beta_hi]. Returns (C0, C1, C2, Dt) or None if the window is too short."""
    rhT = np.sqrt(np.sum((ptx - ptg[None, :]) ** 2, axis=1))
    inst_sqT = np.arcsin(np.clip(np.gradient(rhT, 1 / prf) / vtx, -1.0, 1.0))
    valid_T = np.where((inst_sqT >= beta_lo) & (inst_sqT <= beta_hi))[0]
    rhR = np.sqrt(np.sum((prx - ptg[None, :]) ** 2, axis=1))
    inst_sqR = np.arcsin(np.clip(np.gradient(rhR, 1 / prf) / vrx, -1.0, 1.0))
    valid_R = np.where((inst_sqR >= beta_lo) & (inst_sqR <= beta_hi))[0]
    if valid_T.size < N_time + 3 or valid_R.size < dN_time + 3:
        return None
    taCommon = np.intersect1d(ta[valid_T], ta[valid_R])
    idx = np.nonzero(np.isin(ta, taCommon))[0]
    if idx.size < N_time + 3:
        return None
    rhMS, rhBS = 2 * rhT[idx], (rhT + rhR)[idx]
    fi_ms = -1 / wl * np.diff(rhMS) * prf
    fi_bs = -1 / wl * np.diff(rhBS) * prf
    if fi_ms.size == 0 or fi_bs.size == 0:
        return None
    f_max = np.min([abs(np.max(fi_ms)), abs(np.max(fi_bs))])
    f_min = -(np.min([abs(np.min(fi_ms)), abs(np.min(fi_bs))]))
    vld_ms = np.where((fi_ms < f_max) & (fi_ms > f_min))[0]
    vld_bs = np.where((fi_bs < f_max) & (fi_bs > f_min))[0]
    if vld_ms.size < N_time + 2 or vld_bs.size < dN_time + 2:
        return None
    rh_ms, ta_ms = rhMS[vld_ms], taCommon[vld_ms]
    rh_bs, ta_bs = rhBS[vld_bs], taCommon[vld_bs]
    i_ms, i_bs = int(np.argmin(rh_ms)), int(np.argmin(rh_bs))
    nr = int(np.min([len(rh_ms[i_ms:]), len(rh_bs[i_bs:])]))
    nl = int(np.min([len(rh_ms[:i_ms]), len(rh_bs[:i_bs])]))
    rh_ms, ta_ms = rh_ms[i_ms - nl:i_ms + nr], ta_ms[i_ms - nl:i_ms + nr]
    rh_bs, ta_bs = rh_bs[i_bs - nl:i_bs + nr], ta_bs[i_bs - nl:i_bs + nr]
    if rh_ms.size < N_time + 2 or rh_bs.size < dN_time + 2:
        return None
    tbc_ms = ta_ms[np.where(rh_ms == min(rh_ms))[0][0]]
    tbc_bs = ta_bs[np.where(rh_bs == min(rh_bs))[0][0]]
    c = np.polyfit(ta_ms - tbc_ms, rh_ms, N_time)[::-1]
    dc = np.polyfit(ta_bs - tbc_bs, rh_bs - rh_ms, dN_time)[::-1]
    C0 = dc[0] + c[1] ** 2 / 4 / c[2] - (c[1] + dc[1]) ** 2 / 4 / (c[2] + dc[2])
    C1 = (c[2] * dc[1] - c[1] * dc[2]) / 2 / c[2] / (c[2] + dc[2])
    C2 = dc[2] / 4 / c[2] / (c[2] + dc[2])
    return C0, C1, C2, tbc_bs - tbc_ms


def residual_C0_subband(cfg: ExperimentConfig, tracks: PlatformTracks, ptg_real, channel, k):
    """C0(scatterer) - C0(reference) over the look angles of sub-band k [m] (0 if not fittable)."""
    _f, _b, blo, bhi = subband_frequency_beam(cfg, k)
    args = (tracks.ptx, tracks.prx[channel], tracks.vtx, tracks.vrx[channel],
            cfg.prf, cfg.system.wl, cfg.ta, blo, bhi)
    real, ref = getcoeff_beam(ptg_real, *args), getcoeff_beam(cfg.scene.ptg, *args)
    if real is None or ref is None:
        return 0.0
    return float(real[0] - ref[0])


def build_delta_C0_subband_array(cfg, tracks, channel, k, mode="footprint"):
    """delta_C0 map of one channel for sub-band k: values fitted over the sub-band's
    look angles, alias images for a kernel centred on f_k."""
    f_k = subband_frequency_beam(cfg, k)[0]
    return build_delta_C0_array(cfg, tracks, channel, mode=mode, f_centre=f_k,
                                residual_fn=lambda p, ch: residual_C0_subband(cfg, tracks, p, ch, k))


def subband_axis(prf, Nzp, v, wl, r, f_k):
    """Absolute Doppler of each FFT bin for sub-band k (baseband frequency wrapped into
    [f_k - prf/2, f_k + prf/2)) and its cell offset on the broadside grid."""
    phi = np.fft.fftfreq(Nzp, d=1.0 / prf)
    f_lo = f_k - prf / 2.0
    fsub = f_lo + np.mod(phi - f_lo, prf)
    betasub = np.arcsin(np.clip(wl * fsub / (2.0 * v), -1.0, 1.0))
    return fsub, r * np.tan(betasub) / v * prf


def sata_1d_subband(data, delta_C0_array, rref, prf, v, wl, r, f_k, inverse=False, sata_osf=1):
    """sata_1d for sub-band k: bins labelled with their absolute Doppler in the
    sub-band window (subband_axis), positions on the broadside grid."""
    data = np.asarray(data, dtype=complex)
    dimx = len(data)
    Tsubeff, hop, Nzp = subaperture_sizing(rref, prf, v, wl, sata_osf)
    if Tsubeff <= 2:
        return data.copy()
    win = triangular_window(Tsubeff)
    fsub, azpos = subband_axis(prf, Nzp, v, wl, r, f_k)

    out = np.zeros(dimx, dtype=complex)
    wsum = np.zeros(dimx)
    for start in range(0, dimx, hop):
        seg = data[start:start + Tsubeff]
        L = len(seg)
        if L < 2:
            break
        buf = np.zeros(Nzp, dtype=complex)
        buf[:L] = seg * win[:L]
        spec = np.fft.fft(buf)
        posaux = np.clip(np.round(azpos + start + 0.5 * Tsubeff).astype(int), 0, dimx - 1)
        ph = -2.0 * np.pi / wl * delta_C0_array[posaux]
        ph[~np.isfinite(ph)] = 0.0
        spec *= np.exp(-1j * ph) if inverse else np.exp(1j * ph)
        out[start:start + L] += np.fft.ifft(spec)[:L]
        wsum[start:start + L] += win[:L]
    nz = wsum > 1e-12
    out[nz] /= wsum[nz]
    out[~nz] = data[~nz]
    return out


def sata_channels_subband(cfg, tracks, s_channel, k, remove=True, sata_osf=4, mode="footprint"):
    """Every channel SATA-corrected for output sub-band k."""
    f_k = subband_frequency_beam(cfg, k)[0]
    out = np.asarray(s_channel, dtype=complex).copy()
    for kk in range(cfg.Nrx):
        dC0 = build_delta_C0_subband_array(cfg, tracks, kk, k, mode=mode)
        out[kk] = sata_1d_subband(out[kk], dC0, rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs,
                                  wl=cfg.system.wl, r=cfg.scene.r0, f_k=f_k,
                                  inverse=remove, sata_osf=sata_osf)
    return out


def reconstruct_subband(cfg: ExperimentConfig, tracks: PlatformTracks, s_channel, use_sata=True,
                        remove=True, sata_osf=4, mode="footprint", zeroOutBw=True):
    """Reconstruction where sub-band k uses the channels corrected for sub-band k.
    use_sata=False gives the standard reconstruction."""
    base = np.asarray(s_channel, dtype=complex)
    spec = []
    for k in range(cfg.Nrx):
        corr = sata_channels_subband(cfg, tracks, base, k, remove, sata_osf, mode) if use_sata else base
        spec.append([channel_spectrum(cfg, corr[j]) for j in range(cfg.Nrx)])
    return reconstruct_from_spectra(cfg, tracks, spec, zeroOutBw)
