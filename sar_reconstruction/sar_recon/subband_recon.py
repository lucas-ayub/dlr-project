# -*- coding: utf-8 -*-
"""
Sub-band multichannel azimuth reconstruction with per-sub-band SATA.

The "new reconstruction" model: instead of correcting each channel once for the
whole band and reconstructing every sub-band from that same data, this produces
one SATA-corrected version of each channel PER output sub-band, each conditioned
on that sub-band's own frequency beam, and feeds the matching version into the
reconstruction of that sub-band.

The correct per-sub-band residual (this is the key point)
---------------------------------------------------------
Every output sub-band k occupies a known slice of the reconstructed Doppler band,
centred at

    f_k = (-Nrx/2 + k + 1/2) * PRF_op ,

and PRF_op wide (NOT Nrx*PRF_op wide -- "using only PRF instead of N*PRF"). A
Doppler frequency maps to a beam angle,  beta = asin(lambda f / 2 v_s), so the
sub-band corresponds to the angular WINDOW

    [ beta_k^lo , beta_k^hi ] ,
    beta_k^{lo,hi} = asin( lambda (f_k -/+ PRF_op/2) / (2 v_s) ).

The residual C0 for sub-band k must be computed over THAT window ("consider a
frequency beam to compute -> beta -> x"), not over the full antenna beam. The
package `GetCoeffNu` selects its aperture symmetrically, |inst_sq| <= sq+theta/2
(only exercised broadside), so it cannot restrict to a squinted sub-band window.
`getcoeff_beam` below is a faithful copy of `GetCoeffNu` whose ONLY change is the
aperture selection: it keeps the samples whose instantaneous squint lies in
[beta_lo, beta_hi]. With (beta_lo, beta_hi) = (-theta/2, +theta/2) it reproduces
`GetCoeffNu(sq=0)` exactly, so the per-sub-band residual GENERALISES the existing
whole-band residual.

The frame is consistent throughout: inst_sq = asin(d r_h/dt / v_s) is in the
platform-velocity (v_s) frame, and so are beta_k and the SATA kernel's own
fc = 2 v_s/lambda * sin(squint). One beta_k serves the window and the squint.

What is reused, what is new
---------------------------
Reused unchanged from sar_recon: GetInversionFilters (reconstruction matrix
inverse), sata_1d (the SATA kernel, which already takes squint + Nsb),
az_pixel_of_dx (azimuth-pixel mapping). GetCoeffNu is reused verbatim for the
reconstruction filter (still built at the assumed centre -- the coefficients do
NOT change) and is the template for getcoeff_beam.

New here: getcoeff_beam (windowed coefficients), residual_C0_subband,
build_delta_C0_subband_array, sata_channels_subband, reconstruct_subband.

Scope: C0 term only, as in sata_channels. reconstruct_subband(use_sata=False)
reduces EXACTLY to sar.reconstruct (bit-exact drop-in).

FINDING (measured with getcoeff_beam, see the accompanying document)
-------------------------------------------------------------------
The residual C0 is *angle-invariant*: for a given target it is identical across
all sub-bands (C0 is the constant range term, evaluated at closest approach).
Consequently the per-sub-band split is a no-op at the C0 stage -- the whole-band
SATA (sata_channels) already applies the correct C0. The residual that DOES vary
with the sub-band is C1 (the linear / registration / DPCA term, antisymmetric in
beta_k); C2 varies only weakly. The per-sub-band structure therefore becomes
necessary at C1, not C0. getcoeff_beam already returns (C0, C1, C2) per sub-band,
so the C1 extension reuses this machinery directly.
"""
from __future__ import annotations

import numpy as np

from .reconstruction import GetInversionFilters, GetCoeffNu  # noqa: F401
from .sata import sata_1d, az_pixel_of_dx


# ---------------------------------------------------------------------------
# 1) Windowed coefficient extraction (the sub-band "frequency beam")
# ---------------------------------------------------------------------------
def getcoeff_beam(ptg, ptx, prx, vtx, vrx, pax, vax, prf, wl, ta,
                  beta_lo, beta_hi, N_time=2, dN_time=2):
    """
    Faithful copy of sar_recon.reconstruction.GetCoeffNu whose ONLY change is
    the aperture selection: samples are kept when their instantaneous squint
    lies in the angular window [beta_lo, beta_hi] (rad), instead of the
    symmetric |inst_sq| <= sq + theta/2.

    With (beta_lo, beta_hi) = (-theta/2, +theta/2) this reproduces
    GetCoeffNu(sq=0, theta) exactly. Returns (C0, C1, C2, Dt), or None if the
    window is too small to fit the polynomials robustly.
    """
    rhT = np.sqrt(np.sum((ptx - ptg[np.newaxis, :]) ** 2, axis=1))
    inst_sqT = np.arcsin(np.clip(np.gradient(rhT, 1 / prf) / vtx, -1.0, 1.0))
    valid_idxT = np.where((inst_sqT >= beta_lo) & (inst_sqT <= beta_hi))[0]

    rhA = np.sqrt(np.sum((pax - ptg[np.newaxis, :]) ** 2, axis=1))
    # (rhA / inst_sqA kept for parity with GetCoeffNu; A == TX here)

    rhR = np.sqrt(np.sum((prx - ptg[np.newaxis, :]) ** 2, axis=1))
    inst_sqR = np.arcsin(np.clip(np.gradient(rhR, 1 / prf) / vrx, -1.0, 1.0))
    valid_idxR = np.where((inst_sqR >= beta_lo) & (inst_sqR <= beta_hi))[0]

    if valid_idxT.size < (N_time + 3) or valid_idxR.size < (dN_time + 3):
        return None

    taCommon = np.intersect1d(ta[valid_idxT], ta[valid_idxR])
    idx_com = np.nonzero(np.isin(ta, taCommon))[0]
    if idx_com.size < (N_time + 3):
        return None

    rhMS = 2 * rhT[idx_com]
    rhBS = (rhA + rhR)[idx_com]

    fi_ms = -1 / wl * np.diff(2 * rhT[idx_com]) * prf
    fi_bs = -1 / wl * np.diff(rhT[idx_com] + rhR[idx_com]) * prf
    if fi_ms.size == 0 or fi_bs.size == 0:
        return None
    f_max = np.min([abs(np.max(fi_ms)), abs(np.max(fi_bs))])
    f_min = -(np.min([abs(np.min(fi_ms)), abs(np.min(fi_bs))]))

    vld_ms = np.where((fi_ms < f_max) & (fi_ms > f_min))[0]
    vld_bs = np.where((fi_bs < f_max) & (fi_bs > f_min))[0]
    if vld_ms.size < (N_time + 2) or vld_bs.size < (dN_time + 2):
        return None
    rh_ms = rhMS[vld_ms]; rh_bs = rhBS[vld_bs]
    ta_ms = taCommon[vld_ms]; ta_bs = taCommon[vld_bs]

    idx_ms = int(np.argmin(rh_ms)); idx_bs = int(np.argmin(rh_bs))
    np_r = np.min([len(rh_ms[idx_ms:]), len(rh_bs[idx_bs:])])
    np_l = np.min([len(rh_ms[:idx_ms]), len(rh_bs[:idx_bs])])
    rh_ms = rh_ms[idx_ms - np_l:idx_ms + np_r]; ta_ms = ta_ms[idx_ms - np_l:idx_ms + np_r]
    rh_bs = rh_bs[idx_bs - np_l:idx_bs + np_r]; ta_bs = ta_bs[idx_bs - np_l:idx_bs + np_r]
    if rh_ms.size < (N_time + 2) or rh_bs.size < (dN_time + 2):
        return None

    idx_ms = int(np.where(rh_ms == min(rh_ms))[0][0])
    idx_bs = int(np.where(rh_bs == min(rh_bs))[0][0])
    tbc_ms = ta_ms[idx_ms]; tbc_bs = ta_bs[idx_bs]

    c_time = np.polyfit(ta_ms - tbc_ms, rh_ms, N_time)[::-1]
    dc_time = np.polyfit(ta_bs - tbc_bs, rh_bs - rh_ms, dN_time)[::-1]

    C0 = (dc_time[0] + c_time[1] ** 2 / 4 / c_time[2]
          - (c_time[1] + dc_time[1]) ** 2 / 4 / (c_time[2] + dc_time[2]))
    C1 = ((c_time[2] * dc_time[1] - c_time[1] * dc_time[2])
          / 2 / c_time[2] / (c_time[2] + dc_time[2]))
    C2 = dc_time[2] / 4 / c_time[2] / (c_time[2] + dc_time[2])
    Dt = tbc_bs - tbc_ms
    return C0, C1, C2, Dt


# ---------------------------------------------------------------------------
# 2) Frequency beam of a sub-band:  f_k -> beta_k and its angular window
# ---------------------------------------------------------------------------
def subband_frequency_beam(cfg, k: int):
    """
    Return (f_k, beta_k, beta_lo, beta_hi) for output sub-band k.

    f_k        centre Doppler [Hz]        = (-Nrx/2 + k + 1/2) * PRF_op
    beta_k     centre beam angle [rad]    = asin(wl f_k / (2 vs))
    beta_lo/hi window edges [rad]         = asin(wl (f_k -/+ PRF_op/2) / (2 vs))

    All angles are in the platform-velocity (vs) frame -- the same frame as
    GetCoeffNu's inst_sq and the SATA kernel's fc -- so beta_k serves both the
    getcoeff_beam window and the SATA squint.
    """
    Nrx = cfg.Nrx
    vs = cfg.system.vs
    wl = cfg.system.wl
    f_k = (-Nrx / 2.0 + k + 0.5) * cfg.PRF_op
    a = lambda f: float(np.arcsin(np.clip(wl * f / (2.0 * vs), -1.0, 1.0)))
    return f_k, a(f_k), a(f_k - cfg.PRF_op / 2.0), a(f_k + cfg.PRF_op / 2.0)


# ---------------------------------------------------------------------------
# 3) Per-sub-band residual C0 and its azimuth map
# ---------------------------------------------------------------------------
def residual_C0_subband(cfg, tracks, ptg_real: np.ndarray, channel: int,
                        k: int) -> float:
    """
    Residual C0 [m] for one scatterer, channel, and sub-band k, computed over
    the sub-band's angular window:

        delta_C0 = C0_beam(real target) - C0_beam(assumed centre) ,

    with C0_beam from getcoeff_beam restricted to [beta_lo, beta_hi] of sub-band
    k. Returns 0.0 if the window is too small to fit either target (no reliable
    correction -> leave the data unchanged for that sub-band).
    """
    kk = channel
    _f, _b, blo, bhi = subband_frequency_beam(cfg, k)
    common = (tracks.ptx, tracks.prx[kk], tracks.vtx, tracks.vrx[kk],
              tracks.ptx, tracks.vtx, cfg.prf, cfg.system.wl, cfg.ta, blo, bhi)
    real = getcoeff_beam(ptg_real, *common)
    ref = getcoeff_beam(cfg.scene.ptg, *common)
    if real is None or ref is None:
        return 0.0
    return float(real[0] - ref[0])


def build_delta_C0_subband_array(cfg, tracks, channel: int, k: int,
                                 naz: int | None = None,
                                 pad_zero_outside: bool = False) -> np.ndarray:
    """
    Per-sub-band delta_C0 map for `channel`, sub-band `k` -- the sub-band
    analogue of sar_recon.sata.build_delta_C0_array. Each extra scatterer
    (dx, dy, dh) contributes residual_C0_subband(...) at its azimuth pixel; the
    map is interpolated across azimuth. All zeros if the scene has no extra
    scatterers (-> SATA is a no-op -> baseline).
    """
    if naz is None:
        naz = cfg.Na_ch
    if not cfg.scene.extra_offsets:
        return np.zeros(naz)

    center = cfg.scene.ptg
    from collections import defaultdict
    by_pixel = defaultdict(list)
    for (dx, dy, dh) in cfg.scene.extra_offsets:
        ptg_real = center + np.array([dx, dy, dh], dtype=np.float64)
        by_pixel[az_pixel_of_dx(cfg, dx)].append(
            residual_C0_subband(cfg, tracks, ptg_real, channel, k))

    pix = sorted(by_pixel)
    xs = np.array(pix, dtype=float)
    ys = np.array([max(by_pixel[p], key=abs) for p in pix], dtype=float)

    grid = np.arange(naz)
    if xs.size == 1:
        if pad_zero_outside:
            arr = np.zeros(naz); arr[int(xs[0])] = ys[0]; return arr
        return np.full(naz, ys[0])
    left = 0.0 if pad_zero_outside else ys[0]
    right = 0.0 if pad_zero_outside else ys[-1]
    return np.interp(grid, xs, ys, left=left, right=right)


# ---------------------------------------------------------------------------
# 4) SATA every channel for one sub-band k
# ---------------------------------------------------------------------------
def sata_channels_subband(cfg, tracks, s_channel: np.ndarray, k: int,
                          remove: bool = True, sata_osf: int = 4,
                          verbose: bool = False) -> np.ndarray:
    """
    Copy of s_channel [Nrx, Na_ch] with every channel SATA-corrected for output
    sub-band k: the per-sub-band delta_C0 map (build_delta_C0_subband_array) is
    applied with the SATA frequency beam centred on beta_k (squint = beta_k) and
    Nsb = Nrx. Same kernel and sign convention as sar_recon.sata.sata_channels
    (remove=True removes the residual).
    """
    Nrx = cfg.Nrx
    _f, beta_k, _lo, _hi = subband_frequency_beam(cfg, k)
    out = np.asarray(s_channel, dtype=complex).copy()
    for kk in range(Nrx):
        dC0 = build_delta_C0_subband_array(cfg, tracks, kk, k, naz=cfg.Na_ch)
        out[kk, :] = sata_1d(
            out[kk, :], dC0, rref=cfg.scene.r0, prf=cfg.PRF_op,
            v=cfg.system.vs, wl=cfg.system.wl, r=cfg.scene.r0,
            squint=beta_k, Nsb=Nrx, inverse=remove,
            sata_osf=sata_osf, verbose=verbose,
        )
    return out


# ---------------------------------------------------------------------------
# 5) Orchestrator: SATA per (channel, sub-band) -> reconstruction
# ---------------------------------------------------------------------------
def reconstruct_subband(cfg, tracks, s_channel: np.ndarray,
                        use_sata: bool = True, remove: bool = True,
                        sata_osf: int = 4, zeroOutBw: bool = True,
                        verbose: bool = False) -> np.ndarray:
    """
    Sub-band reconstruction with optional per-sub-band SATA pre-conditioning.

    Mirrors sar_recon.reconstruct(): same inputs, same output (flattened [Na]).
    The reconstruction matrix/inverse is built exactly as in
    ReconstructSignalNumeri (GetCoeffNu at the assumed centre, unchanged). The
    only change: output sub-band k is summed from each channel's version
    SATA-corrected for sub-band k.

    use_sata=False (or a scene with no topography) reduces this EXACTLY to
    sar_recon.reconstruct.
    """
    Nrx = cfg.Nrx
    Na_ch = cfg.Na_ch
    prfCh = cfg.PRF_op
    wl = cfg.system.wl
    Nsb = Nrx
    prfFinal = prfCh * Nrx
    Na = Na_ch * Nrx
    sceneMid = cfg.scene.ptg.reshape([3, 1])
    Nr = 1

    base = np.asarray(s_channel, dtype=complex).reshape([Nrx, Na_ch])

    Nsh = int(Na_ch / 2)
    if Nrx % 2 == 0:
        Nsh = 0

    # spec[k][j] : Doppler spectrum of channel j to use for output sub-band k.
    spec = [[None] * Nrx for _ in range(Nsb)]
    if use_sata:
        for kk in range(Nsb):
            corr = sata_channels_subband(cfg, tracks, base, kk,
                                         remove=remove, sata_osf=sata_osf,
                                         verbose=verbose)
            for jj in range(Nrx):
                spec[kk][jj] = np.roll(np.fft.fft(corr[jj, :]), Nsh)
    else:
        fft_base = [np.roll(np.fft.fft(base[jj, :]), Nsh) for jj in range(Nrx)]
        for kk in range(Nsb):
            for jj in range(Nrx):
                spec[kk][jj] = fft_base[jj]

    fsub = -prfFinal / 2 + np.arange(Na_ch) * prfCh / Na_ch
    srec = np.zeros([Na, Nr], np.complex64)

    for mm in range(Nr):
        C0 = np.zeros(Nrx); C1 = np.zeros(Nrx); C2 = np.zeros(Nrx); Dt = np.zeros(Nrx)
        for kk in range(Nrx):
            C0[kk], C1[kk], C2[kk], Dt[kk] = GetCoeffNu(
                sceneMid[:, mm], tracks.ptx, tracks.prx[kk],
                tracks.vtx, tracks.vrx[kk], tracks.ptx, tracks.vtx,
                prfFinal, wl, cfg.ta,
                cfg.sq_tx, cfg.sq_rx[kk], cfg.theta_tx, cfg.theta_rx[kk])

        hf = np.zeros([Na_ch, Nsb, Nrx], np.complex64)
        for jj in range(Nsb):
            for ii in range(Nrx):
                hf[:, jj, ii] = np.exp(-2j * np.pi * (
                    C0[ii] / wl
                    + (-C1[ii] + Dt[ii]) * (fsub + jj * prfCh)
                    + C2[ii] * (fsub + jj * prfCh) ** 2 * wl))

        iHf = GetInversionFilters(hf)

        for kk in range(Nsb):          # output sub-band
            for jj in range(Nrx):      # channel
                srec[kk * Na_ch:(kk + 1) * Na_ch, mm] += (
                    spec[kk][jj] * iHf[kk * Na_ch:(kk + 1) * Na_ch, jj])

    if zeroOutBw:
        fa = -prfFinal / 2 + np.arange(Na) * prfFinal / Na
        abw_idx = np.concatenate(
            (np.where(fa < -cfg.abw / 2)[0], np.where(fa > cfg.abw / 2)[0]))
        srec[abw_idx, :] *= 0

    srec = np.fft.ifft(np.roll(srec, int(Na / 2), axis=0), axis=0)
    return srec.flatten()