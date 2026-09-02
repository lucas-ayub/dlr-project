# -*- coding: utf-8 -*-
r"""
Explicit per-sub-band SATA kernel (self-contained -- does NOT import sata_1d).

Motivation (reviewer's point)
-----------------------------
The whole-band SATA kernel (`sar_recon.sata.sata_1d`) maps each sub-aperture
FFT bin to a Doppler frequency, then to a beam angle beta, then to an azimuth
image position x_i, via

        f  ->  beta = asin(wl f / 2 v)  ->  x_i = r (tan beta - tan beta_c) ...

For the WHOLE band that mapping is built around broadside (squint = 0). To do
the reconstruction PER OUTPUT SUB-BAND, the frequency array that feeds
f -> beta -> x_i MUST be rebuilt for each sub-band k, centred on that sub-band's
own Doppler centroid f_k, NOT reused from the broadside band. The previous
implementation tried to achieve this by passing `squint = beta_k` into the old
`sata_1d`, which recentres the axis through its `fc / pfc / dfc` bookkeeping:

        fc  = 2 v / wl * sin(squint)
        pfc = round( mod(fc, prf) * Nzp / prf )      # <-- integer bin of the alias
        dfc = round( fc * Nzp / prf ) * prf / Nzp
        fsub = ... + dfc ; fsub = roll(fsub, Nzp/2 + pfc)

That `round(mod(fc, prf) * Nzp / prf)` is exact only when f_k is an integer
multiple of the per-channel PRF. For the sub-band centres

        f_k = (-Nrx/2 + k + 1/2) * PRF_op ,

that holds for ODD Nrx (f_k mod PRF_op = 0) but FAILS for EVEN Nrx, where every
sub-band centre lands exactly on f_k mod PRF_op = PRF_op/2 -- the Nyquist edge of
the per-channel spectrum -- so `pfc` rounds ambiguously by half a bin and the
axis is mis-registered. This is exactly the observed even/odd Nrx split
(odd: SATA-sub == SATA-band; even: a systematic gap).

This module rebuilds the frequency axis EXPLICITLY and continuously per
sub-band, with no `round(mod(...))` step, so it is correct for any f_k
(integer or half-integer multiple of PRF_op):

        freq_block = fftfreq(Nzp, d = 1/PRF_op)      # in [-PRF_op/2, +PRF_op/2)
        fsub_k     = f_k + freq_block                 # absolute Doppler of sub-band k
        beta       = asin( wl * fsub_k / (2 v) )
        x_i offset = r * ( tan(beta) - tan(beta_k) ) / v * PRF_op

`fftfreq` is aligned bin-for-bin with `np.fft.fft`, so no roll/relabel is
needed; the half-integer Nyquist case is handled continuously like any other.

Everything else (WOLA sub-aperture bookkeeping, the ph = -2 pi/wl * dC0
correction, the optional experimental C1/C2 terms) mirrors `sata_1d`.
"""
from __future__ import annotations

import numpy as np

# Residual-geometry helpers are UNCHANGED (they compute the dC0/dC1/dC2 maps
# from GetCoeffNu geometry; the reviewer's point is about the kernel, not these).
from .subband_recon import (build_delta_term_subband_array,
                            subband_frequency_beam, GetInversionFilters, GetCoeffNu)


# ---------------------------------------------------------------------------
# 1) The explicit per-sub-band SATA kernel
# ---------------------------------------------------------------------------
def sata_1d_subband(data, delta_C0_array, rref, prf, v, wl, r,
                    f_k, beta_k, inverse=False, sata_osf=1, verbose=False,
                    delta_C1_array=None, delta_C2_array=None, f_dc=0.0):
    """
    Apply the 1D SATA topography correction for ONE output sub-band k.

    Same WOLA structure as sata_1d, but the sub-aperture frequency axis is built
    EXPLICITLY around this sub-band's Doppler centroid f_k, so the
    f -> beta -> x_i mapping is the one that belongs to sub-band k.

    Parameters that differ from sata_1d:
      f_k     : Doppler centroid of sub-band k [Hz]  (= (-Nrx/2+k+1/2)*PRF_op)
      beta_k  : beam angle of that centroid [rad]     (= asin(wl f_k / 2 v))
      f_dc    : OPTIONAL per-channel Doppler-centroid deviation [Hz] to add to
                the sub-band centre (Nida thesis Eq. 3.38). Default 0.0 leaves
                behaviour identical to before. When non-zero, the whole
                f -> beta -> x_i axis (and the reference beam) is shifted by
                f_dc, i.e. the sub-band centre is repositioned at f_k + f_dc.
    There is no `squint`/`Nsb` here: the squint IS beta_k, and the sub-aperture
    is sized for the full per-channel line (the old Nsb=1 case), which is the
    correct sizing -- see the earlier Nsb-in-the-denominator bug note.
    """
    data = np.asarray(data, dtype=complex).copy()
    dimx = len(data)

    # -- sub-aperture geometry (identical sizing to whole-band SATA, Nsb=1) ----
    deltax = np.sqrt(wl * rref / 2.0)
    Tsubeff = int(np.round(deltax * prf / v * 0.5) * 2)
    if Tsubeff <= 2:
        if verbose:
            print("SATA(sub): sub-aperture length very small, no correction")
        return data

    Nzp = int(2 ** np.ceil(np.log2(Tsubeff)) * sata_osf)
    Tsub = int(Tsubeff * 0.5)                      # 50 % overlap hop
    Tovl = Tsub

    # triangular analysis window (COLA at 50 % overlap)
    rightweight = np.arange(Tovl) / (Tovl - 1)
    leftweight = rightweight[::-1]
    win = np.concatenate((rightweight, leftweight))

    # -- EXPLICIT per-sub-band frequency axis: f -> beta -> x_i ----------------
    # fftfreq is aligned bin-for-bin with np.fft.fft, and spans [-prf/2, prf/2)
    # continuously, so f_k being a half-integer multiple of prf (even Nrx) is
    # handled exactly like any other value -- no round(mod(...)) degeneracy.
    freq_block = np.fft.fftfreq(Nzp, d=1.0 / prf)          # [-prf/2, prf/2)
    # Optional Doppler-centroid recentring (Eq. 3.38): reposition this sub-band's
    # centre at f_k + f_dc. beta_k_eff is the reference beam of the shifted centre
    # so the block centre still maps to azimuth offset 0.
    f_centre = f_k + f_dc
    beta_k_eff = np.arcsin(np.clip(wl * f_centre / (2.0 * v), -1.0, 1.0))
    fsub = f_centre + freq_block                           # absolute Doppler, sub-band k
    betasub = np.arcsin(np.clip(wl * fsub / (2.0 * v), -1.0, 1.0))
    # azimuth-image-position offset [pixels] of a scatterer at beta, measured
    # from the sub-band's own centre beam beta_k_eff (so the block centre maps to 0).
    azpos = r * (np.tan(betasub) - np.tan(beta_k_eff)) / v * prf

    if verbose:
        print(f"SATA(sub): f_k={f_k:.2f} Hz, beta_k={np.degrees(beta_k):.3f} deg, "
              f"Tsubeff={Tsubeff}, Nzp={Nzp}, azpos in [{azpos.min():.1f},{azpos.max():.1f}] px")

    # -- WOLA loop -------------------------------------------------------------
    L = Tsubeff
    hop = Tsub
    out = np.zeros(dimx, dtype=complex)
    wsum = np.zeros(dimx, dtype=float)
    for start in range(0, dimx, hop):
        seg = data[start:start + L]
        Lseg = len(seg)
        if Lseg < 2:
            break
        buf = np.zeros(Nzp, dtype=complex)
        buf[:Lseg] = seg * win[:Lseg]
        spec = np.fft.fft(buf)

        center = start + 0.5 * L
        posaux = np.round(azpos + center).astype(int)
        posaux = np.clip(posaux, 0, dimx - 1)

        ph = -2.0 * np.pi / wl * delta_C0_array[posaux]
        if delta_C1_array is not None:
            ph = ph - 2.0 * np.pi * delta_C1_array[posaux] * fsub
        if delta_C2_array is not None:
            ph = ph - 2.0 * np.pi * delta_C2_array[posaux] * wl * fsub ** 2
        ph[~np.isfinite(ph)] = 0.0

        spec *= np.exp(-1j * ph) if inverse else np.exp(1j * ph)
        rec = np.fft.ifft(spec)[:Lseg]
        out[start:start + Lseg] += rec
        wsum[start:start + Lseg] += win[:Lseg]

    nz = wsum > 1e-12
    out[nz] /= wsum[nz]
    out[~nz] = data[~nz]
    return out


# ---------------------------------------------------------------------------
# 2) SATA every channel for one sub-band k, using the explicit kernel
# ---------------------------------------------------------------------------
def doppler_centroid_deviation(cfg, tracks, channel):
    """Per-channel Doppler-centroid deviation df_DC,i [Hz] from the Nida thesis
    Eq. (3.38):

        df_DC,i = (Dt_i v^2) / (wl sqrt(r_ref^2 + v^2 Dt_i^2))  -  c1(b_i) / wl,

    with Dt_i the bistatic-vs-monostatic beam-centre time offset and c1(b_i) the
    linear coefficient, both taken from GetCoeffNu for the scene centre. This is
    the instantaneous Doppler shift of channel i's acquisition relative to the
    monostatic reference."""
    v, wl, r0 = cfg.system.vs, cfg.system.wl, cfg.scene.r0
    _C0, C1, _C2, Dt = GetCoeffNu(
        cfg.scene.ptg, tracks.ptx, tracks.prx[channel], tracks.vtx,
        tracks.vrx[channel], tracks.ptx, tracks.vtx, cfg.PRF_op * cfg.Nrx, wl,
        cfg.ta, cfg.sq_tx, cfg.sq_rx[channel], cfg.theta_tx, cfg.theta_rx[channel])
    return (Dt * v ** 2) / (wl * np.sqrt(r0 ** 2 + v ** 2 * Dt ** 2)) - C1 / wl


def sata_channels_subband_explicit(cfg, tracks, s_channel, k,
                                   remove=True, sata_osf=4, verbose=False,
                                   correct_terms=("C0",), doppler_centroid=False):
    """Copy of s_channel [Nrx, Na_ch] with every channel SATA-corrected for
    output sub-band k, using the EXPLICIT per-sub-band kernel above.

    doppler_centroid : if True, reposition each channel's sub-band centre by its
    Doppler-centroid deviation df_DC,i (Eq. 3.38). Default False -> unchanged."""
    Nrx = cfg.Nrx
    f_k, beta_k, _lo, _hi = subband_frequency_beam(cfg, k)
    out = np.asarray(s_channel, dtype=complex).copy()
    for kk in range(Nrx):
        f_dc = doppler_centroid_deviation(cfg, tracks, kk) if doppler_centroid else 0.0
        terms = {t: build_delta_term_subband_array(cfg, tracks, kk, k, t, naz=cfg.Na_ch)
                 for t in correct_terms}
        out[kk, :] = sata_1d_subband(
            out[kk, :], terms.get("C0", np.zeros(cfg.Na_ch)),
            rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=cfg.system.wl,
            r=cfg.scene.r0, f_k=f_k, beta_k=beta_k, inverse=remove,
            sata_osf=sata_osf, verbose=verbose,
            delta_C1_array=terms.get("C1"), delta_C2_array=terms.get("C2"),
            f_dc=f_dc,
        )
    return out


# ---------------------------------------------------------------------------
# 3) Orchestrator: identical to reconstruct_subband but with the explicit kernel
# ---------------------------------------------------------------------------
def reconstruct_subband_explicit(cfg, tracks, s_channel, use_sata=True,
                                 remove=True, sata_osf=4, zeroOutBw=True,
                                 verbose=False, correct_terms=("C0",),
                                 doppler_centroid=False):
    """
    Sub-band reconstruction using the EXPLICIT per-sub-band SATA kernel.

    Byte-for-byte identical to subband_recon.reconstruct_subband EXCEPT that the
    SATA pre-conditioning of each channel calls sata_channels_subband_explicit
    (explicit frequency axis) instead of sata_channels_subband (old squint-based
    axis). use_sata=False reduces exactly to sar_recon.reconstruct.
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

    spec = [[None] * Nrx for _ in range(Nsb)]
    if use_sata:
        for kk in range(Nsb):
            corr = sata_channels_subband_explicit(cfg, tracks, base, kk,
                                                  remove=remove, sata_osf=sata_osf,
                                                  verbose=verbose,
                                                  correct_terms=correct_terms,
                                                  doppler_centroid=doppler_centroid)
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

        for kk in range(Nsb):
            for jj in range(Nrx):
                srec[kk * Na_ch:(kk + 1) * Na_ch, mm] += (
                    spec[kk][jj] * iHf[kk * Na_ch:(kk + 1) * Na_ch, jj])

    if zeroOutBw:
        fa = -prfFinal / 2 + np.arange(Na) * prfFinal / Na
        abw_idx = np.concatenate(
            (np.where(fa < -cfg.abw / 2)[0], np.where(fa > cfg.abw / 2)[0]))
        srec[abw_idx, :] *= 0

    srec = np.fft.ifft(np.roll(srec, int(Na / 2), axis=0), axis=0)
    return srec.flatten()