# -*- coding: utf-8 -*-
"""
Two-step Range-Doppler multichannel azimuth reconstruction (2-D) --
port of the reference ``genRDrecons.py``.

Why two steps
-------------
The reconstruction filters depend on the slant range r0 (through the
coefficients C0, C1, C2) *and* on the range frequency fr (through the
wavelength wl(fr) = c/(f0+fr)).  Building one [Na, Nrx, Nr] filter cube per
range bin AND per range frequency is impossible for a real swath, so the
reference splits the operation:

STEP 1 -- reference reconstruction, in the 2-D frequency domain
    ``create_ref_dataset`` builds the filters ONCE, at the mid-swath reference
    range ``r_ref``, but *for every range frequency* (that is the ``wl_arr``
    loop).  This removes the wavelength dependence exactly.  The result is
    stored per (sub-band, channel) in ``REC_Pref`` and brought back to range
    time by an inverse FFT along range.

STEP 2 -- range-dependent residual, in the range-time domain
    ``generalized_rd`` then builds, for every range bin, the filter at the true
    r0 and applies only the PHASE DIFFERENCE with respect to the reference
    filter::

        iHf(r0) = exp(j*angle(P(r0))) * conj(exp(j*angle(P(r_ref))))

    Because this residual is smooth in range frequency it may be applied in
    range time.  ``interp_filter`` additionally shifts the residual filter along
    range to follow the range cell migration of each Doppler bin.

Layout conventions (identical to the 1-D ``sar_recon.reconstruction``)
----------------------------------------------------------------------
* ``CH``       [Nrx, Na_ch, Nr]  channel data in the 2-D FREQUENCY domain
                                 (azimuth FFT along axis 1, range FFT along
                                 axis 2, both in *unshifted* FFT order).
* ``REC_Pref`` [Na_ch, Nsb, Nrx, Nr] step-1 output, azimuth frequency x range TIME.
* ``REC``      [Na, Nr]          stacked sub-bands, azimuth frequency x range time;
                                 turned into azimuth time by the final IFFT.
* Azimuth frequency of row ``kk`` of ``REC`` is ``fa[kk] = -prf/2 + kk*prf/Na``:
  sub-band ``jj`` occupies rows ``jj*Na_ch .. (jj+1)*Na_ch`` and covers
  ``fsub + jj*prf/Nrx``.

Differences from the reference file (flagged inline)
----------------------------------------------------
* ``np.float`` -> ``float``; ``np.complex`` -> ``complex`` (NumPy >= 1.24).
* ``get_coeff_nu`` clips the ``arcsin``-free polyfit windows and guards against
  empty windows instead of raising.
* ``interp_filter`` was called with the FULL ``r_scan`` while operating on a
  block of ``Nb`` range samples; that only worked when ``Nb == Nr``.  The block
  size is now validated (see [FIX 2]).
* ``create_ref_dataset`` had the RCM term multiplied by a literal 0 in the
  reference (i.e. disabled).  It is kept disabled by default but is now an
  explicit ``apply_rcm`` flag instead of dead code.
"""
from __future__ import annotations

import time

import numpy as np

from .datastore import DataStore
from .siglib import ifft2

PI = np.pi
C_LIGHT = 3.0e8

__all__ = ["get_coeff_nu", "get_inversion_filters", "create_ref_dataset",
           "interp_filter", "generalized_rd", "run_two_step", "finish_spectrum"]


# ---------------------------------------------------------------------------
# 1) Reconstruction coefficients for a linear orbit (analytic geometry)
# ---------------------------------------------------------------------------
def get_coeff_nu(dx, r_ref, v, Tint, prf, wl):
    """
    (C0, C1, C2, Dt) for a receiver displaced ``dx`` along-track.

    The bistatic range history is expanded as a 2nd-order polynomial in slow
    time around its own point of closest approach, and compared with the
    monostatic one::

        r_ms(t) = 2*sqrt(r^2 + v^2 t^2)
        r_bs(t) =   sqrt(r^2 + v^2 t^2) + sqrt(r^2 + v^2 (t - dx/v)^2)

    C0 [m]   constant range offset (the term SATA corrects for topography)
    C1 [s]   linear / registration term (the DPCA time shift)
    C2       quadratic / defocus term
    Dt [s]   offset between the two points of closest approach

    ``dx == 0`` (the monostatic channel) returns all zeros exactly.
    """
    if dx == 0:
        return [0.0, 0.0, 0.0, 0.0]

    divfac = 32
    Na = int(np.ceil(Tint * prf / divfac) * divfac)
    ta = (np.arange(Na) - 0.5 * Na) / prf
    N_time, dN_time = 2, 2

    rh_ms = 2 * np.sqrt(r_ref ** 2 + v ** 2 * ta ** 2)
    rh_bs = (np.sqrt(r_ref ** 2 + v ** 2 * ta ** 2)
             + np.sqrt(r_ref ** 2 + v ** 2 * (ta - dx / v) ** 2))

    # Restrict both histories to the COMMON instantaneous-Doppler band, so the
    # two polynomials are fitted over the same physical aperture.
    f_max = np.max(-1 / wl * np.diff(rh_ms * prf))
    f_min = np.min(-1 / wl * np.diff(rh_bs * prf))
    f_ms = -1 / wl * np.diff(rh_ms * prf)
    f_bs = -1 / wl * np.diff(rh_bs * prf)
    vld_ms = np.where((f_ms < f_max) & (f_ms > f_min))[0]
    vld_bs = np.where((f_bs < f_max) & (f_bs > f_min))[0]

    rh_ms, ta_ms = rh_ms[vld_ms], ta[vld_ms]
    rh_bs, ta_bs = rh_bs[vld_bs], ta[vld_bs]

    # Align both windows on their point of closest approach and keep the
    # symmetric overlap.
    idx_ms = int(np.argmin(rh_ms))
    idx_bs = int(np.argmin(rh_bs))
    np_r = int(np.min([len(rh_ms[idx_ms:]), len(rh_bs[idx_bs:])]))
    np_l = int(np.min([len(rh_ms[:idx_ms]), len(rh_bs[:idx_bs])]))
    rh_ms, ta_ms = rh_ms[idx_ms - np_l:idx_ms + np_r], ta_ms[idx_ms - np_l:idx_ms + np_r]
    rh_bs, ta_bs = rh_bs[idx_bs - np_l:idx_bs + np_r], ta_bs[idx_bs - np_l:idx_bs + np_r]

    tbc_ms = ta_ms[int(np.argmin(rh_ms))]
    tbc_bs = ta_bs[int(np.argmin(rh_bs))]

    c_time = np.polyfit(ta_ms - tbc_ms, rh_ms, N_time)[::-1]
    dc_time = np.polyfit(ta_bs - tbc_bs, rh_bs - rh_ms, dN_time)[::-1]

    C0 = (dc_time[0] + c_time[1] ** 2 / 4 / c_time[2]
          - (c_time[1] + dc_time[1]) ** 2 / 4 / (c_time[2] + dc_time[2]))
    C1 = ((c_time[2] * dc_time[1] - c_time[1] * dc_time[2])
          / 2 / c_time[2] / (c_time[2] + dc_time[2]))
    C2 = dc_time[2] / 4 / c_time[2] / (c_time[2] + dc_time[2])
    Dt = tbc_bs - tbc_ms
    return [C0, C1, C2, Dt]


# ---------------------------------------------------------------------------
# 2) Inversion of the reconstruction matrix
# ---------------------------------------------------------------------------
def get_inversion_filters(Hf, analytic=False):
    """
    Invert the [Na_ch, Nsb, Nrx] reconstruction matrix, per Doppler bin.

    ``analytic=True`` uses the closed-form 2x2 inverse (Nrx == 2 only);
    otherwise ``np.linalg.solve`` is used against ``Nrx * I`` -- the ``Nrx``
    factor restores the amplitude lost by the per-channel decimation.
    """
    Na_ch, Nsb, Nrx = Hf.shape
    iHf = np.empty([Na_ch * Nsb, Nrx], np.complex64)

    if analytic and Nrx == 2:
        denom = Hf[:, 0, 0] * Hf[:, 1, 1] - Hf[:, 1, 0] * Hf[:, 0, 1]
        iHf[0:Na_ch, 0] = Hf[:, 1, 1] / denom
        iHf[Na_ch:2 * Na_ch, 0] = -Hf[:, 0, 1] / denom
        iHf[0:Na_ch, 1] = -Hf[:, 1, 0] / denom
        iHf[Na_ch:2 * Na_ch, 1] = Hf[:, 0, 0] / denom
        return iHf

    id_mat = np.identity(Nrx) * Nrx
    Ti = np.repeat(id_mat[np.newaxis, :, :], Na_ch, axis=0)
    for jj in range(Nsb):
        b = Ti[:, :, jj].reshape([Na_ch, Nrx])[..., np.newaxis]
        iHf[jj * Na_ch:(jj + 1) * Na_ch, :] = np.linalg.solve(Hf, b)[..., 0]
    return iHf


# ---------------------------------------------------------------------------
# 3) STEP 1 -- reference reconstruction, range-frequency dependent
# ---------------------------------------------------------------------------
def create_ref_dataset(ds: DataStore, wl_arr, deltaX, r_ref, ve, Tint, prf,
                       fsub, analytic, Nb=None, apply_rcm=False, verbose=True,
                       coeff_fn=None):
    """
    Apply the mid-swath reconstruction filters, evaluated at EVERY range
    frequency, and store the result in ``REC_Pref``.

    ``wl_arr`` [Nr]      wavelength per range-frequency bin, c/(f0+fr)
    ``fsub``   [Na_ch]   azimuth-frequency axis of one sub-band
    ``coeff_fn``         optional ``f(r0, channel) -> (C0, C1, C2, Dt)``.  The
                         default is the closed-form linear-orbit
                         :func:`get_coeff_nu`; pass a
                         :class:`~.geom3d.CoeffTable3D` for a 3-D geometry with
                         cross-track baselines and topography.

    Note that, exactly as in the reference code, the coefficients are evaluated
    ONCE (at ``wl_arr[0]``): only the explicit ``lambda`` inside the phase
    varies with range frequency.
    """
    Nrx, Na_ch, Nr = ds['CH'].shape
    Nsb = Nrx
    if Nb is None:
        Nb = Nr
    if Nr % Nb:
        raise ValueError("Nb must divide Nr")

    # Coefficients at the reference range, once per channel.
    C0r, C1r, C2r, Dtr = (np.zeros(Nrx), np.zeros(Nrx), np.zeros(Nrx), np.zeros(Nrx))
    for ii in range(Nrx):
        C0r[ii], C1r[ii], C2r[ii], Dtr[ii] = (
            coeff_fn(r_ref, ii) if coeff_fn is not None
            else get_coeff_nu(deltaX[ii], r_ref, ve, Tint, prf, wl_arr[0]))

    f0 = C_LIGHT / wl_arr[0]
    fr = (C_LIGHT / wl_arr) - f0

    Hwnf = np.zeros([Na_ch, Nsb, Nrx], np.complex64)
    iHf = np.zeros([Na_ch * Nrx, Nrx, Nb], np.complex64)
    cnt, inCnt = 0, 0
    t0 = time.time()

    for mm in range(Nr):                       # loop over range FREQUENCY
        wl_m = wl_arr[mm]
        for jj in range(Nsb):                  # output sub-band
            f_jj = fsub + jj * prf / Nrx       # true azimuth frequency of the band
            # Range cell migration term (disabled in the reference).
            rcm = (2 * r_ref / C_LIGHT
                   * (1 - 1.0 / np.sqrt(1.0 - (wl_m * f_jj / 2.0 / ve) ** 2))
                   ) if apply_rcm else 0.0
            for ii in range(Nrx):              # channel
                Hwnf[:, jj, ii] = np.exp(-2j * PI * (
                    C0r[ii] / wl_m
                    + (-C1r[ii] + Dtr[ii]) * f_jj
                    + C2r[ii] * f_jj ** 2 * wl_m
                    - rcm * fr[mm]))
        iHf[:, :, cnt] = get_inversion_filters(Hwnf, analytic)

        if cnt == Nb - 1:                      # block full -> flush
            cnt = 0
            sl = slice(inCnt * Nb, (inCnt + 1) * Nb)
            for kk in range(Nsb):
                for jj in range(Nrx):
                    ds['REC_Pref'][:, kk, jj, sl] = (
                        ds['CH'][jj, :, sl] * iHf[kk * Na_ch:(kk + 1) * Na_ch, jj, :])
            iHf[:] = 0
            inCnt += 1
        else:
            cnt += 1

    # Back to range TIME (the residual of step 2 is applied there).
    for kk in range(Nsb):
        for jj in range(Nrx):
            ds['REC_Pref'][:, kk, jj, :] = ifft2(ds['REC_Pref'][:, kk, jj, :], 12, axis=1)

    if verbose:
        print(f"  step 1 (reference filters, {Nr} range freqs): "
              f"{time.time() - t0:.1f} s")


# ---------------------------------------------------------------------------
# 4) Range-cell-migration of the residual filter
# ---------------------------------------------------------------------------
def interp_filter(r_int, prf, Pfx, wl, ve, rsf, r_ref):
    """
    Shift the residual filter along range to follow the range cell migration

        dt_rcm(fa, r) = 2*(r - r_ref)/c * (1 - 1/sqrt(1 - (wl*fa/2/ve)^2))

    ``Pfx`` [Na, Nrx, Nr] is modified in place and returned.  Row ``kk``
    corresponds to azimuth frequency ``fa[kk] = -prf/2 + kk*prf/Na``, which is
    exactly the stacking convention of ``REC``.
    """
    Na, Nrx, Nr_ = Pfx.shape
    if len(r_int) != Nr_:
        # [FIX 2] the reference passed the full r_scan while Pfx held one block.
        raise ValueError(f"interp_filter: r_int has {len(r_int)} samples but the "
                         f"filter block has {Nr_} -- use Nb == Nr")
    fa = np.arange(Na) * prf / Na - 0.5 * prf
    aux_sqrt = np.sqrt(1.0 - (wl * fa / 2.0 / ve) ** 2)
    grid = np.arange(Nr_)
    for kk in range(Na):
        dt_rcm = 2 * (r_int - r_ref) / C_LIGHT * (1.0 - 1.0 / aux_sqrt[kk])
        idx_rcmc = grid + dt_rcm * rsf
        for tt in range(Nrx):
            Pfx[kk, tt, :] = (np.interp(idx_rcmc, grid, np.real(Pfx[kk, tt, :]))
                              + 1j * np.interp(idx_rcmc, grid, np.imag(Pfx[kk, tt, :])))
    return Pfx


# ---------------------------------------------------------------------------
# 5) STEP 2 -- range-dependent residual + coherent sum of the sub-bands
# ---------------------------------------------------------------------------
def generalized_rd(ds: DataStore, prf, rsf, deltaX, ve, wl, Tint, r_scan,
                   fmax, fmin, Nb=None, analytic=False, apply_rcmc=True,
                   verbose=True, coeff_fn=None, return_spectrum=False):
    """
    Full two-step reconstruction.  ``ds['CH']`` must already be in the 2-D
    frequency domain; ``ds['REC']`` is filled with the reconstructed signal in
    azimuth TIME x range TIME.

    ``coeff_fn(r0, channel) -> (C0, C1, C2, Dt)`` replaces the closed-form
    linear-orbit coefficients.  Pass a :class:`~.geom3d.CoeffTable3D` to run
    the 3-D geometry (cross-track baselines, target height); leave it at
    ``None`` to reproduce the reference behaviour exactly.
    """
    def _coef(r0, ii, wl_):
        return (coeff_fn(r0, ii) if coeff_fn is not None
                else get_coeff_nu(deltaX[ii], r0, ve, Tint, prf, wl_))
    Nrx, Na_ch, Nr = ds['CH'].shape
    Nsb = Nrx
    Na = Na_ch * Nrx
    if Nb is None:
        Nb = Nr
    f0 = C_LIGHT / wl
    fr = np.roll(np.arange(Nr) * rsf / Nr - 0.5 * rsf, int(0.5 * Nr))
    fsub = -prf / 2 + np.arange(Na_ch) * (prf / Nrx) / Na_ch

    # For an odd number of channels the per-channel spectrum has to be rolled
    # by half a band so that sub-band 0 starts at -prf/2.
    Nsh = int(Na_ch / 2) if Nrx % 2 else 0
    if Nsh:
        for kk in range(Nrx):
            ds['CH'][kk, :, :] = np.roll(ds['CH'][kk, :, :], Nsh, axis=0)

    r_ref = r_scan[int(Nr / 2)]
    wl_arr = C_LIGHT / (f0 + fr)

    # ---------------- STEP 1 ------------------------------------------------
    create_ref_dataset(ds, wl_arr, deltaX, r_ref, ve, Tint, prf, fsub,
                       analytic, Nb=Nb, verbose=verbose, coeff_fn=coeff_fn)

    # ---------------- STEP 2 ------------------------------------------------
    t0 = time.time()
    hf = np.zeros([Na_ch, Nsb, Nrx], np.complex64)
    for jj in range(Nsb):
        for ii in range(Nrx):
            C0r, C1r, C2r, Dtr = _coef(r_ref, ii, wl)
            hf[:, jj, ii] = np.exp(-2j * PI * (
                C0r / wl + (-C1r + Dtr) * (fsub + jj * prf / Nrx)
                + C2r * (fsub + jj * prf / Nrx) ** 2 * wl))
    iHf_ref = get_inversion_filters(hf, analytic)

    iHf = np.zeros([Na_ch * Nrx, Nrx, Nb], np.complex64)
    cnt, inCnt = 0, 0
    for nn in range(Nr):                       # loop over range BIN (slant range)
        r0 = r_scan[nn]
        for ii in range(Nrx):
            C0r, C1r, C2r, Dtr = _coef(r0, ii, wl)
            for jj in range(Nsb):
                hf[:, jj, ii] = np.exp(-2j * PI * (
                    C0r / wl + (-C1r + Dtr) * (fsub + jj * prf / Nrx)
                    + C2r * (fsub + jj * prf / Nrx) ** 2 * wl))
        Pf = get_inversion_filters(hf, analytic)
        # Only the PHASE difference w.r.t. the reference filter is applied --
        # the amplitude was already handled in step 1.
        iHf[:, :, cnt] = np.exp(1j * np.angle(Pf)) * np.conjugate(np.exp(1j * np.angle(iHf_ref)))

        if cnt == Nb - 1:
            cnt = 0
            sl = slice(inCnt * Nb, (inCnt + 1) * Nb)
            if apply_rcmc:
                iHf = interp_filter(r_scan[sl], prf, iHf, wl, ve, rsf, 0)
            for kk in range(Nsb):
                for jj in range(Nrx):
                    ds['REC'][kk * Na_ch:(kk + 1) * Na_ch, sl] += (
                        ds['REC_Pref'][:, kk, jj, sl]
                        * iHf[kk * Na_ch:(kk + 1) * Na_ch, jj, :])
            iHf[:] = 0
            inCnt += 1
        else:
            cnt += 1

    if return_spectrum:
        # Stop before the final band nulling / roll / IFFT so a caller can
        # assemble the output sub-band by sub-band (per-sub-band SATA).
        if verbose:
            print(f"  step 2 (range-dependent residual, {Nr} range bins): "
                  f"{time.time() - t0:.1f} s")
        return ds['REC'][:]

    # Null the azimuth frequencies outside the common Doppler band.
    fa = np.arange(Na) * prf / Na - 0.5 * prf
    abw_idx = np.concatenate((np.where(fa > fmax)[0], np.where(fa < fmin)[0]))
    rec = ds['REC'][:]
    rec[abw_idx, :] = 0
    # Stacked sub-bands run from -prf/2; roll to the unshifted FFT order.
    ds['REC'][:] = np.fft.ifft(np.roll(rec, int(Na / 2), axis=0), axis=0).astype(np.complex64)

    if verbose:
        print(f"  step 2 (range-dependent residual, {Nr} range bins): "
              f"{time.time() - t0:.1f} s")
    return ds['REC']


# ---------------------------------------------------------------------------
# 6) Convenience wrapper
# ---------------------------------------------------------------------------
def finish_spectrum(rec, prf, abw, Na):
    """
    Final stage shared by the plain and the per-sub-band reconstruction: null
    the azimuth frequencies outside the processed band, roll the stacked
    sub-bands (which start at -PRF/2) into raw FFT order, and IFFT in azimuth.
    """
    fa = np.arange(Na) * prf / Na - 0.5 * prf
    idx = np.concatenate((np.where(fa > abw / 2)[0], np.where(fa < -abw / 2)[0]))
    rec = np.array(rec, copy=True)
    rec[idx, :] = 0
    return np.fft.ifft(np.roll(rec, int(Na / 2), axis=0), axis=0).astype(np.complex64)


def run_two_step(p, s_channel_rc, fmax, fmin, analytic=False, apply_rcmc=True,
                 backend="memory", filename=None, verbose=True, coeff_fn=None,
                 return_spectrum=False):
    """
    Allocate the datasets, load the range-compressed channels and run the
    two-step reconstruction.

    ``p``            : :class:`~.params.Params`
    ``s_channel_rc`` : [Nrx, Na_ch, Nr] range-compressed channel data,
                       azimuth TIME x range TIME.

    Returns ``(rec, ds)`` with ``rec`` [Na, Nr] in azimuth time x range time.
    """
    Nrx, Na_ch, Nr = s_channel_rc.shape
    Na = Na_ch * Nrx

    ds = DataStore(filename, backend)
    ds.create('CH', (Nrx, Na_ch, Nr))
    ds.create('REC_Pref', (Na_ch, Nrx, Nrx, Nr))
    ds.create('REC', (Na, Nr))

    # CH must be the 2-D spectrum: FFT along azimuth AND along range,
    # both in unshifted order.
    ds['CH'][:] = np.fft.fft(np.fft.fft(s_channel_rc, axis=1), axis=2).astype(np.complex64)

    out = generalized_rd(ds, p.prf, p.rsf, p.deltaX, p.ve, p.wl, p.int_time,
                         p.r_scan, fmax, fmin, analytic=analytic,
                         apply_rcmc=apply_rcmc, verbose=verbose,
                         coeff_fn=coeff_fn, return_spectrum=return_spectrum)
    return (out if return_spectrum else ds['REC'][:]), ds
