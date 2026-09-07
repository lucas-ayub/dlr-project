# -*- coding: utf-8 -*-
"""
The two-step Range-Doppler reconstruction, SATA applied to the 3-D geometry,
and the shared machinery for the IRF playground scripts.

Merged from six former files so the package has fewer of them
(datastore.py + siglib.py + rangecomp.py + rd_recons2d.py + recon3d.py +
irf_common.py). Depends on ``geometry.py`` (parameters, tracks, coefficients,
raw-data generation) and ``sata.py`` (the SATA kernel itself, unchanged and
2-D -- only the geometry below is 3-D).

1. ``DataStore`` (formerly datastore.py): ``ds['CH'][...]`` in RAM, API
   compatible with an HDF5-backed store for a much larger problem size.
2. ``fft2`` / ``ifft2`` / ``next_pow2`` (formerly siglib.py): thin FFT shims.
3. ``range_matched_filter`` / ``range_compress`` (formerly rangecomp.py).
4. The two-step Range-Doppler reconstruction (formerly rd_recons2d.py, minus
   ``get_coeff_nu`` which moved to geometry.py next to its 3-D counterpart):

   * STEP 1 -- ``create_ref_dataset``, in the 2-D frequency domain. Build the
     filter ONCE, at the mid-swath reference range, but for EVERY range
     frequency. Removes the wavelength dependence exactly.
   * STEP 2 -- ``generalized_rd``, in the range-time domain. For every range
     bin, build the filter at the true r0 and apply only the phase
     DIFFERENCE with respect to the reference filter. ``interp_filter``
     additionally shifts it along range to follow the range-cell migration
     of each Doppler bin.
5. SATA applied per sub-band to the 3-D geometry (formerly recon3d.py):
   the topographic residual map (``build_delta_C0_map_3d``) and
   ``reconstruct_subband_2d``, which runs the whole two-step reconstruction
   once per output sub-band, each time conditioned on that sub-band via the
   SATA kernel of ``sata.py`` -- not a "3-D SATA", the correction is still
   the same 2-D kernel, just re-evaluated at the true (bistatic) slant range
   of the target via ``geometry.CoeffTable3D`` / ``get_coeff_nu_3d``.
6. Shared build/reconstruct/plot machinery for the IRF playground scripts
   (formerly irf_common.py): ``run_irf`` builds the standard geometry,
   places one target and reconstructs it; ``plot_irf_single`` /
   ``plot_irf_all`` render the azimuth impulse response (dB, full azimuth
   time axis, a Nrx/PRF/Ba/Delta b_at/bxt_max parameter box).
"""
from __future__ import annotations

import time

import numpy as np

from .geometry import (make_params3d, build_tracks_3d, CoeffTable3D,
                       generate_reference_3d, generate_channels_3d,
                       residual_C0_3d, get_coeff_nu, C0_LIGHT)
from .sata import sata_1d, legacy_sata_1d, subband_centre_frequency

PI = np.pi
C_LIGHT = 3.0e8  # kept exactly as in the original rd_recons2d.py -- create_ref_dataset
                 # / generalized_rd use this rounded value; range_bin_of below uses the
                 # precise C0_LIGHT from geometry.py instead, as recon3d.py originally did

__all__ = [
    "DataStore", "fft2", "ifft2", "next_pow2",
    "range_matched_filter", "range_compress",
    "get_inversion_filters", "create_ref_dataset", "interp_filter",
    "generalized_rd", "run_two_step", "finish_spectrum",
    "scatterer_range", "scatterer_range_bistatic", "range_bin_of",
    "az_pixel_of", "build_delta_C0_map_3d", "channels_subband_3d",
    "reconstruct_subband_2d",
    "run_irf", "plot_irf_single", "plot_irf_all",
]



# ============================================================================
# 1) DataStore -- formerly datastore.py
# ============================================================================
class DataStore:
    def __init__(self, filename: str | None = None, backend: str = "memory"):
        self.backend = backend
        self._arrays: dict[str, np.ndarray] = {}
        self._h5 = None
        if backend == "hdf5":
            import h5py  # imported lazily: optional dependency
            self._h5 = h5py.File(filename, "w", libver="latest")
        elif backend != "memory":
            raise ValueError("backend must be 'memory' or 'hdf5'")

    # -- creation ----------------------------------------------------------
    def create(self, name: str, shape, dtype=np.complex64) -> None:
        if self._h5 is not None:
            self._h5.create_dataset(name, shape=shape, dtype=dtype)
        else:
            self._arrays[name] = np.zeros(shape, dtype)

    # -- h5py-like access --------------------------------------------------
    def __getitem__(self, name: str):
        return self._h5[name] if self._h5 is not None else self._arrays[name]

    def __contains__(self, name: str) -> bool:
        return name in (self._h5 if self._h5 is not None else self._arrays)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ============================================================================
# 2) FFT shims -- formerly siglib.py
# ============================================================================
def fft2(x, nthreads: int = 1, axis: int = -1):
    """1-D FFT along ``axis``. ``nthreads`` is accepted for API parity only."""
    return np.fft.fft(x, axis=axis).astype(np.complex64)


def ifft2(x, nthreads: int = 1, axis: int = -1):
    """1-D inverse FFT along ``axis``. ``nthreads`` is ignored."""
    return np.fft.ifft(x, axis=axis).astype(np.complex64)


def next_pow2(n) -> int:
    """Smallest power of two >= n (``np.log10(x)/np.log10(2)`` in the original)."""
    return int(2 ** np.ceil(np.log2(n)))


# ============================================================================
# 3) Range compression -- formerly rangecomp.py
# ============================================================================
def range_matched_filter(fr: np.ndarray, cd: float, rbw: float) -> np.ndarray:
    """Range matched filter evaluated on the range-frequency axis ``fr``."""
    return np.exp(-1j * np.pi * fr * cd * (fr / rbw - 1.0))


def range_compress(data: np.ndarray, cd: float, rbw: float, rsf: float,
                   axis: int = -1) -> np.ndarray:
    """
    Compress ``data`` along ``axis`` (the range/fast-time axis).

    Works for any number of leading dimensions, so it accepts both the
    monostatic reference [Na, Nr] and the channel cube [Nrx, Na_ch, Nr].
    """
    Nr = data.shape[axis]
    # Unshifted FFT ordering, matching np.fft.fft.
    fr = np.roll(np.arange(Nr) * rsf / Nr - 0.5 * rsf, int(0.5 * Nr))
    shape = [1] * data.ndim
    shape[axis] = Nr
    H = range_matched_filter(fr, cd, rbw).reshape(shape)
    return np.fft.ifft(np.fft.fft(data, axis=axis) * H, axis=axis).astype(np.complex64)


# ============================================================================
# 4) Two-step Range-Doppler reconstruction -- formerly rd_recons2d.py
#    (get_coeff_nu moved to geometry.py, next to get_coeff_nu_3d)
# ============================================================================
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


# ============================================================================
# 5) SATA applied to the 3-D geometry -- formerly recon3d.py
# ============================================================================
# ---------------------------------------------------------------------------
# 1) Where a scatterer lands
# ---------------------------------------------------------------------------
def scatterer_range(p, ptg) -> float:
    """Monostatic slant range at closest approach, ``sqrt(y^2 + (H-h)^2)``."""
    return float(np.sqrt(ptg[1] ** 2 + (p.H - ptg[2]) ** 2))


def scatterer_range_bistatic(p, tracks, ptg, channel: int) -> float:
    """
    HALF the minimum two-way path of one TX/RX pair,

        R_i = min_t  ( |p_tx(t) - p| + |p_rx_i(t) - p| ) / 2 ,               (2)

    which is where the range-compressed energy of that channel actually lands.

    This matters as soon as the cross-track baseline is large: a receiver
    displaced by b_xt sees the target at a slant range shorter by roughly
    ``b_xt sin(theta_inc)``, so its half-path moves by ``b_xt sin(theta)/2``.
    For b_xt = 450 m and theta = 20 deg that is 77 m -- about three range bins
    of the default grid.  Using the monostatic range for every channel would
    apply the SATA correction to bins that hold no energy for the outer
    channels, which is exactly how a large-baseline case silently fails.
    """
    ptx, prx, _vtx, _vrx = tracks
    a = np.sqrt(np.sum((ptx - ptg[None, :]) ** 2, axis=1))
    b = np.sqrt(np.sum((prx[channel] - ptg[None, :]) ** 2, axis=1))
    return float(0.5 * np.min(a + b))


def range_bin_of(p, r: float) -> int:
    """Range bin of a slant range: the grid is ``r_scan[0] + n*c/(2*rsf)``."""
    return int(round((r - p.r_scan[0]) * 2.0 * p.rsf / C0_LIGHT))


def az_pixel_of(p, x_az: float) -> float:
    """
    Azimuth pixel of an along-track position, on the per-channel grid.

    The channel line is sampled at ``PRF_op``, so the along-track sample
    spacing is ``vs / PRF_op`` and the scene centre sits at ``Na_ch/2``.
    """
    return p.Na_ch / 2.0 + (x_az - p.x0) / (p.vs / p.PRF_op)


# ---------------------------------------------------------------------------
# 2) The residual map
# ---------------------------------------------------------------------------
def build_delta_C0_map_3d(p, tracks, channel: int, range_halfwidth: int = 3,
                          pad_zero_outside: bool = False, verbose: bool = False):
    """
    ``(dmap, active_bins)`` with ``dmap`` of shape ``[Nr, Na_ch]``.

    For every scatterer, Eq. (1) is evaluated for this channel and deposited at
    its (range bin, azimuth pixel).  Within one range bin the values are
    interpolated across azimuth, so every azimuth pixel gets a correction;
    ``pad_zero_outside=False`` holds the endpoints outside the scatterer span
    (the dominant target governs the line, right for isolated targets).
    The range bin is the BISTATIC one of Eq. (2) -- per channel -- because that
    is where the channel's energy sits.  ``range_halfwidth`` copies each bin's
    profile to its neighbours, covering the residual range cell migration
    (about ``(v T_int/2)^2/(2 r0)`` = 45 m here, i.e. ~2 bins).
    """
    Nr, Na_ch = p.Nr, p.Na_ch
    dmap = np.zeros([Nr, Na_ch])
    pts = p.points
    if len(pts) == 0:
        return dmap, []

    by_bin: dict[int, list] = {}
    for ptg in pts:
        r = scatterer_range_bistatic(p, tracks, ptg, channel)
        n = range_bin_of(p, r)
        if not (0 <= n < Nr):
            continue
        d = residual_C0_3d(p, tracks, ptg, channel)
        by_bin.setdefault(n, []).append((az_pixel_of(p, ptg[0]), d))

    grid = np.arange(Na_ch)
    active = set()
    for n, entries in by_bin.items():
        entries.sort(key=lambda e: e[0])
        xs = np.array([e[0] for e in entries])
        ys = np.array([e[1] for e in entries])
        if xs.size == 1:
            prof = (np.zeros(Na_ch) if pad_zero_outside
                    else np.full(Na_ch, ys[0]))
            if pad_zero_outside:
                prof[int(np.clip(xs[0], 0, Na_ch - 1))] = ys[0]
        else:
            left = 0.0 if pad_zero_outside else ys[0]
            right = 0.0 if pad_zero_outside else ys[-1]
            prof = np.interp(grid, xs, ys, left=left, right=right)
        for m in range(max(0, n - range_halfwidth),
                       min(Nr, n + range_halfwidth + 1)):
            dmap[m, :] = prof
            active.add(m)

    if verbose:
        pk = np.max(np.abs(dmap))
        print(f"  channel {channel}: {len(active)} active range bins, "
              f"max |dC0| = {pk*1e3:.1f} mm = {360*pk/p.wl:.0f} deg")
    return dmap, sorted(active)


# ---------------------------------------------------------------------------
# 3) SATA-condition every channel for one output sub-band
# ---------------------------------------------------------------------------
def channels_subband_3d(p, tracks, s_channel, k: int, remove: bool = True,
                             sata_osf: int = 4, legacy: bool = False,
                             maps=None, f_centre=None,
                             verbose: bool = False) -> np.ndarray:
    """
    Copy of the range-compressed cube ``[Nrx, Na_ch, Nr]`` with every channel
    conditioned for output sub-band ``k``.

    The kernel is called with ``prf_data = PRF_op`` (the channel line's own
    rate), ``f_centre = f_k`` (the sub-band's frequency window) and
    ``squint_image = 0`` (the reconstructed image is broadside).  ``maps`` can
    carry pre-computed residual maps, since they do not depend on ``k``.
    """
    Nrx, Na_ch, Nr = s_channel.shape
    out = np.array(s_channel, dtype=complex, copy=True)
    f_k = (subband_centre_frequency(k, Nrx, p.prf) if f_centre is None
           else float(f_centre))
    beta_k = float(np.arcsin(np.clip(p.wl * f_k / (2.0 * p.vs), -1.0, 1.0)))
    if verbose:
        print(f"  sub-band {k}: f_k = {f_k:8.1f} Hz -> "
              f"beta_k = {np.degrees(beta_k):8.4f} deg")

    for i in range(Nrx):
        dmap, active = (maps[i] if maps is not None
                        else build_delta_C0_map_3d(p, tracks, i))
        for n in active:
            line = out[i, :, n]
            if legacy:
                out[i, :, n] = legacy_sata_1d(
                    line, dmap[n], rref=p.r0, prf=p.PRF_op, Nsb=Nrx, v=p.vs,
                    squint=beta_k, wl=p.wl, r=p.r0, inverse=remove,
                    sata_osf=sata_osf)
            else:
                out[i, :, n] = sata_1d(
                    line, dmap[n], rref=p.r0, prf_data=p.PRF_op, v=p.vs,
                    wl=p.wl, r=p.r0, f_centre=f_k, squint_image=0.0,
                    inverse=remove, sata_osf=sata_osf)
    return out


# ---------------------------------------------------------------------------
# 4) The per-sub-band orchestrator
# ---------------------------------------------------------------------------
def reconstruct_subband_2d(p, tracks, s_channel_rc, coeff_fn,
                           use_sata="subband", sata_osf: int = 4,
                           legacy: bool = False, apply_rcmc: bool = True,
                           analytic: bool = False, maps=None,
                           verbose: bool = True):
    """
    Two-step 2-D reconstruction with SATA.

    ``use_sata``:

    ``False``      no correction -- reduces EXACTLY to the plain reconstruction.
    ``"whole"``    ONE SATA pass on the whole band (``f_centre = 0``), then one
                   reconstruction.  This is the 2-D analogue of the 1-D
                   ``sata_channels``.
    ``"subband"``  one SATA pass and one reconstruction PER output sub-band
                   (``f_centre = f_k``), keeping only that sub-band's rows.
                   Nrx times the cost of ``"whole"``.

    Comparing the two is itself a result: the C0 residual of Eq. (1) is
    evaluated at closest approach and is therefore essentially angle-invariant,
    so at the C0 stage the two should agree.  The per-sub-band machinery becomes
    necessary at C1 (the registration / DPCA term), which IS antisymmetric in
    the sub-band angle.

    Returns the focused-domain array ``[Na, Nr]`` (azimuth time x range time).
    """
    Nrx, Na_ch, Nr = s_channel_rc.shape
    Na = Na_ch * Nrx
    fmax, fmin = p.abw / 2.0, -p.abw / 2.0

    if not use_sata or len(p.extra_offsets) == 0:
        rec, _ = run_two_step(p, s_channel_rc, fmax, fmin, analytic=analytic,
                              apply_rcmc=apply_rcmc, verbose=verbose,
                              coeff_fn=coeff_fn, return_spectrum=True)
        return finish_spectrum(rec, p.prf, p.abw, Na)

    # The residual maps do not depend on the sub-band: build them once.
    if maps is None:
        maps = [build_delta_C0_map_3d(p, tracks, i, verbose=verbose)
                for i in range(Nrx)]

    if use_sata == "whole":
        ch = channels_subband_3d(p, tracks, s_channel_rc, 0, remove=True,
                                      sata_osf=sata_osf, legacy=legacy,
                                      maps=maps, f_centre=0.0, verbose=verbose)
        rec, _ = run_two_step(p, ch, fmax, fmin, analytic=analytic,
                              apply_rcmc=apply_rcmc, verbose=verbose,
                              coeff_fn=coeff_fn, return_spectrum=True)
        return finish_spectrum(rec, p.prf, p.abw, Na)

    stacked = np.zeros([Na, Nr], np.complex64)
    for k in range(Nrx):
        ch_k = channels_subband_3d(p, tracks, s_channel_rc, k,
                                        remove=True, sata_osf=sata_osf,
                                        legacy=legacy, maps=maps,
                                        verbose=verbose)
        rec_k, _ = run_two_step(p, ch_k, fmax, fmin, analytic=analytic,
                                apply_rcmc=apply_rcmc, verbose=False,
                                coeff_fn=coeff_fn, return_spectrum=True)
        stacked[k * Na_ch:(k + 1) * Na_ch, :] = rec_k[k * Na_ch:(k + 1) * Na_ch, :]
    return finish_spectrum(stacked, p.prf, p.abw, Na)


# ============================================================================
# 6) IRF playground machinery -- formerly irf_common.py
# ============================================================================
METHOD_LABEL = {"no": "no SATA", "whole": "SATA whole band", "sub": "SATA per sub-band"}
METHOD_KW = {"no": dict(use_sata=False), "whole": dict(use_sata="whole"),
             "sub": dict(use_sata="subband")}

_text_kwargs = dict(fontsize="small", verticalalignment="top",
                    bbox=dict(boxstyle="round", fc="w", ec="0.5"))


def matched_filter(s, ref):
    """Pure correlation of s against ref, centred -- as in sar_recon.analysis."""
    Na = len(ref)
    return np.roll(np.fft.ifft(np.fft.fft(s) * np.conjugate(np.fft.fft(ref))),
                   int(Na / 2))


def run_irf(target_azimuth, target_height, Nrx, dxt, sata_osf,
           verbose=False, methods=("no", "whole", "sub")):
    """Build the geometry, reconstruct the single target (only the methods
    requested -- default all three), return (p, res) with
    res['mono'|'no'|'whole'|'sub'] the corresponding azimuth IRFs."""
    specs = ((target_azimuth, target_height),) \
        if (target_azimuth, target_height) != (0.0, 0.0) else ()
    p = make_params3d(Nrx=Nrx, dxt=dxt, specs=specs)
    tr = build_tracks_3d(p)

    ptg = p.points[0]
    nb = range_bin_of(p, scatterer_range(p, ptg))
    print(p.summary())
    print(f"target: azimuth={target_azimuth:.1f} m, height={target_height:.1f} m "
         f"-> range bin {nb}/{p.Nr}\n")

    ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)
    ch = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
    ref_line = ref[:, nb]

    tab = CoeffTable3D(p, tr, n_nodes=16)
    maps = [build_delta_C0_map_3d(p, tr, i) for i in range(p.Nrx)]

    res = {}
    for tag in methods:
        rec = reconstruct_subband_2d(p, tr, ch, tab, sata_osf=sata_osf,
                                     maps=maps, verbose=verbose, **METHOD_KW[tag])
        res[tag] = matched_filter(rec[:, nb], ref_line)
    res["mono"] = matched_filter(ref[:, nb], ref_line)

    pm = np.max(np.abs(res["mono"]))
    print(f"  {'':>6}  {'peak':>10}  {'% of monostatic':>16}")
    for tag in ("mono",) + tuple(methods):
        v = float(np.max(np.abs(res[tag])))
        print(f"  {tag:>6}  {v:10.3e}  {100 * v / pm:15.1f}%")
    return p, res


def _param_box(p):
    dx = p.bat[1] - p.bat[0] if p.Nrx > 1 else 0.0
    bxt_max = float(np.max(np.abs(p.bxt))) if p.Nrx > 0 else 0.0
    return (f"Nrx={p.Nrx} | $\\mathrm{{PRF}}={p.prf:.1f}\\,\\mathrm{{Hz}}$ | "
           f"$B_a={p.abw:.1f}\\,\\mathrm{{Hz}}$ | "
           f"$\\Delta b_{{at}}={dx:.1f}\\,\\mathrm{{m}}$ | "
           f"$b_{{xt}}^{{\\max}}={bxt_max:.1f}\\,\\mathrm{{m}}$")


def plot_irf_single(p, res, method, out="sata_irf.png"):
    """Single-method IRF plot: ref vs. rec in dB over the FULL azimuth time
    axis, with the parameter box above the plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Na = p.Na
    t = (np.arange(Na) - Na // 2) / p.prf
    pk = np.max(np.abs(res["mono"]))
    db_ref = 20.0 * np.log10(np.abs(res["mono"]) / pk + 1e-20)
    db_rec = 20.0 * np.log10(np.abs(res[method]) / pk + 1e-20)

    fig, ax = plt.subplots(figsize=(9.0, 5.5), dpi=150)
    fig.suptitle(f"{METHOD_LABEL[method]} | {_param_box(p)}")

    ax.plot(t, db_ref, label="ref")
    ax.plot(t, db_rec, label="rec")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("[dB]")
    ax.set_ylim(-100, 0)
    ax.grid()
    ax.legend(fontsize="small", loc="best")
    ax.text(0.02, 0.95, f"Nrx={p.Nrx}", transform=ax.transAxes, **_text_kwargs)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure written to {out}")


def plot_irf_all(p, res, out="sata_irf_all.png"):
    """The three methods together (no SATA / whole band / per sub-band)
    against the monostatic reference, same style as plot_irf_single."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Na = p.Na
    t = (np.arange(Na) - Na // 2) / p.prf
    pk = np.max(np.abs(res["mono"]))
    order = (("mono", "ref (monostatic)", "#444444"),
             ("no", "no SATA", "#C62828"),
             ("whole", "SATA whole band", "#1F4E79"),
             ("sub", "SATA per sub-band", "#2E7D32"))

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)
    fig.suptitle(f"IRF comparison | {_param_box(p)}")

    for key, lab, col in order:
        if key not in res:
            continue
        db = 20.0 * np.log10(np.abs(res[key]) / pk + 1e-20)
        ax.plot(t, db, label=lab, color=col, lw=1.1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("[dB]")
    ax.set_ylim(-100, 0)
    ax.grid()
    ax.legend(fontsize="small", loc="best")
    ax.text(0.02, 0.95, f"Nrx={p.Nrx}", transform=ax.transAxes, **_text_kwargs)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure written to {out}")
