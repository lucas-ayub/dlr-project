# -*- coding: utf-8 -*-
"""
SATA -- Subaperture Topography- (and Aperture-) dependent Autofocus, 1D.

This is a cleaned, package-integrated port of Pau Prats' `sata_for_topography`
(the 1D routine in SATA1D.txt). It applies a *position-dependent* residual
range/phase correction to a single azimuth line, working in the partially
focused (sub-aperture spectral) domain.

Scope
-----
This implementation corrects the C0 (bulk range / phase) term only, exactly
like the classic 1D SATA. The topographic residual the multichannel
reconstruction leaves is dominated by C0 (proportional to Dh * d_xt); the
higher-order (C1 registration / C2 focus) residuals are negligible in the
broadside regime and are intentionally NOT corrected here.

Why it lives in `sar_recon`
---------------------------
The numerical reconstruction (`GetCoeffNu` / `ReconstructSignalNumeri`) fits a
single coefficient set (C0, C1, C2) per range bin, assuming a flat reference
height h0. A real target at height h != h0 with a non-zero cross-track baseline
d_xt carries a residual range term (the "delta_C0" here) proportional to
Dh * d_xt that the reconstruction filter does NOT account for. SATA injects
that per-position correction back into each channel *before* reconstruction:

    ph(pos) = -2*pi/lambda * delta_C0(pos)

so the two key functions in this module are:

    sata_1d               -- the SATA kernel (faithful to SATA1D.txt)
    build_delta_C0_array  -- turns scene topography into the delta_C0 map,
                             using the very same GetCoeffNu geometry the
                             reconstruction uses (so the correction and the
                             error come from one consistent model)

Differences from SATA1D.txt (all behaviour-preserving fixes)
------------------------------------------------------------
* The overlap-add bookkeeping in the .txt (the `temp`/`leftweight` block and
  the final-block index) had off-by-one bugs and a duplicated `temp = aux`.
  It is replaced here by a standard weighted overlap-add (WOLA) with a
  triangular analysis window and explicit weight normalisation. The physics
  (sub-aperture sizing, Doppler->angle->azimuth-position mapping, and the
  ph = -2*pi/lambda * delta_C0 correction) is untouched.
* `np.log10(x)/np.log10(2)` is written as `np.log2(x)`.
* `arcsin` argument is clipped to [-1, 1] for numerical safety.
"""
from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .geometry import PlatformTracks
from .reconstruction import GetCoeffNu


# ---------------------------------------------------------------------------
# The SATA kernel (1D)
# ---------------------------------------------------------------------------
def sata_1d(data, delta_C0_array, rref, prf, v, wl, r,
            squint=0.0, Nsb=1, inverse=False, sata_osf=1, verbose=True):
    """
    Apply the 1D SATA topography correction (C0 term) to one azimuth line.

    For each sub-aperture and Doppler bin the constant phase
    ph = -2*pi/lambda * delta_C0(x_p) is applied, where x_p is the azimuth
    position the bin maps to. This corrects the bulk range (C0) residual only.

    Parameters
    ----------
    data : (naz,) complex
        Range-compressed azimuth signal of a single channel (before azimuth
        focusing). Modified out of place; the corrected copy is returned.
    delta_C0_array : (naz,) float
        Residual range term [m] as a function of *azimuth image position*.
        delta_C0_array[i] is the extra slant range that a scatterer imaged at
        azimuth pixel i needs, relative to the flat reconstruction reference.
        Build it with `build_delta_C0_array`.
    rref : float
        Reference slant range [m] used to size the sub-aperture (mid-range).
    prf : float
        Azimuth sampling rate of `data` [Hz]. For a reconstruction channel
        this is the per-channel operating PRF (PRF_op = prf_full / Nrx).
    v : float
        Platform (sensor) velocity [m/s].
    wl : float
        Wavelength [m].
    r : float
        Slant range of this azimuth line [m] (used for the Doppler->azimuth
        position mapping). Usually the target/scene range r0.
    squint : float
        Processing squint angle [rad]. 0 for broadside.
    Nsb : int
        Number of sub-bands the azimuth spectrum is split into (1 for a plain
        per-channel line).
    inverse : bool
        If True apply the inverse (de-)correction (exp(-j*ph) instead of
        exp(+j*ph)). Use True to REMOVE the residual (the correction).
    sata_osf : int
        Sub-aperture oversampling factor (zero-padding multiplier). Increase
        if the peak-to-peak correction is large enough to wrap.
    verbose : bool
        Print the sub-aperture sizing report.

    Returns
    -------
    (naz,) complex : the SATA-corrected azimuth line.
    """
    data = np.asarray(data, dtype=complex).copy()
    dimx = len(data)

    # -------------------------------------------------------------------
    # STEP 1 : SATA pre-computations (sub-aperture geometry)
    # -------------------------------------------------------------------
    # Optimum sub-aperture ground extent -- best SATA resolution.
    deltax = np.sqrt(wl * rref / 2.0)
    # Sub-aperture length in samples (forced even).
    Tsubeff = int(np.round(deltax * prf / v * 0.5 / Nsb) * 2)

    if Tsubeff <= 2:
        if verbose:
            print("SATA: sub-aperture length very small, no correction needed")
        return data

    # Zero-padded block length (next power of two, times oversampling).
    Nzp = int(2 ** np.ceil(np.log2(Tsubeff)) * sata_osf)
    Tsub = int(Tsubeff * 0.5)          # hop = half the sub-aperture (50% overlap)
    Tovl = Tsub                        # overlap length
    Nsub = int(dimx // Tsub)
    marg_az = int(np.round(0.5 * (1 + Nzp - Tsubeff)))

    if verbose:
        print(f"SATA: mid-range topography accommodation resolution: {deltax:.2f} m")
        print(f"SATA: sub-aperture length {Tsubeff}, zero-padded to {Nzp} "
              f"(margin {marg_az}), Nsub={Nsub}")

    # Triangular analysis window for weighted overlap-add (50% overlap).
    rightweight = np.arange(Tovl) / (Tovl - 1)           # 0 -> 1
    leftweight = rightweight[::-1]                        # 1 -> 0
    win = np.concatenate((rightweight, leftweight))       # length 2*Tovl = Tsubeff

    # Sub-aperture frequency axis (Doppler), aligned to the data spectrum.
    fc = 2.0 * v / wl * np.sin(squint)                   # Doppler centroid [Hz]
    pfc = np.round(np.mod(fc, prf) * Nzp / prf)          # centroid in pixels
    dfc = np.round(fc * Nzp / prf) * prf / Nzp           # centroid, sample-rounded
    fsub = Nsb * (np.arange(Nzp) * prf / Nzp / Nsb - prf * 0.5 / Nsb) + dfc
    fsub[0] = prf * 0.5 + dfc                            # fix the wrap-around sample
    fsub = np.roll(fsub, int(Nzp / 2 + pfc))            # align with data

    # Doppler -> squint angle -> azimuth image-position offset [pixels].
    betasub = np.arcsin(np.clip(wl * fsub / (2.0 * v), -1.0, 1.0))
    azpos = r * (np.tan(betasub) - np.tan(squint)) / v * prf

    # -------------------------------------------------------------------
    # STEP 2 : weighted overlap-add loop
    # -------------------------------------------------------------------
    L = Tsubeff
    hop = Tsub
    out = np.zeros(dimx, dtype=complex)
    wsum = np.zeros(dimx, dtype=float)
    max_ph = 0.0

    for start in range(0, dimx, hop):
        seg = data[start:start + L]
        Lseg = len(seg)
        if Lseg < 2:
            break

        # Windowed, zero-padded sub-aperture.
        buf = np.zeros(Nzp, dtype=complex)
        buf[:Lseg] = seg * win[:Lseg]
        spec = np.fft.fft(buf)

        # For each Doppler bin, the azimuth image position it maps to.
        center = start + 0.5 * L
        posaux = np.round(azpos + center).astype(int)
        posaux = np.clip(posaux, 0, dimx - 1)

        # C0 residual -> constant-per-position phase correction.
        ph = -2.0 * np.pi / wl * delta_C0_array[posaux]
        ph[~np.isfinite(ph)] = 0.0
        max_ph = max(max_ph, float(np.max(np.abs(ph))))

        spec *= np.exp(-1j * ph) if inverse else np.exp(1j * ph)
        rec = np.fft.ifft(spec)[:Lseg]

        # Weighted overlap-add. The triangular analysis window already weights
        # `rec`; the synthesis is rectangular and we normalise by the summed
        # analysis weights (triangular @ 50% overlap satisfies COLA), so with
        # ph = 0 the identity is reconstructed exactly.
        out[start:start + Lseg] += rec
        wsum[start:start + Lseg] += win[:Lseg]

    nz = wsum > 1e-12
    out[nz] /= wsum[nz]
    out[~nz] = data[~nz]   # untouched edges keep the original samples

    if verbose:
        print(f"SATA: max peak-to-peak phase correction: "
              f"{np.degrees(2 * max_ph):.1f} deg")
        min_osf = sata_osf + int(np.ceil(2 * (0.5 * max_ph / np.pi - marg_az)
                                          * (sata_osf / Nzp)))
        if min_osf > sata_osf:
            print(f"SATA WARNING: margin insufficient; increase sata_osf "
                  f"from {sata_osf} to {min_osf} to avoid artefacts.")

    return out


# ---------------------------------------------------------------------------
# Building the delta_C0 map from the reconstruction geometry
# ---------------------------------------------------------------------------
def residual_C0(cfg: ExperimentConfig, tracks: PlatformTracks,
                ptg_real: np.ndarray, channel: int) -> float:
    """
    Residual C0 [m] for a single scatterer, for one receiver channel.

        residual_C0 = C0(real target) - C0(assumed reconstruction centre)

    Both terms come from `GetCoeffNu`, i.e. exactly the polynomial the
    reconstruction filter uses. This is the physical quantity SATA corrects:
    the extra slant range the flat-earth filter fails to model.
    """
    kk = channel
    common = (tracks.ptx, tracks.prx[kk], tracks.vtx, tracks.vrx[kk],
              tracks.ptx, tracks.vtx, cfg.prf, cfg.system.wl, cfg.ta,
              cfg.sq_tx, cfg.sq_rx[kk], cfg.theta_tx, cfg.theta_rx[kk])
    C0_real = GetCoeffNu(ptg_real, *common)[0]
    C0_ref = GetCoeffNu(cfg.scene.ptg, *common)[0]
    return float(C0_real - C0_ref)


def az_pixel_of_dx(cfg: ExperimentConfig, dx: float) -> int:
    """
    Azimuth image pixel of an along-track offset dx [m] on the per-channel grid.

    The per-channel line is sampled at PRF_op, so along-track sample spacing is
    vs / PRF_op. The scene centre sits at Na_ch/2.
    """
    ds = cfg.system.vs / cfg.PRF_op
    return int(round(cfg.Na_ch / 2 + dx / ds))


def build_delta_C0_array(cfg: ExperimentConfig, tracks: PlatformTracks,
                         channel: int, naz: int | None = None,
                         pad_zero_outside: bool = False) -> np.ndarray:
    """
    Build delta_C0_array[naz] from the extra scatterers of the Scene.

    Each extra scatterer (dx, dy, dh) is a piece of topography at along-track
    position dx and height h0+dh. Its residual C0 (relative to the flat
    reconstruction centre) is computed with `residual_C0`, placed at the
    scatterer's azimuth pixel, and interpolated across azimuth so every pixel
    gets a correction.

    pad_zero_outside : if False (default) pixels beyond the scatterer span are
        held at the nearest endpoint value (the dominant target governs the
        whole line -- correct for isolated targets). If True they are set to 0
        (flat ground outside the topography patch).

    Returns
    -------
    (naz,) float : the delta_C0 map ready for `sata_1d`. All zeros if the
        Scene has no extra scatterers.
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
        by_pixel[az_pixel_of_dx(cfg, dx)].append(residual_C0(cfg, tracks, ptg_real, channel))

    # One value per pixel: the largest-magnitude residual wins (dominant scatterer).
    pix = sorted(by_pixel)
    xs = np.array(pix, dtype=float)
    ys = np.array([max(by_pixel[p], key=abs) for p in pix], dtype=float)

    grid = np.arange(naz)
    if xs.size == 1:                      # single topography position -> constant
        arr = np.zeros(naz) if pad_zero_outside else np.full(naz, ys[0])
        if pad_zero_outside:
            arr[int(xs[0])] = ys[0]
        return arr

    left = 0.0 if pad_zero_outside else ys[0]
    right = 0.0 if pad_zero_outside else ys[-1]
    return np.interp(grid, xs, ys, left=left, right=right)


# ---------------------------------------------------------------------------
# Convenience: apply SATA to every channel before reconstruction
# ---------------------------------------------------------------------------
def sata_channels(cfg: ExperimentConfig, tracks: PlatformTracks,
                  s_channel: np.ndarray, remove: bool = True, sata_osf: int = 4,
                  verbose: bool = False) -> np.ndarray:
    """
    Pre-condition every receiver channel of `s_channel` [Nrx, Na_ch] with SATA
    *before* reconstruction.

    Each channel carries a topography-induced residual C0 (proportional to
    Dh * d_xt) that the flat-earth reconstruction filter does not model.
    `build_delta_C0_array` estimates it per channel; here we REMOVE it
    (remove=True) so the channel matches what the filter expects. In the
    single-target iso-range case this recovers exactly the reconstruction a
    filter with the true height would have produced.

    remove : True removes the residual (correction); False injects it (for
        building test cases / round-trips).

    Returns the corrected copy (same shape).
    """
    Nrx = cfg.Nrx
    out = np.asarray(s_channel, dtype=complex).copy()
    for kk in range(Nrx):
        dC0 = build_delta_C0_array(cfg, tracks, kk, naz=cfg.Na_ch)
        out[kk, :] = sata_1d(
            out[kk, :], dC0, rref=cfg.scene.r0, prf=cfg.PRF_op,
            v=cfg.system.vs, wl=cfg.system.wl, r=cfg.scene.r0,
            squint=cfg.sq_tx, Nsb=1, inverse=remove,
            sata_osf=sata_osf, verbose=verbose,
        )
    return out