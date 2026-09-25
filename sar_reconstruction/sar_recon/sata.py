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
            squint=0.0, Nsb=1, inverse=False, sata_osf=1, verbose=True,
            delta_C1_array=None, delta_C2_array=None, debug_center=False,
            analysis_window="triangular"):
    """
    Apply the 1D SATA topography correction to one azimuth line.

    For each sub-aperture and Doppler bin the constant phase
    ph = -2*pi/lambda * delta_C0(x_p) is applied, where x_p is the azimuth
    position the bin maps to. This corrects the bulk range (C0) residual.

    EXPERIMENTAL C1/C2 extension (opt-in, off by default)
    -------------------------------------------------------
    delta_C1_array and delta_C2_array, if given, add extra per-bin phase
    terms built from the SAME (position -> Doppler bin) map already used for
    C0, mirroring the sign/structure of the reconstruction filter's own phase
    model (see `hf` in `subband_recon.reconstruct_subband`):

        ph_C1(x_p, f) = -2*pi * delta_C1(x_p) * f
        ph_C2(x_p, f) = -2*pi * delta_C2(x_p) * wl * f**2

    These are a first-order, UNVALIDATED extension of the classic (C0-only)
    SATA kernel meant for exploratory A/B comparisons ("which term actually
    moves the result"), not yet a verified physical correction. Confirm with
    a round-trip test (inverse=True then inverse=False reproduces the input)
    before trusting the C1/C2 outputs quantitatively.

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
    debug_center : bool
        If True, also return a dict with the raw (pre-correction) sub-aperture
        spectrum of the window closest to the CENTRE of `data` -- i.e. exactly
        `spec = np.fft.fft(buf)` from STEP 2 below, for the one window nearest
        dimx/2, plus its Doppler axis and sizing. Useful for plotting/
        inspecting the sub-aperture spectrum SATA actually computes, without
        reimplementing any of this function's logic. Return value becomes
        `(out, debug_info)` instead of just `out` when True.
    analysis_window : {"triangular", "rectangular"}
        Analysis window applied to each sub-aperture before the FFT.
        "triangular" (default) is the original SATA behaviour -- required
        for the weighted-overlap-add reconstruction (`out`) to satisfy COLA
        and reproduce the identity when delta_C0=0. "rectangular" applies no
        weighting (all-ones window) -- only meant for inspecting/plotting the
        RAW sub-aperture spectrum (e.g. via debug_center) with narrower main
        lobes and un-tapered (~-13 dB) sidelobes; the reconstructed `out` is
        NOT COLA-guaranteed with this option and should not be trusted.

    Returns
    -------
    (naz,) complex : the SATA-corrected azimuth line.
    (naz,) complex, dict : if debug_center=True, also the debug_info dict
        with keys "spec" (Nzp complex), "fsub" (Nzp float, Doppler axis,
        NOT fftshift'ed -- same order as `spec`), "Tsubeff", "Nzp", "win"
        (the Tsubeff-length triangular analysis window), "start" (the sample
        index the captured window began at).
    """
    data = np.asarray(data, dtype=complex).copy()
    dimx = len(data)

    # -------------------------------------------------------------------
    # STEP 1 : SATA pre-computations (sub-aperture geometry)
    # -------------------------------------------------------------------
    # Optimum sub-aperture ground extent -- best SATA resolution.
    deltax = np.sqrt(wl * rref / 2.0)
    # Sub-aperture length in samples (forced even).
    # Tsubeff = int(np.round(deltax * prf / v * 0.5 / Nsb) * 2)
    Tsubeff = int(np.round(deltax * prf / v * 0.5 ) * 2)

    if Tsubeff <= 2:
        if verbose:
            print("SATA: sub-aperture length very small, no correction needed")
        return (data, None) if debug_center else data

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

    # Analysis window for the weighted overlap-add (50% overlap).
    if analysis_window == "rectangular":
        win = np.ones(Tsubeff)                             # no weighting
    else:
        # Triangular (default, required for COLA / the identity round-trip).
        rightweight = np.arange(Tovl) / (Tovl - 1)          # 0 -> 1
        leftweight = rightweight[::-1]                       # 1 -> 0
        win = np.concatenate((rightweight, leftweight))      # length 2*Tovl = Tsubeff

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
    debug_info = None
    best_center_dist = np.inf

    for start in range(0, dimx, hop):
        seg = data[start:start + L]
        Lseg = len(seg)
        if Lseg < 2:
            break

        # Windowed, zero-padded sub-aperture.
        buf = np.zeros(Nzp, dtype=complex)
        buf[:Lseg] = seg * win[:Lseg]
        spec = np.fft.fft(buf)

        if debug_center:
            center_dist = abs((start + 0.5 * L) - 0.5 * dimx)
            if center_dist < best_center_dist:
                best_center_dist = center_dist
                debug_info = dict(spec=spec.copy(), fsub=fsub.copy(),
                                  Tsubeff=Tsubeff, Nzp=Nzp,
                                  win=win.copy(), start=start)

        # For each Doppler bin, the azimuth image position it maps to.
        center = start + 0.5 * L
        posaux = np.round(azpos + center).astype(int)
        posaux = np.clip(posaux, 0, dimx - 1)

        # C0 residual -> constant-per-position phase correction.
        ph = -2.0 * np.pi / wl * delta_C0_array[posaux]
        if delta_C1_array is not None:
            ph = ph - 2.0 * np.pi * delta_C1_array[posaux] * fsub
        if delta_C2_array is not None:
            ph = ph - 2.0 * np.pi * delta_C2_array[posaux] * wl * fsub ** 2
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

    return (out, debug_info) if debug_center else out


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
    LEGACY. Azimuth pixel of an along-track offset dx [m] on the per-channel
    grid (scene centre at Na_ch/2, sample spacing vs/PRF_op). Ignores the
    centre's own x0 and the channel's along-track baseline; kept only for the
    legacy "hold" map. Use `az_pixel_of_scatterer` instead.
    """
    ds = cfg.system.vs / cfg.PRF_op
    return int(round(cfg.Na_ch / 2 + dx / ds))


def az_pixel_of_scatterer(cfg: ExperimentConfig, dx: float, channel: int) -> int:
    """
    Azimuth pixel [per-channel grid] where scatterer (x0 + dx) appears in the
    line of receiver `channel`.

    The slow-time axis is centred (platform at x = 0 at sample Na_ch/2) and the
    line is sampled every ds = vs / PRF_op. The bistatic phase centre of
    channel kk sits bat[kk]/2 behind the transmitter, so the target's closest
    approach in that channel is delayed by bat[kk]/2 along track.
    """
    ds = cfg.system.vs / cfg.PRF_op
    x_abs = cfg.scene.x0 + dx + 0.5 * cfg.array.bat[channel]
    return int(round(cfg.Na_ch / 2 + x_abs / ds))


def sata_footprint_halfwidth(cfg: ExperimentConfig, squint: float = 0.0) -> int:
    """
    Half-width [pixels] of the region where ONE point target appears in the
    SATA sub-aperture spectrum, expressed on the azimuth-position axis
    (x = posaux).

    SATA looks at the data through windows of Tsubeff samples with a
    triangular analysis window; in every window the target shows up as that
    window's spectrum: a main lobe (first null at 2*PRF_op/Tsubeff) and first
    sidelobes (~ -27 dB, up to the second null at 4*PRF_op/Tsubeff). The
    footprint covers everything up to the second null, i.e. all the energy of
    the target above ~ -30 dB. Mapped through SATA's own ruler
    (f -> beta -> x = r tan(beta) / v * PRF_op). Same sizing as `sata_1d`.
    """
    prf, v, wl, r = cfg.PRF_op, cfg.system.vs, cfg.system.wl, cfg.scene.r0
    deltax = np.sqrt(wl * r / 2.0)
    Tsubeff = int(np.round(deltax * prf / v * 0.5) * 2)
    f_edge = 4.0 * prf / Tsubeff                       # second null of the triangular window
    fc = 2 * v / wl * np.sin(squint)
    b0 = np.arcsin(np.clip(wl * fc / (2 * v), -1, 1))
    b1 = np.arcsin(np.clip(wl * (fc + f_edge) / (2 * v), -1, 1))
    return int(np.ceil(r * (np.tan(b1) - np.tan(b0)) / v * prf))


def sata_image_offsets(cfg: ExperimentConfig, f_centre: float = 0.0) -> np.ndarray:
    """
    Offsets [pixels, relative to the scatterer's own pixel] of every place
    where ONE point target appears on SATA's position axis (x = posaux), for a
    kernel whose bins are labelled with the absolute Doppler in
    [f_centre - PRF_op/2, f_centre + PRF_op/2) on the broadside image grid.

    The channel line is sampled at PRF_op but the target's Doppler history
    spans [-B/2, B/2], B = 4 v sin(theta_tx/2) / wl. A true Doppler f is
    labelled f + m*PRF_op (the m that brings it into the kernel's window), and
    the ruler maps that label to x_target + m*X with
        X = r tan(arcsin(wl*PRF_op/(2v))) / v * PRF_op   (one fold of PRF_op).
    Offsets = {m*X}, m = round((f_centre - B/2)/PRF_op) .. round((f_centre + B/2)/PRF_op).

    f_centre = 0   : whole-band kernel (sata_1d, broadside) -> {-X, 0, +X}.
    f_centre = f_k : per-sub-band kernel of output band k
                     (subband_sata_explicit.sata_1d_subband) -> one-sided set.
    """
    prf, v, wl, r = cfg.PRF_op, cfg.system.vs, cfg.system.wl, cfg.scene.r0
    X = r * np.tan(np.arcsin(np.clip(wl * prf / (2 * v), -1, 1))) / v * prf
    B = 4.0 * v * np.sin(cfg.theta_tx / 2.0) / wl
    m_lo = int(np.round((f_centre - 0.5 * B) / prf))
    m_hi = int(np.round((f_centre + 0.5 * B) / prf))
    return np.array([m * X for m in range(m_lo, m_hi + 1)], dtype=float)


def sata_alias_images(cfg: ExperimentConfig, squint: float = 0.0):
    """
    Whole-band summary of `sata_image_offsets`: (X, M) = image spacing [px]
    and number of images per side for a broadside (f_centre = 0) kernel.
    """
    prf, v, wl, r = cfg.PRF_op, cfg.system.vs, cfg.system.wl, cfg.scene.r0
    X = r * np.tan(np.arcsin(np.clip(wl * prf / (2 * v), -1, 1))) / v * prf
    M = int(max(0, np.max(np.abs(np.round(sata_image_offsets(cfg) / X)))))
    return float(X), M


def footprint_map(values_by_pixel: dict, naz: int, halfwidth: int,
                  offsets=(0.0,)) -> np.ndarray:
    """
    delta map that is NON-ZERO ONLY where the scatterers appear in the SATA
    STFT, 0 elsewhere (flat reference).

    1) Terrain blocks. Scatterers are sorted by pixel (several at the SAME
       pixel: largest |value|). Neighbours whose footprints touch
       (gap <= 2*halfwidth) form one block -- a piece of terrain: over
       [first - halfwidth, last + halfwidth] each cell gets the linear
       interpolation between the neighbouring scatterers. An isolated scatterer
       is its own block: its value over pixel +- halfwidth.
    2) Images. Every block is drawn at each offset (0 = direct image, m*X =
       folded copies, see `sata_image_offsets`). Where images overlap, a cell
       takes the image whose core [first, last] + offset is NEAREST (ties: the
       direct image, then the smaller |offset|). Each image is the footprint
       of one component whose main lobe is centred on its core, so every main
       lobe reads its own value and the overlap is split at the midpoint.
       (Previous rule "the direct image wins the whole overlap" let the block
       of one scatterer overwrite the folded main lobe of another up to
       halfwidth away, which degraded separations near m*X +- halfwidth.)
    3) Zero elsewhere.
    """
    arr = np.zeros(naz)
    if not values_by_pixel:
        return arr
    pix = np.array(sorted(values_by_pixel), dtype=float)
    val = np.array([max(values_by_pixel[p], key=abs) for p in sorted(values_by_pixel)], dtype=float)
    cuts = np.flatnonzero(np.diff(pix) > 2 * halfwidth) + 1
    blocks = list(zip(np.split(pix, cuts), np.split(val, cuts)))
    dist = np.full(naz, np.inf)
    for off in sorted(offsets, key=abs):
        sh = float(np.round(off))
        for bp, bv in blocks:
            lo = int(max(0, np.floor(bp[0] + sh - halfwidth)))
            hi = int(min(naz - 1, np.ceil(bp[-1] + sh + halfwidth)))
            if lo > hi:
                continue
            x = np.arange(lo, hi + 1)
            d = np.maximum(0.0, np.maximum(bp[0] + sh - x, x - (bp[-1] + sh)))
            take = d < dist[x]
            arr[x[take]] = np.interp(x[take] - sh, bp, bv)
            dist[x[take]] = d[take]
    return arr


def build_delta_C0_array(cfg: ExperimentConfig, tracks: PlatformTracks,
                         channel: int, naz: int | None = None,
                         mode: str = "footprint",
                         footprint_halfwidth: int | None = None,
                         include_aliases: bool = True,
                         pad_zero_outside: bool = False) -> np.ndarray:
    """
    Build delta_C0_array[naz] (per-channel azimuth grid) from the extra
    scatterers of the Scene.

    Each extra scatterer (dx, dy, dh) gets its residual C0 (`residual_C0`,
    relative to the flat reconstruction centre).

    mode="footprint" (DEFAULT)
        The residual is present ONLY where the scatterer appears in the SATA
        STFT (see `footprint_map`: neighbouring scatterers whose footprints
        touch are treated as terrain and interpolated): pixels
        az_pixel_of_scatterer(...) +- footprint_halfwidth (default
        `sata_footprint_halfwidth`, the main lobe of the sub-aperture
        spectrum) and, if include_aliases, the same footprint around the alias
        images of the scatterer (`sata_image_offsets`: where its folded Doppler
        lands when the line is sampled below its Doppler bandwidth).
        Everywhere else the map is 0 (flat reference), so bins that do not see
        a scatterer receive no correction.
        CAVEAT: with several scatterers, an image of one can overlap the
        footprint of another; each cell then takes the nearest component (see
        `footprint_map`). Two scatterers whose separation is within the main
        lobe of m*X share the same bins and cannot both be corrected
        (Doppler-alias collision, inherent to any per-channel correction).
    mode="hold" (LEGACY, previous behaviour)
        One value per az_pixel_of_dx; a single pixel is held over the whole
        line (or only that pixel if pad_zero_outside=True); several pixels are
        linearly interpolated with the endpoints held.

    Returns (naz,) float; all zeros if the Scene has no extra scatterers.
    """
    if naz is None:
        naz = cfg.Na_ch
    if not cfg.scene.extra_offsets:
        return np.zeros(naz)

    center = cfg.scene.ptg
    from collections import defaultdict
    by_pixel = defaultdict(list)

    if mode == "footprint":
        hw = sata_footprint_halfwidth(cfg) if footprint_halfwidth is None else int(footprint_halfwidth)
        for (dx, dy, dh) in cfg.scene.extra_offsets:
            ptg_real = center + np.array([dx, dy, dh], dtype=np.float64)
            by_pixel[az_pixel_of_scatterer(cfg, dx, channel)].append(
                residual_C0(cfg, tracks, ptg_real, channel))
        offs = sata_image_offsets(cfg) if include_aliases else (0.0,)
        return footprint_map(by_pixel, naz, hw, offs)

    if mode != "hold":
        raise ValueError(f"unknown mode {mode!r}; expected 'footprint' or 'hold'")

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