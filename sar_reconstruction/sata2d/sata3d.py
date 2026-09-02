# -*- coding: utf-8 -*-
"""
SATA in 2-D with a full 3-D geometry: along-track AND cross-track baselines,
targets at arbitrary height.

What this module adds to `sata2d.py`
------------------------------------
`sata2d.py` fixed the SATA *kernel* (sub-aperture sizing, per-sub-band frequency
bins, azimuth mapping) but its residual map was a 1-D linear-orbit stand-in.
Here the residual comes from the real geometry:

    dC0_i(p) = C0_i(p)  -  C0_i(p_flat(r_p))                               (1)

where ``p`` is the true scatterer (azimuth x, cross-track y, height h),
``r_p = sqrt(y^2 + (H-h)^2)`` is its slant range, and ``p_flat(r_p)`` is the
flat-earth point at that SAME slant range and the same azimuth -- i.e. exactly
what the reconstruction filter of that range bin assumes.  Both C0 come from
`geom3d.get_coeff_nu_3d`, the same fit the filter uses, so correction and error
are one consistent model.

To leading order (1) is the familiar single-pass interferometric sensitivity

    dC0 ~ - b_perp * dh / (r0 sin(theta_inc))                              (2)

which for the default geometry (r0 = 766 km, theta_inc = 20 deg, b_xt = 225 m,
dh = 400 m) gives -0.343 m, i.e. 1.37 wavelengths at L-band -- 494 degrees of
uncorrected phase.  The exact value from (1) is -0.322 m, 6 % away; (2) is only
used as an order-of-magnitude check.

Why the map is 2-D
------------------
Every scatterer lands in the range bin of its own slant range, and the residual
must be applied there and at its own azimuth pixel.  The map is therefore
``[Nr, Na_ch]`` per channel.  Range bins with no scatterer stay at zero, so the
(expensive) SATA kernel only runs on the bins that actually carry topography.

Per-sub-band flow
-----------------
`reconstruct_subband_2d` runs the whole two-step reconstruction once per output
sub-band ``k``, each time on the channel cube conditioned for that sub-band
(``f_centre = f_k``, ``squint_image = 0``), and keeps only sub-band ``k``'s rows
of the stacked spectrum before the common final stage.  That is the 2-D version
of the scheme, and it reduces exactly to the plain reconstruction when the scene
is flat.
"""
from __future__ import annotations

import numpy as np

from .geom3d import residual_C0_3d
from .rd_recons2d import run_two_step, finish_spectrum
from .sata2d import sata_1d, legacy_sata_1d, subband_centre_frequency

C_LIGHT = 299792458.0

__all__ = ["scatterer_range", "scatterer_range_bistatic", "range_bin_of",
           "az_pixel_of",
           "build_delta_C0_map_3d", "sata_channels_subband_3d",
           "reconstruct_subband_2d"]


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
    return int(round((r - p.r_scan[0]) * 2.0 * p.rsf / C_LIGHT))


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
def sata_channels_subband_3d(p, tracks, s_channel, k: int, remove: bool = True,
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
        ch = sata_channels_subband_3d(p, tracks, s_channel_rc, 0, remove=True,
                                      sata_osf=sata_osf, legacy=legacy,
                                      maps=maps, f_centre=0.0, verbose=verbose)
        rec, _ = run_two_step(p, ch, fmax, fmin, analytic=analytic,
                              apply_rcmc=apply_rcmc, verbose=verbose,
                              coeff_fn=coeff_fn, return_spectrum=True)
        return finish_spectrum(rec, p.prf, p.abw, Na)

    stacked = np.zeros([Na, Nr], np.complex64)
    for k in range(Nrx):
        ch_k = sata_channels_subband_3d(p, tracks, s_channel_rc, k,
                                        remove=True, sata_osf=sata_osf,
                                        legacy=legacy, maps=maps,
                                        verbose=verbose)
        rec_k, _ = run_two_step(p, ch_k, fmax, fmin, analytic=analytic,
                                apply_rcmc=apply_rcmc, verbose=False,
                                coeff_fn=coeff_fn, return_spectrum=True)
        stacked[k * Na_ch:(k + 1) * Na_ch, :] = rec_k[k * Na_ch:(k + 1) * Na_ch, :]
    return finish_spectrum(stacked, p.prf, p.abw, Na)
