# -*- coding: utf-8 -*-
"""
Explicit inter-channel range co-registration.

WHAT THIS IS ABOUT
------------------
The two-step reconstruction works ONE RANGE BIN AT A TIME: for range bin ``n``
it takes the ``Nrx`` azimuth lines and interleaves them into a single line at
the full PRF.  It therefore needs range bin ``n`` of every channel to hold the
same target.

A receiver displaced by ``bxt`` across track sees the target at a shorter slant
range, so that is not automatic.  Expanding the bistatic half-path for
``|b| << r`` at the beam centre,

    ||p_rx - p|| = r - b.u + (||b||^2 - (b.u)^2)/(2 r) + O(b^3/r^2)

with ``u`` the platform->target line of sight.  At closest approach ``u`` is
perpendicular to the track, so ``u_x = 0``: the ALONG-track baseline drops out
of the first-order term (it produces the DPCA time offset ``bat/(2 v)``
instead, modelled by ``Dt``).  What survives is

    dR_i(r) = R_i - R_mono = -bxt_i * sin(theta_inc(r)) / 2            (1)

``sin(theta_inc)`` projects the sideways displacement onto the line of sight --
the target is sideways AND down, so moving 1 m across track only closes
``sin(theta_inc)`` m.  The ``/2`` is because only the receive leg moves while
the range axis records the average of the two legs.

For the standard preset (``bxt = +-225 m``, ``theta_inc = 20 deg``) this is
+-38.5 m = +-1.47 range cells: the four channels are spread over almost three
cells.  It matters: with the misalignment left in, the focused peak collapses
to 37 % of the monostatic reference.

THE POINT: THE STANDARD PIPELINE ALREADY DOES THIS
--------------------------------------------------
It is not a missing term.  ``create_ref_dataset`` builds the STEP 1 filter for
EVERY range frequency, and its ``C0`` term reads ``exp(-2j pi C0/wl_m)`` with
``wl_m = c/(f0+fr)``.  Expanding,

    C0/wl(f_r) = C0 f0/c  +  C0 f_r/c                                  (2)
                 \_______/    \________/
                  carrier      envelope shift  (shift theorem)

The second half is linear in range frequency, which by the shift theorem IS a
translation of the envelope by ``C0/2 = dR_i``.  Carrier phase and
co-registration are entangled in one exponential, and building the filter per
range frequency applies both at once.

This module separates them.  It applies (1) explicitly as a range-frequency
ramp and then lets the reconstruction run with a MONOCHROMATIC filter, so the
term is visible and auditable instead of implicit.  The two routes are
interchangeable, and measurably so (``run_coreg.py``)::

    filter          explicit co-reg     peak      err/signal
    wl(f_r)         no                  96.1 %    -6.27 dB     <- standard
    wl(f_r)         yes                 38.6 %    -0.39 dB     <- double count
    wl0             no                  37.4 %    +0.29 dB     <- uncorrected
    wl0             yes                 96.1 %    -6.24 dB     <- this module

Use exactly one of the two.  Doing both double counts, doing neither leaves the
channels misaligned; both failures land in the same place, as they must, since
they differ only in the sign of the residual shift.

WHY BOTHER, IF THEY ARE EQUIVALENT
----------------------------------
Three reasons.  It makes the geometric term explicit rather than hidden inside
a wavelength; it separates a phase correction (``C0``, scale ``lambda``) from a
delay correction (``dR``, scale ``rho_r``), which are physically different
operations; and it survives situations where the filter cannot be built per
range frequency -- a coarser range-frequency grid, or a filter tabulated at one
wavelength.

TOPOGRAPHY
----------
Differentiating ``sin(theta_inc) = y/r`` at fixed ``r``::

    d(dR)/dh = -(bxt/2) cos(theta_inc)/(r sin(theta_inc))

which for ``dh = 240 m``, ``bxt = 225 m`` is 9.7 cm = 0.004 range cells.  The
envelope is insensitive to topography, so this correction needs NO DEM --
unlike SATA, which acts on the ``lambda`` scale and does.

RANGE DEPENDENCE
----------------
``mode="const"`` evaluates (1) once at ``r_ref``.  The error is
``(bxt/(2 rho_r)) [sin th(r_far) - sin th(r_near)]`` cells, printed at run time
and 0.048 cells for the standard preset.  ``mode="block"`` uses
``theta_inc(r_n)`` per overlapping range block, for swaths where that runs out.
"""
from __future__ import annotations

import contextlib

import numpy as np

from .geometry import C0_LIGHT

__all__ = ["sin_theta_inc", "coregistration_shift", "swath_shift_variation",
           "coregister_channels", "monochromatic_filter",
           "reconstruct_explicit_coreg"]


# ---------------------------------------------------------------------------
# 1) the shift
# ---------------------------------------------------------------------------
def sin_theta_inc(p, r, h: float = None) -> np.ndarray:
    """``sin(theta_inc)`` of the flat-earth surface at slant range ``r``."""
    h = p.h0 if h is None else h
    r = np.asarray(r, dtype=float)
    return np.sqrt(np.clip(r ** 2 - (p.H - h) ** 2, 0.0, None)) / r


def coregistration_shift(p, channel: int, r=None, h: float = None) -> np.ndarray:
    """Eq. (1): half-path offset ``dR`` [m] of ``channel`` at slant range ``r``.

    Positive ``dR`` means this channel records the target FARTHER out than the
    monostatic reference does.  ``r`` defaults to the scene reference range.
    """
    r = p.r0 if r is None else r
    return -p.bxt[channel] * sin_theta_inc(p, r, h) / 2.0


def swath_shift_variation(p) -> float:
    """How much ``dR`` changes across the swath, in range cells.

    Decides whether ``mode="const"`` is enough; above ~0.1 cell use
    ``mode="block"``.
    """
    rho_r = C0_LIGHT / (2.0 * p.rsf)
    bxt_max = float(np.max(np.abs(p.bxt))) if p.Nrx else 0.0
    ds = sin_theta_inc(p, p.r_scan[-1]) - sin_theta_inc(p, p.r_scan[0])
    return abs(bxt_max * ds / 2.0) / rho_r


# ---------------------------------------------------------------------------
# 2) applying it
# ---------------------------------------------------------------------------
def _ramp(N: int, delta: float) -> np.ndarray:
    """Baseband range-frequency ramp, in unshifted FFT order.

    ``delta`` is the shift in SAMPLES; the convention is ``x[n] -> x[n+delta]``,
    i.e. a positive ``delta`` moves the envelope towards SMALLER range.
    ``fftfreq(N)*N`` gives the signed bin index directly in FFT order, which
    saves a pair of fftshifts.

    BASEBAND is not optional.  Using the absolute frequency ``f0+f_r`` would
    add ``exp(j 4 pi dR/lambda)`` on top -- the carrier phase that ``C0``
    already owns -- which for the outer channel is a 55411 deg double count.
    """
    k = np.fft.fftfreq(N) * N
    return np.exp(2j * np.pi * k * delta / N)


def coregister_channels(p, s_channel_rc, mode: str = "const",
                        n_blocks: int = 8, olap: float = 0.5,
                        verbose: bool = True):
    """Shift every channel's range envelope onto the monostatic grid.

    ``s_channel_rc`` : ``[Nrx, Na_ch, Nr]`` RANGE-COMPRESSED channel data.
    Returns a new array of the same shape; the input is not modified.

    Envelope only: the carrier phase is deliberately left alone, so ``C0``
    keeps owning it.
    """
    Nrx, Na_ch, Nr = s_channel_rc.shape
    rho_r = C0_LIGHT / (2.0 * p.rsf)

    if verbose:
        var = swath_shift_variation(p)
        print(f"  co-registration ({mode}): swath variation {var:.4f} cells"
              + ("" if var < 0.1 else "  <-- consider mode='block'"))

    if mode == "const":
        out = np.empty_like(s_channel_rc)
        for i in range(Nrx):
            dR = float(coregistration_shift(p, i))
            delta = dR / rho_r
            if verbose:
                print(f"    ch{i}: bxt = {p.bxt[i]:+7.1f} m -> dR = {dR:+7.2f} m "
                      f"= {delta:+6.3f} cells")
            out[i] = np.fft.ifft(np.fft.fft(s_channel_rc[i], axis=-1)
                                 * _ramp(Nr, delta), axis=-1)
        return out

    if mode != "block":
        raise ValueError("mode must be 'const' or 'block'")

    # Range-dependent version: overlapping blocks with triangular weights that
    # sum to one, so with a constant shift it reduces exactly to mode="const".
    step = max(1, int(np.floor(Nr / (n_blocks * (1.0 - olap) + olap))))
    blk = min(Nr, int(round(step / (1.0 - olap))))
    starts = list(range(0, max(1, Nr - blk + 1), step))
    if starts[-1] + blk < Nr:
        starts.append(Nr - blk)
    if verbose:
        print(f"    {len(starts)} range blocks of {blk} samples, step {step}")

    acc = np.zeros(s_channel_rc.shape, np.complex128)
    wsum = np.zeros(Nr)
    tri = 1.0 - np.abs(np.arange(blk) - 0.5 * (blk - 1)) / (0.5 * blk)
    tri = np.clip(tri, 1e-6, None)
    for s0 in starts:
        sl = slice(s0, s0 + blk)
        r_c = p.r_scan[s0 + blk // 2]
        wsum[sl] += tri
        for i in range(Nrx):
            delta = float(coregistration_shift(p, i, r_c)) / rho_r
            seg = np.fft.fft(s_channel_rc[i, :, sl], axis=-1) * _ramp(blk, delta)
            acc[i, :, sl] += np.fft.ifft(seg, axis=-1) * tri
    return (acc / wsum).astype(s_channel_rc.dtype)


# ---------------------------------------------------------------------------
# 3) the monochromatic filter
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def monochromatic_filter():
    """Build the STEP 1 filter at a SINGLE wavelength for the duration.

    ``create_ref_dataset`` normally evaluates the filter at every range
    frequency, which -- per Eq. (2) -- co-registers the channels as a side
    effect.  Inside this context it is evaluated at ``wl_arr[0]`` throughout,
    so the filter carries the carrier phase and nothing else, and the
    co-registration has to be supplied explicitly.

    This is also the experiment of Fig. 4.14 of Sakar's dissertation
    ("the reconstruction coefficients have only been computed for f_r = 0").
    """
    from . import reconstruction as _R

    original = _R.create_ref_dataset

    def _mono(ds, wl_arr, *args, **kwargs):
        return original(ds, np.full_like(wl_arr, wl_arr[0]), *args, **kwargs)

    _R.create_ref_dataset = _mono
    try:
        yield
    finally:
        _R.create_ref_dataset = original


# ---------------------------------------------------------------------------
# 4) the pipeline
# ---------------------------------------------------------------------------
def reconstruct_explicit_coreg(p, tracks, s_channel_rc, coeff_fn,
                               mode: str = "const", n_blocks: int = 8,
                               verbose: bool = True, **kw):
    """Reconstruction with the co-registration done as a separate, visible step.

    Order matters: co-registration comes BEFORE the reconstruction, because
    that is what assumes the channels share a range bin.  It is independent of
    SATA -- one acts on the envelope, the other on the phase -- so every
    ``use_sata`` option behaves exactly as before, and all keyword arguments of
    ``reconstruct_subband_2d`` are forwarded unchanged.

    Numerically equivalent to the standard pipeline (96.1 % of the monostatic
    peak either way, err/signal -6.27 vs -6.24 dB); what changes is that the
    geometric term is explicit.
    """
    from .reconstruction import reconstruct_subband_2d

    ch = coregister_channels(p, s_channel_rc, mode=mode, n_blocks=n_blocks,
                             verbose=verbose)
    with monochromatic_filter():
        return reconstruct_subband_2d(p, tracks, ch, coeff_fn,
                                      verbose=verbose, **kw)
