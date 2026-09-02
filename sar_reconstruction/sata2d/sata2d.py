# -*- coding: utf-8 -*-
"""
SATA (Sub-Aperture Topography- and Aperture-dependent motion compensation)
for the sub-band multichannel reconstruction -- corrected kernel + 2-D driver.

===========================================================================
WHERE THE BUG IS
===========================================================================
Setting.  Each receive channel is sampled at ``PRF_op = prf/Nrx``, so ONE
channel line has ``Na_ch`` samples and a spectrum only ``PRF_op`` wide.  The
reconstruction unmixes the ``Nrx`` aliases: output sub-band ``k`` occupies

    f in [ f_k - PRF_op/2 , f_k + PRF_op/2 ) ,
    f_k = (-Nrx/2 + k + 1/2) * PRF_op                      <- PRF_op wide,
                                                              NOT Nrx*PRF_op

and raw FFT bin ``n`` of a channel carries, *for output sub-band k*, the true
Doppler frequency ``-prf/2 + n*PRF_op/Na_ch + k*PRF_op``.

So a channel line already IS "one sub-band's worth of samples": its sampling
rate is ``prf/Nsb`` with ``Nsb = Nrx``.  Only the frequency LABEL of each bin
changes from sub-band to sub-band.  That is the whole content of the
per-sub-band SATA.

[BUG 1] -- the sub-aperture length (the one to check "after the STFT")
    The kernel sizes the sub-aperture as

        Tsubeff = round(deltax * prf / v * 0.5 / Nsb) * 2

    where the ``/Nsb`` assumes the data is sampled at ``prf/Nsb``.  The call
    site passed ``prf = PRF_op`` (already the per-channel rate) AND
    ``Nsb = Nrx``, so the division was applied a second time:

        Tsubeff_used = Tsubeff_correct / Nrx

    The sub-aperture covered ``deltax/Nrx`` of ground instead of ``deltax``, and
    ``Nzp`` shrank with it, so the STFT Doppler bin spacing became ``Nrx`` times
    coarser than intended.  Since ``azpos`` (the frequency -> azimuth-bin map) is
    read out on exactly that grid, the topography correction was quantised onto
    a grid ``Nrx`` times too coarse -- and with a sub-aperture that short the
    "topography accommodation resolution" ``deltax`` no longer holds at all.
    THIS is the bug that breaks the sub-band SATA + azimuth reconstruction.

[BUG 2] -- the frequency bins were never actually defined per sub-band
    The bin axis in the reference reads

        fsub = Nsb*(arange(Nzp)*prf/Nzp/Nsb - prf*0.5/Nsb) + dfc
             =     arange(Nzp)*prf/Nzp - prf*0.5           + dfc

    The leading ``Nsb`` cancels the ``/Nsb`` inside the bracket: ``Nsb`` has NO
    effect on the frequency axis.  The axis simply spans whatever ``prf`` you
    hand in.  It happened to come out ``PRF_op`` wide only because ``PRF_op``
    was passed by mistake (BUG 1).  Fix BUG 1 alone -- i.e. pass the full
    ``prf`` with ``Nsb = Nrx``, the natural reading of the parameter names --
    and the axis silently becomes ``Nrx`` times too wide, together with
    ``azpos = ... * prf`` (which must also be expressed in samples of the data
    actually handed in).  The two bugs must be fixed together.

[BUG 3] -- the azimuth mapping was measured against a SQUINTED image grid
    The reference maps a Doppler bin to an azimuth offset as

        azpos = r*(tan(betasub) - tan(squint)) / v * prf

    The ``- tan(squint)`` is correct for CLASSIC SATA, where the image is
    processed at the same squint as the beam, so the pixel under the
    sub-aperture centre already sits at ``r*tan(squint)`` down-track.  In the
    sub-band reconstruction that is NOT the case: the reconstructed image is the
    ordinary BROADSIDE image, and only the frequency WINDOW is squinted
    (``squint = beta_k``).  Subtracting ``tan(beta_k)`` therefore shifts the
    whole topography lookup by

        r * tan(beta_k) / v * PRF_op   samples ,

    in OPPOSITE directions for the sub-bands above and below zero Doppler.  For
    the default "small" preset that is +/- 187 azimuth samples per sub-band --
    the correction lands on the wrong piece of topography.  This is why
    whole-band SATA (beta_0 = 0, no shift) works while the per-sub-band version
    does not.  The kernel now takes ``squint_image`` separately from
    ``f_centre``; the sub-band driver passes ``squint_image = 0``.

The fix adopted here: drop the ``(prf, Nsb)`` pair, which is ambiguous, and give
the kernel the two quantities it actually needs --

    ``prf_data``  the sampling rate of the array handed in   ( = PRF_op )
    ``f_centre``  the ABSOLUTE Doppler centre of this sub-band ( = f_k )

Then ``Tsubeff``, ``fsub``, ``betasub`` and ``azpos`` are all expressed in the
same units and cannot drift apart.  :func:`legacy_sata_1d` keeps the original
formulas so the two can be compared -- see ``run_sata_diagnostics.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rd_recons2d import get_coeff_nu

PI = np.pi

__all__ = ["SubapertureGrid", "subaperture_grid", "legacy_subaperture_grid",
           "sata_1d", "legacy_sata_1d", "subband_centre_frequency",
           "build_delta_C0_array", "sata_channels_subband"]


# ---------------------------------------------------------------------------
# 1) STEP 1 of SATA: sub-aperture sizing + frequency bins + azimuth mapping
# ---------------------------------------------------------------------------
@dataclass
class SubapertureGrid:
    """Everything STEP 1 of the kernel computes, exposed so it can be checked."""
    deltax: float          # optimum sub-aperture ground extent [m]
    prf_data: float        # sampling rate of the data handed in [Hz]
    Tsubeff: int           # sub-aperture length [samples of that data]
    Nzp: int               # zero-padded STFT length
    Tsub: int              # hop (50 % overlap) [samples]
    marg_az: int           # zero-padding margin [bins]
    fsub: np.ndarray       # [Nzp] absolute Doppler of each STFT bin [Hz]
    betasub: np.ndarray    # [Nzp] beam angle of each bin [rad]
    azpos: np.ndarray      # [Nzp] azimuth offset of each bin [data samples]
    squint: float          # sub-band centre angle beta_k [rad]

    @property
    def band(self) -> float:
        return float(self.fsub.max() - self.fsub.min())

    @property
    def ground_extent(self) -> float:
        """Ground length actually covered by one sub-aperture [m]."""
        return self.Tsubeff / self.prf_data * self._v

    _v: float = 0.0

    def report(self, indent="  ") -> str:
        i = indent
        return (
            f"{i}deltax (target topo. resolution) : {self.deltax:10.2f} m\n"
            f"{i}data sampling rate prf_data      : {self.prf_data:10.2f} Hz\n"
            f"{i}sub-aperture Tsubeff             : {self.Tsubeff:10d} samples"
            f"  ({self.ground_extent:.2f} m on the ground)\n"
            f"{i}zero-padded Nzp / margin         : {self.Nzp:10d} / {self.marg_az}\n"
            f"{i}hop Tsub (50 % overlap)          : {self.Tsub:10d} samples\n"
            f"{i}STFT Doppler bin spacing         : {self.prf_data/self.Nzp:10.3f} Hz\n"
            f"{i}fsub span                        : {self.fsub.min():10.1f} .."
            f" {self.fsub.max():9.1f} Hz   (band = {self.band:.1f} Hz)\n"
            f"{i}betasub span                     : {np.degrees(self.betasub.min()):10.4f} .."
            f" {np.degrees(self.betasub.max()):9.4f} deg\n"
            f"{i}azpos span                       : {self.azpos.min():10.1f} .."
            f" {self.azpos.max():9.1f} samples\n"
        )


def _common_grid(deltax, prf_data, Tsubeff, Nzp, fsub, v, wl, r, squint,
                 marg_az, squint_image=None) -> SubapertureGrid:
    """``squint_image`` is the squint of the IMAGE GRID that ``delta_C0_array``
    is defined on.  In classic SATA the image is processed at the same squint as
    the beam, so it equals ``squint``; for the sub-band case the image is the
    ordinary broadside image and it must be 0 (see BUG 3)."""
    if squint_image is None:
        squint_image = squint
    betasub = np.arcsin(np.clip(wl * fsub / (2.0 * v), -1.0, 1.0))
    azpos = r * (np.tan(betasub) - np.tan(squint_image)) / v * prf_data
    g = SubapertureGrid(deltax, prf_data, Tsubeff, Nzp, int(Tsubeff * 0.5),
                        marg_az, fsub, betasub, azpos, squint)
    g._v = v
    return g


def subaperture_grid(rref, prf_data, v, wl, r, f_centre=0.0,
                     sata_osf=1, squint_image=0.0) -> SubapertureGrid:
    """
    CORRECTED STEP 1.

    ``prf_data`` : sampling rate of the azimuth line handed to the kernel
                   (for a reconstruction channel: ``PRF_op = prf/Nrx``).
    ``f_centre`` : absolute Doppler centre of the sub-band being processed
                   (``f_k``; 0 for the plain whole-band case).
    ``squint_image`` : squint of the image grid ``delta_C0_array`` lives on.
                   0 for the sub-band reconstruction (the reconstructed image is
                   broadside); set it to ``asin(wl*f_centre/2v)`` to reproduce
                   classic squint-processed SATA.

    The sub-band is ``prf_data`` wide by construction -- it is exactly the band
    a channel line can represent -- so the STFT bins span ``prf_data`` centred
    on ``f_centre``, and ``azpos`` is expressed in samples of that same line.
    """
    deltax = np.sqrt(wl * rref / 2.0)
    Tsubeff = max(2, int(np.round(deltax * prf_data / v * 0.5) * 2))
    Nzp = int(2 ** np.ceil(np.log2(Tsubeff)) * sata_osf)
    marg_az = int(np.round(0.5 * (1 + Nzp - Tsubeff)))

    # Sub-band centre, rounded onto the STFT bin grid, and its alias position
    # inside the (aliased) channel spectrum -- that is what aligns fsub with
    # the ordering of np.fft.fft(data).
    dfc = np.round(f_centre * Nzp / prf_data) * prf_data / Nzp
    pfc = np.round(np.mod(f_centre, prf_data) * Nzp / prf_data)

    fsub = np.arange(Nzp) * prf_data / Nzp - prf_data * 0.5 + dfc
    fsub[0] = prf_data * 0.5 + dfc          # wrap-around sample
    fsub = np.roll(fsub, int(Nzp / 2 + pfc))

    squint = float(np.arcsin(np.clip(wl * f_centre / (2.0 * v), -1.0, 1.0)))
    # [FIX C] azpos is the offset w.r.t. the image grid, which for the sub-band
    # reconstruction is the ordinary BROADSIDE image -> squint_image = 0.
    return _common_grid(deltax, prf_data, Tsubeff, Nzp, fsub, v, wl, r,
                        squint, marg_az, squint_image=squint_image)


def legacy_subaperture_grid(rref, prf, Nsb, v, wl, r, squint=0.0,
                            sata_osf=1) -> SubapertureGrid:
    """
    STEP 1 exactly as in ``SATA1D.txt`` / the current ``sar_recon.sata`` port,
    kept only so the diagnostics can show the difference.  Contains BUG 1 and
    BUG 2 described in the module docstring.
    """
    deltax = np.sqrt(wl * rref / 2.0)
    Tsubeff = max(2, int(np.round(deltax * prf / v * 0.5 / Nsb) * 2))   # BUG 1
    Nzp = int(2 ** np.ceil(np.log2(Tsubeff)) * sata_osf)
    marg_az = int(np.round(0.5 * (1 + Nzp - Tsubeff)))

    fc = 2.0 * v / wl * np.sin(squint)
    pfc = np.round(np.mod(fc, prf) * Nzp / prf)
    dfc = np.round(fc * Nzp / prf) * prf / Nzp
    fsub = Nsb * (np.arange(Nzp) * prf / Nzp / Nsb - prf * 0.5 / Nsb) + dfc  # BUG 2
    fsub[0] = prf * 0.5 + dfc
    fsub = np.roll(fsub, int(Nzp / 2 + pfc))

    g = _common_grid(deltax, prf, Tsubeff, Nzp, fsub, v, wl, r, squint, marg_az)
    return g


# ---------------------------------------------------------------------------
# 2) The SATA kernel (weighted overlap-add over sub-apertures)
# ---------------------------------------------------------------------------
def _run_kernel(data, delta_C0_array, g: SubapertureGrid, wl,
                inverse=False, verbose=False):
    data = np.asarray(data, dtype=complex).copy()
    dimx = len(data)
    if g.Tsubeff <= 2:
        if verbose:
            print("SATA: sub-aperture length very small, no correction needed")
        return data
    if g.Tsubeff > dimx:
        raise ValueError(f"SATA: sub-aperture ({g.Tsubeff}) longer than the "
                         f"azimuth line ({dimx}).")

    L, hop, Nzp = g.Tsubeff, g.Tsub, g.Nzp
    # Triangular analysis window: COLA at 50 % overlap, so delta_C0 == 0
    # reproduces the input to machine precision.
    ramp = np.arange(hop) / (hop - 1)
    win = np.concatenate((ramp, ramp[::-1]))[:L]

    out = np.zeros(dimx, dtype=complex)
    wsum = np.zeros(dimx, dtype=float)
    max_ph = 0.0

    for start in range(0, dimx, hop):
        seg = data[start:start + L]
        Lseg = len(seg)
        if Lseg < 2:
            break
        buf = np.zeros(Nzp, dtype=complex)
        buf[:Lseg] = seg * win[:Lseg]
        spec = np.fft.fft(buf)                       # <- the STFT

        # Each Doppler bin sees a different azimuth position: the centre of the
        # sub-aperture, plus the offset its beam angle implies.
        centre = start + 0.5 * L
        posaux = np.clip(np.round(g.azpos + centre).astype(int), 0, dimx - 1)

        ph = -2.0 * PI / wl * delta_C0_array[posaux]
        ph[~np.isfinite(ph)] = 0.0
        max_ph = max(max_ph, float(np.max(np.abs(ph))))

        spec *= np.exp(-1j * ph) if inverse else np.exp(1j * ph)
        out[start:start + Lseg] += np.fft.ifft(spec)[:Lseg]
        wsum[start:start + Lseg] += win[:Lseg]

    nz = wsum > 1e-12
    out[nz] /= wsum[nz]
    out[~nz] = data[~nz]                             # untouched edges

    if verbose:
        print(f"SATA: max peak-to-peak phase correction "
              f"{np.degrees(2*max_ph):.1f} deg")
    return out


def sata_1d(data, delta_C0_array, rref, prf_data, v, wl, r, f_centre=0.0,
            inverse=False, sata_osf=1, squint_image=0.0, verbose=False):
    """
    CORRECTED SATA. Apply the C0 (bulk-range) topography correction to one
    azimuth line, conditioned on the frequency beam of one sub-band.

    ``data``           [dimx] complex, one channel line at ``prf_data``.
    ``delta_C0_array`` [dimx] residual slant range [m] vs azimuth image
                       position, on the SAME grid as ``data``.
    ``f_centre``       absolute Doppler centre of the sub-band (``f_k``).
    ``inverse``        True REMOVES the residual, False injects it.
    """
    g = subaperture_grid(rref, prf_data, v, wl, r, f_centre, sata_osf,
                         squint_image=squint_image)
    if verbose:
        print(g.report())
    return _run_kernel(data, delta_C0_array, g, wl, inverse, verbose)


def legacy_sata_1d(data, delta_C0_array, rref, prf, Nsb, v, squint, wl, r,
                   inverse=False, sata_osf=1, verbose=False):
    """Buggy reference behaviour, for A/B comparison only."""
    g = legacy_subaperture_grid(rref, prf, Nsb, v, wl, r, squint, sata_osf)
    if verbose:
        print(g.report())
    return _run_kernel(data, delta_C0_array, g, wl, inverse, verbose)


# ---------------------------------------------------------------------------
# 3) Sub-band bookkeeping
# ---------------------------------------------------------------------------
def subband_centre_frequency(k: int, Nrx: int, prf_full: float) -> float:
    """Absolute Doppler centre ``f_k`` of output sub-band ``k``."""
    return (-Nrx / 2.0 + k + 0.5) * (prf_full / Nrx)


def build_delta_C0_array(p, offsets, channel: int, naz: int,
                         r_assumed: float | None = None,
                         pad_zero_outside: bool = False) -> np.ndarray:
    """
    Residual C0 [m] as a function of azimuth image position, on the
    per-channel grid (``naz = Na_ch`` samples at ``PRF_op``).

    ``offsets`` : list of ``(dx_az, dr)`` -- along-track position [m] of a
                  scatterer w.r.t. the scene centre, and the slant-range error
                  [m] the flat-earth reconstruction makes there.  In a 3-D
                  geometry ``dr`` follows from the height error and the
                  cross-track baseline; in this 2-D linear-orbit model it is
                  given directly.
    ``channel`` : receiver index (channel 0 is monostatic -> residual is 0).

    The residual is ``C0(r_assumed + dr) - C0(r_assumed)`` evaluated with the
    same :func:`get_coeff_nu` the reconstruction filter uses, so correction and
    error come from one consistent model.
    """
    if r_assumed is None:
        r_assumed = p.r_ref
    if not offsets:
        return np.zeros(naz)

    dxc = p.deltaX[channel]
    C0_ref = get_coeff_nu(dxc, r_assumed, p.ve, p.int_time, p.prf, p.wl)[0]
    ds = p.ve / p.PRF_op                       # azimuth sample spacing [m]

    xs, ys = [], []
    for (dx_az, dr) in offsets:
        C0_real = get_coeff_nu(dxc, r_assumed + dr, p.ve, p.int_time, p.prf, p.wl)[0]
        xs.append(naz / 2 + dx_az / ds)
        ys.append(C0_real - C0_ref)
    order = np.argsort(xs)
    xs, ys = np.array(xs)[order], np.array(ys)[order]

    grid = np.arange(naz)
    if xs.size == 1:
        if pad_zero_outside:
            arr = np.zeros(naz)
            arr[int(np.clip(xs[0], 0, naz - 1))] = ys[0]
            return arr
        return np.full(naz, ys[0])
    left = 0.0 if pad_zero_outside else ys[0]
    right = 0.0 if pad_zero_outside else ys[-1]
    return np.interp(grid, xs, ys, left=left, right=right)


def sata_channels_subband(p, s_channel, offsets, k: int, remove=True,
                          sata_osf=4, range_bins=None, legacy=False,
                          verbose=False) -> np.ndarray:
    """
    SATA-condition every channel for output sub-band ``k``.

    ``s_channel`` [Nrx, Na_ch, Nr] range-compressed channel cube (azimuth TIME
    x range TIME).  Returns a corrected copy; feed it into the reconstruction
    of sub-band ``k`` only.  ``legacy=True`` reproduces the buggy call.

    ``range_bins`` restricts the correction to a subset of range bins
    (``None`` = all).
    """
    Nrx, Na_ch, Nr = s_channel.shape
    out = np.array(s_channel, dtype=complex, copy=True)
    if not offsets:
        return out

    f_k = subband_centre_frequency(k, Nrx, p.prf)
    beta_k = float(np.arcsin(np.clip(p.wl * f_k / (2.0 * p.ve), -1.0, 1.0)))
    if verbose:
        print(f"sub-band {k}: f_k = {f_k:9.1f} Hz -> beta_k = "
              f"{np.degrees(beta_k):8.4f} deg")

    rb = range(Nr) if range_bins is None else range_bins
    for kk in range(Nrx):
        dC0 = build_delta_C0_array(p, offsets, kk, naz=Na_ch)
        for nn in rb:
            if legacy:
                out[kk, :, nn] = legacy_sata_1d(
                    out[kk, :, nn], dC0, rref=p.r_ref, prf=p.PRF_op, Nsb=Nrx,
                    v=p.ve, squint=beta_k, wl=p.wl, r=p.r_ref,
                    inverse=remove, sata_osf=sata_osf)
            else:
                out[kk, :, nn] = sata_1d(
                    out[kk, :, nn], dC0, rref=p.r_ref, prf_data=p.PRF_op,
                    v=p.ve, wl=p.wl, r=p.r_ref, f_centre=f_k,
                    inverse=remove, sata_osf=sata_osf)
    return out
