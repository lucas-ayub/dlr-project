# -*- coding: utf-8 -*-
"""
Acquisition parameters for the 2-D (range + azimuth) two-step reconstruction.

This is the parameter block at the top of the reference ``MP2DATBF.py``, pulled
out into a dataclass so that the same numbers drive the pipeline, the SATA
diagnostics and the tests.

Two presets are provided:

``"small"``   (default)  -- same physics, reduced bandwidth / aperture so that a
                            full 2-D run finishes in ~1 minute on a laptop and
                            fits in RAM.  Use this while debugging.
``"paper"``              -- the numbers of the reference script (P/X-band, 322
                            MHz, 10 km swath).  Na x Nr ~ 90112 x 32768, i.e.
                            ~24 TB in memory -- it is ONLY usable through the
                            HDF5 out-of-core path and takes hours.

Geometry (linear orbit, the case the reference files cover)
-----------------------------------------------------------
* One transmitter moving along +y at constant speed ``ve``.
* ``Nrx`` receivers on the SAME straight track, displaced along-track by
  ``deltaX[i]`` (so ``deltaX[0] = 0`` is the monostatic channel).
* Every receiver is sampled at ``PRF_op = prf / Nrx``; the union of the Nrx
  channels is an equivalent single channel sampled at ``prf``.
* No cross-track baseline and no topography in the reference generator -- that
  is what the SATA extension in :mod:`sata2d` adds on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C0_LIGHT = 3.0e8
MU_EARTH = 3.9860e14
R_EARTH = 6378.0e3


@dataclass
class Params:
    # ---- system -----------------------------------------------------------
    Nrx: int                 # number of receive channels
    wl: float                # wavelength [m]
    ve: float                # platform velocity [m/s]
    La: float                # transmit antenna length [m]
    theta_tx: float          # transmit beamwidth [rad]
    theta_rx: np.ndarray     # receive beamwidths [rad]
    sq_tx: float             # transmit squint [rad]
    sq_rx: np.ndarray        # receive squints [rad]
    abw: float               # azimuth (Doppler) bandwidth [Hz]
    prf: float               # EQUIVALENT (post-reconstruction) PRF [Hz]
    alpha: float             # oversampling factor

    # ---- range ------------------------------------------------------------
    rbw: float               # chirp bandwidth [Hz]
    rsf: float               # range sampling frequency [Hz]
    cd: float                # chirp duration [s]
    Rmin: float              # near slant range [m]
    swath: float             # slant-range swath [m]
    Nr: int                  # range samples (power of two)
    r_scan: np.ndarray       # slant range of every range bin [m]

    # ---- azimuth ----------------------------------------------------------
    int_time: float          # integration (synthetic aperture) time [s]
    acq_time: float          # simulated acquisition time [s]
    Na: int                  # azimuth samples at the equivalent PRF
    Na_ch: int               # azimuth samples per channel = Na / Nrx
    ta: np.ndarray           # azimuth (slow) time axis [s], centred on 0
    deltaX: np.ndarray       # along-track offset of each receiver [m]

    # ---- scene ------------------------------------------------------------
    r_ref: float             # slant range of the reference point target [m]
    tgt_az: float            # azimuth position of the point target [m]

    preset: str = "small"

    # ---- derived ----------------------------------------------------------
    @property
    def PRF_op(self) -> float:
        """Operating (per-channel) PRF."""
        return self.prf / self.Nrx

    @property
    def f0(self) -> float:
        return C0_LIGHT / self.wl

    @property
    def fr(self) -> np.ndarray:
        """Range-frequency axis in *unshifted* FFT order (matches np.fft.fft)."""
        return np.roll(np.arange(self.Nr) * self.rsf / self.Nr - 0.5 * self.rsf,
                       int(0.5 * self.Nr))

    @property
    def idx_dec(self) -> np.ndarray:
        """Indices that decimate the full-PRF azimuth axis onto one channel."""
        return (np.arange(self.Na_ch) * self.Nrx).astype(int)

    def summary(self) -> str:
        return (
            f"preset      : {self.preset}\n"
            f"Nrx         : {self.Nrx}\n"
            f"wavelength  : {self.wl*100:.2f} cm  (f0 = {self.f0/1e9:.3f} GHz)\n"
            f"ve          : {self.ve:.1f} m/s\n"
            f"theta_tx    : {np.degrees(self.theta_tx):.4f} deg\n"
            f"abw / prf   : {self.abw:.1f} Hz / {self.prf:.1f} Hz "
            f"(PRF_op = {self.PRF_op:.1f} Hz)\n"
            f"rbw / rsf   : {self.rbw/1e6:.2f} MHz / {self.rsf/1e6:.2f} MHz\n"
            f"Na x Nr     : {self.Na} x {self.Nr}   (Na_ch = {self.Na_ch})\n"
            f"int_time    : {self.int_time:.3f} s  (acq {self.acq_time:.3f} s)\n"
            f"range swath : {self.r_scan[0]/1e3:.3f} .. {self.r_scan[-1]/1e3:.3f} km\n"
            f"target      : r = {self.r_ref/1e3:.3f} km, az = {self.tgt_az:.1f} m\n"
            f"deltaX      : {np.array2string(self.deltaX, precision=2)}\n"
        )


def make_params(preset: str = "small", Nrx: int = 2) -> Params:
    """
    Build the parameter set.  The chain of dependencies is exactly the one at
    the top of the reference ``MP2DATBF.py``; only the three "size knobs"
    (``res_az_lambda``, ``swath``, ``divfac``) differ between presets.
    """
    if preset == "small":
        res_az_lambda = 100.0   # azimuth resolution in wavelengths -> antenna len
        swath = 1.0e3           # slant-range swath [m]
        divfac = 128            # Na is forced to a multiple of Nrx*divfac
        acq_factor = 1.5
    elif preset == "paper":
        res_az_lambda = 15.0
        swath = 10.0e3
        divfac = 4096
        acq_factor = 1.5
    else:
        raise ValueError(f"unknown preset {preset!r} (use 'small' or 'paper')")

    c = C0_LIGHT
    wl = 0.031                              # X-band
    alpha = 0.15                            # PRF oversampling factor

    # ---- antenna / Doppler -------------------------------------------------
    res_az = res_az_lambda * wl             # target azimuth resolution [m]
    La = 2.0 * res_az                       # transmit antenna length [m]
    theta_tx = wl / La                      # transmit beamwidth [rad]
    La_rx = La / Nrx                        # receive sub-antenna length [m]
    theta_rx = np.full(Nrx, wl / La_rx)     # receive beamwidth [rad]
    sq_tx = 0.0                             # broadside (linear orbit case)
    sq_rx = np.zeros(Nrx)

    # ---- orbit -------------------------------------------------------------
    Rmin = 700.0e3
    ve = float(int(np.sqrt(MU_EARTH * R_EARTH) / (Rmin + R_EARTH)))
    abw = ve * 2.0 / La                     # Doppler bandwidth [Hz]
    prf = (1.0 + alpha) * abw               # EQUIVALENT PRF (after reconstruction)

    # ---- range -------------------------------------------------------------
    res_rg = res_az
    cd = 1.0e-6                             # chirp duration [s]
    rbw = c / 2.0 / res_rg                  # chirp bandwidth [Hz]
    rsf = (1.0 + alpha) * rbw               # range sampling frequency [Hz]
    Nr = int(2 ** np.ceil(np.log2(((2 * swath / c) + cd) * rsf)))
    r_scan = Rmin + np.arange(Nr) / rsf * c * 0.5

    # ---- scene -------------------------------------------------------------
    # Reference target near the beginning of the swath (as in the reference
    # script) so that its range migration stays inside the simulated window.
    r_ref = float(r_scan[int(0.05 * Nr)])
    tgt_az = 0.0

    # ---- azimuth extent ----------------------------------------------------
    int_time = np.tan(theta_tx * 0.5) * r_ref * 2.0 / ve
    acq_time = int_time * acq_factor
    Na = int(np.ceil(acq_time * prf / Nrx / divfac) * Nrx * divfac)
    Na_ch = int(Na / Nrx)
    ta = (np.arange(Na) - 0.5 * Na) / prf

    # ---- along-track channel offsets --------------------------------------
    # Dx is the DPCA-optimum along-track spacing: the phase centre of channel i
    # sits half-way between TX and RX_i, so a displacement of deltaX/2 per
    # channel must equal ve / PRF_op / ... -> Dx = 2*ve/(prf/Nrx).
    deltaX = np.zeros(Nrx)
    Dx = 2.0 * ve / (prf / Nrx)
    dx_extra = 1.0e2                        # extra physical separation [m]
    deltaX[1:] = Dx * np.arange(1, Nrx) / Nrx + np.ceil(dx_extra / Dx) * Dx

    return Params(
        Nrx=Nrx, wl=wl, ve=ve, La=La, theta_tx=theta_tx, theta_rx=theta_rx,
        sq_tx=sq_tx, sq_rx=sq_rx, abw=abw, prf=prf, alpha=alpha,
        rbw=rbw, rsf=rsf, cd=cd, Rmin=Rmin, swath=swath, Nr=Nr, r_scan=r_scan,
        int_time=int_time, acq_time=acq_time, Na=Na, Na_ch=Na_ch, ta=ta,
        deltaX=deltaX, r_ref=r_ref, tgt_az=tgt_az, preset=preset,
    )


if __name__ == "__main__":
    print(make_params("small").summary())
