# -*- coding: utf-8 -*-
"""
Range compression -- replacement for the DLR-internal ``sarProccessing.mp_rc``.

The generator emits a linear FM down-chirp

    s(t) = exp(-j*pi*(rbw/cd)*(t - t0 - cd/2)^2)

so the matched filter in the range-frequency domain is

    H(f) = exp(-j*pi*f^2*cd/rbw) * exp(+j*pi*f*cd)
         = exp(-1j*pi*f*cd*(f/rbw - 1))

which is exactly the ``ph_rc`` expression used in the reference ``MP2DATBF.py``
(the second factor removes the cd/2 delay of the chirp centre, i.e. it puts the
compressed peak at the true target delay).
"""
from __future__ import annotations

import numpy as np

__all__ = ["range_matched_filter", "range_compress"]


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
