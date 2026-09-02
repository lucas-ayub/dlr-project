# -*- coding: utf-8 -*-
"""
Drop-in replacement for the DLR-internal ``SignalProcessingLibrary`` (imported
as ``sp`` in the reference code).

The reference code only ever uses two entry points::

    sp.fft2(x, nthreads, axis=...)
    sp.ifft2(x, nthreads, axis=...)

which are MKL-threaded 1-D FFTs along ``axis`` (the name ``fft2`` is historical
and does NOT mean "2-D FFT").  Here they are plain ``numpy.fft`` calls; the
thread-count argument is accepted and ignored so that the reference code can be
copied over verbatim.

Nothing else from the proprietary stack is needed by this package.
"""
from __future__ import annotations

import numpy as np

__all__ = ["fft2", "ifft2", "next_pow2"]


def fft2(x, nthreads: int = 1, axis: int = -1):
    """1-D FFT along ``axis``. ``nthreads`` is accepted for API parity only."""
    return np.fft.fft(x, axis=axis).astype(np.complex64)


def ifft2(x, nthreads: int = 1, axis: int = -1):
    """1-D inverse FFT along ``axis``. ``nthreads`` is ignored."""
    return np.fft.ifft(x, axis=axis).astype(np.complex64)


def next_pow2(n) -> int:
    """Smallest power of two >= n (``np.log10(x)/np.log10(2)`` in the original)."""
    return int(2 ** np.ceil(np.log2(n)))
