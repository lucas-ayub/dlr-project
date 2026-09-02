# -*- coding: utf-8 -*-
"""
Platform tracks for the 2-D linear-orbit geometry.

Coordinates are (x, y) = (ground range, azimuth); the orbit is a straight line
along +y at constant speed ``ve``.  The transmitter is at ``y = ve*t``; receiver
``i`` trails it by ``deltaX[i]`` along-track.  There is no cross-track baseline
and no height in this model -- exactly the case the reference files cover.
"""
from __future__ import annotations

import numpy as np

__all__ = ["build_tracks"]


def build_tracks(p, ta=None):
    """
    Returns ``(ptx, prx, vtx, vrx)``:

    ``ptx`` [2, Na]        transmitter track
    ``prx`` [Nrx, 2, Na]   receiver tracks
    ``vtx`` [Na]           transmitter speed
    ``vrx`` [Nrx, Na]      receiver speeds
    """
    if ta is None:
        ta = p.ta
    Na = len(ta)
    Nrx = p.Nrx

    ptx = np.zeros([2, Na], np.float64)
    ptx[1, :] = p.ve * ta

    prx = np.zeros([Nrx, 2, Na], np.float64)
    for kk in range(Nrx):
        prx[kk, 1, :] = p.ve * ta - p.deltaX[kk]

    vtx = np.full(Na, float(p.ve))
    vrx = np.full([Nrx, Na], float(p.ve))
    return ptx, prx, vtx, vrx
