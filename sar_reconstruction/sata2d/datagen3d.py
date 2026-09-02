# -*- coding: utf-8 -*-
"""
2-D raw-data generator for the 3-D geometry (along- + cross-track baselines,
targets at arbitrary height).

Same signal model as `datagen2d.py`, written for 3-D tracks and vectorised over
the range axis.  For one scatterer p and one TX/RX pair the range history is

    rh(t) = |p_tx(t) - p| + |p_rx(t) - p|                                  (1)

and the recorded echo is a linear FM chirp delayed by rh/c and carrying the
two-way carrier phase:

    s(t, tau) = w(t) * rect[(tau - rh/c)/T_p]
                * exp{ -j pi (B_r/T_p) (tau - rh/c - T_p/2)^2 }
                * exp{ -2 pi j rh(t) / wl }                                (2)

where tau is fast time, T_p = ``cd``, B_r = ``rbw``, and w(t) is the two-way
antenna window: a sample is kept when BOTH the transmit and the receive
instantaneous squint fall inside the beam,

    |arcsin( (d r / dt) / v )| <= sq + theta/2 .                           (3)

This is exactly the window used by `sar_recon.signal_model.getRawData1D`, so
the 2-D results here are comparable with the existing 1-D SATA tests.

Vectorisation
-------------
`datagen2d.get_echo` builds one range line per azimuth sample in a Python loop.
Here the whole [Na, Nr] block is built at once by broadcasting fast time
against the range history; the two are numerically equivalent (the rect window
replaces the explicit start/end indices).
"""
from __future__ import annotations

import numpy as np

C_LIGHT = 299792458.0
PI = np.pi

__all__ = ["range_history", "antenna_window", "echo_block",
           "get_raw_data_3d", "generate_reference_3d", "generate_channels_3d"]


# ---------------------------------------------------------------------------
def range_history(ptg, ps_tx, ps_rx):
    """Eq. (1): |p_tx - p| + |p_rx - p| for tracks given as [Na, 3]."""
    a = np.sqrt(np.sum((ps_tx - ptg[None, :]) ** 2, axis=1))
    b = np.sqrt(np.sum((ps_rx - ptg[None, :]) ** 2, axis=1))
    return a + b


def antenna_window(ptg, ps, v, prf, sq, theta):
    """Eq. (3): boolean beam window of one leg."""
    rh = np.sqrt(np.sum((ps - ptg[None, :]) ** 2, axis=1))
    inst_sq = np.arcsin(np.clip(np.gradient(rh, 1.0 / prf) / v, -1.0, 1.0))
    return np.abs(inst_sq) <= (sq + theta / 2.0)


def echo_block(R, w, r_min, Nr, rbw, cd, rsf, wl, out=None):
    """
    Eq. (2), vectorised: returns (or accumulates into ``out``) the [Na, Nr]
    echo of one scatterer.  ``R`` is HALF the two-way path (so the delay is
    2R/c), ``w`` the azimuth window.
    """
    Na = len(R)
    fast_time = 2 * r_min / C_LIGHT + np.arange(Nr) / rsf          # [Nr]
    tau = fast_time[None, :] - 2.0 * R[:, None] / C_LIGHT          # [Na, Nr]
    inside = (tau >= 0.0) & (tau <= cd)
    ph = (-1j * PI * rbw / cd * (tau - 0.5 * cd) ** 2
          - 4j * PI / wl * R[:, None])
    blk = np.where(inside, np.exp(ph), 0.0) * w[:, None]
    if out is None:
        return blk.astype(np.complex64)
    out += blk.astype(np.complex64)
    return out


# ---------------------------------------------------------------------------
def get_raw_data_3d(p, ptgs, ps_tx, ps_rx, v_tx, v_rx, ta, prf,
                    sigma=None) -> np.ndarray:
    """
    [Na, Nr] raw data of a TX/RX pair for all scatterers in ``ptgs`` [Np, 3].

    ``prf`` is the sampling rate of ``ta`` (full PRF for the reference,
    ``PRF_op`` for one decimated channel).  Pass ``ps_rx = ps_tx`` and
    ``v_rx = v_tx`` for the monostatic reference.
    """
    Na, Nr = len(ta), p.Nr
    ptgs = np.atleast_2d(ptgs)
    if sigma is None:
        sigma = np.ones(len(ptgs))
    data = np.zeros([Na, Nr], np.complex64)

    for j, ptg in enumerate(ptgs):
        w_tx = antenna_window(ptg, ps_tx, v_tx, prf, p.sq_tx, p.theta_tx)
        w_rx = antenna_window(ptg, ps_rx, v_rx, prf, p.sq_tx, p.theta_tx)
        w = (w_tx & w_rx).astype(np.float64)
        if not w.any():
            continue
        R = 0.5 * range_history(ptg, ps_tx, ps_rx)
        echo_block(R, sigma[j] * w, p.r_scan[0], Nr, p.rbw, p.cd, p.rsf,
                   p.wl, out=data)
    return data


def generate_reference_3d(p, tracks, ptgs=None) -> np.ndarray:
    """Ideal monostatic signal at the FULL PRF -- the comparison reference."""
    ptx, prx, vtx, vrx = tracks
    if ptgs is None:
        ptgs = p.points
    return get_raw_data_3d(p, ptgs, ptx, ptx, vtx, vtx, p.ta, p.prf)


def generate_channels_3d(p, tracks, ptgs=None) -> np.ndarray:
    """
    Per-channel bistatic signals on the decimated azimuth axis,
    [Nrx, Na_ch, Nr].  Channel ``i`` uses TX at ``ptx`` and RX at ``prx[i]``,
    then keeps every ``Nrx``-th azimuth sample -- the physical statement that
    each channel is read out once per transmitted pulse group.
    """
    ptx, prx, vtx, vrx = tracks
    if ptgs is None:
        ptgs = p.points
    idx = p.idx_dec
    out = np.zeros([p.Nrx, p.Na_ch, p.Nr], np.complex64)
    for i in range(p.Nrx):
        out[i] = get_raw_data_3d(
            p, ptgs, ptx[idx], prx[i][idx], vtx[idx], vrx[i][idx],
            p.ta[idx], p.PRF_op)
    return out
