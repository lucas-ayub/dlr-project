"""Point-target raw data (range-compressed azimuth line) and channel generation."""
from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .geometry import PlatformTracks


def getRawData1D(ptgs, ptx, prx, vtx, vrx, ta, sq_tx, sq_rx, theta_tx, theta_rx, wl, prf, amps=None):
    """Sum of amps[p] exp(-j 2 pi (|ptx-p| + |prx-p|) / wl) over the scatterers ptgs [Np, 3],
    each weighted by its TX and RX antenna illumination window (amps = 1 if not given)."""
    Na = len(ta)
    inst_sq_tx = np.zeros(Na)
    inst_sq_rx = np.zeros(Na)
    datal = np.zeros(Na, np.complex128)
    amps = np.ones(len(ptgs), np.complex128) if amps is None else np.asarray(amps, np.complex128)
    for p, amp in zip(ptgs, amps):
        wa_tx = np.zeros(Na)
        wa_rx = np.zeros(Na)
        rh_ms = np.sqrt(np.sum((ptx - p[None, :]) ** 2, axis=1))
        inst_sq_tx[:Na - 1] = np.arcsin(np.diff(rh_ms) * prf / vtx[1:])
        inst_sq_tx[Na - 1] = 2 * inst_sq_tx[Na - 2] - inst_sq_tx[Na - 3]
        wa_tx[np.abs(inst_sq_tx) <= sq_tx + theta_tx / 2] = 1
        rh_bs = np.sqrt(np.sum((prx - p[None, :]) ** 2, axis=1))
        inst_sq_rx[:Na - 1] = np.arcsin(np.diff(rh_bs) * prf / vrx[1:])
        inst_sq_rx[Na - 1] = 2 * inst_sq_rx[Na - 2] - inst_sq_tx[Na - 3]
        wa_rx[np.abs(inst_sq_rx) <= sq_rx + theta_rx / 2] = 1
        datal += amp * wa_tx * wa_rx * np.exp(-2j * np.pi * (rh_ms + rh_bs) / wl)
    return datal


def raw(cfg: ExperimentConfig, tracks: PlatformTracks, ptgs, prx, vrx, amps=None):
    """getRawData1D with the configuration's time axis, beam and PRF."""
    return getRawData1D(ptgs, tracks.ptx, prx, tracks.vtx, vrx, cfg.ta, cfg.sq_tx, cfg.sq_tx,
                        cfg.theta_tx, cfg.theta_tx, cfg.system.wl, cfg.prf, amps)


def _scene(cfg, ptgs, amps):
    """The scene's scatterers and amplitudes unless explicit ones are given."""
    if ptgs is None:
        return cfg.scene.points[1:], cfg.scene.amps
    return ptgs, amps


def generate_reference(cfg: ExperimentConfig, tracks: PlatformTracks, ptgs=None, amps=None):
    """Monostatic full-PRF signal of the scatterers (ideal reference)."""
    ptgs, amps = _scene(cfg, ptgs, amps)
    return raw(cfg, tracks, ptgs, tracks.ptx, tracks.vtx, amps)


def generate_channels(cfg: ExperimentConfig, tracks: PlatformTracks, ptgs=None, amps=None):
    """Bistatic channels [Nrx, Na_ch], each subsampled by Nrx (per-channel PRF)."""
    ptgs, amps = _scene(cfg, ptgs, amps)
    return np.array([raw(cfg, tracks, ptgs, tracks.prx[i], tracks.vrx[i], amps)[::cfg.Nrx]
                     for i in range(cfg.Nrx)])
