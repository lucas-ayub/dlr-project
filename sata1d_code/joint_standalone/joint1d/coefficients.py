"""Coefficients of the reference point (C0, C1, C2, Dt per channel) and delta_C0 of a point at another height.
Copied without changes from sata1d/joint.py (reference_coefficients) and sata1d/sata.py (residual_C0)."""
from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .geometry import PlatformTracks
from .reconstruction import GetCoeffNu


def reference_coefficients(cfg: ExperimentConfig, tracks: PlatformTracks):
    """(C0, C1, C2, Dt) of the reference point for every channel, as filter_matrix uses them."""
    out = np.zeros((cfg.Nrx, 4))
    for i in range(cfg.Nrx):
        out[i] = GetCoeffNu(cfg.scene.ptg, tracks.ptx, tracks.prx[i], tracks.vtx, tracks.vrx[i],
                            tracks.ptx, tracks.vtx, cfg.prf, cfg.system.wl, cfg.ta,
                            cfg.sq_tx, cfg.sq_rx[i], cfg.theta_tx, cfg.theta_rx[i])
    return out


def residual_C0(cfg: ExperimentConfig, tracks: PlatformTracks, ptg_real, channel):
    """C0(scatterer) - C0(reference point) for one channel [m]."""
    common = (tracks.ptx, tracks.prx[channel], tracks.vtx, tracks.vrx[channel],
              tracks.ptx, tracks.vtx, cfg.prf, cfg.system.wl, cfg.ta,
              cfg.sq_tx, cfg.sq_rx[channel], cfg.theta_tx, cfg.theta_rx[channel])
    return float(GetCoeffNu(ptg_real, *common)[0] - GetCoeffNu(cfg.scene.ptg, *common)[0])
