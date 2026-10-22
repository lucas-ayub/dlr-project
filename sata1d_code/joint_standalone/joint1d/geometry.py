"""Transmitter / receiver tracks: TX at (vs t, 0, H), RX i at (vs t - bat[i], bxt[i], H)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig


@dataclass
class PlatformTracks:
    ptx: np.ndarray   # [Na, 3]
    vtx: np.ndarray   # [Na]
    prx: np.ndarray   # [Nrx, Na, 3]
    vrx: np.ndarray   # [Nrx, Na]


def build_platform_tracks(cfg: ExperimentConfig) -> PlatformTracks:
    vs, H, ta, Na, Nrx = cfg.system.vs, cfg.scene.H, cfg.ta, cfg.Na, cfg.Nrx
    ptx = np.column_stack([vs * ta, np.zeros(Na), H * np.ones(Na)])
    vtx = vs * np.ones(Na)
    prx = np.zeros([Nrx, Na, 3])
    vrx = np.zeros([Nrx, Na])
    for i in range(Nrx):
        prx[i, :, 0] = vs * ta - cfg.array.bat[i]
        prx[i, :, 1] = cfg.array.bxt[i]
        prx[i, :, 2] = H
        vrx[i, :] = vs
    return PlatformTracks(ptx=ptx, vtx=vtx, prx=prx, vrx=vrx)
