# -*- coding: utf-8 -*-
"""
Platform track generation.

Turns an ExperimentConfig (system + scene + array + time axis) into the
transmitter and receiver position/velocity arrays consumed by the signal model
and the reconstruction.

    Transmitter:  ptx(t) = (vs*t,           0,        H)
    Receiver i:   prx(t) = (vs*t - bat[i],  bxt[i],   H)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig


@dataclass
class PlatformTracks:
    ptx: np.ndarray   # [Na, 3]        transmitter positions
    vtx: np.ndarray   # [Na]           transmitter speed (scalar per sample)
    prx: np.ndarray   # [Nrx, Na, 3]   receiver positions
    vrx: np.ndarray   # [Nrx, Na]      receiver speeds


def build_platform_tracks(cfg: ExperimentConfig) -> PlatformTracks:
    vs = cfg.system.vs
    H = cfg.scene.H
    ta = cfg.ta
    Na = cfg.Na
    Nrx = cfg.Nrx

    ptx = np.column_stack([vs * ta, np.zeros(Na), H * np.ones(Na)])
    vtx = vs * np.ones(Na)

    prx = np.zeros([Nrx, Na, 3], np.float64)
    vrx = np.zeros([Nrx, Na], np.float64)
    for jj in range(Nrx):
        prx[jj, :, 0] = vs * ta - cfg.array.bat[jj]
        prx[jj, :, 1] = cfg.array.bxt[jj]
        prx[jj, :, 2] = H
        vrx[jj, :] = vs

    return PlatformTracks(ptx=ptx, vtx=vtx, prx=prx, vrx=vrx)
