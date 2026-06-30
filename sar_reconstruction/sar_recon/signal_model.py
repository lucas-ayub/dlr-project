# -*- coding: utf-8 -*-
"""
Forward signal model.

getRawData1D is kept verbatim from the original pipeline. The two thin wrappers
generate_reference / generate_channels just feed it the right tracks and apply
the per-channel [::Nrx] subsampling.
"""
from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .geometry import PlatformTracks


def getRawData1D(ptgs, ptx, prx, vtx, vrx, ta, sq_tx, sq_rx,
                 theta_tx, theta_rx, wl, prf):
    Na = len(ta)
    Np = len(ptgs[:, 0])

    inst_sq_tx = np.zeros(Na, np.float64)
    inst_sq_rx = np.zeros(Na, np.float64)
    wa_tx = np.zeros(Na)
    wa_rx = np.zeros(Na)
    datal = np.zeros(Na, np.complex128)

    for p_idx in range(Np):
        rh_ms = np.sqrt(np.sum((ptx - ptgs[p_idx, :][np.newaxis, :]) ** 2, axis=1))
        inst_sq_tx[0:Na - 1] = np.arcsin(np.diff(rh_ms) * prf / vtx[1:Na])
        inst_sq_tx[Na - 1] = 2 * inst_sq_tx[Na - 2] - inst_sq_tx[Na - 3]
        wa_tx[np.where(abs(inst_sq_tx) <= (sq_tx + theta_tx / 2))] = 1

        rh_bs = np.sqrt(np.sum((prx - ptgs[p_idx, :][np.newaxis, :]) ** 2, axis=1))
        inst_sq_rx[0:Na - 1] = np.arcsin(np.diff(rh_bs) * prf / vrx[1:Na])
        inst_sq_rx[Na - 1] = 2 * inst_sq_rx[Na - 2] - inst_sq_tx[Na - 3]
        wa_rx[np.where(abs(inst_sq_rx) <= (sq_rx + theta_rx / 2))] = 1

        aPattern = wa_tx * wa_rx
        rh = rh_ms + rh_bs
        datal = aPattern * np.exp(-2j * np.pi * rh / wl)
    return datal


def generate_reference(cfg: ExperimentConfig, tracks: PlatformTracks) -> np.ndarray:
    """Monostatic reference signal (TX acting as both transmitter and receiver)."""
    ptg = cfg.scene.ptg[np.newaxis, :]
    return getRawData1D(
        ptg, tracks.ptx, tracks.ptx, tracks.vtx, tracks.vtx, cfg.ta,
        cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx, cfg.system.wl, cfg.prf,
    )


def generate_channels(cfg: ExperimentConfig, tracks: PlatformTracks) -> np.ndarray:
    """Per-receiver bistatic channels, subsampled by Nrx -> shape [Nrx, Na_ch]."""
    Nrx = cfg.Nrx
    ptg = cfg.scene.ptg[np.newaxis, :]
    s_channel = np.zeros([Nrx, cfg.Na_ch], dtype=np.complex128)
    for ii in range(Nrx):
        s_channel[ii, :] = getRawData1D(
            ptg, tracks.ptx, tracks.prx[ii, :, :],
            tracks.vtx, tracks.vrx[ii, :], cfg.ta,
            cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx,
            cfg.system.wl, cfg.prf,
        )[::Nrx]
    return s_channel
