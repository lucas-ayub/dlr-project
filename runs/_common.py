# -*- coding: utf-8 -*-
"""
runs/_common.py

Shared helpers for the run_*.py driver scripts under runs/. Centralizes:
  - lightweight config/scene building for isolated GetCoeffNu checks
    (NOT the main run_experiment.py pipeline, which keeps its own richer
    factory in sar_recon.config)
  - GetCoeffNu convenience wrappers
  - figure saving with AUTOMATIC output-directory derivation, mirroring
    runs/<category>/<script>.py -> plots/<category>/<script>/
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import sar_recon as sar
from sar_recon.config import (SystemParams, Scene, ArrayGeometry,
                              prf_from_fixed, integration_time, build_time_axis)
from sar_recon.geometry import build_platform_tracks
from sar_recon.reconstruction import GetCoeffNu

_RUNS_ROOT = os.path.dirname(os.path.abspath(__file__))            # .../runs
_PROJECT_ROOT = os.path.dirname(_RUNS_ROOT)                        # .../sar_reconstruction
_PLOTS_ROOT = os.path.join(_PROJECT_ROOT, "plots")


def make_save_dir(caller_file):
    """
    Derives plots/<category>/<script_name>/ from a run script's __file__.
    E.g. runs/validation/run_model_validation_report.py
      -> plots/validation/run_model_validation_report/
    Creates the directory if needed and returns its path.
    """
    rel = os.path.relpath(os.path.abspath(caller_file), _RUNS_ROOT)
    rel_no_ext = os.path.splitext(rel)[0]
    save_dir = os.path.join(_PLOTS_ROOT, rel_no_ext)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_fig(fig, save_dir, name, vector=True):
    fig.savefig(os.path.join(save_dir, name + ".png"), dpi=150, bbox_inches="tight")
    if vector:
        fig.savefig(os.path.join(save_dir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_cfg(wl, bxt, channel, nrx=2, dx=100.0, extra_offsets=(),
              rDelay=0.0051115753, h0=0.0, prf_fixed=2000.0, name="run"):
    """Lightweight ExperimentConfig builder for isolated GetCoeffNu checks."""
    system = SystemParams(wl=wl)
    scene = Scene(rDelay=rDelay, c0=system.c0, h0=h0, extra_offsets=extra_offsets)

    bat = dx * np.arange(nrx)
    bxt_arr = np.zeros(nrx)
    bxt_arr[channel] = bxt
    array = ArrayGeometry(bat=bat, bxt=bxt_arr)

    prf, PRF_op = prf_from_fixed(prf_fixed, nrx)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, nrx, acq_time)

    cfg = sar.ExperimentConfig(
        name=name, system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta, plots_dir=None,
    )
    return cfg, build_platform_tracks(cfg), scene


def get_C0(cfg, tracks, ptg, channel):
    kk = channel
    C0, C1, C2, Dt = GetCoeffNu(
        ptg, tracks.ptx, tracks.prx[kk], tracks.vtx, tracks.vrx[kk],
        tracks.ptx, tracks.vtx,
        cfg.prf, cfg.system.wl, cfg.ta,
        cfg.sq_tx, cfg.sq_rx[kk], cfg.theta_tx, cfg.theta_rx[kk],
    )
    return C0


def get_terms(cfg, tracks, ptg, channel):
    kk = channel
    C0, C1, C2, Dt = GetCoeffNu(
        ptg, tracks.ptx, tracks.prx[kk], tracks.vtx, tracks.vrx[kk],
        tracks.ptx, tracks.vtx,
        cfg.prf, cfg.system.wl, cfg.ta,
        cfg.sq_tx, cfg.sq_rx[kk], cfg.theta_tx, cfg.theta_rx[kk],
    )
    return np.array([C0, Dt - C1, C2])


def reference_geometry(wl, rDelay=0.0051115753, h0=0.0):
    """r0, sin(theta0), cos(theta0) of the fixed reference geometry."""
    _, _, scene = build_cfg(wl, bxt=0.0, channel=1, rDelay=rDelay, h0=h0)
    r0 = scene.r0
    sin_t0 = scene.y0 / r0
    cos_t0 = scene.H / r0
    return r0, sin_t0, cos_t0


def iso_range_offset(r0, H, dh):
    """(dx, dy, dh) extra_offsets placing a target on the iso-range surface
    (same slant range r0 as the reference), at height dh."""
    if dh == 0.0:
        return ()
    y_target = np.sqrt(r0**2 - (H - dh)**2)
    y0 = np.sqrt(r0**2 - H**2)
    return ((0.0, float(y_target - y0), float(dh)),)