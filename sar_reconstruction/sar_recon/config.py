# -*- coding: utf-8 -*-
"""
Configuration layer.

All system and geometry parameters live here as dataclasses, so that adapting
the radar system or the acquisition geometry never requires touching the signal
model, the reconstruction, or the plotting code.

Layers:
    SystemParams      -> radar / platform physics (wl, ve, vs, antenna)
    Scene             -> target + slant range geometry (rDelay -> r0 -> y0)
    ArrayGeometry     -> receiver array layout (along-track + cross-track baselines)
    ExperimentConfig  -> assembles everything for a given Nrx and PRF strategy

The two original cases ("diff" and "dpca") are reproduced exactly by the
factory functions make_diff_config() and make_dpca_config(). To explore a new
system or geometry, either edit a preset or build an ExperimentConfig directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

DEFAULT_C0 = 299792458.0


# ---------------------------------------------------------------------------
# System physics
# ---------------------------------------------------------------------------
@dataclass
class SystemParams:
    """Radar and platform parameters. Antenna length and bandwidth are derived."""
    wl: float = 0.25                      # wavelength [m]
    ve: float = 7408.5313923924796        # effective (Doppler) velocity [m/s]
    vs: float = 7688.53706432             # platform velocity [m/s]
    c0: float = DEFAULT_C0                 # speed of light [m/s]
    da_factor: float = 24.0               # antenna element length: da = da_factor * wl
    La_factor: float = 2.0                # full antenna length:    La = La_factor * da

    @property
    def da(self) -> float:
        return self.da_factor * self.wl

    @property
    def La(self) -> float:
        return self.La_factor * self.da

    @property
    def abw(self) -> float:
        """Azimuth (processed) bandwidth [Hz]."""
        return 2.0 * self.ve / self.La

    @property
    def theta(self) -> float:
        """One-way antenna beamwidth [rad]."""
        return self.wl / self.La


# ---------------------------------------------------------------------------
# Scene / target geometry
# ---------------------------------------------------------------------------
@dataclass
class Scene:
    """Target position and reference slant range, derived from the range delay."""
    rDelay: float                          # two-way range delay [s]
    H: float = 720e3                       # platform height [m]
    x0: float = 20.0                       # target along-track position [m]
    h0: float = 2.0                        # target height [m]
    c0: float = DEFAULT_C0

    @property
    def r0(self) -> float:
        return self.c0 * self.rDelay / 2.0

    @property
    def y0(self) -> float:
        """Cross-track ground position, corrected for target height h0."""
        return float(np.sqrt(max(self.r0 ** 2 - (self.H - self.h0) ** 2, 0.0)))

    @property
    def ptg(self) -> np.ndarray:
        """Target position [x0, y0, h0] as a (3,) vector."""
        return np.array([self.x0, self.y0, self.h0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Receiver array geometry
# ---------------------------------------------------------------------------
@dataclass
class ArrayGeometry:
    """
    Receiver array layout relative to the transmitter.

    bat : along-track baselines  [Nrx]  (receiver i trails TX by bat[i])
    bxt : cross-track baselines  [Nrx]
    """
    bat: np.ndarray
    bxt: np.ndarray

    @property
    def Nrx(self) -> int:
        return len(self.bat)

    @classmethod
    def linear(cls, Nrx: int, dx: float, dxt: float) -> "ArrayGeometry":
        """Uniform linear array: along-track spacing dx, centred cross-track spacing dxt."""
        bat = dx * np.arange(Nrx)
        bxt = dxt * (np.arange(Nrx) - (Nrx - 1) / 2.0)
        return cls(bat=bat, bxt=bxt)


# ---------------------------------------------------------------------------
# Derived acquisition timing
# ---------------------------------------------------------------------------
def integration_time(system: SystemParams, scene: Scene) -> float:
    """Synthetic aperture integration time Tint [s]."""
    return (system.ve / system.da) / 2.0 / system.ve ** 2 * system.wl * scene.r0


def build_time_axis(prf: float, Nrx: int, acq_time: float, divfac: int = 1024):
    """Slow-time axis, with Na padded to a clean multiple of (Nrx * divfac)."""
    Na = int(np.ceil(acq_time * prf / Nrx / divfac) * Nrx * divfac)
    Na_ch = int(Na / Nrx)
    ta = (np.arange(Na) - Na * 0.5) / prf
    return Na, Na_ch, ta


# ---------------------------------------------------------------------------
# Full experiment configuration
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    name: str
    system: SystemParams
    scene: Scene
    array: ArrayGeometry
    prf: float                # final (full) PRF [Hz]
    PRF_op: float             # per-channel operating PRF [Hz]
    Na: int
    Na_ch: int
    ta: np.ndarray
    plots_dir: str
    divfac: int = 1024

    # --- convenience accessors ------------------------------------------
    @property
    def Nrx(self) -> int:
        return self.array.Nrx

    @property
    def abw(self) -> float:
        return self.system.abw

    @property
    def theta_tx(self) -> float:
        return self.system.theta

    @property
    def theta_rx(self) -> np.ndarray:
        return self.system.theta * np.ones(self.Nrx)

    @property
    def sq_tx(self) -> float:
        return 0.0

    @property
    def sq_rx(self) -> np.ndarray:
        return np.zeros(self.Nrx, np.float64)


# ---------------------------------------------------------------------------
# PRF strategy helpers
# ---------------------------------------------------------------------------
def prf_from_fixed(prf: float, Nrx: int):
    """Fixed full PRF; operating PRF derived. Baselines are then free parameters."""
    return prf, prf / Nrx


def prf_from_dpca(system: SystemParams, Nrx: int, dx: float):
    """DPCA spacing dx fixes the operating PRF: PRF_op = 2*vs / (Nrx*dx)."""
    PRF_op = 2.0 * system.vs / (Nrx * dx)
    return PRF_op * Nrx, PRF_op


# ---------------------------------------------------------------------------
# Presets (reproduce the original two cases exactly)
# ---------------------------------------------------------------------------
def make_dpca_config(Nrx: int, base_dir: str) -> ExperimentConfig:
    """DPCA case: small receiver spacing dx fixes the PRF; abw = 2*ve/La."""
    system = SystemParams()
    scene = Scene(rDelay=0.0038659204080400003, c0=system.c0)

    dx, dxt = 11.0, 100.0
    array = ArrayGeometry.linear(Nrx, dx, dxt)

    prf, PRF_op = prf_from_dpca(system, Nrx, dx)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, Nrx, acq_time)

    plots_dir = os.path.join(base_dir, "plots_dpca_prf")
    os.makedirs(plots_dir, exist_ok=True)

    return ExperimentConfig(
        name="dpca", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta, plots_dir=plots_dir,
    )


def make_diff_config(Nrx: int, base_dir: str) -> ExperimentConfig:
    """Large-baseline case: fixed PRF (2000 Hz), large along-track baselines."""
    system = SystemParams()
    scene = Scene(rDelay=0.0051115753, c0=system.c0)

    dx, dxt = 100.0, 200.0
    array = ArrayGeometry.linear(Nrx, dx, dxt)

    prf, PRF_op = prf_from_fixed(2000.0, Nrx)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, Nrx, acq_time)

    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    return ExperimentConfig(
        name="diff", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta, plots_dir=plots_dir,
    )


# Registry so the driver can iterate over named cases.
CONFIG_FACTORIES = {
    "diff": make_diff_config,
    "dpca": make_dpca_config,
}
