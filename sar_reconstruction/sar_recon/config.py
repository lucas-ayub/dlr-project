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
    """
    Target geometry, derived from the range delay.

    The central point (x0, y0, h0) is the one used for RECONSTRUCTION
    (sceneMid is always [3, 1] -> single coefficient fit per range bin).

    extra_offsets lets you add more scatterers to the SCENE used for signal
    GENERATION only: each entry is an (dx, dy, dh) offset, in metres, applied
    to the central point. Reconstruction keeps using the single central point
    regardless of how many scatterers produced the signal.
    """
    rDelay: float                          # two-way range delay [s]
    H: float = 720e3                       # platform height [m]
    x0: float = 20.0                       # central target along-track position [m]
    h0: float = 2.0                        # central target height [m]
    c0: float = DEFAULT_C0
    extra_offsets: tuple = ()              # tuple of (dx, dy, dh) offsets [m]

    @property
    def r0(self) -> float:
        return self.c0 * self.rDelay / 2.0

    @property
    def y0(self) -> float:
        """Cross-track ground position of the central point, corrected for h0."""
        return float(np.sqrt(max(self.r0 ** 2 - (self.H - self.h0) ** 2, 0.0)))

    @property
    def ptg(self) -> np.ndarray:
        """Central point [x0, y0, h0] (3,) -- the one reconstruction uses."""
        return np.array([self.x0, self.y0, self.h0], dtype=np.float64)

    @property
    def points(self) -> np.ndarray:
        """
        All scatterers used for signal GENERATION: the central point plus any
        extra_offsets, shape [Np, 3]. With no extra_offsets this is just the
        central point, identical to the original single-target behaviour.
        """
        center = self.ptg
        pts = [center]
        for ddx, ddy, ddh in self.extra_offsets:
            pts.append(center + np.array([ddx, ddy, ddh], dtype=np.float64))
        return np.array(pts, dtype=np.float64)

    @classmethod
    def from_target(cls, x0: float, y0: float, h0: float, H: float = 720e3,
                    c0: float = DEFAULT_C0, extra_offsets: tuple = ()) -> "Scene":
        """
        Build a Scene directly from the desired central point (x0, y0, h0),
        instead of from rDelay. This back-computes the range delay that would
        place the central point exactly there:

            r0 = sqrt(y0**2 + (H - h0)**2)
            rDelay = 2 * r0 / c0

        Use this when you think in terms of "I want my target at this
        position", rather than "I want this range delay".
        """
        r0 = float(np.sqrt(y0 ** 2 + (H - h0) ** 2))
        rDelay = 2.0 * r0 / c0
        return cls(rDelay=rDelay, H=H, x0=x0, h0=h0, c0=c0, extra_offsets=extra_offsets)


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
# Named multi-scatterer scene presets
# ---------------------------------------------------------------------------
# Each entry is a tuple of (dx, dy, dh) offsets [m] relative to the central
# reconstruction point. "single" (the default) reproduces the original
# one-target behaviour exactly. To add your own scene, just add a new entry
# here -- nothing else in the pipeline needs to change.
SCENE_PRESETS = {
    "single": (),
    "along_track_line": (
        (-60.0, 0.0, 0.0),
        (-20.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
        (60.0, 0.0, 0.0),
    ),
    "cross_track_patch": (
        (25.0, 15.0, 0.0),
        (-25.0, 15.0, 0.0),
        (25.0, -15.0, 0.0),
        (-25.0, -15.0, 0.0),
    ),
    "varied_heights": (
        (15.0, 5.0, 8.0),
        (-15.0, 5.0, 15.0),
        (15.0, -10.0, 3.0),
        (-15.0, -10.0, 25.0),
    ),
}


def _plots_subdir(base_plots_dir: str, scene_name: str) -> str:
    """plots/ for the default 'single' scene, plots/<scene_name>/ otherwise,
    so multi-point runs don't mix their plots with the single-target ones."""
    path = base_plots_dir if scene_name == "single" else os.path.join(base_plots_dir, scene_name)
    os.makedirs(path, exist_ok=True)
    return path


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
# Presets (reproduce the original two cases exactly when scene_name="single")
# ---------------------------------------------------------------------------
def make_dpca_config(Nrx: int, base_dir: str, scene_name: str = "single") -> ExperimentConfig:
    """DPCA case: small receiver spacing dx fixes the PRF; abw = 2*ve/La."""
    system = SystemParams()
    scene = Scene(rDelay=0.0038659204080400003, c0=system.c0,
                  extra_offsets=SCENE_PRESETS[scene_name])

    dx, dxt = 11.0, 100.0
    array = ArrayGeometry.linear(Nrx, dx, dxt)

    prf, PRF_op = prf_from_dpca(system, Nrx, dx)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, Nrx, acq_time)

    plots_dir = _plots_subdir(os.path.join(base_dir, "plots_dpca_prf"), scene_name)

    return ExperimentConfig(
        name="dpca", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta, plots_dir=plots_dir,
    )


def make_diff_config(Nrx: int, base_dir: str, scene_name: str = "single") -> ExperimentConfig:
    """Large-baseline case: fixed PRF (2000 Hz), large along-track baselines."""
    system = SystemParams()
    scene = Scene(rDelay=0.0051115753, c0=system.c0,
                  extra_offsets=SCENE_PRESETS[scene_name])

    dx, dxt = 100.0, 200.0
    array = ArrayGeometry.linear(Nrx, dx, dxt)

    prf, PRF_op = prf_from_fixed(2000.0, Nrx)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, Nrx, acq_time)

    plots_dir = _plots_subdir(os.path.join(base_dir, "plots"), scene_name)

    return ExperimentConfig(
        name="diff", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta, plots_dir=plots_dir,
    )


# Registry so the driver can iterate over named cases.
# Each factory takes (Nrx, base_dir, scene_name="single").
CONFIG_FACTORIES = {
    "diff": make_diff_config,
    "dpca": make_dpca_config,
}
