"""System, scene, array and experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C0_LIGHT = 299792458.0
RDELAY_DEFAULT = 0.0051115753          # two-way range delay -> r0 = 766.2 km


@dataclass
class SystemParams:
    wl: float = 0.25                   # wavelength [m]
    ve: float = 7408.5313923924796     # effective velocity [m/s]
    vs: float = 7688.53706432          # platform velocity [m/s]
    c0: float = C0_LIGHT
    da_factor: float = 24.0            # antenna element length = da_factor * wl
    La_factor: float = 2.0             # full antenna length   = La_factor * da

    @property
    def da(self) -> float:
        return self.da_factor * self.wl

    @property
    def La(self) -> float:
        return self.La_factor * self.da

    @property
    def abw(self) -> float:
        """Processed azimuth bandwidth [Hz]."""
        return 2.0 * self.ve / self.La

    @property
    def theta(self) -> float:
        """One-way antenna beamwidth [rad]."""
        return self.wl / self.La


@dataclass
class Scene:
    """Reference point (x0, y0, h0) used by the reconstruction, plus extra
    scatterers (dx, dy, dh) relative to it that are used to generate the signal."""
    rDelay: float
    H: float = 720e3
    x0: float = 20.0
    h0: float = 0.0
    c0: float = C0_LIGHT
    extra_offsets: tuple = ()

    @property
    def r0(self) -> float:
        return self.c0 * self.rDelay / 2.0

    @property
    def y0(self) -> float:
        return float(np.sqrt(max(self.r0 ** 2 - (self.H - self.h0) ** 2, 0.0)))

    @property
    def ptg(self) -> np.ndarray:
        return np.array([self.x0, self.y0, self.h0], dtype=np.float64)

    @property
    def points(self) -> np.ndarray:
        """Reference point followed by the extra scatterers, shape [Np, 3]."""
        pts = [self.ptg] + [self.ptg + np.array(o, dtype=np.float64) for o in self.extra_offsets]
        return np.array(pts, dtype=np.float64)


@dataclass
class ArrayGeometry:
    bat: np.ndarray                    # along-track baselines [m] (receiver i trails TX by bat[i])
    bxt: np.ndarray                    # cross-track baselines [m]

    @property
    def Nrx(self) -> int:
        return len(self.bat)


@dataclass
class ExperimentConfig:
    system: SystemParams
    scene: Scene
    array: ArrayGeometry
    prf: float                         # full (reconstructed) PRF [Hz]
    PRF_op: float                      # per-channel PRF [Hz]
    Na: int
    Na_ch: int
    ta: np.ndarray

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
        return np.zeros(self.Nrx)


def integration_time(system: SystemParams, scene: Scene) -> float:
    return (system.ve / system.da) / 2.0 / system.ve ** 2 * system.wl * scene.r0


def build_time_axis(prf: float, Nrx: int, acq_time: float, divfac: int = 1024):
    Na = int(np.ceil(acq_time * prf / Nrx / divfac) * Nrx * divfac)
    Na_ch = Na // Nrx
    ta = (np.arange(Na) - Na * 0.5) / prf
    return Na, Na_ch, ta


def make_config(targets, nrx: int = 4, prf: float = 2000.0, bxt_max: float = 100.0,
                seed: int = 0, rdelay: float = RDELAY_DEFAULT) -> ExperimentConfig:
    """
    targets : list of (x [m], h [m]); each target is placed on the iso-range
              surface of the reference point (same slant range r0).
    Array   : along-track baselines on the DPCA condition (step 2 vs / prf),
              cross-track baselines bxt ~ U(0, bxt_max) (bxt = 0 if bxt_max = 0).
    """
    sysp = SystemParams()
    base = Scene(rDelay=rdelay, c0=sysp.c0)
    r0, H, y0 = base.r0, base.H, base.y0
    extra = tuple((float(x), float(np.sqrt(r0 ** 2 - (H - h) ** 2) - y0), float(h)) for x, h in targets)
    scene = Scene(rDelay=rdelay, c0=sysp.c0, extra_offsets=extra)
    dx = 2.0 * sysp.vs / prf
    bat = dx * np.arange(nrx)
    bxt = np.random.default_rng(seed).uniform(0.0, bxt_max, nrx) if bxt_max > 0 else np.zeros(nrx)
    Na, Na_ch, ta = build_time_axis(prf, nrx, 2.0 * integration_time(sysp, scene))
    return ExperimentConfig(system=sysp, scene=scene, array=ArrayGeometry(bat=bat, bxt=bxt),
                            prf=prf, PRF_op=prf / nrx, Na=Na, Na_ch=Na_ch, ta=ta)
