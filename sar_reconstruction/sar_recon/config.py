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
    def linear(cls, Nrx: int, dx: float, dxt: float,
               bat_offset: float = 0.0) -> "ArrayGeometry":
        """
        Uniform linear array: along-track spacing dx, centred cross-track
        spacing dxt.

        bat_offset shifts the whole array along-track relative to the TX.
        Default (0.0) places the first receiver at bat=0 (co-located with
        the TX in along-track), which is the standard DPCA condition.
        Set bat_offset=dx/2 (half-spacing) to ensure every receiver has a
        non-zero bat, which is useful for testing the numerical reconstruction
        in the general bistatic case.

        bat[i] = bat_offset + dx * i
        """
        bat = bat_offset + dx * np.arange(Nrx)
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
# Topographic ramp: defined by elevation angle alpha, not by a direct
# maximum dh. Points lie on a line passing through the center (dy=0, dh=0)
# inclined at alpha degrees with respect to the horizontal plane:
#
#     dh = dy * tan(alpha)
#
# To change the ramp inclination, only modify TOPO_RAMP_ALPHA_DEG -- the
# rest (signal generation, reconstruction, plots) adapts automatically.
# ---------------------------------------------------------------------------
TOPO_RAMP_ALPHA_DEG = 14.04    # ramp elevation angle [degrees]
TOPO_RAMP_LENGTH = 2000.0      # cross-track extent of the ramp at frac=1.0 [m]
TOPO_RAMP_N_POINTS = 9         # number of extra scatterers along the ramp


def _make_topo_ramp(alpha_deg: float, length: float = TOPO_RAMP_LENGTH,
                    n_points: int = TOPO_RAMP_N_POINTS) -> tuple:
    """
    Generate offsets (dx, dy, dh) on a line inclined at alpha_deg degrees,
    passing through the central reconstruction point (dy=0, dh=0):

        dy = frac * length
        dh = dy * tan(alpha_deg)

    n_points equally spaced from 10% to 100% of 'length'.
    """
    alpha = np.radians(alpha_deg)
    tan_a = np.tan(alpha)
    return tuple(
        (0.0, round(frac * length, 2), round(frac * length * tan_a, 2))
        for frac in np.linspace(0.10, 1.0, n_points)
    )


def scene_ramp_angle_deg(scene: "Scene") -> float | None:
    """
    Retrieves the ramp elevation angle (in degrees) from the extra_offsets
    of a Scene, assuming they lie on a line passing through the origin in
    the (cross-track, height) plane -- exactly what _make_topo_ramp produces.

    Returns None if there are no extra_offsets or the angle cannot be
    determined (e.g., dy=0). Used by scene plots to annotate alpha
    directly from the scene, without needing to pass the angle separately.
    """
    if not scene.extra_offsets:
        return None
    _, dy, dh = scene.extra_offsets[-1]
    if dy == 0.0:
        return None
    return float(np.degrees(np.arctan2(dh, dy)))

def _alpha_tag(alpha_deg: float) -> str:
    """Convert an angle to a filename/dict-safe string, e.g. 14.04 -> '14p04'."""
    return f"{alpha_deg:g}".replace(".", "p").replace("-", "m")


# List of ramp angles (degrees) to sweep as separate scenes.
# Each value becomes a preset named "topo_ramp_alpha<tag>" registered in SCENE_PRESETS.
TOPO_RAMP_ALPHA_VALUES_DEG = [5.0, 10.0, 15.0, 20.0, 30.0]
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
    "topo_ramp": _make_topo_ramp(TOPO_RAMP_ALPHA_DEG, TOPO_RAMP_LENGTH, TOPO_RAMP_N_POINTS),
}

for _alpha in TOPO_RAMP_ALPHA_VALUES_DEG:
    SCENE_PRESETS[f"topo_ramp_alpha{_alpha_tag(_alpha)}"] = _make_topo_ramp(_alpha)


def _plots_subdir(base_plots_dir: str, scene_name: str) -> str:
    """
    All cases share a single plots/ root:
        plots/<case_name>/<scene_name>/

    This keeps the output tree clean: one folder to look in, subfolders
    organised first by case, then by scene.
    """
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
# Presets
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

    return ExperimentConfig(
        name="dpca", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta,
        plots_dir=_plots_subdir(base_dir, "dpca", scene_name),
    )


def make_large_bat_config(Nrx: int, base_dir: str, scene_name: str = "single") -> ExperimentConfig:
    """Large along-track baseline case: fixed PRF (2000 Hz), large bat spacing."""
    system = SystemParams()
    scene = Scene(rDelay=0.0051115753, c0=system.c0,
                  extra_offsets=SCENE_PRESETS[scene_name])

    dx, dxt = 100.0, 200.0
    array = ArrayGeometry.linear(Nrx, dx, dxt)

    prf, PRF_op = prf_from_fixed(2000.0, Nrx)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, Nrx, acq_time)

    return ExperimentConfig(
        name="large_bat", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta,
        plots_dir=_plots_subdir(base_dir, "large_bat", scene_name),
    )


def make_dpca_offset_config(Nrx: int, base_dir: str, scene_name: str = "single") -> ExperimentConfig:
    """
    DPCA variant where every receiver has a non-zero bat.

    Same system and PRF as make_dpca_config, but the array is shifted by
    dx/2 (half the receiver spacing), so:

        bat[i] = dx/2 + dx * i   (no receiver at bat=0)

    This tests the numerical reconstruction in the general bistatic case
    without the "coinciding phase center" of the standard DPCA condition.
    Compare the combined plot against the standard 'dpca' case to see how
    the reconstruction quality changes.
    """
    system = SystemParams()
    scene = Scene(rDelay=0.0038659204080400003, c0=system.c0,
                  extra_offsets=SCENE_PRESETS[scene_name])

    dx, dxt = 11.0, 100.0
    array = ArrayGeometry.linear(Nrx, dx, dxt, bat_offset=dx / 2)

    prf, PRF_op = prf_from_dpca(system, Nrx, dx)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, Nrx, acq_time)

    return ExperimentConfig(
        name="dpca_offset", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta,
        plots_dir=_plots_subdir(base_dir, "dpca_offset", scene_name),
    )
    
def make_topo_config(Nrx: int, base_dir: str, scene_name: str = "topo_ramp",
                     dxt: float = 0.0) -> ExperimentConfig:
    """Topographic experiment: central point at h=0, targets in a ramp, variable dxt."""
    system = SystemParams()

    # h0=0.0 → central point on ground
    scene = Scene(rDelay=0.0051115753, c0=system.c0, h0=0.0,
                  extra_offsets=SCENE_PRESETS[scene_name])

    dx = 100.0         # along-track spacing 
    array = ArrayGeometry.linear(Nrx, dx, dxt)   

    prf, PRF_op = prf_from_fixed(2000.0, Nrx)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, Nrx, acq_time)

    plots_dir = _plots_subdir(
        os.path.join(base_dir, "plots", f"topo_dxt{int(dxt)}"), scene_name
    )

    return ExperimentConfig(
        name=f"topo_dxt{int(dxt)}", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta, plots_dir=plots_dir,
    )

def _make_topo_dxt(dxt_val):
    """Create a factory for a fixed dxt, compatible with signature (Nrx, base_dir, scene_name)."""
    def factory(Nrx, base_dir, scene_name="topo_ramp"):
        return make_topo_config(Nrx, base_dir, scene_name, dxt=dxt_val)
    return factory


def make_topo_dpca_config(Nrx: int, base_dir: str, scene_name: str = "topo_ramp",
                          dxt: float = 0.0) -> ExperimentConfig:
    """Topographic experiment with DPCA timing: h0=0, ramp scene, variable dxt."""
    system = SystemParams()

    scene = Scene(rDelay=0.0038659204080400003, c0=system.c0, h0=0.0,
                  extra_offsets=SCENE_PRESETS[scene_name])

    dx = 11.0
    array = ArrayGeometry.linear(Nrx, dx, dxt)

    prf, PRF_op = prf_from_dpca(system, Nrx, dx)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, Nrx, acq_time)

    plots_dir = _plots_subdir(
        os.path.join(base_dir, "plots", f"topo_dpca_dxt{int(dxt)}"), scene_name
    )

    return ExperimentConfig(
        name=f"topo_dpca_dxt{int(dxt)}", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta, plots_dir=plots_dir,
    )


def _make_topo_dpca_dxt(dxt_val):
    def factory(Nrx, base_dir, scene_name="topo_ramp"):
        return make_topo_dpca_config(Nrx, base_dir, scene_name, dxt=dxt_val)
    return factory



# Registry so the driver can iterate over named cases.
# Each factory takes (Nrx, base_dir, scene_name="single").
CONFIG_FACTORIES = {
    "large_bat":    make_large_bat_config,
    "dpca":         make_dpca_config,
    "dpca_offset":  make_dpca_offset_config,
    "topo_dxt0":  _make_topo_dxt(0.0),
    "topo_dxt10": _make_topo_dxt(10.0),
    "topo_dxt20": _make_topo_dxt(20.0),
    "topo_dxt50": _make_topo_dxt(50.0),
    "topo_dxt100":_make_topo_dxt(100.0),
    "topo_dpca_dxt0":   _make_topo_dpca_dxt(0.0),
    "topo_dpca_dxt10":  _make_topo_dpca_dxt(10.0),
    "topo_dpca_dxt20":  _make_topo_dpca_dxt(20.0),
    "topo_dpca_dxt50":  _make_topo_dpca_dxt(50.0),
    "topo_dpca_dxt100": _make_topo_dpca_dxt(100.0),
}