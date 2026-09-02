# -*- coding: utf-8 -*-
"""
3-D acquisition parameters: along-track AND cross-track baselines + topography.

The 2-D linear-orbit model of `params.py` has receivers on the SAME straight
line as the transmitter, so the only baseline is along-track (``bat``). With a
purely along-track array the reconstruction is essentially blind to topography
(dC0/dr = -bat^2/(4 r^2), i.e. 0.008 deg per 100 m of range error), so SATA has
nothing to correct.

This module adds the two ingredients that make SATA meaningful:

* ``bxt`` -- a CROSS-TRACK baseline per receiver.  A target at height
  h0 + dh then produces a residual slant range that the flat-earth
  reconstruction filter does not model, of the standard interferometric size

      dC0  ~  - b_perp * dh / (r0 * sin(theta_inc))                        (1)

  which for the geometry below (r0 = 766 km, theta_inc = 20 deg) is
  0.343 m -- about 1.4 wavelengths at L-band -- for b_perp = 225 m, dh = 400 m.

* ``scene`` -- a list of scatterers with their own (dx_az, dy_ct, dh), so the
  residual varies WITH AZIMUTH POSITION.  That azimuth dependence is precisely
  what SATA exists to correct, and what a single global filter cannot.

Geometry
--------
Coordinates are (x, y, z) = (along-track, cross-track, height).

    transmitter    ptx(t) = ( vs*t,            0,        H )
    receiver i     prx(t) = ( vs*t - bat[i],   bxt[i],   H )
    target         ptg    = ( x0,              y0,       h0 )

with ``y0 = sqrt(r0^2 - (H - h0)^2)`` so that the central target sits at the
requested slant range r0.

Defaults reproduce the geometry of the existing 1-D SATA tests
(``runs/core/run_sata.py``): wl = 0.25 m, H = 720 km, rDelay = 0.0051115753 s
(r0 = 766.21 km, theta_inc = 20.0 deg), da = 24*wl, La = 2*da, PRF = 2000 Hz,
bat spacing 100 m -- so results here are directly comparable with the 1-D ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C0_LIGHT = 299792458.0

__all__ = ["Params3D", "make_params3d", "make_topo_ramp_azimuth",
           "make_bxt", "iso_range_offset"]


# ---------------------------------------------------------------------------
# Array geometry -- same generation rules as sar_recon.config.ArrayGeometry
# ---------------------------------------------------------------------------
def make_bxt(Nrx: int, dxt: float, mode: str = "linear",
             bxt_max: float | None = None, seed=None) -> np.ndarray:
    """
    Cross-track baselines, following ``ArrayGeometry.linear`` of `sar_recon`:

    ``mode="linear"``  bxt[i] = dxt * (i - (Nrx-1)/2)      (symmetric ladder)
    ``mode="random"``  bxt[i] ~ Uniform(0, bxt_max)        (bxt_max defaults to dxt)

    The random mode is the one used by ``make_topo_random_config`` /
    ``make_topo_dpca_random_config``; the seed makes it reproducible.
    """
    if mode == "linear":
        return dxt * (np.arange(Nrx) - (Nrx - 1) / 2.0)
    if mode == "random":
        hi = dxt if bxt_max is None else bxt_max
        gen = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        return gen.uniform(0.0, hi, size=Nrx)
    raise ValueError(f"unknown bxt mode {mode!r}; expected 'linear' or 'random'")


def iso_range_offset(r0: float, H: float, h0: float, dh: float) -> tuple:
    """
    Cross-track offset that keeps a target of height ``h0+dh`` on the SAME
    slant range ``r0`` (the "iso-range" surface):

        y(h) = sqrt(r0^2 - (H - h)^2)   ->   dy = y(h0+dh) - y(h0)

    Iso-range targets all land in the same range bin, so any reconstruction
    error they suffer cannot be blamed on the range-dependent filter -- it is
    purely the topographic residual.  This is the construction used by the 1-D
    SATA tests.
    """
    y0 = np.sqrt(max(r0 ** 2 - (H - h0) ** 2, 0.0))
    yt = np.sqrt(max(r0 ** 2 - (H - h0 - dh) ** 2, 0.0))
    return (0.0, float(yt - y0), float(dh))


def make_topo_ramp_azimuth(r0: float, H: float, h0: float, specs) -> tuple:
    """
    Scene offsets for a topography that varies ALONG AZIMUTH.

    ``specs`` is a sequence of ``(dx_az [m], dh [m])``.  Each entry becomes an
    iso-range scatterer at along-track offset ``dx_az`` and height ``h0+dh``.
    Default of the 1-D tests: ((-400,80), (-200,160), (0,240), (200,320),
    (400,400)) -- a ramp climbing along the flight direction.
    """
    out = []
    for dx_az, dh in specs:
        _, dy, ddh = iso_range_offset(r0, H, h0, dh)
        out.append((float(dx_az), dy, ddh))
    return tuple(out)


# ---------------------------------------------------------------------------
@dataclass
class Params3D:
    """
    Everything the 3-D pipeline needs.  The attribute names deliberately match
    `params.Params`, so `rd_recons2d.run_two_step` and the SATA driver accept
    either object without change.
    """
    # ---- system ----------------------------------------------------------
    Nrx: int
    wl: float
    ve: float                # effective (Doppler) velocity  [m/s]
    vs: float                # platform velocity             [m/s]
    La: float
    theta_tx: float
    theta_rx: np.ndarray
    sq_tx: float
    sq_rx: np.ndarray
    abw: float
    prf: float               # FULL (post-reconstruction) PRF
    # ---- range -----------------------------------------------------------
    rbw: float
    rsf: float
    cd: float
    Nr: int
    r_scan: np.ndarray
    # ---- azimuth ---------------------------------------------------------
    int_time: float
    acq_time: float
    Na: int
    Na_ch: int
    ta: np.ndarray
    # ---- array -----------------------------------------------------------
    bat: np.ndarray          # along-track baselines  [Nrx]
    bxt: np.ndarray          # cross-track baselines  [Nrx]
    # ---- scene -----------------------------------------------------------
    H: float
    x0: float
    h0: float
    r0: float
    extra_offsets: tuple     # ((dx_az, dy_ct, dh), ...) for GENERATION only
    name: str = "topo3d"

    # ---- derived ---------------------------------------------------------
    @property
    def PRF_op(self) -> float:
        return self.prf / self.Nrx

    @property
    def deltaX(self) -> np.ndarray:
        """Alias so code written for the 2-D linear model keeps working."""
        return self.bat

    @property
    def f0(self) -> float:
        return C0_LIGHT / self.wl

    @property
    def fr(self) -> np.ndarray:
        return np.roll(np.arange(self.Nr) * self.rsf / self.Nr - 0.5 * self.rsf,
                       int(0.5 * self.Nr))

    @property
    def idx_dec(self) -> np.ndarray:
        return (np.arange(self.Na_ch) * self.Nrx).astype(int)

    @property
    def y0(self) -> float:
        """Cross-track ground position of the central point."""
        return float(np.sqrt(max(self.r0 ** 2 - (self.H - self.h0) ** 2, 0.0)))

    @property
    def theta_inc(self) -> float:
        """Incidence angle at the central point [rad]."""
        return float(np.arcsin(self.y0 / self.r0))

    @property
    def ptg(self) -> np.ndarray:
        """The central point -- the one the RECONSTRUCTION assumes."""
        return np.array([self.x0, self.y0, self.h0], dtype=np.float64)

    @property
    def points(self) -> np.ndarray:
        """All scatterers used for signal GENERATION, [Np, 3]."""
        c = self.ptg
        pts = [c] if not self.extra_offsets else []
        for d in self.extra_offsets:
            pts.append(c + np.array(d, dtype=np.float64))
        return np.array(pts, dtype=np.float64)

    @property
    def r_ref(self) -> float:
        return float(self.r0)

    def flat_point_at_range(self, r: float) -> np.ndarray:
        """
        The point the flat-earth reconstruction filter assumes for slant range
        ``r``: same height h0, cross-track position back-solved from r.
        """
        y = np.sqrt(max(r ** 2 - (self.H - self.h0) ** 2, 0.0))
        return np.array([self.x0, y, self.h0], dtype=np.float64)

    def summary(self) -> str:
        return (
            f"name        : {self.name}\n"
            f"Nrx         : {self.Nrx}\n"
            f"wavelength  : {self.wl:.3f} m   (f0 = {self.f0/1e6:.1f} MHz)\n"
            f"ve / vs     : {self.ve:.1f} / {self.vs:.1f} m/s\n"
            f"La / theta  : {self.La:.2f} m / {np.degrees(self.theta_tx):.4f} deg\n"
            f"abw / PRF   : {self.abw:.1f} Hz / {self.prf:.1f} Hz "
            f"(PRF_op = {self.PRF_op:.1f} Hz)\n"
            f"rbw / rsf   : {self.rbw/1e6:.2f} / {self.rsf/1e6:.2f} MHz\n"
            f"Na x Nr     : {self.Na} x {self.Nr}   (Na_ch = {self.Na_ch})\n"
            f"Tint / Tacq : {self.int_time:.3f} / {self.acq_time:.3f} s\n"
            f"H           : {self.H/1e3:.1f} km\n"
            f"r0 / y0     : {self.r0/1e3:.3f} / {self.y0/1e3:.3f} km "
            f"(theta_inc = {np.degrees(self.theta_inc):.2f} deg)\n"
            f"range swath : {self.r_scan[0]/1e3:.3f} .. {self.r_scan[-1]/1e3:.3f} km\n"
            f"bat         : {np.array2string(self.bat, precision=1)}\n"
            f"bxt         : {np.array2string(self.bxt, precision=1)}\n"
            f"scatterers  : {len(self.points)}\n"
        )


# ---------------------------------------------------------------------------
def make_params3d(Nrx: int = 4,
                  dx: float = 100.0,
                  dxt: float = 150.0,
                  bxt_mode: str = "linear",
                  bxt_max: float | None = None,
                  seed: int = 0,
                  bat_offset: float = 0.0,
                  specs=((-400.0, 80.0), (-200.0, 160.0), (0.0, 240.0),
                         (200.0, 320.0), (400.0, 400.0)),
                  rDelay: float = 0.0051115753,
                  H: float = 720e3,
                  h0: float = 0.0,
                  x0: float = 0.0,
                  prf: float = 2000.0,
                  wl: float = 0.25,
                  ve: float = 7408.5313923924796,
                  vs: float = 7688.53706432,
                  da_factor: float = 24.0,
                  La_factor: float = 2.0,
                  res_rg: float = 30.0,
                  swath: float = 2000.0,
                  cd: float = 1.0e-6,
                  alpha_r: float = 0.15,
                  divfac: int = 256,
                  acq_factor: float = 2.0,
                  name: str | None = None) -> Params3D:
    """
    Build the 3-D parameter set.  Defaults reproduce the geometry of the 1-D
    SATA tests in ``runs/core/run_sata.py`` (``_build_azimuth_topo_cfg``), so
    the 2-D results are directly comparable.

    ``bxt_mode="random"`` draws bxt ~ U(0, bxt_max) with ``seed``, exactly like
    ``ArrayGeometry.linear(..., bxt_mode="random")`` in the repository.
    ``specs`` are the (azimuth offset, height) pairs of the azimuth-varying
    topography; pass ``()`` for a single flat point target.
    """
    c = C0_LIGHT
    da = da_factor * wl
    La = La_factor * da
    theta = wl / La
    abw = 2.0 * ve / La

    r0 = c * rDelay / 2.0

    # ---- range axis -------------------------------------------------------
    rbw = c / (2.0 * res_rg)
    rsf = (1.0 + alpha_r) * rbw
    Nr = int(2 ** np.ceil(np.log2(((2 * swath / c) + cd) * rsf)))
    # Put the central range r0 at 20 % into the swath, so the topography (which
    # spreads targets slightly in range) stays inside the recorded window.
    r_start = r0 - 0.20 * Nr / rsf * c * 0.5
    r_scan = r_start + np.arange(Nr) / rsf * c * 0.5

    # ---- azimuth axis -----------------------------------------------------
    # Tint = (ve/da)/2/ve^2 * wl * r0 : the same expression as
    # sar_recon.config.integration_time.
    int_time = (ve / da) / 2.0 / ve ** 2 * wl * r0
    acq_time = acq_factor * int_time
    Na = int(np.ceil(acq_time * prf / Nrx / divfac) * Nrx * divfac)
    Na_ch = int(Na / Nrx)
    ta = (np.arange(Na) - 0.5 * Na) / prf

    # ---- array ------------------------------------------------------------
    bat = bat_offset + dx * np.arange(Nrx)
    bxt = make_bxt(Nrx, dxt, mode=bxt_mode, bxt_max=bxt_max, seed=seed)

    extra = make_topo_ramp_azimuth(r0, H, h0, specs) if specs else ()

    if name is None:
        tag = f"bxt{bxt_mode}"
        name = f"topo3d_Nrx{Nrx}_dx{int(dx)}_{tag}"

    return Params3D(
        Nrx=Nrx, wl=wl, ve=ve, vs=vs, La=La, theta_tx=theta,
        theta_rx=np.full(Nrx, theta), sq_tx=0.0, sq_rx=np.zeros(Nrx), abw=abw,
        prf=prf, rbw=rbw, rsf=rsf, cd=cd, Nr=Nr, r_scan=r_scan,
        int_time=int_time, acq_time=acq_time, Na=Na, Na_ch=Na_ch, ta=ta,
        bat=bat, bxt=bxt, H=H, x0=x0, h0=h0, r0=r0, extra_offsets=extra,
        name=name,
    )


if __name__ == "__main__":
    print(make_params3d().summary())
