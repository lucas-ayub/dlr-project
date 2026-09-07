# -*- coding: utf-8 -*-
"""
Geometry: acquisition parameters, tracks, coefficients and raw-data
generation for the 3-D scene (along-track AND cross-track baselines, target
topography).

Merged from three former files so the package has fewer of them (params3d.py
+ geom3d.py + datagen3d.py, plus get_coeff_nu moved here from
rd_recons2d.py -- it is the analytic, linear-orbit counterpart of
get_coeff_nu_3d right below it, so the two belong together):

1. Parameters and array geometry (``Params3D``, ``make_params3d``,
   ``make_bxt``, ``iso_range_offset``, ``make_topo_ramp_azimuth``). The 2-D
   linear-orbit model has receivers on the SAME straight line as the
   transmitter, so the only baseline is along-track (``bat``); with a purely
   along-track array the reconstruction is essentially blind to topography
   (dC0/dr = -bat^2/(4 r^2), i.e. 0.008 deg per 100 m of range error), so SATA
   has nothing to correct. Two ingredients make it meaningful:

   * ``bxt`` -- a CROSS-TRACK baseline per receiver. A target at height
     h0 + dh then produces a residual slant range that the flat-earth
     reconstruction filter does not model, of the standard interferometric
     size ``dC0 = -b_xt * dh / (r0 * tan(theta_inc))``, which for the default
     geometry (r0 = 766 km, theta_inc = 20 deg) is 0.343 m -- about 1.4
     wavelengths at L-band -- for b_perp = 225 m, dh = 400 m.
   * ``scene`` -- a list of scatterers with their own (dx_az, dy_ct, dh), so
     the residual varies WITH AZIMUTH POSITION -- precisely what SATA exists
     to correct, and what a single global filter cannot.

   Coordinates are (x, y, z) = (along-track, cross-track, height)::

       transmitter    ptx(t) = ( vs*t,            0,        H )
       receiver i     prx(t) = ( vs*t - bat[i],   bxt[i],   H )
       target         ptg    = ( x0,              y0,       h0 )

   with ``y0 = sqrt(r0^2 - (H - h0)^2)`` so the central target sits at the
   requested slant range r0.

2. Tracks and coefficients (``build_tracks_3d``, ``get_coeff_nu``,
   ``get_coeff_nu_3d``, ``residual_C0_3d``, ``CoeffTable3D``,
   ``dC0_approx``): ``get_coeff_nu`` is the ANALYTIC linear-orbit
   (C0,C1,C2,Dt) fit (dx == 0 returns all zeros exactly); ``get_coeff_nu_3d``
   is the NUMERIC fit from the real 3-D track geometry that the
   reconstruction actually uses. ``dC0_approx`` is a closed-form check only,
   never called by the reconstruction path.

3. Raw-data generation (``range_history``, ``antenna_window``, ``echo_block``,
   ``get_raw_data_3d``, ``generate_reference_3d``, ``generate_channels_3d``):
   the ideal monostatic signal and the per-channel bistatic signals for a
   scene of one or more scatterers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C0_LIGHT = 299792458.0
C_LIGHT = C0_LIGHT
PI = np.pi

__all__ = [
    "Params3D", "make_params3d", "make_topo_ramp_azimuth", "make_bxt",
    "iso_range_offset",
    "build_tracks_3d", "get_coeff_nu", "get_coeff_nu_3d", "residual_C0_3d",
    "CoeffTable3D", "dC0_approx",
    "range_history", "antenna_window", "echo_block", "get_raw_data_3d",
    "generate_reference_3d", "generate_channels_3d",
]



# ============================================================================
# 1) Parameters and array geometry -- formerly params3d.py
# ============================================================================
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


# ============================================================================
# 2) Tracks and coefficients -- formerly geom3d.py, plus get_coeff_nu
#    (moved here from rd_recons2d.py: the analytic 2-D counterpart of
#    get_coeff_nu_3d right below)
# ============================================================================
# ---------------------------------------------------------------------------
# Analytic linear-orbit coefficients (the 2-D counterpart of get_coeff_nu_3d)
# ---------------------------------------------------------------------------
def get_coeff_nu(dx, r_ref, v, Tint, prf, wl):
    """
    (C0, C1, C2, Dt) for a receiver displaced ``dx`` along-track.

    The bistatic range history is expanded as a 2nd-order polynomial in slow
    time around its own point of closest approach, and compared with the
    monostatic one::

        r_ms(t) = 2*sqrt(r^2 + v^2 t^2)
        r_bs(t) =   sqrt(r^2 + v^2 t^2) + sqrt(r^2 + v^2 (t - dx/v)^2)

    C0 [m]   constant range offset (the term SATA corrects for topography)
    C1 [s]   linear / registration term (the DPCA time shift)
    C2       quadratic / defocus term
    Dt [s]   offset between the two points of closest approach

    ``dx == 0`` (the monostatic channel) returns all zeros exactly.
    """
    if dx == 0:
        return [0.0, 0.0, 0.0, 0.0]

    divfac = 32
    Na = int(np.ceil(Tint * prf / divfac) * divfac)
    ta = (np.arange(Na) - 0.5 * Na) / prf
    N_time, dN_time = 2, 2

    rh_ms = 2 * np.sqrt(r_ref ** 2 + v ** 2 * ta ** 2)
    rh_bs = (np.sqrt(r_ref ** 2 + v ** 2 * ta ** 2)
             + np.sqrt(r_ref ** 2 + v ** 2 * (ta - dx / v) ** 2))

    # Restrict both histories to the COMMON instantaneous-Doppler band, so the
    # two polynomials are fitted over the same physical aperture.
    f_max = np.max(-1 / wl * np.diff(rh_ms * prf))
    f_min = np.min(-1 / wl * np.diff(rh_bs * prf))
    f_ms = -1 / wl * np.diff(rh_ms * prf)
    f_bs = -1 / wl * np.diff(rh_bs * prf)
    vld_ms = np.where((f_ms < f_max) & (f_ms > f_min))[0]
    vld_bs = np.where((f_bs < f_max) & (f_bs > f_min))[0]

    rh_ms, ta_ms = rh_ms[vld_ms], ta[vld_ms]
    rh_bs, ta_bs = rh_bs[vld_bs], ta[vld_bs]

    # Align both windows on their point of closest approach and keep the
    # symmetric overlap.
    idx_ms = int(np.argmin(rh_ms))
    idx_bs = int(np.argmin(rh_bs))
    np_r = int(np.min([len(rh_ms[idx_ms:]), len(rh_bs[idx_bs:])]))
    np_l = int(np.min([len(rh_ms[:idx_ms]), len(rh_bs[:idx_bs])]))
    rh_ms, ta_ms = rh_ms[idx_ms - np_l:idx_ms + np_r], ta_ms[idx_ms - np_l:idx_ms + np_r]
    rh_bs, ta_bs = rh_bs[idx_bs - np_l:idx_bs + np_r], ta_bs[idx_bs - np_l:idx_bs + np_r]

    tbc_ms = ta_ms[int(np.argmin(rh_ms))]
    tbc_bs = ta_bs[int(np.argmin(rh_bs))]

    c_time = np.polyfit(ta_ms - tbc_ms, rh_ms, N_time)[::-1]
    dc_time = np.polyfit(ta_bs - tbc_bs, rh_bs - rh_ms, dN_time)[::-1]

    C0 = (dc_time[0] + c_time[1] ** 2 / 4 / c_time[2]
          - (c_time[1] + dc_time[1]) ** 2 / 4 / (c_time[2] + dc_time[2]))
    C1 = ((c_time[2] * dc_time[1] - c_time[1] * dc_time[2])
          / 2 / c_time[2] / (c_time[2] + dc_time[2]))
    C2 = dc_time[2] / 4 / c_time[2] / (c_time[2] + dc_time[2])
    Dt = tbc_bs - tbc_ms
    return [C0, C1, C2, Dt]


# ---------------------------------------------------------------------------
# 1) Tracks
# ---------------------------------------------------------------------------
def build_tracks_3d(p, ta=None):
    """
    Straight-line orbit with along- and cross-track baselines.

        ptx(t) = ( vs t,            0,       H )
        prx(t) = ( vs t - bat[i],   bxt[i],  H )

    Returns ``(ptx, prx, vtx, vrx)`` with shapes
    ``[Na,3]``, ``[Nrx,Na,3]``, ``[Na]``, ``[Nrx,Na]``.
    """
    if ta is None:
        ta = p.ta
    Na = len(ta)
    Nrx = p.Nrx
    vs = p.vs

    ptx = np.column_stack([vs * ta, np.zeros(Na), np.full(Na, p.H)])
    vtx = np.full(Na, float(vs))

    prx = np.zeros([Nrx, Na, 3], np.float64)
    vrx = np.zeros([Nrx, Na], np.float64)
    for i in range(Nrx):
        prx[i, :, 0] = vs * ta - p.bat[i]
        prx[i, :, 1] = p.bxt[i]
        prx[i, :, 2] = p.H
        vrx[i, :] = vs
    return ptx, prx, vtx, vrx


# ---------------------------------------------------------------------------
# 2) Numeric coefficients
# ---------------------------------------------------------------------------
def get_coeff_nu_3d(ptg, ptx, prx, vtx, vrx, prf, wl, ta,
                    sq_tx, sq_rx, theta_tx, theta_rx,
                    N_time=2, dN_time=2):
    """
    (C0, C1, C2, Dt) for one target and one receiver, from the actual 3-D
    tracks.  ``prf`` is the FULL (post-reconstruction) PRF.

    Returns ``None`` when the common aperture is too short to fit the
    polynomials -- the caller then leaves that (range, channel) uncorrected
    instead of producing a meaningless coefficient.
    """
    ptg = np.asarray(ptg, dtype=np.float64)

    # --- transmit leg: instantaneous squint and beam window ---------------
    rhT = np.sqrt(np.sum((ptx - ptg[None, :]) ** 2, axis=1))
    inst_sqT = np.arcsin(np.clip(np.gradient(rhT, 1.0 / prf) / vtx, -1.0, 1.0))
    validT = np.where(np.abs(inst_sqT) <= (sq_tx + theta_tx / 2)) [0]

    # --- receive leg -------------------------------------------------------
    rhR = np.sqrt(np.sum((prx - ptg[None, :]) ** 2, axis=1))
    inst_sqR = np.arcsin(np.clip(np.gradient(rhR, 1.0 / prf) / vrx, -1.0, 1.0))
    validR = np.where(np.abs(inst_sqR) <= (sq_rx + theta_rx / 2))[0]
    if validT.size < N_time + 3 or validR.size < dN_time + 3:
        return None

    # --- common aperture ---------------------------------------------------
    taCommon = np.intersect1d(ta[validT], ta[validR])
    idx = np.nonzero(np.isin(ta, taCommon))[0]
    if idx.size < N_time + 3:
        return None

    rhMS = 2 * rhT[idx]
    rhBS = (rhT + rhR)[idx]

    # --- restrict to the common instantaneous-Doppler band ----------------
    fi_ms = -1 / wl * np.diff(rhMS) * prf
    fi_bs = -1 / wl * np.diff(rhBS) * prf
    if fi_ms.size == 0 or fi_bs.size == 0:
        return None
    f_max = np.min([abs(np.max(fi_ms)), abs(np.max(fi_bs))])
    f_min = -np.min([abs(np.min(fi_ms)), abs(np.min(fi_bs))])
    vld_ms = np.where((fi_ms < f_max) & (fi_ms > f_min))[0]
    vld_bs = np.where((fi_bs < f_max) & (fi_bs > f_min))[0]
    if vld_ms.size < N_time + 2 or vld_bs.size < dN_time + 2:
        return None

    rh_ms, ta_ms = rhMS[vld_ms], taCommon[vld_ms]
    rh_bs, ta_bs = rhBS[vld_bs], taCommon[vld_bs]

    # --- align on the points of closest approach, keep the symmetric span --
    i_ms, i_bs = int(np.argmin(rh_ms)), int(np.argmin(rh_bs))
    nr = int(np.min([len(rh_ms[i_ms:]), len(rh_bs[i_bs:])]))
    nl = int(np.min([len(rh_ms[:i_ms]), len(rh_bs[:i_bs])]))
    rh_ms, ta_ms = rh_ms[i_ms - nl:i_ms + nr], ta_ms[i_ms - nl:i_ms + nr]
    rh_bs, ta_bs = rh_bs[i_bs - nl:i_bs + nr], ta_bs[i_bs - nl:i_bs + nr]
    if rh_ms.size < N_time + 2 or rh_bs.size < dN_time + 2:
        return None

    tbc_ms = ta_ms[int(np.argmin(rh_ms))]
    tbc_bs = ta_bs[int(np.argmin(rh_bs))]

    c_time = np.polyfit(ta_ms - tbc_ms, rh_ms, N_time)[::-1]
    dc_time = np.polyfit(ta_bs - tbc_bs, rh_bs - rh_ms, dN_time)[::-1]

    C0 = (dc_time[0] + c_time[1] ** 2 / 4 / c_time[2]
          - (c_time[1] + dc_time[1]) ** 2 / 4 / (c_time[2] + dc_time[2]))
    C1 = ((c_time[2] * dc_time[1] - c_time[1] * dc_time[2])
          / 2 / c_time[2] / (c_time[2] + dc_time[2]))
    C2 = dc_time[2] / 4 / c_time[2] / (c_time[2] + dc_time[2])
    Dt = tbc_bs - tbc_ms
    return float(C0), float(C1), float(C2), float(Dt)


def _coeff(p, tracks, ptg, channel):
    ptx, prx, vtx, vrx = tracks
    return get_coeff_nu_3d(ptg, ptx, prx[channel], vtx, vrx[channel],
                           p.prf, p.wl, p.ta, p.sq_tx, p.sq_rx[channel],
                           p.theta_tx, p.theta_rx[channel])


# ---------------------------------------------------------------------------
# 3) The topographic residual
# ---------------------------------------------------------------------------
def residual_C0_3d(p, tracks, ptg_true, channel) -> float:
    """
    dC0 [m] = C0(true target) - C0(flat point at the SAME slant range).

    The 2-D reconstruction builds one filter per range bin, at the flat-earth
    height h0.  A target of height h0+dh lands in the range bin of its own
    slant range r, where the filter assumes the flat point of that same r.
    The difference of the two C0 is exactly what the filter fails to model,
    and exactly what SATA must remove.  Returns 0.0 when either fit fails.
    """
    r = float(np.sqrt(ptg_true[1] ** 2 + (p.H - ptg_true[2]) ** 2))
    ptg_flat = p.flat_point_at_range(r)
    ptg_flat[0] = ptg_true[0]          # same azimuth position: isolate height
    a = _coeff(p, tracks, np.asarray(ptg_true, float), channel)
    b = _coeff(p, tracks, ptg_flat, channel)
    if a is None or b is None:
        return 0.0
    return float(a[0] - b[0])


def dC0_approx(p, channel: int, dh: float) -> float:
    """
    Closed-form residual, for sanity checks:

        dC0  =  - b_xt * dh / (r0 * tan(theta_inc))                         (3)

    Derivation.  A receiver displaced cross-track by b_xt sees the target at

        r' = sqrt((y - b_xt)^2 + (H-h)^2) ~ r - b_xt * (y/r) = r - b_xt sin(th)

    so the constant term of (r_bs - r_ms) at closest approach is
    C0 ~ r' - r = -b_xt sin(th)  (plus the along-track b_at^2/4r term, which
    does not depend on height).  Moving the target along the ISO-RANGE surface
    keeps r fixed while y grows as dy/dh = (H-h)/y = cot(th), hence

        d(sin th)/dh = (1/r) dy/dh = cos(th) / (r sin(th))

    and therefore dC0/dh = -b_xt cos(th)/(r sin(th)) = -b_xt/(r tan(th)).

    Accurate to 0.25 % against the exact `residual_C0_3d` for dh up to 400 m
    in the default geometry -- so it is a genuine check, not just an order of
    magnitude.  (The often-quoted interferometric form -b_perp*dh/(r sin th)
    is the sensitivity for a target moving vertically at fixed GROUND position;
    on the iso-range surface the extra cos(th) appears.)
    """
    return -p.bxt[channel] * dh / (p.r0 * np.tan(p.theta_inc))


# ---------------------------------------------------------------------------
# 4) Coefficient table over range (what the reconstruction consumes)
# ---------------------------------------------------------------------------
class CoeffTable3D:
    """
    (C0, C1, C2, Dt) as a function of slant range, per channel.

    Evaluated on ``n_nodes`` ranges spanning ``r_scan`` and linearly
    interpolated in between.  Call it like a function::

        table = CoeffTable3D(p, tracks)
        C0, C1, C2, Dt = table(r0, channel)

    which is the signature `rd_recons2d.generalized_rd` expects for its
    ``coeff_fn`` argument.
    """

    def __init__(self, p, tracks, n_nodes: int = 24, height: float | None = None,
                 verbose: bool = False):
        self.p = p
        self.tracks = tracks
        self.height = p.h0 if height is None else height
        r = p.r_scan
        self.r_nodes = np.linspace(r[0], r[-1], n_nodes)
        self.tab = np.zeros([n_nodes, p.Nrx, 4])
        for j, rr in enumerate(self.r_nodes):
            ptg = p.flat_point_at_range(rr)
            ptg[2] = self.height
            # re-solve y for the requested height so the point really is at rr
            ptg[1] = np.sqrt(max(rr ** 2 - (p.H - self.height) ** 2, 0.0))
            for i in range(p.Nrx):
                c = _coeff(p, tracks, ptg, i)
                self.tab[j, i, :] = (0, 0, 0, 0) if c is None else c
        if verbose:
            print(f"  CoeffTable3D: {n_nodes} nodes x {p.Nrx} channels, "
                  f"h = {self.height:.1f} m")

    def __call__(self, r0: float, channel: int):
        t = self.tab[:, channel, :]
        return tuple(np.interp(r0, self.r_nodes, t[:, k]) for k in range(4))

    def check(self, n_test: int = 5) -> float:
        """Max relative interpolation error of C0, sampled between the nodes."""
        p, tracks = self.p, self.tracks
        rs = np.linspace(self.r_nodes[0], self.r_nodes[-1], n_test * 2 + 1)[1::2]
        err = 0.0
        for rr in rs:
            ptg = p.flat_point_at_range(rr)
            ptg[2] = self.height
            ptg[1] = np.sqrt(max(rr ** 2 - (p.H - self.height) ** 2, 0.0))
            for i in range(p.Nrx):
                c = _coeff(p, tracks, ptg, i)
                if c is None:
                    continue
                est = self(rr, i)[0]
                den = max(abs(c[0]), 1e-12)
                err = max(err, abs(est - c[0]) / den)
        return float(err)


# ============================================================================
# 3) Raw-data generation for the 3-D scene -- formerly datagen3d.py
# ============================================================================
# ---------------------------------------------------------------------------
def range_history(ptg, ps_tx, ps_rx):
    """Eq. (1): |p_tx - p| + |p_rx - p| for tracks given as [Na, 3]."""
    a = np.sqrt(np.sum((ps_tx - ptg[None, :]) ** 2, axis=1))
    b = np.sqrt(np.sum((ps_rx - ptg[None, :]) ** 2, axis=1))
    return a + b


def antenna_window(ptg, ps, v, prf, sq, theta):
    """Eq. (3): boolean beam window of one leg."""
    rh = np.sqrt(np.sum((ps - ptg[None, :]) ** 2, axis=1))
    inst_sq = np.arcsin(np.clip(np.gradient(rh, 1.0 / prf) / v, -1.0, 1.0))
    return np.abs(inst_sq) <= (sq + theta / 2.0)


def echo_block(R, w, r_min, Nr, rbw, cd, rsf, wl, out=None):
    """
    Eq. (2), vectorised: returns (or accumulates into ``out``) the [Na, Nr]
    echo of one scatterer.  ``R`` is HALF the two-way path (so the delay is
    2R/c), ``w`` the azimuth window.
    """
    Na = len(R)
    fast_time = 2 * r_min / C_LIGHT + np.arange(Nr) / rsf          # [Nr]
    tau = fast_time[None, :] - 2.0 * R[:, None] / C_LIGHT          # [Na, Nr]
    inside = (tau >= 0.0) & (tau <= cd)
    ph = (-1j * PI * rbw / cd * (tau - 0.5 * cd) ** 2
          - 4j * PI / wl * R[:, None])
    blk = np.where(inside, np.exp(ph), 0.0) * w[:, None]
    if out is None:
        return blk.astype(np.complex64)
    out += blk.astype(np.complex64)
    return out


# ---------------------------------------------------------------------------
def get_raw_data_3d(p, ptgs, ps_tx, ps_rx, v_tx, v_rx, ta, prf,
                    sigma=None) -> np.ndarray:
    """
    [Na, Nr] raw data of a TX/RX pair for all scatterers in ``ptgs`` [Np, 3].

    ``prf`` is the sampling rate of ``ta`` (full PRF for the reference,
    ``PRF_op`` for one decimated channel).  Pass ``ps_rx = ps_tx`` and
    ``v_rx = v_tx`` for the monostatic reference.
    """
    Na, Nr = len(ta), p.Nr
    ptgs = np.atleast_2d(ptgs)
    if sigma is None:
        sigma = np.ones(len(ptgs))
    data = np.zeros([Na, Nr], np.complex64)

    for j, ptg in enumerate(ptgs):
        w_tx = antenna_window(ptg, ps_tx, v_tx, prf, p.sq_tx, p.theta_tx)
        w_rx = antenna_window(ptg, ps_rx, v_rx, prf, p.sq_tx, p.theta_tx)
        w = (w_tx & w_rx).astype(np.float64)
        if not w.any():
            continue
        R = 0.5 * range_history(ptg, ps_tx, ps_rx)
        echo_block(R, sigma[j] * w, p.r_scan[0], Nr, p.rbw, p.cd, p.rsf,
                   p.wl, out=data)
    return data


def generate_reference_3d(p, tracks, ptgs=None) -> np.ndarray:
    """Ideal monostatic signal at the FULL PRF -- the comparison reference."""
    ptx, prx, vtx, vrx = tracks
    if ptgs is None:
        ptgs = p.points
    return get_raw_data_3d(p, ptgs, ptx, ptx, vtx, vtx, p.ta, p.prf)


def generate_channels_3d(p, tracks, ptgs=None) -> np.ndarray:
    """
    Per-channel bistatic signals on the decimated azimuth axis,
    [Nrx, Na_ch, Nr].  Channel ``i`` uses TX at ``ptx`` and RX at ``prx[i]``,
    then keeps every ``Nrx``-th azimuth sample -- the physical statement that
    each channel is read out once per transmitted pulse group.
    """
    ptx, prx, vtx, vrx = tracks
    if ptgs is None:
        ptgs = p.points
    idx = p.idx_dec
    out = np.zeros([p.Nrx, p.Na_ch, p.Nr], np.complex64)
    for i in range(p.Nrx):
        out[i] = get_raw_data_3d(
            p, ptgs, ptx[idx], prx[i][idx], vtx[idx], vrx[i][idx],
            p.ta[idx], p.PRF_op)
    return out
