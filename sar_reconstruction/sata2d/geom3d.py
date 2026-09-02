# -*- coding: utf-8 -*-
"""
3-D geometry: platform tracks and NUMERIC reconstruction coefficients.

Why a numeric coefficient extractor
-----------------------------------
`rd_recons2d.get_coeff_nu` is the closed-form linear-orbit version: it takes a
single scalar ``dx`` (along-track offset) and knows nothing about cross-track
baselines or target height.  Once the array has a cross-track baseline
``bxt[i]`` and the target sits at height ``h``, the range history is genuinely
3-D and the coefficients must be extracted from the actual tracks.

`get_coeff_nu_3d` below is the same algorithm as `sar_recon.reconstruction
.GetCoeffNu`: fit

    r_ms(t) = 2 |p_tx(t) - p|                    ~  c0 + c1 t + c2 t^2
    r_bs(t) - r_ms(t) = |p_tx-p| + |p_rx-p| - r_ms  ~  dc0 + dc1 t + dc2 t^2

over the aperture common to both, then combine (stationary-phase result):

    C0 = dc0 + c1^2/(4 c2) - (c1+dc1)^2 / (4 (c2+dc2))          [m]
    C1 = (c2 dc1 - c1 dc2) / (2 c2 (c2+dc2))                    [s]
    C2 = dc2 / (4 c2 (c2+dc2))
    Dt = t_bc^bs - t_bc^ms                                      [s]

so that the channel transfer function is

    H_i(f) = exp{ -2 pi j [ C0/wl + (-C1 + Dt) f + C2 wl f^2 ] } .

Topographic sensitivity
-----------------------
For a target at height h0 + dh and a receiver with cross-track baseline b_xt,
the leading term of the residual that the FLAT-EARTH filter fails to model is
the usual single-pass interferometric one,

    dC0  ~  - b_perp * dh / (r0 sin(theta_inc)) ,       b_perp ~ b_xt cos(...)

but nothing here relies on that approximation: `residual_C0_3d` evaluates the
exact difference C0(true target) - C0(flat point at the same slant range),
using the very same fit the reconstruction filter uses.  The approximation is
only quoted so the order of magnitude can be checked by hand.

Cost
----
One coefficient fit costs a few milliseconds (gradient + intersect1d + two
polyfits over Na samples).  A 2-D reconstruction needs them for every range bin,
which would be N_r x N_rx fits.  `CoeffTable3D` therefore evaluates them on a
coarse grid of ranges and interpolates -- the coefficients are smooth in r
(C0 ~ b^2/4r for the along-track part), and the interpolation error is checked
by `CoeffTable3D.check()`.
"""
from __future__ import annotations

import numpy as np

__all__ = ["build_tracks_3d", "get_coeff_nu_3d", "residual_C0_3d",
           "CoeffTable3D", "dC0_approx"]


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
    Closed-form order of magnitude of the residual, for sanity checks:

        dC0 ~ - b_xt * dh / (r0 * sin(theta_inc)) * sin(theta_inc)
            = - b_xt * dh / r0 * (1/tan(theta_inc)) ... (see the report)

    Implemented as the standard single-pass interferometric sensitivity
    ``-b_perp * dh / (r0 sin theta_inc)`` with ``b_perp = bxt[channel]``.
    """
    return -p.bxt[channel] * dh / (p.r0 * np.sin(p.theta_inc))


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
