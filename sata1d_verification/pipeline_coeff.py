"""
pipeline3d.py
=============
Ported VERBATIM from the repo's sata2d/geom3d.py: the 3-D tracks and the
NUMERIC coefficient extractor that computes C0/C1/C2 from the ACTUAL range
history (the same algorithm as sar_recon.reconstruction.GetCoeffNu), plus
residual_C0_3d = C0(true target) - C0(flat point at the same slant range).

This replaces the closed-form dC0 = -b_xt*h/(r0 tan theta) with the real fit.
"""
from __future__ import annotations
import numpy as np


class Params3D:
    def __init__(self, v=7500.0, H=720000.0, wl=0.25, r0=766206.0,
                 theta_inc=np.deg2rad(20), Nrx=4, prf_full=2000.0,
                 bxt=(0.0, 100.0, 200.0, 300.0), Nfull=8192, band_frac=0.7):
        self.vs = v; self.H = H; self.wl = wl; self.r0 = r0
        self.theta_inc = theta_inc; self.Nrx = Nrx; self.prf = prf_full
        self.bxt = np.asarray(bxt, float)
        self.bat = np.array([2.0 * i * v / prf_full for i in range(Nrx)])  # DPCA
        self.ta = np.arange(Nfull) / prf_full
        Ka = 2 * v**2 / (wl * r0)
        beta = np.arcsin(band_frac * (prf_full / 2) * wl / (2 * v))
        self.theta_tx = 2 * beta
        self.theta_rx = np.full(Nrx, 2 * beta)
        self.sq_tx = 0.0
        self.sq_rx = np.zeros(Nrx)

    def flat_point_at_range(self, r):
        """Ground point (z=0) at slant range r from the TX at (0,0,H)."""
        return np.array([0.0, np.sqrt(r**2 - self.H**2), 0.0])


# ---- build_tracks_3d (verbatim) -------------------------------------------
def build_tracks_3d(p, ta=None):
    if ta is None:
        ta = p.ta
    Na = len(ta); Nrx = p.Nrx; vs = p.vs
    ptx = np.column_stack([vs * ta, np.zeros(Na), np.full(Na, p.H)])
    vtx = np.full(Na, float(vs))
    prx = np.zeros([Nrx, Na, 3], np.float64); vrx = np.zeros([Nrx, Na], np.float64)
    for i in range(Nrx):
        prx[i, :, 0] = vs * ta - p.bat[i]
        prx[i, :, 1] = p.bxt[i]
        prx[i, :, 2] = p.H
        vrx[i, :] = vs
    return ptx, prx, vtx, vrx


# ---- get_coeff_nu_3d (verbatim) -------------------------------------------
def get_coeff_nu_3d(ptg, ptx, prx, vtx, vrx, prf, wl, ta,
                    sq_tx, sq_rx, theta_tx, theta_rx, N_time=2, dN_time=2):
    ptg = np.asarray(ptg, dtype=np.float64)
    rhT = np.sqrt(np.sum((ptx - ptg[None, :]) ** 2, axis=1))
    inst_sqT = np.arcsin(np.clip(np.gradient(rhT, 1.0 / prf) / vtx, -1.0, 1.0))
    validT = np.where(np.abs(inst_sqT) <= (sq_tx + theta_tx / 2))[0]
    rhR = np.sqrt(np.sum((prx - ptg[None, :]) ** 2, axis=1))
    inst_sqR = np.arcsin(np.clip(np.gradient(rhR, 1.0 / prf) / vrx, -1.0, 1.0))
    validR = np.where(np.abs(inst_sqR) <= (sq_rx + theta_rx / 2))[0]
    if validT.size < N_time + 3 or validR.size < dN_time + 3:
        return None
    taCommon = np.intersect1d(ta[validT], ta[validR])
    idx = np.nonzero(np.isin(ta, taCommon))[0]
    if idx.size < N_time + 3:
        return None
    rhMS = 2 * rhT[idx]; rhBS = (rhT + rhR)[idx]
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
    taC = taCommon
    rh_ms, ta_ms = rhMS[vld_ms], taC[vld_ms]
    rh_bs, ta_bs = rhBS[vld_bs], taC[vld_bs]
    i_ms, i_bs = int(np.argmin(rh_ms)), int(np.argmin(rh_bs))
    nr = int(np.min([len(rh_ms[i_ms:]), len(rh_bs[i_bs:])]))
    nl = int(np.min([len(rh_ms[:i_ms]), len(rh_bs[:i_bs])]))
    rh_ms, ta_ms = rh_ms[i_ms - nl:i_ms + nr], ta_ms[i_ms - nl:i_ms + nr]
    rh_bs, ta_bs = rh_bs[i_bs - nl:i_bs + nr], ta_bs[i_bs - nl:i_bs + nr]
    if rh_ms.size < N_time + 2 or rh_bs.size < dN_time + 2:
        return None
    tbc_ms = ta_ms[int(np.argmin(rh_ms))]; tbc_bs = ta_bs[int(np.argmin(rh_bs))]
    c_time = np.polyfit(ta_ms - tbc_ms, rh_ms, N_time)[::-1]
    dc_time = np.polyfit(ta_bs - tbc_bs, rh_bs - rh_ms, dN_time)[::-1]
    C0 = (dc_time[0] + c_time[1] ** 2 / 4 / c_time[2]
          - (c_time[1] + dc_time[1]) ** 2 / 4 / (c_time[2] + dc_time[2]))
    C1 = ((c_time[2] * dc_time[1] - c_time[1] * dc_time[2])
          / 2 / c_time[2] / (c_time[2] + dc_time[2]))
    C2 = dc_time[2] / 4 / c_time[2] / (c_time[2] + dc_time[2])
    Dt = tbc_bs - tbc_ms
    return float(C0), float(C1), float(C2), float(Dt)


def _coeff(p, tracks, ptg, ch):
    ptx, prx, vtx, vrx = tracks
    return get_coeff_nu_3d(ptg, ptx, prx[ch], vtx, vrx[ch], p.prf, p.wl, p.ta,
                           p.sq_tx, p.sq_rx[ch], p.theta_tx, p.theta_rx[ch])


def residual_C0_3d(p, tracks, ptg_true, channel):
    """dC0 [m] = C0(true target) - C0(flat point at the SAME slant range)."""
    r = float(np.sqrt(ptg_true[1] ** 2 + (p.H - ptg_true[2]) ** 2))
    ptg_flat = p.flat_point_at_range(r); ptg_flat[0] = ptg_true[0]
    a = _coeff(p, tracks, np.asarray(ptg_true, float), channel)
    b = _coeff(p, tracks, ptg_flat, channel)
    if a is None or b is None:
        return 0.0
    return float(a[0] - b[0])
