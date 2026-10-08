"""
Previous per-sub-band coefficient fit (kept for reference, no longer used by the pipeline):
the parabolas were fitted only over the samples whose instantaneous look angle falls in the
angular window of sub-band k, and the vertex was extrapolated. Every channel observes every
point over the whole integration time, so the pipeline now uses the full-history value
(sata1d.sata.residual_C0) for every sub-band; in the geometries tested the two agree to
< 1e-3 mm.
"""
import numpy as np

from sata1d.subband import subband_frequency_beam


def getcoeff_beam(ptg, ptx, prx, vtx, vrx, prf, wl, ta, beta_lo, beta_hi, N_time=2, dN_time=2):
    """GetCoeffNu restricted to the samples whose instantaneous squint is in
    [beta_lo, beta_hi]. Returns (C0, C1, C2, Dt) or None if the window is too short."""
    rhT = np.sqrt(np.sum((ptx - ptg[None, :]) ** 2, axis=1))
    inst_sqT = np.arcsin(np.clip(np.gradient(rhT, 1 / prf) / vtx, -1.0, 1.0))
    valid_T = np.where((inst_sqT >= beta_lo) & (inst_sqT <= beta_hi))[0]
    rhR = np.sqrt(np.sum((prx - ptg[None, :]) ** 2, axis=1))
    inst_sqR = np.arcsin(np.clip(np.gradient(rhR, 1 / prf) / vrx, -1.0, 1.0))
    valid_R = np.where((inst_sqR >= beta_lo) & (inst_sqR <= beta_hi))[0]
    if valid_T.size < N_time + 3 or valid_R.size < dN_time + 3:
        return None
    taCommon = np.intersect1d(ta[valid_T], ta[valid_R])
    idx = np.nonzero(np.isin(ta, taCommon))[0]
    if idx.size < N_time + 3:
        return None
    rhMS, rhBS = 2 * rhT[idx], (rhT + rhR)[idx]
    fi_ms = -1 / wl * np.diff(rhMS) * prf
    fi_bs = -1 / wl * np.diff(rhBS) * prf
    if fi_ms.size == 0 or fi_bs.size == 0:
        return None
    f_max = np.min([abs(np.max(fi_ms)), abs(np.max(fi_bs))])
    f_min = -(np.min([abs(np.min(fi_ms)), abs(np.min(fi_bs))]))
    vld_ms = np.where((fi_ms < f_max) & (fi_ms > f_min))[0]
    vld_bs = np.where((fi_bs < f_max) & (fi_bs > f_min))[0]
    if vld_ms.size < N_time + 2 or vld_bs.size < dN_time + 2:
        return None
    rh_ms, ta_ms = rhMS[vld_ms], taCommon[vld_ms]
    rh_bs, ta_bs = rhBS[vld_bs], taCommon[vld_bs]
    i_ms, i_bs = int(np.argmin(rh_ms)), int(np.argmin(rh_bs))
    nr = int(np.min([len(rh_ms[i_ms:]), len(rh_bs[i_bs:])]))
    nl = int(np.min([len(rh_ms[:i_ms]), len(rh_bs[:i_bs])]))
    rh_ms, ta_ms = rh_ms[i_ms - nl:i_ms + nr], ta_ms[i_ms - nl:i_ms + nr]
    rh_bs, ta_bs = rh_bs[i_bs - nl:i_bs + nr], ta_bs[i_bs - nl:i_bs + nr]
    if rh_ms.size < N_time + 2 or rh_bs.size < dN_time + 2:
        return None
    tbc_ms = ta_ms[np.where(rh_ms == min(rh_ms))[0][0]]
    tbc_bs = ta_bs[np.where(rh_bs == min(rh_bs))[0][0]]
    c = np.polyfit(ta_ms - tbc_ms, rh_ms, N_time)[::-1]
    dc = np.polyfit(ta_bs - tbc_bs, rh_bs - rh_ms, dN_time)[::-1]
    C0 = dc[0] + c[1] ** 2 / 4 / c[2] - (c[1] + dc[1]) ** 2 / 4 / (c[2] + dc[2])
    C1 = (c[2] * dc[1] - c[1] * dc[2]) / 2 / c[2] / (c[2] + dc[2])
    C2 = dc[2] / 4 / c[2] / (c[2] + dc[2])
    return C0, C1, C2, tbc_bs - tbc_ms


def residual_C0_subband(cfg, tracks, ptg_real, channel, k):
    """C0(scatterer) - C0(reference) over the look angles of sub-band k [m] (0 if not fittable)."""
    _f, _b, blo, bhi = subband_frequency_beam(cfg, k)
    args = (tracks.ptx, tracks.prx[channel], tracks.vtx, tracks.vrx[channel],
            cfg.prf, cfg.system.wl, cfg.ta, blo, bhi)
    real, ref = getcoeff_beam(ptg_real, *args), getcoeff_beam(cfg.scene.ptg, *args)
    if real is None or ref is None:
        return 0.0
    return float(real[0] - ref[0])


