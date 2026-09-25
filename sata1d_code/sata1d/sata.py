"""
SATA 1D (C0 term): position-dependent phase correction of one azimuth line.

Two stages per channel:
  A) build_delta_C0_array : map delta_C0[x] (correction for energy that the
     sub-aperture ruler assigns to cell x); non-zero only where a scatterer
     appears in the sub-aperture spectra, zero elsewhere.
  B) sata_1d              : sub-aperture loop; every Doppler bin is rotated by
     exp(-j * (-2 pi / wl) * delta_C0[cell the bin points to]).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import ExperimentConfig
from .geometry import PlatformTracks
from .reconstruction import GetCoeffNu


# ---------------------------------------------------------------------------
# Stage B: kernel
# ---------------------------------------------------------------------------
def subaperture_sizing(rref, prf, v, wl, sata_osf=1):
    """Sub-aperture length Tsubeff (even), hop Tsub (50 % overlap), FFT size Nzp."""
    deltax = np.sqrt(wl * rref / 2.0)                      # SATA resolution on ground [m]
    Tsubeff = int(np.round(deltax * prf / v * 0.5) * 2)
    Nzp = int(2 ** np.ceil(np.log2(Tsubeff)) * sata_osf)
    return Tsubeff, Tsubeff // 2, Nzp


def subaperture_axis(prf, Nzp, v, wl, r, squint=0.0):
    """Doppler of each FFT bin (np.fft.fft order) and the cell offset it points to:
    f -> beta = asin(wl f / 2v) -> r (tan beta - tan squint) / v * prf."""
    fc = 2.0 * v / wl * np.sin(squint)
    pfc = np.round(np.mod(fc, prf) * Nzp / prf)
    dfc = np.round(fc * Nzp / prf) * prf / Nzp
    fsub = np.arange(Nzp) * prf / Nzp - prf * 0.5 + dfc
    fsub[0] = prf * 0.5 + dfc
    fsub = np.roll(fsub, int(Nzp / 2 + pfc))
    betasub = np.arcsin(np.clip(wl * fsub / (2.0 * v), -1.0, 1.0))
    return fsub, r * (np.tan(betasub) - np.tan(squint)) / v * prf


def triangular_window(Tsubeff):
    half = np.arange(Tsubeff // 2) / (Tsubeff // 2 - 1)
    return np.concatenate((half, half[::-1]))


def sata_1d(data, delta_C0_array, rref, prf, v, wl, r, squint=0.0, inverse=False, sata_osf=1):
    """
    data           : (naz,) complex azimuth line of one channel at prf (= PRF_op)
    delta_C0_array : (naz,) residual range [m] per azimuth cell
    inverse=True   : remove the residual (spec *= exp(-j ph)); False injects it.
    """
    data = np.asarray(data, dtype=complex)
    dimx = len(data)
    Tsubeff, hop, Nzp = subaperture_sizing(rref, prf, v, wl, sata_osf)
    if Tsubeff <= 2:
        return data.copy()
    win = triangular_window(Tsubeff)

    fsub, azpos = subaperture_axis(prf, Nzp, v, wl, r, squint)

    out = np.zeros(dimx, dtype=complex)
    wsum = np.zeros(dimx)
    for start in range(0, dimx, hop):
        seg = data[start:start + Tsubeff]
        L = len(seg)
        if L < 2:
            break
        buf = np.zeros(Nzp, dtype=complex)
        buf[:L] = seg * win[:L]
        spec = np.fft.fft(buf)
        posaux = np.clip(np.round(azpos + start + 0.5 * Tsubeff).astype(int), 0, dimx - 1)
        ph = -2.0 * np.pi / wl * delta_C0_array[posaux]
        ph[~np.isfinite(ph)] = 0.0
        spec *= np.exp(-1j * ph) if inverse else np.exp(1j * ph)
        out[start:start + L] += np.fft.ifft(spec)[:L]
        wsum[start:start + L] += win[:L]
    nz = wsum > 1e-12
    out[nz] /= wsum[nz]
    out[~nz] = data[~nz]
    return out


# ---------------------------------------------------------------------------
# Stage A: delta_C0 map
# ---------------------------------------------------------------------------
def residual_C0(cfg: ExperimentConfig, tracks: PlatformTracks, ptg_real, channel):
    """C0(scatterer) - C0(reference point) for one channel [m]."""
    common = (tracks.ptx, tracks.prx[channel], tracks.vtx, tracks.vrx[channel],
              tracks.ptx, tracks.vtx, cfg.prf, cfg.system.wl, cfg.ta,
              cfg.sq_tx, cfg.sq_rx[channel], cfg.theta_tx, cfg.theta_rx[channel])
    return float(GetCoeffNu(ptg_real, *common)[0] - GetCoeffNu(cfg.scene.ptg, *common)[0])


def az_pixel_of_scatterer(cfg: ExperimentConfig, dx, channel):
    """Cell of scatterer x0 + dx on the per-channel line; the Tx-Rx phase centre
    trails the transmitter by bat/2."""
    ds = cfg.system.vs / cfg.PRF_op
    return int(round(cfg.Na_ch / 2 + (cfg.scene.x0 + dx + 0.5 * cfg.array.bat[channel]) / ds))


def sata_footprint_halfwidth(cfg: ExperimentConfig, squint=0.0):
    """Half-width [cells] of one scatterer in a sub-aperture spectrum: triangular
    window up to its 2nd null (4 PRF_op / Tsubeff), mapped through the ruler."""
    prf, v, wl, r = cfg.PRF_op, cfg.system.vs, cfg.system.wl, cfg.scene.r0
    Tsubeff, _, _ = subaperture_sizing(r, prf, v, wl)
    f_edge = 4.0 * prf / Tsubeff
    fc = 2 * v / wl * np.sin(squint)
    b0 = np.arcsin(np.clip(wl * fc / (2 * v), -1, 1))
    b1 = np.arcsin(np.clip(wl * (fc + f_edge) / (2 * v), -1, 1))
    return int(np.ceil(r * (np.tan(b1) - np.tan(b0)) / v * prf))


def sata_image_offsets(cfg: ExperimentConfig, f_centre=0.0):
    """Offsets [cells] where one scatterer appears for a kernel whose bins are labelled
    with the absolute Doppler in [f_centre - PRF_op/2, f_centre + PRF_op/2).
    A true Doppler f (|f| <= B/2, B = 4 v sin(theta/2)/wl) is labelled f + m PRF_op,
    so it lands m*X cells away, X = cell shift of one PRF_op fold.
    Whole band (f_centre = 0): {-X, 0, +X}; sub-band k: one-sided set."""
    prf, v, wl, r = cfg.PRF_op, cfg.system.vs, cfg.system.wl, cfg.scene.r0
    X = r * np.tan(np.arcsin(np.clip(wl * prf / (2 * v), -1, 1))) / v * prf
    B = 4.0 * v * np.sin(cfg.theta_tx / 2.0) / wl
    m_lo = int(np.round((f_centre - B / 2) / prf))
    m_hi = int(np.round((f_centre + B / 2) / prf))
    return np.array([m * X for m in range(m_lo, m_hi + 1)], dtype=float)


def footprint_map(values_by_pixel: dict, naz: int, halfwidth: int, offsets=(0.0,)):
    """
    1) terrain blocks: scatterers whose footprints touch (gap <= 2 hw) form one block,
       linearly interpolated between them over [first - hw, last + hw]; an isolated
       scatterer gives a constant block pixel +- hw;
    2) images: every block is drawn at each offset (0 = direct, m X = folded copies);
       where images overlap, a cell takes the image whose core [first, last] + offset
       is nearest (ties: the direct image, then the smaller |offset|), so that the
       main lobe of each component reads its own value;
    3) zero elsewhere.
    """
    arr = np.zeros(naz)
    if not values_by_pixel:
        return arr
    pix = np.array(sorted(values_by_pixel), dtype=float)
    val = np.array([max(values_by_pixel[p], key=abs) for p in sorted(values_by_pixel)])
    cuts = np.flatnonzero(np.diff(pix) > 2 * halfwidth) + 1
    blocks = list(zip(np.split(pix, cuts), np.split(val, cuts)))
    dist = np.full(naz, np.inf)
    for off in sorted(offsets, key=abs):
        sh = float(np.round(off))
        for bp, bv in blocks:
            lo = int(max(0, np.floor(bp[0] + sh - halfwidth)))
            hi = int(min(naz - 1, np.ceil(bp[-1] + sh + halfwidth)))
            if lo > hi:
                continue
            x = np.arange(lo, hi + 1)
            d = np.maximum(0.0, np.maximum(bp[0] + sh - x, x - (bp[-1] + sh)))
            take = d < dist[x]
            arr[x[take]] = np.interp(x[take] - sh, bp, bv)
            dist[x[take]] = d[take]
    return arr


def hold_map(values_by_pixel: dict, naz: int):
    """Previous map (kept for comparison): one value per pixel, linearly
    interpolated between pixels and held constant beyond them."""
    pix = sorted(values_by_pixel)
    ys = np.array([max(values_by_pixel[p], key=abs) for p in pix])
    if len(pix) == 1:
        return np.full(naz, ys[0])
    return np.interp(np.arange(naz), np.array(pix, float), ys, left=ys[0], right=ys[-1])


def build_delta_C0_array(cfg: ExperimentConfig, tracks: PlatformTracks, channel, naz=None,
                         mode="footprint", residual_fn=None, f_centre=0.0):
    """
    delta_C0 map [naz] of one channel from the scene's extra scatterers.
    mode="footprint" (default): footprint_map; mode="hold": previous map.
    residual_fn(ptg, channel) overrides residual_C0 (used per sub-band);
    f_centre is the Doppler centre of the kernel that will read the map.
    """
    naz = cfg.Na_ch if naz is None else naz
    if not cfg.scene.extra_offsets:
        return np.zeros(naz)
    residual_fn = residual_fn or (lambda p, ch: residual_C0(cfg, tracks, p, ch))
    ds = cfg.system.vs / cfg.PRF_op
    by_pixel = defaultdict(list)
    for (dx, dy, dh) in cfg.scene.extra_offsets:
        p = cfg.scene.ptg + np.array([dx, dy, dh])
        pix = az_pixel_of_scatterer(cfg, dx, channel) if mode == "footprint" \
            else int(round(cfg.Na_ch / 2 + dx / ds))
        by_pixel[pix].append(residual_fn(p, channel))
    if mode == "hold":
        return hold_map(by_pixel, naz)
    return footprint_map(by_pixel, naz, sata_footprint_halfwidth(cfg), sata_image_offsets(cfg, f_centre))


def sata_channels(cfg: ExperimentConfig, tracks: PlatformTracks, s_channel, remove=True,
                  sata_osf=4, mode="footprint"):
    """Whole-band SATA: correct every channel once, before the reconstruction."""
    out = np.asarray(s_channel, dtype=complex).copy()
    for kk in range(cfg.Nrx):
        dC0 = build_delta_C0_array(cfg, tracks, kk, mode=mode)
        out[kk] = sata_1d(out[kk], dC0, rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs,
                          wl=cfg.system.wl, r=cfg.scene.r0, squint=cfg.sq_tx,
                          inverse=remove, sata_osf=sata_osf)
    return out
