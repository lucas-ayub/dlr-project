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


def sata_kernel(data, delta_C0_array, Tsubeff, hop, Nzp, azpos, wl, inverse=False, scheme="cola"):
    """
    Sub-aperture loop shared by the whole-band and the sub-band kernels.
    scheme="cola"      : triangular analysis window, zero-padded FFT, overlap-add
                         normalised by the window sum (constant overlap-add);
    scheme="crossfade" : rectangular block, zero-padded FFT, correction, then the
                         corrected blocks are cross-faded with the linear weights of
                         the reference code (triangular synthesis window);
    scheme="margin"    : as "crossfade" but the FFT block holds the Nzp real samples
                         around the sub-aperture (the IDL margin), of which only the
                         central Tsubeff are kept.
    The three are the identity when delta_C0_array == 0.
    """
    data = np.asarray(data, dtype=complex)
    dimx = len(data)
    win = triangular_window(Tsubeff)
    out = np.zeros(dimx, dtype=complex)
    wsum = np.zeros(dimx)
    for start in range(0, dimx, hop):
        seg = data[start:start + Tsubeff]
        L = len(seg)
        if L < 2:
            break
        centre = start + 0.5 * Tsubeff
        posaux = np.clip(np.round(azpos + centre).astype(int), 0, dimx - 1)
        ph = -2.0 * np.pi / wl * delta_C0_array[posaux]
        ph[~np.isfinite(ph)] = 0.0
        rot = np.exp(-1j * ph) if inverse else np.exp(1j * ph)
        if scheme == "margin":
            lo = int(round(centre - Nzp / 2))
            idx = np.clip(np.arange(lo, lo + Nzp), 0, dimx - 1)
            buf = data[idx] * ((np.arange(lo, lo + Nzp) >= 0) & (np.arange(lo, lo + Nzp) < dimx))
            corr = np.fft.ifft(np.fft.fft(buf) * rot)[start - lo:start - lo + L]
        else:
            buf = np.zeros(Nzp, dtype=complex)
            buf[:L] = seg * (win[:L] if scheme == "cola" else 1.0)
            corr = np.fft.ifft(np.fft.fft(buf) * rot)[:L]
        if scheme == "cola":
            out[start:start + L] += corr
        else:
            out[start:start + L] += corr * win[:L]
        wsum[start:start + L] += win[:L]
    nz = wsum > 1e-12
    out[nz] /= wsum[nz]
    out[~nz] = data[~nz]
    return out


def sata_1d(data, delta_C0_array, rref, prf, v, wl, r, squint=0.0, inverse=False, sata_osf=1,
            scheme="cola"):
    """
    data           : (naz,) complex azimuth line of one channel at prf (= PRF_op)
    delta_C0_array : (naz,) residual range [m] per azimuth cell
    inverse=True   : remove the residual (spec *= exp(-j ph)); False injects it.
    """
    data = np.asarray(data, dtype=complex)
    Tsubeff, hop, Nzp = subaperture_sizing(rref, prf, v, wl, sata_osf)
    if Tsubeff <= 2:
        return data.copy()
    fsub, azpos = subaperture_axis(prf, Nzp, v, wl, r, squint)
    return sata_kernel(data, delta_C0_array, Tsubeff, hop, Nzp, azpos, wl, inverse, scheme)


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


def delta_C0_per_target(cfg: ExperimentConfig, tracks: PlatformTracks):
    """delta_C0 [m] of every extra scatterer of the scene in every channel, shape [Np, Nrx],
    from the full range history (residual_C0). Computed once and shared by all the maps."""
    pts = cfg.scene.points[1:]
    return np.array([[residual_C0(cfg, tracks, p, i) for i in range(cfg.Nrx)] for p in pts])


def lookup_table(cfg: ExperimentConfig, tracks: PlatformTracks, channel, naz=None, dC0=None, max_gap=2):
    """
    Look-up table of one channel: the scene IS the targets, so the table holds, at the cell
    of every target (az_pixel_of_scatterer, with the channel's bat/2 shift), that target's
    delta_C0 from the full range history; the mean where several targets share a cell.
    A cell without a target takes the value of the nearest target if that target is at most
    max_gap cells away (a dense grid of targets then fills its whole extent); zero elsewhere.
    No fold images: the same table serves the whole-band kernel and every sub-band kernel.
    dC0: delta_C0_per_target(cfg, tracks), computed here if not given.
    """
    naz = cfg.Na_ch if naz is None else naz
    arr = np.zeros(naz)
    if not cfg.scene.extra_offsets:
        return arr
    dC0 = delta_C0_per_target(cfg, tracks) if dC0 is None else dC0
    cells = np.array([az_pixel_of_scatterer(cfg, dx, channel) for (dx, _, _) in cfg.scene.extra_offsets])
    acc, cnt = np.zeros(naz), np.zeros(naz)
    ok = (cells >= 0) & (cells < naz)
    np.add.at(acc, cells[ok], dC0[ok, channel]); np.add.at(cnt, cells[ok], 1.0)
    filled = cnt > 0
    arr[filled] = acc[filled] / cnt[filled]
    if max_gap > 0 and filled.any():
        idx = np.flatnonzero(filled)
        n = np.arange(naz)
        j = np.clip(np.searchsorted(idx, n), 1, len(idx) - 1)
        left, right = idx[j - 1], idx[j]
        nearest = np.where(np.abs(n - left) <= np.abs(right - n), left, right)
        gap = np.abs(n - nearest)
        fill = ~filled & (gap <= max_gap)
        arr[fill] = arr[nearest[fill]]
    return arr


def terrain_lookup_table(cfg: ExperimentConfig, tracks: PlatformTracks, channel, height_fn, naz=None,
                         step=8, x_max=None):
    """
    Look-up table of one channel from a terrain h(x) (x = azimuth relative to the reference
    point): at every cell, the delta_C0 (full range history, residual_C0) of a point placed on
    the iso-range surface at the height of the terrain there. Computed every step-th cell and
    interpolated linearly in between. x_max=None fills the whole line; with a value, the table
    is zero beyond |x| > x_max. The whole line is needed: the sub-band kernels read the table up
    to Nsb - 1 folds (3 X = 18.7 km) away from a target, the whole-band kernel up to one fold.
    The cell index follows az_pixel_of_scatterer (bat/2 shift of the channel).
    """
    naz = cfg.Na_ch if naz is None else naz
    ds = cfg.system.vs / cfg.PRF_op
    cells = np.arange(naz)
    x = (cells - naz / 2) * ds - 0.5 * cfg.array.bat[channel] - cfg.scene.x0
    H, r0, y0 = cfg.scene.H, cfg.scene.r0, cfg.scene.y0
    inside = np.arange(naz) if x_max is None else np.flatnonzero(np.abs(x) <= x_max)
    coarse = np.unique(np.r_[inside[::step], inside[-1]])
    h = np.asarray(height_fn(x[coarse]), float)
    val = np.zeros(len(coarse))
    for n, c in enumerate(coarse):
        if h[n] != 0.0:
            p = cfg.scene.ptg + np.array([x[c], np.sqrt(r0 ** 2 - (H - h[n]) ** 2) - y0, h[n]])
            val[n] = residual_C0(cfg, tracks, p, channel)
    arr = np.zeros(naz)
    arr[inside] = np.interp(cells[inside], coarse, val)
    return arr


def build_delta_C0_array(cfg: ExperimentConfig, tracks: PlatformTracks, channel, naz=None,
                         mode="footprint", residual_fn=None, f_centre=0.0, dC0=None):
    """
    delta_C0 map [naz] of one channel from the scene's extra scatterers.
    mode="lut":     look-up table (lookup_table): the targets' delta_C0 at the targets' cells,
                    no fold images; the same for every kernel (f_centre is ignored);
    mode="footprint": footprint_map (block +- h_w at every scatterer and at its fold images,
                    zero elsewhere), images placed for a kernel whose Doppler centre is f_centre;
    mode="hold":    previous map (interpolated between scatterers, held beyond, no images).
    dC0: delta_C0_per_target(cfg, tracks) [Np, Nrx], computed once and shared; if not given it
    is computed here (residual_fn(ptg, channel) may override residual_C0 for that computation).
    """
    naz = cfg.Na_ch if naz is None else naz
    if not cfg.scene.extra_offsets:
        return np.zeros(naz)
    if dC0 is None:
        residual_fn = residual_fn or (lambda p, ch: residual_C0(cfg, tracks, p, ch))
        dC0 = np.array([[residual_fn(cfg.scene.ptg + np.array(o), i) for i in range(cfg.Nrx)]
                        for o in cfg.scene.extra_offsets])
    if mode == "lut":
        return lookup_table(cfg, tracks, channel, naz, dC0)
    ds = cfg.system.vs / cfg.PRF_op
    by_pixel = defaultdict(list)
    for j, (dx, dy, dh) in enumerate(cfg.scene.extra_offsets):
        pix = az_pixel_of_scatterer(cfg, dx, channel) if mode == "footprint" \
            else int(round(cfg.Na_ch / 2 + dx / ds))
        by_pixel[pix].append(float(dC0[j, channel]))
    if mode == "hold":
        return hold_map(by_pixel, naz)
    return footprint_map(by_pixel, naz, sata_footprint_halfwidth(cfg), sata_image_offsets(cfg, f_centre))


def sata_channels(cfg: ExperimentConfig, tracks: PlatformTracks, s_channel, remove=True,
                  sata_osf=4, mode="footprint", scheme="cola", dC0=None):
    """Whole-band SATA: correct every channel once, before the reconstruction.
    dC0: delta_C0_per_target (computed once here if not given)."""
    out = np.asarray(s_channel, dtype=complex).copy()
    dC0 = delta_C0_per_target(cfg, tracks) if dC0 is None else dC0
    for kk in range(cfg.Nrx):
        M = build_delta_C0_array(cfg, tracks, kk, mode=mode, dC0=dC0)
        out[kk] = sata_1d(out[kk], M, rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs,
                          wl=cfg.system.wl, r=cfg.scene.r0, squint=cfg.sq_tx,
                          inverse=remove, sata_osf=sata_osf, scheme=scheme)
    return out
