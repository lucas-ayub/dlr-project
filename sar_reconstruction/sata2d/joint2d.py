# -*- coding: utf-8 -*-
"""
Joint reconstruction in 2-D (topography-dependent reconstruction of a 3-D scene with a DEM).

The 2-D version of ``sata1d/joint_line.py``. The unknown is the focused image of the scene, ``g[n, m]``: azimuth
position ``x_n = vs * ta[n]`` (full PRF) and slant range of closest approach ``r_m = r_scan[m]``. The range-compressed
data of channel ``i`` is written from it as

    s_i = decimate_Nrx{ K_i [ M ( Q_i ( S_i g ) ) ] }

    S_i   the terrain, at every pixel: the phase exp(-j 2 pi dC0_i(x, r) / wl) of the topographic residual and the
          shift of the range envelope by dC0_i / 2 (plus the part of the flat-earth offset C0_i(r) / 2 that changes
          along the swath). dC0_i(x, r) comes from the DEM only (``dem_maps``).
    Q_i   in the range-Doppler domain, on the range line of the scatterer: what changes with the range of closest
          approach, relative to the reference range -- the azimuth chirp (exp(-j 4 pi (r - r_ref)(D(f) - 1) / wl),
          D = sqrt(1 - (wl f / 2 vs)^2)) and the coefficients (C0, C1, C2, Dt) of the channel.
    M, K_i in the 2-D frequency domain: the response of channel i to the reference point at the reference range
          (simulated with the system geometry: azimuth chirp, antenna windows of the pair, range cell migration, the
          range offset C0_i / 2 of the channel).

With dC0 = 0 this is the model that the two-step reconstruction inverts. With a terrain, the Nrx channels are solved
together for the image by regularised least squares with conjugate gradients, exactly as in the 1-D joint: every
iteration applies the model once and its adjoint once. The adjoint is "focus every channel (range cell migration
correction and azimuth compression), remove the phase of the terrain, add the channels".

The range band common to the channels. A cross-track baseline shifts the part of the range spectrum of the image that
a channel observes (the wavenumber shift of interferometry): by -(c / 2 wl) d(C0_i + dC0_i) / dr, which grows with the
baseline and with the slope of the terrain towards the radar. An image range frequency that fewer channels observe
than there are azimuth folds to separate is not determined by the data, whatever the method. The output therefore
keeps, at every pixel, only the range frequencies that the channels observe there (``Model2D.common_band`` and
``Model2D.band_limit``): the price is range resolution, by the fraction baseline / critical baseline of the pixel.

Nothing about the targets is used: only the system geometry and the DEM.

    from sata2d.joint2d import dem_maps, reconstruct_joint_2d
    dC0, hmap = dem_maps(p, tracks, terrain)                       # terrain: h(x, dy) [m], x azimuth, dy ground range from the reference point
    rec = reconstruct_joint_2d(p, tracks, channels_rc, dC0)        # [Na, Nr] as reconstruct_subband_2d
"""
from __future__ import annotations

import warnings

import numpy as np

from .geometry import CoeffTable3D, get_raw_data_3d, residual_C0_3d, C0_LIGHT
from .reconstruction import range_compress

__all__ = ["image_positions", "dem_heights", "height_table", "dem_maps", "Model2D", "energy", "processed_band", "image_to_data",
           "reconstruct_joint_2d"]


# ----------------------------------------------------------------------------------------------- the DEM in radar geometry
def image_positions(p, osr=2):
    """(x [Na], r [osr * Nr]): azimuth position of every image row (x = vs * ta: row Na // 2 is the position of the
    transmitter at ta = 0) and slant range of closest approach of every image column. The image is sampled ``osr``
    times finer in range than the data (see ``Model2D``)."""
    return p.vs * p.ta, p.r_scan[0] + np.arange(osr * p.Nr) * (C0_LIGHT / (2.0 * p.rsf)) / osr


def dem_heights(p, terrain, x=None, r=None, iterations=40, osr=2):
    """Height of the terrain at every image pixel, [len(x), len(r)]: the point of the terrain h(x, dy) (dy = ground
    range from the reference point) that sits at slant range r of closest approach, y^2 + (H - h)^2 = r^2 with
    y = y0 + dy. Fixed-point iteration from the flat earth: it converges where the slope of the terrain along ground
    range is below tan(incidence), towards or away from the radar. Also returns the ground-range offset dy."""
    xg, rg = image_positions(p, osr)
    x = xg if x is None else np.asarray(x, float)
    r = rg if r is None else np.asarray(r, float)
    X = np.broadcast_to(x[:, None], (len(x), len(r)))
    R = np.broadcast_to(r[None, :], (len(x), len(r)))
    h = np.full(X.shape, float(p.h0))
    for _ in range(iterations):
        y = np.sqrt(np.clip(R ** 2 - (p.H - h) ** 2, 0.0, None))
        h_new = np.asarray(terrain(X, y - p.y0), float)
        change, h = float(np.max(np.abs(h_new - h))), h_new
        if change < 1e-4:
            break
    else:
        warnings.warn(f"dem_heights: not converged after {iterations} iterations (last change {change:.3g} m); slope along range close to tan(incidence)?")
    y = np.sqrt(np.clip(R ** 2 - (p.H - h) ** 2, 0.0, None))
    return h, y - p.y0


def height_table(p, tracks, h_min, h_max, n_h=17, n_r=5):
    """dC0 of every channel on a grid of (slant range, height): residual_C0_3d of a point at that height and range.
    Returns (ranges [n_r], heights [n_h], table [n_r, n_h, Nrx])."""
    rg = np.linspace(p.r_scan[0], p.r_scan[-1], n_r)
    hg = np.linspace(min(h_min, p.h0) - 1.0, max(h_max, p.h0) + 1.0, n_h)
    tab = np.zeros((n_r, n_h, p.Nrx))
    for a, r in enumerate(rg):
        for b, h in enumerate(hg):
            ptg = np.array([p.x0, np.sqrt(max(r ** 2 - (p.H - h) ** 2, 0.0)), h])
            tab[a, b] = [residual_C0_3d(p, tracks, ptg, i) for i in range(p.Nrx)]
    return rg, hg, tab


def dem_maps(p, tracks, terrain, table=None, osr=2):
    """dC0 [Nrx, Na, osr * Nr] of every channel at every image pixel, from the terrain h(x, dy) only, and the height
    map [Na, osr * Nr] of the terrain in radar geometry. The table is fitted with a cubic in the height at every table
    range and interpolated linearly in range."""
    h, _ = dem_heights(p, terrain, osr=osr)
    rg, hg, tab = height_table(p, tracks, h.min(), h.max()) if table is None else table
    r = image_positions(p, osr)[1]
    w = np.clip((r - rg[0]) / (rg[-1] - rg[0]) * (len(rg) - 1), 0.0, len(rg) - 1 - 1e-9)
    k = w.astype(int)
    f = (w - k)[None, :]
    out = np.zeros((p.Nrx,) + h.shape)
    for i in range(p.Nrx):
        pol = [np.polyfit(hg, tab[a, :, i], 3) for a in range(len(rg))]
        val = np.stack([np.polyval(c, h) for c in pol])                          # [n_r, Na, Nr]: the cubic of every table range
        cols = np.arange(len(r))
        out[i] = val[k, :, cols].T * (1.0 - f) + val[k + 1, :, cols].T * f
    return out, h


# ----------------------------------------------------------------------------------------------- the model and its adjoint
class Model2D:
    """channels = A g and its adjoint (module docstring).

    g is [Na, osr * Nr]: rows in azimuth order (row n at x = vs ta[n]), columns at the slant ranges of
    ``image_positions`` -- ``osr`` times finer than the range bins of the data. The phase of the terrain is multiplied
    on this fine grid: a phase that changes along range moves the range spectrum of the image (the wavenumber shift of
    a cross-track baseline), and on the grid of the data that shift would wrap around as soon as it exceeds the unused
    margin of the sampled band (rsf - rbw) / 2.
    The channels are handled as their 2-D spectra (``data``): the decimation in azimuth is then the sum of the Nrx
    folds of the azimuth spectrum, and no inverse transform is needed inside the iterations."""

    def __init__(self, p, tracks, dC0, shift_order=1, n_nodes=16, dtype=np.complex64):
        self.p = p
        Na, Nr, Nrx, wl = p.Na, p.Nr, p.Nrx, p.wl
        dC0 = np.asarray(dC0)
        if dC0.ndim != 3 or dC0.shape[:2] != (Nrx, Na) or dC0.shape[2] % Nr:
            raise ValueError(f"dC0 must have shape (Nrx, Na, osr * Nr) = ({Nrx}, {Na}, osr * {Nr})")
        self.osr = osr = dC0.shape[2] // Nr
        self.Nr2 = Nr2 = osr * Nr
        self.shape = (Na, Nr2)
        self.dtype = dtype
        self.shift_order = int(shift_order)
        self.mref = Nr // 2
        self.r_ref = float(p.r_scan[self.mref])
        self.r_img = image_positions(p, osr)[1]
        self.cn = float(np.sqrt(p.Na_ch * Nr))                                    # the 2-D FFT of a channel, made unitary
        # range-frequency bins of the data inside the (osr times wider) range spectrum of the image
        self.keep = np.r_[np.arange(Nr // 2), Nr2 - Nr // 2 + np.arange(Nr // 2)]
        ptx, prx, vtx, vrx = tracks
        pref = p.flat_point_at_range(self.r_ref)
        fa = np.fft.fftfreq(Na, 1.0 / p.prf)
        fr = np.fft.fftfreq(Nr, 1.0 / p.rsf)
        # response of every channel (and of the monostatic system) to the reference point, full rate, 2-D spectrum,
        # with the point moved to the origin of the azimuth axis (row Na // 2 -> lag 0) and to range bin 0 of the data
        centre = lambda e: np.fft.fft2(np.roll(range_compress(e, p.cd, p.rbw, p.rsf, axis=1), (-(Na // 2), -self.mref), axis=(0, 1))).astype(dtype)
        self.K0 = centre(get_raw_data_3d(p, pref[None, :], ptx, ptx, vtx, vtx, p.ta, p.prf))
        self.K = [centre(get_raw_data_3d(p, pref[None, :], ptx, prx[i], vtx, vrx[i], p.ta, p.prf)) for i in range(Nrx)]
        # what changes with the range of closest approach, in the range-Doppler domain (image range lines)
        D = np.sqrt(1.0 - (wl * fa / (2.0 * p.vs)) ** 2)
        E = np.exp(-4j * np.pi * (self.r_img[None, :] - self.r_ref) * (D[:, None] - 1.0) / wl)
        self.E = E.astype(dtype)
        tab = CoeffTable3D(p, tracks, n_nodes=n_nodes)
        coef = np.array([[tab(r, i) for i in range(Nrx)] for r in self.r_img])   # [Nr2, Nrx, (C0, C1, C2, Dt)]
        for m in np.flatnonzero(self.r_img > p.r_scan[-1] + 1e-6):               # fine columns after the last range bin: the table
            coef[m] = 2.0 * coef[m - 1] - coef[m - 2]                            # ends there, continue it as a straight line
        d = coef - np.array([tab(self.r_ref, i) for i in range(Nrx)])[None]      # relative to the reference range
        self.Q = [(E * np.exp(-2j * np.pi * (d[:, i, 0][None, :] / wl + (-d[:, i, 1] + d[:, i, 3])[None, :] * fa[:, None]
                                             + d[:, i, 2][None, :] * wl * fa[:, None] ** 2))).astype(dtype) for i in range(Nrx)]
        # the terrain: phase screen and shift of the range envelope [m] (half of the path difference)
        self.screen = [np.exp(-2j * np.pi * dC0[i] / wl).astype(dtype) for i in range(Nrx)]
        self.delta = [(0.5 * (dC0[i] + d[:, i, 0][None, :])).astype(np.float32) for i in range(Nrx)]
        self.jfr = (-4j * np.pi * fr / C0_LIGHT)[None, :].astype(dtype)          # a shift of the envelope by delta [m]: exp(jfr * delta)

    # -- pieces ------------------------------------------------------------------------------------------------
    def _spectrum(self, a, Q, delta=None):
        """2-D spectrum (azimuth frequency x the range-frequency bins of the data) of the echoes of the image a:
        Q on the range line of the scatterer, then the envelope shift (1 + jfr delta + ...), without the kernel."""
        S = np.fft.fft(np.fft.fft(a, axis=0) * Q, axis=1)[:, self.keep]
        if delta is not None and self.shift_order >= 1:
            b = a * delta
            S += self.jfr * np.fft.fft(np.fft.fft(b, axis=0) * Q, axis=1)[:, self.keep]
            if self.shift_order >= 2:
                S += (0.5 * self.jfr ** 2) * np.fft.fft(np.fft.fft(b * delta, axis=0) * Q, axis=1)[:, self.keep]
        return S

    def _image(self, V, Qc):
        """Adjoint of ``_spectrum`` without the shift: back to the image grid (the unused range frequencies are zero)."""
        W = np.zeros(self.shape, self.dtype)
        W[:, self.keep] = V
        return np.fft.ifft(np.fft.ifft(W, axis=1) * Qc, axis=0) * self.osr

    # -- the range band common to the channels -------------------------------------------------------------------
    def spectral_shift(self, i):
        """Shift [Hz] of the range spectrum of the image as channel i observes it, at every pixel [Na, osr * Nr]: the
        image range frequency k appears in the data of the channel at k + shift. It is the range derivative of the
        phase of the channel relative to the monostatic one, -(c / 2 wl) d(C0_i(r) + dC0_i(x, r)) / dr: the flat-earth
        part plus the slope of the terrain along range."""
        return (-(C0_LIGHT / self.p.wl) * np.gradient(self.delta[i], self.r_img, axis=1)).astype(np.float32)

    def common_band(self, min_channels=None):
        """(lo, hi) [Hz], each [Na, osr * Nr]: the image range frequencies between lo and hi of a pixel are observed
        there by at least ``min_channels`` channels (default: by all of them). Where hi <= lo nothing is common."""
        p = self.p
        drop = 0 if min_channels is None else p.Nrx - int(min_channels)
        if not 0 <= drop < p.Nrx:
            raise ValueError("min_channels must be between 1 and Nrx")
        sh = np.stack([self.spectral_shift(i) for i in range(p.Nrx)])
        up = np.sort(np.maximum(sh, 0.0), axis=0)[p.Nrx - 1 - drop]               # what the top of the band loses
        dn = np.sort(np.maximum(-sh, 0.0), axis=0)[p.Nrx - 1 - drop]              # what the bottom of the band loses
        return -0.5 * p.rbw + dn, 0.5 * p.rbw - up

    def band_limit(self, g, lo, hi):
        """Keeps at every pixel of the image g the range frequencies between lo and hi of that pixel (a range filter
        whose band changes from pixel to pixel). The range frequencies kept everywhere go through one inverse FFT; the
        ones kept only at some pixels are added one by one, each at the pixels that keep it (weight going from 0 to 1
        over one frequency bin at the edges of the band)."""
        N = self.Nr2
        df = self.osr * self.p.rsf / N
        f = np.fft.fftfreq(N, 1.0 / (self.osr * self.p.rsf))
        G = np.fft.fft(g, axis=1)
        everywhere = (f <= float(hi.min()) - df) & (f >= float(lo.max()) + df)
        nowhere = (f >= float(hi.max())) | (f <= float(lo.min()))
        out = np.fft.ifft(np.where(everywhere[None, :], G, 0), axis=1)
        m = np.arange(N)
        for k in np.flatnonzero(~everywhere & ~nowhere):
            w = np.clip((hi - f[k]) / df, 0.0, 1.0) * np.clip((f[k] - lo) / df, 0.0, 1.0)
            out += (G[:, k][:, None] * (np.exp(2j * np.pi * k * m / N) / N).astype(self.dtype)[None, :]) * w.astype(np.float32)
        return out.astype(self.dtype)

    def focus(self, u):
        """Image of range-compressed monostatic data at the full PRF, u [Na, Nr]: the inverse of ``monostatic`` inside
        the band of the system (the spectrum divided by the response of the reference point; zero outside)."""
        a = np.abs(self.K0)
        ok = a > 0.05 * a.max()
        W = np.zeros(self.shape, self.dtype)
        W[:, self.keep] = np.where(ok, np.fft.fft2(u) / np.where(ok, self.K0, 1), 0)
        return np.fft.ifft(np.fft.ifft(W, axis=1) * np.conj(self.E), axis=0)

    def to_common_band(self, u, lo, hi):
        """Range-compressed monostatic data u [Na, Nr] with every pixel of its image limited to the band (lo, hi): what
        the reference looks like in the range band that the reconstruction delivers."""
        return self.monostatic(self.band_limit(self.focus(u), lo, hi))

    def data(self, s_channel_rc):
        """Channels [Nrx, Na_ch, Nr] (azimuth time x range) -> the form the model works with (unitary 2-D spectra)."""
        return (np.fft.fft2(np.asarray(s_channel_rc), axes=(1, 2)) / self.cn).astype(self.dtype)

    def channels(self, d):
        """Back from the 2-D spectra to the channels in azimuth time x range."""
        return np.fft.ifft2(d, axes=(1, 2)) * self.cn

    def forward(self, g):
        p = self.p
        out = np.empty((p.Nrx, p.Na_ch, p.Nr), self.dtype)
        for i in range(p.Nrx):
            S = self._spectrum(g * self.screen[i], self.Q[i], self.delta[i]) * self.K[i]
            out[i] = S.reshape(p.Nrx, p.Na_ch, p.Nr).sum(axis=0) / (p.Nrx * self.cn)   # sampling at PRF_op: the folds add up
        return out

    def adjoint(self, d):
        p = self.p
        g = np.zeros(self.shape, self.dtype)
        for i in range(p.Nrx):
            V = np.tile(d[i], (p.Nrx, 1)) * np.conj(self.K[i]) * self.cn
            Qc = np.conj(self.Q[i])
            a = self._image(V, Qc)
            if self.shift_order >= 1:
                a += self.delta[i] * self._image(V * np.conj(self.jfr), Qc)
                if self.shift_order >= 2:
                    a += self.delta[i] ** 2 * self._image(V * np.conj(0.5 * self.jfr ** 2), Qc)
            g += np.conj(self.screen[i]) * a
        return g

    def monostatic(self, g):
        """Range-compressed data of the monostatic system at the full PRF for the image g: the output of the reconstruction."""
        return np.fft.ifft2(self._spectrum(g, self.E) * self.K0)

    def largest_singular_value(self, iterations=12, seed=0):
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal(self.shape) + 1j * rng.standard_normal(self.shape)).astype(self.dtype)
        n = 1.0
        for _ in range(iterations):
            x = self.adjoint(self.forward(x))
            n = float(np.sqrt(energy(x)))
            x /= n
        return float(np.sqrt(n))


def energy(a):
    """Sum of |a|^2 accumulated in double precision."""
    return float(np.sum(a.real.astype(np.float64) ** 2) + np.sum(a.imag.astype(np.float64) ** 2))


def processed_band(p, u):
    """Range-compressed data u [Na, Nr] at the full PRF with the azimuth spectrum cut to the processed bandwidth abw."""
    U = np.fft.fft(u, axis=0)
    U[np.abs(np.fft.fftfreq(p.Na, 1.0 / p.prf)) > p.abw / 2] = 0
    return np.fft.ifft(U, axis=0).astype(np.complex64)


def image_to_data(M, g, band=None, zeroOutBw=True):
    """From the image of the joint to its output: the range-compressed data of the monostatic system at the full PRF,
    [Na, Nr]. band = (lo, hi) of ``Model2D.common_band``: every pixel limited to its band first. zeroOutBw: azimuth
    spectrum cut to the processed bandwidth."""
    u = M.monostatic(g if band is None else M.band_limit(g, *band))
    return processed_band(M.p, u) if zeroOutBw else u.astype(np.complex64)


def reconstruct_joint_2d(p, tracks, s_channel_rc, dC0, reg=0.03, passes=2, iterations=60, tol=1e-4, zeroOutBw=True,
                         shift_order=1, return_info=False, model=None, state=None, callback=None, common_band=True,
                         min_channels=None):
    """
    s_channel_rc [Nrx, Na_ch, Nr] range-compressed channels at PRF_op; dC0 [Nrx, Na, osr * Nr]: delta_C0 of every
    channel at every image pixel (``dem_maps``: from the DEM).
    reg, passes, iterations, tol: as in ``sata1d.joint_line.reconstruct_joint_line`` (regularisation relative to the
    largest singular value of the model; every pass solves for what the previous ones left unexplained, iterated
    Tikhonov; conjugate gradients, at most ``iterations`` per pass).
    state      ``info["state"]`` of a previous call: the passes of this call continue from there (the passes can be
               run in separate calls, e.g. passes=1 twice).
    common_band  True: the output keeps, at every pixel, only the range frequencies that at least ``min_channels``
               channels observe there (default: all the channels; module docstring). False: the whole range band.
    Returns the reconstructed range-compressed data [Na, Nr] at the full PRF (azimuth time x range, as
    ``reconstruct_subband_2d``); with return_info=True also a dict with the number of iterations, the part of the
    channels that the model does not explain (residual / data, rms), the focused image (whole band), the limits of
    the band kept at every pixel and the state.
    callback(n_it, image, residual): called after every iteration (for the convergence plots).
    """
    M = Model2D(p, tracks, dC0, shift_order=shift_order) if model is None else model
    s = M.data(s_channel_rc)
    if state is None:
        sigma = M.largest_singular_value()
        g = np.zeros(M.shape, M.dtype)
        r = s.copy()                                           # part of the channels not explained yet
    else:
        sigma, g, r = float(state["sigma"]), np.array(state["image"], M.dtype), np.array(state["residual"], M.dtype)
    damp2 = (reg * sigma) ** 2
    n_it = 0
    for _ in range(passes):
        d = np.zeros(M.shape, M.dtype)                         # correction of this pass: min |A d - r|^2 + damp2 |d|^2
        rp = r.copy()
        z = M.adjoint(rp)                                      # gradient
        q_dir = z.copy()
        zz = z0 = energy(z)
        for _ in range(iterations):
            q = M.forward(q_dir)
            alpha = zz / (energy(q) + damp2 * energy(q_dir))
            d += np.float32(alpha) * q_dir
            rp -= np.float32(alpha) * q
            n_it += 1
            if callback is not None:
                callback(n_it, g + d, rp)
            z = M.adjoint(rp) - np.float32(damp2) * d
            zz_new = energy(z)
            if zz_new <= tol ** 2 * z0:
                break
            q_dir = z + np.float32(zz_new / zz) * q_dir
            zz = zz_new
        g += d
        r = rp
    band = M.common_band(min_channels) if common_band else None
    out = image_to_data(M, g, band, zeroOutBw)
    if return_info:
        return out, dict(iterations=n_it, unexplained=float(np.sqrt(energy(r) / energy(s))), image=g, band=band,
                         state=dict(sigma=sigma, image=g, residual=r))
    return out
