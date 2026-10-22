from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .geometry import PlatformTracks
from .coefficients import reference_coefficients, residual_C0
from .signal_model import generate_reference


def line_positions(cfg: ExperimentConfig):
    """Azimuth position [m], relative to the reference point, of every sample of the focused full-rate line
    (sample Na // 2 is the reference point; the same axis as the IRF plots, x = vs * t)."""
    return (np.arange(cfg.Na) - cfg.Na // 2) * cfg.system.vs / cfg.prf


def height_table(cfg: ExperimentConfig, tracks: PlatformTracks, h_min, h_max, n=33):
    """delta_C0 of every channel on a grid of heights (residual_C0 of a point on the iso-range surface at that height).
    The residual depends on the height and on the channel, not on the azimuth position. Returns (heights [n],
    table [n, Nrx])."""
    hg = np.linspace(min(h_min, 0.0) - 1.0, max(h_max, 0.0) + 1.0, n)
    H, r0, y0 = cfg.scene.H, cfg.scene.r0, cfg.scene.y0
    tab = np.zeros((n, cfg.Nrx))
    for k, h in enumerate(hg):
        p = cfg.scene.ptg + np.array([0.0, np.sqrt(r0 ** 2 - (H - h) ** 2) - y0, h])
        tab[k] = [residual_C0(cfg, tracks, p, i) for i in range(cfg.Nrx)]
    return hg, tab


def maps_exact(cfg: ExperimentConfig, tracks: PlatformTracks, heights):
    """delta_C0 [Nrx, len(heights)] computed point by point: residual_C0 of the point at each height (no polynomial).
    Positions with the same height share one computation (same result). About 5 ms per height and channel."""
    heights = np.asarray(heights, float)
    H, r0, y0 = cfg.scene.H, cfg.scene.r0, cfg.scene.y0
    hu, inv = np.unique(heights, return_inverse=True)
    tab = np.zeros((len(hu), cfg.Nrx))
    for k, h in enumerate(hu):
        p = cfg.scene.ptg + np.array([0.0, np.sqrt(r0 ** 2 - (H - h) ** 2) - y0, h])
        tab[k] = [residual_C0(cfg, tracks, p, i) for i in range(cfg.Nrx)]
    return tab[inv.ravel()].T


def maps_from_heights(cfg: ExperimentConfig, tracks: PlatformTracks, heights, table=None, polynomial=True):
    """delta_C0 [Nrx, len(heights)] of a terrain given by its height at every position.
    polynomial=True  (default): cubic fit of height_table (33 heights), evaluated at every height;
    polynomial=False: residual_C0 computed for every height (maps_exact), slower."""
    heights = np.asarray(heights, float)
    if not polynomial:
        return maps_exact(cfg, tracks, heights)
    hg, tab = height_table(cfg, tracks, heights.min(), heights.max()) if table is None else table
    return np.array([np.polyval(np.polyfit(hg, tab[:, i], 3), heights) for i in range(cfg.Nrx)])


def line_maps(cfg: ExperimentConfig, tracks: PlatformTracks, terrain, table=None, polynomial=True):
    """delta_C0 [Nrx, Na] of every channel at every position of the full-rate line, from the terrain h(x) only.
    polynomial=True (default): through the height table and the cubic fit; False: point by point (maps_exact)."""
    return maps_from_heights(cfg, tracks, terrain(line_positions(cfg)), table, polynomial)


class LineModel:
    """The model channels = A g and its adjoint (see the module docstring). g is in FFT order: lag 0 = reference point."""

    def __init__(self, cfg: ExperimentConfig, tracks: PlatformTracks, dC0_line):
        self.cfg = cfg
        Na, Nrx, wl = cfg.Na, cfg.Nrx, cfg.system.wl
        f = np.fft.fftfreq(Na, 1.0 / cfg.prf)
        self.chirp = np.fft.fft(generate_reference(cfg, tracks, cfg.scene.ptg[None, :]))       # reference point, full rate
        coef = reference_coefficients(cfg, tracks)                                              # C0, C1, C2, Dt per channel
        self.K = np.array([self.chirp * np.exp(-2j * np.pi * (c[0] / wl + (-c[1] + c[3]) * f + c[2] * wl * f ** 2)) for c in coef])
        dC0 = np.fft.ifftshift(np.asarray(dC0_line, float), axes=1)                             # centred order -> lag order
        if dC0.shape != (Nrx, Na):
            raise ValueError(f"dC0_line must have shape (Nrx, Na) = ({Nrx}, {Na})")
        self.screen = np.exp(-2j * np.pi * dC0 / wl)

    def forward(self, g):
        Nrx = self.cfg.Nrx
        return np.array([np.fft.ifft(np.fft.fft(g * self.screen[i]) * self.K[i])[::Nrx] for i in range(Nrx)])

    def adjoint(self, s):
        Na, Nrx = self.cfg.Na, self.cfg.Nrx
        g = np.zeros(Na, complex)
        v = np.zeros(Na, complex)
        for i in range(Nrx):
            v[:] = 0.0
            v[::Nrx] = s[i]
            g += np.conj(self.screen[i]) * np.fft.ifft(np.fft.fft(v) * np.conj(self.K[i]))
        return g

    def largest_singular_value(self, iterations=12, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(self.cfg.Na) + 1j * rng.standard_normal(self.cfg.Na)
        n = 1.0
        for _ in range(iterations):
            x = self.adjoint(self.forward(x))
            n = np.linalg.norm(x)
            x /= n
        return float(np.sqrt(n))


def reconstruct_joint_line(cfg: ExperimentConfig, tracks: PlatformTracks, s_channel, dC0_line, reg=0.03, passes=2,
                           iterations=60, tol=1e-4, zeroOutBw=True, return_info=False):
    """
    s_channel [Nrx, Na_ch] channel lines at PRF_op; dC0_line [Nrx, Na]: delta_C0 of every channel at every position of
    the full-rate line (line_maps: from the DEM).
    reg        regularisation, relative to the largest singular value of the model;
    passes     number of regularised solutions: pass 1 solves for the line, every further pass solves for what the
               previous ones left unexplained and adds it (iterated Tikhonov). Two passes remove most of the loss that
               the regularisation causes in the directions that the channels measure well, and the directions that
               they do not measure stay damped;
    iterations maximum number of CGLS iterations per pass (a pass stops earlier when its gradient has dropped by tol).
    Returns the reconstructed line [Na] at the full PRF (raw, as reconstruct()); with return_info=True also a dict with
    the number of iterations and the part of the channels that the model does not explain (residual / data, rms).
    """
    s = np.asarray(s_channel, dtype=complex)
    M = LineModel(cfg, tracks, dC0_line)
    damp2 = (reg * M.largest_singular_value()) ** 2
    g = np.zeros(cfg.Na, complex)
    r = s.copy()                                               # part of the channels not explained yet
    n_it = 0
    for _ in range(passes):
        d = np.zeros(cfg.Na, complex)                          # correction of this pass: min |A d - r|^2 + damp2 |d|^2
        rp = r.copy()
        z = M.adjoint(rp)                                      # gradient
        p = z.copy()
        zz = z0 = np.vdot(z, z).real
        for _ in range(iterations):
            q = M.forward(p)
            alpha = zz / (np.vdot(q, q).real + damp2 * np.vdot(p, p).real)
            d += alpha * p
            rp -= alpha * q
            n_it += 1
            z = M.adjoint(rp) - damp2 * d
            zz_new = np.vdot(z, z).real
            if zz_new <= tol ** 2 * z0:
                break
            p = z + (zz_new / zz) * p
            zz = zz_new
        g += d
        r = rp
    U = np.fft.fft(g) * M.chirp                                # back to the raw line of the scene (monostatic, full rate)
    if zeroOutBw:
        U[np.abs(np.fft.fftfreq(cfg.Na, 1.0 / cfg.prf)) > cfg.abw / 2] = 0
    out = np.fft.ifft(U)
    if return_info:
        return out, dict(iterations=n_it, unexplained=float(np.linalg.norm(r) / np.linalg.norm(s)))
    return out
