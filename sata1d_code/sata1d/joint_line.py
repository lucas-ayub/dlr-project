"""
Joint reconstruction over the whole line (topography-dependent reconstruction with a DEM, no sub-apertures).

The joint reconstruction of joint.py puts the delta_C0 of the terrain inside the reconstruction, but reads it once
per sub-aperture bin: the phase of the terrain is taken as constant over the ground that one bin of a 40 ms block
covers (about +-600 m). On steep terrain with a long cross-track baseline that is not true any more (the phase of
channel i changes by 2 pi / wl * d(dC0_i)/dx over that ground), and the result degrades.

Here the same equations are written for the whole line, with the phase of the terrain applied at every azimuth
position. The topographic residual of a scatterer is constant over its whole aperture (every channel sees it during the
whole integration time with the same delta_C0), so in the focused line it is a plain multiplication:

    channel_i = decimate_Nrx{ k_i  (*)  [ g(x) exp(-j 2 pi dC0_i(x) / wl) ] }                         (*) = convolution

    g(x)      focused line of the scene at the full rate (what the reconstruction should deliver, focused)
    dC0_i(x)  delta_C0 of channel i for a point on the terrain at azimuth x: from the DEM only (line_maps)
    k_i       response of channel i to the reference point: the chirp of the reference point filtered by the
              reference-point filter H_i of the standard reconstruction (C0, C1, C2, Dt of GetCoeffNu)

With dC0 = 0 this is the model that the standard reconstruction inverts. With a terrain, the Nrx channels are Na
equations for the Na samples of g, coupled along the line, and they are solved by least squares with conjugate
gradients (CGLS): every iteration applies the model once and its adjoint once (2 FFTs per channel each).
The adjoint is "focus every channel, remove the phase of the terrain, add the channels": the first iterate is the
sum of the channel images with the DEM phase removed, and the following ones cancel the ambiguities that are left.

Regularisation: reg * (largest singular value of the model), applied in two passes (the second pass solves for what
the first left unexplained). Directions of g that the channels do not measure (two effective phase centres on the same
position, see the notes on the common step of the baselines) are not amplified; there the ambiguities stay. No information about the targets is used: only the system geometry and the DEM.

    from sata1d.joint_line import line_maps, reconstruct_joint_line
    dC0 = line_maps(cfg, tracks, terrain)                    # terrain: h(x) [m], x = azimuth from the reference point
    s = reconstruct_joint_line(cfg, tracks, channels, dC0)   # line at the full PRF, as reconstruct()
"""
from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .geometry import PlatformTracks
from .joint import reference_coefficients
from .sata import residual_C0
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


def maps_from_heights(cfg: ExperimentConfig, tracks: PlatformTracks, heights, table=None):
    """delta_C0 [Nrx, len(heights)] of a terrain given by its height at every position (cubic fit of height_table)."""
    heights = np.asarray(heights, float)
    hg, tab = height_table(cfg, tracks, heights.min(), heights.max()) if table is None else table
    return np.array([np.polyval(np.polyfit(hg, tab[:, i], 3), heights) for i in range(cfg.Nrx)])


def line_maps(cfg: ExperimentConfig, tracks: PlatformTracks, terrain, table=None):
    """delta_C0 [Nrx, Na] of every channel at every position of the full-rate line, from the terrain h(x) only."""
    return maps_from_heights(cfg, tracks, terrain(line_positions(cfg)), table)


def channel_maps(cfg: ExperimentConfig, tracks: PlatformTracks, terrain, table=None):
    """Look-up tables [Nrx, Na_ch] for the sub-aperture methods (SATA whole band / per sub-band, reconstruct_joint):
    the same values as terrain_lookup_table (cell of channel i <-> x = (cell - Na_ch / 2) ds - bat_i / 2 - x0),
    computed through the height table instead of one range-history fit per cell."""
    ds = cfg.system.vs / cfg.PRF_op
    cells = np.arange(cfg.Na_ch)
    hx = [np.asarray(terrain((cells - cfg.Na_ch / 2) * ds - 0.5 * cfg.array.bat[i] - cfg.scene.x0), float) for i in range(cfg.Nrx)]
    if table is None:
        table = height_table(cfg, tracks, min(h.min() for h in hx), max(h.max() for h in hx))
    hg, tab = table
    return np.array([np.polyval(np.polyfit(hg, tab[:, i], 3), hx[i]) for i in range(cfg.Nrx)])


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
