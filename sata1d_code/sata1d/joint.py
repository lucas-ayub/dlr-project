"""
Reconstruction inside the SATA sub-apertures ("joint" SATA + reconstruction).

The per-channel SATA multiplies every bin of a channel's sub-aperture spectrum by ONE phase,
although the bin carries the energy of Nsb folded parcels, one per fold q, that come from
Nsb ground positions X apart. When the terrain differs at those positions no single phase
is right, and the parcels that should cancel in the reconstruction leak as ambiguities.

Here the fold bookkeeping is moved into the reconstruction: for every sub-aperture and every
bin phi of the channel spectra, the Nrx x Nsb system

    s_i(phi) = sum_q  H_i(f_q) exp(-j 2 pi dC0_i(x_q) / lambda) u_q ,     f_q = f_q0 + phi,

is solved for the parcels u_q, with x_q = c + a(f_q) the ground position of parcel q (the
same f -> beta -> x ruler as the SATA kernel) and dC0_i(x_q) read from the look-up table of
channel i at THAT position. The reference-point filters H_i are those of the standard
reconstruction, evaluated at the true frequency of every parcel of every bin, so that the
sub-aperture processing is the standard reconstruction done block-wise (zero-padded FFTs,
overlap-add of the analysis windows, which sum to one). With dC0 = 0 the method reduces to the standard reconstruction; with the true dC0 of every
parcel it reproduces the oracle. No fold copies are needed in the table.

occupied=True (default) solves, in every bin, only for the parcels that carry signal,
|f_q| <= B/2 + 4 PRF_op / Tsub (target Doppler bandwidth plus the width of the block window),
by least squares. Since B < (Nsb - 1) PRF_op there are fewer unknowns than channels, and the
system stays well conditioned when a terrain slope along azimuth (with a cross-track baseline)
moves the effective phase centres of two channels onto the same position, where the full
Nrx x Nsb system becomes singular. Cost: with dC0 = 0 the result is no longer bit-identical
to the standard reconstruction (the window tails beyond the bound are dropped; ~1e-4 rms), and
the pseudo-inverse is slower than solve. occupied=False keeps the full square system.
"""
from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .geometry import PlatformTracks
from .reconstruction import GetCoeffNu
from .sata import subaperture_sizing, triangular_window, delta_C0_per_target, lookup_table


def reference_coefficients(cfg: ExperimentConfig, tracks: PlatformTracks):
    """(C0, C1, C2, Dt) of the reference point for every channel, as filter_matrix uses them."""
    out = np.zeros((cfg.Nrx, 4))
    for i in range(cfg.Nrx):
        out[i] = GetCoeffNu(cfg.scene.ptg, tracks.ptx, tracks.prx[i], tracks.vtx, tracks.vrx[i],
                            tracks.ptx, tracks.vtx, cfg.prf, cfg.system.wl, cfg.ta,
                            cfg.sq_tx, cfg.sq_rx[i], cfg.theta_tx, cfg.theta_rx[i])
    return out


def joint_maps(cfg: ExperimentConfig, tracks: PlatformTracks, dC0=None, max_gap=None):
    """Look-up tables for the joint reconstruction, one per channel: the targets' values at
    the targets' cells and, at every other cell, the value of the nearest target (max_gap
    limits how far that extension goes; by default it covers the whole line, so that the
    window tails of the targets' energy, which reach beyond h_w, are also read with the
    terrain of the nearest target instead of a zero)."""
    dC0 = delta_C0_per_target(cfg, tracks) if dC0 is None else dC0
    gap = cfg.Na_ch if max_gap is None else max_gap
    return [lookup_table(cfg, tracks, i, dC0=dC0, max_gap=gap) for i in range(cfg.Nrx)]


def reconstruct_joint(cfg: ExperimentConfig, tracks: PlatformTracks, s_channel, maps=None, dC0=None,
                      sata_osf=4, zeroOutBw=True, occupied=True):
    """
    s_channel [Nrx, Na_ch] channel lines at PRF_op; maps: list of Nrx look-up tables [Na_ch]
    (joint_maps if not given). occupied: solve only for the parcels that carry signal (least
    squares) instead of the full Nrx x Nsb system. Returns the reconstructed line [Na] at the
    full PRF.
    """
    Nrx, Nsb, Na_ch, P = cfg.Nrx, cfg.Nsb, cfg.Na_ch, cfg.PRF_op
    wl, v, r0 = cfg.system.wl, cfg.system.vs, cfg.scene.r0
    Na = Na_ch * Nsb
    s = np.asarray(s_channel, dtype=complex)
    if maps is None:
        maps = joint_maps(cfg, tracks, dC0)
    maps = np.asarray(maps, float)                                    # [Nrx, Na_ch]
    coef = reference_coefficients(cfg, tracks)                        # [Nrx, 4]: C0, C1, C2, Dt

    # 1) sub-aperture grid; parcel frequencies and ground positions of every bin
    Tsub, hop, Nzp = subaperture_sizing(r0, P, v, wl, sata_osf)
    win = triangular_window(Tsub)
    phi = np.arange(Nzp) * P / Nzp                                    # bin b <-> b P / Nzp, as filter_matrix
    fq0 = -Nsb * P / 2.0 + np.arange(Nsb) * P                         # start of parcel q
    f = fq0[:, None] + phi[None, :]                                   # [Nsb, Nzp] true Doppler of parcel q in bin b
    a = r0 * np.tan(np.arcsin(np.clip(wl * f / (2.0 * v), -1.0, 1.0))) / v * P   # cell offset of that position
    # 2) the reference-point filters of the standard reconstruction, on the bin grid
    Hq = np.zeros((Nzp, Nrx, Nsb), complex)                           # [bin, channel (row), parcel (column)]
    for i in range(Nrx):
        C0, C1, C2, Dt = coef[i]
        Hq[:, i, :] = np.exp(-2j * np.pi * (C0 / wl + (-C1 + Dt) * f.T + C2 * wl * f.T ** 2))
    # parcels that carry signal: target Doppler bandwidth plus the width of the block window
    if occupied:
        B = 4.0 * v * np.sin(cfg.theta_tx / 2.0) / wl                # Doppler bandwidth of one target
        occ = np.abs(f.T) <= B / 2.0 + 4.0 * P / Tsub                 # [Nzp, Nsb]

    # 3) sub-aperture loop: solve the Nsb parcels of every bin, overlap-add at the full rate.
    #    The windowed segment sits in the middle of the zero-padded block, so the filtered block
    #    (which spreads a little beyond the segment) is added whole, at its own time.
    lo = (Nzp - Tsub) // 2                                            # margin on each side [channel samples]
    L, Lb = Tsub * Nsb, Nzp * Nsb                                     # segment / block length at the full PRF
    wfull = np.interp(np.arange(L) / Nsb, np.arange(Tsub), win)
    pad = lo * Nsb
    out = np.zeros(Na + Lb + 2 * pad, complex)
    wsum = np.zeros(Na + Lb + 2 * pad)
    spad = np.concatenate([s, np.zeros((Nrx, Tsub), complex)], axis=1)
    roll = (Nzp * Nsb) // 2
    buf = np.zeros((Nrx, Nzp), complex)
    for start in range(0, Na_ch, hop):
        c = start + 0.5 * Tsub
        buf[:] = 0.0
        buf[:, lo:lo + Tsub] = spad[:, start:start + Tsub] * win
        S = np.fft.fft(buf, axis=1)                                             # [Nrx, Nzp]
        pos = np.clip(np.round(c + a).astype(int), 0, Na_ch - 1)               # [Nsb, Nzp]
        theta = maps[:, pos]                                                    # [Nrx, Nsb, Nzp]
        G = Hq * np.exp(-2j * np.pi * theta.transpose(2, 0, 1) / wl)            # [Nzp, Nrx, Nsb]: s_i = sum_q G_iq u_q
        if occupied:                                                            # empty parcels: zero column, left out
            u = Nsb * (np.linalg.pinv(G * occ[:, None, :]) @ S.T[:, :, None])[:, :, 0]
        else:
            u = Nsb * np.linalg.solve(G, S.T[:, :, None])[:, :, 0]              # [Nzp, Nsb]
        U = np.roll(u.T.reshape(-1), roll)                                      # f = fq0[q] + phi[b] -> fft order
        y = np.fft.ifft(U)                                                      # Lb samples at the full PRF
        i0 = (start - lo) * Nsb + pad                                           # block start in the padded output
        out[i0:i0 + Lb] += y
        wsum[pad + start * Nsb:pad + start * Nsb + L] += wfull
    out = out[pad:pad + Na]; wsum = wsum[pad:pad + Na]
    nz = wsum > 1e-12
    out[nz] /= wsum[nz]
    if zeroOutBw:
        fa = np.fft.fftfreq(Na, d=1.0 / cfg.prf)
        Uo = np.fft.fft(out); Uo[np.abs(fa) > cfg.abw / 2] = 0
        out = np.fft.ifft(Uo)
    return out