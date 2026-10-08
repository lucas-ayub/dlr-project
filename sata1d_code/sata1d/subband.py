"""
Per-sub-band SATA and reconstruction (C0 term).

Output sub-band k (k = 0..Nsb-1, Nsb = Nrx) is centred at f_k = (-Nsb/2 + k + 1/2) PRF_op.
For each k, every channel is SATA-corrected with
  - the same delta_C0 values as the whole-band SATA (every channel observes every target over
    the whole integration time; the values are computed once, delta_C0_per_target),
  - the same look-up table (mode="lut") for every k,
  - a kernel whose bins carry the absolute Doppler in [f_k - PRF_op/2, f_k + PRF_op/2)
    (the channel's baseband frequency wrapped into that window), with positions on the
    broadside image grid, so the energy of sub-band k is looked up at its true position,
and sub-band k of the reconstruction uses only that corrected set.
"""
from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .geometry import PlatformTracks
from .reconstruction import channel_spectrum, reconstruct_from_spectra
from .sata import build_delta_C0_array, delta_C0_per_target, subaperture_sizing, sata_kernel


def subband_frequency_beam(cfg: ExperimentConfig, k):
    """(f_k, beta_k, beta_lo, beta_hi) of output sub-band k (the angles are informative only)."""
    vs, wl = cfg.system.vs, cfg.system.wl
    f_k = (-cfg.Nsb / 2.0 + k + 0.5) * cfg.PRF_op
    a = lambda f: float(np.arcsin(np.clip(wl * f / (2.0 * vs), -1.0, 1.0)))
    return f_k, a(f_k), a(f_k - cfg.PRF_op / 2.0), a(f_k + cfg.PRF_op / 2.0)


def build_delta_C0_subband_array(cfg, tracks, channel, k, mode="footprint", dC0=None):
    """delta_C0 map of one channel for the kernel of output sub-band k. The values are the
    full-history delta_C0 of the targets (delta_C0_per_target, shared by all k).
    mode="lut": the look-up table, identical for every k;
    mode="footprint": block map with the fold images of a kernel centred on f_k."""
    if mode == "lut":
        return build_delta_C0_array(cfg, tracks, channel, mode="lut", dC0=dC0)
    return build_delta_C0_array(cfg, tracks, channel, mode=mode, f_centre=subband_frequency_beam(cfg, k)[0], dC0=dC0)


def subband_axis(prf, Nzp, v, wl, r, f_k):
    """Absolute Doppler of each FFT bin for sub-band k (baseband frequency wrapped into
    [f_k - prf/2, f_k + prf/2)) and its cell offset on the broadside grid."""
    phi = np.fft.fftfreq(Nzp, d=1.0 / prf)
    f_lo = f_k - prf / 2.0
    fsub = f_lo + np.mod(phi - f_lo, prf)
    betasub = np.arcsin(np.clip(wl * fsub / (2.0 * v), -1.0, 1.0))
    return fsub, r * np.tan(betasub) / v * prf


def sata_1d_subband(data, delta_C0_array, rref, prf, v, wl, r, f_k, inverse=False, sata_osf=1,
                    scheme="cola"):
    """sata_1d for sub-band k: bins labelled with their absolute Doppler in the
    sub-band window (subband_axis), positions on the broadside grid."""
    data = np.asarray(data, dtype=complex)
    Tsubeff, hop, Nzp = subaperture_sizing(rref, prf, v, wl, sata_osf)
    if Tsubeff <= 2:
        return data.copy()
    fsub, azpos = subband_axis(prf, Nzp, v, wl, r, f_k)
    return sata_kernel(data, delta_C0_array, Tsubeff, hop, Nzp, azpos, wl, inverse, scheme)


def sata_channels_subband(cfg, tracks, s_channel, k, remove=True, sata_osf=4, mode="footprint",
                          scheme="cola", maps=None, dC0=None):
    """Every channel SATA-corrected for output sub-band k.
    maps: ready-made delta_C0 maps, one per channel (the look-up table shared by all k);
    dC0:  delta_C0_per_target, shared by all k (computed here if neither is given)."""
    f_k = subband_frequency_beam(cfg, k)[0]
    out = np.asarray(s_channel, dtype=complex).copy()
    if maps is None and dC0 is None:
        dC0 = delta_C0_per_target(cfg, tracks)
    for kk in range(cfg.Nrx):
        M = maps[kk] if maps is not None else build_delta_C0_subband_array(cfg, tracks, kk, k, mode=mode, dC0=dC0)
        out[kk] = sata_1d_subband(out[kk], M, rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs,
                                  wl=cfg.system.wl, r=cfg.scene.r0, f_k=f_k,
                                  inverse=remove, sata_osf=sata_osf, scheme=scheme)
    return out


def reconstruct_subband(cfg: ExperimentConfig, tracks: PlatformTracks, s_channel, use_sata=True,
                        remove=True, sata_osf=4, mode="footprint", zeroOutBw=True, scheme="cola", dC0=None):
    """Reconstruction where output sub-band k (k = 0..Nsb-1) uses the channels corrected by
    the kernel of sub-band k. The targets' delta_C0 are computed once (delta_C0_per_target) and
    shared by every sub-band; with mode="lut" the table itself is built once per channel.
    use_sata=False gives the standard reconstruction."""
    base = np.asarray(s_channel, dtype=complex)
    maps = None
    if use_sata:
        dC0 = delta_C0_per_target(cfg, tracks) if dC0 is None else dC0
        if mode == "lut":
            maps = [build_delta_C0_array(cfg, tracks, j, mode="lut", dC0=dC0) for j in range(cfg.Nrx)]
    spec = []
    for k in range(cfg.Nsb):
        corr = sata_channels_subband(cfg, tracks, base, k, remove, sata_osf, mode, scheme, maps, dC0) if use_sata else base
        spec.append([channel_spectrum(cfg, corr[j]) for j in range(cfg.Nrx)])
    return reconstruct_from_spectra(cfg, tracks, spec, zeroOutBw)
