# -*- coding: utf-8 -*-
"""
Post-reconstruction analysis: matched filtering and peak zoom for IRF inspection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig


def zoom1Dpeak(s, N, zpf, Ndc=0):
    """Zero-padded zoom around the peak of |s| for fine IRF inspection."""
    Na = len(s)
    ii = np.argmax(abs(s))
    sc = (np.roll(s, int(0.5 * Na - ii)))[int(0.5 * Na - N):int(0.5 * Na + N)]
    sc = np.roll(np.fft.fft(sc), int(-Ndc))
    s2 = np.zeros(N * 2 * zpf, dtype=np.complex128)
    s2[0:N] = sc[0:N]
    s2[N * 2 * zpf - N:] = sc[N:]
    s2 = np.fft.ifft(np.roll(s2, int(Ndc)))
    return s2


def matched_filter(s: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Correlate s against ref in the frequency domain, fftshift to centre."""
    Na = len(ref)
    return np.roll(
        np.fft.ifft(np.fft.fft(s) * np.conjugate(np.fft.fft(ref))),
        int(Na / 2),
    )


@dataclass
class ReconResult:
    sref: np.ndarray
    srecN: np.ndarray
    srefF: np.ndarray            # reference autocorrelation (focused)
    srecNF: np.ndarray           # reconstructed matched against reference
    u_refFocC: np.ndarray        # zoomed reference IRF
    u_interpFocCN: np.ndarray    # zoomed reconstructed IRF
    taz: np.ndarray              # zoom time axis [s]
    fa: np.ndarray               # Doppler frequency axis [Hz]
    abw_idx: np.ndarray          # indices outside the processed bandwidth


def analyze(cfg: ExperimentConfig, sref: np.ndarray, srecN: np.ndarray,
            N_factor: int = 16, zpf: int = 64) -> ReconResult:
    Na = cfg.Na
    prf = cfg.prf
    abw = cfg.abw

    srefF = matched_filter(sref, sref)
    srecNF = matched_filter(srecN, sref)

    N = int(N_factor * prf / abw)
    taz = (np.arange(2 * N * zpf) - N * zpf) / prf / zpf
    u_refFocC = zoom1Dpeak(srefF, N, zpf)
    u_interpFocCN = zoom1Dpeak(srecNF, N, zpf)

    fa = np.roll((np.arange(Na) / Na - 0.5) * prf, int(Na / 2))
    abw_idx = np.concatenate((np.where(fa < -0.5 * abw)[0],
                              np.where(fa > 0.5 * abw)[0]))

    return ReconResult(
        sref=sref, srecN=srecN, srefF=srefF, srecNF=srecNF,
        u_refFocC=u_refFocC, u_interpFocCN=u_interpFocCN,
        taz=taz, fa=fa, abw_idx=abw_idx,
    )
