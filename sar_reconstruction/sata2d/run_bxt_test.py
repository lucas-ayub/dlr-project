from __future__ import annotations
import os, sys
import numpy as np
if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)
from .geometry import (make_params3d, build_tracks_3d, generate_reference_3d,
                       generate_channels_3d, CoeffTable3D)
from .reconstruction import (range_compress, build_delta_C0_map_3d, reconstruct_subband_2d,
                             scatterer_range, range_bin_of, matched_filter)
C = 3.0e8
print(f"{'dxt [m]':>8} {'bxt_max':>8} {'rg misalign [bins]':>19} {'oracle peak %':>14} {'err/sig [dB]':>13}")
for dxt in (150.0, 30.0, 5.0):
    p = make_params3d(Nrx=4, dxt=dxt, specs=((0.0, 240.0),))
    tr = build_tracks_3d(p); ptg = np.asarray(p.points[0], float)
    nb = range_bin_of(p, scatterer_range(p, ptg))
    bx = float(np.max(np.abs(p.bxt)))
    th = np.arcsin(p.y0 / p.r0)
    mis = bx * np.sin(th) / 2 / (C / 2 / p.rsf)     # half-path shift, in range bins
    ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)
    ch  = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
    tab = CoeffTable3D(p, tr, n_nodes=16, height=float(ptg[2]))   # oracle: phase is exact
    maps = [build_delta_C0_map_3d(p, tr, i) for i in range(p.Nrx)]
    rec = reconstruct_subband_2d(p, tr, ch, tab, sata_osf=4, maps=maps, use_sata=False)
    rl = ref[:, nb]
    pm = float(np.abs(matched_filter(rl, rl)).max())
    v = float(np.abs(matched_filter(rec[:, nb], rl)).max())
    S_ref = np.fft.fftshift(np.fft.fft2(ref)); S = np.fft.fftshift(np.fft.fft2(rec))
    fa = np.fft.fftshift(np.fft.fftfreq(p.Na, 1/p.prf)); inb = np.abs(fa) < p.abw/2
    e = np.abs(S-S_ref)[inb]; s = np.abs(S_ref)[inb]
    esr = 20*np.log10(np.sqrt(np.mean(e**2))/np.sqrt(np.mean(s**2)))
    print(f"{dxt:8.0f} {bx:8.0f} {mis:19.2f} {100*v/pm:13.1f}% {esr:12.2f}")
