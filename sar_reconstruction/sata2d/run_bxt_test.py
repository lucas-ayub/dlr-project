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
import argparse
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--bxt-mode", default="random", dest="bxt_mode",
                 choices=("linear", "random"))
_ap.add_argument("--seed", type=int, default=0)
_a = _ap.parse_args()
# In random mode the sweep is over the upper bound of the uniform draw; in
# linear mode it is over the ladder step.  What matters for the reconstruction
# is the SPREAD between channels -- a shift common to all of them is a global
# image shift, not a misalignment -- so that is what is reported.
_SWEEP = (300.0, 100.0, 20.0) if _a.bxt_mode == "random" else (150.0, 30.0, 5.0)
print(f"array mode: {_a.bxt_mode}\n")
print(f"{'dxt/bxt_max':>12} {'bxt':>34} {'spread [cells]':>15} {'oracle peak %':>14} {'err/sig [dB]':>13}")
for dxt in _SWEEP:
    from .arrays import make_params_dpca
    p = make_params_dpca(Nrx=4, dxt=dxt, specs=((0.0, 240.0),),
                         bxt_mode=_a.bxt_mode,
                         bxt_max=dxt if _a.bxt_mode == "random" else None,
                         seed=_a.seed)
    tr = build_tracks_3d(p); ptg = np.asarray(p.points[0], float)
    nb = range_bin_of(p, scatterer_range(p, ptg))
    th = np.arcsin(p.y0 / p.r0)
    dR = -p.bxt * np.sin(th) / 2.0                 # half-path shift per channel [m]
    mis = (dR.max() - dR.min()) / (C / 2 / p.rsf)  # spread between channels [cells]
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
    print(f"{dxt:12.0f} {np.array2string(p.bxt, precision=1):>34} "
          f"{mis:15.2f} {100*v/pm:13.1f}% {esr:12.2f}")
