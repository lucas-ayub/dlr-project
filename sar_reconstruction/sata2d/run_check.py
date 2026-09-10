# -*- coding: utf-8 -*-
"""Is the 'phase error' panel a valid diagnostic here?  Measure error-to-signal."""
from __future__ import annotations
import os, sys
import numpy as np
if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)
from .geometry import (make_params3d, build_tracks_3d, generate_reference_3d,
                       generate_channels_3d)
from .reconstruction import (range_compress, CoeffTable3D, build_delta_C0_map_3d,
                             reconstruct_subband_2d, scatterer_range, range_bin_of,
                             matched_filter, METHOD_KW)
C = 3.0e8
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"); os.makedirs(OUT, exist_ok=True)
p = make_params3d(Nrx=4, dxt=150.0, specs=((0.0, 240.0),))
tr = build_tracks_3d(p); nb = range_bin_of(p, scatterer_range(p, p.points[0]))
ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)
ch  = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
tab = CoeffTable3D(p, tr, n_nodes=16); maps = [build_delta_C0_map_3d(p, tr, i) for i in range(p.Nrx)]
rec = {t: reconstruct_subband_2d(p, tr, ch, tab, sata_osf=4, maps=maps, **METHOD_KW[t])
       for t in ("no", "sub")}
S_ref = np.fft.fftshift(np.fft.fft2(ref))
S = {k: np.fft.fftshift(np.fft.fft2(v)) for k, v in rec.items()}
fa = np.fft.fftshift(np.fft.fftfreq(p.Na, 1/p.prf)); fr = np.fft.fftshift(np.fft.fftfreq(p.Nr, 1/p.rsf))
inb = np.abs(fa) < p.abw/2
print("\n=== error-to-signal ratio, in-band ===")
esr = {}
for k, v in S.items():
    e = np.abs(v - S_ref)[inb]; s = np.abs(S_ref)[inb]
    esr[k] = 20*np.log10(np.sqrt(np.mean(e**2))/np.sqrt(np.mean(s**2)))
    print(f"  {k:>4}: ||S_rec - S_ref|| / ||S_ref|| = {esr[k]:6.2f} dB  "
          f"(linear {10**(esr[k]/20):.3f})")
    # how much of that is a pure phase screen (multiplicative) vs additive?
    ph = np.exp(1j*np.angle(v*np.conj(S_ref)))
    e2 = np.abs(v - S_ref*ph)[inb]
    print(f"        residual after removing the best per-bin phase: "
          f"{20*np.log10(np.sqrt(np.mean(e2**2))/np.sqrt(np.mean(s**2))):6.2f} dB (amplitude-only part)")
esr_fa = {k: 20*np.log10(np.sqrt(np.mean(np.abs(v-S_ref)**2,1))/np.sqrt(np.mean(np.abs(S_ref)**2,1)))
          for k, v in S.items()}
esr_2d = {k: 20*np.log10(np.abs(v-S_ref)/(np.abs(S_ref)+1e-30)) for k, v in S.items()}
ref_line = ref[:, nb]
foc = lambda d: np.stack([matched_filter(d[:, n], ref_line) for n in range(p.Nr)], 1)
img = {"mono": foc(ref), "no": foc(rec["no"]), "sub": foc(rec["sub"])}
pk = int(np.argmax(np.abs(img["mono"][:, nb])))
np.savez_compressed(os.path.join(OUT, "cache_check.npz"),
    fa=fa, fr=fr, abw=p.abw, prf=p.prf, ve=p.ve, pk=pk,
    esr_fa_no=esr_fa["no"], esr_fa_sub=esr_fa["sub"],
    esr2d_no=esr_2d["no"].astype(np.float32), esr2d_sub=esr_2d["sub"].astype(np.float32),
    mag_ref=np.abs(S_ref).astype(np.float32),
    irf_mono=img["mono"][:, nb], irf_no=img["no"][:, nb], irf_sub=img["sub"][:, nb])
print("cache written")
