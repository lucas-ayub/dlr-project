# -*- coding: utf-8 -*-
"""Do the C1 / C2 residuals matter?  Measure the coefficients, then run an oracle."""
from __future__ import annotations
import os, sys
import numpy as np
if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)
from .geometry import (make_params3d, build_tracks_3d, generate_reference_3d,
                       generate_channels_3d, CoeffTable3D, _coeff)
from .reconstruction import (range_compress, build_delta_C0_map_3d, reconstruct_subband_2d,
                             scatterer_range, range_bin_of, matched_filter, METHOD_KW)

p = make_params3d(Nrx=4, dxt=150.0, specs=((0.0, 240.0),))
tr = build_tracks_3d(p)
ptg = np.asarray(p.points[0], float)
r = float(np.sqrt(ptg[1]**2 + (p.H - ptg[2])**2))
flat = p.flat_point_at_range(r); flat[0] = ptg[0]
T, wl = p.int_time, p.wl
print(f"target h = {ptg[2]:.0f} m, r = {r/1e3:.3f} km, T_int = {T:.3f} s, wl = {wl} m\n")
print(f"{'ch':>3} {'dC0 [mm]':>11} {'dC1 [mm/s]':>12} {'dC2 [mm/s2]':>12} "
      f"{'ph0 [deg]':>11} {'ph1 pp [deg]':>13} {'ph2 [deg]':>11}")
tot = []
for i in range(p.Nrx):
    a, b = _coeff(p, tr, ptg, i), _coeff(p, tr, flat, i)
    d = np.array(a[:3]) - np.array(b[:3])
    ph0 = 360*d[0]/wl
    ph1 = 360*d[1]*T/wl              # peak-to-peak over the aperture
    ph2 = 360*d[2]*(T/2)**2/wl       # edge value
    tot.append((ph0, ph1, ph2))
    print(f"{i:>3} {d[0]*1e3:11.3f} {d[1]*1e3:12.4f} {d[2]*1e3:12.4f} "
          f"{ph0:11.1f} {ph1:13.3f} {ph2:11.3f}")
t = np.array(tot)
print(f"\nmax |phase|:  C0 {np.abs(t[:,0]).max():.1f} deg | "
      f"C1 {np.abs(t[:,1]).max():.3f} deg (pp) | C2 {np.abs(t[:,2]).max():.3f} deg (edge)")
print(f"ratio C1/C0 = {np.abs(t[:,1]).max()/np.abs(t[:,0]).max():.2e} ; "
      f"C2/C0 = {np.abs(t[:,2]).max()/np.abs(t[:,0]).max():.2e}")

# ---- reconstructions -------------------------------------------------------
ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)
ch  = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
nb  = range_bin_of(p, scatterer_range(p, ptg))
tab_flat = CoeffTable3D(p, tr, n_nodes=16)                       # filter at h0 (current)
tab_true = CoeffTable3D(p, tr, n_nodes=16, height=float(ptg[2])) # ORACLE: all of C0,C1,C2 right
maps = [build_delta_C0_map_3d(p, tr, i) for i in range(p.Nrx)]

runs = {
    "no SATA":        (tab_flat, dict(use_sata=False)),
    "SATA C0 only":   (tab_flat, dict(use_sata="subband")),
    "oracle h_true":  (tab_true, dict(use_sata=False)),
}
S_ref = np.fft.fftshift(np.fft.fft2(ref))
fa = np.fft.fftshift(np.fft.fftfreq(p.Na, 1/p.prf)); inb = np.abs(fa) < p.abw/2
ref_line = ref[:, nb]; pm = None
print(f"\n{'case':>15} {'peak':>11} {'% of mono':>10} {'err/sig in-band [dB]':>22}")
mono = matched_filter(ref[:, nb], ref_line)
pm = float(np.abs(mono).max())
print(f"{'monostatic':>15} {pm:11.3e} {100.0:9.1f}% {'--':>22}")
for lab, (tb, kw) in runs.items():
    rec = reconstruct_subband_2d(p, tr, ch, tb, sata_osf=4, maps=maps, **kw)
    v = float(np.abs(matched_filter(rec[:, nb], ref_line)).max())
    S = np.fft.fftshift(np.fft.fft2(rec))
    e = np.abs(S - S_ref)[inb]; s = np.abs(S_ref)[inb]
    esr = 20*np.log10(np.sqrt(np.mean(e**2))/np.sqrt(np.mean(s**2)))
    print(f"{lab:>15} {v:11.3e} {100*v/pm:9.1f}% {esr:21.2f}")
