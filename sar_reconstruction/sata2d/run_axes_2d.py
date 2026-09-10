# -*- coding: utf-8 -*-
"""
Diagnostic: which axis is which.

Reproduces the two figures the reference 2-D reconstruction code produced
(azimuth IRF + 2-D spectrum phase difference) and adds the RANGE (fast-time)
cut, so the slow-time / fast-time distinction is explicit.
"""
from __future__ import annotations
import os, sys
import numpy as np

if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)

from .geometry import (make_params3d, build_tracks_3d, generate_reference_3d,
                       generate_channels_3d)
from .reconstruction import (range_compress, CoeffTable3D, build_delta_C0_map_3d,
                             reconstruct_subband_2d, scatterer_range, range_bin_of,
                             matched_filter, _param_box)

C = 3.0e8
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT, exist_ok=True)

TARGET_AZ, TARGET_H, NRX, DXT, OSF = 0.0, 240.0, 4, 150.0, 4

p = make_params3d(Nrx=NRX, dxt=DXT, specs=((TARGET_AZ, TARGET_H),))
tr = build_tracks_3d(p)
nb = range_bin_of(p, scatterer_range(p, p.points[0]))
print(p.summary()); print("target range bin", nb, "/", p.Nr)

ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)
ch  = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
tab = CoeffTable3D(p, tr, n_nodes=16)
maps = [build_delta_C0_map_3d(p, tr, i) for i in range(p.Nrx)]
rec = reconstruct_subband_2d(p, tr, ch, tab, sata_osf=OSF, maps=maps,
                             use_sata="subband")

ref_line = ref[:, nb]
# azimuth focusing of every range bin with the same reference azimuth chirp
ref_img = np.stack([matched_filter(ref[:, n], ref_line) for n in range(p.Nr)], axis=1)
rec_img = np.stack([matched_filter(rec[:, n], ref_line) for n in range(p.Nr)], axis=1)
pk_az = int(np.argmax(np.abs(ref_img[:, nb])))
print("peak azimuth sample", pk_az, "of", p.Na)

# ---- axes -----------------------------------------------------------------
t_slow = (np.arange(p.Na) - pk_az) / p.prf                 # [s]  LONG time
x_az   = t_slow * p.ve                                     # [m]  ground azimuth
fa = np.fft.fftshift(np.fft.fftfreq(p.Na, 1.0 / p.prf))    # [Hz]
fr = np.fft.fftshift(np.fft.fftfreq(p.Nr, 1.0 / p.rsf))    # [Hz] baseband range freq
t_fast = (np.arange(p.Nr) - nb) / p.rsf                    # [s]  SHORT time
r_rel  = t_fast * C / 2.0                                  # [m]  slant range offset

S_ref = np.fft.fftshift(np.fft.fft2(ref))
S_rec = np.fft.fftshift(np.fft.fft2(rec))
mag = np.abs(S_ref)
mask = mag > 0.02 * mag.max()
dph = np.angle(S_rec * np.conj(S_ref), deg=True)
dph = np.where(mask, dph, np.nan)
off = np.nanmedian(dph)
print("median 2-D phase offset %.3f deg, rms in-band %.3f deg"
      % (off, np.sqrt(np.nanmean((dph - off) ** 2))))

# ---- plots ----------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

db = lambda z, n: 20 * np.log10(np.abs(z) / n + 1e-20)
nrm = np.abs(ref_img[:, nb]).max()

fig, ax = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
fig.suptitle("SATA2D / two-step RD -- which axis is which | " + _param_box(p),
             fontsize="medium")

# (a) azimuth IRF, SLOW time, full axis
ax[0, 0].plot(t_slow, db(ref_img[:, nb], nrm), color="#444", lw=1.0, label="ref (mono)")
ax[0, 0].plot(t_slow, db(rec_img[:, nb], nrm), color="#2E7D32", lw=1.0, label="rec (SATA sub-band)")
ax[0, 0].set_xlabel("slow (azimuth) time $t_a$ [s]   -- 1/PRF per sample")
ax[0, 0].set_ylabel("[dB]"); ax[0, 0].set_ylim(-100, 2); ax[0, 0].grid(alpha=.3)
ax[0, 0].legend(fontsize="small"); ax[0, 0].set_title("(a) azimuth IRF, full slow-time axis")

# (b) azimuth IRF zoom, slow time in ms + ground azimuth
w = 400
sl = slice(pk_az - w, pk_az + w)
ax[0, 1].plot(t_slow[sl] * 1e3, db(ref_img[sl, nb], nrm), color="#444", label="ref")
ax[0, 1].plot(t_slow[sl] * 1e3, db(rec_img[sl, nb], nrm), color="#2E7D32", label="rec")
ax[0, 1].set_xlabel("slow time $t_a$ [ms]"); ax[0, 1].set_ylabel("[dB]")
ax[0, 1].set_ylim(-60, 2); ax[0, 1].grid(alpha=.3); ax[0, 1].legend(fontsize="small")
ax[0, 1].set_title("(b) main lobe -- zoom")
sec = ax[0, 1].secondary_xaxis("top", functions=(lambda t: t * 1e-3 * p.ve,
                                                lambda x: x / p.ve * 1e3))
sec.set_xlabel("ground azimuth $x = v_e t_a$ [m]")

# (c) range IRF -- FAST time, for contrast
ax[1, 0].plot(t_fast * 1e6, db(ref_img[pk_az, :], np.abs(ref_img[pk_az, :]).max()),
              color="#444", marker=".", ms=3, label="ref")
ax[1, 0].plot(t_fast * 1e6, db(rec_img[pk_az, :], np.abs(ref_img[pk_az, :]).max()),
              color="#C62828", marker=".", ms=3, label="rec")
ax[1, 0].set_xlabel(r"fast (range) time $\tau$ [$\mu$s]  -- 1/rsf per sample")
ax[1, 0].set_ylabel("[dB]"); ax[1, 0].set_ylim(-60, 2); ax[1, 0].grid(alpha=.3)
ax[1, 0].legend(fontsize="small"); ax[1, 0].set_title("(c) RANGE IRF (fast time) -- the other axis")
sec2 = ax[1, 0].secondary_xaxis("top", functions=(lambda t: t * 1e-6 * C / 2,
                                                 lambda r: r / (C / 2) * 1e6))
sec2.set_xlabel(r"slant range offset $c\tau/2$ [m]")

# (d) 2-D spectrum phase difference
im = ax[1, 1].imshow(dph - off, origin="lower", aspect="auto", cmap="bwr",
                     vmin=-5, vmax=5,
                     extent=[fr.min() / 1e6, fr.max() / 1e6,
                             fa.min() / 1e3, fa.max() / 1e3])
fig.colorbar(im, ax=ax[1, 1], label="phase difference [deg]")
ax[1, 1].set_xlabel("range frequency $f_r$ [MHz]")
ax[1, 1].set_ylabel("azimuth (Doppler) frequency $f_a$ [kHz]")
ax[1, 1].set_title("(d) reconstructed vs ideal, 2-D spectrum")
ax[1, 1].axhline(+p.abw / 2e3, color="k", ls="-.", lw=.8)
ax[1, 1].axhline(-p.abw / 2e3, color="k", ls="-.", lw=.8)

fig.tight_layout()
out = os.path.join(OUT, "axes_2d.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("figure written to", out)
