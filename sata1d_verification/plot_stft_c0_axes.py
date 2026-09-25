"""
STFT of the raw data and the delta_C0 read by each bin, against three x-axes:
look angle theta, azimuth frequency f_a, and azimuth cell x.
Same window, same bins -- only the x-axis changes (f_a -> theta -> x).
Produces stft_c0_axes.png.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sata_kernel import step1_mapping
from pipeline_signal import pipeline_point_line
from pipeline_coeff import Params3D, build_tracks_3d, residual_C0_3d

wl, r0, v, H = 0.25, 766206.0, 7500.0, 720000.0
PRFop, rref, dimx, ch = 500.0, 766206.0, 512, 3
h = 240.0

# raw data: one channel line of a point target (exact range history)
s = pipeline_point_line(r0, H, v, wl, PRFop, dimx)

# one STFT window centred on the target (exactly what the SATA loop takes)
g = step1_mapping(dimx, rref, PRFop, 1, v, 0.0, wl, r0)
Tsub, Nzp = int(g["Tsub"]), g["Nzp"]
start = dimx//2 - Tsub; centre = start + Tsub
buf = s[start:start+2*Tsub]
buf = np.concatenate([buf, np.zeros(Nzp-len(buf), complex)])
spec = np.fft.fft(buf)

# the three axes of every bin
fa    = g["fsub"]                          # [Hz]
theta = np.rad2deg(g["betasub"])           # [deg]
x     = np.round(g["azpos"] + centre)      # absolute azimuth cell [samples]

# delta_C0 map: real residual (pipeline fit) over the target footprint
p = Params3D(v=v, H=H, wl=wl, r0=r0, prf_full=4*PRFop, bxt=(0., 100., 200., 300.))
tracks = build_tracks_3d(p)
ptg = np.array([0.0, np.sqrt(r0**2-(H-h)**2), h])
dC0 = residual_C0_3d(p, tracks, ptg, ch)
foot = int(round(np.sqrt(wl*rref/2)/(v/PRFop)))
dmap = np.zeros(dimx); dmap[dimx//2-foot:dimx//2+foot+1] = dC0
read = dmap[np.clip(x.astype(int), 0, dimx-1)]      # what each bin reads

o = np.argsort(fa)
stft_db = 20*np.log10(np.abs(spec)[o]/np.abs(spec).max() + 1e-9)
axes = [(theta[o], r"look angle $\theta$ [deg]"),
        (fa[o],    r"azimuth frequency $f_a$ [Hz]"),
        (x[o],     r"azimuth cell $x$ [samples]")]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 130})
fig, A = plt.subplots(2, 3, figsize=(14, 7.5), sharey="row")
for j, (xx, lab) in enumerate(axes):
    A[0, j].plot(xx, stft_db, "o-", ms=3, lw=1.2, color="#185FA5")
    A[0, j].set_xlabel(lab); A[0, j].grid(alpha=.3); A[0, j].set_ylim(-40, 3)
    A[1, j].step(xx, read[o]*100, where="mid", color="#D85A30", lw=1.5)
    A[1, j].plot(xx, read[o]*100, "o", ms=3, color="#D85A30")
    A[1, j].set_xlabel(lab); A[1, j].grid(alpha=.3)
A[0, 0].set_ylabel("STFT of raw data [dB]")
A[1, 0].set_ylabel(r"$\Delta C_0$ read by the bin [cm]")
A[0, 1].set_title("STFT of one window (point target) vs three x-axes")
A[1, 1].set_title(rf"$\Delta C_0$ each bin reads (footprint $\pm${foot} cells, ch {ch})")
A[1, 2].axvspan(dimx//2-foot, dimx//2+foot, color="#D85A30", alpha=.08)
A[1, 2].axvline(dimx//2, color="grey", ls=":", lw=.8)
plt.tight_layout(); plt.savefig("stft_c0_axes.png"); plt.close()
print(f"dC0(ch{ch}) = {dC0*100:.3f} cm, foot = {foot} cells, "
      f"bins reading dC0: {int(np.sum(read!=0))} of {Nzp}")
