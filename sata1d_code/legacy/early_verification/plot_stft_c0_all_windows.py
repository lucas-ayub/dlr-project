"""
STFT of the raw data and the delta_C0 read by each bin, for ALL windows of the
SATA loop (hop Tsub). x-axis: window centre. y-axis: the bin, named three ways
(theta, f_a, absolute azimuth cell x). Produces stft_c0_all_windows.png.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sata_kernel import step1_mapping
from pipeline_signal import pipeline_point_line
from pipeline_coeff import Params3D, build_tracks_3d, residual_C0_3d

wl, r0, v, H = 0.25, 766206.0, 7500.0, 720000.0
PRFop, rref, dimx, ch, h = 500.0, 766206.0, 512, 3, 240.0

s = pipeline_point_line(r0, H, v, wl, PRFop, dimx)
g = step1_mapping(dimx, rref, PRFop, 1, v, 0.0, wl, r0)
Tsub, Nzp = int(g["Tsub"]), g["Nzp"]
o = np.argsort(g["fsub"])
fa, theta, azpos = g["fsub"][o], np.rad2deg(g["betasub"][o]), g["azpos"][o]

# delta_C0 map (real residual from the pipeline fit, over the target footprint)
p = Params3D(v=v, H=H, wl=wl, r0=r0, prf_full=4*PRFop, bxt=(0., 100., 200., 300.))
dC0 = residual_C0_3d(p, build_tracks_3d(p), np.array([0.0, np.sqrt(r0**2-(H-h)**2), h]), ch)
foot = int(round(np.sqrt(wl*rref/2)/(v/PRFop)))
dmap = np.zeros(dimx); dmap[dimx//2-foot:dimx//2+foot+1] = dC0

# every window of the SATA loop: start = k*Tsub, length 2*Tsub, centre = start+Tsub
starts = np.arange(0, dimx-2*Tsub+1, Tsub)
centres = starts + Tsub
S = np.zeros((Nzp, len(starts))); R = np.full((Nzp, len(starts)), np.nan)
X = np.zeros((Nzp, len(starts)))
for k, st in enumerate(starts):
    buf = np.concatenate([s[st:st+2*Tsub], np.zeros(Nzp-2*Tsub, complex)])
    S[:, k] = np.abs(np.fft.fft(buf))[o]
    X[:, k] = np.round(azpos + centres[k])                 # posaux of each bin
    inside = (X[:, k] >= 0) & (X[:, k] < dimx)
    R[inside, k] = dmap[X[inside, k].astype(int)]          # delta_C0[posaux]
S_db = 20*np.log10(S/S.max() + 1e-9)
S_db[:, S.max(axis=0) == 0] = np.nan          # windows outside the beam: no signal

# consistency: in every window where the target is seen, does the peak bin
# point to cell 256 and read delta_C0?
seen = S_db.max(axis=0) > -20
pk = S.argmax(axis=0)
print(f"dC0(ch{ch}) = {dC0*100:.3f} cm, windows = {len(starts)}, windows seeing target = {seen.sum()}")
print("peak cell - 256 (seen windows):", np.unique(X[pk[seen], np.where(seen)[0]] - 256))
print("peak bin reads dC0 in", int(np.sum(R[pk[seen], np.where(seen)[0]] == dC0)), "of", seen.sum(), "seen windows")

C = np.tile(centres, (Nzp, 1))
ys = [(np.tile(theta[:, None], (1, len(starts))), r"look angle $\theta$ [deg]"),
      (np.tile(fa[:, None], (1, len(starts))), r"azimuth frequency $f_a$ [Hz]"),
      (X, r"azimuth cell $x$ = posaux [samples]")]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 130})
fig, A = plt.subplots(2, 3, figsize=(17, 9.5))
plt.subplots_adjust(wspace=.32, hspace=.3)
for j, (Y, lab) in enumerate(ys):
    im0 = A[0, j].pcolormesh(C, Y, S_db, shading="nearest", cmap="viridis", vmin=-40, vmax=0)
    im1 = A[1, j].pcolormesh(C, Y, np.where(np.isnan(R), np.nan, R*100), shading="nearest",
                             cmap="Oranges_r", vmin=dC0*100, vmax=0)
    for a in A[:, j]:
        a.set_ylabel(lab); a.set_xlabel("window centre [samples]")
        a.axvline(dimx//2, color="w" if a is A[0, j] else "grey", ls=":", lw=.8)
    if j == 2:
        for a in A[:, j]:
            a.axhline(dimx//2, color="r", ls="--", lw=.8)
            a.axhspan(dimx//2-foot, dimx//2+foot, color="r", alpha=.08)
fig.colorbar(im0, ax=A[0, :], label="STFT [dB]", shrink=.9)
fig.colorbar(im1, ax=A[1, :], label=r"$\Delta C_0$ read [cm]", shrink=.9)
A[0, 1].set_title("STFT of every SATA window (point target at cell 256)")
A[1, 1].set_title(rf"$\Delta C_0$[posaux] read by every bin of every window (ch {ch}, {dC0*100:.1f} cm)")
plt.savefig("stft_c0_all_windows.png", bbox_inches="tight"); plt.close()
print("saved stft_c0_all_windows.png")
