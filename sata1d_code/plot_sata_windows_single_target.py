"""
SATA window by window for one point target (whole-band kernel, one channel).
One page per sub-aperture window, columns = bin axis (look angle, Doppler, cell x = posaux):
  1) |STFT| of the window (dots: bins where the target appears, > -30 dB;
     red = the bin reads the target's delta_C0, open = it does not)
  2) delta_C0[posaux] read by every bin (shaded: cells where the map is non-zero)
  3) phase applied, ph = -2 pi / wl * delta_C0
  4) rotation measured on the target's bins
Last page: the map and the phase added to every sample vs the ideal correction.
Output: sata_windows_single_target.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sata1d as S
from sata1d.sata import subaperture_sizing, subaperture_axis, triangular_window

H_TGT, CH, OSF, DB_MIN = 240.0, 0, 4, -30.0

cfg = S.make_config([(0.0, H_TGT)])
tr = S.build_platform_tracks(cfg)
wl, v, r, P = cfg.system.wl, cfg.system.vs, cfg.scene.r0, cfg.PRF_op
s = S.generate_channels(cfg, tr)[CH]
dimx = len(s)
dC0 = S.residual_C0(cfg, tr, cfg.scene.points[1], CH)
dmap = S.build_delta_C0_array(cfg, tr, CH)
xt = S.az_pixel_of_scatterer(cfg, 0.0, CH)

# replay of the sata_1d loop (identical operations), keeping every window
Tsubeff, hop, Nzp = subaperture_sizing(r, P, v, wl, OSF)
win = triangular_window(Tsubeff)
fsub, azpos = subaperture_axis(P, Nzp, v, wl, r)
log = []
for start in range(0, dimx, hop):
    L = len(s[start:start + Tsubeff])
    if L < 2:
        break
    buf = np.zeros(Nzp, complex); buf[:L] = s[start:start + L] * win[:L]
    spec = np.fft.fft(buf)
    c = start + 0.5 * Tsubeff
    posaux = np.clip(np.round(azpos + c).astype(int), 0, dimx - 1)
    ph = -2 * np.pi / wl * dmap[posaux]
    log.append(dict(start=start, L=L, c=c, posaux=posaux, ph=ph, spec=spec, after=spec * np.exp(-1j * ph)))

out = S.sata_1d(s, dmap, r, P, v, wl, r, inverse=True, sata_osf=OSF)
ideal = s * np.exp(1j * 2 * np.pi / wl * dC0)
ill = np.abs(s) > 0
err = np.linalg.norm((out - ideal)[ill]) / np.linalg.norm(ideal[ill])
print(f"channel {CH}: dC0 = {dC0 * 100:.3f} cm, target cell {xt}, error vs ideal correction {err:.3f}")

o = np.argsort(fsub)
gmax = max(np.abs(w["spec"]).max() for w in log)
expected = np.rad2deg(np.angle(np.exp(1j * 2 * np.pi / wl * dC0)))
segs = np.flatnonzero(np.diff(np.r_[0, (dmap != 0).astype(int), 0])).reshape(-1, 2)
idx_ill = np.flatnonzero(ill)
show = [k for k, w in enumerate(log) if idx_ill[0] - 40 <= w["c"] <= idx_ill[-1] + 40]

plt.rcParams.update({"font.size": 10, "figure.dpi": 90})
with PdfPages("sata_windows_single_target.pdf") as pdf:
    for k in show:
        w = log[k]
        db = 20 * np.log10(np.abs(w["spec"])[o] / gmax + 1e-12)
        E, C = db > DB_MIN, (w["ph"] != 0)[o]
        rot = np.rad2deg(np.angle(w["after"] * np.conj(w["spec"])))[o]
        fig, A = plt.subplots(4, 3, figsize=(14, 12), sharey="row")
        for j, (xx, lab, lim) in enumerate([
                (np.rad2deg(np.arcsin(wl * fsub / (2 * v)))[o], r"look angle $\theta$ [deg]", (-.25, .25)),
                (fsub[o], r"Doppler $f_a$ [Hz]", (-260, 260)),
                (w["posaux"][o], "cell x = posaux", (xt - 700, xt + 700))]):
            A[0, j].plot(xx, db, "-", lw=.8, color="tab:blue")
            A[0, j].plot(xx[E & C], db[E & C], "o", ms=3.5, color="tab:red", label="target, corrected")
            A[0, j].plot(xx[E & ~C], db[E & ~C], "o", ms=3.5, mfc="none", color="k", label="target, not corrected")
            A[1, j].plot(xx, dmap[w["posaux"]][o] * 100, "o", ms=2.5, color="tab:red")
            A[2, j].plot(xx, w["ph"][o], "o", ms=2.5, color="tab:purple")
            A[3, j].plot(xx[E], rot[E], "o", ms=3.5, color="tab:green")
            A[3, j].axhline(expected, color="grey", ls=":")
            for a in A[:, j]:
                a.set_xlim(*lim); a.grid(alpha=.3)
            A[3, j].set_xlabel(lab)
        for a in A[:, 2]:
            for lo, hi in segs:
                a.axvspan(lo, hi - 1, color="tab:red", alpha=.1)
            a.axvline(xt, color="grey", ls=":"); a.axvline(w["c"], color="tab:blue", ls="--")
        A[0, 0].legend(fontsize=8, loc="lower left")
        A[0, 0].set_ylim(-60, 3); A[3, 0].set_ylim(-180, 180)
        A[0, 0].set_ylabel("|STFT| [dB]"); A[1, 0].set_ylabel(r"$\Delta C_0$[posaux] [cm]")
        A[2, 0].set_ylabel("ph [rad]"); A[3, 0].set_ylabel("rotation [deg]")
        fig.suptitle(f"window {k + 1}/{len(log)}, centre {w['c']:.0f} | channel {CH}, "
                     f"$\\Delta C_0$ = {dC0 * 100:.2f} cm (ideal rotation {-expected:+.1f} deg removed)",
                     fontsize=12)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    n = np.arange(dimx)
    fig, A = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    A[0].plot(n, dmap * 100, color="tab:red"); A[0].axvline(xt, color="grey", ls=":")
    A[0].set_ylabel(r"$\Delta C_0$ map [cm]"); A[0].set_title("delta_C0 map (target block and alias images)")
    A[1].plot(n[ill], np.rad2deg(np.angle(out[ill] * np.conj(s[ill]))), ".", ms=2, color="tab:red",
              label=f"SATA output (error vs ideal {err:.3f})")
    A[1].axhline(expected, color="k", ls=":", label=f"ideal {expected:.1f} deg")
    A[1].set_ylim(-180, 180); A[1].set_ylabel("arg(out) - arg(in) [deg]"); A[1].legend()
    A[1].set_xlabel("azimuth sample"); A[1].set_xlim(idx_ill[0] - 100, idx_ill[-1] + 100)
    for a in A: a.grid(alpha=.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
print(f"saved sata_windows_single_target.pdf ({len(show) + 1} pages)")
