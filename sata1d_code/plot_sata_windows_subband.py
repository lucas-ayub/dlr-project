"""
SATA window by window for one point target, per-sub-band kernel of ONE output band k (one channel).
Same layout as plot_sata_windows_single_target.py, with the bins labelled by the absolute Doppler of band k:
  columns = bin axis: look angle theta(f~), absolute Doppler f~ in [f_k - PRF/2, f_k + PRF/2), cell x = posaux
  rows    = 1) |STFT| (dots: bins where the target appears, > -30 dB; red = the bin reads the target's
               delta_C0, open = it does not; the target's true Doppler in this window is marked, and
               whether it lies in band k)
            2) delta_C0[posaux] read by every bin (shaded: cells where the band-k map is non-zero)
            3) phase applied, ph = -2 pi / wl * delta_C0
            4) rotation measured on the target's bins
Last page: the band-k map and the phase added to every sample vs the ideal correction.
Usage: python plot_sata_windows_subband.py [--band 2] [--channel 0] [--height 240]
Output: sata_windows_subband_k<band>.pdf
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sata1d as S
from sata1d.sata import subaperture_sizing, triangular_window
from sata1d.subband import subband_axis, subband_frequency_beam

ap = argparse.ArgumentParser()
ap.add_argument("--band", type=int, default=2)
ap.add_argument("--channel", type=int, default=0)
ap.add_argument("--height", type=float, default=240.0)
a = ap.parse_args()
K, CH, H_TGT, OSF, DB_MIN = a.band, a.channel, a.height, 4, -30.0

cfg = S.make_config([(0.0, H_TGT)])
tr = S.build_platform_tracks(cfg)
wl, v, r, P = cfg.system.wl, cfg.system.vs, cfg.scene.r0, cfg.PRF_op
f_k, beta_k, _, _ = subband_frequency_beam(cfg, K)
s = S.generate_channels(cfg, tr)[CH]
dimx = len(s)
tgt = cfg.scene.points[1]
dC0 = S.residual_C0_subband(cfg, tr, tgt, CH, K)
dmap = S.build_delta_C0_subband_array(cfg, tr, CH, K)
xt = S.az_pixel_of_scatterer(cfg, 0.0, CH)
X = r * np.tan(np.arcsin(wl * P / (2 * v))) / v * P

# true (unfolded) Doppler of the target at every channel sample, from the simulated range history
rh = np.linalg.norm(tr.ptx - tgt, axis=1) + np.linalg.norm(tr.prx[CH] - tgt, axis=1)
f_true = (-np.gradient(rh, 1 / cfg.prf) / wl)[::cfg.Nrx]

# replay of the sata_1d_subband loop (identical operations), keeping every window
Tsubeff, hop, Nzp = subaperture_sizing(r, P, v, wl, OSF)
win = triangular_window(Tsubeff)
fsub, azpos = subband_axis(P, Nzp, v, wl, r, f_k)
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
    ft = f_true[min(int(c), dimx - 1)]
    log.append(dict(start=start, L=L, c=c, posaux=posaux, ph=ph, spec=spec, after=spec * np.exp(-1j * ph),
                    f_true=ft, own=abs(ft - f_k) < P / 2))

out = S.sata_1d_subband(s, dmap, r, P, v, wl, r, f_k=f_k, inverse=True, sata_osf=OSF)
ideal = s * np.exp(1j * 2 * np.pi / wl * dC0)
ill = np.abs(s) > 0
err = np.linalg.norm((out - ideal)[ill]) / np.linalg.norm(ideal[ill])
print(f"band {K} (f_k = {f_k:+.0f} Hz), channel {CH}: dC0 = {dC0 * 100:.3f} cm, target cell {xt}, "
      f"error vs ideal correction {err:.3f}")

o = np.argsort(fsub)
gmax = max(np.abs(w["spec"]).max() for w in log)
expected = np.rad2deg(np.angle(np.exp(1j * 2 * np.pi / wl * dC0)))
segs = np.flatnonzero(np.diff(np.r_[0, (dmap != 0).astype(int), 0])).reshape(-1, 2)
idx_ill = np.flatnonzero(ill)
show = [k for k, w in enumerate(log) if idx_ill[0] - 40 <= w["c"] <= idx_ill[-1] + 40]
theta = np.rad2deg(np.arcsin(wl * fsub / (2 * v)))

plt.rcParams.update({"font.size": 10, "figure.dpi": 90})
fname = f"sata_windows_subband_k{K}.pdf"
with PdfPages(fname) as pdf:
    for k in show:
        w = log[k]
        db = 20 * np.log10(np.abs(w["spec"])[o] / gmax + 1e-12)
        E, C = db > DB_MIN, (w["ph"] != 0)[o]
        rot = np.rad2deg(np.angle(w["after"] * np.conj(w["spec"])))[o]
        # label of the target's energy on this band's axis, and the cell that label points to
        f_lab = f_k - P / 2 + np.mod(w["f_true"] - (f_k - P / 2), P)
        m_fold = int(np.round((w["f_true"] - f_lab) / P))
        fig, A = plt.subplots(4, 3, figsize=(14, 12), sharey="row")
        for j, (xx, lab, lim) in enumerate([
                (theta[o], r"look angle $\theta$ [deg]", (theta.min() - .01, theta.max() + .01)),
                (fsub[o], r"absolute Doppler $\tilde f$ in band $k$ [Hz]", (f_k - P / 2 - 10, f_k + P / 2 + 10)),
                (w["posaux"][o], "cell x = posaux", (xt - 900, xt + 900))]):
            A[0, j].plot(xx, db, "-", lw=.8, color="tab:blue")
            A[0, j].plot(xx[E & C], db[E & C], "o", ms=3.5, color="tab:red", label="target, corrected")
            A[0, j].plot(xx[E & ~C], db[E & ~C], "o", ms=3.5, mfc="none", color="k", label="target, not corrected")
            A[1, j].plot(xx, dmap[w["posaux"]][o] * 100, "o", ms=2.5, color="tab:red")
            A[2, j].plot(xx, w["ph"][o], "o", ms=2.5, color="tab:purple")
            A[3, j].plot(xx[E], rot[E], "o", ms=3.5, color="tab:green")
            A[3, j].axhline(expected, color="grey", ls=":")
            for aa in A[:, j]:
                aa.set_xlim(*lim); aa.grid(alpha=.3)
            A[3, j].set_xlabel(lab)
        for aa in A[:, 1]:
            aa.axvline(f_lab, color="tab:orange", ls="--", lw=1)
        for aa in A[:, 2]:
            for lo, hi in segs:
                aa.axvspan(lo, hi - 1, color="tab:red", alpha=.1)
            aa.axvline(xt, color="grey", ls=":"); aa.axvline(w["c"], color="tab:blue", ls="--")
            for mm in (-3, -2, -1, 1, 2, 3):
                aa.axvline(xt + mm * X, color="grey", ls=":", lw=.6)
        A[0, 0].legend(fontsize=8, loc="lower left")
        A[0, 0].set_ylim(-60, 3); A[3, 0].set_ylim(-180, 180)
        A[0, 0].set_ylabel("|STFT| [dB]"); A[1, 0].set_ylabel(r"$\Delta C_0$[posaux] [cm]")
        A[2, 0].set_ylabel("ph [rad]"); A[3, 0].set_ylabel("rotation [deg]")
        where = "own band: points at $x_t$" if w["own"] else f"other band: folded, points at $x_t{m_fold * -1:+d}X$"
        fig.suptitle(f"band k = {K} ($f_k$ = {f_k:+.0f} Hz) | window {k + 1}/{len(log)}, centre {w['c']:.0f} | "
                     f"channel {CH}, $\\Delta C_0$ = {dC0 * 100:.2f} cm\n"
                     f"true Doppler of the target {w['f_true']:+.0f} Hz $\\to$ label {f_lab:+.0f} Hz (orange) | {where} "
                     f"| ideal rotation {expected:+.1f} deg", fontsize=11)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # summary of the mapping: every window's peak -> label -> cell
    cw = np.array([w["c"] for w in log]); keep = np.array([idx_ill[0] <= w["c"] <= idx_ill[-1] for w in log])
    pk = np.array([int(np.argmax(np.abs(w["spec"]))) for w in log])
    lab_pk = fsub[pk]; cell_pk = np.array([w["posaux"][j] for w, j in zip(log, pk)])
    own = np.array([w["own"] for w in log]); ftw = np.array([w["f_true"] for w in log])
    rot_pk = np.array([np.rad2deg(np.angle(w["after"][j] * np.conj(w["spec"][j]))) for w, j in zip(log, pk)])
    fig, A = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    A[0].plot(cw[keep], ftw[keep], "-", color="0.5", label="true Doppler")
    A[0].plot(cw[keep & own], lab_pk[keep & own], "o", ms=3, color="tab:green", label="peak label, own band")
    A[0].plot(cw[keep & ~own], lab_pk[keep & ~own], "o", ms=3, color="tab:orange", label="peak label, folded")
    A[0].axhspan(f_k - P / 2, f_k + P / 2, color="tab:orange", alpha=.1)
    A[0].set_ylabel("Doppler [Hz]"); A[0].legend(fontsize=8)
    A[0].set_title(f"band k = {K}: the f_a map, window by window (peak of each window)")
    A[1].plot(cw[keep], np.rad2deg(np.arcsin(wl * lab_pk[keep] / (2 * v))), "o", ms=3, color="tab:blue")
    A[1].set_ylabel(r"$\theta(\tilde f)$ of the peak [deg]")
    A[2].plot(cw[keep & own], cell_pk[keep & own], "o", ms=3, color="tab:green", label="own band")
    A[2].plot(cw[keep & ~own], cell_pk[keep & ~own], "o", ms=3, color="tab:orange", label="folded")
    for mm in range(-3, 4):
        A[2].axhline(xt + mm * X, color="grey", ls=":", lw=.8 if mm else 1.2)
    A[2].set_ylabel("cell the peak points to"); A[2].legend(fontsize=8)
    A[3].plot(cw[keep], rot_pk[keep], "o", ms=3, color="tab:purple")
    A[3].axhline(expected, color="k", ls=":"); A[3].set_ylim(-180, 180)
    A[3].set_ylabel("rotation on the peak [deg]"); A[3].set_xlabel("window centre [cell]")
    for aa in A: aa.grid(alpha=.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    n = np.arange(dimx)
    fig, A = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    A[0].plot(n[ill], f_true[ill], color="tab:blue", label="true Doppler of the target")
    A[0].axhspan(f_k - P / 2, f_k + P / 2, color="tab:orange", alpha=.15, label=f"band k = {K}")
    A[0].set_ylabel("Doppler [Hz]"); A[0].legend(fontsize=8)
    A[0].set_title("which part of the illumination belongs to band k (own band points at $x_t$, the rest at $x_t + mX$)")
    A[1].plot(n, dmap * 100, color="tab:red"); A[1].axvline(xt, color="grey", ls=":")
    for mm in (-3, -2, -1, 1, 2, 3):
        A[1].axvline(xt + mm * X, color="grey", ls=":", lw=.6)
    A[1].set_ylabel(r"$\Delta C_0$ map [cm]"); A[1].set_title(f"delta_C0 map of band {K} (direct block and fold images)")
    A[2].plot(n[ill], np.rad2deg(np.angle(out[ill] * np.conj(s[ill]))), ".", ms=2, color="tab:red",
              label=f"SATA band {K} output (error vs ideal {err:.3f})")
    A[2].axhline(expected, color="k", ls=":", label=f"ideal {expected:.1f} deg")
    A[2].set_ylim(-180, 180); A[2].set_ylabel("arg(out) - arg(in) [deg]"); A[2].legend()
    A[2].set_xlabel("azimuth sample"); A[2].set_xlim(0, dimx)
    for aa in A: aa.grid(alpha=.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
print(f"saved {fname} ({len(show) + 2} pages)")
