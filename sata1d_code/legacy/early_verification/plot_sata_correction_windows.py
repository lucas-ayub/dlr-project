"""
Where and how SATA applies the C0 correction -- window by window -- with the
pipeline's delta_C0 map "only where the target appears in the STFT".

Everything comes from the repo's sar_recon package:
  * config  : make_topo_config (Nrx=4, bxt = -150..150 m, PRF_op = 500 Hz)
  * target  : ONE scatterer at h = 240 m, same slant range as the scene centre
  * signal  : getRawData1D of that target only, channel line at PRF_op
  * map     : build_delta_C0_array (mode="footprint": main lobe + alias images)
  * kernel  : sata_1d (triangular WOLA, sata_osf=4, inverse=True), whose STEP-2
              loop is replayed here and checked to give the SAME output.

One page per window (columns = bin axis theta / f_a / x=posaux):
  1) |STFT|: where the target appears (dots = bins with energy; red = corrected)
  2) delta_C0[posaux] each bin reads (shaded = where the map is non-zero)
  3) ph = -2pi/wl * delta_C0 computed for each bin
  4) arg(spec after) - arg(spec before) on the bins with energy
Last pages: the map, and the line after SATA vs the ideal correction
(new map vs the legacy constant map).
Produces sata_correction_windows.pdf.
"""
import os, sys, tempfile, dataclasses
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sar_reconstruction"))
from sar_recon.config import make_topo_config
from sar_recon.geometry import build_platform_tracks
from sar_recon.signal_model import getRawData1D
from sar_recon.sata import (sata_1d, residual_C0, build_delta_C0_array,
                            az_pixel_of_scatterer, sata_footprint_halfwidth, sata_alias_images)

H_TGT, CH, OSF, INVERSE, DB_MIN = 240.0, 3, 4, True, -30.0

# ---- configuration, target, signal ----------------------------------------
cfg = make_topo_config(4, tempfile.mkdtemp(), "single", dxt=100.0)
sc = cfg.scene
dy = np.sqrt(sc.r0**2 - (sc.H - H_TGT)**2) - sc.y0          # iso-range with the centre
cfg = dataclasses.replace(cfg, scene=dataclasses.replace(sc, extra_offsets=((0.0, dy, H_TGT),)))
tr = build_platform_tracks(cfg)
wl, v, r, P = cfg.system.wl, cfg.system.vs, cfg.scene.r0, cfg.PRF_op
tgt = cfg.scene.points[1:]                                   # the elevated target only
s = getRawData1D(tgt, tr.ptx, tr.prx[CH], tr.vtx, tr.vrx[CH], cfg.ta, cfg.sq_tx, cfg.sq_tx,
                 cfg.theta_tx, cfg.theta_tx, wl, cfg.prf)[::cfg.Nrx].astype(complex)
dimx = len(s)

dC0 = residual_C0(cfg, tr, tgt[0], CH)
dmap = build_delta_C0_array(cfg, tr, CH)                     # NEW default: footprint (+ aliases)
dmap_old = build_delta_C0_array(cfg, tr, CH, mode="hold")    # legacy: constant over the line
xt = az_pixel_of_scatterer(cfg, 0.0, CH); hw = sata_footprint_halfwidth(cfg)
X, M = sata_alias_images(cfg)

# ---- STEP 1 / STEP 2 of sata_1d, replayed and recorded -------------------
deltax = np.sqrt(wl*r/2.0)
Tsubeff = int(np.round(deltax*P/v*0.5)*2); Tsub = Tsubeff//2
Nzp = int(2**np.ceil(np.log2(Tsubeff))*OSF)
win = np.concatenate((np.arange(Tsub)/(Tsub-1), (np.arange(Tsub)/(Tsub-1))[::-1]))
fsub = np.arange(Nzp)*P/Nzp - P*0.5; fsub[0] = P*0.5; fsub = np.roll(fsub, Nzp//2)
betasub = np.arcsin(np.clip(wl*fsub/(2*v), -1, 1))
azpos = r*np.tan(betasub)/v*P

out = np.zeros(dimx, complex); wsum = np.zeros(dimx); log = []
for start in range(0, dimx, Tsub):
    seg = s[start:start+Tsubeff]; L = len(seg)
    if L < 2: break
    buf = np.zeros(Nzp, complex); buf[:L] = seg*win[:L]
    spec = np.fft.fft(buf)
    centre = start + 0.5*Tsubeff
    posaux = np.clip(np.round(azpos + centre).astype(int), 0, dimx-1)
    ph = -2*np.pi/wl*dmap[posaux]
    after = spec*(np.exp(-1j*ph) if INVERSE else np.exp(1j*ph))
    out[start:start+L] += np.fft.ifft(after)[:L]; wsum[start:start+L] += win[:L]
    log.append(dict(start=start, L=L, centre=centre, posaux=posaux, ph=ph, spec=spec, after=after))
nz = wsum > 1e-12; out[nz] /= wsum[nz]; out[~nz] = s[~nz]

kw = dict(rref=r, prf=P, v=v, wl=wl, r=r, inverse=INVERSE, sata_osf=OSF, verbose=False)
ref = sata_1d(s, dmap, **kw); ref_old = sata_1d(s, dmap_old, **kw)
ideal = s*np.exp(1j*2*np.pi/wl*dC0)                         # every sample rotated by exactly dC0
ill = np.abs(s) > 0
err = lambda o: np.linalg.norm((o-ideal)[ill])/np.linalg.norm(ideal[ill])
print(f"ch{CH}: dC0 = {dC0*100:.3f} cm, target pixel {xt}, footprint +-{hw} px, alias images +-{X:.0f} px (M={M})")
print(f"replayed loop vs sar_recon.sata_1d: {np.max(np.abs(out-ref))/np.max(np.abs(ref)):.1e}")
print(f"error vs ideal: new map {err(ref):.3f}, legacy constant map {err(ref_old):.3f}")

# ---- which windows / bins carry the target, and are they corrected? ------
o = np.argsort(fsub)
gmax = max(np.abs(w["spec"]).max() for w in log)
for w in log:
    w["db"] = 20*np.log10(np.abs(w["spec"])/gmax + 1e-12)
    w["E"] = w["db"] > DB_MIN                                # bins where the target appears
    w["C"] = w["ph"] != 0                                    # bins that get corrected
tot_E = sum(w["E"].sum() for w in log); tot_EC = sum((w["E"] & w["C"]).sum() for w in log)
tot_C_noE = sum((~w["E"] & w["C"] & (w["db"] > -300)).sum() for w in log)
print(f"bins with target energy: {tot_E}, of which corrected: {tot_EC} ({100*tot_EC/tot_E:.1f}%)")

idx_ill = np.flatnonzero(ill)
show = [k for k, w in enumerate(log) if idx_ill[0]-40 <= w["centre"] <= idx_ill[-1]+40]
expected = np.rad2deg(np.angle(np.exp(1j*2*np.pi/wl*dC0)))
segs = np.flatnonzero(np.diff(np.r_[0, (dmap != 0).astype(int), 0])).reshape(-1, 2)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "figure.dpi": 90})
with PdfPages("sata_correction_windows.pdf") as pdf:
    for k in show:
        w = log[k]; E, C = w["E"][o], w["C"][o]
        rot = np.rad2deg(np.angle(w["after"]*np.conj(w["spec"])))[o]
        fig, A = plt.subplots(4, 3, figsize=(14, 12), sharey="row")
        for j, (xx, lab, lim) in enumerate([
                (np.rad2deg(betasub[o]), r"look angle $\theta$ [deg]", (-.25, .25)),
                (fsub[o], r"azimuth frequency $f_a$ [Hz]", (-260, 260)),
                (w["posaux"][o], r"azimuth cell $x$ = posaux [samples]", (xt-700, xt+700))]):
            A[0, j].plot(xx, w["db"][o], "-", lw=.8, color="#185FA5")
            A[0, j].plot(xx[E & C], w["db"][o][E & C], "o", ms=3.5, color="#D85A30", label="target, corrected")
            A[0, j].plot(xx[E & ~C], w["db"][o][E & ~C], "o", ms=3.5, mfc="none", color="k", label="target, NOT corrected")
            A[1, j].plot(xx, dmap[w["posaux"]][o]*100, "o", ms=2.5, color="#D85A30")
            A[2, j].plot(xx, w["ph"][o], "o", ms=2.5, color="#7F4FC9")
            if E.any():
                A[3, j].plot(xx[E], rot[E], "o", ms=3.5, color="#1D9E75")
            A[3, j].axhline(expected, color="grey", ls=":", lw=1)
            for a in A[:, j]:
                a.set_xlim(*lim); a.grid(alpha=.3)
            A[3, j].set_xlabel(lab)
        for a in A[:, 2]:
            for lo, hi in segs:
                a.axvspan(lo, hi-1, color="#D85A30", alpha=.10)
            a.axvline(xt, color="grey", ls=":", lw=.8)
            a.axvline(w["centre"], color="#185FA5", ls="--", lw=.8)
        A[0, 0].legend(fontsize=8, loc="lower left")
        A[0, 0].set_ylim(-60, 3); A[1, 0].set_ylim(min(dC0*100, 0)-3, max(dC0*100, 0)+3)
        A[2, 0].set_ylim(min(-2*np.pi/wl*dC0, 0)-1, max(-2*np.pi/wl*dC0, 0)+1); A[3, 0].set_ylim(-180, 180)
        A[0, 0].set_ylabel("|STFT| [dB]"); A[1, 0].set_ylabel(r"$\Delta C_0$[posaux] [cm]")
        A[2, 0].set_ylabel("ph [rad]"); A[3, 0].set_ylabel("rotation applied\nto the bin [deg]")
        A[0, 1].set_title("1) where the target appears in the STFT (dots: > -30 dB)")
        A[1, 1].set_title(r"2) $\Delta C_0$ each bin reads (shaded: map $\neq$ 0 = target footprint + alias images)")
        A[2, 1].set_title(r"3) phase computed for each bin, ph $= -2\pi/\lambda\,\Delta C_0$")
        A[3, 1].set_title(f"4) rotation that happened on the target bins (dotted = {expected:.1f} deg)")
        nE, nEC = int(E.sum()), int((E & C).sum())
        fig.suptitle(f"Window {k+1}/{len(log)}: samples {w['start']}-{w['start']+w['L']-1}, centre {w['centre']:.0f}"
                     f"  |  target bins: {nE}, corrected: {nEC}  |  ch {CH}, $\\Delta C_0$ = {dC0*100:.2f} cm",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- summary: the map and the result -------------------------------------
    n = np.arange(dimx)
    fig, A = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    A[0].plot(n, dmap_old*100, color="grey", lw=1.2, label="legacy (hold): constant over the line")
    A[0].plot(n, dmap*100, color="#D85A30", lw=1.5, label="new (footprint): only where the target appears")
    A[0].axvline(xt, color="grey", ls=":", lw=.8)
    A[0].set_ylabel(r"$\Delta C_0$ map [cm]"); A[0].legend(fontsize=9)
    A[0].set_title(f"delta_C0 map, ch {CH}: target at pixel {xt}, main lobe +-{hw} px, alias images at +-{X:.0f} px")
    A[1].plot(n[ill], np.rad2deg(np.angle(ref_old[ill]*np.conj(s[ill]))), ".", ms=2, color="grey",
              label=f"legacy map (err {err(ref_old):.3f})")
    A[1].plot(n[ill], np.rad2deg(np.angle(ref[ill]*np.conj(s[ill]))), ".", ms=2, color="#D85A30",
              label=f"new map (err {err(ref):.3f})")
    A[1].axhline(expected, color="k", ls=":", lw=1, label=f"ideal {expected:.1f} deg")
    A[1].set_ylabel("arg(out) - arg(in) [deg]"); A[1].set_ylim(-180, 180); A[1].legend(fontsize=9)
    A[1].set_title("Phase SATA added to every sample of the line (err = ||out - ideal|| / ||ideal||)")
    A[2].plot(n, np.abs(s), color="#185FA5", lw=1, label="|input|")
    A[2].plot(n, np.abs(ref), color="#D85A30", lw=1, ls="--", label="|output|, new map")
    A[2].set_ylabel("amplitude"); A[2].set_xlabel("azimuth sample"); A[2].legend(fontsize=9)
    A[2].set_title("Amplitude")
    for a in A: a.grid(alpha=.3)
    A[2].set_xlim(idx_ill[0]-100, idx_ill[-1]+100)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
print(f"saved sata_correction_windows.pdf ({len(show)+1} pages)")
