"""
Two point targets P1, P2 with overlapping synthetic apertures (Lsa1, Lsa2).
In every SATA window where both are illuminated the STFT shows TWO peaks;
through the ruler f_a <-> theta <-> x each peak lands on its own target's
position, and the delta_C0 map (build_delta_C0_array, mode="footprint")
holds dC0_1 around x1 and dC0_2 around x2 -- so each peak is corrected with
its own value.

Everything from the repo's sar_recon (make_topo_config, getRawData1D,
build_delta_C0_array, sata_1d). Channel 3.
One page per window: columns theta / f_a / x = posaux; rows
  1) |STFT| -- dots = bins where P1 (orange) / P2 (green) appear (> -30 dB);
     filled = the bin reads THAT target's dC0, open = it does not
  2) delta_C0[posaux] each bin reads (dotted: dC0_1, dC0_2)
  3) rotation applied to the bins of each target (dotted: ideal for each)
Last page: the map and the per-target result vs the ideal correction
(new footprint map vs legacy hold map).
Produces sata_two_targets_windows.pdf.
"""
import os, sys, tempfile, dataclasses
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sar_reconstruction"))
from sar_recon.config import make_topo_config
from sar_recon.geometry import build_platform_tracks
from sar_recon.signal_model import getRawData1D
from sar_recon.sata import (sata_1d, residual_C0, build_delta_C0_array, az_pixel_of_scatterer)

CH, OSF, INVERSE, DB_MIN = 3, 4, True, -30.0
TARGETS = ((-1500.0, 240.0), (1500.0, -120.0))            # (dx [m], height [m])
COL = ("#D85A30", "#1D9E75")

# ---- scene: two targets, each at the same slant range as the centre --------
cfg = make_topo_config(4, tempfile.mkdtemp(), "single", dxt=100.0)
sc = cfg.scene
off = tuple((dx, np.sqrt(sc.r0**2 - (sc.H - h)**2) - sc.y0, h) for dx, h in TARGETS)
cfg = dataclasses.replace(cfg, scene=dataclasses.replace(sc, extra_offsets=off))
tr = build_platform_tracks(cfg)
wl, v, r, P = cfg.system.wl, cfg.system.vs, cfg.scene.r0, cfg.PRF_op
pts = cfg.scene.points[1:]
sig = [getRawData1D(p[None], tr.ptx, tr.prx[CH], tr.vtx, tr.vrx[CH], cfg.ta, cfg.sq_tx, cfg.sq_tx,
                    cfg.theta_tx, cfg.theta_tx, wl, cfg.prf)[::cfg.Nrx].astype(complex) for p in pts]
s = sig[0] + sig[1]
dimx = len(s)
dC0 = [residual_C0(cfg, tr, p, CH) for p in pts]
xt = [az_pixel_of_scatterer(cfg, dx, CH) for dx, _ in TARGETS]
dmap = build_delta_C0_array(cfg, tr, CH)                     # footprint (new default)
dmap_old = build_delta_C0_array(cfg, tr, CH, mode="hold")

# ---- sata_1d STEP 1 / STEP 2 replayed; the STFT of each target kept apart ---
deltax = np.sqrt(wl*r/2.0)
Tsubeff = int(np.round(deltax*P/v*0.5)*2); Tsub = Tsubeff//2
Nzp = int(2**np.ceil(np.log2(Tsubeff))*OSF)
win = np.concatenate((np.arange(Tsub)/(Tsub-1), (np.arange(Tsub)/(Tsub-1))[::-1]))
fsub = np.arange(Nzp)*P/Nzp - P*0.5; fsub[0] = P*0.5; fsub = np.roll(fsub, Nzp//2)
betasub = np.arcsin(np.clip(wl*fsub/(2*v), -1, 1))
azpos = r*np.tan(betasub)/v*P
fft_w = lambda x, st: np.fft.fft(np.concatenate([x[st:st+Tsubeff]*win[:len(x[st:st+Tsubeff])],
                                                 np.zeros(Nzp-len(x[st:st+Tsubeff]), complex)]))
log = []
for start in range(0, dimx, Tsub):
    if dimx - start < 2: break
    spec = fft_w(s, start); parts = [fft_w(x, start) for x in sig]
    centre = start + 0.5*Tsubeff
    posaux = np.clip(np.round(azpos + centre).astype(int), 0, dimx-1)
    ph = -2*np.pi/wl*dmap[posaux]
    rotf = np.exp(-1j*ph) if INVERSE else np.exp(1j*ph)
    log.append(dict(start=start, centre=centre, posaux=posaux, ph=ph, spec=spec, parts=parts, rotf=rotf))

kw = dict(rref=r, prf=P, v=v, wl=wl, r=r, inverse=INVERSE, sata_osf=OSF, verbose=False)
def per_target_err(m):
    """SATA is linear: SATA(s1+s2) = SATA(s1) + SATA(s2); score each target's part."""
    out = []
    for x, d in zip(sig, dC0):
        o = sata_1d(x, m, **kw); ideal = x*np.exp(1j*2*np.pi/wl*d); ill = np.abs(x) > 0
        out.append((o, np.linalg.norm((o-ideal)[ill])/np.linalg.norm(ideal[ill])))
    return out
new, old = per_target_err(dmap), per_target_err(dmap_old)
for i in range(2):
    print(f"P{i+1}: dC0 {dC0[i]*100:.2f} cm at pixel {xt[i]} -> err new {new[i][1]:.3f}, legacy {old[i][1]:.3f}")

# ---- bins where each target appears -----------------------------------------
o = np.argsort(fsub)
gmax = max(np.abs(w["spec"]).max() for w in log)
for w in log:
    a = [np.abs(p) for p in w["parts"]]
    w["db"] = 20*np.log10(np.abs(w["spec"])/gmax + 1e-12)
    w["own"] = [(20*np.log10(a[i]/gmax + 1e-12) > DB_MIN) & (a[i] >= a[1-i]) for i in range(2)]
    w["ok"] = [w["own"][i] & np.isclose(dmap[w["posaux"]], dC0[i]) for i in range(2)]
tot = [sum(w["own"][i].sum() for w in log) for i in range(2)]
okk = [sum(w["ok"][i].sum() for w in log) for i in range(2)]
for i in range(2):
    print(f"P{i+1}: bins where it appears {tot[i]}, reading its own dC0 {okk[i]} ({100*okk[i]/tot[i]:.1f}%)")

ill_any = np.flatnonzero(np.abs(sig[0]) + np.abs(sig[1]) > 0)
both = [np.flatnonzero(np.abs(x) > 0) for x in sig]
ov = (max(both[0][0], both[1][0]), min(both[0][-1], both[1][-1]))
show = [k for k, w in enumerate(log) if ill_any[0]-30 <= w["centre"] <= ill_any[-1]+30]
exp_deg = [np.rad2deg(np.angle(np.exp(1j*2*np.pi/wl*d))) for d in dC0]
segs = np.flatnonzero(np.diff(np.r_[0, (dmap != 0).astype(int), 0])).reshape(-1, 2)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "figure.dpi": 90})
with PdfPages("sata_two_targets_windows.pdf") as pdf:
    for k in show:
        w = log[k]
        fig, A = plt.subplots(3, 3, figsize=(14, 10), sharey="row")
        for j, (xx, lab, lim) in enumerate([
                (np.rad2deg(betasub[o]), r"look angle $\theta$ [deg]", (-.25, .25)),
                (fsub[o], r"azimuth frequency $f_a$ [Hz]", (-260, 260)),
                (w["posaux"][o], r"azimuth cell $x$ = posaux [samples]", (xt[0]-600, xt[1]+600))]):
            A[0, j].plot(xx, w["db"][o], "-", lw=.8, color="#185FA5")
            for i in range(2):
                own, ok = w["own"][i][o], w["ok"][i][o]
                A[0, j].plot(xx[own & ok], w["db"][o][own & ok], "o", ms=3.5, color=COL[i],
                             label=f"P{i+1}, reads its own dC0")
                A[0, j].plot(xx[own & ~ok], w["db"][o][own & ~ok], "o", ms=4, mfc="none", color=COL[i],
                             label=f"P{i+1}, does NOT")
                rot = np.rad2deg(np.angle(w["rotf"]))[o]
                A[2, j].plot(xx[own], rot[own], "o", ms=3.5, color=COL[i])
                A[1, j].axhline(dC0[i]*100, color=COL[i], ls=":", lw=1)
                A[2, j].axhline(exp_deg[i], color=COL[i], ls=":", lw=1)
            A[1, j].plot(xx, dmap[w["posaux"]][o]*100, "o", ms=2.5, color="k")
            for a in A[:, j]:
                a.set_xlim(*lim); a.grid(alpha=.3)
            A[2, j].set_xlabel(lab)
        for a in A[:, 2]:
            for lo, hi in segs:
                a.axvspan(lo, hi-1, color="grey", alpha=.10)
            for i in range(2):
                a.axvline(xt[i], color=COL[i], ls=":", lw=1)
            a.axvline(w["centre"], color="#185FA5", ls="--", lw=.8)
        A[0, 0].legend(fontsize=7.5, loc="lower left", ncol=2)
        A[0, 0].set_ylim(-60, 3); A[2, 0].set_ylim(-180, 180)
        lo_, hi_ = min(dC0+[0])*100, max(dC0+[0])*100
        A[1, 0].set_ylim(lo_-3, hi_+3)
        A[0, 0].set_ylabel("|STFT| [dB]"); A[1, 0].set_ylabel(r"$\Delta C_0$[posaux] [cm]")
        A[2, 0].set_ylabel("rotation applied\nto the bin [deg]")
        A[0, 1].set_title("1) where P1 / P2 appear in the STFT of this window")
        A[1, 1].set_title(r"2) $\Delta C_0$ each bin reads (grey shade: map $\neq$ 0; dotted: $\Delta C_{0,1}$, $\Delta C_{0,2}$)")
        A[2, 1].set_title("3) rotation applied to each target's bins (dotted: ideal for each)")
        n1, n2 = int(w["own"][0].sum()), int(w["own"][1].sum())
        tag = "BOTH illuminated" if ov[0] <= w["centre"] <= ov[1] else "one target"
        fig.suptitle(f"Window {k+1}/{len(log)}, centre {w['centre']:.0f} ({tag})  |  P1 bins {n1} "
                     f"(own dC0: {int(w['ok'][0].sum())}), P2 bins {n2} (own dC0: {int(w['ok'][1].sum())})",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    n = np.arange(dimx)
    fig, A = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    A[0].plot(n, dmap_old*100, color="grey", lw=1.2, label="legacy (hold): interpolated, ends held")
    A[0].plot(n, dmap*100, color="k", lw=1.4, label="new (footprint): each target's value where it appears")
    for i in range(2):
        A[0].axvline(xt[i], color=COL[i], ls=":", lw=1)
        A[0].axvspan(both[i][0], both[i][-1], color=COL[i], alpha=.06)
    A[0].set_ylabel(r"$\Delta C_0$ map [cm]"); A[0].legend(fontsize=9)
    A[0].set_title("delta_C0 map (shaded: synthetic aperture of each target, Lsa1 / Lsa2)")
    for i in range(2):
        ill = np.abs(sig[i]) > 0
        for res, ls, name in ((new, "-", "new"), (old, ":", "legacy")):
            A[1+i].plot(n[ill], np.rad2deg(np.angle(res[i][0][ill]*np.conj(sig[i][ill]))), ls,
                        lw=1.2, color=COL[i] if name == "new" else "grey",
                        label=f"{name} map (err {res[i][1]:.3f})")
        A[1+i].axhline(exp_deg[i], color="k", ls=":", lw=1, label=f"ideal {exp_deg[i]:.1f} deg")
        A[1+i].set_ylim(-180, 180); A[1+i].legend(fontsize=9)
        A[1+i].set_ylabel("arg(out) - arg(in) [deg]")
        A[1+i].set_title(f"P{i+1}: phase SATA added to its samples (dC0 = {dC0[i]*100:.2f} cm)")
    A[2].set_xlabel("azimuth sample")
    for a in A: a.grid(alpha=.3)
    A[2].set_xlim(ill_any[0]-100, ill_any[-1]+100)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
print(f"saved sata_two_targets_windows.pdf ({len(show)+1} pages)")
