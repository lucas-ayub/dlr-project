import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
d = np.load(os.path.join(OUT, "cache_check.npz"))
fa, fr, abw, prf, ve, pk = d["fa"], d["fr"], float(d["abw"]), float(d["prf"]), float(d["ve"]), int(d["pk"])
mag = d["mag_ref"]; m = mag > 0.02*mag.max()
Na = len(fa); t = (np.arange(Na) - pk)/prf
irf = {"mono": d["irf_mono"], "no": d["irf_no"], "sub": d["irf_sub"]}
nrm = np.abs(irf["mono"]).max()
db = lambda v: 20*np.log10(np.abs(v)/nrm + 1e-20)

fig, ax = plt.subplots(2, 2, figsize=(13, 9.5), dpi=150)
fig.suptitle(r"Reconstruction error budget | $\delta h$=240 m, $b_{xt}^{max}$=225 m, $N_{rx}$=4, "
             r"PRF=2000 Hz, $B_a$=1234.8 Hz", fontsize="medium")

im = ax[0,0].imshow(np.where(m, d["esr2d_no"], np.nan), origin="lower", aspect="auto",
                    cmap="magma", vmin=-30, vmax=10,
                    extent=[fr.min()/1e6, fr.max()/1e6, fa.min()/1e3, fa.max()/1e3])
fig.colorbar(im, ax=ax[0,0], label=r"$|S_{rec}-S_{ref}|/|S_{ref}|$ [dB]")
ax[0,0].axhline(abw/2e3, color="w", ls="-.", lw=.8); ax[0,0].axhline(-abw/2e3, color="w", ls="-.", lw=.8)
ax[0,0].set_xlabel("Range freq. [MHz]"); ax[0,0].set_ylabel("Azimuth freq. [kHz]")
ax[0,0].set_title("(a) error-to-signal, no SATA")

ax[0,1].plot(fa/1e3, d["esr_fa_no"], color="#C62828", lw=.9, label="no SATA (in-band +1.2 dB)")
ax[0,1].plot(fa/1e3, d["esr_fa_sub"], color="#2E7D32", lw=.9, label="SATA sub-band (in-band -6.4 dB)")
ax[0,1].axhline(0, color="k", lw=.8, ls=":")
ax[0,1].axvline(abw/2e3, color="k", ls="-.", lw=.8); ax[0,1].axvline(-abw/2e3, color="k", ls="-.", lw=.8)
ax[0,1].set_xlim(-1, 1); ax[0,1].set_ylim(-25, 15)
ax[0,1].set_xlabel("Azimuth frequency [kHz]"); ax[0,1].set_ylabel("error / signal [dB]")
ax[0,1].grid(alpha=.3); ax[0,1].legend(fontsize="small")
ax[0,1].set_title("(b) error-to-signal vs Doppler\n(above 0 dB the phase-difference plot is meaningless)")

for k, lab, c in (("mono","ref (mono)","#444"),("no","no SATA","#C62828"),("sub","SATA sub-band","#2E7D32")):
    ax[1,0].plot(t, db(irf[k]), lw=.7, color=c, label=lab)
ax[1,0].set_xlabel("Azimuth time $t_a$ [s]"); ax[1,0].set_ylabel("Impulse Response [dB]")
ax[1,0].set_ylim(-100, 3); ax[1,0].grid(alpha=.3); ax[1,0].legend(fontsize="small")
ax[1,0].set_title("(c) azimuth IRF with ambiguities")

w = 400; sl = slice(pk-w, pk+w)
for k, lab, c in (("mono","ref","#444"),("no","no SATA","#C62828"),("sub","SATA sub-band","#2E7D32")):
    ax[1,1].plot(t[sl]*1e3, db(irf[k])[sl], lw=1.0, color=c, label=lab)
ax[1,1].set_xlabel("Azimuth time $t_a$ [ms]"); ax[1,1].set_ylabel("[dB]")
ax[1,1].set_ylim(-45, 3); ax[1,1].set_xlim(-80, 80); ax[1,1].grid(alpha=.3)
ax[1,1].legend(fontsize="small"); ax[1,1].set_title("(d) main lobe -- zoom")
sec = ax[1,1].secondary_xaxis("top", functions=(lambda s: s*1e-3*ve, lambda x: x/ve*1e3))
sec.set_xlabel(r"ground azimuth $x = v_e t_a$ [m]")
fig.tight_layout()
o = os.path.join(OUT, "error_budget.png"); fig.savefig(o, dpi=150, bbox_inches="tight"); print("written", o)
