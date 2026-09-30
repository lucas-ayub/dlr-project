"""
Test 3 -- one point target: the ambiguity pulses and how SATA lowers them.
No analytic approximation:
  * signal: EXACT range history (repo's getRawData1D);
  * dC0:    computed like the pipeline (residual_C0_3d = C0(true)-C0(flat)).
Three curves: no SATA / SATA (whole band) / SATA per sub-band.
For a single target the residual is angle-invariant, so whole-band and
per-sub-band coincide (the documented case-matrix result); both bury the
ambiguities that "no SATA" leaves. Produces test3_ambiguity.png.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sata_kernel import sata_1d_corrected
from pipeline_signal import getRawData1D
from pipeline_coeff import Params3D, build_tracks_3d, residual_C0_3d

wl, v, H, r0 = 0.25, 7500.0, 720000.0, 766206.0
theta_inc = np.deg2rad(20); Nrx = 4; PRFfull = 2000.0; PRFop = PRFfull/Nrx
rref = r0; h = 240.0; Nfull = 8192
Ka = 2*v**2/(wl*r0)                                  # exact Doppler rate (locates ambiguities)

p = Params3D(v=v, H=H, wl=wl, r0=r0, theta_inc=theta_inc, Nrx=Nrx, prf_full=PRFfull,
             bxt=(0.0, 100.0, 200.0, 300.0), Nfull=Nfull, band_frac=0.7)
tracks = build_tracks_3d(p); ptx, prx, vtx, vrx = tracks

# target: azimuth 0, height h, ground range so that slant range = r0
y = np.sqrt(r0**2 - (H - h)**2); ptg = np.array([0.0, y, h])

# --- dC0 the pipeline way: C0(true) - C0(flat at same range), real fit ---
dC0 = np.array([residual_C0_3d(p, tracks, ptg, i) for i in range(Nrx)])
print("dC0 (exact fit) [cm]:", np.round(dC0*100, 3))

# --- ideal full-PRF signal from the EXACT range history ---
s_full = getRawData1D(ptg[None, :], ptx, ptx, vtx, vtx, p.ta, 0.0, 0.0,
                      p.theta_tx, p.theta_tx, wl, PRFfull).astype(complex)
Lc = Nfull//Nrx

def channels(dC0v):
    ch = np.empty((Nrx, Lc), complex)
    for i in range(Nrx):
        ch[i] = s_full[i::Nrx][:Lc]*np.exp(-1j*2*np.pi/wl*dC0v[i])
    return ch

def reconstruct(ch):
    full = np.empty(Nrx*Lc, complex)
    for i in range(Nrx):
        full[i::Nrx] = ch[i]
    return full

def correct_whole(ch):
    out = ch.copy()
    for i in range(Nrx):
        out[i] = sata_1d_corrected(ch[i].copy(), np.full(Lc, dC0[i]),
                                   rref, PRFop, 1, v, 0.0, wl, r0, inverse=True)
    return out

def correct_subband(ch):
    """Split each channel band into Nrx sub-bands, correct each with its beta_k."""
    out = ch.copy(); edges = np.linspace(-PRFop/2, PRFop/2, Nrx+1)
    for i in range(Nrx):
        L = Lc; spec = np.fft.fftshift(np.fft.fft(ch[i])); fb = (np.arange(L)-L//2)*(PRFop/L)
        acc = np.zeros(L, complex)
        for k in range(Nrx):
            band = (fb >= edges[k]) & (fb < edges[k+1])
            sl = np.zeros(L, complex); sl[band] = spec[band]
            seg = np.fft.ifft(np.fft.ifftshift(sl))
            fk = 0.5*(edges[k]+edges[k+1]); beta_k = np.arcsin(np.clip(wl*fk/(2*v), -1, 1))
            acc += sata_1d_corrected(seg, np.full(L, dC0[i]), rref, PRFop, 1, v,
                                     beta_k, wl, r0, inverse=True)
        out[i] = acc
    return out

def irf_db(sig):
    n = min(len(sig), len(s_full))
    c = np.fft.fftshift(np.fft.ifft(np.fft.fft(sig[:n])*np.conj(np.fft.fft(s_full[:n]))))
    a = np.abs(c); a /= a.max(); return 20*np.log10(a+1e-12)

def amb_db(db):
    N = len(db); pk = N//2; off = int(round(PRFop/Ka*PRFfull)); w = 6
    seg = db[pk+off-w:pk+off+w]; return seg.max() if len(seg) else -99

ch0 = channels(dC0)
dbn = irf_db(reconstruct(ch0))
dbw = irf_db(reconstruct(correct_whole(ch0)))
dbs = irf_db(reconstruct(correct_subband(ch0)))
print(f"no SATA        : {amb_db(dbn):.1f} dB")
print(f"SATA           : {amb_db(dbw):.1f} dB")
print(f"SATA per sub-b : {amb_db(dbs):.1f} dB")

lag = (np.arange(len(dbn)) - len(dbn)//2)/PRFfull
RD, AZ, GR = "#E24B4A", "#185FA5", "#1D9E75"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "figure.dpi": 130})
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(lag, dbn, color=RD, lw=1.2, label=f"no SATA (amb {amb_db(dbn):.0f} dB)")
ax.plot(lag, dbw, color=AZ, lw=1.4, label=f"SATA (amb {amb_db(dbw):.0f} dB)")
ax.plot(lag, dbs, color=GR, lw=1.0, ls="--", label=f"SATA per sub-band (amb {amb_db(dbs):.0f} dB)")
ax.set_xlim(lag.min(), lag.max()); ax.set_ylim(-70, 3)   # full azimuth scene (no zoom)
ax.set_xlabel("azimuth time [s]"); ax.set_ylabel("[dB]")
ax.set_title("Test 3 - one point target: ambiguity pulses and how SATA lowers them")
for mm in (1, 2):
    ax.axvline(+mm*PRFop/Ka, color="grey", ls=":", lw=.6)
    ax.axvline(-mm*PRFop/Ka, color="grey", ls=":", lw=.6)
off = int(round(PRFop/Ka*PRFfull))
ax.annotate("ambiguity pulses\nat +- PRF_op/Ka", xy=(PRFop/Ka, dbn[len(dbn)//2+off]),
            xytext=(0.95, -18), fontsize=10, color=RD, arrowprops=dict(arrowstyle="->", color=RD))
ax.legend(fontsize=10, loc="upper right"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("test3_ambiguity.png"); plt.close()
print("saved test3_ambiguity.png")
