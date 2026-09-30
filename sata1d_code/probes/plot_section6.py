"""Figures for the detailed section 6: how the delta C0 map is built (current / look-up table) and read
(whole band / per sub-band), and the resulting IRFs. Scene: A, B, C at x = -2.6, 0, +2.6 km, h = 0, 500, 1000 m."""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "."); sys.path.insert(0, "probes")
import sata1d as S
import sata1d.sata as SS
from sata1d.subband import (subband_frequency_beam, subband_axis, build_delta_C0_subband_array, sata_1d_subband)
from sata1d.reconstruction import channel_spectrum, reconstruct_from_spectra
import run_geometry_sweep as G
import check_lut as CL

OUT = "section6_figs"; os.makedirs(OUT, exist_ok=True)
WL = 0.25
XS = [-2600.0, 0.0, 2600.0]; HS = [0.0, 500.0, 1000.0]; NAMES = ["A", "B", "C"]
BAND = ["#6A1B9A", "#1565C0", "#00897B", "#9E9D24"]
cfg = S.make_config(list(zip(XS, HS)), bxt_max=G.BXT_MAX, seed=G.BXT_SEED); tr = S.build_platform_tracks(cfg)
ds = cfg.system.vs / cfg.PRF_op
X = SS.sata_image_offsets(cfg, 0.0).max()
cells = np.arange(cfg.Na_ch); xkm = (cells - cfg.Na_ch / 2) * ds / 1e3
pix = [SS.az_pixel_of_scatterer(cfg, x, 0) for x in XS]

# ---------- table: delta C0 of each target and channel
with open(f"{OUT}/dc0_table.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["target", "x_m", "h_m", "y_m", "pixel_ch0"] + [f"dC0_ch{i}_mm" for i in range(4)])
    for n, p in zip(NAMES, cfg.scene.points[1:]):
        w.writerow([n, p[0], p[2], round(p[1], 2), SS.az_pixel_of_scatterer(cfg, p[0] - cfg.scene.x0, 0)] +
                   [round(1e3 * S.residual_C0(cfg, tr, p, i), 3) for i in range(4)])
print(open(f"{OUT}/dc0_table.csv").read())

# ---------- maps of channel 0
Mw = SS.build_delta_C0_array(cfg, tr, 0)
Mk = [build_delta_C0_subband_array(cfg, tr, 0, k) for k in range(4)]
L = CL.lut(cfg, tr, 0, XS, HS, step=1)
np.save(f"{OUT}/maps.npy", dict(Mw=Mw, Mk=Mk, L=L, xkm=xkm, pix=pix, X=X))

fig, axs = plt.subplots(6, 1, figsize=(15, 15), sharex=True); fig.subplots_adjust(hspace=0.55)
rows = [("mapa atual, SATA whole band: cópias em $m \\in \\{-1, 0, +1\\}$", Mw, "#1565C0")]
for k in range(4):
    fk = subband_frequency_beam(cfg, k)[0]
    ms = [int(round(o / X)) for o in SS.sata_image_offsets(cfg, fk)]
    rows.append((f"mapa atual, SATA per sub-band, sub-banda {k} ($f_k$ = {fk:+.0f} Hz): cópias em $m \\in \\{{{', '.join(f'{m:+d}' for m in ms)}\\}}$", Mk[k], BAND[k]))
rows.append(("look-up table (Nida): terreno em todas as células, história completa; a mesma para whole band e para as 4 sub-bandas", L, "#2E7D32"))
for ax, (title, M, col) in zip(axs, rows):
    ax.fill_between(xkm, 0, M * 1e3, color=col, alpha=0.25, step="mid"); ax.plot(xkm, M * 1e3, color=col, lw=1.2)
    for xx, n in zip(XS, NAMES):
        ax.axvline(xx / 1e3, color="k", lw=0.6, ls=":"); ax.text(xx / 1e3, 8, n, ha="center", fontsize=8)
    ax.set_ylim(-260, 20); ax.set_ylabel("$\\Delta C_0$ [mm]"); ax.grid(alpha=.3); ax.set_title(title, fontsize=9.5, loc="left")
axs[-1].set_xlabel("posição em azimute na linha do canal 0 [km]  (pontilhado: alvos A, B, C)")
axs[-1].set_xlim(-22, 22)
fig.savefig(f"{OUT}/s6_maps.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# ---------- the read of one bin: B seen at +400 Hz in a sub-aperture
prf, v, r = cfg.PRF_op, cfg.system.vs, cfg.scene.r0
Tsub, hop, Nzp = SS.subaperture_sizing(r, prf, v, WL, G.SATA_OSF)
a = lambda f: r * np.tan(np.arcsin(WL * f / (2 * v))) / v * prf
fB = 400.0
centre = pix[1] - a(fB)
start = int(round(centre - 0.5 * Tsub)); centre = start + 0.5 * Tsub
lineB = S.generate_channels(cfg, tr, ptgs=[cfg.scene.points[2]])[0]
win = SS.triangular_window(Tsub)
buf = np.zeros(Nzp, complex); buf[:Tsub] = lineB[start:start + Tsub] * win
E = np.abs(np.fft.fft(buf)); E /= E.max()
phi = np.fft.fftfreq(Nzp, 1 / prf)
need = S.residual_C0(cfg, tr, cfg.scene.points[2], 0) * 1e3
kerns = [("whole band", SS.subaperture_axis(prf, Nzp, v, WL, r, 0.0)[1], Mw, L)]
for k in range(4):
    kerns.append((f"sub-banda {k}", subband_axis(prf, Nzp, v, WL, r, subband_frequency_beam(cfg, k)[0])[1], Mk[k], L))
fig, axs = plt.subplots(5, 2, figsize=(16, 16), sharex=True); fig.subplots_adjust(hspace=0.55, wspace=0.12, top=0.93)
rows = []
b0 = int(np.argmax(E))
for n, (name, az, Mcur, Ll) in enumerate(kerns):
    pos = np.clip(np.round(az + centre).astype(int), 0, cfg.Na_ch - 1)
    for c, (Mname, M, col) in enumerate((("mapa atual", Mcur, "#1565C0" if n == 0 else BAND[n - 1]), ("look-up table", Ll, "#2E7D32"))):
        ax = axs[n, c]
        ax.plot(xkm, M * 1e3, color=col, lw=1.1)
        o = np.argsort(pos)
        ax2 = ax.twinx(); ax2.fill_between(xkm[pos[o]], 0, E[o], color="0.3", alpha=0.35, step="mid"); ax2.set_ylim(0, 3.2); ax2.set_yticks([])
        xb = xkm[pos[b0]]; val = M[pos[b0]] * 1e3
        ax.plot([xb], [val], "o", ms=9, mfc="none", mec="k", mew=1.8)
        ok = abs(val - need) < 3
        ax.set_title(f"{name} + {Mname}\npico de B lido em {xb:+.1f} km: $\\Delta C_0$ = {val:.1f} mm "
                     f"({'certo' if ok else 'errado'}; B precisa {need:.1f} mm)", fontsize=9.5, color="k" if ok else "#C62828")
        for xx, nm in zip(XS, NAMES):
            ax.axvline(xx / 1e3, color="k", lw=0.5, ls=":")
        ax.set_ylim(-260, 20); ax.grid(alpha=.3); ax.set_xlim(-16, 16)
        if c == 0: ax.set_ylabel("$\\Delta C_0$ [mm]")
        rows.append(dict(kernel=name, map=Mname, x_read_km=round(xb, 2), dC0_read_mm=round(val, 2), needed_mm=round(need, 2)))
for ax in axs[-1]: ax.set_xlabel("posição [km]; sombra cinza: energia de B na sub-abertura, onde cada kernel a coloca")
fig.suptitle(f"Uma sub-abertura em que B é visto com Doppler +{fB:.0f} Hz (o bin de pico em banda-base é {phi[b0]:+.0f} Hz): onde cada kernel lê o mapa", fontsize=11, y=0.975)
fig.savefig(f"{OUT}/s6_read.png", dpi=110, bbox_inches="tight"); plt.close(fig)
with open(f"{OUT}/read_table.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(open(f"{OUT}/read_table.csv").read())

# ---------- delta C0 applied to B as a function of its true Doppler, for every kernel and map
fs = np.linspace(-990, 990, 397); app = {}
for name, az, Mcur, Ll in kerns:
    for Mname, M in (("mapa atual", Mcur), ("look-up table", Ll)):
        vals = []
        for f in fs:
            c0 = pix[1] - a(f)
            fb = (f + prf / 2) % prf - prf / 2
            b = int(np.argmin(np.abs(phi - fb)))
            vals.append(M[int(np.clip(round(az[b] + c0), 0, cfg.Na_ch - 1))] * 1e3)
        app[(name, Mname)] = np.array(vals)
edges = [subband_frequency_beam(cfg, k) for k in range(4)]
fig, axs = plt.subplots(1, 2, figsize=(16, 5.2), sharey=True); fig.subplots_adjust(wspace=0.08)
cols = ["k"] + BAND; lss = ["-", (0, (5, 2)), (0, (3, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2))]
for ax, Mname in zip(axs, ("mapa atual", "look-up table")):
    for k in range(4):
        fk = edges[k][0]; ax.axvspan(fk - prf / 2, fk + prf / 2, color=BAND[k], alpha=0.08)
        ax.text(fk, 12, f"banda {k}", ha="center", fontsize=8.5, color=BAND[k])
    Bw = cfg.system.abw
    for sgn in (-1, 1): ax.axvspan(sgn * Bw / 2, sgn * 1000, color="0.55", alpha=0.25, lw=0)
    ax.text(-820, -245, "fora da banda\nprocessada", ha="center", fontsize=8.5, color="0.25")
    ax.text(820, -245, "fora da banda\nprocessada", ha="center", fontsize=8.5, color="0.25")
    ax.axhline(need, color="tab:red", lw=2.5, alpha=0.5, label=f"valor de B ({need:.1f} mm)")
    for (name, _, _, _), c, ls in zip(kerns, cols, lss):
        ax.plot(fs, app[(name, Mname)], color=c, ls=ls, lw=1.6, label=name)
    ax.set_xlabel("Doppler verdadeiro com que B é visto na sub-abertura [Hz]"); ax.grid(alpha=.3); ax.set_xlim(-1000, 1000)
    ax.set_title(Mname, fontsize=11)
axs[0].set_ylabel("$\\Delta C_0$ aplicado ao sinal de B [mm]"); axs[0].set_ylim(-260, 25)
h, l = axs[1].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=6, fontsize=9.5, bbox_to_anchor=(0.5, -0.07))
fig.savefig(f"{OUT}/s6_applied.png", dpi=120, bbox_inches="tight"); plt.close(fig)
np.save(f"{OUT}/applied.npy", dict(fs=fs, app=app))

# ---------- the four channels: current whole-band map and look-up table
MWc = [SS.build_delta_C0_array(cfg, tr, i) for i in range(4)]
LLc = [CL.lut(cfg, tr, i, XS, HS, step=1) for i in range(4)]
CHC = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"]
fig, axs = plt.subplots(2, 1, figsize=(15, 7.5), sharex=True); fig.subplots_adjust(hspace=0.3)
for ax, Ms, ttl in zip(axs, (MWc, LLc), ("mapa atual, whole band", "look-up table")):
    for i in range(4):
        ax.plot(xkm, Ms[i] * 1e3, color=CHC[i], lw=1.3, label=f"canal {i}")
    for xx, n in zip(XS, NAMES):
        ax.axvline(xx / 1e3, color="k", lw=0.6, ls=":"); ax.text(xx / 1e3, 8, n, ha="center", fontsize=8)
    ax.set_ylim(-260, 20); ax.grid(alpha=.3); ax.set_ylabel("$\\Delta C_0$ [mm]"); ax.set_title(ttl, fontsize=10, loc="left")
axs[0].legend(fontsize=9, ncol=2, loc="lower left"); axs[-1].set_xlim(-12, 12); axs[-1].set_xlabel("posição em azimute [km]")
fig.savefig(f"{OUT}/s6_channels.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# ---------- IRFs for every mapping
ch = S.generate_channels(cfg, tr)
sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
ref = F(S.generate_reference(cfg, tr)); pk = ref.max()
t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
kw = dict(rref=r, prf=prf, v=v, wl=WL, r=r, inverse=True, sata_osf=G.SATA_OSF)
def whole(Ms): return S.reconstruct(cfg, tr, np.array([S.sata_1d(ch[i], Ms[i], **kw) for i in range(4)]))
def sub(Mks):
    spec = []
    for k in range(4):
        fk = subband_frequency_beam(cfg, k)[0]
        C = [sata_1d_subband(ch[i], Mks[k][i], f_k=fk, **kw) for i in range(4)]
        spec.append([channel_spectrum(cfg, C[q]) for q in range(4)])
    return reconstruct_from_spectra(cfg, tr, spec)
MW = [SS.build_delta_C0_array(cfg, tr, i) for i in range(4)]
MK = [[build_delta_C0_subband_array(cfg, tr, i, k) for i in range(4)] for k in range(4)]
LL = [CL.lut(cfg, tr, i, XS, HS, step=1) for i in range(4)]
rec = {"no SATA": S.reconstruct(cfg, tr, ch), "mapa atual, whole band": whole(MW), "mapa atual, per sub-band": sub(MK),
       "look-up table, whole band": whole(LL), "look-up table, per sub-band": sub([LL] * 4)}
amb = {k: G.ambiguity_level(t, F(s), pk, np.array(XS), cfg.system.vs, G.ambiguity_times(cfg)) for k, s in rec.items()}
amb["Reference"] = G.ambiguity_level(t, ref, pk, np.array(XS), cfg.system.vs, G.ambiguity_times(cfg))
print(amb)
db = lambda e: 20 * np.log10(e / pk + 1e-12)
fig, axs = plt.subplots(3, 1, figsize=(15, 12.5), sharex=True); fig.subplots_adjust(hspace=0.35)
panels = [(("mapa atual, whole band", "tab:blue", (0, (5, 3))), ("mapa atual, per sub-band", "tab:orange", (0, (1, 2)))),
          (("look-up table, whole band", "#0D47A1", (0, (5, 3))), ("look-up table, per sub-band", "#E65100", (0, (1, 2)))),
          (("no SATA", "tab:red", (0, (5, 3))),)]
for ax, pan in zip(axs, panels):
    ax.plot(t, db(ref), color="tab:green", lw=2.2, label="Reference")
    for k, col, ls in pan:
        ax.plot(t, db(F(rec[k])), color=col, lw=1.5, ls=ls, label=f"{k}")
    ax.set_ylim(-90, 5); ax.set_xlim(t[0], t[-1]); ax.grid(alpha=.3); ax.set_ylabel("[dB]"); ax.legend(fontsize=9, loc="upper left")
    ax.set_title("highest ambiguity: " + " | ".join(f"{k} {amb[k]:.1f} dB" for k, _, _ in pan) + f" | reference {amb['Reference']:.1f} dB", fontsize=9.5)
axs[-1].set_xlabel("Azimuth time [s]")
fig.savefig(f"{OUT}/s6_irf.png", dpi=110, bbox_inches="tight"); plt.close(fig)
with open(f"{OUT}/irf_table.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["case", "highest_ambiguity_db"]); [w.writerow([k, round(v, 2)]) for k, v in amb.items()]

# ---------- the triangular window spectrum and the block half-width
Tsub, hop, Nzp = SS.subaperture_sizing(r, prf, v, WL, G.SATA_OSF)
w = SS.triangular_window(Tsub); N = 8192
W = np.abs(np.fft.fft(w, N)) ** 2; fw = np.fft.fftfreq(N, 1 / prf); o = np.argsort(fw); fw, W = fw[o], W[o] / W.max()
frac = lambda lim: W[np.abs(fw) <= lim].sum() / W.sum()
fig, axs = plt.subplots(1, 2, figsize=(15, 4.6)); fig.subplots_adjust(wspace=0.22)
ax = axs[0]; ax.plot(fw, 10 * np.log10(W + 1e-12), color="#1565C0", lw=1.5)
for k in (1, 2, 3):
    fn = k * 2 * prf / Tsub
    for sg in (-1, 1): ax.axvline(sg * fn, color="k", ls=":", lw=0.8)
    ax.text(fn + 3, -6 - 4 * k, f"{k}º nulo ({fn:.0f} Hz)", ha="left", fontsize=8.5)
ax.axvspan(-2 * 2 * prf / Tsub, 2 * 2 * prf / Tsub, color="#1565C0", alpha=0.08)
ax.set_xlim(-250, 250); ax.set_ylim(-62, 3); ax.text(0, -40, "sombreado: $\\pm h_w$ (até o 2º nulo)", ha="center", fontsize=8.5, color="#1565C0"); ax.grid(alpha=.3)
ax.set_xlabel("Doppler em relação ao alvo [Hz]"); ax.set_ylabel("|W(f)|² [dB]")
ax.set_title(f"espectro da janela triangular de {Tsub} amostras: energia dentro do 1º nulo {100*frac(50):.2f} %, do 2º nulo {100*frac(100):.3f} %", fontsize=9.5)
ax = axs[1]; ff = np.linspace(-1000, 1000, 801); ax.plot(ff, [a(x) for x in ff], color="#1565C0", lw=1.8, label="$a(f) = r_0 \\tan\\beta(f)/\\Delta x$")
ax.plot(ff, ff * a(1.0), color="k", ls="--", lw=1, label="reta $0{,}81\\,f$")
for fx, lab in ((100, "$h_w$ = 82 células"), (500, "$X$ = 405 células")):
    ax.plot([fx], [a(fx)], "o", color="#C62828"); ax.annotate(lab, (fx, a(fx)), (fx - 330, a(fx) + 60), fontsize=9, arrowprops=dict(arrowstyle="-", color="0.4"))
ax.set_xlabel("Doppler rotulado $\\hat f$ [Hz]"); ax.set_ylabel("deslocamento em células"); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="upper left")
ax.set_title("a régua $f \\to \\beta \\to$ célula", fontsize=10)
fig.savefig(f"{OUT}/s6_window_ruler.png", dpi=120, bbox_inches="tight"); plt.close(fig)
