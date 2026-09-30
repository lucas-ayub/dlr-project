"""2-D view of the delta C0 that the SATA applies: azimuth position of the energy x true Doppler f,
for channel 0, scene A/B/C. 'joined' = output band k uses the channels corrected by the sub-band-k kernel."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
sys.path.insert(0, "."); sys.path.insert(0, "probes")
import sata1d as S, sata1d.sata as SS
from sata1d.subband import subband_frequency_beam, subband_axis, build_delta_C0_subband_array
import run_geometry_sweep as G, check_lut as CL

OUT = "section6_figs"; os.makedirs(OUT, exist_ok=True)
WL = 0.25; XS = [-2600.0, 0.0, 2600.0]; HS = [0.0, 500.0, 1000.0]; NAMES = ["A", "B", "C"]
cfg = S.make_config(list(zip(XS, HS)), bxt_max=G.BXT_MAX, seed=G.BXT_SEED); tr = S.build_platform_tracks(cfg)
prf, v, r = cfg.PRF_op, cfg.system.vs, cfg.scene.r0; ds = v / prf; Bw = cfg.system.abw
Tsub, hop, Nzp = SS.subaperture_sizing(r, prf, v, WL, G.SATA_OSF)
a = lambda f: r * np.tan(np.arcsin(WL * f / (2 * v))) / v * prf
phi = np.fft.fftfreq(Nzp, 1 / prf)
pix = np.array([SS.az_pixel_of_scatterer(cfg, x, 0) for x in XS])
Mw = SS.build_delta_C0_array(cfg, tr, 0); Mk = [build_delta_C0_subband_array(cfg, tr, 0, k) for k in range(4)]
L = CL.lut(cfg, tr, 0, XS, HS, step=1)
kern = {"whole": SS.subaperture_axis(prf, Nzp, v, WL, r, 0.0)[1]}
for k in range(4): kern[k] = subband_axis(prf, Nzp, v, WL, r, subband_frequency_beam(cfg, k)[0])[1]
fk = [subband_frequency_beam(cfg, k)[0] for k in range(4)]

cells = np.arange(900, 2200); xkm = (cells - cfg.Na_ch / 2) * ds / 1e3
fs = np.linspace(-1000, 1000, 401)
fb = (fs + prf / 2) % prf - prf / 2
bins = np.array([int(np.argmin(np.abs(phi - f))) for f in fb])

def applied(M, az):
    """[len(fs), len(cells)]: value applied to energy from cell n seen at true Doppler f."""
    out = np.zeros((len(fs), len(cells)))
    for i, (f, b) in enumerate(zip(fs, bins)):
        read = np.clip(np.round(cells - a(f) + az[b]).astype(int), 0, cfg.Na_ch - 1)
        out[i] = M[read]
    return out * 1e3

def joined(Ms):
    out = np.zeros((len(fs), len(cells)))
    for k in range(4):
        sel = (fs >= fk[k] - prf / 2) & (fs < fk[k] + prf / 2)
        out[sel] = applied(Ms[k], kern[k])[sel]
    return out

A = {("mapa atual", "whole band"): applied(Mw, kern["whole"]), ("mapa atual", "per sub-band, juntas"): joined(Mk),
     ("look-up table", "whole band"): applied(L, kern["whole"]), ("look-up table", "per sub-band, juntas"): joined([L] * 4)}
K = {("mapa atual", k): applied(Mk[k], kern[k]) for k in range(4)}
K.update({("look-up table", k): applied(L, kern[k]) for k in range(4)})
need = np.tile(L[cells] * 1e3, (len(fs), 1))       # terrain of the DEM (what the LUT assumes)
np.save(f"{OUT}/maps2d.npy", dict(fs=fs, xkm=xkm, A=A, K=K, need=need))

norm = Normalize(-235, 5); cmap = "Spectral"
def panel(ax, Z, title):
    im = ax.imshow(Z, aspect="auto", origin="lower", extent=[xkm[0], xkm[-1], fs[0], fs[-1]], cmap=cmap, norm=norm, interpolation="nearest")
    for sg in (-1, 1): ax.axhspan(sg * Bw / 2, sg * 1000, color="0.35", alpha=0.55, lw=0)
    for k in range(4): ax.axhline(fk[k] - prf / 2, color="k", lw=0.5, ls="--")
    for x, n in zip(XS, NAMES): ax.axvline(x / 1e3, color="k", lw=0.7, ls=":"); ax.text(x / 1e3, 930, n, ha="center", fontsize=8)
    ax.set_title(title, fontsize=9.5); ax.set_xlim(-9.5, 9.5); return im

# ---- figure 1: needed + whole + joined, current vs LUT
fig, axs = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, sharey=True); fig.subplots_adjust(wspace=0.12, hspace=0.25, right=0.9)
for rr, mn in enumerate(("mapa atual", "look-up table")):
    if rr == 0:
        Zn = np.full_like(need, np.nan)
        for p, h in zip(pix, HS):
            sel = np.abs(cells - p) <= 2; Zn[:, sel] = 1e3 * S.residual_C0(cfg, tr, cfg.scene.points[1 + list(pix).index(p)], 0)
        panel(axs[rr, 0], Zn, "necessário (só onde há alvo: A, B, C)")
    else:
        panel(axs[rr, 0], need, "necessário se o terreno fosse o DEM da look-up table")
    for cc, kn in enumerate(("whole band", "per sub-band, juntas")):
        im = panel(axs[rr, cc + 1], A[(mn, kn)], f"aplicado: {mn}, {kn}")
    axs[rr, 0].set_ylabel("Doppler verdadeiro da energia [Hz]")
for ax in axs[-1]: ax.set_xlabel("posição de origem da energia em azimute [km]")
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("$\\Delta C_0$ aplicado [mm]")
fig.text(0.5, 0.94, "canal 0 — faixas cinza: fora da banda processada; tracejado: limites das sub-bandas de saída", ha="center", fontsize=9.5)
fig.savefig(f"{OUT}/s6_2d_applied.png", dpi=115, bbox_inches="tight"); plt.close(fig)

# ---- figure 2: the four sub-band kernels, before joining
fig, axs = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True); fig.subplots_adjust(wspace=0.1, hspace=0.25, right=0.9)
for rr, mn in enumerate(("mapa atual", "look-up table")):
    for k in range(4):
        im = panel(axs[rr, k], K[(mn, k)], f"{mn}, kernel da sub-banda {k}")
        axs[rr, k].add_patch(plt.Rectangle((-9.5, fk[k] - prf / 2), 19, prf, fill=False, ec="k", lw=2.2))
    axs[rr, 0].set_ylabel("Doppler verdadeiro da energia [Hz]")
for ax in axs[-1]: ax.set_xlabel("posição de origem da energia [km]")
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("$\\Delta C_0$ aplicado [mm]")
fig.text(0.5, 0.94, "canal 0 — cada kernel corrige o canal inteiro; a caixa preta é a banda de saída que ele alimenta (a parte que entra na figura 'juntas')", ha="center", fontsize=9.5)
fig.savefig(f"{OUT}/s6_2d_kernels.png", dpi=115, bbox_inches="tight"); plt.close(fig)

# ---- figure 3: phase error left on the energy of A, B, C by each kernel (all parcels matter)
fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharey=True, sharex=True); fig.subplots_adjust(wspace=0.08, hspace=0.25)
KC = ["#6A1B9A", "#1565C0", "#00897B", "#9E9D24"]; LS = [(0, (5, 2)), (0, (3, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2))]
for rr, mn in enumerate(("mapa atual", "look-up table")):
    for j, (p, n) in enumerate(zip(pix, NAMES)):
        ax = axs[rr, j]; col = int(np.argmin(np.abs(cells - p)))
        needj = 1e3 * S.residual_C0(cfg, tr, cfg.scene.points[1 + j], 0)
        ax.plot(fs, (A[(mn, "whole band")][:, col] - needj) * 360 / (WL * 1e3), color="k", lw=2, label="kernel whole band")
        for k in range(4):
            ax.plot(fs, (K[(mn, k)][:, col] - needj) * 360 / (WL * 1e3), color=KC[k], ls=LS[k], lw=1.7, label=f"kernel da sub-banda {k}")
        for k in range(4): ax.axvspan(fk[k] - prf / 2, fk[k] + prf / 2, color=KC[k], alpha=0.06); ax.text(fk[k], 350, f"banda {k}", ha="center", fontsize=8, color=KC[k])
        for sg in (-1, 1): ax.axvspan(sg * Bw / 2, sg * 1000, color="0.55", alpha=0.3, lw=0)
        ax.set_title(f"{mn} — alvo {n} (precisa {needj:.1f} mm)", fontsize=10); ax.grid(alpha=.3); ax.set_xlim(-1000, 1000)
    axs[rr, 0].set_ylabel("erro de fase que sobra [graus]")
for ax in axs[-1]: ax.set_xlabel("Doppler verdadeiro da energia do alvo [Hz]")
axs[0, 0].set_ylim(-200, 380)
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=5, fontsize=9.5, bbox_to_anchor=(0.5, -0.04))
fig.savefig(f"{OUT}/s6_2d_error.png", dpi=115, bbox_inches="tight"); plt.close(fig)
print("ok")
