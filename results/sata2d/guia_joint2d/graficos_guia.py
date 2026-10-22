"""Figures of the guides of the 2-D joint, from dados_*.npz / .json (dados_guia.py) and the numbers of the sweep."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight", "axes.formatter.use_locale": False})
CH = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
REF, JNT, NON = "tab:green", "tab:purple", "tab:red"
vg = lambda s: s.replace(".", ",")
save = lambda fig, name: (fig.savefig(os.path.join(HERE, "figs", name)), plt.close(fig))
nums = {}

# ------------------------------------------------------------------------------------------------ 1) range cell migration
z = np.load(os.path.join(HERE, "dados_rcm.npz"))
r, wl, vs = float(z["r"]), float(z["wl"]), float(z["vs"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(wspace=0.26))
T = z["time"] / z["time"].max()
a1.pcolormesh(z["ta"], z["dr"], T.T, cmap="magma", vmin=0, vmax=1, shading="auto", rasterized=True)
tt = np.linspace(-1.08, 1.08, 400)
a1.plot(tt, np.sqrt(r ** 2 + (vs * tt) ** 2) - r, color="c", lw=1.3, ls=(0, (4, 3)))
a1.set_xlim(-1.3, 1.3); a1.set_ylim(-45, 110); a1.set_xlabel("tempo de azimute $t$ [s]"); a1.set_ylabel("distância menos a de maior aproximação [m]")
a1.set_title("Eco de um alvo, comprimido em range: a curva $R(t)-r$", fontsize=10.5)
D_ = z["rd"] / z["rd"].max()
im = a2.pcolormesh(z["fa"], z["dr"], D_.T, cmap="magma", vmin=0, vmax=1, shading="auto", rasterized=True)
ff = np.linspace(-640, 640, 400)
a2.plot(ff, r * (1 / np.sqrt(1 - (wl * ff / (2 * vs)) ** 2) - 1), color="c", lw=1.3, ls=(0, (4, 3)))
a2.set_xlim(-800, 800); a2.set_ylim(-45, 110); a2.set_xlabel("frequência Doppler $f$ [Hz]")
a2.set_title("O mesmo eco em range-Doppler: a curva $\\Delta R(f)$", fontsize=10.5)
for a in (a1, a2):
    a.grid(False)
cb = fig.colorbar(im, ax=[a1, a2], pad=0.015, fraction=0.03); cb.set_label("amplitude (1 = máximo)")
nums["rcm_640"] = float(r * (1 / np.sqrt(1 - (wl * 640.7 / (2 * vs)) ** 2) - 1))
save(fig, "rcm.pdf")

# ------------------------------------------------------------------------------------------------ 2) one target on flat terrain
z = np.load(os.path.join(HERE, "dados_ponto.npz"))
lev = json.loads(str(z["levels"])); nums["ponto"] = lev; nums["ponto_shift_MHz"] = [float(v) / 1e6 for v in z["shift"]]
nums["ponto_unexplained_dB"] = float(20 * np.log10(z["unexplained"])); nums["ponto_iterations"] = int(z["iterations"])
fr, err, sig, shift, bxt = z["fr"], z["err"], z["sig"], z["shift"], z["bxt"]
o = np.argsort(fr)
fig, ax = plt.subplots(figsize=(11, 3.9))
rel = 10 * np.log10(err[o] / np.maximum(sig[o], 1e-12) + 1e-12)
inb = np.abs(fr[o]) <= 2.5e6
ax.plot(fr[o][inb] / 1e6, rel[inb], color="k", lw=1.6, label="erro da saída (faixa inteira) em relação ao sinal da referência, por frequência de range")
for i in range(1, len(bxt)):
    e = 2.5 - shift[i] / 1e6
    ax.axvline(e, color=CH[i], lw=1.5, ls="--", label=vg(f"canal {i} ($b_{{xt}}$ = {bxt[i]:g} m): vê a imagem até {e:.2f} MHz"))
ax.axvspan(2.5 - shift.max() / 1e6, 2.5, color="0.86", zorder=0, label="frequências que nem todos os canais veem")
ax.set_xlim(-2.6, 2.6); ax.set_ylim(-55, 2)
ax.set_xlabel("frequência de range da imagem [MHz]  (o canal 0, o transmissor, vê de $-$2,5 a +2,5 MHz)"); ax.set_ylabel("erro / sinal [dB]")
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
save(fig, "erro_fr.pdf")
nums["erro_fr_medio_comum_dB"] = float(10 * np.log10(err[(fr > -2.4e6) & (fr < 2.5e6 - shift.max() - 1e5)].sum() / sig[(fr > -2.4e6) & (fr < 2.5e6 - shift.max() - 1e5)].sum()))
nums["erro_fr_topo_dB"] = float(10 * np.log10(err[(fr > 2.5e6 - np.sort(shift)[-2]) & (fr < 2.5e6)].sum() / sig[(fr > 2.5e6 - np.sort(shift)[-2]) & (fr < 2.5e6)].sum()))

Na = len(z["env_ref"]); t = (np.arange(Na) - Na // 2) / 2000.0
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.7), gridspec_kw=dict(wspace=0.2), sharey=True)
pk = z["env_ref_whole"].max()
a1.plot(t, 20 * np.log10(z["env_ref_whole"] / pk + 1e-12), color=REF, lw=2.0, label="Reference")
a1.plot(t, 20 * np.log10(z["env_joint_whole"] / pk + 1e-12), color=JNT, lw=1.1, label="joint, faixa de range inteira")
a1.set_title(vg(f"Faixa inteira: maior ambiguidade {lev['joint_whole']:.1f} dB (referência {lev['ref_whole']:.1f} dB)"), fontsize=10.5)
pk = z["env_ref"].max()
a2.plot(t, 20 * np.log10(z["env_ref"] / pk + 1e-12), color=REF, lw=2.0, label="Reference, na mesma faixa")
a2.plot(t, 20 * np.log10(z["env_joint"] / pk + 1e-12), color=JNT, lw=1.1, label="joint, faixa comum aos canais")
a2.set_title(vg(f"Faixa comum: maior ambiguidade {lev['joint']:.1f} dB (referência {lev['ref']:.1f} dB)"), fontsize=10.5)
for a in (a1, a2):
    a.set_xlim(t[0], t[-1]); a.set_ylim(-95, 5); a.set_xlabel("tempo de azimute [s]"); a.legend(fontsize=8.5, loc="upper left")
a1.set_ylabel("[dB]")
save(fig, "irf_ponto.pdf")
hist_p = z["hist"]

# ------------------------------------------------------------------------------------------------ 3) the mountain
z = np.load(os.path.join(HERE, "dados_relevo.npz"))
lev = json.loads(str(z["levels"])); nums["relevo"] = lev
nums["relevo_unexplained_dB"] = float(20 * np.log10(z["unexplained"])); nums["relevo_iterations"] = int(z["iterations"])
x, r, r0, xy, hs = z["x"], z["r"], float(z["r0"]), z["xy"], z["hs"]
rt = np.sqrt((float(z["y0"]) + xy[:, 1]) ** 2 + (float(z["H"]) - hs) ** 2) - r0
nums["relevo_kept_median"] = float(np.median(z["kept"])); nums["relevo_kept_min"] = float(z["kept"].min())
nums["relevo_shift_max_MHz"] = float(z["shift"].max() / 1e6); nums["relevo_shift_min_MHz"] = float(z["shift"].min() / 1e6)
nums["relevo_dC0_min_m"] = float(z["dC0"].min()); nums["relevo_h_max"] = float(z["hmap"].max()); nums["relevo_bxt"] = [float(b) for b in z["bxt"]]
fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.6), gridspec_kw=dict(wspace=0.12, hspace=0.42), sharey=True)
items = [(z["hmap"], "terrain", "Altura do terreno em geometria de radar [m]", None),
         (1e3 * z["dC0"], "RdBu_r", vg(f"$\\Delta C_0$ do canal 1 ($b_{{xt}}$ = {z['bxt'][1]:g} m) [mm]"), "sym"),
         (z["shift"] / 1e6, "viridis", "Deslocamento do espectro de range do canal 1 [MHz]", None),
         (100 * z["kept"], "cividis", "Fração da faixa de range comum aos 4 canais [%]", None)]
for a, (M_, cmap, title, mode) in zip(axs.ravel(), items):
    kw = dict(vmin=-np.abs(M_).max(), vmax=np.abs(M_).max()) if mode == "sym" else {}
    im = a.pcolormesh(x / 1e3, r - r0, M_.T, cmap=cmap, shading="auto", rasterized=True, **kw)
    fig.colorbar(im, ax=a, pad=0.02, fraction=0.05)
    a.plot(xy[:, 0] / 1e3, rt, "o", ms=4.5, mfc="none", mec="k", mew=0.9, ls="none")
    a.set_title(title, fontsize=10); a.grid(False); a.set_xlim(-8, 8)
    a.set_xlabel("posição em azimute $x$ [km]")
for a in axs[:, 0]:
    a.set_ylabel("range inclinado menos o do\nponto de referência [m]")
save(fig, "relevo_mapas.pdf")

Na = len(z["env_ref"]); t = (np.arange(Na) - Na // 2) / 2000.0
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), gridspec_kw=dict(hspace=0.42), sharex=True)
pk = z["env_ref"].max()
a1.plot(t, 20 * np.log10(z["env_ref"] / pk + 1e-12), color=REF, lw=2.0, label="Reference")
a1.plot(t, 20 * np.log10(z["env_joint"] / pk + 1e-12), color=JNT, lw=1.1, label="joint em 2-D (DEM em cada pixel da imagem)")
a1.set_title(vg(f"maior ambiguidade: referência {lev['ref']:.1f} dB | joint {lev['joint']:.1f} dB"), fontsize=10.5)
a2.plot(t, 20 * np.log10(z["env_ref"] / pk + 1e-12), color=REF, lw=2.0, label="Reference")
a2.plot(t, 20 * np.log10(z["env_none"] / pk + 1e-12), color=NON, lw=1.4, ls=(0, (5, 3)), label="no SATA")
a2.set_title(vg(f"maior ambiguidade: referência {lev['ref']:.1f} dB | no SATA {lev['none']:.1f} dB"), fontsize=10.5)
for a in (a1, a2):
    a.set_xlim(t[0], t[-1]); a.set_ylim(-90, 5); a.set_ylabel("[dB]"); a.legend(fontsize=8.5, loc="upper left"); a.tick_params(labelbottom=True)
a2.set_xlabel("tempo de azimute [s]")
save(fig, "irf_relevo.pdf")

# the image itself: whole band and common band
img = z["img"]; pk = img.max()
xi = (np.arange(img.shape[0]) * 2 - 2 * (img.shape[0] // 2)) * (vs / 2000.0)
keep = np.abs(xi) <= 8000
fig, axs = plt.subplots(2, 1, figsize=(11, 6.0), gridspec_kw=dict(hspace=0.36), sharex=True)
for ax, key, title in ((axs[0], "img", "A imagem que as iterações entregam, $|g(x, r)|$, com a faixa de range inteira"),
                       (axs[1], "img_common", "A mesma imagem só com as frequências de range que todos os canais veem em cada pixel")):
    im = ax.pcolormesh(xi[keep] / 1e3, r - r0, 20 * np.log10(z[key][keep] / pk + 1e-6).T, cmap="magma", vmin=-50, vmax=0, shading="auto", rasterized=True)
    fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03).set_label("[dB]")
    ax.plot(xy[:, 0] / 1e3, rt, "o", ms=11, mfc="none", mec="c", mew=0.8, ls="none")
    ax.grid(False); ax.set_ylabel("range inclinado menos o do\nponto de referência [m]"); ax.set_title(title, fontsize=10.5); ax.tick_params(labelbottom=True)
axs[1].set_xlabel("posição em azimute $x$ [km]  (círculos: onde os alvos estão de verdade)")
save(fig, "imagem.pdf")
ghost = lambda key: float(20 * np.log10(np.max([z[key][np.abs(xi - (x0 + sgn * 6228.0)) < 150].max() for x0 in xy[:, 0] for sgn in (-1, 1) if abs(x0 + sgn * 6228.0) < 8000]) / pk))
nums["imagem_fantasma_dB"] = dict(inteira=ghost("img"), comum=ghost("img_common"))

# ------------------------------------------------------------------------------------------------ 4) convergence
fig, ax = plt.subplots(figsize=(11, 3.4))
for h, c, lab in ((hist_p, "tab:blue", "um alvo em terreno plano"), (z["hist"], "tab:orange", "cinco alvos na montanha")):
    ax.plot(np.arange(1, len(h) + 1), h / 2 * 2, color=c, lw=1.6, label=lab)
    nums.setdefault("hist", {})[lab] = [float(h[0]), float(h[9]), float(h[59]), float(h[-1])]
ax.axvline(60.5, color="k", lw=0.8, ls=":"); ax.text(61.5, -14, "começa a 2ª passagem", fontsize=8.5)
ax.set_xlabel("iteração"); ax.set_ylabel("energia do resíduo / energia dos canais [dB]"); ax.legend(fontsize=8.5, loc="upper right")
ax.set_xlim(0, 121)
save(fig, "convergencia.pdf")

# ------------------------------------------------------------------------------------------------ 5) the band that is left, in theory
c, r0_, th = 299792458.0, 766206.0, np.radians(20.0)
b = np.linspace(0, 600, 301)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw=dict(wspace=0.22), sharey=True)
for a, Br, name in ((a1, 5e6, "5 MHz (30 m): a configuração das simulações"), (a2, 50e6, "50 MHz (3 m)")):
    for al, col in ((-8, "tab:blue"), (0, "k"), (8, "tab:orange"), (12, "tab:red")):
        s = c * b * np.cos(th) / (2 * wl * r0_ * np.tan(th - np.radians(al)))
        a.plot(b, 100 * np.clip(1 - s / Br, 0, 1), color=col, lw=1.6, label="terreno plano" if al == 0 else f"rampa de {abs(al)}$^\\circ$ " + ("de frente para o radar" if al > 0 else "de costas para o radar"))
    a.set_title("Largura de faixa de range " + name, fontsize=10.5); a.set_xlabel("linha de base cross-track $b_{xt}$ [m]"); a.set_xlim(0, 600); a.set_ylim(0, 102)
a1.set_ylabel("faixa de range comum\nao canal e ao transmissor [%]"); a1.legend(fontsize=8.5, loc="lower left")
save(fig, "faixa_teoria.pdf")
nums["bcrit_bperp_30m"] = float(wl * r0_ * np.tan(th) / (c / (2 * 5e6))); nums["bcrit_bxt_30m"] = nums["bcrit_bperp_30m"] / float(np.cos(th))
nums["shift_per_100m_MHz"] = float(c * 100 * np.cos(th) / (2 * wl * r0_ * np.tan(th)) / 1e6)

# ------------------------------------------------------------------------------------------------ 6) the sweep
SWEEP = [q for q in (os.path.join(HERE, "sweep", "all_numbers.json"), os.path.join(HERE, "..", "sweep_joint2d", "all_numbers.json")) if os.path.exists(q)][0]
R = json.load(open(SWEEP))                                    # written by: python -m sata2d.run_sweep_joint2d --table
fig, ax = plt.subplots(figsize=(11.5, 4.3))
k = np.arange(len(R))
ax.plot(k, [q["none"] for q in R], "x", color=NON, ms=6, mew=1.6, ls="none", label="no SATA")
ax.plot(k, [q["ref"] for q in R], "o", color=REF, ms=7, ls="none", label="Reference")
ax.plot(k, [q["joint"] for q in R], "D", color=JNT, ms=4.5, ls="none", label="joint em 2-D")
fam = [q["family"] for q in R]
for i in range(1, len(R)):
    if fam[i] != fam[i - 1]:
        ax.axvline(i - 0.5, color="0.4", lw=0.8)
names = {"Ramps": "rampas", "Sines and cosines": "senos e cossenos", "Realistic terrains": "relevos realistas"}
for f in dict.fromkeys(fam):
    idx = [i for i, q in enumerate(fam) if q == f]
    ax.text(np.mean(idx), 2, names[f], ha="center", fontsize=9)
ax.set_xticks(k); ax.set_xticklabels([str(q["nrx"]) for q in R], fontsize=7.5)
ax.set_xlabel("os 40 casos, na ordem das páginas do PDF (o número é $N_{rx}$)"); ax.set_ylabel("maior ambiguidade [dB]")
ax.set_ylim(-75, 8); ax.set_xlim(-0.7, len(R) - 0.3); ax.legend(fontsize=8.5, loc="lower right", ncol=3)
save(fig, "resultados.pdf")
med = lambda key, sel=lambda q: True: float(np.median([q[key] for q in R if sel(q)]))
nums["sweep"] = dict(n=len(R), ref=med("ref"), none=med("none"), joint=med("joint"), ref_whole=med("ref_whole"), none_whole=med("none_whole"), joint_whole=med("joint_whole"),
                     kept_median=med("band_kept_median"), kept_min=float(min(q["band_kept_min"] for q in R)),
                     within3=int(sum(q["joint"] <= q["ref"] + 3 for q in R)), within1=int(sum(q["joint"] <= q["ref"] + 1 for q in R)),
                     worse10=[(q["scene"], q["baseline"], q["ref"], q["joint"]) for q in R if q["joint"] > q["ref"] + 10],
                     worse3=[(q["scene"], q["baseline"], q["ref"], q["joint"]) for q in R if q["joint"] > q["ref"] + 3],
                     n2=[(q["scene"], q["baseline"], q["ref"], q["none"], q["joint"]) for q in R if q["nrx"] == 2],
                     s50=[(q["scene"], q["baseline"], q["ref"], q["none"], q["joint"]) for q in R if q["baseline"] == "s50"],
                     gain_median=float(np.median([q["none"] - q["joint"] for q in R])),
                     by_nrx={n: (int(sum(q["nrx"] == n for q in R)), med("ref", lambda q: q["nrx"] == n), med("none", lambda q: q["nrx"] == n), med("joint", lambda q: q["nrx"] == n)) for n in (2, 3, 4, 6, 8)})
json.dump(nums, open(os.path.join(HERE, "numeros_guia.json"), "w"), indent=1)
print(json.dumps(nums, indent=1))
