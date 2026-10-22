"""Figuras da seção "Casos que o método não resolve" do guia do algoritmo.

  figs/nao_resolve_previsao.pdf   as 680 rodadas: o acoplamento previsto com a geometria e o DEM contra o resultado
  figs/nao_resolve_mapa.pdf       o acoplamento ao longo da linha em três casos (um por causa)

Usa as cenas de audit_dem.py (pasta joint_cases; variável JOINT_CASES, ou a pasta acima desta) e os números de
acoplamento_todas.json (paginas_nao_resolve.py --all)."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
JC = os.environ.get("JOINT_CASES", os.path.dirname(HERE))
sys.path.insert(0, JC); sys.path.insert(0, os.path.dirname(JC))
import audit_dem as AD
import sata1d as S
from sata1d.config import ArrayGeometry
from sata1d.joint import reference_coefficients
from sata1d.joint_line import height_table, maps_from_heights

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})
TERRA, TERRAD = "#D7C9A7", "#6D4C2F"
C_OUT, C_N2, C_PASSO = "tab:blue", "tab:orange", "tab:purple"
pt = lambda s: s.replace(".", ",")
nums = {}

# ---------------------------------------------------------------- 1) the 680 runs: predicted coupling against the result
a = json.load(open(os.path.join(HERE, "acoplamento_todas.json")))
rows = []
for k, v in a.items():
    cause = "n2" if v["nrx"] == 2 else ("passo" if v["baseline"] in ("s10", "s50") else "outros")
    rows.append((v["c"][0][0], v["line"] - v["ref"], cause, v["line"] > max(-30.0, v["ref"] + 10.0), v["line"]))
fig, ax = plt.subplots(figsize=(11, 4.6))
eps = 1e-4
xval = lambda c: -np.log10(np.maximum(1.0 - np.asarray(c, float), eps))          # 0 -> c = 0 ; 2 -> c = 0.99 ; 4 -> c >= 0.9999
rng = np.random.default_rng(3)
for cause, col, lab in (("outros", C_OUT, "três ou mais canais, sem passo comum (510 rodadas)"),
                        ("n2", C_N2, "dois canais (102)"), ("passo", C_PASSO, "passo comum, 10 m e 50 m (68)")):
    S_ = [r for r in rows if r[2] == cause]
    for solved, kw in ((True, dict(mfc="none", mew=1.0, alpha=0.75)), (False, dict(mfc=col, mew=0.6, mec="k", alpha=0.95))):
        T = [r for r in S_ if (not r[3]) == solved]
        if not T:
            continue
        xx = xval([r[0] for r in T]) + np.where(np.array([r[0] for r in T]) > 1 - eps, rng.uniform(-0.07, 0.07, len(T)), 0.0)
        ax.plot(xx, [r[1] for r in T], "o", ms=5.5, color=col, ls="none", label=lab if solved else None, **kw)
ax.plot([], [], "o", ms=5.5, mfc="0.35", mec="k", mew=0.6, ls="none", label="símbolo cheio: não resolvida\n(ambiguidade acima de $-$30 dB)")
ax.axvline(xval(0.98), color="0.3", lw=0.9, ls=":")
ax.text(xval(0.98) + 0.03, 47.0, "$|c_m|$ = 0,98", fontsize=8.5, color="0.3", va="top")
ax.axhline(10.0, color="0.3", lw=0.9, ls=":")
tk = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
ax.set_xticks(xval(tk)); ax.set_xticklabels([pt(f"{t:g}") if t < 0.9999 else "1" for t in tk])
ax.set_xlim(-0.05, 4.2); ax.set_ylim(-6, 48)
ax.set_xlabel("maior acoplamento $|c_m|$ entre o lugar de um alvo e os lugares a $m$ dobras (calculado só com a geometria e o DEM)")
ax.set_ylabel("maior ambiguidade acima\nda referência [dB]")
ax.set_title("As 680 rodadas: todas as não resolvidas têm um par de lugares com $|c_m|$ $\\geq$ 0,98", fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.93), framealpha=0.95)
fig.savefig(os.path.join(HERE, "figs", "nao_resolve_previsao.pdf")); plt.close(fig)
hi = [r for r in rows if r[0] >= 0.98]; lo = [r for r in rows if r[0] < 0.98]; top = [r for r in rows if r[0] >= 0.995]
nums["previsao"] = dict(n=len(rows), acima_098=len(hi), nr_acima_098=int(sum(r[3] for r in hi)), abaixo_098=len(lo),
                        nr_abaixo_098=int(sum(r[3] for r in lo)), acima_0995=len(top), nr_acima_0995=int(sum(r[3] for r in top)),
                        mediana_gap_abaixo=float(np.median([r[1] for r in lo])), mediana_gap_acima_0995=float(np.median([r[1] for r in top])),
                        longe_abaixo_098=int(sum(r[1] > 10 for r in lo)))

# ---------------------------------------------------------------- 2) the coupling along the line, one case per cause
CASES = [("many_ramp8_49", "N2b100", "dois canais"), ("few_ramp4_5", "s50", "passo comum de 50 m"),
         ("many_peak_33", "N4b300", "quatro canais, $b_{xt}$ até 300 m")]
fig, axs = plt.subplots(2, 3, figsize=(11, 5.4), gridspec_kw=dict(hspace=0.18, wspace=0.22, height_ratios=[0.8, 1.0]), sharex="col")
MCOL = {1: "tab:red", -1: "tab:blue", 2: "tab:green", -2: "tab:olive"}
for j, (sc, tag, title) in enumerate(CASES):
    c = AD.SCENES[sc][1]()
    bxt, _ = AD.baseline(tag)
    cfg = S.make_config(list(zip(c["xs"], c["hs"])), nrx=len(bxt), bxt_max=0.0, amplitudes=c["amps"], tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy()))
    tr = S.build_platform_tracks(cfg)
    wl, v, r0, P = cfg.system.wl, cfg.system.vs, cfg.scene.r0, cfg.PRF_op
    X = wl * r0 * P / (2.0 * v); B = 4.0 * v / wl * np.sin(cfg.theta_tx / 2.0)
    tauP = (lambda q: (q[:, 3] - q[:, 1]) * P)(reference_coefficients(cfg, tr))
    terrain, xs = c["terrain"], np.asarray(c["xs"], float)
    half = 0.5 * cfg.Na / cfg.prf * v
    tab = height_table(cfg, tr, terrain(np.linspace(-half - 100, half + 100, 8001)).min(), terrain(np.linspace(-half - 100, half + 100, 8001)).max())
    psi = lambda x: 2.0 * np.pi * maps_from_heights(cfg, tr, terrain(np.asarray(x, float)), tab) / wl           # [Nrx, len(x)]
    pad = max(0.25 * np.ptp(xs), 1500.0)
    xg = np.linspace(xs.min() - pad, xs.max() + pad, 1601)
    a1, a2 = axs[0, j], axs[1, j]
    a1.fill_between(xg / 1e3, 0, terrain(xg), color=TERRA, alpha=0.6, lw=0); a1.plot(xg / 1e3, terrain(xg), color=TERRAD, lw=1.2)
    cm_t = {}
    for m in (1, -1, 2, -2):
        beta = (B - abs(m) * P) / B
        if beta <= 0:
            continue
        cm = np.abs(np.mean(np.exp(1j * (2.0 * np.pi * m * tauP[:, None] + psi(xg + m * X) - psi(xg))), axis=0))
        a2.plot(xg / 1e3, cm, color=MCOL[m], lw=1.3, label=f"$m$ = {m:+d}".replace("-", "$-$"))
        cm_t[m] = np.abs(np.mean(np.exp(1j * (2.0 * np.pi * m * tauP[:, None] + psi(xs + m * X) - psi(xs))), axis=0))
    worst = np.max(np.array(list(cm_t.values())), axis=0)
    bad = worst >= 0.98
    a1.plot(xs[~bad] / 1e3, terrain(xs[~bad]), "o", ms=4.5, color="#1F4E79", mec="k", mew=0.5, ls="none", label="alvo")
    a1.plot(xs[bad] / 1e3, terrain(xs[bad]), "o", ms=6.5, color="tab:red", mec="k", mew=0.6, ls="none", label="alvo com $|c_m|$ $\\geq$ 0,98")
    a2.plot(xs / 1e3, worst, "o", ms=4, color="k", mfc="none", ls="none", label="nos alvos (o maior)")
    a2.axhline(1.0, color="0.3", lw=0.8, ls=":")
    a2.set_ylim(0, 1.08); a2.set_xlim(xg[0] / 1e3, xg[-1] / 1e3)
    a1.set_title(f"{title}\n$b_{{xt}}$ = [{'; '.join(pt(f'{b:g}') for b in np.round(bxt, 0))}] m, uma dobra = {pt(f'{X / 1e3:.1f}')} km", fontsize=9.5)
    a2.set_xlabel("posição x no chão [km]")
    if j == 0:
        a1.set_ylabel("altura h [m]"); a2.set_ylabel("acoplamento $|c_m(x)|$")
    if j == 0:
        a1.legend(fontsize=7.5, loc="upper left")
    if j == 1:
        hand = a2.get_legend_handles_labels()
    fmt = matplotlib.ticker.FuncFormatter(lambda val, pos: pt(f"{val:g}").replace("-", "\u2212"))
    a2.xaxis.set_major_formatter(fmt); a2.yaxis.set_major_formatter(fmt)
    nums[f"{sc}__{tag}"] = dict(X=float(X), n_alvos=int(len(xs)), n_alvos_098=int(bad.sum()), maior=float(worst.max()),
                                x_piores=[float(x) for x in xs[np.argsort(-worst)[:4]]], tauP=tauP.tolist(), B=float(B))
fig.legend(*hand, fontsize=8.5, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), frameon=False)
fig.savefig(os.path.join(HERE, "figs", "nao_resolve_mapa.pdf")); plt.close(fig)
json.dump(nums, open(os.path.join(HERE, "numeros_nao_resolve.json"), "w"), indent=1)
print(json.dumps(nums, indent=1))
