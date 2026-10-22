"""Figuras da nota "Por que o joint na linha inteira ganha do SATA".
  figs/resumo_680.pdf       distribuição da maior ambiguidade nas 680 rodadas, por método
  figs/frente_a_frente.pdf  cada rodada: SATA (banda inteira / por sub-banda) contra o joint na linha inteira
  figs/exemplo_irf.pdf      o exemplo dos guias (montanha de 1000 m, 5 alvos)
  figs/controle_dobra.pdf   terreno periódico: o SATA funciona quando o período é a dobra, e só então
  figs/sata_piora.pdf       um alvo na altura de referência: o SATA cria ambiguidades
  figs/previsao_sata.pdf    o que o SATA deixa, previsto só com a geometria e o DEM, contra a simulação
Entradas: all_numbers.json (auditoria), irfs.npz / irfs.json (calc_irfs.py), prev_sata.json (prev_sata.py)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
HERE = os.path.dirname(os.path.abspath(__file__))
JC = os.environ.get("JOINT_CASES", os.path.dirname(HERE))          # joint_cases (a pasta acima desta)
sys.path.insert(0, JC); sys.path.insert(0, os.path.dirname(JC))
import audit_dem as AD


def dados(nome):
    """Arquivo da auditoria: nesta pasta, se houver uma cópia, senão em joint_cases/audit_dem/."""
    p = os.path.join(HERE, nome)
    return p if os.path.exists(p) else os.path.join(JC, "audit_dem", nome)


plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight", "axes.axisbelow": True})
TERRA, TERRAD = "#D7C9A7", "#6D4C2F"
COL = {"ref": "tab:green", "none": "tab:red", "whole": "tab:blue", "sub": "tab:orange", "joint_occ": "tab:pink", "joint_line": "tab:purple"}
NOME = {"ref": "referência", "none": "sem SATA", "whole": "SATA banda inteira", "sub": "SATA por sub-banda",
        "joint_occ": "joint em blocos", "joint_line": "joint na linha inteira"}
LS = {"ref": "-", "none": (0, (5, 3)), "whole": "-", "sub": (0, (3, 1.5)), "joint_occ": (0, (5, 3)), "joint_line": "-"}
LW = {"ref": 2.0, "none": 1.3, "whole": 1.4, "sub": 1.4, "joint_occ": 1.4, "joint_line": 2.0}
pt = lambda s: s.replace(".", ",").replace("-", "−")
fmt = FuncFormatter(lambda v, _: pt(f"{v:g}"))
def ptaxes(*axs):
    for a in axs:
        a.xaxis.set_major_formatter(fmt); a.yaxis.set_major_formatter(fmt)
F = lambda n: os.path.join(HERE, "figs", n)
rows = [r for r in json.load(open(dados("all_numbers.json"))) if r["group"] != "control"]
M = ["none", "whole", "sub", "joint_occ", "joint_line"]
A = {m: np.array([r[m] for r in rows]) for m in M + ["ref"]}
which = set(sys.argv[1:]) or {"resumo", "frente", "exemplo", "controle", "piora", "previsao"}

# ---------------------------------------------------------------- 1) the 680 runs, distribution per method
if "resumo" in which:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(wspace=0.22))
    for m in ["ref"] + M:
        v = np.sort(A[m]); y = 100.0 * np.arange(1, len(v) + 1) / len(v)
        a1.step(v, y, where="post", color=COL[m], ls=LS[m], lw=LW[m], label=NOME[m])
    a1.axvline(-30, color="0.35", lw=0.9, ls=":"); a1.text(-29.2, 97, "$-$30 dB", fontsize=8.5, color="0.3", ha="left", va="top")
    a1.set_xlim(-75, 5); a1.set_ylim(0, 100)
    a1.set_xlabel("maior ambiguidade [dB]"); a1.set_ylabel("rodadas com a maior ambiguidade\naté este valor [%]")
    a1.set_title("680 rodadas: quanto mais à esquerda, melhor", fontsize=10)
    for m in M:
        v = np.sort(A[m] - A["ref"]); y = 100.0 * np.arange(1, len(v) + 1) / len(v)
        a2.step(v, y, where="post", color=COL[m], ls=LS[m], lw=LW[m])
    a2.axvline(10, color="0.35", lw=0.9, ls=":"); a2.text(10.8, 97, "10 dB", fontsize=8.5, color="0.3", ha="left", va="top")
    a2.set_xlim(-3, 66); a2.set_ylim(0, 100)
    a2.set_xlabel("maior ambiguidade acima da referência [dB]")
    a2.set_title("As mesmas rodadas, em relação à referência de cada uma", fontsize=10)
    fig.legend(*a1.get_legend_handles_labels(), fontsize=9, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.07), frameon=False, handlelength=2.6)
    ptaxes(a1, a2)
    fig.savefig(F("resumo_680.pdf")); plt.close(fig)

# ---------------------------------------------------------------- 2) every run: SATA against the whole line
if "frente" in which:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw=dict(wspace=0.18), sharey=True)
    G = {"one": ("um alvo", "o", "tab:blue"), "few": ("poucos alvos (2 a 9)", "s", "tab:orange"), "many": ("muitos alvos (13 a 49)", "^", "tab:green")}
    grp = np.array([r["group"] for r in rows])
    for ax, m in zip(axs, ("whole", "sub")):
        ax.fill_between([-80, 10], [-80, 10], [10, 10], color="0.92", lw=0, zorder=0)
        ax.plot([-80, 10], [-80, 10], color="0.25", lw=1.0)
        for g, (lab, mk, col) in G.items():
            s = grp == g
            ax.plot(A[m][s], A["joint_line"][s], mk, ms=4.6, mfc="none", mec=col, mew=0.9, alpha=0.8, ls="none", label=f"{lab}: {int(s.sum())} rodadas")
        ax.set_xlim(-72, 4); ax.set_ylim(-72, 4); ax.set_aspect("equal")
        ax.set_xlabel(f"maior ambiguidade com o {NOME[m]} [dB]")
        ax.text(-69, 1, "acima da diagonal:\no SATA seria melhor", fontsize=8.5, color="0.3", va="top")
        ax.text(1.5, -70.5, "abaixo da diagonal: o joint na linha inteira é melhor", fontsize=8.5, color="0.3", ha="right", va="bottom",
                bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))
        d = A[m] - A["joint_line"]
        ax.set_title(f"contra o {NOME[m]}: melhor em {int((d > 0).sum())} de {len(d)}\n(por mais de 3 dB em {int((d > 3).sum())}; ganho mediano de {pt(f'{np.median(d):.1f}')} dB)", fontsize=9.5)
    axs[0].set_ylabel("maior ambiguidade com o joint na linha inteira [dB]")
    axs[1].legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.84), framealpha=0.95)
    ptaxes(*axs)
    fig.savefig(F("frente_a_frente.pdf")); plt.close(fig)

Z = np.load(os.path.join(HERE, "irfs.npz")) if os.path.exists(os.path.join(HERE, "irfs.npz")) else None
J = json.load(open(os.path.join(HERE, "irfs.json"))) if Z is not None else None
dB = lambda e: 20 * np.log10(np.asarray(e, float) + 1e-12)

def irf(ax, key, methods, xlim=None, ylim=(-90, 5), legend=True, fs=8.3, loc="upper left"):
    """IRFs of one case; the worst method is drawn first, the reference last (on top, thin)."""
    x = Z[f"{key}__t"] * J[key]["vs"] / 1e3
    order = sorted([m for m in methods if m != "ref"], key=lambda m: -J[key]["amb"][m]) + ["ref"]
    hand = {}
    for z, m in enumerate(order):
        hand[m], = ax.plot(x, dB(Z[f"{key}__{m}"]), color=COL[m], ls="-", lw=0.8 if m == "ref" else 0.9, zorder=2 + z,
                           label=f"{NOME[m]}: {pt(format(J[key]['amb'][m], '.1f'))} dB")
    ax.set_xlim(*(xlim or (x[0], x[-1]))); ax.set_ylim(*ylim)
    if legend:
        leg = ax.legend([hand[m] for m in methods], [hand[m].get_label() for m in methods], fontsize=fs, loc=loc, framealpha=0.95,
                        title="maior ambiguidade", title_fontsize=fs)
        leg.set_zorder(20)
        for l in leg.get_lines():
            l.set_linewidth(1.8)

# ---------------------------------------------------------------- 3) the example of the guides
if "exemplo" in which:
    key = "few_peak_5__u500"
    c = AD.SCENES["few_peak_5"][1](); terrain, xs = c["terrain"], np.asarray(c["xs"])
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.3), gridspec_kw=dict(hspace=0.3, wspace=0.08), sharex=True)
    X = J[key]["X"]; lim = (-21.5, 21.5)
    a0 = axs[0, 0]
    xg = np.linspace(-21.5e3, 21.5e3, 3001)
    a0.fill_between(xg / 1e3, 0, terrain(xg), color=TERRA, alpha=0.6, lw=0); a0.plot(xg / 1e3, terrain(xg), color=TERRAD, lw=1.2)
    a0.plot(xs / 1e3, terrain(xs), "o", ms=6, color="#1F4E79", mec="k", ls="none", label="os cinco alvos")
    for m in (-2, -1, 1, 2):
        a0.plot((xs + m * X) / 1e3, np.full(len(xs), -110.0), "|", ms=9, color="0.3", ls="none", label="onde caem as ambiguidades\n(a 1 e 2 dobras de cada alvo)" if m == -2 else None)
    a0.set_ylim(-220, 1900); a0.set_ylabel("altura h [m]"); a0.legend(fontsize=8.0, loc="upper left")
    a0.set_title("a cena (o DEM que todos leem)", fontsize=9.5)
    for ax, m in zip(axs.ravel()[1:], ["none", "whole", "sub", "joint_occ", "joint_line"]):
        irf(ax, key, [m, "ref"], xlim=lim, fs=8.0)
        ax.set_title(NOME[m], fontsize=9.5, color=COL[m], fontweight="bold")
    for ax in (axs[0, 2], axs[1, 1], axs[1, 2]):
        ax.tick_params(labelleft=False)
    axs[0, 1].yaxis.tick_right(); axs[0, 1].tick_params(labelright=False)
    axs[1, 0].set_ylabel("resposta [dB]"); axs[0, 2].yaxis.set_label_position("right"); axs[0, 2].yaxis.tick_right(); axs[0, 2].tick_params(labelright=True); axs[0, 2].set_ylabel("resposta [dB]")
    for ax in axs[1]:
        ax.set_xlabel("posição x no chão [km]")
    ptaxes(*axs.ravel())
    fig.savefig(F("exemplo_irf.pdf")); plt.close(fig)

# ---------------------------------------------------------------- 4) periodic terrain: SATA works when the period is the fold, and only then
if "controle" in which:
    CASES = [("ctrl_periodX_5__N4b100", "ctrl_periodX_5", "4 canais, dobra de 6,23 km\nperíodo do terreno = 1 dobra"),
             ("ctrl_period1p5X_5__N4b100", "ctrl_period1p5X_5", "4 canais, dobra de 6,23 km\nperíodo do terreno = 1,5 dobra"),
             ("ctrl_periodX_5__N3b100", "ctrl_periodX_5", "3 canais, dobra de 8,30 km\no mesmo terreno da esquerda")]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.1), gridspec_kw=dict(hspace=0.12, wspace=0.08, height_ratios=[0.62, 1.0]), sharex=True, sharey="row")
    for j, (key, sc, title) in enumerate(CASES):
        c = AD.SCENES[sc][1](); terrain, xs = c["terrain"], np.asarray(c["xs"]); X = J[key]["X"]
        a1, a2 = axs[0, j], axs[1, j]
        xg = np.linspace(-20e3, 20e3, 3001)
        a1.fill_between(xg / 1e3, 0, terrain(xg), color=TERRA, alpha=0.6, lw=0); a1.plot(xg / 1e3, terrain(xg), color=TERRAD, lw=1.2)
        a1.plot(xs / 1e3, terrain(xs), "o", ms=5, color="#1F4E79", mec="k", mew=0.5, ls="none", label="alvos")
        x0 = xs[len(xs) // 2]                                               # the middle target and the places one fold away
        xf = np.array([x0 - X, x0 + X])
        a1.plot([x0 / 1e3], [terrain(np.array([x0]))[0]], "o", ms=8.5, mfc="none", mec="tab:red", mew=1.6, ls="none")
        a1.plot(xf / 1e3, terrain(xf), "D", ms=6.5, color="tab:red", mec="k", mew=0.5, ls="none", label="a uma dobra do alvo do meio")
        for x_ in xf:
            a1.annotate("", xy=(x_ / 1e3, 262), xytext=(x0 / 1e3, 262), arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.0))
        h0, hf = terrain(np.array([x0]))[0], terrain(xf)
        a1.text(x0 / 1e3, 300, f"alvo: {h0:.0f} m;  a uma dobra: {hf[0]:.0f} m e {hf[1]:.0f} m", ha="center", fontsize=8.3, color="tab:red")
        a1.set_ylim(-15, 345); a1.set_title(title, fontsize=9.5)
        irf(a2, key, ["ref", "whole", "joint_line"], xlim=(-20, 20), fs=8.0)
        a2.set_xticks([-15, -10, -5, 0, 5, 10, 15])
        a2.set_xlabel("posição x no chão [km]")
    axs[0, 0].set_ylabel("altura h [m]"); axs[1, 0].set_ylabel("resposta [dB]")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), fontsize=8.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.045), frameon=False)
    ptaxes(*axs.ravel())
    fig.savefig(F("controle_dobra.pdf")); plt.close(fig)

# ---------------------------------------------------------------- 5) one target at the reference height: SATA creates ambiguities
if "piora" in which:
    key = "one_refpoint__N4b100"
    c = AD.SCENES["one_refpoint"][1](); terrain, xs = c["terrain"], np.asarray(c["xs"]); X = J[key]["X"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw=dict(wspace=0.2, width_ratios=[0.8, 1.2]))
    xg = np.linspace(-20e3, 20e3, 2001)
    a1.fill_between(xg / 1e3, 0, terrain(xg), color=TERRA, alpha=0.6, lw=0); a1.plot(xg / 1e3, terrain(xg), color=TERRAD, lw=1.2)
    a1.plot(xs / 1e3, terrain(xs), "o", ms=7, color="#1F4E79", mec="k", ls="none", label="o alvo (h = 0)")
    xf = np.array([xs[0] - X, xs[0] + X]); a1.plot(xf / 1e3, terrain(xf), "D", ms=6.5, color="tab:red", mec="k", mew=0.5, ls="none", label="a uma dobra do alvo")
    for x_ in xf:
        a1.annotate(f"{terrain(np.array([x_]))[0]:.0f} m", (x_ / 1e3, terrain(np.array([x_]))[0]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color="tab:red")
    a1.set_xlabel("posição x no chão [km]"); a1.set_ylabel("altura h [m]"); a1.set_ylim(-8, 135); a1.legend(fontsize=8.3, loc="upper center")
    a1.set_title("terreno h = 100 $-$ 100 cos(2$\\pi$x / 50 km)", fontsize=9.5)
    irf(a2, key, ["ref", "none", "whole", "sub"], xlim=(-20, 20))
    a2.set_xlabel("posição x no chão [km]"); a2.set_ylabel("resposta [dB]")
    a2.set_title("4 canais, $b_{xt}$ = [0; 100; 40; 34] m".replace("[0; 100; 40; 34]", "[" + "; ".join(pt(f"{b:g}") for b in np.round(J[key]["bxt"], 0)) + "]"), fontsize=9.5)
    ptaxes(a1, a2)
    fig.savefig(F("sata_piora.pdf")); plt.close(fig)

# ---------------------------------------------------------------- 6) what SATA leaves: predicted against simulated
if "previsao" in which:
    P = json.load(open(os.path.join(HERE, "prev_sata.json")))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.7), gridspec_kw=dict(wspace=0.18), sharey=True)
    G = {"one": ("um alvo", "o", "tab:blue"), "few": ("poucos alvos", "s", "tab:orange"), "many": ("muitos alvos", "^", "tab:green")}
    nums = {}
    for ax, m in zip(axs, ("whole", "sub")):
        ax.plot([-80, 10], [-80, 10], color="0.25", lw=1.0, zorder=1)
        for g, (lab, mk, col) in G.items():
            v = [(q[m], q[f"sim_{m}"]) for q in P.values() if q["group"] == g]
            if v:
                v = np.array(v); ax.plot(v[:, 0], v[:, 1], mk, ms=4.6, mfc="none", mec=col, mew=0.9, alpha=0.8, ls="none", label=lab)
                d = v[:, 1] - v[:, 0]; nums[f"{m}_{g}"] = dict(n=len(d), media=float(d.mean()), mediana=float(np.median(d)), rms=float(np.sqrt((d ** 2).mean())), maior=float(np.abs(d).max()),
                                                               dentro3=float(100 * np.mean(np.abs(d) <= 3)))
        ax.set_xlim(-54, 0); ax.set_ylim(-54, 0); ax.set_aspect("equal")
        ax.set_xlabel("previsto só com a geometria e o DEM [dB]")
        dd = np.array([q[f"sim_{m}"] - q[m] for q in P.values()])
        ax.set_title(f"{NOME[m]}: {len(dd)} rodadas, diferença rms de {pt(format(np.sqrt((dd ** 2).mean()), '.1f'))} dB", fontsize=10)
        nums[f"{m}_todas"] = dict(n=len(dd), media=float(dd.mean()), mediana=float(np.median(dd)), rms=float(np.sqrt((dd ** 2).mean())), maior=float(np.abs(dd).max()),
                                  dentro1=float(100 * np.mean(np.abs(dd) <= 1)), dentro2=float(100 * np.mean(np.abs(dd) <= 2)), dentro3=float(100 * np.mean(np.abs(dd) <= 3)))
    axs[0].set_ylabel("maior ambiguidade simulada [dB]"); axs[0].legend(fontsize=8.5, loc="upper left")
    ptaxes(*axs)
    fig.savefig(F("previsao_sata.pdf")); plt.close(fig)
    json.dump(nums, open(os.path.join(HERE, "numeros_previsao.json"), "w"), indent=1); print(json.dumps(nums, indent=1))
