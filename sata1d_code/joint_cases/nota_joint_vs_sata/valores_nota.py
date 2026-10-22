"""Números da nota (numeros_nota.json), a partir de audit_dem/all_numbers.json e das cenas de audit_dem.py."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
JC = os.environ.get("JOINT_CASES", os.path.dirname(HERE))          # joint_cases (a pasta acima desta)
sys.path.insert(0, JC); sys.path.insert(0, os.path.dirname(JC))
import audit_dem as AD


def dados(nome):
    """Arquivo da auditoria: nesta pasta, se houver uma cópia, senão em joint_cases/audit_dem/."""
    p = os.path.join(HERE, nome)
    return p if os.path.exists(p) else os.path.join(JC, "audit_dem", nome)

rows = [r for r in json.load(open(dados("all_numbers.json"))) if r["group"] != "control"]
ctrl = [r for r in json.load(open(dados("all_numbers.json"))) if r["group"] == "control"]
M = ["none", "whole", "sub", "joint_occ", "joint_line"]
A = {m: np.array([r[m] for r in rows]) for m in M + ["ref"]}
N = {}
N["n"] = len(rows)
N["mediana"] = {m: float(np.median(A[m])) for m in M + ["ref"]}
N["mediana_gap"] = {m: float(np.median(A[m] - A["ref"])) for m in M}
N["ate10"] = {m: float(100 * np.mean(A[m] - A["ref"] <= 10)) for m in M}
N["abaixo30"] = {m: float(100 * np.mean(A[m] <= -30)) for m in M}
N["pior_que_nada"] = {m: float(100 * np.mean(A[m] > A["none"] + 1)) for m in M[1:]}
N["pior_que_nada_n"] = {m: int(np.sum(A[m] > A["none"] + 1)) for m in M[1:]}
for other in ("whole", "sub", "joint_occ", "none"):
    d = A[other] - A["joint_line"]
    N[f"linha_vs_{other}"] = dict(melhor=int(np.sum(d > 3)), empate=int(np.sum(np.abs(d) <= 3)), pior=int(np.sum(d < -3)), ganho_mediano=float(np.median(d)),
                                  ganho_min=float(d.min()), ganho_max=float(d.max()), nunca_pior_que=float(-d.min()))
N["whole_ate10_n"] = int(np.sum(A["whole"] - A["ref"] <= 10)); N["sub_ate10_n"] = int(np.sum(A["sub"] - A["ref"] <= 10))
N["whole_ate10_cenas"] = sorted({r["scene"] for r in rows if r["whole"] - r["ref"] <= 10})
N["whole_abaixo30_n"] = int(np.sum(A["whole"] <= -30)); N["sub_abaixo30_n"] = int(np.sum(A["sub"] <= -30)); N["linha_abaixo30_n"] = int(np.sum(A["joint_line"] <= -30))
N["melhora_sata_whole"] = float(np.median(A["none"] - A["whole"])); N["melhora_sata_sub"] = float(np.median(A["none"] - A["sub"]))
N["melhora_linha"] = float(np.median(A["none"] - A["joint_line"]))
N["por_nrx"] = {}
for n in (2, 3, 4, 6, 8):
    s = np.array([r["nrx"] == n for r in rows])
    N["por_nrx"][n] = dict(n=int(s.sum()), **{m: float(np.median(A[m][s])) for m in M + ["ref"]})
N["por_bxt"] = {}
for b in (100, 300, 500):
    s = np.array([r["nrx"] >= 3 and r["baseline"].endswith(f"b{b}") for r in rows])
    N["por_bxt"][b] = dict(n=int(s.sum()), **{m: float(np.median(A[m][s])) for m in M + ["ref"]})
N["por_grupo"] = {}
for g in ("one", "few", "many"):
    s = np.array([r["group"] == g for r in rows])
    N["por_grupo"][g] = dict(n=int(s.sum()), **{m: float(np.median(A[m][s])) for m in M + ["ref"]}, linha_melhor_whole=int(np.sum((A["whole"] - A["joint_line"])[s] > 3)))
# three or more channels, no common step
s = np.array([r["nrx"] >= 3 and r["baseline"] not in ("s10", "s50") for r in rows])
N["n3_sem_passo"] = dict(n=int(s.sum()), **{f"ate10_{m}": float(100 * np.mean((A[m] - A["ref"])[s] <= 10)) for m in M}, **{f"abaixo30_{m}": float(100 * np.mean(A[m][s] <= -30)) for m in M},
                         **{f"mediana_{m}": float(np.median(A[m][s])) for m in M})
N["tempo_n4"] = {m: float(np.median([r["seconds"][m] for r in rows if r["nrx"] == 4])) for m in M}
N["controles"] = {f"{r['scene']}__{r['baseline']}": {m: r[m] for m in M + ["ref"]} for r in ctrl}
# fraction of the band that gets the phase of its own place with SATA whole band: P / B
B = 1281.4
N["fracao_certa"] = {n: float(min(1.0, 2000.0 / n / B)) for n in (2, 3, 4, 6, 8)}
json.dump(N, open(os.path.join(HERE, "numeros_nota.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in N.items() if k != "controles"}, indent=1))
for k, v in N["controles"].items():
    if "period" in k and k.split("__")[1].startswith("N") and k.endswith("b100"):
        print(k, {m: round(x, 1) for m, x in v.items()})

# ------------------------------------------------------------------ extras
N["por_nrx_aleatorias"] = {}
for n in (2, 3, 4, 6, 8):
    s = np.array([r["baseline"].startswith(f"N{n}b") for r in rows])
    N["por_nrx_aleatorias"][n] = dict(n=int(s.sum()), **{m: float(np.median(A[m][s])) for m in M + ["ref"]},
                                      linha_ate10=float(100 * np.mean((A["joint_line"] - A["ref"])[s] <= 10)), whole_ate10=float(100 * np.mean((A["whole"] - A["ref"])[s] <= 10)))
D = json.load(open(dados("distributed.json")))
N["distribuida"] = {}
for b in ("rnd100", "rnd500", "s50"):
    R = [r for r in D if r["baseline"] == b and r["case"] != "FLAT"]
    N["distribuida"][b] = dict(n=len(R), bxt=R[0]["bxt"], **{m: [float(min(r["metrics"][m]["esr"] for r in R)), float(np.median([r["metrics"][m]["esr"] for r in R])), float(max(r["metrics"][m]["esr"] for r in R))]
                                                              for m in M + ["oracle"]})
    N["distribuida"][b]["casos"] = {r["case"]: {m: float(r["metrics"][m]["esr"]) for m in M + ["oracle"]} for r in R}
N["distribuida_plano"] = {r["baseline"]: {m: float(r["metrics"][m]["esr"]) for m in M + ["oracle"]} for r in D if r["case"] == "FLAT"}
unres = np.array([r["joint_line"] > max(-30.0, r["ref"] + 10.0) for r in rows])
N["nao_resolvidas"] = dict(n=int(unres.sum()), **{f"mediana_{m}": float(np.median(A[m][unres])) for m in M}, ganho_sobre_whole=float(np.median((A["whole"] - A["joint_line"])[unres])),
                           whole_resolve=int(np.sum((A["whole"] <= np.maximum(-30.0, A["ref"] + 10.0))[unres])), sub_resolve=int(np.sum((A["sub"] <= np.maximum(-30.0, A["ref"] + 10.0))[unres])))
res = lambda m: A[m] <= np.maximum(-30.0, A["ref"] + 10.0)
N["resolvidas"] = {m: int(res(m).sum()) for m in M}
N["melhora_sobre_nada"] = {m: dict(mediana=float(np.median(A["none"] - A[m])), mais10=float(100 * np.mean(A["none"] - A[m] > 10)), mais20=float(100 * np.mean(A["none"] - A[m] > 20))) for m in M[1:]}
# hand formula, two channels, one target at h = 0 on h = 100 - 100 cos(2 pi x / 50 km)
wl, r0, v, th = 0.25, 766206.0, 7688.5, np.radians(20.0)
K = wl * r0 * np.tan(th); P2 = 1000.0; X2 = wl * r0 * P2 / (2 * v)
dh = 100 - 100 * np.cos(2 * np.pi * X2 / 50e3)
N["mao"] = dict(lam_r0_tan=float(K), X2=float(X2), dh=float(dh), fracao=float((B - P2) / (2 * B)))
for b in (100.0, 300.0):
    al = 2 * np.pi * b * dh / K
    N["mao"][f"b{b:g}"] = dict(alpha=float(al), alpha_graus=float(np.degrees(al)), seno=float(abs(np.sin(al / 2))), nivel=float(20 * np.log10((B - P2) / (2 * B) * abs(np.sin(al / 2)))))
N["mao"]["dh_30dB_b100"] = float(2 * np.arcsin(10 ** (-30 / 20)) * K / (2 * np.pi * 100.0))
N["mao"]["dh_30dB_b500"] = float(2 * np.arcsin(10 ** (-30 / 20)) * K / (2 * np.pi * 500.0))
N["h2pi"] = {b: float(K / b) for b in (100, 134.9, 300, 500)}
N["dobra"] = {n: float(wl * r0 * (2000.0 / n) / (2 * v)) for n in (2, 3, 4, 6, 8)}
N["cenas"] = len({r["scene"] for r in rows}); N["conjuntos"] = len({r["baseline"] for r in rows})
json.dump(N, open(os.path.join(HERE, "numeros_nota.json"), "w"), indent=1)
for k in ("por_nrx_aleatorias", "distribuida", "distribuida_plano", "nao_resolvidas", "resolvidas", "melhora_sobre_nada", "mao", "h2pi", "dobra", "cenas", "conjuntos"):
    v_ = N[k]
    if k == "distribuida":
        v_ = {b: {m: [round(x, 1) for x in q[m]] for m in M + ["oracle"]} for b, q in v_.items()}
    print(k, json.dumps(v_))
