"""Páginas de seis painéis dos casos que o joint na linha inteira não resolve (Seção "Casos que o método não resolve"
do guia do algoritmo), com o acoplamento entre lugares a m dobras previsto só com a geometria e o DEM.

    python paginas_nao_resolve.py            # faz as páginas que faltam em ./paginas e escreve nao_resolve.json
    python paginas_nao_resolve.py --stats    # só as contagens por causa, a partir de audit_dem/all_numbers.json
    python paginas_nao_resolve.py --all      # o acoplamento previsto para as 680 rodadas -> acoplamento_todas.json
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # joint_cases
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # sata1d_code
import audit_dem as AD
from sata1d.joint import reference_coefficients
from sata1d.joint_line import height_table, maps_from_heights

CASES = [  # (causa, cena, linhas de base)
    ("dois canais", "few_plateau_9", "N2b100"), ("dois canais", "many_sin50km_31", "N2b300"),
    ("dois canais", "many_ramp8_49", "N2b100"), ("dois canais", "many_mountains_49", "N2b300"),
    ("passo comum", "few_escarpment_9", "s50"), ("passo comum", "few_mountains_5", "s50"),
    ("passo comum", "few_ramp4_5", "s50"), ("passo comum", "many_ramp3_49", "s50"),
    ("linha de base grande em declive", "many_peak_33", "N4b300"),
    ("linha de base grande em declive", "many_ramp8_49", "N4b500"),
    ("linha de base grande em declive", "one_ramp8", "N8b100"),
]
OUT = os.path.join(HERE, "paginas")


def coupling(cfg, tr, terrain, xs):
    """|c_m| entre o lugar de cada alvo e os lugares a m dobras, para os m em que as duas parcelas se sobrepõem.
    Só usa a geometria (filtros do ponto de referência, linhas de base) e o DEM. Devolve [(|c|, beta, m, x)], o maior
    primeiro."""
    wl, v, r0, P = cfg.system.wl, cfg.system.vs, cfg.scene.r0, cfg.PRF_op
    coef = reference_coefficients(cfg, tr)
    tauP = (coef[:, 3] - coef[:, 1]) * P
    X = wl * r0 * P / (2.0 * v)
    B = 4.0 * v / wl * np.sin(cfg.theta_tx / 2.0)
    half = 0.5 * cfg.Na / cfg.prf * v
    hg = terrain(np.linspace(-half - 100.0, half + 100.0, 8001))
    tab = height_table(cfg, tr, hg.min(), hg.max())
    psi = lambda x: 2.0 * np.pi * maps_from_heights(cfg, tr, terrain(np.atleast_1d(np.asarray(x, float))), tab)[:, 0] / wl
    out = []
    for m in range(-cfg.Nrx, cfg.Nrx + 1):
        beta = (B - abs(m) * P) / B
        if m == 0 or beta <= 0:
            continue
        for x in xs:
            c = np.mean(np.exp(1j * (2.0 * np.pi * m * tauP + psi(x + m * X) - psi(x))))
            out.append((float(abs(c)), float(beta), int(m), float(x)))
    return sorted(out, reverse=True), float(X)


def stats():
    rows = json.load(open(os.path.join(AD.OUT, "all_numbers.json")))
    rows = rows if isinstance(rows, list) else list(rows.values())
    R = [r for r in rows if r.get("group") != "control"]
    cause = lambda r: "dois canais" if r["nrx"] == 2 else ("passo comum" if r["baseline"] in ("s10", "s50") else "outros")
    out = {}
    for name in ("dois canais", "passo comum", "outros"):
        S = [r for r in R if cause(r) == name]
        far = [r for r in S if r["joint_line"] > r["ref"] + 10.0]
        vis = [r for r in far if r["joint_line"] > -30.0]
        blk = [r for r in S if r["joint_line"] > r["joint_occ"] + 3.0]
        out[name] = dict(runs=len(S), far=len(far), visible=len(vis), worse_than_blocks=len(blk),
                         visible_list=sorted(f"{r['scene']}__{r['baseline']}" for r in vis),
                         worse_than_blocks_list=sorted(f"{r['scene']}__{r['baseline']} {r['joint_line']:.1f}/{r['joint_occ']:.1f}/{r['ref']:.1f}" for r in blk))
        print(name, {k: v for k, v in out[name].items() if not k.endswith("list")})
    s50 = [r for r in R if r["baseline"] == "s50"]; s10 = [r for r in R if r["baseline"] == "s10"]
    for nm, S in (("s50", s50), ("s10", s10)):
        print(nm, len(S), "longe da ref:", sum(r["joint_line"] > r["ref"] + 10 for r in S), "visíveis:", sum(r["joint_line"] > max(-30, r["ref"] + 10) for r in S))
    return out, {(r["scene"], r["baseline"]): r for r in R}


def _one(job):
    """O acoplamento de uma rodada qualquer (cena, linhas de base), sem simular nada: só a geometria e o DEM."""
    from dataclasses import replace
    import sata1d as S
    from sata1d.config import ArrayGeometry
    sc, tag = job
    c = AD.SCENES[sc][1]()
    bxt, _ = AD.baseline(tag)
    cfg = S.make_config(list(zip(c["xs"], c["hs"])), nrx=len(bxt), bxt_max=0.0, amplitudes=c["amps"], tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy()))
    tr = S.build_platform_tracks(cfg)
    cp, X = coupling(cfg, tr, c["terrain"], c["xs"])
    return f"{sc}__{tag}", cp[:4]


def all_couplings():
    """max |c_m| nos alvos para as 680 rodadas da auditoria -> acoplamento_todas.json (retomável)."""
    from multiprocessing import Pool
    _, allr = stats()
    out_file = os.path.join(HERE, "acoplamento_todas.json")
    out = json.load(open(out_file)) if os.path.exists(out_file) else {}
    jobs = [k for k in allr if f"{k[0]}__{k[1]}" not in out]
    print(len(jobs), "rodadas a calcular", flush=True)
    with Pool(4) as pool:
        for n, (key, cp) in enumerate(pool.imap_unordered(_one, jobs)):
            r = allr[tuple(key.split("__"))]
            out[key] = dict(c=cp, line=r["joint_line"], ref=r["ref"], blocks=r["joint_occ"], none=r["none"], nrx=r["nrx"],
                            baseline=r["baseline"], n_targets=r["n_targets"])
            if n % 40 == 39:
                json.dump(out, open(out_file, "w"))
                print(len(out), flush=True)
    json.dump(out, open(out_file, "w"))
    print("feito:", len(out), flush=True)


def main():
    if "--all" in sys.argv:
        return all_couplings()
    st, allr = stats()
    if "--stats" in sys.argv:
        for k, v in st.items():
            print(k, "\n   visíveis:", v["visible_list"], "\n   pior que os blocos:", v["worse_than_blocks_list"])
        return
    os.makedirs(OUT, exist_ok=True)
    res_file = os.path.join(HERE, "nao_resolve.json")
    done = json.load(open(res_file)) if os.path.exists(res_file) else {}
    for causa, sc, tag in CASES:
        key = f"{sc}__{tag}"
        png = os.path.join(OUT, key + ".png")
        if key in done and os.path.exists(png):
            continue
        res, d = AD.compute(sc, tag)
        AD.page(sc, tag, res, d, png)
        cp, X = coupling(d["cfg"], d["tr"], d["c"]["terrain"], d["c"]["xs"])
        same = {t: r["joint_line"] for (s, t), r in allr.items() if s == sc}       # a mesma cena com as outras linhas de base
        done[key] = dict(causa=causa, res={k: res[k] for k in ("ref", "none", "whole", "sub", "joint_occ", "joint_line", "bxt", "nrx",
                                                                 "n_targets", "label", "max_slope_deg", "line_unexplained_dB")},
                         X=X, coupling=cp[:6], same_scene=same, ref_same_scene={t: r["ref"] for (s, t), r in allr.items() if s == sc})
        json.dump(done, open(res_file, "w"), indent=1)
        print(f"{key:34s} line {res['joint_line']:6.1f} ref {res['ref']:6.1f} | maior |c_m| {cp[0][0]:.2f} (m = {cp[0][2]}, beta {cp[0][1]:.2f})", flush=True)
    json.dump(dict(done, _stats=st), open(res_file, "w"), indent=1)


if __name__ == "__main__":
    main()
