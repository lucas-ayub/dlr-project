"""Nível que o SATA por canal deixa, previsto só com a geometria e o DEM (endereços das dobras), para as 680 rodadas.
Para cada alvo: [G^-1 diag(e^{j eps}) G] integrado na banda; a previsão da rodada é a do pior alvo. -> prev_sata.json"""
import os, sys, json, time
import numpy as np
from dataclasses import replace
HERE = os.path.dirname(os.path.abspath(__file__))
JC = os.environ.get("JOINT_CASES", os.path.dirname(HERE))          # joint_cases (a pasta acima desta)
sys.path.insert(0, JC); sys.path.insert(0, os.path.dirname(JC))
import audit_dem as AD


def dados(nome):
    """Arquivo da auditoria: nesta pasta, se houver uma cópia, senão em joint_cases/audit_dem/."""
    p = os.path.join(HERE, nome)
    return p if os.path.exists(p) else os.path.join(JC, "audit_dem", nome)

import sata1d as S
from sata1d.config import ArrayGeometry
from sata1d.joint import reference_coefficients
from sata1d.joint_line import height_table

def predict_run(sc, tag):
    c = AD.SCENES[sc][1]()
    xs, terrain = np.asarray(c["xs"], float), c["terrain"]
    bxt, _ = AD.baseline(tag)
    cfg = S.make_config(list(zip(c["xs"], c["hs"])), nrx=len(bxt), bxt_max=0.0, amplitudes=c["amps"], tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy()))
    tr = S.build_platform_tracks(cfg)
    Nrx, Nsb, P, wl, v, r0 = cfg.Nrx, cfg.Nsb, cfg.PRF_op, cfg.system.wl, cfg.system.vs, cfg.scene.r0
    B = 4.0 * v * np.sin(cfg.theta_tx / 2.0) / wl
    coef = reference_coefficients(cfg, tr)
    nb = 200
    phi = (np.arange(nb) + 0.5) * P / nb
    f = -Nsb * P / 2.0 + np.mod(phi + Nsb * P / 2.0, P)[None, :] + (np.arange(Nsb) * P)[:, None]
    a = lambda ff: r0 * np.tan(np.arcsin(np.clip(wl * ff / (2 * v), -1, 1)))
    G = np.zeros((nb, Nrx, Nsb), complex)
    for i in range(Nrx):
        C0, C1, C2, Dt = coef[i]
        G[:, i, :] = np.exp(-2j * np.pi * (C0 / wl + (-C1 + Dt) * f.T + C2 * wl * f.T ** 2))
    Gi = np.linalg.inv(G)
    hh = terrain(np.linspace(-60e3, 60e3, 24001)); hg, tab = height_table(cfg, tr, hh.min(), hh.max())
    pol = [np.polyfit(hg, tab[:, i], 3) for i in range(Nrx)]
    fbase = np.mod(phi + P / 2, P) - P / 2
    out = {}
    for mode in ("whole", "sub"):
        best = 0.0
        for xp in xs:
            hp = float(terrain(np.array([xp]))[0])
            for d in (-2, -1, 1, 2):
                acc = 0j
                for q in range(Nsb):
                    qd = q + d
                    if not 0 <= qd < Nsb:
                        continue
                    mask = (np.abs(f[q]) <= B / 2) & (np.abs(f[qd]) <= cfg.abw / 2)
                    if not mask.any():
                        continue
                    xread = xp - a(f[q]) + a(fbase if mode == "whole" else f[qd])
                    hr = terrain(xread)
                    L = np.zeros(nb, complex)
                    for i in range(Nrx):
                        eps = 2 * np.pi / wl * (np.polyval(pol[i], hr) - np.polyval(pol[i], hp))
                        L += Gi[:, qd, i] * np.exp(1j * eps) * G[:, i, q]
                    acc += (L * mask).sum() * (P / nb)
                best = max(best, abs(acc) / B)
        out[mode] = float(20 * np.log10(best + 1e-12))
    return out

if __name__ == "__main__":
    rows = [r for r in json.load(open(dados("all_numbers.json"))) if r["group"] != "control"]
    fo = os.path.join(HERE, "prev_sata.json")
    res = json.load(open(fo)) if os.path.exists(fo) else {}
    t0 = time.time()
    for n, r in enumerate(rows):
        k = f"{r['scene']}__{r['baseline']}"
        if k in res:
            continue
        res[k] = dict(predict_run(r["scene"], r["baseline"]), sim_whole=r["whole"], sim_sub=r["sub"], group=r["group"], nrx=r["nrx"], n_targets=r["n_targets"])
        if n % 40 == 0:
            json.dump(res, open(fo, "w")); print(n, k, res[k], f"{time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(fo, "w"), indent=0)
    for mode in ("whole", "sub"):
        for g in ("one", "few", "many"):
            d = np.array([v[f"sim_{mode}"] - v[mode] for v in res.values() if v["group"] == g])
            print(mode, g, len(d), f"mean {d.mean():+.2f} median {np.median(d):+.2f} rms {np.sqrt((d**2).mean()):.2f} max|.| {np.abs(d).max():.1f}")
