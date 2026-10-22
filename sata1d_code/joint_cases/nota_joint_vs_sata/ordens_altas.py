"""Com 6 e 8 canais há ambiguidades possíveis a 3 ou mais dobras (|m| P < B), que a medida da auditoria (ordens +-1, +-2)
não olha. Este script refaz as rodadas de 6 e 8 canais e mede a maior ambiguidade também com as ordens até +-5.
-> ordens_altas.json (pode ser interrompido e retomado).   python ordens_altas.py [parte] [de quantas]"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
JC = os.environ.get("JOINT_CASES", os.path.dirname(HERE))          # joint_cases (a pasta acima desta)
sys.path.insert(0, JC); sys.path.insert(0, os.path.dirname(JC))
import audit_dem as AD
part, nparts = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
fo = os.path.join(HERE, f"ordens_altas_{part}.json" if nparts > 1 else "ordens_altas.json")
res = json.load(open(fo)) if os.path.exists(fo) else {}
jobs = [(sc, f"N{n}b{b}") for sc, (g, _) in AD.SCENES.items() if g != "control" for n in (6, 8) for b in (100, 300, 500)]
M = ["ref", "none", "whole", "sub", "joint_occ", "joint_line"]
t0 = time.time()
for j, (sc, tag) in enumerate(jobs):
    k = f"{sc}__{tag}"
    if j % nparts != part or k in res:
        continue
    r, d = AD.compute(sc, tag)
    cfg, t, pk, xs = d["cfg"], d["t"], d["pk"], d["c"]["xs"]
    dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    hi = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
    res[k] = dict(nrx=cfg.Nrx, group=r["group"], o2={m: r[m] for m in M},
                  o5={m: float(AD.ambiguity_level(t, d["env"][m], pk, xs, cfg.system.vs, dt, orders=hi)) for m in M},
                  so345={m: float(AD.ambiguity_level(t, d["env"][m], pk, xs, cfg.system.vs, dt, orders=(-5, -4, -3, 3, 4, 5))) for m in M})
    json.dump(res, open(fo, "w"))
    print(j, k, {m: (round(res[k]["o2"][m], 1), round(res[k]["o5"][m], 1)) for m in ("whole", "joint_line")}, f"{time.time() - t0:.0f}s", flush=True)
