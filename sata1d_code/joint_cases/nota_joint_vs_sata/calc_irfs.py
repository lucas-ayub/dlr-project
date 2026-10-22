"""Respostas (IRFs) dos casos mostrados na nota -> irfs.npz, irfs.json (nesta pasta).
Usa as cenas e os métodos de audit_dem.py (pasta joint_cases: variável JOINT_CASES, ou a pasta acima desta)."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
JC = os.environ.get("JOINT_CASES", os.path.dirname(HERE))          # joint_cases (a pasta acima desta)
sys.path.insert(0, JC); sys.path.insert(0, os.path.dirname(JC))
import audit_dem as AD
CASES = [("few_peak_5", "u500"), ("ctrl_periodX_5", "N4b100"), ("ctrl_period1p5X_5", "N4b100"), ("ctrl_periodX_5", "N3b100"),
         ("ctrl_flat500_1", "N4b300"), ("one_refpoint", "N4b100")]
out, nums = {}, {}
for sc, tag in CASES:
    t0 = time.time(); res, d = AD.compute(sc, tag)
    k = f"{sc}__{tag}"
    cfg = d["cfg"]
    for m, e in d["env"].items():
        out[f"{k}__{m}"] = (e / d["pk"]).astype(np.float32)
    out[f"{k}__t"] = d["t"]
    X = cfg.system.wl * cfg.scene.r0 * cfg.PRF_op / (2 * cfg.system.vs)
    nums[k] = dict(amb={m: res[m] for m in ("ref", "none", "whole", "sub", "joint_occ", "joint_line")}, bxt=res["bxt"], xs=[float(x) for x in d["c"]["xs"]],
                   hs=[float(h) for h in d["c"]["hs"]], X=float(X), vs=float(cfg.system.vs), label=res["label"], sec=time.time() - t0)
    print(k, {m: round(v, 1) for m, v in nums[k]["amb"].items()}, f"{time.time() - t0:.1f}s", flush=True)
np.savez_compressed(os.path.join(HERE, "irfs.npz"), **out)
json.dump(nums, open(os.path.join(HERE, "irfs.json"), "w"), indent=1)
