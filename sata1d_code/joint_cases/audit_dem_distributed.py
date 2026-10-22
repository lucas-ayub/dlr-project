"""
Distributed scenes (distributed_test.py: ~11 900 scatterers on the terrain, three dark gaps) with the joint
reconstruction over the whole line added. Uses the channels cached by distributed_test.py (DIST_CACHE); every method
reads only the DEM. Metrics as there: error-to-signal ratio and gap ratio inside |x| <= 7 km.
    python joint_cases/audit_dem_distributed.py            # from sata1d_code -> joint_cases/audit_dem/distributed.json
"""
import json
import os
import sys
import time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
from sata1d.joint import reconstruct_joint
from sata1d.joint_line import height_table, channel_maps, line_maps, reconstruct_joint_line
import distributed_test as DT
import run_joint_sweep as W

OUT = os.path.join(HERE, "audit_dem", "distributed.json")


def run(cid, btag):
    cfg, tr, terrain, label, xs, amps = DT.make(cid, btag)
    fg = os.path.join(DT.CACHE, f"gen_{cid}_{btag}.npz")
    if not os.path.exists(fg):
        return None
    z = np.load(fg); ref, ch, och = z["ref"], z["ch"], z["och"]
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    hg = terrain(np.linspace(-half - 100, half + 100, 8001))
    tab = height_table(cfg, tr, hg.min(), hg.max())
    Mch, Ml = channel_maps(cfg, tr, terrain, tab), line_maps(cfg, tr, terrain, tab)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    fr = np.fft.fftfreq(cfg.Na, 1.0 / cfg.prf)
    band = (np.abs(fr) <= cfg.abw / 2.0) * (0.54 + 0.46 * np.cos(2 * np.pi * fr / cfg.abw))
    F = lambda s: np.roll(np.fft.ifft(np.fft.fft(s) * Hf * band), cfg.Na // 2)
    t0 = time.time(); yl, info = reconstruct_joint_line(cfg, tr, ch, Ml, return_info=True); tl = time.time() - t0
    y = {"ref": F(ref), "none": F(S.reconstruct(cfg, tr, ch)), "whole": F(DT.P5.sata_whole_table(cfg, tr, ch, Mch)),
         "sub": F(DT.P5.sata_sub_table(cfg, tr, ch, Mch)),
         "joint_occ": F(reconstruct_joint(cfg, tr, ch, maps=Mch, sata_osf=W.SATA_OSF, occupied=True)),
         "joint_line": F(yl), "oracle": F(S.reconstruct(cfg, tr, och))}
    _, _, m, _ = DT.metrics(cfg, y)
    return dict(case=cid, baseline=btag, label=label, bxt=[float(b) for b in cfg.array.bxt], metrics=m, seconds_line=tl,
                line_unexplained_dB=float(20 * np.log10(info["unexplained"])))


if __name__ == "__main__":
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["case"], r["baseline"]) for r in res}
    deadline = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else 1e18
    for b in DT.BASELINES:
        for c in DT.CASES:
            if (c, b) in done or time.time() > deadline:
                continue
            r = run(c, b)
            if r is None:
                print(c, b, "no cached channels (run distributed_test.py first)"); continue
            res.append(r); json.dump(res, open(OUT, "w"), indent=1)
            m = r["metrics"]
            print(f"{c:5s} {b:7s} error/signal [dB]: " + " ".join(f"{k} {m[k]['esr']:6.1f}" for k in ("none", "whole", "sub", "joint_occ", "joint_line", "oracle"))
                  + " | gap ratio [dB]: " + " ".join(f"{k} {m[k]['gap']:6.1f}" for k in ("ref", "none", "whole", "joint_occ", "joint_line", "oracle")), flush=True)
