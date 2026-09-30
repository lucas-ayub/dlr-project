"""
SATA 1D sweep, v4: 3 distances x 3 height scales, reference = ideal monostatic signal.
  distance: 200 m (small), 2.6 km (normal), 10 km (large); N <= 5 beyond 200 m
  heights : scale 0.1 (small), 1 (normal), 13 (large) of the base scene
            ramp base h_j = j 200 tan(alpha); random heights U(0, hmax); piecewise slopes over 200 m
Metric: ambiguity peak = largest |s| within +-20 ms of x_j/v + m 0.81 s (m = +-1, +-2),
        outside +-20 ms of every main lobe, dB re. the reference peak.
Run:  python sweep_v4.py [--workers 2]
"""
import argparse
import csv
import os
import time
from dataclasses import replace

import numpy as np

import sata1d as S
import run_geometry_sweep as G

OUT = "sweep_v4"
DIST = {"dS": 200.0, "dN": 2600.0, "dL": 10000.0}
HSCALE = {"hS": 0.1, "hN": 1.0, "hL": 13.0}
FAMS = {"ramp": "RA", "random_heights": "RH", "piecewise": "PW"}


def simulate(case):
    cfg = G.make_cfg(case, G.BXT_MAX)
    tr = S.build_platform_tracks(cfg)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    ch = S.generate_channels(cfg, tr)
    rec = {"ref": S.generate_reference(cfg, tr),
           "none": S.reconstruct(cfg, tr, ch),
           "whole": S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=G.SATA_OSF)),
           "sub": S.reconstruct_subband(cfg, tr, ch, sata_osf=G.SATA_OSF)}
    H = np.conj(np.fft.fft(sref))
    return cfg, {k: np.roll(np.fft.ifft(np.fft.fft(v) * H), cfg.Na // 2) for k, v in rec.items()}


def job(args):
    case, cid, npz, tag = args
    cfg, foc = simulate(case)
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    pk = np.abs(foc["ref"]).max()
    dt = G.ambiguity_times(cfg)
    m = {f"ambpk_db_{k}": G.ambiguity_level(t, np.abs(v), pk, case.xs, cfg.system.vs, dt) for k, v in foc.items()}
    xs, dys, hs = G.scene_coordinates(case.xs, case.heights, case.offsets)
    np.savez_compressed(npz, t=t, xs=xs, dys=dys, hs=hs, bxt=cfg.array.bxt,
                        **{f"env_{k}": (np.abs(v) / pk).astype(np.float32) for k, v in foc.items()})
    row = {"config": tag, "case_id": cid, "family": case.family, "slug": case.slug, "N": case.N,
           "spacing": case.spacing, "dh_span": float(np.ptp(case.heights)),
           "heights": " ".join(f"{h:.2f}" for h in case.heights), "npz": npz}
    row.update({k: round(float(v), 3) for k, v in m.items()})
    return row


def cases(fam, spacing, scale):
    out = []
    if fam == "ramp":
        for N in G.N_TARGETS:
            if spacing > 200 and N > 5:
                continue
            for a in G.ALPHAS_DEG:
                out.append(G.Case("ramp", f"N{N}_a{a:g}", "", G.heights_from_slopes([a] * (N - 1), 200.0 * scale),
                                  spacing, dict(alpha=a)))
    else:
        for c in G.build_cases([fam]):
            if spacing > 200 and c.N != 5:
                continue
            out.append(replace(c, heights=c.heights * scale, spacing=spacing))
    return out


def all_jobs():
    jobs = []
    for fam, pre in FAMS.items():
        for dk, sp in DIST.items():
            for hk, sc in HSCALE.items():
                tag = f"{pre}_{dk}_{hk}"
                d = os.path.join(OUT, tag); os.makedirs(d, exist_ok=True)
                for i, c in enumerate(cases(fam, sp, sc), 1):
                    cid = f"{pre}-{dk[1]}{hk[1]}-{i:03d}"
                    jobs.append((c, cid, os.path.join(d, f"{cid}_{c.slug}.npz"), tag))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); a = ap.parse_args()
    jobs = all_jobs()
    print(len(jobs), "cases", flush=True)
    from multiprocessing import Pool
    t0, rows = time.time(), []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap(job, jobs, chunksize=1), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)
