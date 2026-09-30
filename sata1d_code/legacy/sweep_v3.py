"""
SATA 1D sweep, v3: topographic ambiguity measured against the flat-terrain reconstruction.

For every case:
  flat  : the same reconstruction with b_xt = 0 (the heights then leave no phase error, so this is
          the reconstruction the scene would give without topography; it carries the reconstruction's
          own band-edge skirts and floor, which do not depend on the heights)
  none, whole, sub : reconstruction without SATA, SATA whole band, SATA per sub-band (b_xt ~ U(0, 100))
Metrics (dB re. the peak of the flat reconstruction):
  ambt_db_<k>  topographic ambiguity: largest |s_k - s_flat| within +-20 ms of x_j/v + m dt_amb
               (m = +-1, +-2), outside +-20 ms of every main lobe
  ambpk_db_<k> the same windows read on |s_k| (previous metric)
  floor_db     the same windows read on |s_flat| (floor of the reconstruction, independent of heights)

Ramp configurations (heights normal = j 200 tan(alpha), large = j 2600 tan(alpha)):
  A 200 m normal | B 2.6 km normal | C 200 m large | D 2.6 km large
  E 5.2 km normal | F 5.2 km large | G 10 km normal | H 10 km large
Random heights / piecewise ramps: 200 m (all N), 2.6, 5.2 and 10 km (N = 5), same heights.
Spacing sweep: 200 m ... 12.5 km for three N = 5 scenes.

Run:  python sweep_v3.py [--workers 2] [--only A B ...] [--no-spacing]
"""
import argparse
import csv
import os
import time
from dataclasses import replace

import numpy as np

import sata1d as S
import run_geometry_sweep as G

OUT = "sweep_v3"
X_KM = 6.2287
RAMP_CONFIGS = {"A": (200.0, 200.0), "B": (2600.0, 200.0), "C": (200.0, 2600.0), "D": (2600.0, 2600.0),
                "E": (5200.0, 200.0), "F": (5200.0, 2600.0), "G": (10000.0, 200.0), "H": (10000.0, 2600.0)}
OTHER_SPACINGS = [200.0, 2600.0, 5200.0, 10000.0]
SPACINGS = [200, 400, 600, 800, 1000, 1200, 1600, 2000, 2600, 3200, 3800, 4400, 5200, 6200, 7400, 8800,
            10000, 11300, 12500]
KEYS = ("flat", "none", "whole", "sub")


def simulate(case):
    cfg = G.make_cfg(case, G.BXT_MAX)
    tr = S.build_platform_tracks(cfg)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    ch = S.generate_channels(cfg, tr)
    cfg0 = G.make_cfg(case, 0.0)
    tr0 = S.build_platform_tracks(cfg0)
    rec = {"flat": S.reconstruct(cfg0, tr0, S.generate_channels(cfg0, tr0)),
           "none": S.reconstruct(cfg, tr, ch),
           "whole": S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=G.SATA_OSF)),
           "sub": S.reconstruct_subband(cfg, tr, ch, sata_osf=G.SATA_OSF)}
    H = np.conj(np.fft.fft(sref))
    return cfg, {k: np.roll(np.fft.ifft(np.fft.fft(v) * H), cfg.Na // 2) for k, v in rec.items()}


def metrics(cfg, foc, case):
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    pk = np.abs(foc["flat"]).max()
    dt = G.ambiguity_times(cfg)
    L = lambda x: G.ambiguity_level(t, np.abs(x), pk, case.xs, cfg.system.vs, dt)
    out = {"floor_db": L(foc["flat"])}
    for k in ("none", "whole", "sub"):
        out[f"ambt_db_{k}"] = L(foc[k] - foc["flat"])
        out[f"ambpk_db_{k}"] = L(foc[k])
    return out, t, pk


def job(args):
    case, cid, npz, tag = args
    cfg, foc = simulate(case)
    m, t, pk = metrics(cfg, foc, case)
    xs, dys, hs = G.scene_coordinates(case.xs, case.heights, case.offsets)
    env = {f"env_{k}": (np.abs(v) / pk).astype(np.float32) for k, v in foc.items()}
    err = {f"err_{k}": (np.abs(foc[k] - foc["flat"]) / pk).astype(np.float32) for k in ("none", "whole", "sub")}
    np.savez_compressed(npz, t=t, xs=xs, dys=dys, hs=hs, bxt=cfg.array.bxt, **env, **err)
    row = {"config": tag, "case_id": cid, "family": case.family, "slug": case.slug, "N": case.N,
           "spacing": case.spacing, "dh_span": float(np.ptp(case.heights)),
           "heights": " ".join(f"{h:.2f}" for h in case.heights), "npz": npz}
    row.update({k: round(float(v), 3) for k, v in m.items()})
    return row


def ramp_cases(spacing, step):
    out = []
    for N in G.N_TARGETS:
        if spacing > 200 and N > 5:
            continue
        for a in G.ALPHAS_DEG:
            out.append(G.Case("ramp", f"N{N}_a{a:g}", "", G.heights_from_slopes([a] * (N - 1), step),
                              spacing, dict(alpha=a, step=step)))
    return out


def other_cases(family, spacing):
    out = []
    for c in G.build_cases([family]):
        if spacing > 200 and c.N != 5:
            continue
        out.append(replace(c, spacing=spacing))
    return out


def all_jobs(only):
    jobs = []
    for tag, (sp, step) in RAMP_CONFIGS.items():
        if only and tag not in only:
            continue
        d = os.path.join(OUT, f"ramp_{tag}"); os.makedirs(d, exist_ok=True)
        for i, c in enumerate(ramp_cases(sp, step), 1):
            jobs.append((c, f"{tag}-{i:03d}", os.path.join(d, f"{tag}-{i:03d}_{c.slug}.npz"), tag))
    for fam, pre in (("random_heights", "RH"), ("piecewise", "PW")):
        for sp in OTHER_SPACINGS:
            tag = f"{pre}{sp:.0f}"
            if only and tag not in only:
                continue
            d = os.path.join(OUT, f"{fam}_{sp:.0f}"); os.makedirs(d, exist_ok=True)
            for i, c in enumerate(other_cases(fam, sp), 1):
                jobs.append((c, f"{tag}-{i:03d}", os.path.join(d, f"{tag}-{i:03d}_{c.slug}.npz"), tag))
    return jobs


def collides(spacing, N):
    d = np.arange(1, N) * spacing / 1e3
    m = np.maximum(1, np.round(d / X_KM))
    return bool(np.any(np.abs(d - X_KM * m) < 0.63))


def spacing_jobs():
    scenes = {"ramp_n20": ("ramp", G.heights_from_slopes([20] * 4, 200.0)),
              "ramp_l20": ("ramp", G.heights_from_slopes([20] * 4, 2600.0)),
              "random500": ("random_heights",
                            [c for c in G.build_cases(["random_heights"]) if c.slug == "N5_hmax500_seed0"][0].heights)}
    jobs = []
    d = os.path.join(OUT, "spacing"); os.makedirs(d, exist_ok=True)
    for name, (fam, h) in scenes.items():
        for s in SPACINGS:
            c = G.Case(fam, f"{name}_s{s}", "", np.asarray(h, float), float(s), {})
            jobs.append((c, f"SP-{name}-{s}", os.path.join(d, f"{name}_s{s}.npz"), name))
    return jobs


def run(jobs, workers, csv_path):
    from multiprocessing import Pool
    t0, rows = time.time(), []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap(job, jobs, chunksize=1), 1):
            rows.append(r)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-spacing", action="store_true")
    ap.add_argument("--spacing-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.spacing_only:
        jobs = all_jobs(a.only)
        print(len(jobs), "cases", flush=True)
        run(jobs, a.workers, os.path.join(OUT, "results.csv" if not a.only else f"results_{'_'.join(a.only)}.csv"))
    if not a.no_spacing:
        sj = spacing_jobs()
        print(len(sj), "spacing cases", flush=True)
        run(sj, a.workers, os.path.join(OUT, "spacing.csv"))
