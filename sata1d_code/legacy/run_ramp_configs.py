"""
Along-track ramps with large heights: h_j = j * 2600 m * tan(alpha) (the ramp of slope alpha over
2.6-km steps), with the targets
  - 200 m apart   (N = 2 ... 13)       -> geometry_sweep_ramp_largeh_close/
  - 2.6 km apart  (N <= 5, collisions)  -> geometry_sweep_ramp_largeh_separated/
The normal-height ramps (h_j = j * 200 m * tan(alpha)) are in geometry_sweep_v2 (200 m apart) and
geometry_sweep_separated (2.6 km apart).
Run:  python run_ramp_configs.py [--workers 2]
"""
import argparse
import csv
import os
import time

import run_geometry_sweep as G

STEP_H = 2600.0


def cases(spacing, n_max):
    out = []
    for N in G.N_TARGETS:
        if N > n_max:
            continue
        for a in G.ALPHAS_DEG:
            h = G.heights_from_slopes([a] * (N - 1), STEP_H)
            out.append(G.Case("ramp", f"N{N}_a{a:g}",
                              rf"$N={N}$, $\alpha={a:g}^\circ$, heights $j\cdot2600\tan\alpha$", h, spacing, dict(alpha=a)))
    return out


def run(out_dir, spacing, n_max, prefix, workers):
    jobs = []
    for i, c in enumerate(cases(spacing, n_max), 1):
        cid = f"{prefix}-{i:03d}"
        d = os.path.join(out_dir, "bxt100", "ramp"); os.makedirs(d, exist_ok=True)
        jobs.append((c, G.BXT_MAX, cid, os.path.join(d, f"{cid}_{c.slug}.png")))
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(workers) as pool:
        rows = pool.map(G._job, jobs, chunksize=1)
    with open(os.path.join(out_dir, "sweep_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
    print(out_dir, len(rows), f"cases in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); a = ap.parse_args()
    run("geometry_sweep_ramp_largeh_close", 200.0, 99, "RAH", a.workers)
    run("geometry_sweep_ramp_largeh_separated", 2600.0, 5, "RAHS", a.workers)
