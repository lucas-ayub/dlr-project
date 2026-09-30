"""Adds ambpk_db_* (peak level at the azimuth-ambiguity positions x_j/vs + m dt_amb, m = +-1, +-2)
to a sweep CSV, computed from the saved envelopes (.npz) of every case."""
import csv, glob, os, sys
import numpy as np
import sata1d as S
from run_geometry_sweep import ambiguity_level, ambiguity_times

def recompute(sweep_dir):
    cfg = S.make_config([(0.0, 0.0)])
    vs, dt = cfg.system.vs, ambiguity_times(cfg)
    rows = list(csv.DictReader(open(os.path.join(sweep_dir, "sweep_results.csv"))))
    for r in rows:
        f = glob.glob(os.path.join(sweep_dir, "bxt100", r["family"], f"{r['case_id']}_*.npz"))[0]
        z = np.load(f, allow_pickle=True)
        pk = z["env_ref"].max()
        for k in ("ref", "none", "whole", "sub", "oracle"):
            r[f"ambpk_db_{k}"] = round(ambiguity_level(z["t"], z[f"env_{k}"], pk, z["xs"], vs, dt), 3)
    with open(os.path.join(sweep_dir, "sweep_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(sweep_dir, len(rows), "cases, dt_amb = %.4f s" % dt)

if __name__ == "__main__":
    for d in sys.argv[1:]:
        recompute(d)
