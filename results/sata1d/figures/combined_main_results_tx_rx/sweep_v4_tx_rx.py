"""
The distance x height sweep of sweep_v4 (1053 cases: along-track ramps and the random geometries, same
cases and ids as sweep_csv/sweep_distance_height_results.csv) with channel 0 on the transmitter.
Resumable, in pieces of limited time (several workers may share the folder):

    python sweep_v4_tx_rx.py --worker 0/4 --time-budget 140     # repeat until it prints "all done"
    python sweep_v4_tx_rx.py --csv                               # rows -> sweep_distance_height_results.csv (here)

(PYTHONPATH must contain sata1d_code.)
"""
import argparse
import csv
import glob
import json
import os
import tempfile
import time

import tx_rx
tx_rx.enable()
import numpy as np
import sweep_v4 as V

HERE = os.path.dirname(os.path.abspath(__file__))
V.FAMS = {"ramp": "RA", "random_heights": "RH", "piecewise": "PW", "zigzag": "ZI", "steepening": "ST",
          "flattening": "FL", "cliff": "CL", "spike": "SK", "mesa": "ME"}          # the families of the CSV
V.OUT = os.path.join(tempfile.gettempdir(), "sweep_v4_tx_rx")                  # sweep_v4 makes its folders there
np.savez_compressed = lambda *a, **k: None                                    # the envelopes are not kept, only the metrics
FIELDS = ["config", "case_id", "family", "slug", "N", "spacing", "dh_span", "heights", "npz",
          "ambpk_db_ref", "ambpk_db_none", "ambpk_db_whole", "ambpk_db_sub"]


def rows_done():
    rows = {}
    for path in glob.glob(os.path.join(HERE, "sweep_rows_w*.jsonl")):
        for line in open(path):
            if line.strip():
                r = json.loads(line); rows[r["case_id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="0/1"); ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()
    jobs = V.all_jobs()
    done = rows_done()
    if a.csv:
        missing = [j[1] for j in jobs if j[1] not in done]
        if missing:
            raise SystemExit(f"{len(missing)} cases missing, e.g. {missing[:3]}")
        with open(os.path.join(HERE, "sweep_distance_height_results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
            w.writerows({k: "" if k == "npz" else done[j[1]][k] for k in FIELDS} for j in jobs)     # no envelopes kept
        print("wrote", len(jobs), "rows")
        return
    i, n = (int(v) for v in a.worker.split("/"))
    t0 = time.time()
    with open(os.path.join(HERE, f"sweep_rows_w{i}.jsonl"), "a") as fh:
        for job in jobs[i::n]:
            if job[1] in done:
                continue
            if a.time_budget is not None and time.time() - t0 > a.time_budget:
                print(f"worker {i}: time budget reached, {sum(j[1] not in done for j in jobs[i::n])} cases left", flush=True)
                return
            r = V.job(job); done[r["case_id"]] = r
            fh.write(json.dumps(r) + "\n"); fh.flush()
    print(f"worker {i}: all done", flush=True)


if __name__ == "__main__":
    main()
