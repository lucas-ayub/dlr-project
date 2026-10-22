"""
Case 32 (plateau 500 m high, 2.4 km wide, 9 targets 500 m apart) with b_xt = [0, 100, 200, 300] and with the same
values on channels 1, 2, 3 in three other orders (no arithmetic progression in channel order). One sweep page
(4 panels) per order in joint_cases/ordem_canais/<tag>/C32.png and a PDF with the four pages.
    SATA_LUT_CACHE=/tmp/lutcache python joint_cases/ordem_canais.py --budget 140      # from sata1d_code
Already computed orders are skipped; --budget stops starting new runs after that many seconds.
"""
import argparse
import os
import sys
import time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import run_joint_sweep as W
import joint_sweep_summary as JS

CASE = "C32"
ORDERS = [("s100", [0, 100, 200, 300], "steps of 100 m"),
          ("o312", [0, 300, 100, 200], "same values, other order"),
          ("o231", [0, 200, 300, 100], "same values, other order"),
          ("o132", [0, 100, 300, 200], "same values, other order")]
OUT = os.path.join(HERE, "ordem_canais")
for _t, _b, _k in ORDERS:
    W.BXT.setdefault(_t, np.asarray(_b, float))
    W.BKIND.setdefault(_t, _k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1e9, help="seconds after which no new run is started")
    ap.add_argument("--pdf", default=os.path.join(OUT, "C32_ordem_dos_canais.pdf"))
    a = ap.parse_args()
    t0 = time.time()
    for t, _, _ in ORDERS:
        if os.path.exists(os.path.join(OUT, t, f"{CASE}.json")):
            continue
        if time.time() - t0 > a.budget:
            print("budget reached, run again", flush=True)
            return
        r = W.run(CASE, t, out=OUT)
        print(t, {k: round(r[k], 1) for k in ("ref", "none", "whole", "sub", "joint_full", "joint_occ")},
              f"{r['seconds']:.0f} s", flush=True)
    pngs = [os.path.join(OUT, t, f"{CASE}.png") for t, _, _ in ORDERS]
    outline = [(f"b_xt = {W.bxt_text(b)}", k, 0) for k, (t, b, _) in enumerate(ORDERS)]
    print("wrote", JS.bundle(pngs, a.pdf, outline), flush=True)


if __name__ == "__main__":
    main()
