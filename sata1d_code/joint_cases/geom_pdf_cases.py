"""
The six scenes of pages 21-24 of 'SATA 1D: Algorithm, Implementation and Verification' (RA-061, PW-007, CL-010, ME-005,
SK-002, RH-005), run with SATA 1D (whole band and per sub-band) and with the final joint (whole line), every method
reading only the DEM. Same computation and page layout as sweep_final.py (version 'joint_sata').

Differences from the PDF: channel 0 is the transmitter (bxt = [0, 27.0, 4.1, 1.7] m, the 'u100' set: the same random
draw as the PDF's U(0, 100) = [63.7, 27.0, 4.1, 1.7] with channel 0 moved onto the transmitter), and the DEM is the
terrain through the targets (linear between them, flat beyond), the same for every method. Heights of PW-007 and RH-005
read from the PDF figures.

    python joint_cases/geom_pdf_cases.py          # from sata1d_code
Output: joint_cases/geom_pdf_cases/<scene>.png, numbers.json, geom_pdf_cases.pdf
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import audit_dem as AD
import sweep_final as SF

OUT = os.path.join(HERE, "geom_pdf_cases")
X9, X5 = np.arange(-800.0, 801.0, 200.0), np.arange(-400.0, 401.0, 200.0)
SC = {   # id -> (xs, hs, description)
    "RA-061": (X9, np.tan(np.radians(15.0)) * (X9 + 800.0), "Uniform ramp of 15 deg (as RA-061)"),
    "PW-007": (X9, np.array([0, -168, -203, -317, -319, -338, -331, -404, -572.0]), "Piecewise ramp, random slopes up to 40 deg (as PW-007)"),
    "CL-010": (X9, np.array([0, 0, 0, 0, 100, 100, 100, 100, 100.0]), "Cliff of 100 m (as CL-010)"),
    "ME-005": (X9, np.array([0, 500, 500, 500, 500, 500, 500, 500, 0.0]), "Mesa of 500 m (as ME-005)"),
    "SK-002": (X5, np.array([0, 0, 500, 0, 0.0]), "Spike: middle target 500 m up (as SK-002)"),
    "RH-005": (X5, np.array([0, 656, 27, 507, 647.0]), "Random heights U(0, 1000 m) (as RH-005)"),
}
for k, (xs, hs, desc) in SC.items():
    terr = (lambda x, xs=xs, hs=hs: np.interp(np.asarray(x, float), xs, hs))
    AD.SCENES[k] = ("pdf", (lambda terr=terr, desc=desc, xs=xs: AD._on(terr, desc + ", DEM linear between the targets, flat beyond", xs)))

TAG = "u100"


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, pngs = {}, []
    for k in SC:
        res, d = SF.compute(k, TAG)
        p = os.path.join(OUT, f"{k}.png")
        SF.page(res, d, p, "joint_sata")
        pngs.append(p)
        rows[k] = {m: res[m] for m in ("ref", "none", "whole", "sub", "joint_occ", "joint_line")}
        rows[k]["unexplained_dB"] = res["line_unexplained_dB"]
        print(f"{k}: ref {res['ref']:6.1f} | no SATA {res['none']:6.1f} | SATA whole {res['whole']:6.1f} | SATA sub {res['sub']:6.1f} | "
              f"joint blocks {res['joint_occ']:6.1f} | joint final {res['joint_line']:6.1f}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "numbers.json"), "w"), indent=1)
    from joint_sweep_summary import bundle
    print("wrote", bundle(pngs, os.path.join(OUT, "geom_pdf_cases.pdf"), [(k, i, 0) for i, k in enumerate(SC)]))


if __name__ == "__main__":
    main()
