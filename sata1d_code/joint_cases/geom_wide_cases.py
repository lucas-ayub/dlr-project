"""
Wider versions of the six terrains of pages 21-24 of 'SATA 1D: Algorithm, Implementation and Verification' (ramp,
piecewise ramp, cliff, mesa, spikes, random heights): more targets, larger and irregular azimuth spacing, spread over
several km, so that targets X = 6.2 km apart (one fold of the channel PRF) sit on different heights.
SATA 1D (whole band, per sub-band), joint in blocks and the final joint, every method reading only the DEM.
Same computation and page layout as sweep_final.py (version 'joint_sata'). Channel 0 is the transmitter.

    python joint_cases/geom_wide_cases.py [--workers 4]      # from sata1d_code; resumable
Output: joint_cases/geom_wide_cases/numbers/*.json, pages/*.png, geom_wide_cases.pdf, table.txt
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import audit_dem as AD
import sweep_final as SF

OUT = os.path.join(HERE, "geom_wide_cases")
TAGS = ["u100", "u500"]
VERSION = 2                    # 2: spikes centred on targets, labels with the real spacings
DENS = {"15": (15, 600.0, 11), "25": (25, 650.0, 12)}      # n targets, mean spacing [m], seed of the spacings


def positions(n, mean, seed):
    s = mean * np.random.default_rng(seed).uniform(0.4, 1.6, n - 1)          # irregular spacing
    x = np.concatenate([[0.0], np.cumsum(s)])
    return np.round(x - x.mean(), 1)


def shapes(W, xs):
    """Terrains h(x) for targets spread over [-W, W] (flat beyond the shapes). The spikes are centred on two targets
    (a property of the scene, not something given to the methods)."""
    c1 = xs[np.argmin(np.abs(xs + 0.4 * W))]; c2 = xs[np.argmin(np.abs(xs - 0.3 * W))]
    lin = lambda xk, hk: (lambda x: np.interp(np.asarray(x, float), xk, hk))
    kn = np.arange(-W, W + 1e-6, 1200.0)
    sl = np.tan(np.radians(np.random.default_rng(7).uniform(-25.0, 25.0, len(kn) - 1)))
    hpw = np.concatenate([[0.0], np.cumsum(sl * np.diff(kn))])
    kr = np.arange(-W, W + 1e-6, 1000.0)
    hr = np.random.default_rng(0).uniform(0.0, 1000.0, len(kr))
    return {
        "ramp": (lin([-W, W], [0.0, 1000.0]), f"ramp 0 -> 1000 m over {2 * W / 1e3:.1f} km"),
        "piecewise": (lin(kn, hpw), "piecewise ramp, random slopes up to 25 deg every 1.2 km"),
        "cliff": (lin([-W, 670.0, 730.0, W], [0.0, 0.0, 300.0, 300.0]), "cliff of 300 m at x = 0.7 km"),
        "mesa": (lin([-0.35 * W - 250, -0.35 * W, 0.35 * W, 0.35 * W + 250], [0.0, 500.0, 500.0, 0.0]),
                 f"mesa of 500 m, {0.7 * W / 1e3:.1f} km wide, flanks of 250 m"),
        "spikes": (lin([c1 - 150, c1, c1 + 150, c2 - 200, c2, c2 + 200],
                       [0.0, 500.0, 0.0, 0.0, 800.0, 0.0]), "two spikes, 500 m and 800 m"),
        "random": (lin(kr, hr), "random heights U(0, 1000 m) every 1 km, linear between"),
    }


def register():
    cases = []
    for dk, (n, mean, seed) in DENS.items():
        xs = positions(n, mean, seed)
        W = max(abs(xs.min()), xs.max()) + 300.0
        for sk, (terr, desc) in shapes(W, xs).items():
            sid = f"{sk}_{dk}"
            lab = f"{desc}, {n} targets irregularly spaced ({np.diff(xs).min():.0f} to {np.diff(xs).max():.0f} m, mean {np.diff(xs).mean():.0f} m)"
            def build(terr=terr, xs=xs, lab=lab):
                c = AD._on(terr, lab, xs); c["label"] = lab; return c
            AD.SCENES[sid] = ("wide", build)
            for tag in TAGS:
                cases.append((sid, tag))
    return cases


def _job(case):
    sid, tag = case
    pj = os.path.join(OUT, "numbers", f"{sid}__{tag}.json"); pp = os.path.join(OUT, "pages", f"{sid}__{tag}.png")
    if os.path.exists(pj) and os.path.exists(pp) and json.load(open(pj)).get("version_wide") == VERSION:
        return json.load(open(pj))
    register()
    res, d = SF.compute(sid, tag)
    SF.page(res, d, pp, "joint_sata")
    keep = {k: v for k, v in res.items() if not isinstance(v, (dict,))}
    keep["version_wide"] = VERSION
    json.dump(keep, open(pj, "w"), indent=1)
    return keep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    for s in ("numbers", "pages"):
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
    cases = register()
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(_job, cases):
            print(f"{r['scene']:14s} {r['baseline']:5s} ref {r['ref']:6.1f} none {r['none']:6.1f} whole {r['whole']:6.1f} "
                  f"sub {r['sub']:6.1f} blocks {r['joint_occ']:6.1f} joint {r['joint_line']:6.1f}", flush=True)
    rows = [json.load(open(os.path.join(OUT, "numbers", f"{s}__{t}.json"))) for s, t in cases]
    keys = ["ref", "none", "whole", "sub", "joint_occ", "joint_line"]
    lines = [f"{'case':24s} " + " ".join(f"{k:>10s}" for k in keys)]
    lines += [f"{r['scene'] + ' ' + r['baseline']:24s} " + " ".join(f"{r[k]:10.1f}" for k in keys) for r in rows]
    open(os.path.join(OUT, "table.txt"), "w").write("\n".join(lines) + "\n")
    from joint_sweep_summary import bundle
    pngs = [os.path.join(OUT, "pages", f"{s}__{t}.png") for s, t in cases]
    print("wrote", bundle(pngs, os.path.join(OUT, "geom_wide_cases.pdf"), [(f"{s} | {t}", i, 0) for i, (s, t) in enumerate(cases)]))


if __name__ == "__main__":
    main()
