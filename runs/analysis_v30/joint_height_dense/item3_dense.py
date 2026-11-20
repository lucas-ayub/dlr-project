"""Item 3: scenes full of point targets. N targets (unit amplitude, random phase, random positions in |x| <= 10 km,
on the terrain surface) for N = 1, 5, 20, 50, 100, 200 and three seeds, on simulated terrains (isolated 1000 m
mountain, random mountains, fractal alpine relief) and on SRTM lines (western foothills = slide 25, eastern
escarpment = cut 6, High Sierra = cut 3). Positions stay within +-10 km so that every ambiguity (+-2X = +-12.5 km)
stays on the processed line (+-23.6 km).
Metrics per run (README): image error of the focused complex line vs the reference (whole line, same band), highest
ambiguity (talk metric; only where an ambiguity window is not inside +-154 m of a target, else NaN), weakest peak.
    python3 -I item3_dense.py [terrain ...]       (cache/dense_<terrain>_N<n>_s<seed>.npz, table cache/item3.json)"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import CACHE, relative, gauss_mountain, synth_terrain, real_terrain, simulate, amb_new, nmse_db

NS = [1, 5, 20, 50, 100, 200]
SEEDS = [0, 1, 2]
LIM = 10000.0
TERRAINS = {"mountain": "isolated mountain 1000 m", "mountains": "random mountains", "alpine": "fractal alpine relief",
            "foothills": "SRTM western foothills", "escarpment": "SRTM eastern escarpment",
            "highsierra": "SRTM High Sierra"}
KEEP = {(n, 0) for n in (5, 20, 200)}          # lines kept for the example figures


def terrain(name):
    if name == "mountain":
        return relative(gauss_mountain(1000.0))
    if name in ("mountains", "alpine"):
        return relative(synth_terrain(name))
    return real_terrain(name)


def scene(n, seed):
    g = np.random.default_rng(1000 * n + seed)
    xs = np.sort(g.uniform(-LIM, LIM, n))
    amps = np.exp(2j * np.pi * g.uniform(0, 1, n))
    return xs, amps


def one(name, ter, n, seed):
    p = os.path.join(CACHE, f"dense_{name}_N{n:03d}_s{seed}.npz")
    if os.path.exists(p):
        return json.loads(str(np.load(p, allow_pickle=True)["metrics"]))
    xs, amps = scene(n, seed)
    t0 = time.time()
    r = simulate(xs, ter, amps=amps)
    m = {}
    for k in ("ref", "std", "joint"):
        try:
            a, b, c = amb_new(r, k)
        except ValueError:          # no ambiguity window clear of the targets
            a, b, c = np.nan, np.nan, np.nan
        m[k] = dict(amb=a, floor=b, peak_loss=c, err=(nmse_db(r, k) if k != "ref" else None))
    m["sec"] = time.time() - t0
    m["iterations"] = r["iterations"]
    out = dict(metrics=json.dumps(m), xs=xs, hs=r["hs"], X=r["X"])
    if (n, seed) in KEEP:
        out.update(t=r["t"].astype(np.float32), vs=r["vs"], ref=r["ref"].astype(np.complex64),
                   std=r["std"].astype(np.complex64), joint=r["joint"].astype(np.complex64),
                   tgrid=np.linspace(-23600, 23600, 4721), hgrid=ter(np.linspace(-23600, 23600, 4721)))
    np.savez_compressed(p, **out)
    print(f"{name:11s} N={n:3d} s{seed}  err std {m['std']['err']:6.1f} joint {m['joint']['err']:6.1f} dB | amb std "
          f"{m['std']['amb']:6.1f} joint {m['joint']['amb']:6.1f} ref {m['ref']['amb']:6.1f} | {m['sec']:.1f}s",
          flush=True)
    return m


if __name__ == "__main__":
    names = sys.argv[1:] or list(TERRAINS)
    for name in names:
        ter, href = terrain(name)
        for n in NS:
            for s in SEEDS:
                one(name, ter, n, s)
    rows = []
    for name in TERRAINS:
        for n in NS:
            for s in SEEDS:
                p = os.path.join(CACHE, f"dense_{name}_N{n:03d}_s{s}.npz")
                if os.path.exists(p):
                    rows.append(dict(terrain=name, N=n, seed=s,
                                     metrics=json.loads(str(np.load(p, allow_pickle=True)["metrics"]))))
    json.dump(rows, open(os.path.join(CACHE, "item3.json"), "w"), indent=1)
