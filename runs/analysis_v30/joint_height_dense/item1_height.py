"""Item 1: the slide-8 case (isolated Gaussian mountain, sigma 1.5 km, five targets at x = -8, -0.55, 0.65, 2.9,
4.1 km on the surface; heights relative to the mean height of the line) for mountain heights 0 ... 3000 m.
Per height: reference, standard and joint focused lines; metrics: highest ambiguity (talk metric), highest level
anywhere off the targets, weakest true peak, 3-dB width of every target's main lobe.
    python3 -I item1_height.py        (cache: cache/height_*.npz, table: cache/item1_height.json)"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import CACHE, relative, gauss_mountain, simulate, amb_old, amb_new, width3db

XS = np.array([-8000.0, -550.0, 650.0, 2900.0, 4100.0])          # facts.md: the mountain's five targets
HEIGHTS = np.arange(0.0, 3001.0, 100.0)


def one(H):
    p = os.path.join(CACHE, f"height_{int(H):04d}.npz")
    if os.path.exists(p):
        return dict(np.load(p, allow_pickle=True))
    ter, href = relative(gauss_mountain(H))
    r = simulate(XS, ter)
    m = {}
    for k in ("ref", "std", "joint"):
        a, b, c = amb_new(r, k)
        w = [width3db(r, k, x)[0] for x in XS]
        m[k] = dict(amb=a, amb_old=amb_old(r, k), floor=b, peak_loss=c, w3db=w)
    tg = np.linspace(-23600.0, 23600.0, 4721)
    out = dict(H=H, href=href, xs=XS, hs=r["hs"], X=r["X"], t=r["t"].astype(np.float32), vs=r["vs"],
               ref=r["ref"].astype(np.complex64), std=r["std"].astype(np.complex64),
               joint=r["joint"].astype(np.complex64), tgrid=tg, hgrid=ter(tg), metrics=json.dumps(m),
               iterations=r["iterations"])
    np.savez_compressed(p, **out)
    print(f"H={H:5.0f} m  std {m['std']['amb']:6.1f}  joint {m['joint']['amb']:6.1f}  ref {m['ref']['amb']:6.1f}"
          f"  | floor joint {m['joint']['floor']:6.1f}  peak loss std {m['std']['peak_loss']:6.1f}"
          f"  joint {m['joint']['peak_loss']:5.2f}  w3db joint {np.mean(w):.2f} m", flush=True)
    return dict(np.load(p, allow_pickle=True))


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for H in HEIGHTS:
        d = one(H)
        rows.append(dict(H=float(H), href=float(d["href"]), metrics=json.loads(str(d["metrics"]))))
    json.dump(rows, open(os.path.join(CACHE, "item1_height.json"), "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s")
