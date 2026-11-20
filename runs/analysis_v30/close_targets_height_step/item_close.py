"""Close point targets next to an abrupt height change.
Two unit point targets (random-free: phases 0 and pi/2) a distance d apart, on terrain that is flat except for
  'step':  a cliff of height dh between the two targets, h(x) = dh/2 (1 + tanh(2 (x - xs) / w)), w = 10 m,
  'bump':  a narrow bump of height dh between them, h(x) = dh exp(-(x - xs)^2 / (2 s^2)), s = 10 m,
  'width': the step at dh = 300 m with w = 1 m (sharper than one sample, dx = 3.84 m) and w = 50 m,
  'three': three targets at xs - d, xs, xs + d (the middle one on the cliff edge), w = 10 m.
xs = dx / 2 (between two samples of the line); the targets sit on the surface; heights relative to the mean of the
line. Default system, exact distances, no noise; reference, standard and joint as in joint_height_dense.
    python3 -I item_close.py          (cache/close_<case>_d<d>_h<dh>.npz, table cache/close.json)"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from close_common import CACHE, relative, simulate, amb_new, nmse_db, pair_metrics, local_err
from style import mismatch, B300

DX = 7689.0 / 2000.0 * 1.0          # replaced below by the exact sample spacing of the line
DS = [6.0, 12.0, 18.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]
DH = [0.0, 10.0, 50.0, 100.0, 300.0, 1000.0]
KEEP = {("step", 6.0, 300.0), ("step", 50.0, 300.0), ("step", 1000.0, 300.0), ("step", 18.0, 1000.0),
        ("bump", 50.0, 300.0), ("step", 6.0, 1000.0)}


def terrain(case, dh, xs, w=10.0):
    if case in ("step", "three", "width"):
        f = lambda x: 0.5 * dh * (1.0 + np.tanh(2.0 * (np.asarray(x, float) - xs) / w))
    else:
        f = lambda x: dh * np.exp(-0.5 * ((np.asarray(x, float) - xs) / 10.0) ** 2)
    return relative(f)


def blind(ter, xs, X):
    best = 9.0
    for x in xs:
        for m in (-2, -1, 1, 2):
            best = min(best, mismatch(B300, float(ter(x + m * X) - ter(x)), m))
    return best


def one(case, d, dh, w=10.0, dx=3.8442685):
    tag = f"close_{case}_w{w:g}_d{d:g}_h{dh:g}"
    p = os.path.join(CACHE, tag + ".npz")
    if os.path.exists(p):
        return json.loads(str(np.load(p, allow_pickle=True)["metrics"]))
    xs0 = 0.5 * dx
    xs = np.array([xs0 - d, xs0, xs0 + d]) if case == "three" else np.array([xs0 - d / 2, xs0 + d / 2])
    amps = np.exp(1j * np.pi / 2 * np.arange(xs.size))
    ter, href = terrain(case, dh, xs0, w)
    t0 = time.time()
    r = simulate(xs, ter, amps=amps)
    m = dict(case=case, d=d, dh=dh, w=w, xs=xs.tolist(), hs=r["hs"].tolist(), href=href)
    for k in ("ref", "std", "joint"):
        a, b, c = amb_new(r, k)
        pk, pos, dips = pair_metrics(r, k, xs)
        m[k] = dict(amb=a, floor=b, pk=pk, pos=pos, dip=dips,
                    err=(nmse_db(r, k) if k != "ref" else None), lerr=(local_err(r, k, xs) if k != "ref" else None))
    m["mismatch"] = blind(ter, xs, r["X"])
    m["sec"] = time.time() - t0
    out = dict(metrics=json.dumps(m))
    if (case, d, dh) in KEEP and w == 10.0:
        xc = np.linspace(xs.min() - 400, xs.max() + 400, 2001)
        out.update(t=r["t"].astype(np.float32), vs=r["vs"], ref=r["ref"].astype(np.complex64),
                   std=r["std"].astype(np.complex64), joint=r["joint"].astype(np.complex64), xc=xc, hc=ter(xc), X=r["X"])
    np.savez_compressed(p, **out)
    g = m["joint"]["amb"] - m["ref"]["amb"]
    print(f"{case:5s} w={w:4g} d={d:6g} dh={dh:6g} | gap joint {g:6.1f} std {m['std']['amb'] - m['ref']['amb']:6.1f}"
          f" | pk joint-ref {np.max(np.abs(np.subtract(m['joint']['pk'], m['ref']['pk']))):5.2f} dB"
          f" | dip ref {m['ref']['dip'][0]:6.1f} joint {m['joint']['dip'][0]:6.1f} | lerr {m['joint']['lerr']:6.1f}"
          f" | mm {m['mismatch']:.2f} | {m['sec']:.1f}s", flush=True)
    return m


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for case in ("step", "bump"):
        for dh in DH:
            for d in DS:
                rows.append(one(case, d, dh))
    for w in (1.0, 50.0):
        for d in DS:
            rows.append(one("width", d, 300.0, w=w))
    for dh in (100.0, 300.0, 1000.0):
        for d in (6.0, 18.0, 50.0, 300.0):
            rows.append(one("three", d, dh))
    json.dump(rows, open(os.path.join(CACHE, "close.json"), "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s, {len(rows)} runs")
