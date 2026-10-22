"""
Stress tests with only what a real case would have: every method (SATA whole band, SATA per sub-band, joint with
occupied parcels, joint full system) reads the same look-up table from the terrain (DEM); none knows where the targets
are. Scenes with one target, a few targets and many targets; Nrx = 2, 3, 4, 6, 8 at the same total PRF (2000 Hz, DPCA);
channel 0 on the transmitter, the other bxt random with the largest exactly bmax (baselines).
For the sweep ramps (point targets without a terrain) the DEM is the terrain through the targets, flat beyond them.
    SATA_LUT_CACHE=/tmp/lutcache python joint_cases/stress_dem.py --budget 160      # from sata1d_code, repeat
    python joint_cases/stress_dem.py --summary
Pages: joint_cases/stress_dem/b<bmax>/<scene>_N<Nrx>.png (+ .json); PDFs: stress_dem/stress_dem_{one,few,many}.pdf
"""
import argparse
import glob
import json
import os
import sys
import textwrap
import time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
from sata1d.config import ArrayGeometry
import run_joint_sweep as W
import joint_sweep_summary as JS
import sweep_5panel as P5

OUT = os.path.join(HERE, "stress_dem")
NRX = [2, 3, 4, 6, 8]
BMAX = [100.0, 300.0, 500.0]
VERSION = 1


def baselines(nrx, bmax, seed=0):
    """Channel 0 on the transmitter; the others random in (0, bmax], the largest exactly bmax."""
    u = np.random.default_rng(seed).uniform(0.1, 1.0, nrx)
    u[0] = 0.0
    return (bmax * u / u.max()).round(1)


def phases(n, seed=1):
    return np.exp(2j * np.pi * np.random.default_rng(seed).uniform(size=n))


def sweep_case(cid):
    """A case of the joint sweep; point-target ramps get the terrain through the targets (flat beyond) as DEM."""
    c = dict(W.case_info(cid))
    if c["terrain"] is None:
        xs, hs = np.asarray(c["xs"], float), np.asarray(c["hs"], float)
        c["terrain"] = lambda x, xs=xs, hs=hs: np.interp(np.asarray(x, float), xs, hs)
        c["label"] += " (DEM: terrain through the targets)"
    if c.get("old"):
        c["label"] += f" [{c['old']}]"
    return c


def one_target(name, x):
    h, desc = W.TERRAINS[name]
    xs = np.array([float(x)])
    return dict(xs=xs, hs=h(xs), terrain=h, amps=None,
                label=f"{desc}, 1 target at x = {x / 1e3:g} km (h = {float(h(xs)[0]):.0f} m)")


def many(name, half, d=250.0):
    h, desc = W.TERRAINS[name]
    xs = np.arange(-half, half + 1e-6, d)
    return dict(xs=xs, hs=h(xs), terrain=h, amps=phases(len(xs)), label=f"{desc}, {len(xs)} targets {d:g} m apart")


def ramp49(slope):
    xs = np.arange(-6000.0, 6000.0 + 1e-6, 250.0)
    hs = np.tan(np.radians(slope)) * (xs - xs[0])
    hmax = hs.max()
    terr = lambda x: np.clip(np.tan(np.radians(slope)) * (np.asarray(x, float) - xs[0]), 0.0, hmax)
    return dict(xs=xs, hs=hs, terrain=terr, amps=phases(len(xs)),
                label=f"ramp {slope:g}$^\\circ$ between two flats (h 0 -> {hmax:.0f} m), {len(xs)} targets 250 m apart")


GROUPS = {   # group -> {scene name: builder}
    "one": {
        "C20": lambda: sweep_case("C20"),                      # REF-A100-P50, target at h(0) = 0
        "C10": lambda: sweep_case("C10"),
        "mountain_top": lambda: one_target("mountain", 0.0),
        "mountain_flank": lambda: one_target("mountain", 1500.0),
        "escarpment_mid": lambda: one_target("escarpment", 0.0),
    },
    "few": {cid: (lambda cid=cid: sweep_case(cid)) for cid in
            ("C01", "C02", "C03", "C05", "C15", "C22", "C24", "C28", "C29", "C31", "C32", "C33")},
    "many": {
        "C09": lambda: sweep_case("C09"),
        "C13": lambda: sweep_case("C13"),
        "C27": lambda: sweep_case("C27"),
        "ramp08_49": lambda: ramp49(8.0),
        "hills_49": lambda: many("hills", 6000.0),
        "mountains_49": lambda: many("mountains", 6000.0),
        "escarpment_25": lambda: many("escarpment", 3000.0),
    },
}
SCENES = {k: (g, f) for g, d in GROUPS.items() for k, f in d.items()}


def config(c, nrx, bmax):
    cfg = S.make_config(list(zip(c["xs"], c["hs"])), nrx=nrx, bxt_max=0.0, amplitudes=c.get("amps"), tx_rx_channel0=True)
    return replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=baselines(nrx, bmax)))


def dem_table(sc, cfg, tr, i, terrain):
    cache = os.environ.get("SATA_LUT_CACHE", "/tmp/lutcache")
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, f"dem_{sc}_bat{cfg.array.bat[i]:.3f}_bxt{cfg.array.bxt[i]:g}_n{cfg.Na_ch}.npy")
    if os.path.exists(f):
        return np.load(f)
    m = S.terrain_lookup_table(cfg, tr, i, terrain, step=4)
    np.save(f, m)
    return m


def page(job):
    sc, n, bm = job
    grp, fn = SCENES[sc]
    c = fn()
    cfg = config(c, n, bm)
    tag = f"b{bm:g}"
    p, amb = P5.run(sc, tag, out=OUT, cfg=cfg, case=c, name=f"{sc}_N{n}", realistic=True,
                    lut_fn=lambda cfg_, tr_, i, terrain, cid: dem_table(sc, cfg_, tr_, i, terrain),
                    kind=f"random, largest {bm:g} m, $N_{{rx}}$ = {n}")
    json.dump(dict(amb, scene=sc, group=grp, nrx=n, bmax=bm, version=VERSION, bxt=cfg.array.bxt.tolist(),
                   n_targets=len(c["xs"]), label=c["label"]),
              open(os.path.join(OUT, tag, f"{sc}_N{n}.json"), "w"), indent=1)
    return job


LINES = [("ref", "Reference", dict(color="tab:green", marker="o", lw=2)),
         ("none", "no SATA", dict(color="tab:red", marker="s", ls="--")),
         ("whole", "SATA whole band (DEM)", dict(color="tab:blue", marker="^", ls=":")),
         ("sub", "SATA per sub-band (DEM)", dict(color="tab:orange", marker="v", ls="-.")),
         ("joint_full", "joint, full system (DEM)", dict(color="tab:pink", marker="d", ls="--")),
         ("joint_occ", "joint, occupied parcels (DEM)", dict(color="tab:purple", marker="o", lw=2))]


def summary_pages(res, scenes, tag):
    pngs = []
    for k0 in range(0, len(scenes), 6):
        chunk = scenes[k0:k0 + 6]
        fig, axs = plt.subplots(len(BMAX), len(chunk), figsize=(4.4 * len(chunk) + 1, 12.5), sharex=True, sharey=True,
                                squeeze=False)
        for r_, bm in enumerate(BMAX):
            for k, sc in enumerate(chunk):
                ax = axs[r_, k]
                ns = [n for n in NRX if (sc, bm, n) in res]
                for key, lab, st in LINES:
                    if ns:
                        ax.plot(ns, [res[(sc, bm, n)][key] for n in ns], label=lab, **st)
                ax.set_xticks(NRX); ax.grid(alpha=0.3); ax.set_ylim(-80, 10)
                if ns:
                    ax.set_title(textwrap.fill(res[(sc, bm, ns[0])]["label"], 44) + f"\nlargest $b_{{xt}}$ = {bm:g} m", fontsize=8.5)
                if r_ == len(BMAX) - 1:
                    ax.set_xlabel("number of channels $N_{rx}$")
                if k == 0:
                    ax.set_ylabel("highest ambiguity [dB]")
        axs[0, 0].legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        p = os.path.join(OUT, f"summary_{tag}_{k0 // 6}.png"); fig.savefig(p, dpi=75); plt.close(fig)
        pngs.append(p)
    return pngs


def summary():
    res = {}
    for f in glob.glob(os.path.join(OUT, "b*", "*.json")):
        r = json.load(open(f))
        res[(r["scene"], r["bmax"], r["nrx"])] = r
    out = []
    for grp, d in GROUPS.items():
        scenes = list(d)
        pngs = summary_pages(res, scenes, grp)
        outline = [(f"Summary {i + 1}", i, 0) for i in range(len(pngs))]
        for sc in scenes:
            for bm in BMAX:
                for n in NRX:
                    if (sc, bm, n) in res:
                        outline.append((f"{sc} | largest bxt {bm:g} m | Nrx = {n}", len(pngs), 0))
                        pngs.append(os.path.join(OUT, f"b{bm:g}", f"{sc}_N{n}.png"))
        out.append(JS.bundle(pngs, os.path.join(OUT, f"stress_dem_{grp}.pdf"), outline))
    return out


def table():
    rows = sorted((json.load(open(f)) for f in glob.glob(os.path.join(OUT, "b*", "*.json"))),
                  key=lambda r: (list(SCENES).index(r["scene"]), r["bmax"], r["nrx"]))
    for r in rows:
        print(f"{r['group']:4s} {r['scene']:15s} b{r['bmax']:<4g} N{r['nrx']} | ref {r['ref']:6.1f} none {r['none']:6.1f} "
              f"whole {r['whole']:6.1f} sub {r['sub']:6.1f} full {r['joint_full']:6.1f} occ {r['joint_occ']:6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return print("wrote", summary())
    if a.table:
        return table()
    todo = [(sc, n, bm) for sc in SCENES for bm in BMAX for n in NRX
            if not (os.path.exists(f := os.path.join(OUT, f"b{bm:g}", f"{sc}_N{n}.json"))
                    and json.load(open(f)).get("version") == VERSION)]
    print(len(todo), "pages to do", flush=True)
    deadline = time.time() + a.budget
    with Pool(a.workers) as pool:
        for job in pool.imap_unordered(page, todo):
            if time.time() > deadline:
                pool.terminate()
                return print("budget reached, run again", flush=True)
    print("all done:", summary(), flush=True)


if __name__ == "__main__":
    main()
