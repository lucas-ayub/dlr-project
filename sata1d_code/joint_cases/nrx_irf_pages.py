"""
IRF pages (sweep layout, 6 panels) of the joint for several numbers of channels and baselines, on ramps and realistic
terrains with many targets. Same total PRF (2000 Hz, PRF_op = 2000 / Nrx, DPCA); channel 0 on the transmitter; the other
bxt random with the largest exactly bmax (stress_joint.baselines). Ramps: map of the nearest target; terrains: look-up
table from the terrain. The first panel compares the joint with occupied parcels and the same with the positions without
targets left out (support = scene_support).
    SATA_LUT_CACHE=/tmp/lutcache python joint_cases/nrx_irf_pages.py --budget 160      # from sata1d_code, repeat
    python joint_cases/nrx_irf_pages.py --summary
Pages: joint_cases/nrx_irf/b<bmax>/<scene>_N<Nrx>.png (+ .json); PDF: joint_cases/nrx_irf/nrx_irf.pdf
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
import stress_joint as SJ

OUT = os.path.join(HERE, "nrx_irf")
NRX = [2, 3, 4, 5, 6, 8]
BMAX = [100.0, 300.0, 500.0]
VERSION = 2


def _ramp(slope):
    xs, hs = SJ.ramp(slope, 250.0, 6000.0)
    return dict(xs=xs, hs=hs, terrain=None, amps=SJ.phases(len(xs)),
                label=f"ramp, {len(xs)} targets 250 m apart, slope {slope:g}$^\\circ$ (h 0 -> {hs.max():.0f} m)")


def _terrain(name, half):
    h, desc = W.TERRAINS[name]
    xs = np.arange(-half, half + 1e-6, 250.0)
    return dict(xs=xs, hs=h(xs), terrain=h, amps=SJ.phases(len(xs)), tname=name,
                label=f"{desc}, {len(xs)} targets 250 m apart")


SCENES = {  # file name -> builder
    "ramp03": lambda: _ramp(3.0),
    "ramp08": lambda: _ramp(8.0),
    "ramp15": lambda: _ramp(15.0),
    "hills": lambda: _terrain("hills", 6000.0),
    "mountains": lambda: _terrain("mountains", 6000.0),
    "escarpment": lambda: _terrain("escarpment", 3000.0),
}


def config(c, nrx, bmax):
    cfg = S.make_config(list(zip(c["xs"], c["hs"])), nrx=nrx, bxt_max=0.0, amplitudes=c["amps"], tx_rx_channel0=True)
    return replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=SJ.baselines(nrx, bmax)))


def page(job):
    sc, n, bm = job
    c = SCENES[sc]()
    cfg = config(c, n, bm)
    lut = (lambda cfg_, tr_, i, terrain, cid: SJ.dem_table(cfg_, tr_, i, c["tname"])) if c["terrain"] is not None else None
    tag = f"b{bm:g}"
    p, amb = P5.run(sc, tag, out=OUT, cfg=cfg, case=c, lut_fn=lut, with_support=True, name=f"{sc}_N{n}",
                    kind=f"random, largest {bm:g} m, $N_{{rx}}$ = {n}")
    json.dump(dict(amb, scene=sc, nrx=n, bmax=bm, version=VERSION, bxt=cfg.array.bxt.tolist(), label=c["label"]),
              open(os.path.join(OUT, tag, f"{sc}_N{n}.json"), "w"), indent=1)
    return sc, n, bm


def max_slope(sc):
    c = SCENES[sc]()
    return float(np.max(np.abs(np.diff(c["hs"]) / np.diff(c["xs"]))))


def limit_figure(res):
    """Joint (positions without targets left out) minus reference against the change of the terrain phase across one
    SATA resolution cell, 2 pi / lambda * |d dC0 / dx| * dx_SATA, with dC0 ~ 3.59e-6 bxt h (largest bxt, steepest slope)."""
    cfg = S.make_config([], nrx=4)
    wl, r0 = cfg.system.wl, cfg.scene.r0
    dx = np.sqrt(wl * r0 / 2.0)
    fig, ax = plt.subplots(figsize=(13, 7))
    mk = {"ramp03": "o", "ramp08": "s", "ramp15": "D", "hills": "^", "mountains": "v", "escarpment": "P"}
    cols = dict(zip(NRX, ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple", "tab:brown"]))
    slopes = {sc: max_slope(sc) for sc in SCENES}
    for (sc, bm, n), r in res.items():
        x = 2 * np.pi / wl * 3.59e-6 * bm * slopes[sc] * dx
        ax.plot(x, r["joint_sup"] - r["ref"], mk[sc], color=cols[n], ms=8, mec="k", alpha=0.85)
    for sc in SCENES:
        ax.plot([], [], mk[sc], color="0.6", mec="k", label=textwrap.shorten(SCENES[sc]()["label"], 70))
    for n in NRX:
        ax.plot([], [], "o", color=cols[n], label=f"$N_{{rx}}$ = {n}")
    ax.axhline(10, color="0.4", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("change of the terrain phase across one SATA resolution cell (%.0f m) [rad]\n"
                  "= 2$\\pi$/$\\lambda$ $\\cdot$ 3.59e-6 $\\cdot$ largest $b_{xt}$ $\\cdot$ steepest slope $\\cdot$ %.0f m" % (dx, dx))
    ax.set_ylabel("joint (positions without targets left out) minus reference\nhighest ambiguity [dB]")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    p = os.path.join(OUT, "limit.png"); fig.savefig(p, dpi=85); plt.close(fig)
    return p


def summary():
    res = {}
    for f in glob.glob(os.path.join(OUT, "b*", "*.json")):
        r = json.load(open(f))
        res[(r["scene"], r["bmax"], r["nrx"])] = r
    sc_list = list(SCENES)
    fig, axs = plt.subplots(len(BMAX), len(sc_list), figsize=(26, 12.5), sharex=True, sharey=True)
    lines = [("ref", "Reference", dict(color="tab:green", marker="o", lw=2)),
             ("none", "no SATA", dict(color="tab:red", marker="s", ls="--")),
             ("whole", "SATA whole band (block map)", dict(color="tab:blue", marker="^", ls=":")),
             ("joint_occ", "joint, occupied parcels", dict(color="tab:pink", marker="d", ls="--")),
             ("joint_sup", "joint, occupied parcels, positions without targets left out", dict(color="tab:purple", marker="o", lw=2))]
    for r_, bm in enumerate(BMAX):
        for k, sc in enumerate(sc_list):
            ax = axs[r_, k]
            ns = [n for n in NRX if (sc, bm, n) in res]
            for key, lab, st in lines:
                if ns:
                    ax.plot(ns, [res[(sc, bm, n)][key] for n in ns], label=lab, **st)
            ax.set_xticks(NRX); ax.grid(alpha=0.3); ax.set_ylim(-60, 10)
            if ns:
                ax.set_title(textwrap.fill(res[(sc, bm, ns[0])]['label'], 46) + f"\nlargest $b_{{xt}}$ = {bm:g} m", fontsize=9)
            if r_ == len(BMAX) - 1:
                ax.set_xlabel("number of channels $N_{rx}$")
            if k == 0:
                ax.set_ylabel("highest ambiguity [dB]")
    axs[0, 0].legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    p = os.path.join(OUT, "summary.png"); fig.savefig(p, dpi=80); plt.close(fig)
    p2 = limit_figure(res)
    pngs, outline = [p, p2], [("Summary", 0, 0), ("Where the joint departs from the reference", 1, 0)]
    for sc in sc_list:
        for bm in BMAX:
            for n in NRX:
                if (sc, bm, n) in res:
                    outline.append((f"{sc} | largest bxt {bm:g} m | Nrx = {n}", len(pngs), 0))
                    pngs.append(os.path.join(OUT, f"b{bm:g}", f"{sc}_N{n}.png"))
    return JS.bundle(pngs, os.path.join(OUT, "nrx_irf.pdf"), outline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return print("wrote", summary())
    todo = []
    for sc in SCENES:
        for bm in BMAX:
            for n in NRX:
                f = os.path.join(OUT, f"b{bm:g}", f"{sc}_N{n}.json")
                if not (os.path.exists(f) and json.load(open(f)).get("version") == VERSION):
                    todo.append((sc, n, bm))
    print(len(todo), "pages to do", flush=True)
    deadline = time.time() + a.budget
    with Pool(a.workers) as pool:
        for sc, n, bm in pool.imap_unordered(page, todo):
            if time.time() > deadline:
                pool.terminate()
                return print("budget reached, run again", flush=True)
    print("all done:", summary(), flush=True)


if __name__ == "__main__":
    main()
