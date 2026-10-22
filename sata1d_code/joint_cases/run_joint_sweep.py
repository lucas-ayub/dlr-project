"""
Sweep of the joint reconstruction (sata1d/joint.py, reconstruct_joint) over geometries and cross-track baselines.

Cases, in the order of the PDF:
  1. along-track ramps    point targets on a ramp (the ramp cases of combined_main_results_tx_rx.pdf)
  2. sines and cosines    point targets on the sine / cosine terrains of run_terrain_lut_cases
  3. random terrains      smooth terrains (random hills and mountains, escarpment, isolated mountain, plateaus) with
                          point targets on them; they replace the random-heights, cliff, spike, mesa, zig-zag and
                          piecewise point-target scenes, whose heights jumped from one target to the next
Baselines: channel 0 is always the transmitter (bat = bxt = 0); along-track baselines on the DPCA condition.
  two random draws as before, bxt ~ U(0, 100 m) and U(0, 500 m) (seed 0), and systematic sets, some with
  positive and negative bxt (BASELINES below).
Maps of the joint reconstruction: always from the DEM (terrain_lookup_table); for the point targets on a ramp the DEM
is the ramp through the targets (linear between them, held beyond the ends). No map reads the targets. SATA whole band / per sub-band: block map with copies (footer and heatmaps).

    python joint_cases/run_joint_sweep.py --workers 4              # from sata1d_code -> joint_cases/sweep/<baseline>/
    python joint_cases/run_joint_sweep.py --cases C03 --baselines s110
    python joint_cases/joint_sweep_summary.py                       # heatmaps + joint_cases/joint_sweep.pdf
Already computed (case, baseline) pairs are skipped; --budget stops starting new ones after that many seconds.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))   # sata1d_code
import sata1d as S
import sata1d.sata as SS
from sata1d.config import ArrayGeometry
from sata1d.joint import reconstruct_joint, joint_maps
import run_geometry_sweep as G
import run_terrain_lut_cases as T
import run_joint_cases as R
from run_one_case import ambiguity_level
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy

OUT = os.path.join(HERE, "sweep")
SATA_OSF = G.SATA_OSF
CH_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]


# ----------------------------------------------------------------------------------------------- baselines
def _random_bxt(bmax):
    b = np.random.default_rng(G.BXT_SEED).uniform(0.0, bmax, 4)
    b[0] = 0.0
    return b.round(1)


BASELINES = [  # (tag, bxt [m] of channels 0..3, kind)
    ("rnd100", _random_bxt(100.0), "random, U(0, 100 m)"),
    ("rnd500", _random_bxt(500.0), "random, U(0, 500 m)"),
    ("s10", [0, 10, 20, 30], "steps of 10 m"),
    ("s50", [0, 50, 100, 150], "steps of 50 m"),
    ("s100", [0, 100, 200, 300], "steps of 100 m"),
    ("s110", [0, 110, 120, 130], "close together, ~120 m"),
    ("s300", [0, 300, 310, 320], "close together, ~310 m"),
    ("pm50", [0, -50, 50, 100], "negative and positive"),
    ("pm120", [0, -120, 120, -60], "negative and positive"),
]
EXTRA_BASELINES = [  # more baselines, for joint_breaks.py (not in the main sweep PDF)
    ("s20", [0, 20, 40, 60], "steps of 20 m"),
    ("s30", [0, 30, 60, 90], "steps of 30 m"),
    ("s75", [0, 75, 150, 225], "steps of 75 m"),
    ("s150", [0, 150, 300, 450], "steps of 150 m"),
    ("c50", [0, 50, 60, 70], "close together, ~60 m"),
    ("c200", [0, 200, 210, 220], "close together, ~210 m"),
    ("pm25", [0, -25, 25, 50], "negative and positive"),
    ("pm100", [0, -100, 100, 200], "negative and positive"),
    ("pm60", [0, -60, 60, -30], "negative and positive"),
    ("pm150", [0, -150, 150, -75], "negative and positive"),
    ("perm50", [0, 100, 50, 150], "steps of 50 m, other order"),
    ("dec50", [0, 150, 100, 50], "decreasing, steps of 50 m"),
    ("uneven", [0, 20, 60, 140], "uneven steps"),
]
BXT = {t: np.asarray(b, float) for t, b, _ in BASELINES + EXTRA_BASELINES}
BKIND = {t: k for t, _, k in BASELINES + EXTRA_BASELINES}


def bxt_text(b):
    return "[" + ", ".join(f"{v:g}" for v in b) + "] m"


# ----------------------------------------------------------------------------------------------- terrains
def random_terrain(seed, relief, corr, half=80e3, dx=20.0, region=22e3):
    """Smooth random terrain: white noise filtered by a Gaussian (correlation exp(-x^2 / corr^2)), scaled so that
    min -> max over |x| <= region is 0 -> relief."""
    g = np.random.default_rng(seed)
    x = np.arange(-half, half + dx, dx)
    k = np.fft.rfftfreq(x.size, dx)
    h = np.fft.irfft(np.fft.rfft(g.standard_normal(x.size)) * np.exp(-2.0 * (np.pi * 0.5 * corr * k) ** 2), x.size)
    m = np.abs(x) <= region
    h = (h - h[m].min()) * relief / np.ptp(h[m])
    return lambda xq: np.interp(np.asarray(xq, float), x, h)


TERRAINS = {  # name -> (h(x), description)
    "hills": (random_terrain(3, 300.0, 1500.0), "random hills, relief 300 m, correlation length 1.5 km"),
    "mountains": (random_terrain(3, 1000.0, 3000.0), "random mountains, relief 1000 m, correlation length 3 km"),
    "escarpment": (lambda x: 250.0 * (1.0 + np.tanh(np.asarray(x, float) / 500.0)), "escarpment 0 -> 500 m, 1 km wide"),
    "mountain": (lambda x: 1000.0 * np.exp(-0.5 * (np.asarray(x, float) / 1500.0) ** 2),
                 "isolated mountain 1000 m high (Gaussian, sigma 1.5 km)"),
    "plateau_n": (lambda x: 250.0 * (np.tanh((np.asarray(x, float) + 1200.0) / 500.0) - np.tanh((np.asarray(x, float) - 1200.0) / 500.0)),
                  "plateau 500 m high, 2.4 km wide"),
    "plateau_w": (lambda x: 250.0 * (np.tanh((np.asarray(x, float) + 3900.0) / 500.0) - np.tanh((np.asarray(x, float) - 3900.0) / 500.0)),
                  "plateau 500 m high, 7.8 km wide"),
}


def max_slope_deg(h, lo, hi):
    x = np.linspace(lo, hi, 4001)
    return float(np.degrees(np.arctan(np.abs(np.gradient(h(x), x)).max())))


def spaced(n, d):
    return (np.arange(n) - (n - 1) / 2.0) * d


def dist_text(d):
    return f"{d:g} m" if d < 1000 else f"{d / 1e3:g} km"


# ----------------------------------------------------------------------------------------------- cases
def _ramp(cid):
    kind = "v4" if cid.count("-") == 2 else ("many" if cid.startswith("MP") else "sweep")
    xs, hs, amps, _ = {"v4": R.v4_case, "sweep": R.sweep_case, "many": R.many_case}[kind](cid)
    d = float(np.diff(xs)[0])
    return dict(xs=xs, hs=hs, amps=amps, terrain=None,
                label=f"ramp, {len(xs)} targets {dist_text(d)} apart, h 0 -> {hs.max():.0f} m",
                old=cid)


def _sincos(preset, n, centre):
    expr, desc = T.ALL_PRESETS[preset]
    terrain = T.terrain_from(expr)
    xs = np.atleast_1d(T.target_positions(n, centre=centre)).astype(float)
    desc = desc.replace("2 pi", "2$\\pi$")
    return dict(xs=xs, hs=terrain(xs), amps=None, terrain=terrain,
                label=f"{desc}, {n} target{'s' if n > 1 else ''}", old=f"{preset}-N{n}")


def _rand(name, n, d):
    h, desc = TERRAINS[name]
    xs = spaced(n, d)
    return dict(xs=xs, hs=h(xs), amps=None, terrain=h, label=f"{desc}, {n} targets {dist_text(d)} apart", old=None)


GROUPS = [("ramp", "along-track ramps"), ("sincos", "sine and cosine terrains"), ("random", "random terrains")]
_CASE_LIST = (
    [("ramp", lambda c=c: _ramp(c)) for c in ("RA-SS-012", "RA-012", "RA-049", "RA-NN-036", "RA-NN-038", "RA-LN-038",
                                                "MP-R13", "MP-R25", "MP-R49")]
    + [("sincos", lambda a=a: _sincos(*a)) for a in R.TERRAIN_CASES]
    + [("random", lambda a=a: _rand(*a)) for a in (
        ("hills", 13, 200.0), ("hills", 5, 2600.0),
        ("mountains", 13, 200.0), ("mountains", 5, 2600.0), ("mountains", 5, 10000.0),
        ("mountains", 13, 1000.0), ("mountains", 25, 500.0),
        ("escarpment", 9, 200.0), ("escarpment", 5, 2600.0),
        ("mountain", 9, 200.0), ("mountain", 5, 2600.0),
        ("plateau_n", 9, 500.0), ("plateau_w", 5, 2600.0))])
CASES = {}   # "C01", ... in the order of the PDF -> (group, builder)
for k, (grp, fn) in enumerate(_CASE_LIST, 1):
    CASES[f"C{k:02d}"] = (grp, fn)


def case_info(cid):
    grp, fn = CASES[cid]
    c = fn()
    c.update(group=grp, number=int(cid[1:]))
    return c


def make_cfg(c, btag):
    cfg = S.make_config(list(zip(c["xs"], c["hs"])), bxt_max=0.0, seed=G.BXT_SEED, amplitudes=c["amps"], tx_rx_channel0=True)
    return replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=BXT[btag].copy()))


# ----------------------------------------------------------------------------------------------- one run
def _lut(cfg, tr, i, terrain, cid):
    """terrain_lookup_table(step=4) of channel i; with the environment variable SATA_LUT_CACHE set to a folder, the
    table is kept there (it depends only on the case, the channel index and its bxt)."""
    cache = os.environ.get("SATA_LUT_CACHE")
    if not cache:
        return S.terrain_lookup_table(cfg, tr, i, terrain, step=4)
    f = os.path.join(cache, f"{cid}_ch{i}_bxt{cfg.array.bxt[i]:g}.npy")
    if os.path.exists(f):
        return np.load(f)
    lut = S.terrain_lookup_table(cfg, tr, i, terrain, step=4)
    os.makedirs(cache, exist_ok=True); np.save(f, lut)
    return lut


def run(cid, btag, out=OUT, map_kind=None):
    """One sweep page. map_kind=None: look-up table from the DEM (for the ramps of point targets, the ramp through the
    targets); map_kind="targets": nearest target (joint_maps), only for explicit comparisons."""
    t0 = time.time()
    c = case_info(cid)
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    cfg = make_cfg(c, btag)
    tr = S.build_platform_tracks(cfg)
    dC0 = S.delta_C0_per_target(cfg, tr)
    ch = S.generate_channels(cfg, tr)
    points_only = terrain is None
    if points_only:                                             # ramp of point targets: its DEM is the ramp through them
        o = np.argsort(xs)
        terrain = lambda x, xo=np.asarray(xs, float)[o], ho=np.asarray(hs, float)[o]: np.interp(np.asarray(x, float), xo, ho)
    if map_kind == "targets":
        jmaps, jlabel = np.array(joint_maps(cfg, tr, dC0)), "nearest target (joint_maps)"
    else:
        jmaps, jlabel = np.array([_lut(cfg, tr, i, terrain, cid) for i in range(cfg.Nrx)]), "look-up table from the terrain"
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    tj = {}
    env = {"ref": F(S.generate_reference(cfg, tr)),
           "none": F(S.reconstruct(cfg, tr, ch)),
           "whole": F(S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=SATA_OSF, mode="footprint", dC0=dC0))),
           "sub": F(S.reconstruct_subband(cfg, tr, ch, sata_osf=SATA_OSF, mode="footprint", dC0=dC0))}
    for key, occ in (("joint_full", False), ("joint_occ", True)):
        t1 = time.time(); env[key] = F(reconstruct_joint(cfg, tr, ch, maps=jmaps, sata_osf=SATA_OSF, occupied=occ)); tj[key] = time.time() - t1
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    amb = {k: float(ambiguity_level(t, v, pk, xs, cfg.system.vs, G.ambiguity_times(cfg))) for k, v in env.items()}
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}

    fig = plt.figure(figsize=(18, 9.6))
    gs = GridSpec(2, 2, width_ratios=[1.45, 1.25], hspace=0.36, wspace=0.14, figure=fig)
    fig.suptitle(f"Case {c['number']}: {c['label']} | $b_{{xt}}$ = {bxt_text(BXT[btag])} ({BKIND[btag]}) | "
                 f"joint reconstruction (map: {jlabel})", fontsize=12, y=0.995)
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["joint_full"], color="tab:pink", lw=1.4, ls=(0, (5, 3)), label="joint, full Nrx x Nsb system (occupied=False)")
    ax.plot(t, db["joint_occ"], color="tab:purple", lw=1.3, label="joint, occupied parcels only (occupied=True)")
    ax.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | joint, occupied {amb['joint_occ']:.1f} dB | "
                 f"joint, full system {amb['joint_full']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | improvement: "
                 f"joint, occupied {amb['none'] - amb['joint_occ']:.1f} dB, joint, full system {amb['none'] - amb['joint_full']:.1f} dB", fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")

    # geometry: 2D zoom on the targets and the reference point, 3D over the whole azimuth line
    xs_, dys, hs_ = scene_coordinates(xs, hs)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    gg = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.18, width_ratios=[1.0, 1.1])
    g2 = fig.add_subplot(gg[0, 0]); g3 = fig.add_subplot(gg[0, 1], projection="3d")
    lo2, hi2 = min(xs_.min(), 0.0), max(xs_.max(), 0.0); pad2 = max(0.12 * (hi2 - lo2), 500.0)
    slope = ""
    if points_only:
        geometry_2d(g2, xs_, dys, hs_, False, title=None, annotate=len(xs_) <= 13)
        geometry_3d(g3, xs_, dys, hs_, False, title=None)
        dh = np.ptp(hs_)
    else:
        xg = np.linspace(-half, half, 4801); hg = terrain(xg)
        hz = hg[(xg >= lo2 - pad2) & (xg <= hi2 + pad2)]; dh = np.ptp(hz)
        slope = f" | max slope {max_slope_deg(terrain, lo2 - pad2, hi2 + pad2):.0f}$^\\circ$"
        geometry_2d(g2, xs_, dys, hs_, False, title=None, connect=False, annotate=len(xs_) <= 13)
        g2.fill_between(xg, min(hg.min(), 0) - 0.05 * np.ptp(hg), hg, color="#D7C9A7", alpha=0.45, lw=0)
        g2.plot(xg, hg, color="#6D4C2F", lw=1.2, zorder=2, label="terrain (the scene)")
        ylo, yhi = min(hz.min(), 0.0), max(hz.max(), hs_.max(), 1.0)
        g2.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))
        geometry_3d(g3, xs_, dys, hs_, False, connect=False, title=None)
        g3.plot(xg, iso_range_dy(hg), hg, color="#6D4C2F", lw=1.0)
    g2.plot(0.0, 0.0, marker="*", ms=13, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $h$ = 0)")
    g2.set_xlim(lo2 - pad2, hi2 + pad2)
    g2.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.1, handlelength=1.6, columnspacing=1.2)
    g2.set_title(f"Zoom on the point targets and the reference point | $\\Delta h_{{max}}$ = {dh:.0f} m{slope}", fontsize=9.5, pad=36)
    g2.xaxis.set_major_formatter(km); g2.set_xlabel("azimuth position $x$ [km]")
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=6)
    g3.set_xlim(-half, half); g3.xaxis.set_major_formatter(km); g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4)
    g3.set_title(f"Whole azimuth line ({2 * half / 1e3:.0f} km) in 3D (x, $\\Delta y$, h)", fontsize=9.5)
    for t_ in list(g3.texts):
        t_.remove()

    # the map read by the joint reconstruction
    am = fig.add_subplot(gs[1, 1])
    n = np.arange(cfg.Na_ch)
    for i in range(cfg.Nrx):
        am.step(n, 1e3 * jmaps[i], where="mid", color=CH_COLORS[i], lw=1.3, label=f"channel {i} ($b_{{xt}}$ = {cfg.array.bxt[i]:g} m)")
    cells1 = np.array([SS.az_pixel_of_scatterer(cfg, x, 1) for x in xs])
    if len(cells1) <= 60:
        for cc in cells1:
            am.axvline(cc, color="0.75", lw=0.6, zorder=0)
    am.plot(cells1, 1e3 * dC0[:, 1], "o", ms=5, color=CH_COLORS[1], mec="k", zorder=5, label="point targets: own $\\Delta C_0$ (channel 1)")
    am.set_xlim(0, cfg.Na_ch - 1); am.grid(alpha=0.3)
    am.set_xlabel("azimuth cell of the channel line"); am.set_ylabel("$\\Delta C_0$ in the map [mm]")
    lo_, hi_ = am.get_ylim(); am.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    am.set_title(f"$\\Delta C_0$ per cell read by the joint reconstruction ({jlabel})", fontsize=10)
    am.legend(fontsize=8, loc="upper center", ncol=3)
    fig.text(0.5, 0.012, f"Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions of every "
             f"target. For comparison (block map with copies): SATA whole band {amb['whole']:.1f} dB, SATA per sub-band {amb['sub']:.1f} dB. "
             f"Joint: {tj['joint_occ']:.2f} s (occupied) and {tj['joint_full']:.2f} s (full system) per line. "
             f"$N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz, channel 0 is the transmitter "
             f"($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {bxt_text(cfg.array.bxt)}.", ha="center", fontsize=8.5, color="0.35")
    fig.subplots_adjust(bottom=0.1)
    d = os.path.join(out, btag); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"{cid}.png"), dpi=90, bbox_inches="tight"); plt.close(fig)
    res = dict(amb, case=cid, baseline=btag, bxt=list(map(float, cfg.array.bxt)), label=c["label"], group=c["group"],
               old=c["old"], seconds=time.time() - t0)
    json.dump(res, open(os.path.join(d, f"{cid}.json"), "w"), indent=1)
    return res


def _job(a):
    cid, btag, out, deadline = a
    if time.time() > deadline:
        return None
    return run(cid, btag, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--baselines", nargs="+", default=[b[0] for b in BASELINES], choices=list(BXT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=float, default=1e9, help="seconds after which no new run is started")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    deadline = time.time() + a.budget
    todo = [(c, b, a.out, deadline) for c in a.cases for b in a.baselines
            if not os.path.exists(os.path.join(a.out, b, f"{c}.json"))]
    todo.sort(key=lambda j: CASES[j[0]][0] == "ramp")        # terrains first (slowest)
    print(len(todo), "runs to do", flush=True)
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as p:
            for r in p.imap_unordered(_job, todo):
                if r:
                    print(f"{r['case']} {r['baseline']:6s} none {r['none']:.1f} whole {r['whole']:.1f} joint_occ {r['joint_occ']:.1f} "
                          f"({r['seconds']:.0f} s)", flush=True)
    else:
        for j in todo:
            r = _job(j)
            if r:
                print(r["case"], r["baseline"], f"{r['joint_occ']:.1f}", flush=True)


if __name__ == "__main__":
    main()
