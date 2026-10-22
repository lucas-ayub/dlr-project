"""
Complete sweep of the final joint (joint reconstruction over the whole line, reconstruct_joint_line) in 1-D, in the
four-panel layout of the sweep pages:

    top left      azimuth IRF: reference + joint (final version)
    top right     the scene: 2D zoom on the targets and the reference point | 3D over the whole azimuth line
    bottom left   azimuth IRF: reference + no SATA
    bottom right  delta_C0 of every channel at every position of the line, from the DEM (what the joint reads)

Cases: every scene of the final sweep (sweep_final.FAMILIES: ramps, sines / cosines, realistic terrains; 1, a few and
many targets) plus 18 new scenes on more realistic terrains (fractal alpine relief, rolling hills, glacial valley,
folded ridges, canyon with terraces, volcano; 1, 7 and 49 targets each), and EVERY baseline: Nrx = 2, 3, 4, 6, 8 with
the largest b_xt 100, 300 and 500 m (random, the largest exactly that value), plus the fixed sets of four channels
(u100, u500, s10, s50, pm50). Channel 0 is always the transmitter (bat = bxt = 0). The joint reads only the DEM.
Same computation and ambiguity measure as sweep_final.py / audit_dem.py (orders up to |m| PRF_op < B).

    python joint_cases/sweep_joint_all.py --workers 4        # numbers and pages (resumable), from sata1d_code
    python joint_cases/sweep_joint_all.py --pdf              # one PDF per family of geometries
    python joint_cases/sweep_joint_all.py --table | --list | --check
Output: joint_cases/sweep_joint_all/numbers/<job>.json, pages/<job>.png, sweep_joint_all_1d_<family>.pdf, all_numbers.json
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
sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
from sata1d.config import ArrayGeometry
from sata1d.joint_line import height_table, line_maps, line_positions, reconstruct_joint_line
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy
import audit_dem as AD
import sweep_final as SF
import run_joint_sweep as W

OUT = os.path.join(HERE, "sweep_joint_all")
VERSION = 2
CH = AD.CH


# ----------------------------------------------------------------------------------------------- new realistic terrains
def fractal_profile(seed, relief, hurst, outer, inner, half=80e3, dx=10.0, region=22e3):
    """Random profile with the power spectrum of real terrain, ~ k^-(2H+1) between the scales outer and inner [m]
    (Gaussian roll-off below inner), scaled so that min -> max over |x| <= region is 0 -> relief."""
    x = np.arange(-half, half + dx, dx)
    k = np.fft.rfftfreq(x.size, dx)
    amp = (k ** 2 + outer ** -2) ** (-(hurst + 0.5) / 2.0) * np.exp(-(k * inner) ** 2)
    h = np.fft.irfft(np.fft.rfft(np.random.default_rng(seed).standard_normal(x.size)) * amp, x.size)
    m = np.abs(x) <= region
    h = (h - h[m].min()) * relief / np.ptp(h[m])
    return x, h


def _tab(x, h):
    return lambda xq: np.interp(np.asarray(xq, float), x, h)


def _new_terrains():
    xa, alpine = fractal_profile(21, 2000.0, 0.8, 30000.0, 100.0)
    _, rolling = fractal_profile(12, 150.0, 1.0, 20000.0, 100.0)
    _, rough30 = fractal_profile(13, 30.0, 0.7, 2000.0, 30.0)
    _, rough40 = fractal_profile(14, 40.0, 0.7, 2000.0, 30.0)
    _, rough20 = fractal_profile(15, 20.0, 0.7, 2000.0, 30.0)
    d = np.maximum(np.abs(xa + 500.0) - 900.0, 0.0)
    valley = 1300.0 * (1.0 - np.exp(-(d / 2500.0) ** 2)) + rough40
    ridges = 350.0 + 220.0 * np.sin(2 * np.pi * xa / 3500.0) + 90.0 * np.sin(2 * np.pi * xa / 1300.0 + 1.0) + rough30
    u = np.abs(xa - 700.0)
    step = lambda w: 0.5 * (1.0 - np.tanh((u - w) / 80.0))
    canyon = 600.0 - 150.0 * (step(1500.0) + step(1000.0) + step(500.0)) + rough20
    volcano = 1800.0 * np.exp(-np.sqrt(xa ** 2 + 300.0 ** 2) / 3000.0) - 200.0 * np.exp(-(xa / 350.0) ** 2)
    return {
        "alpine": (_tab(xa, alpine), "fractal alpine relief 0 -> 2000 m (power spectrum of real mountain DEMs, detail down to 100 m)"),
        "rolling": (_tab(xa, rolling), "fractal rolling hills 0 -> 150 m (power spectrum of real DEMs, detail down to 100 m)"),
        "valley": (_tab(xa, valley), "glacial U-shaped valley, 1.8 km wide floor, walls up to 1300 m, 40 m roughness"),
        "ridges": (_tab(xa, ridges), "folded ridges and valleys (periods 3.5 km and 1.3 km, 130 -> 660 m), 30 m roughness"),
        "canyon": (_tab(xa, canyon), "plateau at 600 m cut by a canyon 450 m deep with three terraces (steep walls), 20 m roughness"),
        "volcano": (_tab(xa, volcano), "volcano about 1450 m high with a summit crater 200 m deep, concave flanks"),
    }


NEW_T = _new_terrains()


def _steep_point(terrain, lo=-3000.0, hi=3000.0):
    """The position of largest slope in [lo, hi] (a property of the scene: where the single target is put)."""
    x = np.arange(lo, hi + 1.0, 10.0)
    return float(x[np.argmax(np.abs(np.gradient(terrain(x), x)))])


NEW_SCENES = {}
for _k, (_t, _d) in NEW_T.items():
    NEW_SCENES[f"one_{_k}"] = ("one", (lambda t=_t, d=_d: AD._on(t, d, [_steep_point(t)])))
    NEW_SCENES[f"few_{_k}_7"] = ("few", (lambda t=_t, d=_d: AD._on(t, d, W.spaced(7, 1000.0) + 130.0)))
    NEW_SCENES[f"many_{_k}_49"] = ("many", (lambda t=_t, d=_d: AD._on(t, d, AD.span(6000.0, 250.0))))
AD.SCENES.update(NEW_SCENES)

FAMILIES = [  # (title, file suffix, scenes)
    ("Ramps", "ramps", SF.FAMILIES[0][1]),
    ("Sines and cosines", "sines_cosines", SF.FAMILIES[1][1]),
    ("Realistic terrains", "realistic", SF.FAMILIES[2][1]),
    ("More realistic terrains (fractal relief, valley, ridges, canyon, volcano)", "realistic_fractal",
     [f"{g}_{k}{n}" for k in NEW_T for g, n in (("one", ""), ("few", "_7"), ("many", "_49"))]),
]
TAGS = ([f"N2b{b:g}" for b in AD.BMAX] + [f"N3b{b:g}" for b in AD.BMAX] + [f"N4b{b:g}" for b in AD.BMAX] + list(AD.FIXED)
        + [f"N6b{b:g}" for b in AD.BMAX] + [f"N8b{b:g}" for b in AD.BMAX])


def jobs():
    return [(sc, t) for _, _, scenes in FAMILIES for sc in scenes for t in TAGS]


# ----------------------------------------------------------------------------------------------- one job
def compute(sc, tag):
    """Reference, no SATA and the final joint (DEM only) of one job, as in audit_dem.compute."""
    grp, fn = AD.SCENES[sc]
    c = fn()
    xs, terrain = c["xs"], c["terrain"]
    bxt, kind = AD.baseline(tag)
    cfg = S.make_config(list(zip(xs, c["hs"])), nrx=len(bxt), bxt_max=0.0, amplitudes=c["amps"], tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy()))
    tr = S.build_platform_tracks(cfg)
    ch = S.generate_channels(cfg, tr)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    hgrid = terrain(np.linspace(-half - 100.0, half + 100.0, 8001))
    tab = height_table(cfg, tr, hgrid.min(), hgrid.max())
    Mline = line_maps(cfg, tr, terrain, tab)                             # the only knowledge of the scene given to the joint
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    t1 = time.time()
    y, info = reconstruct_joint_line(cfg, tr, ch, Mline, return_info=True)
    sec = time.time() - t1
    env = {"ref": F(S.generate_reference(cfg, tr)), "none": F(S.reconstruct(cfg, tr, ch)), "joint_line": F(y)}
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt_amb = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    orders = SF.orders_of(cfg)
    amb = {k: float(AD.ambiguity_level(t, v, pk, xs, cfg.system.vs, dt_amb, orders=orders)) for k, v in env.items()}
    lo, hi = min(xs.min(), 0.0), max(xs.max(), 0.0); pad = max(0.12 * (hi - lo), 500.0)
    res = dict(amb, scene=sc, baseline=tag, group=grp, nrx=cfg.Nrx, bxt=[float(b) for b in cfg.array.bxt], kind=kind,
               n_targets=int(len(xs)), label=c["label"], max_slope_deg=W.max_slope_deg(terrain, lo - pad, hi + pad),
               dh=float(np.ptp(terrain(np.linspace(lo - pad, hi + pad, 2001)))), orders=list(orders), seconds_joint=sec,
               line_iterations=info["iterations"], line_unexplained_dB=float(20 * np.log10(info["unexplained"])), version=VERSION)
    return res, dict(c=c, cfg=cfg, env=env, t=t, pk=pk, Mline=Mline)


def page(res, d, out_png):
    c, cfg, env, t, pk, Mline = d["c"], d["cfg"], d["env"], d["t"], d["pk"], d["Mline"]
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    fig = plt.figure(figsize=(18, 9.6))
    gs = GridSpec(2, 2, width_ratios=[1.45, 1.25], hspace=0.36, wspace=0.14, figure=fig, left=0.045, right=0.97, top=0.86, bottom=0.1)
    lab = c["label"] if c["label"].startswith("h =") else c["label"][0].upper() + c["label"][1:]
    head = f"{lab} | $N_{{rx}}$ = {cfg.Nrx}, $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)} ({res['kind']})"
    tail = "the joint reads only the DEM; no method is given the positions of the targets"
    if len(head) + len(tail) > 175:
        fig.suptitle(head + "\n" + tail, fontsize=12, y=0.995)
        gs.update(top=0.84)
    else:
        fig.suptitle(head + " | " + tail, fontsize=12, y=0.985)
    # ---- IRFs
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["joint_line"], color="tab:purple", lw=1.3, label="joint, final version (over the whole line, DEM applied at every position)")
    ax.set_title(f"highest ambiguity: reference {res['ref']:.1f} dB | joint {res['joint_line']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"highest ambiguity: reference {res['ref']:.1f} dB | no SATA {res['none']:.1f} dB | improvement of the joint "
                 f"{res['none'] - res['joint_line']:.1f} dB", fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")
    # ---- the scene: 2D zoom on the targets and the reference point, 3D over the whole azimuth line
    xs_, dys, hs_ = scene_coordinates(xs, hs)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    gg = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.18, width_ratios=[1.0, 1.1])
    g2 = fig.add_subplot(gg[0, 0]); g3 = fig.add_subplot(gg[0, 1], projection="3d")
    lo2, hi2 = min(xs_.min(), 0.0), max(xs_.max(), 0.0); pad2 = max(0.12 * (hi2 - lo2), 500.0)
    xg = np.linspace(-half, half, 9601); hg = terrain(xg)
    hz = hg[(xg >= lo2 - pad2) & (xg <= hi2 + pad2)]
    geometry_2d(g2, xs_, dys, hs_, False, title=None, connect=False, annotate=len(xs_) <= 13)
    g2.fill_between(xg, min(hg.min(), 0) - 0.05 * max(np.ptp(hg), 1.0), hg, color="#D7C9A7", alpha=0.45, lw=0)
    g2.plot(xg, hg, color="#6D4C2F", lw=1.2, zorder=2, label="terrain (the scene)")
    ylo, yhi = min(hz.min(), 0.0), max(hz.max(), hs_.max(), 1.0)
    g2.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))
    g2.plot(0.0, 0.0, marker="*", ms=13, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $h$ = 0)")
    g2.set_xlim(lo2 - pad2, hi2 + pad2)
    g2.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.1, handlelength=1.6, columnspacing=1.2)
    g2.set_title(f"Zoom on the point targets and the reference point | $\\Delta h_{{max}}$ = {res['dh']:.0f} m | max slope "
                 f"{res['max_slope_deg']:.0f}$^\\circ$", fontsize=9.5, pad=36)
    g2.xaxis.set_major_formatter(km); g2.set_xlabel("azimuth position $x$ [km]")
    geometry_3d(g3, xs_, dys, hs_, False, connect=False, title=None)
    g3.plot(xg, iso_range_dy(hg), hg, color="#6D4C2F", lw=1.0)
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=6)
    g3.set_xlim(-half, half); g3.xaxis.set_major_formatter(km); g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4)
    g3.set_zlim(min(0.0, hg.min()), max(hg.max(), 50.0) * 1.15)
    g3.set_title(f"Whole azimuth line ({2 * half / 1e3:.0f} km) in 3D (x, $\\Delta y$, h)", fontsize=9.5)
    for t_ in list(g3.texts):
        t_.remove()
    # ---- what the joint reads: delta_C0 of every channel at every position of the line, from the DEM
    am = fig.add_subplot(gs[1, 1])
    xl = line_positions(cfg)
    for i in range(cfg.Nrx):
        am.plot(xl, 1e3 * Mline[i], color=CH[i], lw=1.2, label=f"channel {i} ($b_{{xt}}$ = {cfg.array.bxt[i]:g} m)")
    if len(xs) <= 60:
        for x in xs:
            am.axvline(x, color="0.75", lw=0.6, zorder=0)
    i1 = int(np.argmax(np.abs(cfg.array.bxt)))
    am.plot(xs, 1e3 * np.interp(xs, xl, Mline[i1]), "o", ms=5, color=CH[i1], mec="k", zorder=5,
            label=f"positions of the point targets (channel {i1}; not given to the joint)")
    am.set_xlim(xl[0], xl[-1]); am.xaxis.set_major_formatter(km); am.grid(alpha=0.3)
    am.set_xlabel("azimuth position $x$ [km]"); am.set_ylabel("$\\Delta C_0$ from the DEM [mm]")
    lo_, hi_ = am.get_ylim(); am.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    am.set_title("$\\Delta C_0$ of every channel at every position of the line, from the DEM (what the joint reads)", fontsize=10)
    am.legend(fontsize=7.8 if cfg.Nrx <= 4 else 6.6, loc="upper center", ncol=2 if cfg.Nrx <= 4 else 3)
    fig.text(0.5, 0.012, "Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions of every target "
             f"(up to {max(res['orders'])} times the ambiguity distance of the channel PRF, on both sides).\n$N_{{rx}}$ = {cfg.Nrx}, "
             f"PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz, channel 0 is the transmitter ($b_{{at}}$ = $b_{{xt}}$ = 0), "
             f"$b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)}.", ha="center", fontsize=8.5, color="0.35")
    fig.savefig(out_png, dpi=90, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------------------------- driver
def _paths(sc, tag):
    k = f"{sc}__{tag}"
    return os.path.join(OUT, "numbers", k + ".json"), os.path.join(OUT, "pages", k + ".png")


def _done(sc, tag):
    pj, pp = _paths(sc, tag)
    return os.path.exists(pj) and os.path.exists(pp) and json.load(open(pj)).get("version") == VERSION


def _job(a):
    sc, tag, deadline = a
    if time.time() > deadline:
        return None
    t0 = time.time()
    res, d = compute(sc, tag)
    pj, pp = _paths(sc, tag)
    page(res, d, pp)
    json.dump(res, open(pj, "w"), indent=1)                              # written last: marks the job as done
    return f"{sc:22s} {tag:7s} ref {res['ref']:6.1f} none {res['none']:6.1f} joint {res['joint_line']:6.1f} ({time.time() - t0:.0f} s)"


def load():
    out = {}
    for sc, tag in jobs():
        pj = _paths(sc, tag)[0]
        if os.path.exists(pj):
            out[(sc, tag)] = json.load(open(pj))
    return out


def table():
    res = load()
    json.dump([res[j] for j in jobs() if j in res], open(os.path.join(OUT, "all_numbers.json"), "w"), indent=1)
    for fam, _, scenes in FAMILIES:
        print(fam)
        for sc in scenes:
            for tag in TAGS:
                r = res.get((sc, tag))
                if r:
                    print(f"  {sc:22s} {tag:7s} Nrx {r['nrx']} | ref {r['ref']:6.1f} none {r['none']:6.1f} joint {r['joint_line']:6.1f}")


def check():
    """The jobs in common with sweep_final.py must give the same numbers (same computation)."""
    res, n, worst = load(), 0, 0.0
    for (sc, tag), r in res.items():
        pj = os.path.join(SF.OUT, "numbers", f"{sc}__{tag}.json")
        if os.path.exists(pj):
            o = json.load(open(pj))
            dd = max(abs(r[k] - o[k]) for k in ("ref", "none", "joint_line"))
            worst = max(worst, dd); n += 1
            if dd > 0.05:
                print(f"  differs: {sc} {tag}: " + ", ".join(f"{k} {r[k]:.2f} vs {o[k]:.2f}" for k in ("ref", "none", "joint_line")))
    print(f"{n} jobs in common with sweep_final, largest difference {worst:.3f} dB")


def bundle(pngs, out, outline):
    """Multi-page PDF (JPEG pages) with bookmarks, one page at a time (little memory)."""
    from PIL import Image
    from pypdf import PdfReader, PdfWriter
    import tempfile
    w = PdfWriter()
    tmp = os.path.join(tempfile.gettempdir(), "sweep_joint_all_page.pdf")
    for p in pngs:
        Image.open(p).convert("RGB").save(tmp, resolution=90, quality=75)
        w.add_page(PdfReader(tmp).pages[0])
    parent = None
    for title, k, lvl in outline:
        it = w.add_outline_item(title, k, parent=parent if lvl else None)
        if lvl == 0:
            parent = it
    w.write(out)
    os.remove(tmp)
    return out


def pdf():
    res, done = load(), []
    for fam, suffix, scenes in FAMILIES:
        pngs, outline = [], []
        for sc in scenes:
            first = True
            for tag in TAGS:
                pp = _paths(sc, tag)[1]
                if (sc, tag) not in res or not os.path.exists(pp):
                    continue
                r = res[(sc, tag)]
                if first:
                    lab = r["label"].replace("$\\pi$", "pi").replace("$^\\circ$", " deg").replace("$\\pm$", "+-")
                    outline.append((lab, len(pngs), 0)); first = False
                outline.append((f"{r['nrx']} channels, bxt {W.bxt_text(r['bxt'])}", len(pngs), 1))
                pngs.append(pp)
        if pngs:
            out = os.path.join(OUT, f"sweep_joint_all_1d_{suffix}.pdf")
            done.append(f"{bundle(pngs, out, outline)} ({len(pngs)} pages)")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, help="scene ids (default: all)")
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for sc, tag in jobs():
            print(f"{sc:22s} {tag:7s} bxt {W.bxt_text(AD.baseline(tag)[0])}")
        return print(len(jobs()), "jobs")
    if a.table:
        return table()
    if a.check:
        return check()
    if a.pdf:
        return print("wrote\n" + "\n".join(pdf()))
    for sub in ("numbers", "pages"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    todo = [(sc, tag) for sc, tag in jobs() if (a.only is None or sc in a.only) and not _done(sc, tag)]
    print(len(todo), "jobs to do", flush=True)
    deadline = time.time() + a.budget
    args = [(sc, tag, deadline) for sc, tag in todo]
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers, maxtasksperchild=20) as pool:
            for line in pool.imap_unordered(_job, args):
                if line:
                    print(line, flush=True)
    else:
        for x in args:
            line = _job(x)
            if line:
                print(line, flush=True)
    print("left:", sum(not _done(sc, tag) for sc, tag in jobs() if a.only is None or sc in a.only), flush=True)


if __name__ == "__main__":
    main()
