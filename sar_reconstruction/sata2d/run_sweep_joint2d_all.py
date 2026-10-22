"""
Complete sweep of the 2-D joint reconstruction (joint2d.py, final version in 2-D) on 3-D scenes, in the four-panel
layout of the sweep pages:

    top left      azimuth IRF: reference + joint (2-D, final version)
    top right     the scene: from above with a zoom on the targets | in 3D over the whole imaged area
    bottom left   azimuth IRF: reference + no SATA (the two-step reconstruction with the flat-earth filters)
    bottom right  delta_C0 in the DEM table of the channel with the largest baseline, in radar geometry (what the joint reads)

Cases: every scene of run_sweep_joint2d.py (ramps along azimuth, along range and oblique; sines / cosines; realistic
terrains) plus 12 scenes on more realistic terrains (fractal relief, fractal rolling hills, glacial valley
oblique to the flight direction, volcano; 1, 5 and 49 targets each), and EVERY baseline: Nrx = 2, 3, 4, 6, 8 with the
largest b_xt 100, 300 and 500 m (random, the largest exactly that value) and the fixed sets of four channels (u100,
u500, s10, s50, pm50). Channel 0 is always the transmitter (bat = bxt = 0). The joint reads only the DEM. Computation,
metrics and stages are those of run_sweep_joint2d.py (its run_stage is used as it is).

Run from sar_reconstruction/:
    python -m sata2d.run_sweep_joint2d_all --workers 3            # numbers (resumable; jobs already done in sweep_joint2d are copied)
    python -m sata2d.run_sweep_joint2d_all --pages                # the PNG pages
    python -m sata2d.run_sweep_joint2d_all --pdf                  # one PDF per family of geometries
    python -m sata2d.run_sweep_joint2d_all --table | --list
Output: ../results/sata2d/sweep_joint2d_all/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator

if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)

from . import run_sweep_joint2d as R
from .joint2d import dem_heights, image_positions
from .geometry import C0_LIGHT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "sata2d", "sweep_joint2d_all"))
WORK = os.path.join(tempfile.gettempdir(), "sweep_joint2d_all_work")
R.OUT, R.WORK = OUT, WORK
OSR = R.OSR


# ----------------------------------------------------------------------------------------------- new realistic terrains h(x, dy)
def fractal_terrain_2d(seed, relief, hurst, outer, inner, half_x=30e3, half_y=9e3, step=50.0, region=(22e3, 4e3), max_range_slope=None):
    """Random terrain with the power spectrum of real terrain, ~ k^-(2H+2) between the scales outer and inner [m]
    (Gaussian roll-off below inner), scaled so that inside |x| <= region[0], |dy| <= region[1] it goes from 0 to the relief;
    with max_range_slope [deg], the relief is instead set so that the steepest slope along ground range (|x| <= 20 km,
    |dy| <= 7 km) is that angle (no layover: the slope along range must stay below the incidence, ~20 deg)."""
    nx, ny = int(2 * half_x / step), int(2 * half_y / step)
    w = np.random.default_rng(seed).standard_normal((nx, ny))
    kx, ky = np.fft.fftfreq(nx, step)[:, None], np.fft.fftfreq(ny, step)[None, :]
    k2 = kx ** 2 + ky ** 2
    f = np.fft.ifft2(np.fft.fft2(w) * (k2 + outer ** -2) ** (-(hurst + 1.0) / 2.0) * np.exp(-k2 * inner ** 2)).real
    xa, ya = (np.arange(nx) - nx // 2) * step, (np.arange(ny) - ny // 2) * step
    sel = f[np.ix_(np.abs(xa) <= region[0], np.abs(ya) <= region[1])]
    f = (f - sel.min()) / np.ptp(sel)
    if max_range_slope is not None:
        box = np.ix_(np.abs(xa) <= 20e3, np.abs(ya) <= 7e3)
        relief = np.tan(np.radians(max_range_slope)) / np.abs(np.gradient(f, step, axis=1)[box]).max()
    return _bilinear(xa, ya, f * relief, step)


def _bilinear(xa, ya, f, step):
    nx, ny = f.shape

    def h(x, dy):
        x, dy = R._bc(x, dy)
        u = np.clip((x - xa[0]) / step, 0.0, nx - 1.001)
        v = np.clip((dy - ya[0]) / step, 0.0, ny - 1.001)
        i, j = u.astype(int), v.astype(int)
        a, b = u - i, v - j
        return (1 - a) * (1 - b) * f[i, j] + a * (1 - b) * f[i + 1, j] + (1 - a) * b * f[i, j + 1] + a * b * f[i + 1, j + 1]
    return h


def _valley():
    rough = fractal_terrain_2d(24, 30.0, 0.7, 2000.0, 200.0)
    c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))          # normal of the valley axis 30 deg off the flight direction

    def h(x, dy):
        x, dy = R._bc(x, dy)
        d = np.maximum(np.abs(x * c + dy * s - 300.0) - 700.0, 0.0)
        return 900.0 * (1.0 - np.exp(-(d / 3000.0) ** 2)) + rough(x, dy)
    return h


def _volcano():
    def h(x, dy):
        x, dy = R._bc(x, dy)
        rho2 = x ** 2 + (dy - 500.0) ** 2
        return 1000.0 * np.exp(-np.sqrt(rho2 + 300.0 ** 2) / 3500.0) - 150.0 * np.exp(-rho2 / 350.0 ** 2)
    return h


NEW_TERRAINS = {
    "fractal": (fractal_terrain_2d(31, None, 0.8, 20000.0, 300.0, max_range_slope=12.0),
                "fractal relief 0 -> 430 m (power spectrum of real mountain DEMs, detail down to 300 m), slopes along range up to 12$^\\circ$ (no layover)"),
    "rolling": (fractal_terrain_2d(32, 150.0, 1.0, 15000.0, 150.0),
                "fractal rolling hills 0 -> 150 m (power spectrum of real DEMs, detail down to 150 m)"),
    "valley": (_valley(), "glacial U-shaped valley, axis 60$^\\circ$ off the flight direction, walls up to 900 m, 30 m roughness"),
    "volcano": (_volcano(), "volcano about 850 m high with a summit crater 150 m deep, concave flanks"),
}
R.TERRAINS.update(NEW_TERRAINS)
NEW = [(f"{t}_{lay.replace('one_', '')}", t, lay) for t in NEW_TERRAINS for lay in ("one_flank", "diag5", "grid49")]
FAMILIES = [(fam, suffix, lst) for (fam, lst), suffix in zip(R.FAMILIES, ("ramps", "sines_cosines", "realistic"))]
FAMILIES.append(("More realistic terrains (fractal relief, valley, volcano)", "realistic_fractal", NEW))
R.SCENES.update({sid: (FAMILIES[-1][0], terr, lay) for sid, terr, lay in NEW})
TAGS = ([f"N2b{b}" for b in (100, 300, 500)] + [f"N3b{b}" for b in (100, 300, 500)] + [f"N4b{b}" for b in (100, 300, 500)]
        + list(R.FIXED) + [f"N6b{b}" for b in (100, 300, 500)] + [f"N8b{b}" for b in (100, 300, 500)])


PRIORITY = ["N4b300", "N2b300", "N8b300", "N3b300", "N6b300", "N4b100", "N4b500", "N2b100", "N2b500", "N8b100", "N8b500",
            "N3b100", "N3b500", "N6b100", "N6b500", "u100", "u500", "s10", "s50", "pm50"]                # order of the runs


def jobs():
    return [(sid, tag) for _, _, lst in FAMILIES for sid, _, _ in lst for tag in TAGS]


def adopt_done():
    """Copy the jobs already computed by run_sweep_joint2d.py (same code, same version) instead of running them again."""
    old = os.path.normpath(os.path.join(OUT, "..", "sweep_joint2d"))
    n = 0
    for sid, tag in jobs():
        k = f"{sid}__{tag}"
        src = {s: os.path.join(old, s, k + e) for s, e in (("numbers", ".json"), ("data", ".npz"))}
        if R._done(sid, tag) or not all(os.path.exists(v) for v in src.values()):
            continue
        if json.load(open(src["numbers"])).get("version") != R.VERSION:
            continue
        shutil.copyfile(src["data"], os.path.join(OUT, "data", k + ".npz"))
        shutil.copyfile(src["numbers"], os.path.join(OUT, "numbers", k + ".json"))
        n += 1
    return n


# ----------------------------------------------------------------------------------------------- page
def page(sid, tag):
    f = R._paths(sid, tag)
    res, z = json.load(open(f["num"])), np.load(f["data"])
    s = R.scene(sid, tag)
    p, terrain, xy, hs = s["p"], s["terrain"], s["xy"], s["hs"]
    pk = float(z["pk"])
    db = {k: 20 * np.log10(z[f"env_{k}"] / pk + 1e-12) for k in ("ref", "none", "joint")}
    t = (np.arange(p.Na) - p.Na // 2) / p.prf
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    fig = plt.figure(figsize=(19, 10.2))
    gs = GridSpec(2, 2, width_ratios=[1.3, 1.4], hspace=0.40, wspace=0.13, figure=fig, left=0.045, right=0.975, top=0.86, bottom=0.12)
    lab = res["label"] if res["label"].startswith("h =") else res["label"][0].upper() + res["label"][1:]
    fig.suptitle(f"{lab}\n$N_{{rx}}$ = {p.Nrx}, $b_{{xt}}$ = {R.bxt_text(p.bxt)} ({res['kind']}) | the joint reads only the DEM; "
                 "no method is given the positions of the targets", fontsize=12, y=0.99)
    # ---- azimuth IRFs
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["joint"], color="tab:purple", lw=1.3, label="joint, final version in 2-D (DEM applied at every pixel of the image)")
    ax.set_title(f"highest ambiguity: reference {res['ref']:.1f} dB | joint {res['joint']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"highest ambiguity: reference {res['ref']:.1f} dB | no SATA {res['none']:.1f} dB | improvement of the joint "
                 f"{res['none'] - res['joint']:.1f} dB", fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")
    # ---- the scene: from above (zoom on the targets) and in 3D (whole imaged area)
    gg = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.22, width_ratios=[1.0, 1.15])
    g2 = fig.add_subplot(gg[0, 0])
    lo = np.minimum(xy.min(axis=0), 0.0); hi = np.maximum(xy.max(axis=0), 0.0)
    pad = np.maximum(0.25 * (hi - lo), [1500.0, 1000.0])
    xg, yg = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 241), np.linspace(lo[1] - pad[1], hi[1] + pad[1], 161)
    hg = terrain(xg[:, None], yg[None, :])
    half = 0.5 * p.Na / p.prf * p.vs
    _, dyr = dem_heights(p, terrain, x=np.array([0.0]), r=p.r_scan[[0, -1]], osr=OSR)
    ylim = (min(float(dyr.min()), lo[1] - pad[1]), max(float(dyr.max()), hi[1] + pad[1]))
    X3, Y3 = np.meshgrid(np.linspace(-half, half, 161), np.linspace(ylim[0], ylim[1], 71), indexing="ij")
    H3 = terrain(X3, Y3)
    vmin, vmax = min(0.0, float(H3.min()), float(hg.min())), max(float(H3.max()), float(hs.max()), 1.0)
    cf = g2.contourf(xg, yg, hg.T, levels=np.linspace(vmin, vmax, 21), cmap="terrain", alpha=0.85)
    cb = fig.colorbar(cf, ax=g2, pad=0.02, fraction=0.06); cb.set_label("terrain height [m]", fontsize=8); cb.ax.tick_params(labelsize=7)
    cb.locator = MaxNLocator(6); cb.formatter = FuncFormatter(lambda v, _: f"{v:.0f}"); cb.update_ticks()
    g2.plot(xy[:, 0], xy[:, 1], "o", ms=6, color="#1F4E79", mec="w", mew=0.8, ls="none", zorder=5, label="point targets")
    if len(xy) <= 9:
        for (x, y), h in zip(xy, hs):
            g2.annotate(f"{h:.0f} m", (x, y), xytext=(6, 5), textcoords="offset points", fontsize=7.5, color="k",
                        bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.7))
    g2.plot(0.0, 0.0, marker="*", ms=14, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $\\Delta y$ = 0, $h$ = 0)")
    g2.annotate("", xy=(0.97, 0.06), xytext=(0.80, 0.06), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    g2.text(0.79, 0.06, "flight", transform=g2.transAxes, ha="right", va="center", fontsize=8)
    g2.text(0.03, 0.04, "radar on this side", transform=g2.transAxes, fontsize=8, va="bottom")
    g2.legend(fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False, borderaxespad=0.1)
    g2.set_title(f"From above, zoom on the targets | $\\Delta h_{{max}}$ = {res['dh']:.0f} m\nmax slope {res['max_slope_azimuth_deg']:.0f}$^\\circ$ "
                 f"along azimuth, {res['max_slope_range_deg']:.0f}$^\\circ$ along ground range", fontsize=9.5, pad=40)
    g2.xaxis.set_major_formatter(km); g2.yaxis.set_major_formatter(km)
    g2.set_xlabel("azimuth position $x$ [km]"); g2.set_ylabel("ground range $\\Delta y$ from the reference point [km]")
    g3 = fig.add_subplot(gg[0, 1], projection="3d")
    g3.plot_surface(X3, Y3, H3, cmap="terrain", vmin=vmin, vmax=vmax, alpha=0.75, linewidth=0, antialiased=True, rstride=1, cstride=1)
    g3.scatter(xy[:, 0], xy[:, 1], hs + 0.02 * vmax, s=16, color="#1F4E79", edgecolor="w", linewidths=0.4, depthshade=False, zorder=10)
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=11)
    g3.set_xlim(-half, half); g3.set_ylim(*ylim); g3.set_zlim(vmin, max(float(H3.max()), 50.0) * 1.6)
    g3.xaxis.set_major_formatter(km); g3.yaxis.set_major_formatter(km)
    g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4); g3.set_ylabel("ground range $\\Delta y$ [km]", fontsize=8, labelpad=4)
    g3.set_zlabel("height [m]", fontsize=8, labelpad=2); g3.tick_params(labelsize=7, pad=0)
    g3.view_init(elev=28, azim=-60); g3.set_box_aspect((1.7, 1.0, 0.55), zoom=1.2)
    g3.locator_params(axis="x", nbins=6); g3.locator_params(axis="y", nbins=4); g3.locator_params(axis="z", nbins=4)
    g3.set_title(f"In 3D: the whole imaged area\n({2 * half / 1e3:.0f} km of azimuth x {np.ptp(ylim) / 1e3:.0f} km of ground range)", fontsize=9.5)
    # ---- what the joint reads: the DEM table of the channel with the largest baseline, in radar geometry
    am = fig.add_subplot(gs[1, 1])
    i1 = int(z["channel"])
    xi, ri = image_positions(p, OSR)
    D = 1e3 * z["dC0"]
    vm = max(float(np.abs(D).max()), 1e-3)
    im = am.pcolormesh(xi[::4], (ri - p.r0), D.T, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto", rasterized=True)
    cb = fig.colorbar(im, ax=am, pad=0.015, fraction=0.04); cb.ax.tick_params(labelsize=7); cb.set_label("$\\Delta C_0$ [mm]", fontsize=8)
    rt = np.sqrt((p.y0 + xy[:, 1]) ** 2 + (p.H - hs) ** 2) - p.r0
    am.plot(xy[:, 0], rt, "o", ms=4.5, mfc="none", mec="k", mew=0.9, ls="none", label="where the point targets are (not given to the joint)")
    am.set_xlim(xi[0], xi[-1]); am.xaxis.set_major_formatter(km)
    am.set_xlabel("azimuth position $x$ [km]"); am.set_ylabel("slant range relative to the reference point [m]")
    am.set_title(f"$\\Delta C_0$ from the DEM, channel {i1} ($b_{{xt}}$ = {p.bxt[i1]:g} m, the largest baseline), azimuth x slant range "
                 "(what the joint reads)", fontsize=10)
    am.legend(fontsize=7.8, loc="upper right", framealpha=0.9)
    rho = C0_LIGHT / (2.0 * p.rsf)
    fig.text(0.5, 0.012, "Azimuth IRF: every target focused with its own 2-D matched filter, envelope over the targets. Highest ambiguity in dB relative to "
             f"the reference peak, inside +-20 ms of the ambiguity positions of every target (up to {max(res['orders'])} times the ambiguity distance of "
             "the channel PRF, on both sides).\nThe three curves are in the same processed band: azimuth bandwidth $B_a$ and, at every pixel, the range "
             f"frequencies that all the channels observe there ({100 * res['band_kept_median']:.0f} % of the range band at the median pixel, "
             f"{100 * res['band_kept_min']:.0f} % at the worst one).\n"
             f"$N_{{rx}}$ = {p.Nrx}, PRF = {p.prf:.0f} Hz, $B_a$ = {p.abw:.0f} Hz, range resolution {C0_LIGHT / (2 * p.rbw):.0f} m (range bin {rho:.0f} m), "
             f"channel 0 is the transmitter ($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {R.bxt_text(p.bxt)}.", ha="center", fontsize=8.5, color="0.35")
    out = os.path.join(OUT, "pages4", f"{sid}__{tag}.png")
    fig.savefig(out, dpi=85, bbox_inches="tight"); plt.close(fig)
    return out


def _page_path(sid, tag):
    return os.path.join(OUT, "pages4", f"{sid}__{tag}.png")


def _page_job(a):
    sid, tag = a
    try:
        return page(sid, tag)
    except Exception as e:                                                   # report and go on with the other pages
        return f"page failed: {sid} {tag}: {e!r}"


# ----------------------------------------------------------------------------------------------- driver
def load():
    return {(sid, tag): json.load(open(R._paths(sid, tag)["num"])) for sid, tag in jobs() if R._done(sid, tag)}


def table():
    res = load()
    json.dump([res[j] for j in jobs() if j in res], open(os.path.join(OUT, "all_numbers.json"), "w"), indent=1)
    for fam, _, lst in FAMILIES:
        print(fam)
        for sid, _, _ in lst:
            for tag in TAGS:
                r = res.get((sid, tag))
                if r:
                    print(f"  {sid:20s} {tag:7s} Nrx {r['nrx']} targets {r['n_targets']:2d} | ref {r['ref']:6.1f} none {r['none']:6.1f} "
                          f"joint {r['joint']:6.1f} | unexplained {r['unexplained_dB']:6.1f} dB")


def bundle(pngs, out, outline):
    """Multi-page PDF (JPEG pages) with bookmarks, one page at a time (little memory)."""
    from PIL import Image
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter()
    tmp = os.path.join(tempfile.gettempdir(), "sweep_joint2d_all_page.pdf")
    for q in pngs:
        Image.open(q).convert("RGB").save(tmp, resolution=85, quality=75)
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
    for fam, suffix, lst in FAMILIES:
        pngs, outline = [], []
        for sid, _, _ in lst:
            first = True
            for tag in TAGS:
                pp = _page_path(sid, tag)
                if (sid, tag) not in res or not os.path.exists(pp):
                    continue
                r = res[(sid, tag)]
                if first:
                    lab = r["label"].replace("$\\pi$", "pi").replace("$^\\circ$", " deg").replace("$\\Delta y$", "dy")
                    outline.append((lab, len(pngs), 0)); first = False
                outline.append((f"{r['nrx']} channels, bxt {R.bxt_text(r['bxt'])}", len(pngs), 1))
                pngs.append(pp)
        if pngs:
            done.append(f"{bundle(pngs, os.path.join(OUT, f'sweep_joint_all_2d_{suffix}.pdf'), outline)} ({len(pngs)} pages)")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, help="scene ids (default: all)")
    ap.add_argument("--budget", type=float, default=1e9, help="seconds: no stage is started that would not fit")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pages", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tags", nargs="+", default=None, help="baselines (default: all)")
    ap.add_argument("--pack", type=float, nargs=2, default=None, metavar=("A", "B"),
                    help="duration of one stage taken as A * Nrx + B seconds (to fill a time budget with stages that fit)")
    a = ap.parse_args()
    if a.pack:
        R.stage_time = lambda tag, stage: a.pack[0] * len(R.baseline(tag)[0]) + a.pack[1]
    sel = [(sid, tag) for sid, tag in jobs() if (a.only is None or sid in a.only) and (a.tags is None or tag in a.tags)]
    if a.list:
        for sid, tag in sel:
            print(f"{sid:20s} {tag:7s} bxt {R.bxt_text(R.baseline(tag)[0])} | {R.TERRAINS[R.SCENES[sid][1]][1]}; {R.LAYOUTS[R.SCENES[sid][2]][1]}")
        return print(len(sel), "jobs")
    if a.table:
        return table()
    if a.pdf:
        return print("wrote\n" + "\n".join(pdf()))
    for sub in ("numbers", "data", "pages4"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    if a.pages:
        todo = [(sid, tag) for sid, tag in sel if R._done(sid, tag) and not os.path.exists(_page_path(sid, tag))]
        print(len(todo), "pages to do", flush=True)
        if a.workers > 1 and len(todo) > 1:
            from multiprocessing import Pool
            with Pool(a.workers, maxtasksperchild=25) as pool:
                for line in pool.imap_unordered(_page_job, todo):
                    print(line, flush=True)
        else:
            for x in todo:
                print(_page_job(x), flush=True)
        return
    if not a.tags:
        print(adopt_done(), "jobs copied from sweep_joint2d", flush=True)
    deadline = time.time() + a.budget
    todo = [(sid, tag) for sid, tag in sel if not R._done(sid, tag)]
    todo.sort(key=lambda j: (not os.path.exists(R._paths(*j)["state"]), PRIORITY.index(j[1])))   # every scene with the main baselines first
    print(len(todo), "jobs to do", flush=True)
    args = [(sid, tag, deadline) for sid, tag in todo]
    if a.workers > 1 and len(args) > 1:
        from multiprocessing import Pool
        with Pool(a.workers, initializer=R._set_out, initargs=(OUT, WORK), maxtasksperchild=10) as pool:
            for lines in pool.imap_unordered(R._work, args):
                for line in lines:
                    print(line, flush=True)
    else:
        for x in args:
            for line in R._work(x):
                print(line, flush=True)
    print("left:", sum(not R._done(sid, tag) for sid, tag in sel), "jobs", flush=True)


if __name__ == "__main__":
    main()
