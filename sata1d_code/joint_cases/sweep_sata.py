"""
Small sweep of SATA 1D (DEM only) before the standard azimuth reconstruction: ramps and sines / cosines, 1, a few and
many targets, four channels and one other number of channels per scene (largest b_xt 100, 300 or 500 m, channel 0 at
the transmitter). Six-panel pages as in sweep_final.py, with the two SATA versions in the IRF panels:
    top IRF:    Reference + no SATA + SATA whole band
    bottom IRF: Reference + no SATA + SATA per sub-band
plus a first page with the highest ambiguity of every case. The scenes, baselines, DEM table, SATA calls and the
measure of the highest ambiguity are those of audit_dem.py / sweep_final.py (not changed).

    python joint_cases/sweep_sata.py --workers 4     # numbers and pages (resumable), from sata1d_code
    python joint_cases/sweep_sata.py --pdf
    python joint_cases/sweep_sata.py --table
Output: joint_cases/sweep_sata/numbers/<job>.json, pages/<job>.png, sweep_sata.pdf
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
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
import sata1d.sata as SS
from sata1d.config import ArrayGeometry
import audit_dem as AD
import sweep_final as SF
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy
import run_joint_sweep as W

OUT = os.path.join(HERE, "sweep_sata")
VERSION = 1
FAMILIES = [
    ("Ramps", ["one_ramp8", "few_ramp20_7", "few_ramp15_9", "many_ramp3_49", "many_ramp8_49"]),
    ("Sines and cosines", ["one_refpoint", "one_sine", "few_cos20km_5", "few_cos5km_7", "many_sin50km_31", "many_sin5km_31"]),
]
FOUR = ["N4b100", "N4b300", "N4b500"]
OTHER = ["N2b300", "N3b500", "N6b100", "N8b300", "N2b500", "N3b100", "N6b500", "N8b100", "N2b100", "N3b300"]
# scenes with their own baselines (outside the rotation): the case REF-A100-P50 of the stress tests, with the same
# four-channel set and channel 0 at the transmitter
EXPLICIT = {"one_refpoint": ["u100"]}


def jobs():
    out, i = [], 0
    for _, scenes in FAMILIES:
        for sc in scenes:
            if sc in EXPLICIT:
                out += [(sc, t) for t in EXPLICIT[sc]]
                continue
            tags = [FOUR[i % len(FOUR)], OTHER[i % len(OTHER)]]
            out += [(sc, t) for t in sorted(tags, key=lambda t: len(AD.baseline(t)[0]))]
            i += 1
    return out


def compute(sc, tag):
    """audit_dem.compute without the joint versions: reference, no SATA, SATA whole band, SATA per sub-band."""
    grp, fn = AD.SCENES[sc]
    c = fn()
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    bxt, kind = AD.baseline(tag)
    cfg = S.make_config(list(zip(xs, hs)), nrx=len(bxt), bxt_max=0.0, amplitudes=c["amps"], tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy()))
    tr = S.build_platform_tracks(cfg)
    ch = S.generate_channels(cfg, tr)                                    # the simulated echoes of the point targets
    # ---- the only knowledge of the scene given to SATA: the terrain (DEM)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    hgrid = terrain(np.linspace(-half - 100.0, half + 100.0, 8001))
    tab = AD.height_table(cfg, tr, hgrid.min(), hgrid.max())
    Mch = AD.channel_maps(cfg, tr, terrain, tab)                         # [Nrx, Na_ch] DEM tables of the channels
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    env, sec = {"ref": F(S.generate_reference(cfg, tr))}, {}
    for key, f in (("none", lambda: S.reconstruct(cfg, tr, ch)), ("whole", lambda: AD.sata_whole(cfg, tr, ch, Mch)),
                   ("sub", lambda: AD.sata_sub(cfg, tr, ch, Mch))):
        t1 = time.time(); env[key] = F(f()); sec[key] = time.time() - t1
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt_amb = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    orders = SF.orders_of(cfg)
    amb = {k: float(AD.ambiguity_level(t, v, pk, xs, cfg.system.vs, dt_amb, orders=orders)) for k, v in env.items()}
    lo, hi = min(xs.min(), 0.0), max(xs.max(), 0.0); pad = max(0.12 * (hi - lo), 500.0)
    res = dict(amb, scene=sc, baseline=tag, group=grp, nrx=cfg.Nrx, bxt=[float(b) for b in cfg.array.bxt], kind=kind,
               n_targets=int(len(xs)), label=c["label"], max_slope_deg=W.max_slope_deg(terrain, lo - pad, hi + pad),
               dh=float(np.ptp(terrain(np.linspace(lo - pad, hi + pad, 2001)))), seconds=sec, orders=list(orders),
               version=VERSION)
    return res, dict(c=c, cfg=cfg, tr=tr, env=env, amb=amb, t=t, pk=pk, Mch=Mch, kind=kind)


# ----------------------------------------------------------------------------------------------- page of one case
def page(res, d, out_png):
    """sweep_final.page with the two SATA versions in the IRF panels (the other four panels unchanged)."""
    c, cfg, env, amb, t, pk, Mch = d["c"], d["cfg"], d["env"], d["amb"], d["t"], d["pk"], d["Mch"]
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    fig = plt.figure(figsize=(22, 10.6))
    gs = GridSpec(2, 3, width_ratios=[1.32, 0.86, 1.0], hspace=0.38, wspace=0.2, figure=fig,
                  left=0.04, right=0.985, top=0.86, bottom=0.1)
    lab = c["label"] if c["label"].startswith("h =") else c["label"][0].upper() + c["label"][1:]
    head = f"{lab} | $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)} ({d['kind']})"
    tail = "SATA reads only the DEM (look-up table from the terrain)"
    if len(head) + len(tail) > 185:
        fig.suptitle(head + "\n" + tail, fontsize=12.5, y=0.985)
    else:
        fig.suptitle(head + " | " + tail, fontsize=13, y=0.975)
    for row, (key, name, col, ls) in enumerate((("whole", "SATA whole band", "tab:blue", (0, (1, 1.5))),
                                                 ("sub", "SATA per sub-band", "tab:orange", (0, (3, 1.5))))):
        a_ = fig.add_subplot(gs[row, 0])
        a_.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
        a_.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
        a_.plot(t, db[key], color=col, lw=1.3, ls=ls, label=f"{name} (DEM)")
        a_.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | {name} {amb[key]:.1f} dB", fontsize=10)
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")
    # scene, zoom on the targets
    xs_, dys, hs_ = scene_coordinates(xs, hs)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    g2 = fig.add_subplot(gs[0, 1])
    lo2, hi2 = min(xs_.min(), 0.0), max(xs_.max(), 0.0); pad2 = max(0.12 * (hi2 - lo2), 500.0)
    xg = np.linspace(-half, half, 4801); hg = terrain(xg)
    hz = hg[(xg >= lo2 - pad2) & (xg <= hi2 + pad2)]
    geometry_2d(g2, xs_, dys, hs_, False, title=None, connect=False, annotate=len(xs_) <= 13)
    g2.fill_between(xg, min(hg.min(), 0) - 0.05 * max(np.ptp(hg), 1.0), hg, color="#D7C9A7", alpha=0.45, lw=0)
    g2.plot(xg, hg, color="#6D4C2F", lw=1.2, zorder=2, label="terrain (the scene)")
    ylo, yhi = min(hz.min(), 0.0), max(hz.max(), hs_.max(), 1.0)
    g2.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))
    g2.plot(0.0, 0.0, marker="*", ms=13, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $h$ = 0)")
    g2.set_xlim(lo2 - pad2, hi2 + pad2)
    g2.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.1)
    g2.set_title(f"Scene: zoom on the targets | $\\Delta h_{{max}}$ = {res['dh']:.0f} m | max slope {res['max_slope_deg']:.0f}$^\\circ$", fontsize=10, pad=34)
    g2.xaxis.set_major_formatter(km); g2.set_xlabel("azimuth position $x$ [km]")
    # scene, whole line in 3D
    g3 = fig.add_subplot(gs[0, 2], projection="3d")
    geometry_3d(g3, xs_, dys, hs_, False, connect=False, title=None)
    g3.plot(xg, iso_range_dy(hg), hg, color="#6D4C2F", lw=1.0)
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=6)
    g3.set_xlim(-half, half); g3.xaxis.set_major_formatter(km); g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4)
    g3.set_zlim(min(0.0, hg.min()), max(hg.max(), 50.0) * 1.15)
    g3.set_title(f"Scene: whole azimuth line ({2 * half / 1e3:.0f} km) in 3D (x, $\\Delta y$, h)", fontsize=10)
    for t_ in list(g3.texts):
        t_.remove()
    # channels
    AD.channel_geometry(fig.add_subplot(gs[1, 1]), cfg)
    # the DEM table
    am = fig.add_subplot(gs[1, 2])
    n = np.arange(cfg.Na_ch)
    for i in range(cfg.Nrx):
        am.step(n, 1e3 * Mch[i], where="mid", color=AD.CH[i], lw=1.3, label=f"channel {i} ($b_{{xt}}$ = {cfg.array.bxt[i]:g} m)")
    i1 = int(np.argmax(np.abs(cfg.array.bxt)))
    cells1 = np.array([SS.az_pixel_of_scatterer(cfg, x, i1) for x in xs])
    if len(cells1) <= 60:
        for cc in cells1:
            am.axvline(cc, color="0.75", lw=0.6, zorder=0)
    am.plot(cells1, 1e3 * Mch[i1][np.clip(cells1, 0, cfg.Na_ch - 1)], "o", ms=5, color=AD.CH[i1], mec="k", zorder=5,
            label=f"cells of the point targets (channel {i1}; not given to SATA)")
    am.set_xlim(0, cfg.Na_ch - 1); am.grid(alpha=0.3)
    am.set_xlabel("azimuth cell of the channel line"); am.set_ylabel("$\\Delta C_0$ in the table [mm]")
    lo_, hi_ = am.get_ylim(); am.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    am.set_title("$\\Delta C_0$ per cell in the DEM table of every channel (look-up table from the terrain)", fontsize=10)
    am.legend(fontsize=7.8 if cfg.Nrx <= 4 else 6.6, loc="upper center", ncol=2)
    nmax = max(res["orders"])
    fig.text(0.5, 0.012, "Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions of every "
             f"target (up to {nmax} times the ambiguity distance of the channel PRF, on both sides). Both SATA versions use the same "
             f"look-up table from the terrain (DEM); neither uses the positions of the targets.\n$N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, "
             f"$B_a$ = {cfg.abw:.0f} Hz, channel 0 is the transmitter ($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)}.",
             ha="center", fontsize=8.5, color="0.35")
    fig.savefig(out_png, dpi=90); plt.close(fig)


# ----------------------------------------------------------------------------------------------- summary page
SHORT = {"one_ramp8": "ramp of 8 deg (h 0 -> 1686 m), 1 target", "few_ramp20_7": "ramp of 20 deg (h 0 -> 437 m), 7 targets",
         "few_ramp15_9": "ramp of 15 deg (h 0 -> 804 m), 9 targets", "many_ramp3_49": "ramp of 3 deg (h 0 -> 629 m), 49 targets",
         "many_ramp8_49": "ramp of 8 deg (h 0 -> 1686 m), 49 targets",
         "one_refpoint": "h = 100 - 100 cos(2pi x / 50 km), 1 target at h(0) = 0"}


def short(r):
    """Case name for the summary page and the PDF outline (the pages carry the full b_xt)."""
    if r["scene"] in SHORT:
        geo = SHORT[r["scene"]]
    else:
        n = r["n_targets"]
        geo = r["label"].split(",")[0].replace("$\\pi$", "pi") + f", {n} target" + ("s" if n > 1 else "")
    return f"{geo} | {r['nrx']} channels, largest b_xt {max(abs(b) for b in r['bxt']):g} m"


def summary(res, out_png):
    rows = [(fam, sc, tag) for fam, scenes in FAMILIES for sc, tag in jobs() if sc in scenes and (sc, tag) in res]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(15, 0.42 * n + 2.2))
    fig.subplots_adjust(left=0.38, right=0.97, top=1 - 0.9 / (0.42 * n + 2.2), bottom=1.1 / (0.42 * n + 2.2))
    y = np.arange(n)[::-1]
    sty = [("ref", "Reference", "tab:green", "o", 70), ("none", "no SATA", "tab:red", "x", 60),
           ("whole", "SATA whole band", "tab:blue", "s", 46), ("sub", "SATA per sub-band", "tab:orange", "D", 40)]
    for k, name, col, m, s in sty:
        v = [res[(sc, tag)][k] for _, sc, tag in rows]
        ax.scatter(v, y, marker=m, s=s, color=col, label=name, zorder=3, lw=1.6 if m == "x" else 0.6,
                   edgecolor=None if m == "x" else "k")
    for i, (_, sc, tag) in enumerate(rows):
        r = res[(sc, tag)]
        ax.plot([min(r[k] for k, *_ in sty), max(r[k] for k, *_ in sty)], [y[i]] * 2, color="0.8", lw=0.8, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([short(res[(sc, tag)]) for _, sc, tag in rows], fontsize=8.6)
    split = sum(1 for f, *_ in rows if f == FAMILIES[0][0])
    ax.axhline(n - split - 0.5, color="0.4", lw=1)
    ax.text(1.0, n - 0.6, FAMILIES[0][0], transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=10, weight="bold")
    ax.text(1.0, n - split - 0.7, FAMILIES[1][0], transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=10, weight="bold")
    ax.set_xlabel("highest ambiguity [dB relative to the reference peak]")
    ax.set_ylim(-0.7, n - 0.3); ax.grid(axis="x", alpha=0.3); ax.grid(axis="y", alpha=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=10)
    fig.savefig(out_png, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------------------------- driver
def _paths(sc, tag):
    k = f"{sc}__{tag}"
    return os.path.join(OUT, "numbers", k + ".json"), os.path.join(OUT, "pages", k + ".png")


def _done(sc, tag):
    pj, pp = _paths(sc, tag)
    return os.path.exists(pj) and json.load(open(pj)).get("version") == VERSION and os.path.exists(pp)


def _job(a):
    sc, tag = a
    t0 = time.time()
    res, d = compute(sc, tag)
    pj, pp = _paths(sc, tag)
    page(res, d, pp)
    json.dump(res, open(pj, "w"), indent=1)
    return f"{sc:18s} {tag:7s} ref {res['ref']:6.1f} none {res['none']:6.1f} whole {res['whole']:6.1f} sub {res['sub']:6.1f} ({time.time() - t0:.0f} s)"


def load():
    out = {}
    for sc, tag in jobs():
        pj = _paths(sc, tag)[0]
        if os.path.exists(pj):
            out[(sc, tag)] = json.load(open(pj))
    return out


def table():
    res = load()
    for fam, scenes in FAMILIES:
        print(fam)
        for sc, tag in jobs():
            r = res.get((sc, tag))
            if r and sc in scenes:
                print(f"  {short(r):95s} ref {r['ref']:6.1f} none {r['none']:6.1f} whole {r['whole']:6.1f} sub {r['sub']:6.1f}")
    v = lambda k: np.median([r[k] for r in res.values()])
    print(f"median of {len(res)}: ref {v('ref'):.1f} none {v('none'):.1f} whole {v('whole'):.1f} sub {v('sub'):.1f}")


def pdf():
    from joint_sweep_summary import bundle
    res = load()
    first = os.path.join(OUT, "pages", "_summary.png")
    summary(res, first)
    pngs, outline = [first], [("Highest ambiguity of every case", 0, 0)]
    for fam, scenes in FAMILIES:
        head = True
        for sc, tag in jobs():
            pp = _paths(sc, tag)[1]
            if sc not in scenes or (sc, tag) not in res or not os.path.exists(pp):
                continue
            if head:
                outline.append((fam, len(pngs), 0)); head = False
            outline.append((short(res[(sc, tag)]), len(pngs), 1))
            pngs.append(pp)
    return f"{bundle(pngs, os.path.join(OUT, 'sweep_sata.pdf'), outline)} ({len(pngs)} pages)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for sc, tag in jobs():
            print(f"{sc:18s} {tag:7s} bxt {W.bxt_text(AD.baseline(tag)[0])}")
        return print(len(jobs()), "jobs")
    if a.table:
        return table()
    if a.pdf:
        return print("wrote", pdf())
    for sub in ("numbers", "pages"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    todo = [j for j in jobs() if not _done(*j)]
    print(len(todo), "jobs to do", flush=True)
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for line in pool.imap_unordered(_job, todo):
                print(line, flush=True)
    else:
        for j in todo:
            print(_job(j), flush=True)
    print("left:", sum(not _done(*j) for j in jobs()), flush=True)


if __name__ == "__main__":
    main()
