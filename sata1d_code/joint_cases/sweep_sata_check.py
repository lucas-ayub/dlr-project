"""
Check sweep of SATA 1D (DEM only) before the standard azimuth reconstruction: SATA whole band and SATA per sub-band,
with the oracle (every target corrected with its own delta_C0) as the level a correct per-target phase reaches.
Controls (elevated flat terrain, terrain of period one fold), one target, a few and many targets on ramps and realistic
terrains, channel 0 at the transmitter. Page layout of sweep_sata.py (IRF whole band / IRF per sub-band, scene zoom,
whole line in 3D, channels seen from above, DEM table), with the oracle added to both IRF panels.

    python joint_cases/sweep_sata_check.py --out <folder> --workers 4     # numbers and pages (resumable)
    python joint_cases/sweep_sata_check.py --out <folder> --pdf
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
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
import audit_dem as AD
import sweep_sata as SW
import run_joint_sweep as W
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy

VERSION = 1
FAMILIES = [
    ("Controls: terrains where one phase per Doppler bin is right (flat, period of one fold) and one where it is not", [
        ("ctrl_flat500_1", "N4b300"), ("ctrl_flat500_25", "N4b500"), ("ctrl_periodX_5", "N4b100"), ("ctrl_period1p5X_5", "N4b100")]),
    ("One target", [
        ("one_refpoint", "u100"), ("one_ramp8", "N4b300"), ("one_flank", "N4b100"), ("one_flank", "N4b500"), ("one_peak", "N4b300"),
        ("one_escarpment", "N4b300"), ("one_mountains", "N4b500"), ("one_hill", "N2b300")]),
    ("A few targets", [
        ("few_ramp4_5", "N4b100"), ("few_ramp20_7", "N4b300"), ("few_ramp15_9", "N4b500"), ("few_hills_5", "N4b300"),
        ("few_mountains_5", "N4b100"), ("few_mountains_5", "N8b500"), ("few_escarpment_9", "N4b300"), ("few_peak_5", "N6b300"),
        ("few_plateau_5", "N4b500"), ("few_cos20km_5", "N3b300")]),
    ("Many targets", [
        ("many_ramp3_49", "N4b100"), ("many_ramp3_49", "N4b500"), ("many_ramp8_49", "N4b300"), ("many_ramp15_25", "N4b300"),
        ("many_hills_49", "N4b300"), ("many_hills_49", "N8b100"), ("many_mountains_49", "N4b100"), ("many_mountains_49", "N4b500"),
        ("many_mountains_49", "N6b300"), ("many_escarpment_25", "N3b500"), ("many_peak_33", "N4b300"), ("many_sin50km_31", "N2b500")]),
]
OUT = None
ORACLE = "oracle (every target corrected with its own phase)"


def jobs():
    return [j for _, js in FAMILIES for j in js]


def compute(sc, tag):
    res, d = SW.compute(sc, tag)
    cfg, tr = d["cfg"], d["tr"]
    dC0 = SS.delta_C0_per_target(cfg, tr)
    och = np.zeros((cfg.Nrx, cfg.Na_ch), complex)
    for j, p in enumerate(cfg.scene.points[1:]):
        och += S.generate_channels(cfg, tr, ptgs=[p], amps=[cfg.scene.amps[j]]) * np.exp(2j * np.pi * dC0[j] / cfg.system.wl)[:, None]
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    d["env"]["oracle"] = np.abs(np.roll(np.fft.ifft(np.fft.fft(S.reconstruct(cfg, tr, och)) * Hf), cfg.Na // 2))
    amb = AD.ambiguity_level(d["t"], d["env"]["oracle"], d["pk"], d["c"]["xs"], cfg.system.vs,
                             cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2), orders=tuple(res["orders"]))
    res["oracle"] = d["amb"]["oracle"] = float(amb)
    res["version"] = VERSION
    return res, d


def page(res, d, out_png):
    c, cfg, env, amb, t, pk, Mch = d["c"], d["cfg"], d["env"], d["amb"], d["t"], d["pk"], d["Mch"]
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    fig = plt.figure(figsize=(22, 10.6))
    gs = GridSpec(2, 3, width_ratios=[1.32, 0.86, 1.0], hspace=0.38, wspace=0.2, figure=fig,
                  left=0.04, right=0.985, top=0.86, bottom=0.115)
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
        a_.plot(t, db["oracle"], color="0.45", lw=0.9, label=ORACLE)
        a_.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
        a_.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
        a_.plot(t, db[key], color=col, lw=1.3, ls=ls, label=f"{name} (DEM)")
        a_.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | {name} {amb[key]:.1f} dB"
                     f" | oracle {amb['oracle']:.1f} dB", fontsize=10)
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]")
        hd, lb = a_.get_legend_handles_labels()
        a_.legend([hd[i] for i in (1, 2, 3, 0)], [lb[i] for i in (1, 2, 3, 0)], fontsize=8.5, loc="upper left")
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
             "look-up table from the terrain (DEM); neither uses the positions of the targets.\nThe oracle multiplies the echo of every "
             "target by its own phase before the reconstruction (it knows the targets, so it is only a yardstick for what a correct phase per target reaches).\n"
             f"$N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz, channel 0 is the transmitter "
             f"($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)}.",
             ha="center", fontsize=8.5, color="0.35")
    fig.savefig(out_png, dpi=90); plt.close(fig)


def short(r):
    s = r["label"].replace("$\\pi$", "pi").replace("$^\\circ$", " deg").replace("$\\pm$", "+-")
    for cut in (", flat before and after", " (6.23 km = one fold of the channel PRF on the ground)",
                " (the DEM is the constant 500 m)"):
        s = s.replace(cut, "")
    return f"{s} | {r['nrx']} channels, largest b_xt {max(abs(b) for b in r['bxt']):g} m"


def summary(res, out_png):
    rows = [(fi, j) for fi, (_, js) in enumerate(FAMILIES) for j in js if j in res]
    n = len(rows)
    H = 0.42 * n + 2.6
    fig, ax = plt.subplots(figsize=(21, H))
    fig.subplots_adjust(left=0.47, right=0.97, top=1 - 0.9 / H, bottom=1.1 / H)
    y = np.arange(n)[::-1]
    sty = [("ref", "Reference", "tab:green", "o", 70), ("oracle", "oracle", "0.35", "|", 160), ("none", "no SATA", "tab:red", "x", 60),
           ("whole", "SATA whole band", "tab:blue", "s", 46), ("sub", "SATA per sub-band", "tab:orange", "D", 40)]
    for k, name, col, m, s in sty:
        v = [res[j][k] for _, j in rows]
        ax.scatter(v, y, marker=m, s=s, color=col, label=name, zorder=3, lw=1.6 if m in "x|" else 0.6,
                   edgecolor=None if m in "x|" else "k")
    for i, (_, j) in enumerate(rows):
        r = res[j]
        ax.plot([min(r[k] for k, *_ in sty), max(r[k] for k, *_ in sty)], [y[i]] * 2, color="0.8", lw=0.8, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([short(res[j]) for _, j in rows], fontsize=8.2)
    prev = None
    for i, (fi, _) in enumerate(rows):
        if fi != prev:
            if prev is not None:
                ax.axhline(y[i] + 0.5, color="0.4", lw=1)
            ax.text(0.995, y[i] - 0.1, FAMILIES[fi][0].split(":")[0], transform=ax.get_yaxis_transform(), ha="right", va="top",
                    fontsize=10, weight="bold")
            prev = fi
    ax.set_xlabel("highest ambiguity [dB relative to the reference peak]")
    ax.set_ylim(-0.7, n - 0.3); ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=10)
    fig.savefig(out_png, dpi=110); plt.close(fig)


def _paths(sc, tag):
    k = f"{sc}__{tag}"
    return os.path.join(OUT, "numbers", k + ".json"), os.path.join(OUT, "pages", k + ".png")


def _done(j):
    pj, pp = _paths(*j)
    return os.path.exists(pj) and json.load(open(pj)).get("version") == VERSION and os.path.exists(pp)


def _job(j):
    t0 = time.time()
    res, d = compute(*j)
    pj, pp = _paths(*j)
    page(res, d, pp)
    json.dump(res, open(pj, "w"), indent=1)
    return (f"{j[0]:18s} {j[1]:7s} ref {res['ref']:6.1f} none {res['none']:6.1f} whole {res['whole']:6.1f} "
            f"sub {res['sub']:6.1f} oracle {res['oracle']:6.1f} ({time.time() - t0:.0f} s)")


def load():
    out = {}
    for j in jobs():
        pj = _paths(*j)[0]
        if os.path.exists(pj):
            out[j] = json.load(open(pj))
    return out


def pdf():
    from joint_sweep_summary import bundle
    res = load()
    first = os.path.join(OUT, "pages", "_summary.png")
    summary(res, first)
    pngs, outline = [first], [("Highest ambiguity of every case", 0, 0)]
    for fam, js in FAMILIES:
        outline.append((fam.split(":")[0], len(pngs), 0))
        for j in js:
            if j in res:
                outline.append((short(res[j]), len(pngs), 1)); pngs.append(_paths(*j)[1])
    return f"{bundle(pngs, os.path.join(OUT, 'sweep_sata_check.pdf'), outline)} ({len(pngs)} pages)"


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "sweep_sata_check"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    OUT = a.out
    if a.pdf:
        return print("wrote", pdf())
    for sub in ("numbers", "pages"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    todo = [j for j in jobs() if not _done(j) and (not a.only or j[0] in a.only)]
    print(len(todo), "jobs", flush=True)
    if a.workers > 1:
        with Pool(a.workers, initializer=_init, initargs=(OUT,)) as p:
            for line in p.imap_unordered(_job, todo):
                print(line, flush=True)
    else:
        for j in todo:
            print(_job(j), flush=True)


def _init(out):
    global OUT
    OUT = out


if __name__ == "__main__":
    main()
