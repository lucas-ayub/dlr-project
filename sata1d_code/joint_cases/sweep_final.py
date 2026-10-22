"""
Sweep of the final joint (joint reconstruction over the whole line, the DEM applied at every position), six-panel pages
in two versions (same cases, same numbers):

    A  sweep_final/sweep_final_joint.pdf        top IRF: reference + joint | bottom IRF: reference + no SATA
    B  sweep_final/sweep_final_joint_sata.pdf   top IRF: reference + joint | bottom IRF: reference + no SATA
                                                                              + SATA whole band + SATA per sub-band

Cases: the scenes of audit_dem.py (ramps, sines / cosines, realistic terrains; 1, a few and many targets) with the
number of channels (2, 3, 4, 6, 8) and the cross-track baselines (largest 100, 300, 500 m, and the fixed sets of four
channels) rotating over the scenes: every scene with four channels and with one other number of channels, the realistic
terrains with a third one. Channel 0 is always the transmitter (bat = bxt = 0). Every method reads only the DEM.

    python joint_cases/sweep_final.py --budget 160 --workers 4     # numbers and pages (resumable), from sata1d_code
    python joint_cases/sweep_final.py --pdf                        # the two PDFs from the pages
    python joint_cases/sweep_final.py --table                      # text table of the numbers
Output: joint_cases/sweep_final/numbers/<job>.json, pages_joint/<job>.png, pages_joint_sata/<job>.png
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import audit_dem as AD
import sata1d.sata as SS
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy
import run_joint_sweep as W

OUT = os.path.join(HERE, "sweep_final")
VERSION = 1
VARIANTS = {"joint": ("pages_joint", "sweep_final_joint.pdf"), "joint_sata": ("pages_joint_sata", "sweep_final_joint_sata.pdf")}
FAMILIES = [   # (title in the PDF outline, scenes in the order of the pages)
    ("Ramps", ["one_ramp8", "few_ramp5_2", "few_ramp20_7", "few_ramp4_5", "few_ramp15_9", "many_ramp15_25", "many_ramp3_49",
               "many_ramp5_49", "many_ramp8_49"]),
    ("Sines and cosines", ["one_refpoint", "one_sine", "few_sin50km_5", "few_cos20km_5", "few_cos5km_7", "many_sin50km_31",
                           "many_sin5km_31"]),
    ("Realistic terrains", ["one_hill", "one_mountains", "one_peak", "one_flank", "one_escarpment", "few_hills_5",
                            "few_mountains_5", "few_peak_5", "few_escarpment_5", "few_escarpment_9", "few_plateau_5",
                            "few_plateau_9", "many_mountains_13", "many_mountains_25", "many_mountains_49", "many_hills_49",
                            "many_escarpment_25", "many_peak_33"]),
]
FOUR = ["N4b300", "u500", "N4b100", "s50", "N4b500", "u100", "pm50", "s10"]                    # four channels
OTHER = ["N3b300", "N6b100", "N2b500", "N8b300", "N3b100", "N6b500", "N2b300", "N8b100", "N3b500", "N6b300", "N2b100", "N8b500"]


def jobs():
    """(scene, baseline) in the order of the pages: per family, per scene, by number of channels."""
    out, i = [], 0
    for fam, scenes in FAMILIES:
        for sc in scenes:
            tags = [FOUR[i % len(FOUR)], OTHER[i % len(OTHER)]]
            if fam == "Realistic terrains":
                tags.append(OTHER[(i + 6) % len(OTHER)])
            out += [(sc, t) for t in sorted(tags, key=lambda t: len(AD.baseline(t)[0]))]
            i += 1
    return out


def orders_of(cfg):
    """Ambiguity orders that can hold energy: |m| PRF_op below the Doppler bandwidth (at least +-1, +-2 as in the audit)."""
    B = 4.0 * cfg.system.vs * np.sin(cfg.theta_tx / 2.0) / cfg.system.wl
    n = max(2, int(np.floor(B / cfg.PRF_op)))
    return tuple(m for m in range(-n, n + 1) if m != 0)


def compute(sc, tag):
    res, d = AD.compute(sc, tag)
    cfg, xs = d["cfg"], d["c"]["xs"]
    dt_amb = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    orders = orders_of(cfg)
    amb = {k: float(AD.ambiguity_level(d["t"], v, d["pk"], xs, cfg.system.vs, dt_amb, orders=orders)) for k, v in d["env"].items()}
    res.update(amb, orders=list(orders), version=VERSION)
    d["amb"] = amb
    return res, d


# ----------------------------------------------------------------------------------------------- page
def page(res, d, out_png, variant):
    """Six panels: two IRF panels (left), the scene in 2D (zoom on the targets) and in 3D (whole line), the channels seen
    from above and the DEM table. variant 'joint': bottom IRF = reference + no SATA; 'joint_sata': + the two SATA."""
    c, cfg, env, amb, t, pk, Mch = d["c"], d["cfg"], d["env"], d["amb"], d["t"], d["pk"], d["Mch"]
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    fig = plt.figure(figsize=(22, 10.6))
    gs = GridSpec(2, 3, width_ratios=[1.32, 0.86, 1.0], hspace=0.38, wspace=0.2, figure=fig,
                  left=0.04, right=0.985, top=0.86, bottom=0.1)
    lab = c["label"] if c["label"].startswith("h =") else c["label"][0].upper() + c["label"][1:]
    head = f"{lab} | $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)} ({d['kind']})"
    tail = "every method reads only the DEM (look-up table from the terrain)"
    if len(head) + len(tail) > 185:
        fig.suptitle(head + "\n" + tail, fontsize=12.5, y=0.985)
    else:
        fig.suptitle(head + " | " + tail, fontsize=13, y=0.975)
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["joint_line"], color="tab:purple", lw=1.3, label="joint, final version (over the whole line, DEM applied at every position)")
    ax.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | joint {amb['joint_line']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    title = f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB"
    if variant == "joint_sata":
        az.plot(t, db["whole"], color="tab:blue", lw=1.2, ls=(0, (1, 1.5)), label="SATA whole band (DEM)")
        az.plot(t, db["sub"], color="tab:orange", lw=1.2, ls=(0, (3, 1.5)), label="SATA per sub-band (DEM)")
        title += f" | SATA whole band {amb['whole']:.1f} dB | SATA per sub-band {amb['sub']:.1f} dB"
    az.set_title(title, fontsize=10)
    for a_ in (ax, az):
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
            label=f"cells of the point targets (channel {i1}; not given to any method)")
    am.set_xlim(0, cfg.Na_ch - 1); am.grid(alpha=0.3)
    am.set_xlabel("azimuth cell of the channel line"); am.set_ylabel("$\\Delta C_0$ in the table [mm]")
    lo_, hi_ = am.get_ylim(); am.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    am.set_title("$\\Delta C_0$ per cell in the DEM table of every channel (look-up table from the terrain)", fontsize=10)
    am.legend(fontsize=7.8 if cfg.Nrx <= 4 else 6.6, loc="upper center", ncol=2)
    nmax = max(res["orders"])
    fig.text(0.5, 0.012, "Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions of every "
             f"target (up to {nmax} times the ambiguity distance of the channel PRF, on both sides). All methods use the same look-up "
             f"table from the terrain (DEM); none uses the positions of the targets.\n$N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, "
             f"$B_a$ = {cfg.abw:.0f} Hz, channel 0 is the transmitter ($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)}.",
             ha="center", fontsize=8.5, color="0.35")
    fig.savefig(out_png, dpi=90); plt.close(fig)


# ----------------------------------------------------------------------------------------------- driver
def _paths(sc, tag):
    k = f"{sc}__{tag}"
    return os.path.join(OUT, "numbers", k + ".json"), {v: os.path.join(OUT, VARIANTS[v][0], k + ".png") for v in VARIANTS}


def _done(sc, tag):
    pj, pp = _paths(sc, tag)
    return os.path.exists(pj) and json.load(open(pj)).get("version") == VERSION and all(os.path.exists(p) for p in pp.values())


def _job(a):
    sc, tag, deadline = a
    if time.time() > deadline:
        return None
    t0 = time.time()
    res, d = compute(sc, tag)
    pj, pp = _paths(sc, tag)
    for v, p in pp.items():
        page(res, d, p, v)
    json.dump(res, open(pj, "w"), indent=1)
    return f"{sc:22s} {tag:8s} ref {res['ref']:6.1f} none {res['none']:6.1f} whole {res['whole']:6.1f} sub {res['sub']:6.1f} " \
           f"joint {res['joint_line']:6.1f} ({time.time() - t0:.0f} s)"


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
                print(f"  {sc:22s} {tag:8s} Nrx {r['nrx']} | ref {r['ref']:6.1f} none {r['none']:6.1f} whole {r['whole']:6.1f} sub {r['sub']:6.1f} "
                      f"joint {r['joint_line']:6.1f}")


def pdf():
    from joint_sweep_summary import bundle
    res = load()
    done = []
    for v, (folder, name) in VARIANTS.items():
        pngs, outline = [], []
        for fam, scenes in FAMILIES:
            first = True
            for sc, tag in jobs():
                pp = _paths(sc, tag)[1][v]
                if sc not in scenes or (sc, tag) not in res or not os.path.exists(pp):
                    continue
                if first:
                    outline.append((fam, len(pngs), 0)); first = False
                r = res[(sc, tag)]
                lab = r["label"].replace("$\\pi$", "pi").replace("$^\\circ$", " deg").replace("$\\pm$", "+-")
                outline.append((f"{lab} | {r['nrx']} channels, bxt {W.bxt_text(r['bxt'])}", len(pngs), 1))
                pngs.append(pp)
        done.append(f"{bundle(pngs, os.path.join(OUT, name), outline)} ({len(pngs)} pages)")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, help="scene ids (default: all)")
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--list", action="store_true", help="the jobs, without running anything")
    ap.add_argument("--force", action="store_true", help="redo the selected jobs even if they are done")
    a = ap.parse_args()
    if a.list:
        for sc, tag in jobs():
            print(f"{sc:22s} {tag:8s} bxt {W.bxt_text(AD.baseline(tag)[0])}")
        return print(len(jobs()), "jobs")
    if a.table:
        return table()
    if a.pdf:
        return print("wrote\n" + "\n".join(pdf()))
    for sub in ["numbers"] + [f for f, _ in VARIANTS.values()]:
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    todo = [(sc, tag) for sc, tag in jobs() if (a.only is None or sc in a.only) and (a.force or not _done(sc, tag))]
    print(len(todo), "jobs to do", flush=True)
    deadline = time.time() + a.budget
    args = [(sc, tag, deadline) for sc, tag in todo]
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for line in pool.imap_unordered(_job, args):
                if line:
                    print(line, flush=True)
    else:
        for x in args:
            line = _job(x)
            if line:
                print(line, flush=True)
    left = sum(not _done(sc, tag) for sc, tag in jobs() if a.only is None or sc in a.only)
    print("left:", left, flush=True)


if __name__ == "__main__":
    main()
