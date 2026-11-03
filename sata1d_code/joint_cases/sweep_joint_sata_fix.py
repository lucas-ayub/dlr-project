"""
Sweep after the registration fix of the sub-aperture joint (2026-10-27): the 86 cases of sweep_final.py (ramps, sines /
cosines, realistic terrains; 1, a few and many targets; 2 to 8 channels; channel 0 on the transmitter), every method
reading only the DEM. Six-panel pages:

    top IRF     Reference, oracle, joint over the whole line, joint in the sub-apertures (fixed)
    bottom IRF  Reference, oracle, no SATA, SATA whole band, SATA per sub-band
    right       scene zoomed on the targets, whole line in 3D, channels seen from above, DEM table of the channels

The oracle multiplies the echo of every target by its own phase before the standard reconstruction (it knows the
targets): it is only the yardstick of what a correct per-target phase reaches. With --old-joint <file> the joint in the
sub-apertures before the fix (a copy of the old sata1d/joint.py) is also computed, for the summary only.

    python joint_cases/sweep_joint_sata_fix.py --workers 2 [--old-joint <old joint.py>]   # numbers and pages (resumable)
    python joint_cases/sweep_joint_sata_fix.py --pdf                                        # summary pages + PDF
Output: joint_cases/sweep_joint_sata_fix/{numbers,pages}/, sweep_joint_sata_fix.pdf
"""
import argparse
import json
import os
import sys
import time
import types
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
import sweep_final as SF
import run_joint_sweep as W
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy

OUT = os.path.join(HERE, "sweep_joint_sata_fix")
VERSION = 1
OLD_JOINT = None                                   # path of the old joint.py (optional)
STYLE = {   # key: (legend, colour, line style, width)
    "ref": ("Reference", "tab:green", "-", 2.2),
    "oracle": ("oracle (every target corrected with its own phase)", "0.45", "-", 0.9),
    "joint_line": ("joint over the whole line (DEM at every position)", "tab:purple", "-", 1.3),
    "joint_occ": ("joint in the sub-apertures (fixed)", "tab:brown", (0, (4, 1.5)), 1.2),
    "none": ("no SATA", "tab:red", (0, (5, 3)), 1.6),
    "whole": ("SATA whole band (DEM)", "tab:blue", (0, (1, 1.5)), 1.2),
    "sub": ("SATA per sub-band (DEM)", "tab:orange", (0, (3, 1.5)), 1.2),
}
SHORT = {"ref": "reference", "oracle": "oracle", "joint_line": "joint line", "joint_occ": "joint sub-ap.", "none": "no SATA",
         "whole": "SATA whole", "sub": "SATA sub-band", "joint_old": "joint sub-ap. before fix"}


def _old_joint():
    m = types.ModuleType("sata1d.joint_before_fix"); m.__package__ = "sata1d"
    exec(compile(open(OLD_JOINT).read(), OLD_JOINT, "exec"), m.__dict__)
    return m


def compute(sc, tag):
    res, d = SF.compute(sc, tag)                    # ref, none, whole, sub, joint_occ (fixed), joint_line; orders of sweep_final
    cfg, tr, xs = d["cfg"], d["tr"], d["c"]["xs"]
    ch = S.generate_channels(cfg, tr)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    dC0 = SS.delta_C0_per_target(cfg, tr)
    och = np.zeros((cfg.Nrx, cfg.Na_ch), complex)
    for j, p in enumerate(cfg.scene.points[1:]):
        och += S.generate_channels(cfg, tr, ptgs=[p], amps=[cfg.scene.amps[j]]) * np.exp(2j * np.pi * dC0[j] / cfg.system.wl)[:, None]
    d["env"]["oracle"] = F(S.reconstruct(cfg, tr, och))
    keys = ["oracle"]
    if OLD_JOINT:
        d["env"]["joint_old"] = F(_old_joint().reconstruct_joint(cfg, tr, ch, maps=d["Mch"], sata_osf=AD.SATA_OSF, occupied=True))
        keys.append("joint_old")
    dt_amb = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    for k in keys:
        res[k] = d["amb"][k] = float(AD.ambiguity_level(d["t"], d["env"][k], d["pk"], xs, cfg.system.vs, dt_amb, orders=tuple(res["orders"])))
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
    tail = "every method reads only the DEM (look-up table from the terrain)"
    if len(head) + len(tail) > 185:
        fig.suptitle(head + "\n" + tail, fontsize=12.5, y=0.985)
    else:
        fig.suptitle(head + " | " + tail, fontsize=13, y=0.975)
    for row, keys in enumerate((("ref", "joint_line", "joint_occ", "oracle"), ("ref", "none", "whole", "sub", "oracle"))):
        a_ = fig.add_subplot(gs[row, 0])
        for k in keys[::-1]:
            name, col, ls, lw = STYLE[k]
            a_.plot(t, db[k], color=col, ls=ls, lw=lw, label=name)
        a_.set_title("highest ambiguity: " + " | ".join(f"{SHORT[k]} {amb[k]:.1f} dB" for k in keys), fontsize=9.6)
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]")
        hd, lb = a_.get_legend_handles_labels()
        a_.legend(hd[::-1], lb[::-1], fontsize=8.3, loc="upper left")
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
             "table from the terrain (DEM); none uses the positions of the targets.\nThe oracle multiplies the echo of every target by "
             "its own phase before the standard reconstruction (it knows the targets: only a yardstick). "
             f"$N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz, channel 0 is the transmitter "
             f"($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)}.",
             ha="center", fontsize=8.5, color="0.35")
    fig.savefig(out_png, dpi=90); plt.close(fig)


# ----------------------------------------------------------------------------------------------- summary
def short(r):
    s = r["label"].replace("$\\pi$", "pi").replace("$^\\circ$", " deg").replace("$\\pm$", "+-")
    for cut in (", flat before and after",):
        s = s.replace(cut, "")
    return f"{s} | {r['nrx']} ch., largest b_xt {max(abs(b) for b in r['bxt']):g} m"


SUMMARY = [("ref", "Reference", "tab:green", "o", 60), ("oracle", "oracle", "0.3", "|", 150), ("none", "no SATA", "tab:red", "x", 50),
           ("whole", "SATA whole band", "tab:blue", "s", 36), ("sub", "SATA per sub-band", "tab:orange", "D", 30),
           ("joint_old", "joint sub-ap. (before fix)", "none", "^", 46), ("joint_occ", "joint sub-ap. (fixed)", "tab:brown", "^", 46),
           ("joint_line", "joint over the whole line", "tab:purple", "*", 110)]


def summary(res, fam, scenes, out_png):
    rows = [j for j in SF.jobs() if j[0] in scenes and j in res]
    n = len(rows)
    H = 0.4 * n + 2.4
    fig, ax = plt.subplots(figsize=(22, H))
    fig.subplots_adjust(left=0.43, right=0.97, top=1 - 1.0 / H, bottom=1.0 / H)
    y = np.arange(n)[::-1]
    for k, name, col, m, s in SUMMARY:
        v = [res[j].get(k, np.nan) for j in rows]
        if np.all(np.isnan(v)):
            continue
        if col == "none":
            ax.scatter(v, y, marker=m, s=s, facecolors="none", edgecolors="tab:brown", lw=1.0, label=name, zorder=3)
        else:
            ax.scatter(v, y, marker=m, s=s, color=col, label=name, zorder=3, lw=1.6 if m in "x|" else 0.5,
                       edgecolor=None if m in "x|" else "k")
    for i, j in enumerate(rows):
        vals = [res[j][k] for k, *_ in SUMMARY if k in res[j]]
        ax.plot([min(vals), max(vals)], [y[i]] * 2, color="0.85", lw=0.8, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([short(res[j]) for j in rows], fontsize=8)
    ax.set_xlabel("highest ambiguity [dB relative to the reference peak]")
    ax.set_ylim(-0.7, n - 0.3); ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"{fam}: highest ambiguity of every case (every method reads only the DEM)", fontsize=12, pad=30)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(SUMMARY), frameon=False, fontsize=9.5)
    fig.savefig(out_png, dpi=100); plt.close(fig)


def table_page(res, out_png):
    """Median over the cases of every family, and how often each method is within 3 dB of the oracle."""
    keys = [k for k, *_ in SUMMARY if any(k in r for r in res.values())]
    fams = SF.FAMILIES + [("All cases", sum((s for _, s in SF.FAMILIES), []))]
    lines = []
    for fam, scenes in fams:
        rs = [r for (sc, _), r in res.items() if sc in scenes]
        med = {k: np.median([r[k] for r in rs if k in r]) for k in keys}
        near = {k: np.mean([r[k] <= r["oracle"] + 3.0 for r in rs if k in r]) for k in keys}
        lines.append((fam, len(rs), med, near))
    fig, ax = plt.subplots(figsize=(22, 6.5)); ax.axis("off")
    col = ["family (cases)"] + [SHORT[k] for k in keys]
    cells = []
    for fam, n, med, near in lines:
        cells.append([f"{fam} ({n})  median [dB]"] + [f"{med[k]:.1f}" for k in keys])
        cells.append([f"{fam}  within 3 dB of the oracle"] + [f"{100 * near[k]:.0f} %" for k in keys])
    tb = ax.table(cellText=cells, colLabels=col, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(1, 2.0)
    for (r_, c_), cell in tb.get_celld().items():
        if c_ == 0:
            cell.set_text_props(ha="left"); cell.set_width(0.26)
        if r_ == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Sweep after the registration fix of the joint in the sub-apertures: highest ambiguity, median per family "
                 "and share of cases within 3 dB of the oracle", fontsize=13)
    fig.savefig(out_png, dpi=100); plt.close(fig)


# ----------------------------------------------------------------------------------------------- driver
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
    old = f" (before fix {res['joint_old']:6.1f})" if "joint_old" in res else ""
    return (f"{j[0]:20s} {j[1]:7s} ref {res['ref']:6.1f} oracle {res['oracle']:6.1f} none {res['none']:6.1f} whole {res['whole']:6.1f} "
            f"sub {res['sub']:6.1f} joint sub-ap {res['joint_occ']:6.1f}{old} line {res['joint_line']:6.1f} ({time.time() - t0:.0f} s)")


def load():
    out = {}
    for j in SF.jobs():
        pj = _paths(*j)[0]
        if os.path.exists(pj):
            out[j] = json.load(open(pj))
    return out


def pdf():
    from joint_sweep_summary import bundle
    res = load()
    pngs, outline = [], []
    tp = os.path.join(OUT, "pages", "_table.png"); table_page(res, tp)
    outline.append(("Summary", 0, 0)); pngs.append(tp)
    for i, (fam, scenes) in enumerate(SF.FAMILIES):
        sp = os.path.join(OUT, "pages", f"_summary_{i}.png"); summary(res, fam, scenes, sp)
        outline.append((f"Highest ambiguity: {fam}", len(pngs), 1)); pngs.append(sp)
    for fam, scenes in SF.FAMILIES:
        first = True
        for j in SF.jobs():
            if j[0] not in scenes or j not in res or not os.path.exists(_paths(*j)[1]):
                continue
            if first:
                outline.append((fam, len(pngs), 0)); first = False
            outline.append((short(res[j]), len(pngs), 1)); pngs.append(_paths(*j)[1])
    return f"{bundle(pngs, os.path.join(OUT, 'sweep_joint_sata_fix.pdf'), outline)} ({len(pngs)} pages)"


def _init(out, old):
    global OUT, OLD_JOINT
    OUT, OLD_JOINT = out, old


def main():
    global OUT, OLD_JOINT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--old-joint", default=None, help="old sata1d/joint.py, to also compute the joint before the fix")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    OUT, OLD_JOINT = a.out, a.old_joint
    if a.pdf:
        return print("wrote", pdf())
    for sub in ("numbers", "pages"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    todo = [j for j in SF.jobs() if not _done(j) and (not a.only or j[0] in a.only)]
    print(len(todo), "jobs", flush=True)
    if a.workers > 1:
        with Pool(a.workers, initializer=_init, initargs=(OUT, OLD_JOINT)) as p:
            for line in p.imap_unordered(_job, todo):
                print(line, flush=True)
    else:
        for j in todo:
            print(_job(j), flush=True)


if __name__ == "__main__":
    main()
