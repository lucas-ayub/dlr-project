"""
Every case page of combined_main_results_tx_rx.pdf, with the joint reconstruction (sata1d/joint.py,
reconstruct_joint). Top: the two joint variants; bottom: no SATA (the SATA whole band and per sub-band with the
block map with copies, as in the original pages, are given in the footer and in results.json). Channel 0 is the
transmitter (tx_rx_channel0=True: bat = bxt = 0); the other channels have bxt ~ U(0, bxt_max), seed 0. The default
bxt_max = 100 m is the configuration of the original pages, bxt = [0, 27.0, 4.1, 1.7] m.

Cases:
  sweep_v4 ids      RA-SS-012, RA-NN-036, ..., ZI-NN-012, ... (<family>-<distance><heights>-<index>, as sweep_v4.cases)
  sweep ids         RA-012, RA-049, RH-017, CL-012, SK-007, ME-005 (run_geometry_sweep.build_cases, as run_sweep_lut_cases)
  many targets      MP-R13, MP-R25, MP-R49, MP-H13, MP-H25 (run_sweep_lut_cases.many_cases)
  terrains          <preset>-N<n> (run_terrain_lut_cases presets, n point targets on the terrain)
Maps of the joint reconstruction: point-target scenes -> joint_maps (each cell takes the delta_C0 of the nearest
target, whole line); terrains -> the look-up table from the terrain (terrain_lookup_table, the scene's DEM).
Joint variants: occupied=True (only the parcels that carry signal, least squares) and occupied=False (full
Nrx x Nsb system). Needs the joint.py with the `occupied` argument.

    python joint_cases/run_joint_cases.py              # from sata1d_code: every case -> joint_cases/bxt100/
    python joint_cases/run_joint_cases.py --cases RA-SS-012 COS-B80-A80-P20-N6
    python joint_cases/run_joint_cases.py --bxt-max 1000      # larger cross-track baselines -> joint_cases/bxt1000/
    python joint_cases/joint_summary.py joint_cases/bxt100 joint_cases/bxt500 joint_cases/bxt1000 --pdf joint_cases/joint_cases.pdf
"""
import argparse
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # sata1d_code
import sata1d as S
import sata1d.sata as SS
from sata1d.joint import reconstruct_joint, joint_maps
import run_geometry_sweep as G
import run_terrain_lut_cases as T
import sweep_v4 as V
from run_one_case import ambiguity_level
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy

SATA_OSF = G.SATA_OSF
CH_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
V4_DIST = {"S": "200 m apart", "N": "2.6 km apart", "L": "10 km apart"}
V4_H = {"S": "small heights", "N": "normal heights", "L": "large heights"}
V4_SPACING = {"S": 200.0, "N": 2600.0, "L": 10000.0}
V4_SCALE = {"S": 0.1, "N": 1.0, "L": 13.0}


INV_PREFIX = {v: k for k, v in G.PREFIX.items()}


def v4_case(cid):
    """(xs, hs, amps, title) of a case of the v4/v5 sweep by id (<prefix>-<dist><heights>-<index>), as sweep_v4.cases."""
    pre, dh, idx = cid.split("-")
    c = V.cases(INV_PREFIX[pre], V4_SPACING[dh[0]], V4_SCALE[dh[1]])[int(idx) - 1]
    title = (f"{cid}: {G.LABEL[c.family]}, {c.N} targets {V4_DIST[dh[0]]}, {V4_H[dh[1]]}"
             + (f", $\\alpha$ = {c.params['alpha']:g}$^\\circ$" if "alpha" in c.params else f" ({c.slug})"))
    return np.asarray(c.xs, float), np.asarray(c.heights, float), None, title


_SWEEP = None


def sweep_case(cid):
    """(xs, hs, amps, title) of a case of run_geometry_sweep by id (RA-012, ...), numbered as run_sweep_lut_cases does."""
    global _SWEEP
    if _SWEEP is None:
        _SWEEP, counter = {}, {}
        for c in G.build_cases(G.FAMILIES):
            counter[c.family] = counter.get(c.family, 0) + 1
            _SWEEP[f"{G.PREFIX[c.family]}-{counter[c.family]:03d}"] = c
    c = _SWEEP[cid]
    assert c.offsets is None, "iso-range cases only"
    return np.asarray(c.xs, float), np.asarray(c.heights, float), None, f"{cid}: {G.LABEL[c.family]}, {c.desc}, spacing {c.spacing:g} m"


def many_case(cid):
    import run_sweep_lut_cases as R
    for k, title, c in R.many_cases():
        if k == cid:
            return np.asarray(c.params["xs"], float), np.asarray(c.heights, float), c.params.get("amps"), f"{cid}: {title}"
    raise KeyError(cid)


TERRAIN_CASES = [("SIN-B100-A100-P50", 1, -100.0), ("SIN-B100-A100-P50", 5, -100.0), ("SIN-B100-A100-P50", 6, -100.0),
                 ("SIN-B100-A100-P50", 31, -100.0), ("SIN-B500-A100-P50", 6, -100.0), ("COS-B80-A80-P20", 5, -100.0),
                 ("COS-B80-A80-P20", 6, -100.0), ("COS-B40-A40-P5", 6, -100.0), ("COS-B40-A40-P5", 7, -100.0),
                 ("SIN-B40-A40-P5", 31, -100.0), ("REF-A100-P50", 1, 0.0)]
CASES = {}   # id -> (kind, argument), in the order of the pages of the PDF
for k in ("RA-SS-012",):
    CASES[k] = ("v4", k)
for k in ("RA-012", "RA-049"):
    CASES[k] = ("sweep", k)
for k in ("RA-NN-036", "RA-NN-038", "RA-LN-038"):
    CASES[k] = ("v4", k)
for k in ("MP-R13", "MP-R25", "MP-R49"):
    CASES[k] = ("many", k)
for k in ("RH-017", "CL-012", "SK-007", "ME-005"):
    CASES[k] = ("sweep", k)
for k in ("RH-NN-005", "PW-NN-002", "ZI-NN-012", "CL-NN-005", "SK-NN-003", "ME-NN-002", "RH-LN-005"):
    CASES[k] = ("v4", k)
for k in ("MP-H13", "MP-H25"):
    CASES[k] = ("many", k)
for p, n, c in TERRAIN_CASES:
    CASES[f"{p}-N{n}"] = ("terrain", (p, n, c))


BXT_MAX = G.BXT_MAX          # set by --bxt-max


def run(case_id, out_dir):
    t0 = time.time()
    kind, arg = CASES[case_id]
    terrain, amps = None, None
    if kind == "terrain":
        preset, n, centre = arg
        expr, desc = T.ALL_PRESETS[preset]
        terrain = T.terrain_from(expr)
        xs = np.atleast_1d(T.target_positions(n, centre=centre)).astype(float); hs = terrain(xs)
        title = f"{preset}: terrain {desc}, {n} point target{'s' if n > 1 else ''} on it"
    else:
        xs, hs, amps, title = {"v4": v4_case, "sweep": sweep_case, "many": many_case}[kind](arg)
    cfg = S.make_config(list(zip(xs, hs)), bxt_max=BXT_MAX, seed=G.BXT_SEED, amplitudes=amps, tx_rx_channel0=True)
    tr = S.build_platform_tracks(cfg)
    dC0 = S.delta_C0_per_target(cfg, tr)
    ch = S.generate_channels(cfg, tr)
    if terrain is None:
        jmaps, jlabel = np.array(joint_maps(cfg, tr, dC0)), "nearest target (joint_maps)"
    else:
        jmaps, jlabel = np.array([S.terrain_lookup_table(cfg, tr, i, terrain, step=4) for i in range(cfg.Nrx)]), "look-up table from the terrain"
    oracle = sum(S.generate_channels(cfg, tr, ptgs=[p], amps=[cfg.scene.amps[j]]) * np.exp(2j * np.pi * dC0[j] / cfg.system.wl)[:, None]
                 for j, p in enumerate(cfg.scene.points[1:]))
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    tj = {}
    env = {"ref": F(S.generate_reference(cfg, tr)),
           "none": F(S.reconstruct(cfg, tr, ch)),
           "whole": F(S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=SATA_OSF, mode="footprint", dC0=dC0))),
           "sub": F(S.reconstruct_subband(cfg, tr, ch, sata_osf=SATA_OSF, mode="footprint", dC0=dC0)),
           "oracle": F(S.reconstruct(cfg, tr, oracle))}
    for key, occ in (("joint_full", False), ("joint_occ", True)):
        t1 = time.time(); env[key] = F(reconstruct_joint(cfg, tr, ch, maps=jmaps, sata_osf=SATA_OSF, occupied=occ)); tj[key] = time.time() - t1
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    amb = {k: float(ambiguity_level(t, v, pk, xs, cfg.system.vs, G.ambiguity_times(cfg))) for k, v in env.items()}
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}

    fig = plt.figure(figsize=(18, 9.6))
    gs = GridSpec(2, 2, width_ratios=[1.45, 1.25], hspace=0.36, wspace=0.14, figure=fig)
    fig.suptitle(f"{title} | joint reconstruction (map: {jlabel})", fontsize=12, y=0.995)
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["joint_full"], color="tab:pink", lw=1.4, ls=(0, (5, 3)), label="joint, full Nrx x Nsb system (occupied=False)")
    ax.plot(t, db["joint_occ"], color="tab:purple", lw=1.3, label="joint, occupied parcels only (occupied=True)")
    ax.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | joint, occupied {amb['joint_occ']:.1f} dB | "
                 f"joint, full system {amb['joint_full']:.1f} dB | oracle {amb['oracle']:.1f} dB", fontsize=10)
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
    if terrain is None:
        geometry_2d(g2, xs_, dys, hs_, False, title=None, annotate=len(xs_) <= 13)
        geometry_3d(g3, xs_, dys, hs_, False, title=None)
        dh = np.ptp(hs_)
    else:
        xg = np.linspace(-half, half, 2401); hg = terrain(xg)
        hz = hg[(xg >= lo2 - pad2) & (xg <= hi2 + pad2)]; dh = np.ptp(hz)
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
    g2.set_title(f"Zoom on the point targets and the reference point | $\\Delta h_{{max}}$ = {dh:.0f} m", fontsize=9.5, pad=36)
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
        am.step(n, 1e3 * jmaps[i], where="mid", color=CH_COLORS[i], lw=1.3, label=f"channel {i} ($b_{{xt}}$ = {cfg.array.bxt[i]:.1f} m)")
    cells1 = np.array([SS.az_pixel_of_scatterer(cfg, x, 1) for x in xs])
    if len(cells1) <= 60:
        for c in cells1:
            am.axvline(c, color="0.75", lw=0.6, zorder=0)
    am.plot(cells1, 1e3 * dC0[:, 1], "o", ms=5, color=CH_COLORS[1], mec="k", zorder=5, label="point targets: own $\\Delta C_0$ (channel 1)")
    am.set_xlim(0, cfg.Na_ch - 1); am.grid(alpha=0.3)
    am.set_xlabel("azimuth cell of the channel line"); am.set_ylabel("$\\Delta C_0$ in the map [mm]")
    lo_, hi_ = am.get_ylim(); am.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    am.set_title(f"$\\Delta C_0$ per cell read by the joint reconstruction ({jlabel})", fontsize=10)
    am.legend(fontsize=8, loc="upper center", ncol=3)
    fig.text(0.5, 0.012, f"Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions of every "
             f"target. For comparison (block map with copies): SATA whole band {amb['whole']:.1f} dB, SATA per sub-band {amb['sub']:.1f} dB. "
             f"Joint: {tj['joint_occ']:.2f} s (occupied) and {tj['joint_full']:.2f} s (full system) per line. "
             f"$N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz, $b_{{xt}}$ = ["
             + ", ".join(f"{b:.1f}" for b in cfg.array.bxt) + "] m.", ha="center", fontsize=8.5, color="0.35")
    fig.subplots_adjust(bottom=0.1)
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{case_id}_joint.png")
    fig.savefig(png, dpi=95, bbox_inches="tight"); plt.close(fig)
    print(f"{case_id:22s} " + " | ".join(f"{k} {v:.1f}" for k, v in amb.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    return amb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--bxt-max", type=float, default=G.BXT_MAX,
                    help="cross-track baselines bxt ~ U(0, bxt_max) (seed 0), channel 0 on the transmitter (bat = bxt = 0)")
    ap.add_argument("--out", default=None, help="default joint_cases/bxt<bxt_max>")
    ap.add_argument("--pdf", action="store_true", help="also bundle the pages of every case in results.json into <out>.pdf")
    a = ap.parse_args()
    global BXT_MAX
    BXT_MAX = a.bxt_max
    a.out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"bxt{a.bxt_max:g}")
    res = {}
    path = os.path.join(a.out, "results.json")
    if os.path.exists(path):
        res = json.load(open(path))
    for cid in a.cases:
        res[cid] = run(cid, a.out)
        res["_bxt"] = [0.0] + list(np.random.default_rng(G.BXT_SEED).uniform(0.0, BXT_MAX, 4)[1:].round(3))
        json.dump(res, open(path, "w"), indent=1)
    if a.pdf:
        pngs = [os.path.join(a.out, f"{cid}_joint.png") for cid in CASES if cid in res]
        print("wrote", T.bundle_pdf(pngs, a.out + ".pdf"))


if __name__ == "__main__":
    main()
