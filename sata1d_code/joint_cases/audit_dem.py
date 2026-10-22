"""
Audit of the SATA 1D with a DEM: every method reads ONLY the look-up table from the terrain (no method knows where the
targets are, their delta_C0 or how many they are).

Methods, all with the same DEM:
    none        standard reconstruction (reference point at h = 0)
    whole       SATA whole band before the reconstruction        (sata_1d, table of the channel)
    sub         SATA per sub-band before the reconstruction      (sata_1d_subband, the same table)
    joint_occ   joint reconstruction in the SATA sub-apertures, occupied parcels (reconstruct_joint)
    joint_line  joint reconstruction over the whole line         (reconstruct_joint_line, sata1d/joint_line.py)
Scenes: one target, a few targets, many targets on along-track ramps, sine / cosine terrains and realistic terrains
(random hills and mountains, escarpment, isolated mountain, plateaus). Channel 0 is always the transmitter
(bat = bxt = 0); along-track baselines on the DPCA condition; total PRF 2000 Hz for every Nrx.

    python joint_cases/audit_dem.py --budget 160 --workers 4            # numbers of every job (resumable), from sata1d_code
    python joint_cases/audit_dem.py --pages --budget 160 --workers 4    # the pages of PAGES (PNG, layout of the sweep pages)
    python joint_cases/audit_dem.py --table                             # text table of the numbers
    python joint_cases/audit_dem.py --pdf                               # joint_cases/audit_dem/audit_dem.pdf from the pages
Output: joint_cases/audit_dem/numbers/<job>.json, joint_cases/audit_dem/pages/<job>.png
"""
import argparse
import glob
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
from sata1d.joint import reconstruct_joint
from sata1d.joint_line import height_table, channel_maps, line_maps, reconstruct_joint_line
from sata1d.subband import sata_channels_subband, channel_spectrum
from sata1d.reconstruction import reconstruct_from_spectra
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy
import run_joint_sweep as W

OUT = os.path.join(HERE, "audit_dem")
SATA_OSF = 4
VERSION = 1
NRX = [2, 3, 4, 6, 8]
BMAX = [100.0, 300.0, 500.0]
CH = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive"]


# ----------------------------------------------------------------------------------------------- baselines
def random_baselines(nrx, bmax, seed=0):
    """Channel 0 on the transmitter; the others random in (0, bmax], the largest exactly bmax."""
    u = np.random.default_rng(seed).uniform(0.1, 1.0, nrx)
    u[0] = 0.0
    return (bmax * u / u.max()).round(1)


FIXED = {  # tag -> (bxt of channels 0..3, text) : Nrx = 4 only
    "u100": (W.BXT["rnd100"], "random, U(0, 100 m)"),
    "u500": (W.BXT["rnd500"], "random, U(0, 500 m)"),
    "s10": (np.array([0.0, 10.0, 20.0, 30.0]), "steps of 10 m"),
    "s50": (np.array([0.0, 50.0, 100.0, 150.0]), "steps of 50 m"),
    "pm50": (np.array([0.0, -50.0, 50.0, 100.0]), "negative and positive"),
}


def baseline(tag):
    """tag: 'N<nrx>b<bmax>' (random, largest exactly bmax) or a key of FIXED. Returns (bxt, text)."""
    if tag in FIXED:
        return FIXED[tag]
    n, b = tag[1:].split("b")
    return random_baselines(int(n), float(b)), f"random, largest {float(b):g} m, $N_{{rx}}$ = {int(n)}"


# ----------------------------------------------------------------------------------------------- scenes
def phases(n, seed=1):
    return np.exp(2j * np.pi * np.random.default_rng(seed).uniform(size=n))


def ramp_terrain(slope_deg, x0, length):
    """Flat at h = 0 before x0, ramp of slope_deg over [x0, x0 + length], flat on top after."""
    t = np.tan(np.radians(slope_deg))
    return lambda x: t * np.clip(np.asarray(x, float) - x0, 0.0, length)


def _on(terrain, desc, xs, random_phase=None):
    xs = np.atleast_1d(np.asarray(xs, float))
    n = len(xs)
    if n == 1:
        what = f"1 target at x = {xs[0] / 1e3:g} km"
    else:
        what = f"{n} targets {W.dist_text(float(np.diff(xs)[0]))} apart"
    amps = phases(n) if (n > 13 if random_phase is None else random_phase) else None
    return dict(xs=xs, hs=np.asarray(terrain(xs), float), terrain=terrain, amps=amps, label=f"{desc}, {what}")


def _ramp(slope, length, xs):
    hmax = np.tan(np.radians(slope)) * length
    return _on(ramp_terrain(slope, -length / 2.0, length), f"along-track ramp of {slope:g}$^\\circ$, {length / 1e3:g} km long (h 0 -> {hmax:.0f} m), flat before and after", xs)


def _sweep(cid):
    """A case of the joint sweep; the point-target ramps get the terrain through the targets (flat beyond) as DEM."""
    c = dict(W.case_info(cid))
    if c["terrain"] is None:
        xs, hs = np.asarray(c["xs"], float), np.asarray(c["hs"], float)
        c["terrain"] = lambda x, xs=xs, hs=hs: np.interp(np.asarray(x, float), xs, hs)
        c["label"] += ", flat before and after"
    return c


def _t(name):
    return W.TERRAINS[name]


span = lambda half, d: np.arange(-half, half + 1e-6, d)
X_FOLD = 6228.7                                   # one fold of PRF_op on the ground for Nrx = 4 (PRF_op = 500 Hz) [m]


def _flat(h):
    return (lambda x: np.full(np.shape(np.asarray(x, float)), float(h))), f"flat terrain at h = {h:g} m (the DEM is the constant {h:g} m)"


def _periodic(factor):
    per = factor * X_FOLD
    txt = "the same height at x and x $\\pm$ 6.23 km" if factor == 1 else "different heights at x and x $\\pm$ 6.23 km"
    return (lambda x: 100.0 - 100.0 * np.cos(2 * np.pi * np.asarray(x, float) / per),
            f"h = 100 - 100 cos(2$\\pi$ x / {per / 1e3:.2f} km), {txt} (6.23 km = one fold of the channel PRF on the ground)")


SCENES = {   # id -> (group, builder)
    # ---- controls: terrains for which one phase per bin is right (SATA before the reconstruction must work) and one for which it is not
    "ctrl_flat500_1": ("control", lambda: _on(*_flat(500.0), [1300.0])),
    "ctrl_flat500_25": ("control", lambda: _on(*_flat(500.0), span(6000.0, 500.0) + 130.0, random_phase=True)),
    "ctrl_periodX_5": ("control", lambda: _on(*_periodic(1.0), W.spaced(5, 2600.0) + 700.0)),
    "ctrl_periodX_25": ("control", lambda: _on(*_periodic(1.0), span(6000.0, 500.0) + 130.0, random_phase=True)),
    "ctrl_period1p5X_5": ("control", lambda: _on(*_periodic(1.5), W.spaced(5, 2600.0) + 700.0)),
    # ---- one target
    "one_sine": ("one", lambda: _sweep("C10")),
    "one_refpoint": ("one", lambda: _sweep("C20")),
    "one_ramp8": ("one", lambda: _ramp(8.0, 12000.0, [0.0])),
    "one_hill": ("one", lambda: _on(*_t("hills"), [0.0])),
    "one_mountains": ("one", lambda: _on(*_t("mountains"), [0.0])),
    "one_peak": ("one", lambda: _on(*_t("mountain"), [0.0])),
    "one_flank": ("one", lambda: _on(*_t("mountain"), [1500.0])),
    "one_escarpment": ("one", lambda: _on(*_t("escarpment"), [0.0])),
    # ---- a few targets
    "few_ramp5_2": ("few", lambda: _sweep("C01")),
    "few_ramp20_7": ("few", lambda: _sweep("C03")),
    "few_ramp4_5": ("few", lambda: _sweep("C05")),
    "few_ramp15_9": ("few", lambda: _ramp(15.0, 3000.0, span(1400.0, 350.0))),
    "few_cos20km_5": ("few", lambda: _sweep("C15")),
    "few_cos5km_7": ("few", lambda: _sweep("C18")),
    "few_sin50km_5": ("few", lambda: _sweep("C11")),
    "few_hills_5": ("few", lambda: _sweep("C22")),
    "few_mountains_5": ("few", lambda: _sweep("C24")),
    "few_escarpment_5": ("few", lambda: _sweep("C29")),
    "few_escarpment_9": ("few", lambda: _sweep("C28")),
    "few_peak_5": ("few", lambda: _sweep("C31")),
    "few_plateau_9": ("few", lambda: _sweep("C32")),
    "few_plateau_5": ("few", lambda: _sweep("C33")),
    # ---- many targets
    "many_ramp5_49": ("many", lambda: _sweep("C09")),
    "many_ramp3_49": ("many", lambda: _ramp(3.0, 12000.0, span(6000.0, 250.0))),
    "many_ramp8_49": ("many", lambda: _ramp(8.0, 12000.0, span(6000.0, 250.0))),
    "many_ramp15_25": ("many", lambda: _ramp(15.0, 3000.0, span(1500.0, 125.0))),
    "many_sin50km_31": ("many", lambda: _sweep("C13")),
    "many_sin5km_31": ("many", lambda: _sweep("C19")),
    "many_hills_49": ("many", lambda: _on(*_t("hills"), span(6000.0, 250.0))),
    "many_mountains_49": ("many", lambda: _on(*_t("mountains"), span(6000.0, 250.0))),
    "many_mountains_25": ("many", lambda: _sweep("C27")),
    "many_mountains_13": ("many", lambda: _sweep("C23")),
    "many_escarpment_25": ("many", lambda: _on(*_t("escarpment"), span(3000.0, 250.0))),
    "many_peak_33": ("many", lambda: _on(*_t("mountain"), span(4000.0, 250.0))),
}
GROUP_TEXT = {"control": "controls", "one": "one target", "few": "a few targets", "many": "many targets"}


def all_jobs():
    tags = [f"N{n}b{b:g}" for b in BMAX for n in NRX] + list(FIXED)
    return [(sc, t) for sc in SCENES for t in tags]


PAGES = [  # (scene, baseline): the pages of the PDF
    # controls
    ("ctrl_flat500_1", "N2b300"), ("ctrl_flat500_25", "N8b500"), ("ctrl_periodX_5", "N4b100"), ("ctrl_periodX_25", "u100"), ("ctrl_period1p5X_5", "N4b100"),
    # one target
    ("one_refpoint", "N4b100"), ("one_sine", "N4b300"), ("one_ramp8", "N4b300"), ("one_peak", "N4b300"), ("one_flank", "N4b100"),
    ("one_flank", "N4b500"), ("one_flank", "N2b300"), ("one_escarpment", "N4b500"), ("one_escarpment", "N8b300"), ("one_mountains", "N3b500"),
    # a few targets
    ("few_ramp5_2", "N4b300"), ("few_ramp20_7", "N4b100"), ("few_ramp20_7", "N4b300"), ("few_ramp4_5", "u500"), ("few_ramp4_5", "s50"),
    ("few_ramp15_9", "N4b300"), ("few_cos20km_5", "N4b500"), ("few_cos20km_5", "N2b300"), ("few_cos5km_7", "N4b300"), ("few_sin50km_5", "N6b500"),
    ("few_hills_5", "N4b300"), ("few_mountains_5", "u500"), ("few_mountains_5", "N4b100"), ("few_mountains_5", "N3b300"), ("few_mountains_5", "N8b500"),
    ("few_escarpment_5", "N4b300"), ("few_escarpment_9", "N4b100"), ("few_peak_5", "N6b300"), ("few_plateau_9", "pm50"), ("few_plateau_5", "N4b500"),
    # many targets
    ("many_ramp5_49", "u500"), ("many_ramp3_49", "N4b500"), ("many_ramp8_49", "N4b300"), ("many_ramp8_49", "N2b100"), ("many_ramp15_25", "N4b300"),
    ("many_sin50km_31", "N4b500"), ("many_sin5km_31", "N4b300"), ("many_hills_49", "N4b300"), ("many_hills_49", "N8b500"), ("many_mountains_49", "N4b100"),
    ("many_mountains_49", "N4b300"), ("many_mountains_49", "N6b500"), ("many_mountains_25", "N4b500"), ("many_mountains_13", "s10"), ("many_escarpment_25", "N3b300"),
    ("many_peak_33", "N4b300"),
]


# ----------------------------------------------------------------------------------------------- methods
def sata_whole(cfg, tr, ch, M):
    """SATA whole band with the DEM table of every channel, then the standard reconstruction."""
    o = np.array(ch, complex).copy()
    for i in range(cfg.Nrx):
        o[i] = SS.sata_1d(o[i], M[i], rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=cfg.system.wl,
                          r=cfg.scene.r0, squint=cfg.sq_tx, inverse=True, sata_osf=SATA_OSF)
    return S.reconstruct(cfg, tr, o)


def sata_sub(cfg, tr, ch, M):
    """SATA per sub-band with the same DEM table for every sub-band."""
    spec = []
    for k in range(cfg.Nsb):
        corr = sata_channels_subband(cfg, tr, ch, k, sata_osf=SATA_OSF, maps=list(M))
        spec.append([channel_spectrum(cfg, corr[j]) for j in range(cfg.Nrx)])
    return reconstruct_from_spectra(cfg, tr, spec, True)


def ambiguity_level(t, env, pk, xs, vs, dt_amb, orders=(-2, -1, 1, 2), win=0.02, guard=0.02):
    """Largest |IRF| inside +-win s of every ambiguity position x_j / vs + m dt_amb, in dB re. pk, leaving out
    +-guard s around every target's own main lobe (as in the sweeps)."""
    lobe = np.zeros(len(t), bool)
    sel = np.zeros(len(t), bool)
    for x in xs:
        lobe |= np.abs(t - x / vs) <= guard
        for m in orders:
            sel |= np.abs(t - (x / vs + m * dt_amb)) <= win
    sel &= ~lobe
    return 20 * np.log10(env[sel].max() / pk + 1e-15) if sel.any() else np.nan


def compute(sc, tag):
    """All the methods of one job. Returns (numbers dict, things needed by the page)."""
    grp, fn = SCENES[sc]
    c = fn()
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    bxt, kind = baseline(tag)
    cfg = S.make_config(list(zip(xs, hs)), nrx=len(bxt), bxt_max=0.0, amplitudes=c["amps"], tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy()))
    tr = S.build_platform_tracks(cfg)
    ch = S.generate_channels(cfg, tr)                                    # the simulated echoes of the point targets
    # ---- the only knowledge of the scene given to the methods: the terrain (DEM)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    hgrid = terrain(np.linspace(-half - 100.0, half + 100.0, 8001))
    tab = height_table(cfg, tr, hgrid.min(), hgrid.max())
    Mch = channel_maps(cfg, tr, terrain, tab)                            # [Nrx, Na_ch] tables of the channels
    Mline = line_maps(cfg, tr, terrain, tab)                             # [Nrx, Na] the same DEM on the full-rate line
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    env, sec = {}, {}

    def timed(key, f):
        t1 = time.time(); env[key] = F(f()); sec[key] = time.time() - t1
    env["ref"] = F(S.generate_reference(cfg, tr))
    timed("none", lambda: S.reconstruct(cfg, tr, ch))
    timed("whole", lambda: sata_whole(cfg, tr, ch, Mch))
    timed("sub", lambda: sata_sub(cfg, tr, ch, Mch))
    timed("joint_occ", lambda: reconstruct_joint(cfg, tr, ch, maps=Mch, sata_osf=SATA_OSF, occupied=True))
    info = {}

    def line():
        y, i = reconstruct_joint_line(cfg, tr, ch, Mline, return_info=True)
        info.update(i)
        return y
    timed("joint_line", line)
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt_amb = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    amb = {k: float(ambiguity_level(t, v, pk, xs, cfg.system.vs, dt_amb)) for k, v in env.items()}
    lo, hi = min(xs.min(), 0.0), max(xs.max(), 0.0); pad = max(0.12 * (hi - lo), 500.0)
    res = dict(amb, scene=sc, baseline=tag, group=grp, nrx=cfg.Nrx, bxt=[float(b) for b in cfg.array.bxt], kind=kind,
               n_targets=int(len(xs)), label=c["label"], max_slope_deg=W.max_slope_deg(terrain, lo - pad, hi + pad),
               dh=float(np.ptp(terrain(np.linspace(lo - pad, hi + pad, 2001)))), seconds=sec,
               line_iterations=info["iterations"], line_unexplained_dB=float(20 * np.log10(info["unexplained"])), version=VERSION)
    return res, dict(c=c, cfg=cfg, tr=tr, env=env, amb=amb, t=t, pk=pk, Mch=Mch, kind=kind)


# ----------------------------------------------------------------------------------------------- page
def channel_geometry(ax, cfg):
    """Transmitter and receivers seen from above, in the frame of the transmitter."""
    bat, bxt, n = cfg.array.bat, cfg.array.bxt, cfg.Nrx
    xr, yr = -bat, bxt
    ylo, yhi = min(0.0, yr.min()), max(0.0, yr.max())
    span_y = max(yhi - ylo, 1.0)
    span_x = max(np.ptp(np.r_[0.0, xr]), 1.0)
    for i in range(n):
        ax.plot([0.0, xr[i]], [0.0, yr[i]], color=CH[i], lw=1.0, ls=":", zorder=1)
    for i in range(n):
        ax.plot(xr[i], yr[i], "o", ms=13, color=CH[i], mec="k", zorder=4,
                label=f"Rx {i}: $b_{{at}}$ = {bat[i]:.2f} m, $b_{{xt}}$ = {bxt[i]:g} m")
        at_tx = np.hypot(xr[i], yr[i]) < 0.02 * max(span_x, span_y)
        ax.annotate(f"Rx {i}", (xr[i], yr[i]), xytext=(-12, 10) if at_tx else (9, 5), textcoords="offset points",
                    ha="right" if at_tx else "left", color=CH[i], fontsize=10, fontweight="bold")
    ax.plot(0.0, 0.0, marker="*", ms=17, color="gold", mec="k", ls="none", zorder=5, label="Tx")
    ax.annotate("Tx", (0.0, 0.0), xytext=(9, -14), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.annotate("", xy=(0.10 * span_x, yhi + 0.13 * span_y), xytext=(-0.28 * span_x, yhi + 0.13 * span_y),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.4))
    ax.text(-0.29 * span_x, yhi + 0.13 * span_y, "flight direction ", ha="right", va="center", fontsize=9, color="0.25")
    ax.annotate("", xy=(0.12 * span_x, ylo + 0.95 * span_y), xytext=(0.12 * span_x, ylo + 0.65 * span_y),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.4))
    ax.text(0.135 * span_x, ylo + 0.80 * span_y, "towards\nthe scene", fontsize=8.5, color="0.25", va="center")
    many = n > 4
    ax.set_xlim(-1.5 * span_x, (2.3 if many else 0.62) * span_x)
    ax.set_ylim(ylo - 0.15 * span_y, yhi + 0.22 * span_y)
    ax.grid(alpha=0.3)
    ax.set_xlabel("along-track position relative to the Tx [m]")
    ax.set_ylabel("cross-track position (horizontal) [m]")
    ax.set_title("Channels seen from above (all at the same altitude; axes not to scale)", fontsize=10)
    if many:
        ax.legend(fontsize=7.2, loc="upper right", framealpha=0.92)
    else:
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        covered = np.any((xr < x0 + 0.5 * (x1 - x0)) & (yr > y1 - (0.14 + 0.07 * n) * (y1 - y0)))   # a receiver under the legend
        if covered:
            ax.legend(fontsize=7.6, loc="lower left", framealpha=0.92)
        else:
            ax.legend(fontsize=7.6, loc="upper left", bbox_to_anchor=(0.0, 0.92), framealpha=0.92)


def page(sc, tag, res, d, out_png):
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
    ax.plot(t, db["joint_occ"], color="tab:pink", lw=1.4, ls=(0, (5, 3)), label="joint in the sub-apertures, occupied parcels only (occupied=True)")
    ax.plot(t, db["joint_line"], color="tab:purple", lw=1.3, label="joint over the whole line (DEM applied at every position)")
    ax.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | joint in the sub-apertures {amb['joint_occ']:.1f} dB | "
                 f"joint over the whole line {amb['joint_line']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.plot(t, db["whole"], color="tab:blue", lw=1.2, ls=(0, (1, 1.5)), label="SATA whole band (DEM)")
    az.plot(t, db["sub"], color="tab:orange", lw=1.2, ls=(0, (3, 1.5)), label="SATA per sub-band (DEM)")
    az.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | SATA whole band "
                 f"{amb['whole']:.1f} dB | SATA per sub-band {amb['sub']:.1f} dB", fontsize=10)
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
    channel_geometry(fig.add_subplot(gs[1, 1]), cfg)
    # the DEM table
    am = fig.add_subplot(gs[1, 2])
    n = np.arange(cfg.Na_ch)
    for i in range(cfg.Nrx):
        am.step(n, 1e3 * Mch[i], where="mid", color=CH[i], lw=1.3, label=f"channel {i} ($b_{{xt}}$ = {cfg.array.bxt[i]:g} m)")
    i1 = int(np.argmax(np.abs(cfg.array.bxt)))
    cells1 = np.array([SS.az_pixel_of_scatterer(cfg, x, i1) for x in xs])
    if len(cells1) <= 60:
        for cc in cells1:
            am.axvline(cc, color="0.75", lw=0.6, zorder=0)
    am.plot(cells1, 1e3 * Mch[i1][np.clip(cells1, 0, cfg.Na_ch - 1)], "o", ms=5, color=CH[i1], mec="k", zorder=5,
            label=f"cells of the point targets (channel {i1}; not given to any method)")
    am.set_xlim(0, cfg.Na_ch - 1); am.grid(alpha=0.3)
    am.set_xlabel("azimuth cell of the channel line"); am.set_ylabel("$\\Delta C_0$ in the table [mm]")
    lo_, hi_ = am.get_ylim(); am.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    am.set_title("$\\Delta C_0$ per cell in the DEM table of every channel (look-up table from the terrain)", fontsize=10)
    am.legend(fontsize=7.8 if cfg.Nrx <= 4 else 6.6, loc="upper center", ncol=2)
    fig.text(0.5, 0.012, "Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions of every "
             "target. All methods use the same look-up table from the terrain (DEM); none uses the positions of the targets. "
             f"$N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz, channel 0 is the transmitter "
             f"($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)}.", ha="center", fontsize=8.5, color="0.35")
    fig.savefig(out_png, dpi=90); plt.close(fig)


# ----------------------------------------------------------------------------------------------- driver
def _paths(sc, tag):
    return os.path.join(OUT, "numbers", f"{sc}__{tag}.json"), os.path.join(OUT, "pages", f"{sc}__{tag}.png")


def _done(sc, tag, pages):
    pj, pp = _paths(sc, tag)
    ok = os.path.exists(pj) and json.load(open(pj)).get("version") == VERSION
    return ok and (not pages or os.path.exists(pp))


def _job(a):
    sc, tag, pages, deadline = a
    if time.time() > deadline:
        return None
    t0 = time.time()
    res, d = compute(sc, tag)
    pj, pp = _paths(sc, tag)
    if pages:
        page(sc, tag, res, d, pp)
    json.dump(res, open(pj, "w"), indent=1)
    return f"{sc:22s} {tag:8s} ref {res['ref']:6.1f} none {res['none']:6.1f} whole {res['whole']:6.1f} sub {res['sub']:6.1f} " \
           f"joint {res['joint_occ']:6.1f} line {res['joint_line']:6.1f} ({time.time() - t0:.0f} s)"


def load():
    return {(r["scene"], r["baseline"]): r for r in (json.load(open(f)) for f in glob.glob(os.path.join(OUT, "numbers", "*.json")))}


def table():
    res = load()
    for sc, tag in all_jobs():
        r = res.get((sc, tag))
        if r:
            print(f"{r['group']:4s} {sc:22s} {tag:8s} | ref {r['ref']:6.1f} none {r['none']:6.1f} whole {r['whole']:6.1f} sub {r['sub']:6.1f} "
                  f"joint {r['joint_occ']:6.1f} line {r['joint_line']:6.1f}")


def pdf():
    from joint_sweep_summary import bundle
    res = load()
    pngs, outline = [], []
    for grp, text in GROUP_TEXT.items():
        first = True
        for sc, tag in PAGES:
            pp = _paths(sc, tag)[1]
            if SCENES[sc][0] != grp or not os.path.exists(pp):
                continue
            if first:
                outline.append((text.capitalize(), len(pngs), 0)); first = False
            r = res[(sc, tag)]
            lab = r["label"].replace("$\\pi$", "pi").replace("$^\\circ$", " deg").replace("$\\pm$", "+-")
            outline.append((f"{lab} | bxt {W.bxt_text(r['bxt'])}", len(pngs), 1))
            pngs.append(pp)
    return bundle(pngs, os.path.join(OUT, "audit_dem.pdf"), outline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", action="store_true", help="the jobs of PAGES, with the PNG page")
    ap.add_argument("--only", nargs="+", default=None, help="scene ids (default: all)")
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo the selected jobs even if they are done")
    a = ap.parse_args()
    if a.table:
        return table()
    if a.pdf:
        return print("wrote", pdf())
    for sub in ("numbers", "pages"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    jobs = PAGES if a.pages else all_jobs()
    todo = [(sc, tag) for sc, tag in jobs if (a.only is None or sc in a.only) and (a.force or not _done(sc, tag, a.pages))]
    print(len(todo), "jobs to do", flush=True)
    deadline = time.time() + a.budget
    args = [(sc, tag, a.pages, deadline) for sc, tag in todo]
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
    left = sum(not _done(sc, tag, a.pages) for sc, tag in jobs if a.only is None or sc in a.only)
    print("left:", left, flush=True)


if __name__ == "__main__":
    main()
