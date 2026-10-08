"""
Scene = a terrain h(x) (the whole scene; the look-up table is built from it at every cell,
terrain_lookup_table) + a few point targets placed ON the terrain, whose echoes are simulated
and whose IRFs are the sanity check of the table. Figure in the layout of the sweep pages.

    python run_terrain_lut_cases.py                       # the default sine / cosine cases, table from the terrain
    python run_terrain_lut_cases.py --map footprint       # same scenes, block map with copies built from the point targets
    python run_terrain_lut_cases.py --cases SIN-A250-P4 COS-A500-P8
    python run_terrain_lut_cases.py --terrain "300*sin(2*pi*x/5000)" --targets=-5000,-2000,1000,4000 --name MY
    python run_terrain_lut_cases.py --n-targets 1 3 7 15 31 --pdf   # every preset with 1, 3, ... point targets, one PDF
    python run_terrain_lut_cases.py --n-targets --cases SIN-A250-P4 COS-A250-P4 --tag a    # part of a sweep (several processes
    python run_terrain_lut_cases.py --n-targets --cases SIN-A500-P8 COS-A500-P8 --tag b    #  sharing the same --out), then
    python run_terrain_lut_cases.py --bundle-only                                           # summary + PDF from the results
    python run_terrain_lut_cases.py --n-targets --worker 0/4 --resume --time-budget 140      # 1 of 4 workers, resumable

terrain   : numpy expression in x [m] (azimuth relative to the reference point), or a preset id
targets   : azimuth positions [m]; the height is the terrain there
n-targets : numbers of point targets (target_positions: evenly spaced around x = -100 m over <= 10.8 km,
            spacing nudged so that no ambiguity x_j +- X lands on another target); overrides --targets
"""
import argparse
import glob
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import sata1d as S
import sata1d.sata as SS
from sata1d.subband import sata_channels_subband, channel_spectrum
from sata1d.reconstruction import reconstruct_from_spectra
from run_one_case import ambiguity_level, oracle_channels
from run_geometry_sweep import SATA_OSF, BXT_MAX, BXT_SEED
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy

X_FOLD = 6228.6                                      # one fold of PRF_ch on the ground [m] (405.06 cells)
PRESETS = {
    # h(x) = A + A sin(2 pi x / P) and A + A cos(2 pi x / P): the profile touches the reference surface
    # (h = 0) at its minima and reaches 2A. Gentle terrains first (less than one period across the
    # +-8 km scene, max slope 2 pi A / P < 1.5 deg), then a few that do undulate (2-3 periods, < 3 deg).
    "SIN-B100-A100-P50": ("100+100*sin(2*pi*x/50000)", "h = 100 + 100 sin(2 pi x / 50 km)"),
    "COS-B100-A100-P50": ("100+100*cos(2*pi*x/50000)", "h = 100 + 100 cos(2 pi x / 50 km)"),
    "SIN-B50-A50-P30": ("50+50*sin(2*pi*x/30000)", "h = 50 + 50 sin(2 pi x / 30 km)"),
    "COS-B50-A50-P30": ("50+50*cos(2*pi*x/30000)", "h = 50 + 50 cos(2 pi x / 30 km)"),
    "SIN-B80-A80-P20": ("80+80*sin(2*pi*x/20000)", "h = 80 + 80 sin(2 pi x / 20 km)"),
    "COS-B80-A80-P20": ("80+80*cos(2*pi*x/20000)", "h = 80 + 80 cos(2 pi x / 20 km)"),
    "SIN-B60-A60-P8": ("60+60*sin(2*pi*x/8000)", "h = 60 + 60 sin(2 pi x / 8 km)"),
    "COS-B60-A60-P8": ("60+60*cos(2*pi*x/8000)", "h = 60 + 60 cos(2 pi x / 8 km)"),
    "SIN-B40-A40-P5": ("40+40*sin(2*pi*x/5000)", "h = 40 + 40 sin(2 pi x / 5 km)"),
    "COS-B40-A40-P5": ("40+40*cos(2*pi*x/5000)", "h = 40 + 40 cos(2 pi x / 5 km)"),
    # h(x) = A - A cos(2 pi x / P) = A + A sin(2 pi x / P - pi/2): h(0) = 0, so a point target at x = 0 sits
    # exactly on the reference point (same azimuth, h = 0) and on the terrain (--target-centre 0, odd N)
    "REF-A100-P50": ("100-100*cos(2*pi*x/50000)", "h = 100 - 100 cos(2 pi x / 50 km), h(0) = 0"),
    "REF-A50-P30": ("50-50*cos(2*pi*x/30000)", "h = 50 - 50 cos(2 pi x / 30 km), h(0) = 0"),
    "REF-A80-P20": ("80-80*cos(2*pi*x/20000)", "h = 80 - 80 cos(2 pi x / 20 km), h(0) = 0"),
    "REF-A60-P8": ("60-60*cos(2*pi*x/8000)", "h = 60 - 60 cos(2 pi x / 8 km), h(0) = 0"),
    "REF-A40-P5": ("40-40*cos(2*pi*x/5000)", "h = 40 - 40 cos(2 pi x / 5 km), h(0) = 0"),
    "RAMP-H500-L12": ("500*clip((x+6000)/12000,0,1)", "h = ramp 0 -> 500 m from x = -6 km to +6 km, flat outside"),
    "RAMP-H1000-L12": ("1000*clip((x+6000)/12000,0,1)", "h = ramp 0 -> 1000 m from x = -6 km to +6 km, flat outside"),
    "RAMP-H1000-L4": ("1000*clip((x+2000)/4000,0,1)", "h = ramp 0 -> 1000 m from x = -2 km to +2 km, flat outside"),
}
LEGACY_PRESETS = {                                   # earlier profiles (B > A), available through --cases
    "COS-B500-A100-P50": ("500+100*cos(2*pi*x/50000)", "h = 500 + 100 cos(2 pi x / 50 km)"),
    "SIN-B300-A50-P30": ("300+50*sin(2*pi*x/30000)", "h = 300 + 50 sin(2 pi x / 30 km)"),
    "COS-B400-A80-P20": ("400+80*cos(2*pi*x/20000)", "h = 400 + 80 cos(2 pi x / 20 km)"),
    "SIN-B250-A60-P8": ("250+60*sin(2*pi*x/8000)", "h = 250 + 60 sin(2 pi x / 8 km)"),
    "COS-B200-A40-P5": ("200+40*cos(2*pi*x/5000)", "h = 200 + 40 cos(2 pi x / 5 km)"),
    "SIN-B500-A100-P50": ("500+100*sin(2*pi*x/50000)", "h = 500 + 100 sin(2 pi x / 50 km)"),
    "COS-B300-A50-P30": ("300+50*cos(2*pi*x/30000)", "h = 300 + 50 cos(2 pi x / 30 km)"),
    "SIN-B400-A80-P20": ("400+80*sin(2*pi*x/20000)", "h = 400 + 80 sin(2 pi x / 20 km)"),
    "COS-B250-A60-P8": ("250+60*cos(2*pi*x/8000)", "h = 250 + 60 cos(2 pi x / 8 km)"),
    "SIN-B200-A40-P5": ("200+40*sin(2*pi*x/5000)", "h = 200 + 40 sin(2 pi x / 5 km)"),
    "SIN-B150-A30-P2": ("150+30*sin(2*pi*x/2000)", "h = 150 + 30 sin(2 pi x / 2 km)"),
    "COS-B150-A30-P2": ("150+30*cos(2*pi*x/2000)", "h = 150 + 30 cos(2 pi x / 2 km)"),
    "SIN-B200-A50-P4": ("200+50*sin(2*pi*x/4000)", "h = 200 + 50 sin(2 pi x / 4 km)"),
    "COS-B200-A50-P4": ("200+50*cos(2*pi*x/4000)", "h = 200 + 50 cos(2 pi x / 4 km)"),
    "SIN-B300-A100-P8": ("300+100*sin(2*pi*x/8000)", "h = 300 + 100 sin(2 pi x / 8 km)"),
    "COS-B300-A100-P8": ("300+100*cos(2*pi*x/8000)", "h = 300 + 100 cos(2 pi x / 8 km)"),
    "SIN-B400-A200-P10": ("400+200*sin(2*pi*x/10000)", "h = 400 + 200 sin(2 pi x / 10 km)"),
    "COS-B400-A200-P10": ("400+200*cos(2*pi*x/10000)", "h = 400 + 200 cos(2 pi x / 10 km)"),
    "SIN-B600-A400-P25": ("600+400*sin(2*pi*x/25000)", "h = 600 + 400 sin(2 pi x / 25 km)"),
    "COS-B600-A400-P25": ("600+400*cos(2*pi*x/25000)", "h = 600 + 400 cos(2 pi x / 25 km)"),
    "SIN-B800-A600-P60": ("800+600*sin(2*pi*x/60000)", "h = 800 + 600 sin(2 pi x / 60 km)"),
    "COS-B800-A600-P60": ("800+600*cos(2*pi*x/60000)", "h = 800 + 600 cos(2 pi x / 60 km)"),
    "SIN-A250-P4": ("250*sin(2*pi*x/4000)", "h = 250 sin(2 pi x / 4 km)"),
    "COS-A250-P4": ("250*cos(2*pi*x/4000)", "h = 250 cos(2 pi x / 4 km)"),
    "SIN-A500-P8": ("500*sin(2*pi*x/8000)", "h = 500 sin(2 pi x / 8 km)"),
    "COS-A500-P8": ("500*cos(2*pi*x/8000)", "h = 500 cos(2 pi x / 8 km)"),
    "SIN-A250-PX": (f"250*sin(2*pi*x/{X_FOLD})", "h = 250 sin(2 pi x / X), period = fold X = 6.23 km"),
    "COS-A250-PX": (f"250*cos(2*pi*x/{X_FOLD})", "h = 250 cos(2 pi x / X), period = fold X = 6.23 km"),
}
ALL_PRESETS = {**PRESETS, **LEGACY_PRESETS}
DEFAULT_TARGETS = [-5500.0, -3700.0, -1900.0, -100.0, 1700.0, 3500.0, 5300.0]
DEFAULT_COUNTS = [1, 2, 3, 4, 5, 6, 7, 9, 11, 15, 31]
CH_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]


def terrain_from(expr):
    ns = {k: getattr(np, k) for k in ("sin", "cos", "exp", "tan", "sqrt", "abs", "pi", "tanh", "clip", "maximum", "minimum")}
    return lambda x: np.broadcast_to(eval(expr, {"__builtins__": {}}, dict(ns, x=np.asarray(x, float))), np.shape(x)) * 1.0


MAP_LABEL = {"terrain": "look-up table from the terrain", "footprint": "block map with copies from the targets"}


def target_positions(n, span=10800.0, centre=-100.0, keep_clear=300.0):
    """n azimuths centred at `centre`, evenly spaced over at most `span` m (n = 7 gives DEFAULT_TARGETS).
    When an ambiguity position x_j +- X would fall within `keep_clear` m of another target, the spacing
    is nudged to X / (k + 1/2) so that the ambiguities fall half-way between two targets and the
    +-20 ms ambiguity windows of the metric do not sit on a neighbour's main lobe."""
    if n == 1:
        return np.array([centre])
    d = span / (n - 1)
    k = np.arange(1, int(X_FOLD / d) + 2)
    if np.abs(X_FOLD - k * d).min() < min(keep_clear, 0.5 * d - 1.0):
        d = X_FOLD / (np.round(X_FOLD / d - 0.5) + 0.5)
    return centre + (np.arange(n) - (n - 1) / 2.0) * d


def png_name(out_dir, name, mp, tag=""):
    return os.path.join(out_dir, f"{name}{tag}_{'terrain_lut' if mp == 'terrain' else 'footprint'}.png")


def run_case(name, expr, desc, target_x, out_dir, mp="terrain", tag="", full_line=False, compare=False):
    t0 = time.time()
    terrain = terrain_from(expr)
    xs = np.asarray(target_x, float); hs = terrain(xs)
    cfg = S.make_config(list(zip(xs, hs)), bxt_max=BXT_MAX, seed=BXT_SEED)
    tr = S.build_platform_tracks(cfg)
    dC0 = S.delta_C0_per_target(cfg, tr)
    tmaps = np.array([S.terrain_lookup_table(cfg, tr, i, terrain, step=4) for i in range(cfg.Nrx)])    # the scene -> table
    fmaps = np.array([SS.build_delta_C0_array(cfg, tr, i, mode="footprint", dC0=dC0) for i in range(cfg.Nrx)])   # targets -> blocks + copies
    maps = tmaps if mp == "terrain" else fmaps
    ch = S.generate_channels(cfg, tr)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))

    def whole(M):
        out = ch.copy()
        for i in range(cfg.Nrx):
            out[i] = SS.sata_1d(out[i], M[i], rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=cfg.system.wl,
                                r=cfg.scene.r0, squint=cfg.sq_tx, inverse=True, sata_osf=SATA_OSF)
        return S.reconstruct(cfg, tr, out)

    def sub(M):
        spec = []
        for k in range(cfg.Nsb):
            corr = sata_channels_subband(cfg, tr, ch, k, sata_osf=SATA_OSF, maps=list(M))
            spec.append([channel_spectrum(cfg, corr[j]) for j in range(cfg.Nrx)])
        return reconstruct_from_spectra(cfg, tr, spec, True)

    env = {"ref": F(S.generate_reference(cfg, tr)),
           "none": F(S.reconstruct(cfg, tr, ch)),
           "whole": F(whole(maps)),
           "sub": F(sub(tmaps)) if mp == "terrain" else F(S.reconstruct_subband(cfg, tr, ch, sata_osf=SATA_OSF, mode="footprint", dC0=dC0)),
           "other": F(whole(fmaps if mp == "terrain" else tmaps)),
           "oracle": F(S.reconstruct(cfg, tr, oracle_channels(cfg, tr, dC0)))}
    if compare:                                   # the other map per sub-band too: its IRFs replace the no-SATA panel
        env["other_sub"] = (F(S.reconstruct_subband(cfg, tr, ch, sata_osf=SATA_OSF, mode="footprint", dC0=dC0)) if mp == "terrain"
                            else F(sub(tmaps)))
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    amb = {k: ambiguity_level(t, v, pk, xs, cfg.system.vs, dt) for k, v in env.items()}
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}

    # table value at the targets' cells vs their own delta_C0 (the sanity check of the table)
    cells0 = np.array([SS.az_pixel_of_scatterer(cfg, x, 0) for x in xs])
    err = 1e3 * np.abs(tmaps[0, cells0] - dC0[:, 0]).max()

    fig = plt.figure(figsize=(18, 9.6))
    gs = GridSpec(2, 2, width_ratios=[1.45, 1.25], hspace=0.36, wspace=0.14, figure=fig)
    fig.suptitle(f"{name}: terrain {desc}, {len(xs)} point target{'s' if len(xs) > 1 else ''} on it | "
                 + (f"{MAP_LABEL[mp]} (top) and {MAP_LABEL['footprint' if mp == 'terrain' else 'terrain']} (bottom)" if compare else MAP_LABEL[mp]),
                 fontsize=12, y=0.995)
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    lab = "look-up table" if mp == "terrain" else "block map with copies"
    ax.plot(t, db["whole"], color="tab:blue", lw=1.6, ls=(0, (5, 3)), label=f"SATA whole band ({lab})")
    ax.plot(t, db["sub"], color="tab:orange", lw=1.8, ls=(0, (1, 2)), label=f"SATA per sub-band ({lab})")
    ax.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | SATA whole band {amb['whole']:.1f} dB | "
                 f"SATA per sub-band {amb['sub']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    if compare:
        lab_o = "block map with copies" if mp == "terrain" else "look-up table"
        az.plot(t, db["other"], color="tab:blue", lw=1.6, ls=(0, (5, 3)), label=f"SATA whole band ({lab_o})")
        az.plot(t, db["other_sub"], color="tab:orange", lw=1.8, ls=(0, (1, 2)), label=f"SATA per sub-band ({lab_o})")
        az.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | SATA whole band {amb['other']:.1f} dB | "
                     f"SATA per sub-band {amb['other_sub']:.1f} dB", fontsize=10)
    else:
        az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
        az.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | improvement: "
                     f"whole band {amb['none'] - amb['whole']:.1f} dB, per sub-band {amb['none'] - amb['sub']:.1f} dB", fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")

    xs_, dys, hs_ = scene_coordinates(xs, hs)
    # full_line: the 3D panel spans the whole azimuth line of the acquisition (as the IRF and the map do) and the
    # 2D panel is a zoom on the point targets and the reference point
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs if full_line else 8000.0
    xg = np.linspace(-half, half, 2401 if full_line else 801); hg = terrain(xg)
    lim = half if full_line else 8500.0
    zoom = (-lim, lim)
    if full_line:
        lo2, hi2 = min(xs_.min(), 0.0), max(xs_.max(), 0.0)
        pad2 = max(0.12 * (hi2 - lo2), 500.0)
        zoom = (lo2 - pad2, hi2 + pad2)
    hz = hg[(xg >= zoom[0]) & (xg <= zoom[1])]              # the terrain inside the 2D panel
    gg = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.18, width_ratios=[1.0, 1.1])
    g2 = fig.add_subplot(gg[0, 0]); g3 = fig.add_subplot(gg[0, 1], projection="3d")
    geometry_2d(g2, xs_, dys, hs_, False, title=None, connect=False)
    g2.set_title(("Zoom on the point targets and the reference point" if full_line else "Scene (terrain) and point targets")
                 + f" | $\\Delta h_{{max}}$ = {np.ptp(hz):.0f} m", fontsize=9.5, pad=36)
    g2.fill_between(xg, min(hg.min(), 0) - 0.05 * np.ptp(hg), hg, color="#D7C9A7", alpha=0.45, lw=0)
    g2.plot(xg, hg, color="#6D4C2F", lw=1.2, zorder=2, label="terrain (the scene)")
    g2.plot(0.0, 0.0, marker="*", ms=13, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $h$ = 0)")
    g2.set_xlim(*zoom)
    if full_line:
        ylo, yhi = min(hz.min(), 0.0), max(hz.max(), hs_.max(), 1.0)
        g2.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))
    else:
        g2.set_ylim(min(hg.min(), 0) - 0.25 * np.ptp(hg), hg.max() + 0.25 * np.ptp(hg))
    g2.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.1, handlelength=1.6, columnspacing=1.2)
    geometry_3d(g3, xs_, dys, hs_, False, connect=False,
                title=(f"Whole azimuth line ({2 * half / 1e3:.0f} km) in 3D" if full_line else "in 3D") + " (x, $\\Delta y$, h)")
    g3.plot(xg, iso_range_dy(hg), hg, color="#6D4C2F", lw=1.0)
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=6)
    g3.set_xlim(-lim, lim)
    if full_line:                                        # azimuth in km; no height labels in 3D (they overlap at this scale)
        from matplotlib.ticker import FuncFormatter
        km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
        g2.xaxis.set_major_formatter(km); g2.set_xlabel("azimuth position $x$ [km]")
        g3.xaxis.set_major_formatter(km); g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4)
        for t_ in list(g3.texts):
            t_.remove()

    am = fig.add_subplot(gs[1, 1])
    n = np.arange(cfg.Na_ch)
    for i in range(cfg.Nrx):
        am.plot(n, 1e3 * maps[i], color=CH_COLORS[i], lw=1.2, label=f"channel {i} ($b_{{xt}}$ = {cfg.array.bxt[i]:.1f} m)")
    if mp == "footprint":
        am.plot(n, 1e3 * tmaps[0], color="0.5", lw=0.8, ls=":", label="terrain $\\Delta C_0$, channel 0 (for reference)")
    for c in cells0:
        am.axvline(c, color="0.75", lw=0.6, zorder=0)
    am.plot(cells0, 1e3 * dC0[:, 0], "o", ms=5, color=CH_COLORS[0], mec="k", zorder=5, label="point targets: own $\\Delta C_0$ (channel 0)")
    am.set_xlim(0, cfg.Na_ch - 1); am.grid(alpha=0.3)
    am.set_xlabel("azimuth cell of the channel line"); am.set_ylabel("$\\Delta C_0$ in the map [mm]")
    lo_, hi_ = am.get_ylim(); am.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    am.set_title("$\\Delta C_0$ per cell", fontsize=10); am.legend(fontsize=8, loc="upper center", ncol=3)
    other = MAP_LABEL["footprint" if mp == "terrain" else "terrain"]
    fig.text(0.5, 0.012, f"Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions "
             f"of every target. For comparison: " + (f"no SATA {amb['none']:.1f} dB, " if compare else f"{other} (whole band) {amb['other']:.1f} dB, ") +
             f"oracle {amb['oracle']:.1f} dB. Terrain table vs targets' own $\\Delta C_0$ at their cells: max |diff| = {err:.3f} mm. "
             f"$N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz, $b_{{xt}}$ = [" +
             ", ".join(f"{b:.1f}" for b in cfg.array.bxt) + "] m.", ha="center", fontsize=8.5, color="0.35")
    fig.subplots_adjust(bottom=0.1)
    os.makedirs(out_dir, exist_ok=True)
    png = png_name(out_dir, name, mp, tag)
    fig.savefig(png, dpi=95, bbox_inches="tight"); plt.close(fig)
    np.savez_compressed(png[:-4] + ".npz", t=t, xs=xs, hs=hs, dC0=dC0, maps=maps, terrain_maps=tmaps,
                        **{f"env_{k}": (v / pk).astype(np.float32) for k, v in env.items()})
    print(f"{name + tag:16s} " + " | ".join(f"{k} {v:.1f}" for k, v in amb.items()) + f" | table err {err:.3f} mm ({time.time() - t0:.0f} s)", flush=True)
    return png, {k: float(v) for k, v in amb.items()}


def save_results(results, out_dir, tag=""):
    """results[(preset, n)] -> <out>/results<tag>.json (one file per process sharing --out)."""
    with open(os.path.join(out_dir, f"results{tag}.json"), "w") as f:
        json.dump({f"{cid}|{n}": amb for (cid, n), amb in results.items()}, f, indent=1)


def load_results(out_dir):
    res = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "results*.json"))):
        with open(path) as f:
            for k, amb in json.load(f).items():
                cid, n = k.split("|")
                res[(cid, int(n))] = amb
    return res


def summary_figure(results, counts, out_dir, mp):
    """Highest ambiguity against the number of point targets, one panel per preset.
    results[(preset, n)] = amb dict of run_case."""
    cids = [c for c in ALL_PRESETS if any((c, n) in results for n in counts)]
    rows = int(np.ceil(len(cids) / 3.0))
    fig, axs = plt.subplots(rows, 3, figsize=(18, 3.3 * rows + 0.6), sharex=True, sharey=True, squeeze=False)
    lab = "look-up table" if mp == "terrain" else "block map with copies"
    other = "block map with copies" if mp == "terrain" else "look-up table"
    series = [("none", "no SATA", "tab:red", "-"), ("whole", f"SATA whole band ({lab})", "tab:blue", "-"),
              ("sub", f"SATA per sub-band ({lab})", "tab:orange", "-"), ("other", f"SATA whole band ({other})", "0.45", "--"),
              ("oracle", "oracle", "k", ":"), ("ref", "reference", "tab:green", "-")]
    for ax, cid in zip(axs.ravel(), cids):
        ns = [n for n in counts if (cid, n) in results]
        for key, label, col, ls in series:
            ax.plot(ns, [results[(cid, n)][key] for n in ns], marker="o", ms=4, color=col, ls=ls, lw=1.5, label=label)
        ax.set_title(f"{cid}: {ALL_PRESETS[cid][1]}", fontsize=10); ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    for ax in axs.ravel()[len(cids):]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("number of point targets")
    for ax in axs[:, 0]:
        ax.set_ylabel("highest ambiguity [dB]")
    axs[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Highest ambiguity against the number of point targets on the terrain | {MAP_LABEL[mp]}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(out_dir, f"summary_vs_n_targets_{'terrain_lut' if mp == 'terrain' else 'footprint'}.png")
    fig.savefig(png, dpi=95); plt.close(fig)
    return png


def bundle_pdf(pngs, out):
    """One page per PNG, at the PNG's own size."""
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out) as pdf:
        for png in pngs:
            im = mpimg.imread(png)
            h, w = im.shape[:2]
            fig = plt.figure(figsize=(w / 95.0, h / 95.0))
            ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(im); ax.axis("off")
            pdf.savefig(fig); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(PRESETS), choices=list(ALL_PRESETS))
    ap.add_argument("--terrain", default=None, help="numpy expression in x [m] (overrides --cases)")
    ap.add_argument("--name", default="CUSTOM")
    ap.add_argument("--targets", default=",".join(f"{x:g}" for x in DEFAULT_TARGETS), help="azimuths [m], comma-separated")
    ap.add_argument("--n-targets", type=int, nargs="*", default=None,
                    help=f"numbers of point targets (target_positions); no value = {' '.join(map(str, DEFAULT_COUNTS))}")
    ap.add_argument("--target-centre", type=float, default=-100.0,
                    help="centre of the evenly spaced targets of --n-targets [m]; 0 with odd N puts a target on the reference point")
    ap.add_argument("--map", default="terrain", choices=["terrain", "footprint"])
    ap.add_argument("--out", default=None, help="default sweep_terrain_<map>_cases (or _ntargets with --n-targets)")
    ap.add_argument("--tag", default="", help="suffix of the results json, for several processes sharing one --out")
    ap.add_argument("--pdf", action="store_true", help="also bundle the pages (and the summary, with --n-targets) into <out>.pdf")
    ap.add_argument("--bundle-only", action="store_true", help="only the summary and the PDF, from the results already in --out")
    ap.add_argument("--worker", default=None, help="i/n: this process runs every n-th case of the sweep starting at i (json tag _w<i>)")
    ap.add_argument("--resume", action="store_true", help="skip the cases already in the results*.json of --out")
    ap.add_argument("--time-budget", type=float, default=None, help="do not start a new case after this many seconds")
    ap.add_argument("--full-line", action="store_true",
                    help="geometry panels over the whole azimuth line of the acquisition (the extent of the IRF and of the map)")
    ap.add_argument("--compare-maps", action="store_true",
                    help="lower IRF panel: SATA with the other map (whole band and per sub-band) instead of no SATA")
    a = ap.parse_args()
    stem = "sweep_terrain_lut" if a.map == "terrain" else "sweep_terrain_footprint"
    sweep = a.n_targets is not None or a.bundle_only
    out = a.out or (stem + ("_ntargets" if sweep else "_cases"))
    if a.bundle_only:
        results = load_results(out)
        counts = sorted({n for (_, n) in results})
        pngs = [png_name(out, cid, a.map, f"_N{n:02d}") for cid in ALL_PRESETS for n in counts if (cid, n) in results]
        pngs.append(summary_figure(results, counts, out, a.map))
        print("wrote", bundle_pdf(pngs, out + ".pdf"))
        return
    targets = [float(v) for v in a.targets.split(",") if v]
    if a.terrain:
        run_case(a.name, a.terrain, a.terrain, targets, out, a.map, full_line=a.full_line, compare=a.compare_maps)
        return
    pngs, results = [], {}
    counts = (a.n_targets or DEFAULT_COUNTS) if sweep else [None]
    jobs = [(cid, n) for cid in a.cases for n in counts]
    tag = a.tag
    if a.worker:
        i, nw = (int(v) for v in a.worker.split("/"))
        jobs, tag = jobs[i::nw], a.tag or f"_w{i}"
    done = load_results(out) if a.resume else {}
    own = os.path.join(out, f"results{tag}.json")
    if a.resume and os.path.exists(own):
        results = {(k.split("|")[0], int(k.split("|")[1])): v for k, v in json.load(open(own)).items()}
    t_start = time.time()
    for cid, n in jobs:
        if (cid, n) in done:
            continue
        if a.time_budget is not None and time.time() - t_start > a.time_budget:
            print(f"time budget reached, {sum(j not in done for j in jobs)} cases left", flush=True)
            break
        expr, desc = ALL_PRESETS[cid]
        xs = targets if n is None else target_positions(n, centre=a.target_centre)
        png, amb = run_case(cid, expr, desc, xs, out, a.map, tag="" if n is None else f"_N{n:02d}", full_line=a.full_line, compare=a.compare_maps)
        pngs.append(png); results[(cid, n)] = amb; done[(cid, n)] = amb
        if sweep:
            save_results(results, out, tag)
    if sweep and not a.worker and all(j in done for j in jobs):
        pngs.append(summary_figure(done, counts, out, a.map))
    if a.pdf:
        print("wrote", bundle_pdf(pngs, out + ".pdf"))


if __name__ == "__main__":
    main()
