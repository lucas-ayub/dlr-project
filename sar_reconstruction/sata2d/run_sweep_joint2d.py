"""
Sweep of the 2-D joint reconstruction (joint2d.py) on 3-D scenes: terrains h(x, dy) that change along azimuth AND
along ground range, point targets on the terrain spread in both directions, several numbers of channels and
cross-track baselines. Six-panel pages, in the layout of the 1-D sweep pages:

    top left      azimuth IRF: reference + joint (2-D, final version)
    bottom left   azimuth IRF: reference + no SATA (the two-step reconstruction with the flat-earth filters)
    top middle    the scene seen from above, zoom on the targets (terrain height, targets, reference point)
    top right     the scene in 3D, the whole imaged area
    bottom middle the channels seen from above
    bottom right  the DEM table of the channel with the largest baseline, in radar geometry (azimuth x slant range)

Every method reads only the DEM (dem_maps: the terrain in radar geometry); none uses the positions of the targets.
Channel 0 is always the transmitter (bat = bxt = 0); along-track baselines on the DPCA condition; total PRF 2000 Hz.

Run from sar_reconstruction/ (the joint is run in two stages, one regularised pass each, so that an interrupted run
continues where it stopped):
    python -m sata2d.run_sweep_joint2d --workers 3            # numbers (resumable)
    python -m sata2d.run_sweep_joint2d --pages                # the PNG pages from the numbers
    python -m sata2d.run_sweep_joint2d --pdf                  # the PDF from the pages
    python -m sata2d.run_sweep_joint2d --table | --list
Output (default ../results/sata2d/sweep_joint2d/): numbers/<job>.json, data/<job>.npz, pages/<job>.png,
sweep_joint2d.pdf, all_numbers.json (written by --table). The state between the two stages of a job goes to --work (default: a temporary folder).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator

if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)

from .arrays import make_params_dpca
from .geometry import build_tracks_3d, generate_reference_3d, generate_channels_3d, CoeffTable3D, C0_LIGHT
from .reconstruction import range_compress, reconstruct_subband_2d
from .joint2d import dem_maps, dem_heights, image_positions, Model2D, reconstruct_joint_2d, image_to_data, processed_band

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "sata2d", "sweep_joint2d"))
WORK = os.path.join(tempfile.gettempdir(), "sweep_joint2d_work")
VERSION = 2
R0_FRAC = 0.6            # the reference range at 60 % of the swath: heights move the targets towards the radar
OSR = 2                  # range oversampling of the image (joint2d.Model2D)
CH = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive"]


# ----------------------------------------------------------------------------------------------- baselines
def random_baselines(nrx, bmax, seed=0):
    """Channel 0 on the transmitter; the others random in (0, bmax], the largest exactly bmax (as in the 1-D sweep)."""
    u = np.random.default_rng(seed).uniform(0.1, 1.0, nrx)
    u[0] = 0.0
    return (bmax * u / u.max()).round(1)


FIXED = {  # tag -> (bxt of channels 0..3, text): four channels
    "u100": (np.array([0.0, 27.0, 4.1, 1.7]), "random, U(0, 100 m)"),
    "u500": (np.array([0.0, 134.9, 20.5, 8.3]), "random, U(0, 500 m)"),
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


def bxt_text(b):
    return "[" + ", ".join(f"{v:g}" for v in b) + "] m"


# ----------------------------------------------------------------------------------------------- terrains h(x, dy)
def _bc(x, dy):
    return np.broadcast_arrays(np.asarray(x, float), np.asarray(dy, float))


def ramp_azimuth(slope_deg, length):
    """Flat at h = 0 before -length / 2, ramp along azimuth over the length, flat on top after; the same at every range."""
    t = np.tan(np.radians(slope_deg))
    return lambda x, dy: t * np.clip(_bc(x, dy)[0] + length / 2.0, 0.0, length)


def ramp_range(slope_deg, length):
    """Ramp along ground range (positive slope: the terrain rises away from the radar, it faces the radar); the same at
    every azimuth."""
    t = np.tan(np.radians(abs(slope_deg)))
    s = np.sign(slope_deg)
    return lambda x, dy: t * np.clip(s * _bc(x, dy)[1] + length / 2.0, 0.0, length)


def ramp_oblique(slope_deg, length, angle_deg):
    """Ramp along a direction that makes angle_deg with the flight direction."""
    t, c, s = np.tan(np.radians(slope_deg)), np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    return lambda x, dy: t * np.clip(_bc(x, dy)[0] * c + _bc(x, dy)[1] * s + length / 2.0, 0.0, length)


def random_terrain_2d(seed, relief, corr_x, corr_y, half_x=30e3, half_y=9e3, step=100.0, region=(22e3, 4e3)):
    """Smooth random terrain: white noise filtered by a Gaussian (correlation exp(-(x / corr_x)^2 - (dy / corr_y)^2)),
    scaled so that inside |x| <= region[0], |dy| <= region[1] it goes from 0 to the relief."""
    nx, ny = int(2 * half_x / step), int(2 * half_y / step)
    w = np.random.default_rng(seed).standard_normal((nx, ny))
    kx, ky = np.fft.fftfreq(nx, step)[:, None], np.fft.fftfreq(ny, step)[None, :]
    f = np.fft.ifft2(np.fft.fft2(w) * np.exp(-0.5 * np.pi ** 2 * ((kx * corr_x) ** 2 + (ky * corr_y) ** 2))).real
    xa, ya = (np.arange(nx) - nx // 2) * step, (np.arange(ny) - ny // 2) * step
    sel = f[np.ix_(np.abs(xa) <= region[0], np.abs(ya) <= region[1])]
    f = (f - sel.min()) * relief / np.ptp(sel)

    def h(x, dy):
        x, dy = _bc(x, dy)
        u = np.clip((x - xa[0]) / step, 0.0, nx - 1.001)
        v = np.clip((dy - ya[0]) / step, 0.0, ny - 1.001)
        i, j = u.astype(int), v.astype(int)
        a, b = u - i, v - j
        return (1 - a) * (1 - b) * f[i, j] + a * (1 - b) * f[i + 1, j] + (1 - a) * b * f[i, j + 1] + a * b * f[i + 1, j + 1]
    return h


TERRAINS = {  # name -> (h(x, dy), description)
    "ramp_az8": (ramp_azimuth(8.0, 12000.0), "ramp of 8$^\\circ$ along azimuth, 12 km long (h 0 -> 1686 m), the same at every range"),
    "ramp_az4": (ramp_azimuth(4.0, 12000.0), "ramp of 4$^\\circ$ along azimuth, 12 km long (h 0 -> 839 m), the same at every range"),
    "ramp_rg8": (ramp_range(8.0, 5000.0), "ramp of 8$^\\circ$ along ground range, facing the radar (h 0 -> 703 m over 5 km), the same at every azimuth"),
    "ramp_rg-8": (ramp_range(-8.0, 5000.0), "ramp of 8$^\\circ$ along ground range, facing away from the radar (h 703 -> 0 m over 5 km), the same at every azimuth"),
    "ramp_obl6": (ramp_oblique(6.0, 12000.0, 35.0), "ramp of 6$^\\circ$ along a direction 35$^\\circ$ off the flight direction (h 0 -> 1261 m)"),
    "sine_az": (lambda x, dy: 100.0 + 100.0 * np.sin(2 * np.pi * _bc(x, dy)[0] / 20e3) + 0.0 * _bc(x, dy)[1],
                "h = 100 + 100 sin(2$\\pi$ x / 20 km), the same at every range"),
    "eggbox": (lambda x, dy: 150.0 + 150.0 * np.cos(2 * np.pi * _bc(x, dy)[0] / 8e3) * np.cos(2 * np.pi * _bc(x, dy)[1] / 5e3),
               "h = 150 + 150 cos(2$\\pi$ x / 8 km) cos(2$\\pi$ $\\Delta y$ / 5 km)"),
    "mountain": (lambda x, dy: 1000.0 * np.exp(-0.5 * ((_bc(x, dy)[0] / 1500.0) ** 2 + (_bc(x, dy)[1] / 4000.0) ** 2)),
                 "isolated mountain 1000 m high (Gaussian, sigma 1.5 km along azimuth, 4 km along ground range)"),
    "hills": (random_terrain_2d(3, 300.0, 1500.0, 2500.0), "random hills, relief 300 m, correlation length 1.5 km (azimuth) x 2.5 km (ground range)"),
    "mountains": (random_terrain_2d(5, 1000.0, 3000.0, 6000.0), "random mountains, relief 1000 m, correlation length 3 km (azimuth) x 6 km (ground range)"),
    "escarpment": (lambda x, dy: 250.0 * (1.0 + np.tanh((_bc(x, dy)[0] * np.cos(np.radians(25.0)) + _bc(x, dy)[1] * np.sin(np.radians(25.0))) / 500.0)),
                   "escarpment 0 -> 500 m, 1 km wide, 25$^\\circ$ off the cross-track direction"),
    "plateau": (lambda x, dy: 500.0 * (np.tanh((_bc(x, dy)[0] + 3000.0) / 500.0) - np.tanh((_bc(x, dy)[0] - 3000.0) / 500.0)) / (2.0 * np.tanh(6.0))
                * (np.tanh((_bc(x, dy)[1] + 2500.0) / 1200.0) - np.tanh((_bc(x, dy)[1] - 2500.0) / 1200.0)) / (2.0 * np.tanh(2500.0 / 1200.0)),
                "plateau 500 m high, 6 km (azimuth) x 5 km (ground range)"),
}


# ----------------------------------------------------------------------------------------------- targets (x, dy)
def grid(nx, ny, sx, sy, x0=0.0, y0=0.0):
    return [(x0 + (i - (nx - 1) / 2.0) * sx, y0 + (j - (ny - 1) / 2.0) * sy) for i in range(nx) for j in range(ny)]


def diagonal(n, sx, sy, x0=0.0, y0=0.0):
    return [(x0 + (i - (n - 1) / 2.0) * sx, y0 + (i - (n - 1) / 2.0) * sy) for i in range(n)]


LAYOUTS = {  # name -> (ground positions, text)
    "one_centre": ([(0.0, 300.0)], "1 target at x = 0, $\\Delta y$ = 0.3 km"),
    "one_flank": ([(1500.0, 1000.0)], "1 target at x = 1.5 km, $\\Delta y$ = 1 km"),
    "diag5": (diagonal(5, 2300.0, 600.0, 150.0, 80.0), "5 targets on a diagonal, 2.3 km apart in azimuth and 0.6 km in ground range"),
    "grid9": (grid(3, 3, 2300.0, 1200.0, 150.0, 80.0), "9 targets, 3 x 3, 2.3 km apart in azimuth and 1.2 km in ground range"),
    "grid25": (grid(5, 5, 1300.0, 600.0, 150.0, 80.0), "25 targets, 5 x 5, 1.3 km apart in azimuth and 0.6 km in ground range"),
    "grid49": (grid(7, 7, 900.0, 400.0, 150.0, 80.0), "49 targets, 7 x 7, 0.9 km apart in azimuth and 0.4 km in ground range"),
}

FAMILIES = [   # (title in the PDF outline, [(scene id, terrain, layout)])
    ("Ramps", [("ramp_az8_one", "ramp_az8", "one_centre"), ("ramp_az4_grid9", "ramp_az4", "grid9"), ("ramp_rg8_diag5", "ramp_rg8", "diag5"),
               ("ramp_rg-8_grid25", "ramp_rg-8", "grid25"), ("ramp_obl6_grid49", "ramp_obl6", "grid49")]),
    ("Sines and cosines", [("sine_az_grid9", "sine_az", "grid9"), ("eggbox_one", "eggbox", "one_flank"), ("eggbox_grid49", "eggbox", "grid49")]),
    ("Realistic terrains", [("mountain_one", "mountain", "one_centre"), ("mountain_flank", "mountain", "one_flank"), ("mountain_diag5", "mountain", "diag5"),
                            ("mountain_grid49", "mountain", "grid49"), ("hills_grid9", "hills", "grid9"), ("hills_grid49", "hills", "grid49"),
                            ("mountains_diag5", "mountains", "diag5"), ("mountains_grid25", "mountains", "grid25"),
                            ("escarpment_grid9", "escarpment", "grid9"), ("escarpment_grid25", "escarpment", "grid25"),
                            ("plateau_diag5", "plateau", "diag5"), ("plateau_grid49", "plateau", "grid49")]),
]
SCENES = {sid: (fam, terr, lay) for fam, lst in FAMILIES for sid, terr, lay in lst}
FOUR = ["N4b300", "u500", "N4b100", "s50", "N4b500", "u100", "pm50", "s10"]                    # four channels
OTHER = ["N3b300", "N6b100", "N2b500", "N8b300", "N3b100", "N6b500", "N2b300", "N8b100", "N3b500", "N6b300", "N2b100", "N8b500"]


def jobs():
    """(scene, baseline) in the order of the pages: every scene with four channels and with another number of channels."""
    out, i = [], 0
    for _, lst in FAMILIES:
        for sid, _, _ in lst:
            tags = [FOUR[i % len(FOUR)], OTHER[i % len(OTHER)]]
            out += [(sid, t) for t in sorted(tags, key=lambda t: len(baseline(t)[0]))]
            i += 1
    return out


def scene(sid, tag):
    """Parameters, tracks, terrain and texts of one job. The targets sit on the terrain at the positions of the layout."""
    fam, terr, lay = SCENES[sid]
    terrain, tdesc = TERRAINS[terr]
    xy, ldesc = LAYOUTS[lay]
    xy = np.asarray(xy, float)
    hs = np.asarray(terrain(xy[:, 0], xy[:, 1]), float)
    bxt, kind = baseline(tag)
    p = make_params_dpca(Nrx=len(bxt), specs=(), r0_frac=R0_FRAC)
    p.bxt = np.asarray(bxt, float).copy()
    p.extra_offsets = tuple((float(x), float(dy), float(h)) for (x, dy), h in zip(xy, hs))
    return dict(p=p, tr=build_tracks_3d(p), terrain=terrain, label=f"{tdesc}; {ldesc}", kind=kind, family=fam, xy=xy, hs=hs)


# ----------------------------------------------------------------------------------------------- metrics
def envelopes(p, tracks, images):
    """|IRF| along azimuth of every image of the dict (range-compressed, [Na, Nr]): every scatterer is focused with its
    own 2-D matched filter, the azimuth cut through its range row is placed at its along-track position and the
    envelope over the scatterers is kept (reconstruction.azimuth_envelope, with the transforms of the images shared)."""
    ds = p.vs / p.prf
    F = {k: np.fft.fft2(v) for k, v in images.items()}
    env = {k: np.zeros(p.Na) for k in images}
    for ptg in p.points:
        R = np.conj(np.fft.fft2(range_compress(generate_reference_3d(p, tracks, ptgs=ptg[None, :]), p.cd, p.rbw, p.rsf, axis=1)))
        sh = int(round((ptg[0] - p.x0) / ds))
        for k in images:
            row = np.abs(np.fft.fftshift(np.fft.ifft((F[k] * R).mean(axis=1))))        # column Nr // 2 of the focused image
            env[k] = np.maximum(env[k], np.roll(row, sh))
    return env


def ambiguity_orders(p):
    """Ambiguity orders that can hold energy: |m| PRF_op below the Doppler bandwidth (at least +-1, +-2)."""
    B = 4.0 * p.vs * np.sin(p.theta_tx / 2.0) / p.wl
    n = max(2, int(np.floor(B / p.PRF_op)))
    return tuple(m for m in range(-n, n + 1) if m != 0)


def ambiguity_level(p, env, pk, xs, win=0.02, guard=0.02):
    """Largest |IRF| inside +-win s of every ambiguity position x_j / vs + m dt_amb, in dB re. pk, leaving out
    +-guard s around every target's own main lobe (the measure of the 1-D sweep pages)."""
    t = (np.arange(p.Na) - p.Na // 2) / p.prf
    dt_amb = p.PRF_op * p.wl * p.r0 / (2.0 * p.vs ** 2)
    lobe = np.zeros(p.Na, bool)
    sel = np.zeros(p.Na, bool)
    for x in xs:
        lobe |= np.abs(t - x / p.vs) <= guard
        for m in ambiguity_orders(p):
            sel |= np.abs(t - (x / p.vs + m * dt_amb)) <= win
    sel &= ~lobe
    return float(20 * np.log10(env[sel].max() / pk + 1e-15)) if sel.any() else float("nan")


def max_slopes_deg(terrain, xy):
    """Largest slope of the terrain along azimuth and along ground range around the targets (+-1 km) [deg]."""
    lo, hi = xy.min(axis=0) - 1000.0, xy.max(axis=0) + 1000.0
    x, y = np.linspace(lo[0], hi[0], 801), np.linspace(lo[1], hi[1], 401)
    h = terrain(x[:, None], y[None, :])
    return (float(np.degrees(np.arctan(np.abs(np.gradient(h, x, axis=0)).max()))),
            float(np.degrees(np.arctan(np.abs(np.gradient(h, y, axis=1)).max()))), float(np.ptp(h)))


# ----------------------------------------------------------------------------------------------- the stages of one job
def _paths(sid, tag):
    k = f"{sid}__{tag}"
    return dict(num=os.path.join(OUT, "numbers", k + ".json"), data=os.path.join(OUT, "data", k + ".npz"),
                state=os.path.join(WORK, k + "_pass1.npz"), page=os.path.join(OUT, "pages", k + ".png"))


def _done(sid, tag):
    f = _paths(sid, tag)
    return os.path.exists(f["num"]) and os.path.exists(f["data"]) and json.load(open(f["num"])).get("version") == VERSION


def stage_time(tag, stage):
    """Rough duration [s] of one stage on a loaded machine: used to decide whether it still fits in the budget."""
    return (25.0 if stage == 1 else 40.0) * len(baseline(tag)[0]) / 4.0 + 15.0


def run_stage(sid, tag):
    """Advance the job by one stage: stage 1 = first regularised pass of the joint (state saved), stage 2 = second pass,
    the two-step reconstruction without SATA, the envelopes and the numbers. Returns a text line."""
    f = _paths(sid, tag)
    t0 = time.time()
    s = scene(sid, tag)
    p, tr, terrain = s["p"], s["tr"], s["terrain"]
    ch = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)       # the simulated echoes of the point targets
    dC0, hmap = dem_maps(p, tr, terrain, osr=OSR)                                       # the only knowledge of the scene given to the joint
    M = Model2D(p, tr, dC0)
    if not os.path.exists(f["state"]):
        _, info = reconstruct_joint_2d(p, tr, ch, dC0, model=M, passes=1, return_info=True)
        st = info["state"]
        np.savez(f["state"] + ".tmp.npz", sigma=st["sigma"], image=st["image"], residual=st["residual"], iterations=info["iterations"])
        os.replace(f["state"] + ".tmp.npz", f["state"])
        return f"{sid:22s} {tag:8s} stage 1 ({time.time() - t0:.0f} s)"
    z = np.load(f["state"])
    joint, info = reconstruct_joint_2d(p, tr, ch, dC0, model=M, passes=1, return_info=True,
                                       state=dict(sigma=float(z["sigma"]), image=z["image"], residual=z["residual"]))
    t_joint = time.time() - t0
    ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)
    none = reconstruct_subband_2d(p, tr, ch, CoeffTable3D(p, tr, n_nodes=16), use_sata=False, verbose=False)
    # the three curves of the page in the same processed band: azimuth bandwidth abw and, at every pixel, the range
    # frequencies that all the channels observe (the band that the joint delivers)
    lo, hi = info["band"]
    kept = np.clip(hi - lo, 0.0, None) / p.rbw
    env = envelopes(p, tr, dict(ref=processed_band(p, M.to_common_band(ref, lo, hi)), none=M.to_common_band(none, lo, hi), joint=joint,
                                ref_whole=ref, none_whole=none, joint_whole=image_to_data(M, info["image"])))
    pk = env["ref"].max()
    xs = p.points[:, 0]
    amb = {k: ambiguity_level(p, env[k], pk, xs) for k in ("ref", "none", "joint")}
    # for the record: the whole range band, against the reference as the package uses it (not cut in azimuth)
    amb.update({k: ambiguity_level(p, env[k], env["ref_whole"].max(), xs) for k in ("ref_whole", "none_whole", "joint_whole")})
    sa, sr, dh = max_slopes_deg(terrain, s["xy"])
    res = dict(amb, scene=sid, baseline=tag, family=s["family"], nrx=p.Nrx, bxt=[float(b) for b in p.bxt], kind=s["kind"], label=s["label"],
               n_targets=int(len(xs)), max_slope_azimuth_deg=sa, max_slope_range_deg=sr, dh=dh, orders=list(ambiguity_orders(p)),
               iterations=int(z["iterations"]) + info["iterations"], unexplained_dB=float(20 * np.log10(info["unexplained"])),
               band_kept_median=float(np.median(kept)), band_kept_min=float(kept.min()),
               seconds_stage2=time.time() - t0, version=VERSION)
    # what the page needs: envelopes, the DEM table of the channel with the largest baseline (every 4th azimuth row)
    i1 = int(np.argmax(np.abs(p.bxt)))
    np.savez_compressed(f["data"], **{f"env_{k}": v.astype(np.float32) for k, v in env.items()}, pk=pk,
                        dC0=dC0[i1, ::4].astype(np.float32), hmap=hmap[::4].astype(np.float32), channel=i1)
    json.dump(res, open(f["num"], "w"), indent=1)                                       # written last: marks the job as done
    os.remove(f["state"])
    return (f"{sid:22s} {tag:8s} ref {amb['ref']:6.1f} none {amb['none']:6.1f} joint {amb['joint']:6.1f} dB | unexplained "
            f"{res['unexplained_dB']:5.1f} dB ({t_joint:.0f} + {time.time() - t0 - t_joint:.0f} s)")


def _work(a):
    sid, tag, deadline = a
    lines = []
    while not _done(sid, tag):
        stage = 2 if os.path.exists(_paths(sid, tag)["state"]) else 1
        if time.time() + stage_time(tag, stage) > deadline:
            break
        lines.append(run_stage(sid, tag))
    return lines


# ----------------------------------------------------------------------------------------------- page
def channel_geometry(ax, p):
    """Transmitter and receivers seen from above, in the frame of the transmitter."""
    bat, bxt, n = p.bat, p.bxt, p.Nrx
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
        covered = np.any((xr < x0 + 0.5 * (x1 - x0)) & (yr > y1 - (0.14 + 0.07 * n) * (y1 - y0)))
        ax.legend(fontsize=7.6, loc="lower left" if covered else "upper left", bbox_to_anchor=None if covered else (0.0, 0.92), framealpha=0.92)


def page(sid, tag):
    """The six-panel page of one job, from its numbers and data files."""
    f = _paths(sid, tag)
    res, z = json.load(open(f["num"])), np.load(f["data"])
    s = scene(sid, tag)
    p, terrain, xy, hs = s["p"], s["terrain"], s["xy"], s["hs"]
    pk = float(z["pk"])
    db = {k: 20 * np.log10(z[f"env_{k}"] / pk + 1e-12) for k in ("ref", "none", "joint")}
    t = (np.arange(p.Na) - p.Na // 2) / p.prf
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    fig = plt.figure(figsize=(22, 10.6))
    gs = GridSpec(2, 3, width_ratios=[1.32, 0.86, 1.0], hspace=0.38, wspace=0.2, figure=fig, left=0.04, right=0.985, top=0.86, bottom=0.115)
    lab = res["label"] if res["label"].startswith("h =") else res["label"][0].upper() + res["label"][1:]
    fig.suptitle(f"{lab}\n$b_{{xt}}$ = {bxt_text(p.bxt)} ({res['kind']}) | the joint reads only the DEM (look-up table from the terrain); no method is given the positions of the targets",
                 fontsize=12.5, y=0.985)
    # ---- azimuth IRFs
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["joint"], color="tab:purple", lw=1.3, label="joint, final version in 2-D (DEM applied at every pixel of the image)")
    ax.set_title(f"highest ambiguity: reference {res['ref']:.1f} dB | joint {res['joint']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"highest ambiguity: reference {res['ref']:.1f} dB | no SATA {res['none']:.1f} dB", fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")
    # ---- the scene from above, zoom on the targets
    g2 = fig.add_subplot(gs[0, 1])
    lo = np.minimum(xy.min(axis=0), 0.0); hi = np.maximum(xy.max(axis=0), 0.0)
    pad = np.maximum(0.25 * (hi - lo), [1500.0, 1000.0])
    xg, yg = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 241), np.linspace(lo[1] - pad[1], hi[1] + pad[1], 161)
    hg = terrain(xg[:, None], yg[None, :])
    vmax = max(float(hg.max()), float(hs.max()), 1.0)
    cf = g2.contourf(xg, yg, hg.T, levels=np.linspace(min(0.0, float(hg.min())), vmax, 21), cmap="terrain", alpha=0.85)
    cb = fig.colorbar(cf, ax=g2, pad=0.02, fraction=0.05); cb.set_label("terrain height [m]", fontsize=8); cb.ax.tick_params(labelsize=7)
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
    g2.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.1)
    g2.set_title(f"Scene from above: zoom on the targets | $\\Delta h_{{max}}$ = {res['dh']:.0f} m | max slope {res['max_slope_azimuth_deg']:.0f}$^\\circ$ "
                 f"along azimuth, {res['max_slope_range_deg']:.0f}$^\\circ$ along ground range", fontsize=9.5, pad=30)
    g2.xaxis.set_major_formatter(km); g2.yaxis.set_major_formatter(km)
    g2.set_xlabel("azimuth position $x$ [km]"); g2.set_ylabel("ground range $\\Delta y$ from the reference point [km]")
    # ---- the scene in 3D, the whole imaged area
    g3 = fig.add_subplot(gs[0, 2], projection="3d")
    half = 0.5 * p.Na / p.prf * p.vs
    _, dyr = dem_heights(p, terrain, x=np.array([0.0]), r=p.r_scan[[0, -1]], osr=OSR)
    ylim = (min(float(dyr.min()), lo[1] - pad[1]), max(float(dyr.max()), hi[1] + pad[1]))
    X3, Y3 = np.meshgrid(np.linspace(-half, half, 121), np.linspace(ylim[0], ylim[1], 61), indexing="ij")
    H3 = terrain(X3, Y3)
    g3.plot_surface(X3, Y3, H3, cmap="terrain", vmin=min(0.0, float(hg.min())), vmax=vmax, alpha=0.75, linewidth=0, antialiased=True, rstride=1, cstride=1)
    g3.scatter(xy[:, 0], xy[:, 1], hs + 0.02 * vmax, s=16, color="#1F4E79", edgecolor="w", linewidths=0.4, depthshade=False, zorder=10)
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=11)
    g3.set_xlim(-half, half); g3.set_ylim(*ylim); g3.set_zlim(min(0.0, float(H3.min())), max(float(H3.max()), 50.0) * 1.6)
    g3.xaxis.set_major_formatter(km); g3.yaxis.set_major_formatter(km)
    g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4); g3.set_ylabel("ground range $\\Delta y$ [km]", fontsize=8, labelpad=4)
    g3.set_zlabel("height [m]", fontsize=8, labelpad=2); g3.tick_params(labelsize=7, pad=0)
    g3.view_init(elev=28, azim=-60); g3.set_box_aspect((1.7, 1.0, 0.55), zoom=1.25)
    g3.locator_params(axis="x", nbins=6); g3.locator_params(axis="y", nbins=4); g3.locator_params(axis="z", nbins=4)
    g3.set_title(f"Scene in 3D: the whole imaged area ({2 * half / 1e3:.0f} km of azimuth x {np.ptp(ylim) / 1e3:.0f} km of ground range)", fontsize=10)
    # ---- channels
    channel_geometry(fig.add_subplot(gs[1, 1]), p)
    # ---- the DEM table of the channel with the largest baseline, in radar geometry
    am = fig.add_subplot(gs[1, 2])
    i1 = int(z["channel"])
    xi, ri = image_positions(p, OSR)
    D = 1e3 * z["dC0"]
    vm = max(float(np.abs(D).max()), 1e-3)
    im = am.pcolormesh(xi[::4], (ri - p.r0), D.T, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto", rasterized=True)
    cb = fig.colorbar(im, ax=am, pad=0.02, fraction=0.05); cb.ax.tick_params(labelsize=7)
    rt = np.sqrt((p.y0 + xy[:, 1]) ** 2 + (p.H - hs) ** 2) - p.r0
    am.plot(xy[:, 0], rt, "o", ms=4.5, mfc="none", mec="k", mew=0.9, ls="none", label="where the point targets are (not given to any method)")
    am.set_xlim(xi[0], xi[-1]); am.xaxis.set_major_formatter(km)
    am.set_xlabel("azimuth position $x$ [km]"); am.set_ylabel("slant range relative to the reference point [m]")
    am.set_title(f"$\\Delta C_0$ [mm] in the DEM table of channel {i1} ($b_{{xt}}$ = {p.bxt[i1]:g} m): azimuth x slant range", fontsize=10)
    am.legend(fontsize=7.8, loc="upper center", framealpha=0.9)
    nmax = max(res["orders"])
    rho = C0_LIGHT / (2.0 * p.rsf)
    fig.text(0.5, 0.012, "Azimuth IRF: every target focused with its own 2-D matched filter, envelope over the targets. Highest ambiguity in dB relative to "
             f"the reference peak, inside +-20 ms of the ambiguity positions of every target (up to {nmax} times the ambiguity distance of the channel "
             "PRF, on both sides).\nThe three curves are in the same processed band: azimuth bandwidth $B_a$ and, at every pixel, the range frequencies "
             f"that all the channels observe there ({100 * res['band_kept_median']:.0f} % of the range band at the median pixel, "
             f"{100 * res['band_kept_min']:.0f} % at the worst one).\n"
             f"$N_{{rx}}$ = {p.Nrx}, PRF = {p.prf:.0f} Hz, $B_a$ = {p.abw:.0f} Hz, range resolution {C0_LIGHT / (2 * p.rbw):.0f} m (range bin {rho:.0f} m), "
             f"channel 0 is the transmitter ($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {bxt_text(p.bxt)}.", ha="center", fontsize=8.5, color="0.35")
    fig.savefig(f["page"], dpi=90); plt.close(fig)
    return f["page"]


# ----------------------------------------------------------------------------------------------- driver
def load():
    return {(sid, tag): json.load(open(_paths(sid, tag)["num"])) for sid, tag in jobs() if _done(sid, tag)}


def table():
    res = load()
    json.dump([res[j] for j in jobs() if j in res], open(os.path.join(OUT, "all_numbers.json"), "w"), indent=1)   # all the runs in one file
    for fam, lst in FAMILIES:
        print(fam)
        for sid, tag in jobs():
            r = res.get((sid, tag))
            if r and SCENES[sid][0] == fam:
                print(f"  {sid:22s} {tag:8s} Nrx {r['nrx']} targets {r['n_targets']:2d} | ref {r['ref']:6.1f} none {r['none']:6.1f} joint {r['joint']:6.1f} | "
                      f"unexplained {r['unexplained_dB']:6.1f} dB")


def pdf():
    """Multi-page PDF (JPEG pages) with bookmarks per family."""
    from PIL import Image
    res = load()
    pngs, outline = [], []
    for fam, _ in FAMILIES:
        first = True
        for sid, tag in jobs():
            pp = _paths(sid, tag)["page"]
            if SCENES[sid][0] != fam or (sid, tag) not in res or not os.path.exists(pp):
                continue
            if first:
                outline.append((fam, len(pngs), 0)); first = False
            r = res[(sid, tag)]
            lab = r["label"].replace("$\\pi$", "pi").replace("$^\\circ$", " deg").replace("$\\Delta y$", "dy")
            outline.append((f"{lab} | {r['nrx']} channels, bxt {bxt_text(r['bxt'])}", len(pngs), 1))
            pngs.append(pp)
    out = os.path.join(OUT, "sweep_joint2d.pdf")
    ims = [Image.open(q).convert("RGB") for q in pngs]
    tmp = os.path.join(tempfile.gettempdir(), "sweep_joint2d_pages.pdf")               # outside the output folder
    ims[0].save(tmp, save_all=True, append_images=ims[1:], resolution=90, quality=75)
    try:
        from pypdf import PdfReader, PdfWriter
        w = PdfWriter(clone_from=PdfReader(tmp))
        parent = None
        for title, k, lvl in outline:
            it = w.add_outline_item(title, k, parent=parent if lvl else None)
            if lvl == 0:
                parent = it
        w.write(out)
        os.remove(tmp)
    except ImportError:
        import shutil
        shutil.copyfile(tmp, out)
    return f"{out} ({len(pngs)} pages)"


def main():
    global OUT, WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, help="scene ids (default: all)")
    ap.add_argument("--budget", type=float, default=1e9, help="seconds: no stage is started that would not fit")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--work", default=WORK, help="folder for the state between the two stages of a job")
    ap.add_argument("--pages", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--list", action="store_true", help="the jobs, without running anything")
    a = ap.parse_args()
    OUT, WORK = a.out, a.work
    sel = [(sid, tag) for sid, tag in jobs() if a.only is None or sid in a.only]
    if a.list:
        for sid, tag in sel:
            print(f"{sid:22s} {tag:8s} bxt {bxt_text(baseline(tag)[0])} | {TERRAINS[SCENES[sid][1]][1]}; {LAYOUTS[SCENES[sid][2]][1]}")
        return print(len(sel), "jobs")
    if a.table:
        return table()
    if a.pdf:
        return print("wrote", pdf())
    for sub in ("numbers", "data", "pages"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    deadline = time.time() + a.budget
    if a.pages:
        todo = [(sid, tag) for sid, tag in sel if _done(sid, tag) and not os.path.exists(_paths(sid, tag)["page"])]
        print(len(todo), "pages to do", flush=True)
        fn, args = _page_job, [(sid, tag, deadline) for sid, tag in todo]
    else:
        todo = [(sid, tag) for sid, tag in sel if not _done(sid, tag)]
        todo.sort(key=lambda j: not os.path.exists(_paths(*j)["state"]))               # the jobs already at stage 2 first
        print(len(todo), "jobs to do", flush=True)
        fn, args = _work, [(sid, tag, deadline) for sid, tag in todo]
    if a.workers > 1 and len(args) > 1:
        from multiprocessing import Pool
        with Pool(a.workers, initializer=_set_out, initargs=(OUT, WORK)) as pool:
            for lines in pool.imap_unordered(fn, args):
                for line in lines:
                    print(line, flush=True)
    else:
        for x in args:
            for line in fn(x):
                print(line, flush=True)
    print("left:", sum(not _done(sid, tag) for sid, tag in sel), "jobs,", sum(_done(sid, tag) and not os.path.exists(_paths(sid, tag)["page"]) for sid, tag in sel),
          "pages", flush=True)


def _set_out(out, work):
    global OUT, WORK
    OUT, WORK = out, work


def _page_job(a):
    sid, tag, deadline = a
    if time.time() + 15.0 > deadline:
        return []
    return [page(sid, tag)]


if __name__ == "__main__":
    main()
