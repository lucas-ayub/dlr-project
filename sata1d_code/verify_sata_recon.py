"""
SATA 1D with the DEM, together with the azimuth reconstruction: is the implementation correct?

The package (sata1d) is compared with an INDEPENDENT model of what SATA does, written here from the theory only:

    ideal SATA   every target, every channel, the full-rate echo is corrected in the Doppler domain, frequency by
                 frequency, with the phase exp(+j 2 pi dC0_i / wl) read on the DEM at the ground position that the
                 kernel assigns to that frequency:  x_label = x_target + R(f_label) - R(f),
                 R(f) = r0 tan(asin(wl f / 2 v)),  f_label = f wrapped into the window of the kernel
                 (whole band: [-P/2, P/2); output sub-band k: [f_k - P/2, f_k + P/2)).
                 Then decimated to the channel PRF and reconstructed exactly like the package does.
                 This is SATA with an infinitely fine sub-aperture resolution and the exact DEM.
    oracle       the same, with x_label = x_target for every frequency (each target with its own dC0).

dC0_i(h) comes from a height table made here with residual_C0 (the 1-D geometry is invariant along azimuth), not from
terrain_lookup_table. The cells of the table (bat/2 shift) are tested directly (cell_checks): a shift of a few cells
changes the results too little to be seen in the reconstructions.

Checks (PASS / FAIL in sata_recon/checks.json):
  M  the model of this script: sum of the per-target echoes = generate_channels; ideal with a flat DEM = standard
     reconstruction; ideal with x_label = x_target = oracle_channels of the package
  Z  zero DEM: SATA whole band and per sub-band = standard reconstruction (Nrx = 2 ... 8)
  E  scenes where theory says SATA before the reconstruction is EXACT (constant height; terrain periodic with the
     fold of that Nrx): package = oracle
  P  every scene: package SATA = ideal SATA (the implementation does what the method says), and how far both are from
     the oracle (what the method itself leaves)
  X  the same comparison with the package broken on purpose (ruler sign, labels of the sub-bands, phase sign, DEM moved
     by 1 km): the test must see it

    python verify_sata_recon.py --list
    python verify_sata_recon.py --jobs 0-29 --budget 160      # resumable, one JSON per run in sata_recon/runs
    python verify_sata_recon.py --checks                      # M, Z, X and the PASS / FAIL of E and P from the runs
    python verify_sata_recon.py --pdf
"""
import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np

import sata1d as S
import sata1d.sata as SS
from sata1d.config import ArrayGeometry
from sata1d.reconstruction import reconstruct_from_spectra, channel_spectrum
from sata1d.subband import sata_channels_subband, subband_frequency_beam, subband_axis

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sata_recon")
WL, OSF = 0.25, 4


# ================================================================================================ scenes
def fold(nrx, prf=2000.0):
    """One fold of the channel PRF on the ground [m]: R(P) = r0 tan(asin(wl P / 2 v))."""
    cfg = S.make_config([(0.0, 0.0)], nrx=nrx, bxt_max=0.0)
    v, r0, P = cfg.system.vs, cfg.scene.r0, prf / nrx
    return float(r0 * np.tan(np.arcsin(WL * P / (2 * v))))


def gauss(x, h, s):
    return h * np.exp(-0.5 * (np.asarray(x, float) / s) ** 2)


def hills(x):
    x = np.asarray(x, float)
    return 150 + 60 * np.sin(2 * np.pi * x / 3700 + 0.4) + 50 * np.sin(2 * np.pi * x / 2300 + 1.9) + 40 * np.sin(2 * np.pi * x / 9100 + 2.7)


SCENES = {  # name -> (terrain h(x) or None for the fold-periodic terrain, targets, description)
    "flat500": (lambda x: 0 * np.asarray(x, float) + 500.0, np.linspace(-5200, 5200, 5), "flat terrain at h = 500 m (constant DEM)"),
    "periodic": (None, np.linspace(-5200, 5200, 5), "h = 40 (1 - cos(2 pi x / X)), X = one fold of the channel PRF"),
    "ramp1": (lambda x: np.tan(np.radians(1.0)) * (np.asarray(x, float) + 9000), np.linspace(-5200, 5200, 5), "ramp of 1 deg along azimuth"),
    "ramp3": (lambda x: np.tan(np.radians(3.0)) * (np.asarray(x, float) + 9000), np.linspace(-5200, 5200, 5), "ramp of 3 deg along azimuth"),
    "ramp8": (lambda x: np.tan(np.radians(8.0)) * (np.asarray(x, float) + 9000), np.linspace(-5200, 5200, 5), "ramp of 8 deg along azimuth"),
    "sine": (lambda x: 100 + 100 * np.sin(2 * np.pi * np.asarray(x, float) / 20e3), np.linspace(-5200, 5200, 5), "h = 100 + 100 sin(2 pi x / 20 km)"),
    "mountain": (lambda x: gauss(x, 1000.0, 1500.0), np.linspace(-5200, 5200, 5), "isolated mountain 1000 m (Gaussian, sigma 1.5 km)"),
    "mountain1": (lambda x: gauss(x, 1000.0, 1500.0), np.array([1300.0]), "isolated mountain 1000 m, one target on the flank"),
    "hills25": (hills, np.linspace(-4800, 4800, 25), "hills 50 - 300 m, 25 targets every 400 m"),
    "pairX": (lambda x: np.tan(np.radians(1.0)) * (np.asarray(x, float) + 9000), lambda n: np.array([-0.5, 0.5]) * fold(n),
              "ramp of 1 deg, two targets exactly one fold X apart"),
    "pairX2": (lambda x: np.tan(np.radians(1.0)) * (np.asarray(x, float) + 9000), lambda n: np.array([-0.25, 0.25]) * fold(n),
               "ramp of 1 deg, two targets half a fold apart"),
}


def targets(scene, nrx):
    t = SCENES[scene][1]
    return np.asarray(t(nrx) if callable(t) else t, float)
NRX = [2, 3, 4, 6, 8]
BMAX = [100.0, 500.0]


def jobs():
    return [(s, n, b) for s in SCENES for n in NRX for b in BMAX]


def baselines(n, bmax, seed=0):
    """Channel 0 on the transmitter; the others random, the largest exactly bmax (as in the sweeps)."""
    u = np.random.default_rng(seed).uniform(0.1, 1.0, n)
    u[0] = 0.0
    return (bmax * u / u.max()).round(1)


def terrain_of(scene, nrx):
    t = SCENES[scene][0]
    if t is None:
        X = fold(nrx)
        return lambda x: 40.0 * (1.0 - np.cos(2 * np.pi * np.asarray(x, float) / X))
    return t


def setup(scene, nrx, bmax):
    terrain = terrain_of(scene, nrx)
    xs = targets(scene, nrx)
    amps = np.exp(2j * np.pi * np.random.default_rng(7).uniform(size=len(xs)))
    cfg = S.make_config([(float(x), float(terrain(x))) for x in xs], nrx=nrx, bxt_max=0.0, amplitudes=amps)
    bat = cfg.array.bat
    cfg = replace(cfg, array=ArrayGeometry(bat=bat, bxt=baselines(nrx, bmax)))
    return cfg, S.build_platform_tracks(cfg), terrain


# ================================================================================================ the independent model
def height_table(cfg, tr, hmin, hmax, n=25):
    """dC0_i as a function of the height (the 1-D geometry does not change along azimuth): residual_C0 of a point at
    x = 0 on the iso-range surface, polynomial of degree 4 in h. Returns a list of callables g_i(h)."""
    H, r0, y0 = cfg.scene.H, cfg.scene.r0, cfg.scene.y0
    hg = np.linspace(min(hmin, 0.0) - 1.0, max(hmax, 0.0) + 1.0, n)
    out = []
    for i in range(cfg.Nrx):
        if cfg.array.bxt[i] == 0 and cfg.array.bat[i] == 0:
            out.append(lambda h: 0.0 * np.asarray(h, float))
            continue
        v = [SS.residual_C0(cfg, tr, cfg.scene.ptg + np.array([0.0, np.sqrt(r0 ** 2 - (H - h) ** 2) - y0, h]), i) for h in hg]
        c = np.polyfit(hg, v, 4)
        out.append(lambda h, c=c: np.polyval(c, np.asarray(h, float)))
    return out


def ruler(cfg, f):
    """Ground offset [m] of the Doppler frequency f: R(f) = r0 tan(asin(wl f / 2 v))."""
    return cfg.scene.r0 * np.tan(np.arcsin(np.clip(WL * f / (2 * cfg.system.vs), -1, 1)))


def wrap(f, lo, P):
    return lo + np.mod(f - lo, P)


def model_channels(cfg, tr, terrain, g):
    """Per-target, per-channel full-rate echoes corrected in the Doppler domain. Returns dict of channel sets
    [Nrx, Na_ch]: 'raw', 'oracle', 'flat' (ideal with a zero DEM), 'whole' (ideal, whole-band labels),
    'sub' (list over k: ideal, labels of output sub-band k)."""
    N, P = cfg.Nrx, cfg.PRF_op
    f = np.fft.fftfreq(cfg.Na, 1.0 / cfg.prf)
    labels = {"whole": wrap(f, -P / 2, P), "flat": None, "oracle": f}
    for k in range(cfg.Nsb):
        # reconstruct_from_spectra: output block k holds fa = -N P / 2 + k P + n P / Na_ch, n = 0 .. Na_ch - 1
        f_lo = -cfg.Nsb * P / 2.0 + k * P
        labels[f"sub{k}"] = wrap(f, f_lo, P)
    out = {k: np.zeros((N, cfg.Na_ch), complex) for k in list(labels) + ["raw"]}
    pts = cfg.scene.points[1:]
    for j, (p, a) in enumerate(zip(pts, cfg.scene.amps)):
        xj = p[0] - cfg.scene.x0                               # ground position relative to the reference point
        for i in range(N):
            s = S.signal_model.raw(cfg, tr, [p], tr.prx[i], tr.vrx[i], [a])
            out["raw"][i] += s[::N]
            F = np.fft.fft(s)
            for name, lab in labels.items():
                if name == "flat":
                    out[name][i] += s[::N]
                    continue
                xl = xj + ruler(cfg, lab) - ruler(cfg, f)
                out[name][i] += np.fft.ifft(F * np.exp(2j * np.pi * g[i](terrain(xl)) / WL))[::N]
    out["sub"] = [out.pop(f"sub{k}") for k in range(cfg.Nsb)]
    return out


def recon_whole(cfg, tr, ch):
    return S.reconstruct(cfg, tr, ch)


def recon_sub(cfg, tr, chs):
    """chs: list over output sub-bands k of channel sets [Nrx, Na_ch]."""
    spec = [[channel_spectrum(cfg, chs[k][j]) for j in range(cfg.Nrx)] for k in range(cfg.Nsb)]
    return reconstruct_from_spectra(cfg, tr, spec, True)


# ================================================================================================ the package
def package(cfg, tr, ch, maps, mutate=None, scheme="cola"):
    """SATA whole band and per sub-band of the package with the DEM tables `maps` (one per channel).
    mutate: None or one of 'ruler_sign', 'whole_labels_in_sub', 'phase_sign' (the package broken on purpose)."""
    kern = dict(rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=WL, r=cfg.scene.r0)
    Tsubeff, hop, Nzp = SS.subaperture_sizing(cfg.scene.r0, cfg.PRF_op, cfg.system.vs, WL, OSF)
    inv = mutate != "phase_sign"
    sgn = -1.0 if mutate == "ruler_sign" else 1.0
    _, az_w = SS.subaperture_axis(cfg.PRF_op, Nzp, cfg.system.vs, WL, cfg.scene.r0)
    whole = np.array([SS.sata_kernel(ch[i], maps[i], Tsubeff, hop, Nzp, sgn * az_w, WL, inv, scheme) for i in range(cfg.Nrx)])
    if mutate is None:                                         # the public entry point gives the same
        ref = np.array([SS.sata_1d(ch[i], maps[i], inverse=True, sata_osf=OSF, scheme=scheme, **kern) for i in range(cfg.Nrx)])
        assert np.allclose(ref, whole, atol=1e-12)
    subs = []
    for k in range(cfg.Nsb):
        if mutate is None:
            subs.append(sata_channels_subband(cfg, tr, ch, k, sata_osf=OSF, maps=list(maps), scheme=scheme))
            continue
        f_k = subband_frequency_beam(cfg, k)[0]
        if mutate == "sub_windows_reversed":
            f_k = subband_frequency_beam(cfg, cfg.Nsb - 1 - k)[0]
        az = az_w if mutate == "whole_labels_in_sub" else subband_axis(cfg.PRF_op, Nzp, cfg.system.vs, WL, cfg.scene.r0, f_k)[1]
        subs.append(np.array([SS.sata_kernel(ch[i], maps[i], Tsubeff, hop, Nzp, sgn * az, WL, inv, "cola") for i in range(cfg.Nrx)]))
    return recon_whole(cfg, tr, whole), recon_sub(cfg, tr, subs)


# ================================================================================================ metrics
def focuser(cfg, tr):
    Hf = np.conj(np.fft.fft(S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])))
    return lambda s: np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2)


def orders(cfg):
    B = 4.0 * cfg.system.vs * np.sin(cfg.theta_tx / 2.0) / WL
    n = max(2, int(np.floor(B / cfg.PRF_op)))
    return [m for m in range(-n, n + 1) if m]


def amb_level(cfg, env, pk, xs, win=0.02, guard=0.02):
    """Largest |IRF| within +-20 ms of the ambiguity positions of every target (orders |m| PRF_op < B, at least +-2),
    leaving out +-20 ms around every target, in dB re. the peak of the reference (the measure of the sweeps)."""
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    v = cfg.system.vs
    dt = cfg.PRF_op * WL * cfg.scene.r0 / (2 * v ** 2)
    lobe = np.zeros(cfg.Na, bool)
    sel = np.zeros(cfg.Na, bool)
    for x in xs:
        lobe |= np.abs(t - x / v) <= guard
        for m in orders(cfg):
            sel |= np.abs(t - (x / v + m * dt)) <= win
    sel &= ~lobe
    return float(20 * np.log10(env[sel].max() / pk + 1e-15))


def rel(a, b):
    return float(20 * np.log10(np.linalg.norm(a - b) / np.linalg.norm(b) + 1e-300))


# ================================================================================================ one run
def run(scene, nrx, bmax, mutations=False, step=4):
    t0 = time.time()
    cfg, tr, terrain = setup(scene, nrx, bmax)
    xs = targets(scene, nrx)
    hs = terrain(np.linspace(-30e3, 30e3, 6001))
    g = height_table(cfg, tr, float(hs.min()), float(hs.max()))
    m = model_channels(cfg, tr, terrain, g)
    ch = S.generate_channels(cfg, tr)
    maps = [S.terrain_lookup_table(cfg, tr, i, terrain, step=step) for i in range(cfg.Nrx)]
    foc = focuser(cfg, tr)
    ref = S.generate_reference(cfg, tr)
    I = dict(ref=foc(ref), none=foc(S.reconstruct(cfg, tr, ch)), oracle=foc(recon_whole(cfg, tr, m["oracle"])),
             ideal_whole=foc(recon_whole(cfg, tr, m["whole"])), ideal_sub=foc(recon_sub(cfg, tr, m["sub"])),
             flat_model=foc(recon_whole(cfg, tr, m["flat"])))
    pw, ps = package(cfg, tr, ch, maps)
    I["pkg_whole"], I["pkg_sub"] = foc(pw), foc(ps)
    pw, ps = package(cfg, tr, ch, maps, scheme="margin")         # the scheme closest to the reference routine
    I["margin_whole"], I["margin_sub"] = foc(pw), foc(ps)
    oracle_pkg = foc(S.reconstruct(cfg, tr, __import__("run_one_case").oracle_channels(cfg, tr, S.delta_C0_per_target(cfg, tr))))
    pk = np.abs(I["ref"]).max()
    lev = {k: amb_level(cfg, np.abs(v), pk, xs) for k, v in I.items()}
    dC0 = S.delta_C0_per_target(cfg, tr)
    res = dict(scene=scene, nrx=nrx, bmax=bmax, bxt=[float(b) for b in cfg.array.bxt], desc=SCENES[scene][2], n_targets=len(xs),
               levels=lev,
               # the model of this script
               model_raw_vs_generate=rel(m["raw"], ch), model_flat_vs_none=rel(I["flat_model"], I["none"]),
               model_oracle_vs_package_oracle=rel(I["oracle"], oracle_pkg),
               table_vs_targets_mm=float(1e3 * np.max(np.abs(np.array([[g[i](h) for i in range(cfg.Nrx)] for h in terrain(xs)]) - dC0))),
               # the implementation against the method, and the method against the oracle
               pkg_vs_ideal_whole=rel(I["pkg_whole"], I["ideal_whole"]), pkg_vs_ideal_sub=rel(I["pkg_sub"], I["ideal_sub"]),
               margin_vs_ideal_whole=rel(I["margin_whole"], I["ideal_whole"]), margin_vs_ideal_sub=rel(I["margin_sub"], I["ideal_sub"]),
               err={k: rel(I[k], I["oracle"]) for k in ("none", "pkg_whole", "pkg_sub", "ideal_whole", "ideal_sub", "margin_whole", "margin_sub")},
               seconds=0.0)
    # complex differences inside the ambiguity windows [dB re. the reference peak]
    A = lambda a, b: amb_level(cfg, np.abs(a - b), pk, xs)
    res["amb_diff"] = {f"pkg_vs_ideal_{k}": A(I["pkg_" + k], I["ideal_" + k]) for k in ("whole", "sub")}
    res["amb_diff"].update({f"margin_vs_ideal_{k}": A(I["margin_" + k], I["ideal_" + k]) for k in ("whole", "sub")})
    res["amb_diff"].update({f"ideal_vs_oracle_{k}": A(I["ideal_" + k], I["oracle"]) for k in ("whole", "sub")})
    # per-channel SATA with maps made from the TARGET POSITIONS (mode="footprint": not allowed in the method, only to
    # show what a per-channel correction can do at best)
    fw = S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=OSF, mode="footprint", dC0=dC0))
    fs = S.reconstruct_subband(cfg, tr, ch, sata_osf=OSF, mode="footprint", dC0=dC0)
    res["levels"]["footprint_whole"] = amb_level(cfg, np.abs(foc(fw)), pk, xs)
    res["levels"]["footprint_sub"] = amb_level(cfg, np.abs(foc(fs)), pk, xs)
    # largest complex difference to the oracle over the WHOLE image, peaks included [dB re. the reference peak]:
    # with two targets one fold apart the leakage of one lands on the peak of the other, where amb_level does not look
    D = lambda a: float(20 * np.log10(np.abs(a - I["oracle"]).max() / pk + 1e-15))
    res["full_diff"] = dict(footprint_whole=D(foc(fw)), footprint_sub=D(foc(fs)),
                            pkg_whole=D(I["pkg_whole"]), pkg_sub=D(I["pkg_sub"]), ref=D(I["ref"]))
    if mutations:
        res["mutations"] = {}
        variants = {mu: package(cfg, tr, ch, maps, mutate=mu) for mu in ("ruler_sign", "whole_labels_in_sub", "sub_windows_reversed", "phase_sign")}
        for n_cells, name in ((2, "dem_moved_31m"), (8, "dem_moved_123m"), (int(round(1000.0 / (cfg.system.vs / cfg.PRF_op))), "dem_moved_1km")):
            variants[name] = package(cfg, tr, ch, [np.roll(M, n_cells) for M in maps])
        for mu, (a, b) in variants.items():
            a, b = foc(a), foc(b)
            res["mutations"][mu] = dict(diff_whole=A(a, I["ideal_whole"]), diff_sub=A(b, I["ideal_sub"]),
                                        level_whole=amb_level(cfg, np.abs(a), pk, xs), level_sub=amb_level(cfg, np.abs(b), pk, xs))
    ds = cfg.system.vs / cfg.PRF_op
    xg = np.linspace(-25000, 25000, 5001)                     # the kernels read the table up to (N - 1) folds away
    grad = max(float(np.max(np.abs(np.gradient(g[i](terrain(xg)), xg)))) for i in range(cfg.Nrx))
    res["max_rad_per_sata_cell"] = float(2 * np.pi * grad * np.sqrt(WL * cfg.scene.r0 / 2) / WL)   # phase change of the DEM correction over 310 m
    res["seconds"] = time.time() - t0
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    curves = {k: (20 * np.log10(np.abs(v) / pk + 1e-12)).astype(np.float32) for k, v in I.items() if k != "flat_model"}
    return res, dict(t=t.astype(np.float32), **curves)


# ================================================================================================ cells
def cell_checks():
    """Cell of a target in the line of every channel: closest approach of the bistatic range history (Tx + Rx_i)
    on the full-rate time axis, divided by Nrx, against az_pixel_of_scatterer (which the DEM table uses)."""
    out = []
    for nrx in (2, 4, 8):
        cfg, tr, _ = setup("ramp1", nrx, 500.0)
        for dx in (-5200.0, 0.0, 3100.0):
            p = cfg.scene.ptg + np.array([dx, 0.0, 0.0])
            for i in range(nrx):
                rr = np.linalg.norm(tr.ptx - p, axis=1) + np.linalg.norm(tr.prx[i] - p, axis=1)
                k = int(np.argmin(rr))
                a, b, c = rr[k - 1], rr[k], rr[k + 1]
                t_min = k + 0.5 * (a - c) / (a - 2 * b + c)            # sub-sample minimum (parabola)
                cell_true = t_min / nrx                                # channel i is sampled at ta[::nrx]
                ds = cfg.system.vs / cfg.PRF_op
                formula = cfg.Na_ch / 2 + (cfg.scene.x0 + dx + 0.5 * cfg.array.bat[i]) / ds     # the formula of az_pixel_of_scatterer, before rounding
                c_tab = int(round(formula))                                                     # the cell whose x terrain_lookup_table evaluates:
                x_tab = (c_tab - cfg.Na_ch / 2) * ds - 0.5 * cfg.array.bat[i] - cfg.scene.x0    # x of that cell in the table
                out.append(dict(nrx=nrx, dx=dx, channel=i, err_cells=float(SS.az_pixel_of_scatterer(cfg, dx, i) - cell_true),
                                formula_err_cells=float(formula - cell_true), table_x_err_m=float(x_tab - dx - (c_tab - formula) * ds)))
    return out


# ================================================================================================ zero DEM
def zero_checks():
    out = []
    for nrx in (2, 3, 4, 5, 6, 8):
        cfg, tr, _ = setup("mountain", nrx, 300.0)
        ch = S.generate_channels(cfg, tr)
        none = S.reconstruct(cfg, tr, ch)
        a, b = package(cfg, tr, ch, [np.zeros(cfg.Na_ch)] * cfg.Nrx)
        out.append(dict(nrx=nrx, whole=rel(a, none), sub=rel(b, none)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--jobs", default=None, help="e.g. 0-29 (indices of --list)")
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)
    J = jobs()
    if a.list:
        for n, j in enumerate(J):
            print(n, *j)
        return
    if a.jobs is not None:
        lo, hi = (int(v) for v in a.jobs.split("-"))
        t_end = time.time() + a.budget
        for n in range(lo, hi + 1):
            scene, nrx, bmax = J[n]
            f = os.path.join(OUT, "runs", f"{scene}_N{nrx}_b{bmax:g}")
            if os.path.exists(f + ".json"):
                continue
            if time.time() + 25 > t_end:
                break
            res, cur = run(scene, nrx, bmax, mutations=((scene, nrx, bmax) in (("ramp1", 4, 100.0), ("sine", 4, 500.0), ("periodic", 4, 100.0),
                                                                                ("mountain", 4, 300.0), ("hills25", 4, 100.0))))
            np.savez_compressed(f + ".npz", **cur)
            json.dump(res, open(f + ".json", "w"), indent=1)
            L = res["levels"]
            print(f"{n:3d} {scene:10s} N{nrx} b{bmax:g}: ref {L['ref']:6.1f} none {L['none']:6.1f} oracle {L['oracle']:6.1f} | "
                  f"whole pkg {L['pkg_whole']:6.1f} ideal {L['ideal_whole']:6.1f} | sub pkg {L['pkg_sub']:6.1f} ideal {L['ideal_sub']:6.1f} | "
                  f"pkg-ideal {res['pkg_vs_ideal_whole']:6.1f}/{res['pkg_vs_ideal_sub']:6.1f} dB ({res['seconds']:.0f} s)", flush=True)
        left = sum(not os.path.exists(os.path.join(OUT, "runs", f"{s}_N{n}_b{b:g}.json")) for s, n, b in J)
        print("left:", left)
    if a.checks:
        import verify_sata_recon_report as R
        R.checks(zero_checks(), cell_checks())
    if a.pdf:
        import verify_sata_recon_report as R
        R.pdf()


if __name__ == "__main__":
    main()
