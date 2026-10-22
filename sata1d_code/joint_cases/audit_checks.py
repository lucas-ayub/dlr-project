"""
Checks of the SATA 1D implementation (whole band, per sub-band, joint in the sub-apertures, joint over the whole
line). Every check prints PASS / FAIL with the measured number and its tolerance; the numbers go to
joint_cases/audit_dem/checks.json. Nothing in the package is changed.

    python joint_cases/audit_checks.py --part kernel      # from sata1d_code; parts: kernel base periodic predict cells line blind noise
    python joint_cases/audit_checks.py --report           # all the PASS / FAIL lines stored so far

kernel    the sub-aperture loop: identity with a zero map, rotation by a constant map (sign of inverse), the package loop
          against the reference routine sata_for_topography (runs/sata/reference_code_SATA1D.py), the ruler
          (which cell every Doppler frequency reads; whole band and every sub-band), the cell of a scatterer in every
          channel, the delta_C0 values against the phase measured between two simulated echoes
base      Nrx = 2 ... 8: zero map = standard reconstruction for every method; one target on a flat terrain at height h
          (the DEM is the constant h): every method must reach the oracle
periodic  SATA before the reconstruction with a DEM: terrain with the same height at x and x +- X (X = one fold of the
          channel PRF on the ground): whole band and per sub-band must reach the oracle; terrain with different
          heights at x and x +- X: they must fail and the joint must not
predict   the level left by SATA whole band / per sub-band with a DEM, computed from the addresses of the folds only
          (which ground position every folded parcel reads), against the simulation
cells     the DEM tables: channel_maps (height table) against terrain_lookup_table (one range-history fit per cell),
          the table at the cells of the targets against the targets' own delta_C0
line      joint over the whole line: adjoint of the model, wrong DEMs must not work, flat DEM = no SATA
blind     no method uses the targets: same result with the list of targets removed from the configuration it receives
noise     (no PASS / FAIL) noise scaling of the joint reconstructions and sensitivity to DEM errors
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import types
from dataclasses import replace
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
import sata1d.sata as SS
from sata1d.config import ArrayGeometry
from sata1d.joint import reconstruct_joint, reference_coefficients
from sata1d.joint_line import height_table, channel_maps, line_maps, reconstruct_joint_line, LineModel
from sata1d.subband import sata_1d_subband
from sata1d.signal_model import raw
import audit_dem as A

OUT = os.path.join(A.OUT, "checks.json")
LINES = []


def check(name, value, ok, tol):
    line = f"{'PASS' if ok else 'FAIL'} | {name}: {value} (tolerance: {tol})"
    LINES.append(line)
    print(line, flush=True)


def info(name, value):
    line = f"     | {name}: {value}"
    LINES.append(line)
    print(line, flush=True)


# ------------------------------------------------------------------------------------------------- helpers
def config(xs, hs, bxt, amps=None):
    cfg = S.make_config(list(zip(np.atleast_1d(xs), np.atleast_1d(hs))), nrx=len(bxt), bxt_max=0.0, amplitudes=amps, tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float)))
    return cfg, S.build_platform_tracks(cfg)


def iso_point(cfg, x, h):
    H, r0, y0 = cfg.scene.H, cfg.scene.r0, cfg.scene.y0
    return cfg.scene.ptg + np.array([x, np.sqrt(r0 ** 2 - (H - h) ** 2) - y0, h])


def focus(cfg, tr):
    Hf = np.conj(np.fft.fft(S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])))
    return lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))


def level(cfg, env, pk, xs):
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt_amb = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    return float(A.ambiguity_level(t, env, pk, np.atleast_1d(xs), cfg.system.vs, dt_amb))


def oracle_channels(cfg, tr):
    """Every target corrected with its own delta_C0 before the channels are summed (uses the targets: a bound only)."""
    dC0 = S.delta_C0_per_target(cfg, tr)
    out = np.zeros((cfg.Nrx, cfg.Na_ch), complex)
    for j, p in enumerate(cfg.scene.points[1:]):
        out += S.generate_channels(cfg, tr, ptgs=[p], amps=[cfg.scene.amps[j]]) * np.exp(2j * np.pi * dC0[j] / cfg.system.wl)[:, None]
    return out


def all_methods(cfg, tr, terrain, xs, with_oracle=True):
    ch = S.generate_channels(cfg, tr); F = focus(cfg, tr); pk = F(S.generate_reference(cfg, tr)).max()
    hg = terrain(np.linspace(-40e3, 40e3, 8001)); tab = height_table(cfg, tr, hg.min(), hg.max())
    Mc, Ml = channel_maps(cfg, tr, terrain, tab), line_maps(cfg, tr, terrain, tab)
    r = dict(none=level(cfg, F(S.reconstruct(cfg, tr, ch)), pk, xs), whole=level(cfg, F(A.sata_whole(cfg, tr, ch, Mc)), pk, xs),
             sub=level(cfg, F(A.sata_sub(cfg, tr, ch, Mc)), pk, xs),
             joint=level(cfg, F(reconstruct_joint(cfg, tr, ch, maps=Mc, sata_osf=4, occupied=True)), pk, xs),
             line=level(cfg, F(reconstruct_joint_line(cfg, tr, ch, Ml)), pk, xs))
    if with_oracle:
        r["oracle"] = level(cfg, F(S.reconstruct(cfg, tr, oracle_channels(cfg, tr))), pk, xs)
    return r


def reference_routine():
    """sata_for_topography of the reference code, unchanged (only the function, which needs numpy alone)."""
    path = os.path.join(os.path.dirname(os.path.dirname(HERE)), "runs", "sata", "reference_code_SATA1D.py")
    src = open(path).read()
    src = "import numpy as np\n" + src[src.index("def sata_for_topography"):src.index("def sata(data, pos")]
    mod = types.ModuleType("reference_sata"); exec(compile(src, path, "exec"), mod.__dict__)

    def run(data, M, prf, v, wl, r0, inverse=True, osf=4):
        with contextlib.redirect_stdout(io.StringIO()):
            return mod.sata_for_topography(np.array(data, complex).copy(), np.asarray(M, float), r0, prf, 1, v, 0.0, wl, np.array([r0]), inverse=inverse, sata_osf=osf)
    return run


# ------------------------------------------------------------------------------------------------- parts
def part_kernel():
    sysp = S.SystemParams(); P, v, wl = 500.0, sysp.vs, sysp.wl
    r0 = S.Scene(rDelay=S.config.RDELAY_DEFAULT).r0
    rng = np.random.default_rng(0); n = 3072
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    for sch in ("cola", "crossfade", "margin"):
        e = np.abs(SS.sata_1d(x, np.zeros(n), r0, P, v, wl, r0, inverse=True, sata_osf=4, scheme=sch) - x).max()
        check(f"zero map leaves the line unchanged, whole-band loop, scheme {sch} (max |difference|)", f"{e:.1e}", e < 1e-12, "1e-12")
    e = max(np.abs(sata_1d_subband(x, np.zeros(n), r0, P, v, wl, r0, f_k=(-2 + k + 0.5) * P, inverse=True, sata_osf=4) - x).max() for k in range(4))
    check("zero map leaves the line unchanged, sub-band loops k = 0..3 (max |difference|)", f"{e:.1e}", e < 1e-12, "1e-12")
    ref = reference_routine()
    e = np.abs(ref(x, np.zeros(n), P, v, wl, r0) - x)[:-22].max()
    check("zero map leaves the line unchanged, reference routine (all but its last 22 samples)", f"{e:.1e}", e < 1e-12, "1e-12")
    c0 = 0.031
    for name, fn in (("package", lambda inv: SS.sata_1d(x, np.full(n, c0), r0, P, v, wl, r0, inverse=inv, sata_osf=4)),
                     ("reference routine", lambda inv: ref(x, np.full(n, c0), P, v, wl, r0, inverse=inv))):
        for inv, sign in ((True, 1.0), (False, -1.0)):
            y = fn(inv); rot = float(np.angle(np.vdot(x[30:-30], y[30:-30]))); want = sign * 2 * np.pi * c0 / wl
            check(f"constant map of 31 mm, {name}, inverse={inv}: rotation of the line [rad]", f"{rot:+.6f} (expected {want:+.6f})", abs(rot - want) < 1e-9, "1e-9")
    M = 0.05 * np.sin(2 * np.pi * np.arange(n) / 700.0) + 0.02 * (np.arange(n) > 1500)
    a = SS.sata_1d(x, M, r0, P, v, wl, r0, inverse=True, sata_osf=4, scheme="crossfade"); b = ref(x, M, P, v, wl, r0)
    e = float(np.linalg.norm((a - b)[30:-60]) / np.linalg.norm(x[30:-60]))
    check("package loop (scheme crossfade) against the reference routine, same line and map (rms difference / rms)", f"{e:.4f}", e < 0.01, "0.01")
    # ruler
    slope = 1e-5; Mr = slope * np.arange(n, dtype=float); tt = np.arange(n) / P; c = 1500
    ruler = lambda f: r0 * np.tan(np.arcsin(wl * f / (2 * v))) / v * P
    info("one fold on the ground X [cells of a channel line] / [m]", f"{ruler(P):.2f} / {ruler(P) * v / P:.1f}")
    errs = []
    for f in np.linspace(-200, 200, 41):
        tone = np.exp(2j * np.pi * f * tt); y = SS.sata_1d(tone, Mr, r0, P, v, wl, r0, inverse=True, sata_osf=4)
        cell = np.angle(y[c] * np.conj(tone[c])) * wl / (2 * np.pi * slope)
        cell += np.round((c + ruler(f) - cell) / (wl / slope)) * (wl / slope)
        errs.append(cell - (c + ruler(f)))
    check("ruler, whole band: cell read minus (block centre + x(f)), |f| <= 200 Hz [cells]", f"mean {np.mean(errs):+.2f}, largest {np.abs(errs).max():.2f}",
          abs(np.mean(errs)) < 0.6 and np.abs(errs).max() < 3, "mean 0.6, largest 3 (resolution of a block: 20 cells)")
    for k in range(4):
        fk = (-2 + k + 0.5) * P; errs = []
        for f in np.r_[np.linspace(-190, -70, 9), np.linspace(70, 190, 9)]:
            tone = np.exp(2j * np.pi * f * tt); y = sata_1d_subband(tone, Mr, r0, P, v, wl, r0, f_k=fk, inverse=True, sata_osf=4)
            fabs = fk - P / 2 + np.mod(f - (fk - P / 2), P)
            cell = np.angle(y[c] * np.conj(tone[c])) * wl / (2 * np.pi * slope)
            cell += np.round((c + ruler(fabs) - cell) / (wl / slope)) * (wl / slope)
            errs.append(cell - (c + ruler(fabs)))
        check(f"ruler, sub-band k = {k}: cell read minus (block centre + x(absolute Doppler in the sub-band)) [cells]",
              f"mean {np.mean(errs):+.2f}, largest {np.abs(errs).max():.2f}", abs(np.mean(errs)) < 0.6 and np.abs(errs).max() < 3, "mean 0.6, largest 3")
    # cell of a scatterer and delta_C0
    cfg, tr = config([0.0], [0.0], [0, 134.9, 20.5, 8.3])
    worst = 0.0
    for xq in (-5000.0, -333.3, 0.0, 1234.5, 6000.0):
        for hq in (0.0, 800.0):
            p = iso_point(cfg, xq, hq)
            for i in range(cfg.Nrx):
                rb = np.linalg.norm(tr.ptx - p, axis=1) + np.linalg.norm(tr.prx[i] - p, axis=1)
                k0 = np.argmin(rb); aa, bb, _ = np.polyfit(np.arange(-3, 4), rb[k0 - 3:k0 + 4], 2)
                true_cell = (k0 - bb / (2 * aa)) / cfg.Nrx
                formula = cfg.Na_ch / 2 + (cfg.scene.x0 + xq + 0.5 * cfg.array.bat[i]) / (cfg.system.vs / cfg.PRF_op)
                worst = max(worst, abs(true_cell - formula))
    check("cell of a scatterer in every channel (minimum of the Tx + Rx range) against Na_ch/2 + (x0 + x + bat/2)/ds [cells]", f"{worst:.1e}", worst < 1e-3, "1e-3")
    worst, coh = 0.0, 1.0
    for hq in (50.0, 300.0, 1000.0, 3000.0):
        for xq in (0.0, 4000.0):
            p, p0 = iso_point(cfg, xq, hq), iso_point(cfg, xq, 0.0)
            for i in range(cfg.Nrx):
                sh, s0 = raw(cfg, tr, [p], tr.prx[i], tr.vrx[i]), raw(cfg, tr, [p0], tr.prx[i], tr.vrx[i])
                d = S.residual_C0(cfg, tr, p, i)
                meas = -np.angle(np.vdot(s0, sh)) * wl / (2 * np.pi); meas += np.round((d - meas) / wl) * wl
                worst = max(worst, abs(meas - d)); coh = min(coh, abs(np.vdot(s0, sh)) / (np.linalg.norm(s0) * np.linalg.norm(sh)))
    check("delta_C0 (residual_C0) against the phase measured between the echo at height h and the echo at h = 0, h up to 3 km [mm]", f"{worst * 1e3:.3f}", worst < 1e-4, "0.1")
    check("the height changes only a constant phase of the echo of a channel (coherence between the two echoes)", f"{coh:.8f}", coh > 0.999999, "0.999999")
    th = np.arctan2(cfg.scene.y0, cfg.scene.H)
    info("delta_C0 / (bxt h): measured, and -1 / (r0 tan(theta)) [1/m]", f"{S.residual_C0(cfg, tr, iso_point(cfg, 0, 300.0), 1) / (134.9 * 300.0):.4e}, {-1 / (r0 * np.tan(th)):.4e}")


def part_base():
    flat = lambda h: (lambda x: np.full(np.shape(np.asarray(x, float)), float(h)))
    for n in (2, 3, 4, 5, 6, 7, 8):
        bxt = A.random_baselines(n, 300.0)
        xs = np.linspace(-6000, 6000, 9)
        cfg, tr = config(xs, 0 * xs, bxt, A.phases(len(xs))); ch = S.generate_channels(cfg, tr)
        std = S.reconstruct(cfg, tr, ch); Z = np.zeros((n, cfg.Na_ch))
        rel = lambda y: float(np.linalg.norm(y - std) / np.linalg.norm(std))
        e = dict(whole=rel(A.sata_whole(cfg, tr, ch, Z)), sub=rel(A.sata_sub(cfg, tr, ch, Z)),
                 joint_full=rel(reconstruct_joint(cfg, tr, ch, maps=Z, occupied=False)), joint_occ=rel(reconstruct_joint(cfg, tr, ch, maps=Z, occupied=True)),
                 line=rel(reconstruct_joint_line(cfg, tr, ch, np.zeros((n, cfg.Na)))))
        check(f"Nrx = {n}: zero map = standard reconstruction (rms difference / rms): SATA whole band, SATA per sub-band, joint full system",
              f"{e['whole']:.1e}, {e['sub']:.1e}, {e['joint_full']:.1e}", max(e["whole"], e["sub"], e["joint_full"]) < 1e-5, "1e-5")
        check(f"Nrx = {n}: zero map = standard reconstruction: joint with occupied parcels, joint over the whole line",
              f"{e['joint_occ']:.1e}, {e['line']:.1e}", max(e["joint_occ"], e["line"]) < 5e-3, "5e-3 (they leave out what is outside the Doppler band / regularise)")
        for h in (500.0, 1500.0):
            cfg, tr = config([1300.0], [h], bxt)
            r = all_methods(cfg, tr, flat(h), [1300.0])
            d = max(abs(r[k] - r["oracle"]) for k in ("whole", "sub", "joint", "line"))
            check(f"Nrx = {n}: one target on a flat terrain at {h:g} m, DEM = {h:g} m: every method against the oracle [dB]",
                  f"no SATA {r['none']:.1f}, whole band {r['whole']:.1f}, per sub-band {r['sub']:.1f}, joint {r['joint']:.1f}, whole line {r['line']:.1f}, oracle {r['oracle']:.1f}",
                  d < 0.3, "0.3 dB from the oracle")


def part_periodic():
    X = A.X_FOLD
    scenes = {"1 target": np.array([X / 4]), "5 targets 2.6 km apart": (np.arange(5) - 2) * 2600.0 + 700.0, "25 targets 500 m apart": (np.arange(25) - 12) * 500.0 + 130.0}
    for fac, name in ((1.0, "period X"), (0.5, "period X / 2"), (1.5, "period 1.5 X"), (2.0, "period 2 X")):
        h = lambda x, fac=fac: 100.0 - 100.0 * np.cos(2 * np.pi * np.asarray(x, float) / (fac * X))
        for sname, xs in scenes.items():
            cfg, tr = config(xs, h(xs), [0, 27.0, 4.1, 1.7], A.phases(len(xs)) if len(xs) > 5 else None)
            r = all_methods(cfg, tr, h, xs)
            txt = f"no SATA {r['none']:.1f}, whole band {r['whole']:.1f}, per sub-band {r['sub']:.1f}, joint {r['joint']:.1f}, whole line {r['line']:.1f}, oracle {r['oracle']:.1f}"
            if fac in (1.0, 0.5):
                d = max(abs(r[k] - r["oracle"]) for k in ("whole", "sub"))
                tol = 2.0 if fac == 1.0 else 7.0     # period X / 2: slopes up to 11 deg, the terrain changes inside the ground of one sub-aperture bin
                check(f"terrain 100 - 100 cos, {name} (same height at x and x +- X), {sname}: SATA whole band and per sub-band reach the oracle [dB]", txt, d < tol,
                      f"{tol:g} dB from the oracle" + ("" if fac == 1.0 else " (slopes up to 11 deg: the terrain changes inside the ground covered by one sub-aperture bin)"))
            else:
                worse = min(r["whole"], r["sub"]) - r["oracle"]; dj = max(abs(r["joint"] - r["oracle"]), abs(r["line"] - r["oracle"]))
                check(f"terrain 100 - 100 cos, {name} (different heights at x and x +- X), {sname}: SATA before the reconstruction fails, the joint does not [dB]",
                      txt, worse > 1.5 and dj < 2.0, "whole band and per sub-band more than 1.5 dB above the oracle; joint within 2 dB of it")


def predicted_level(cfg, tr, terrain, xp, mode):
    """Highest ambiguity left by SATA whole band (mode='whole') or per sub-band ('sub') reading a DEM, for one target at
    xp, from the fold addresses only: energy of true Doppler f_q (fold q) is given the phase of the ground position
    xp - a(f_q) + a(label), label = baseband frequency (whole band) or the absolute Doppler of the output sub-band;
    what is left after the inversion is [G^-1 diag(exp(j eps_i)) G]_(q', q)."""
    Nrx, Nsb, P, wl, v, r0 = cfg.Nrx, cfg.Nsb, cfg.PRF_op, cfg.system.wl, cfg.system.vs, cfg.scene.r0
    B = 4.0 * v * np.sin(cfg.theta_tx / 2.0) / wl
    coef = reference_coefficients(cfg, tr)
    nb = 400
    phi = (np.arange(nb) + 0.5) * P / nb
    f = -Nsb * P / 2.0 + np.mod(phi + Nsb * P / 2.0, P)[None, :] + (np.arange(Nsb) * P)[:, None]
    a = lambda ff: r0 * np.tan(np.arcsin(np.clip(wl * ff / (2 * v), -1, 1)))
    G = np.zeros((nb, Nrx, Nsb), complex)
    for i in range(Nrx):
        C0, C1, C2, Dt = coef[i]
        G[:, i, :] = np.exp(-2j * np.pi * (C0 / wl + (-C1 + Dt) * f.T + C2 * wl * f.T ** 2))
    Gi = np.linalg.inv(G)
    hh = terrain(np.linspace(-60e3, 60e3, 24001)); hg, tab = height_table(cfg, tr, hh.min(), hh.max())
    pol = [np.polyfit(hg, tab[:, i], 3) for i in range(Nrx)]
    hp = float(terrain(np.array([xp]))[0])
    fbase = np.mod(phi + P / 2, P) - P / 2
    best = 0.0
    for d in (-2, -1, 1, 2):
        acc = 0j
        for q in range(Nsb):
            qd = q + d
            if not 0 <= qd < Nsb:
                continue
            mask = (np.abs(f[q]) <= B / 2) & (np.abs(f[qd]) <= cfg.abw / 2)
            xread = xp - a(f[q]) + a(fbase if mode == "whole" else f[qd])
            L = np.zeros(nb, complex)
            for i in range(Nrx):
                eps = 2 * np.pi / wl * (np.polyval(pol[i], terrain(xread)) - np.polyval(pol[i], hp))
                L += Gi[:, qd, i] * np.exp(1j * eps) * G[:, i, q]
            acc += (L * mask).sum() * (P / nb)
        best = max(best, abs(acc) / B)
    return 20 * np.log10(best + 1e-12)


def part_predict():
    dif = []
    for sc in ("one_refpoint", "one_sine", "one_peak"):
        for n in (2, 4, 6):
            for bm in (100.0, 300.0):
                c = A.SCENES[sc][1](); xs, terr = c["xs"], c["terrain"]
                cfg, tr = config(xs, c["hs"], A.random_baselines(n, bm))
                ch = S.generate_channels(cfg, tr); F = focus(cfg, tr); pk = F(S.generate_reference(cfg, tr)).max()
                M = channel_maps(cfg, tr, terr)
                sw, sp = level(cfg, F(A.sata_whole(cfg, tr, ch, M)), pk, xs), level(cfg, F(A.sata_sub(cfg, tr, ch, M)), pk, xs)
                pw, ps = predicted_level(cfg, tr, terr, xs[0], "whole"), predicted_level(cfg, tr, terr, xs[0], "sub")
                dif.append((sw - pw, sp - ps))
                check(f"{c['label'].replace('$', '')}, Nrx = {n}, largest bxt {bm:g} m: level left by SATA with the DEM, simulated against predicted from the fold addresses [dB]",
                      f"whole band {sw:.1f} / {pw:.1f}, per sub-band {sp:.1f} / {ps:.1f}", abs(sw - pw) < 2.0 and abs(sp - ps) < 1.0, "whole band 2 dB, per sub-band 1 dB")
    dif = np.array(dif)
    info("simulated minus predicted over the cases [dB]", f"whole band mean {dif[:, 0].mean():+.2f}, largest {np.abs(dif[:, 0]).max():.2f}; per sub-band mean {dif[:, 1].mean():+.2f}, largest {np.abs(dif[:, 1]).max():.2f}")


def part_cells():
    c = A.SCENES["few_mountains_5"][1](); xs, terr = c["xs"], c["terrain"]
    cfg, tr = config(xs, c["hs"], A.FIXED["u500"][0])
    t0 = time.time(); Tpk = np.array([S.terrain_lookup_table(cfg, tr, i, terr, step=8) for i in range(cfg.Nrx)]); t1 = time.time() - t0
    t0 = time.time(); Tme = channel_maps(cfg, tr, terr); t2 = time.time() - t0
    e = float(np.abs(Tpk - Tme).max() * 1e3)
    check("DEM tables of the channels: channel_maps (height table) against terrain_lookup_table (one range-history fit every 8 cells) [mm]",
          f"largest difference {e:.3f} of a range of {np.ptp(Tme) * 1e3:.0f} ({t2:.1f} s against {t1:.1f} s)", e < 0.5, "0.5")
    dC0 = S.delta_C0_per_target(cfg, tr)
    cells = np.array([[SS.az_pixel_of_scatterer(cfg, x, i) for i in range(cfg.Nrx)] for x in xs])
    e = float(np.abs(Tme[np.arange(cfg.Nrx)[None, :], cells] - dC0).max() * 1e3)
    check("DEM table at the cell of every target against the target's own delta_C0, slopes up to 21 deg, bxt up to 135 m [mm]", f"{e:.3f}", e < 2.0,
          "2 (the target is anywhere inside its 15.4 m cell)")


def part_line():
    c = A.SCENES["few_mountains_5"][1](); xs, terr = c["xs"], c["terrain"]
    cfg, tr = config(xs, c["hs"], A.FIXED["u500"][0])
    M = LineModel(cfg, tr, line_maps(cfg, tr, terr)); rng = np.random.default_rng(0)
    a = rng.standard_normal(cfg.Na) + 1j * rng.standard_normal(cfg.Na); b = rng.standard_normal((cfg.Nrx, cfg.Na_ch)) + 1j * rng.standard_normal((cfg.Nrx, cfg.Na_ch))
    e = float(abs(np.vdot(M.forward(a), b) - np.vdot(a, M.adjoint(b))) / abs(np.vdot(M.forward(a), b)))
    check("joint over the whole line: the adjoint is the adjoint of the model, |<A a, b> - <a, A^H b>| / |<A a, b>|", f"{e:.1e}", e < 1e-10, "1e-10")
    for sc, tag in (("few_mountains_5", "u500"), ("many_hills_49", "N4b300")):
        c = A.SCENES[sc][1](); xs, terr = c["xs"], c["terrain"]
        cfg, tr = config(xs, c["hs"], A.baseline(tag)[0], c["amps"])
        ch = S.generate_channels(cfg, tr); F = focus(cfg, tr); pk = F(S.generate_reference(cfg, tr)).max()
        none = level(cfg, F(S.reconstruct(cfg, tr, ch)), pk, xs)
        dems = {"true DEM": terr, "DEM mirrored, h(-x)": lambda x: terr(-np.asarray(x, float)), "DEM shifted by 1 km": lambda x: terr(np.asarray(x, float) - 1000.0),
                "flat DEM": lambda x: 0 * np.asarray(x, float)}
        r = {}
        for name, d in dems.items():
            y, i_ = reconstruct_joint_line(cfg, tr, ch, line_maps(cfg, tr, d), return_info=True)
            r[name] = (level(cfg, F(y), pk, xs), 20 * np.log10(i_["unexplained"]))
        txt = "; ".join(f"{k}: {v[0]:.1f} dB (channels not explained by the model: {v[1]:.1f} dB)" for k, v in r.items())
        ok = (r["true DEM"][0] < min(r["DEM mirrored, h(-x)"][0], r["DEM shifted by 1 km"][0]) - 20) and abs(r["flat DEM"][0] - none) < 0.5
        check(f"joint over the whole line, {c['label']}, {tag}: only the true DEM works; flat DEM = no SATA ({none:.1f} dB)", txt, ok,
              "true DEM more than 20 dB below the wrong DEMs; flat DEM within 0.5 dB of no SATA")


def part_noise():
    rng = np.random.default_rng(5)

    def smooth_error(rms, corr=500.0, seed=3, half=80e3, dx=20.0):
        g = np.random.default_rng(seed); x = np.arange(-half, half + dx, dx); k = np.fft.rfftfreq(x.size, dx)
        e = np.fft.irfft(np.fft.rfft(g.standard_normal(x.size)) * np.exp(-2.0 * (np.pi * 0.5 * corr * k) ** 2), x.size); e *= rms / e.std()
        return lambda xq: np.interp(np.asarray(xq, float), x, e)
    for sc, tag in (("few_mountains_5", "u100"), ("few_mountains_5", "u500"), ("few_ramp4_5", "s50"), ("many_mountains_49", "N4b300"), ("many_mountains_49", "N2b300"), ("many_hills_49", "N4b300")):
        c = A.SCENES[sc][1](); xs, terr = c["xs"], c["terrain"]
        cfg, tr = config(xs, c["hs"], A.baseline(tag)[0], c["amps"])
        hh = terr(np.linspace(-60e3, 60e3, 24001)); tab = height_table(cfg, tr, hh.min() - 40, hh.max() + 40)
        Mc, Ml = channel_maps(cfg, tr, terr, tab), line_maps(cfg, tr, terr, tab)
        noise = (rng.standard_normal((cfg.Nrx, cfg.Na_ch)) + 1j * rng.standard_normal((cfg.Nrx, cfg.Na_ch))) / np.sqrt(2)
        band = np.abs(np.fft.fftfreq(cfg.Na, 1 / cfg.prf)) <= cfg.abw / 2
        pw = lambda y: float((np.abs(np.fft.fft(y)[band]) ** 2).sum())
        p0 = pw(S.reconstruct(cfg, tr, noise))
        info(f"noise at the output re. the standard reconstruction, {c['label'].replace('$', '')}, bxt = {W_text(cfg)} [dB]",
             f"joint in the sub-apertures {10 * np.log10(pw(reconstruct_joint(cfg, tr, noise, maps=Mc, occupied=True)) / p0):+.1f}, "
             f"joint over the whole line {10 * np.log10(pw(reconstruct_joint_line(cfg, tr, noise, Ml)) / p0):+.1f}")
        if tag in ("u100", "u500", "N4b300"):
            ch = S.generate_channels(cfg, tr); F = focus(cfg, tr); pk = F(S.generate_reference(cfg, tr)).max()
            parts = []
            for rms in (0.0, 2.0, 5.0, 10.0):
                e = smooth_error(rms) if rms else (lambda x: 0 * np.asarray(x, float))
                td = lambda x: terr(x) + e(x)
                parts.append(f"{rms:g} m: {level(cfg, F(reconstruct_joint_line(cfg, tr, ch, line_maps(cfg, tr, td, tab))), pk, xs):.1f}")
            info(f"highest ambiguity of the joint over the whole line with a DEM error of rms (correlation length 500 m), same case [dB]", ", ".join(parts))


def part_blind():
    """No method may use the targets: the same channels and DEM, with the list of targets removed from the configuration
    that the method receives, must give the same line, bit by bit."""
    for sc, tag in (("few_mountains_5", "u500"), ("many_hills_49", "N3b300")):
        c = A.SCENES[sc][1](); xs, terr = c["xs"], c["terrain"]
        cfg, tr = config(xs, c["hs"], A.baseline(tag)[0], c["amps"])
        ch = S.generate_channels(cfg, tr)                               # the echoes (made with the targets)
        blind = replace(cfg, scene=replace(cfg.scene, extra_offsets=(), amplitudes=()))   # what the methods get: no targets
        tb = S.build_platform_tracks(blind)
        out = {}
        for name, k, t in (("with", cfg, tr), ("without", blind, tb)):
            hg = terr(np.linspace(-40e3, 40e3, 8001)); tab = height_table(k, t, hg.min(), hg.max())
            Mc, Ml = channel_maps(k, t, terr, tab), line_maps(k, t, terr, tab)
            out[name] = dict(none=S.reconstruct(k, t, ch), whole=A.sata_whole(k, t, ch, Mc), sub=A.sata_sub(k, t, ch, Mc),
                             joint=reconstruct_joint(k, t, ch, maps=Mc, sata_osf=4, occupied=True), line=reconstruct_joint_line(k, t, ch, Ml))
        d = {m: float(np.abs(out["with"][m] - out["without"][m]).max()) for m in out["with"]}
        check(f"{c['label'].replace('$', '')}, {tag}: the line of every method with and without the list of targets in the configuration (max |difference|)",
              ", ".join(f"{m} {v:.1e}" for m, v in d.items()), max(d.values()) == 0.0, "exactly 0")


def W_text(cfg):
    return "[" + ", ".join(f"{b:g}" for b in cfg.array.bxt) + "] m"


PARTS = dict(kernel=part_kernel, base=part_base, periodic=part_periodic, predict=part_predict, cells=part_cells, line=part_line, blind=part_blind, noise=part_noise)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=list(PARTS), nargs="+", default=[])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for p in a.part:
        LINES.clear(); t0 = time.time()
        PARTS[p]()
        store[p] = list(LINES)
        os.makedirs(A.OUT, exist_ok=True); json.dump(store, open(OUT, "w"), indent=1)
        print(f"part {p}: {sum(l.startswith('PASS') for l in LINES)} PASS, {sum(l.startswith('FAIL') for l in LINES)} FAIL ({time.time() - t0:.0f} s)", flush=True)
    if a.report:
        for p in PARTS:
            for l in store.get(p, []):
                print(f"[{p}] {l}")
        allp = [l for p in store.values() for l in p]
        print(f"total: {sum(l.startswith('PASS') for l in allp)} PASS, {sum(l.startswith('FAIL') for l in allp)} FAIL")
