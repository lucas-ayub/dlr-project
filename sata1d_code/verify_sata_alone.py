"""
SATA 1D on its own: the kernels on ONE channel WITHOUT folds, i.e. without the multichannel reconstruction.

Three parts, each with PASS / FAIL checks and numbers (sata_alone/results.json) and figures (sata_alone/*.png, sata_alone.pdf):

  A  The kernel alone on a monostatic line at the full PRF (2000 Hz > Doppler bandwidth: no folds). Every target
     carries a KNOWN phase exp(-j 2 pi D / wl), with D read from a map D(x); the kernel is given the same map and
     must return the line without the phases. No geometry, no reconstruction: only the sub-aperture loop.
       A1 zero map (identity)        A2 constant map (uniform rotation)
       A3 two plateaus: residual of a target against its distance to the step (the resolution of the kernel)
       A4 the same test with the kernel broken on purpose (the test must fail)
       A5 linear maps (what a slope does)                     A6 whole band = sub-band when Nrx = 1
  S  The kernels of the 4-channel system (PRF_op = 500 Hz: whole band and the four sub-band kernels) on lines made
     free of folds: the full-PRF line is limited to one Doppler window of 500 Hz and only then sampled at 500 Hz.
  B  One receiver with a cross-track baseline, sampled at the full PRF (Nrx = 1), with the real geometry and the
     DEM table (terrain_lookup_table): channel -> SATA -> reconstruction (for one channel: only the filter of the
     reference point) against the monostatic reference. bxt = 0 is the monostatic case (the map is zero).

    python verify_sata_alone.py            # everything (1 to 4 minutes; part B keeps every case in sata_alone/cache)
    python verify_sata_alone.py --fresh    # part B computed again
    python verify_sata_alone.py --parts A S --no-figures
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
from sata1d.subband import sata_1d_subband, sata_channels_subband, subband_frequency_beam

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sata_alone")
WL = 0.25
OSF = 4
CHECKS = []


def check(part, name, value, limit, larger=False, unit=""):
    ok = bool(value >= limit) if larger else bool(value <= limit)
    CHECKS.append(dict(part=part, name=name, value=float(value), limit=float(limit), larger=larger, unit=unit, ok=ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:.4g} {unit} ({'>=' if larger else '<='} {limit:g})", flush=True)
    return ok


def rel_db(a, b):
    """Energy of a - b relative to the energy of b [dB]."""
    return float(20 * np.log10(float(np.linalg.norm(a - b)) / float(np.linalg.norm(b)) + 1e-300))


def focuser(reference_point_line):
    """Azimuth compression with the echo of the reference point; the output is complex, target at its own cell."""
    Hf = np.conj(np.fft.fft(reference_point_line))
    n = len(reference_point_line)
    return lambda s: np.roll(np.fft.ifft(np.fft.fft(s) * Hf), n // 2)


def peak(I, c, up=32, half=64):
    """Position [cells] and complex value of the peak of |I| next to cell c (band-limited interpolation)."""
    w = np.arange(c - half, c + half)
    F = np.fft.fft(I[w])
    Z = np.zeros(len(w) * up, complex)
    Z[:half] = F[:half]
    Z[-half:] = F[-half:]
    z = np.fft.ifft(Z) * up
    k = int(np.argmax(np.abs(z)))
    return w[0] + k / up, z[k]


def target_numbers(I, I0, c, ds, lobe=5):
    """Of one target: shift of the peak [m], amplitude [dB] and phase [deg] of the peak relative to the reference
    image I0, and the largest difference between the two images outside the main lobe [dB re. the peak] with its
    distance to the target [m]."""
    p0, v0 = peak(I0, c)
    p1, v1 = peak(I, c)
    d = np.abs(I - I0)
    d[max(c - lobe, 0):c + lobe + 1] = 0
    k = int(np.argmax(d))
    return dict(shift_m=float((p1 - p0) * ds), amp_dB=float(20 * np.log10(abs(v1 / v0))), phase_deg=float(np.degrees(np.angle(v1 / v0))),
                side_dB=float(20 * np.log10(d[k] / abs(I0[c]) + 1e-300)), side_at_m=float((k - c) * ds))


# ================================================================================================ part A
def mono_setup(nrx=1):
    """One monostatic channel (bat = bxt = 0). nrx = 1: PRF_op = 2000 Hz; nrx = 4 only sets PRF_op = 500 Hz."""
    cfg = S.make_config([(0.0, 0.0)], nrx=nrx, bxt_max=0.0, tx_rx_channel0=True)
    return cfg, S.build_platform_tracks(cfg)


def kernel_args(cfg):
    return dict(rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=WL, r=cfg.scene.r0)


def broken_kernel(cfg, data, M, ruler_scale=1.0, centre_shift=0.0):
    """sata_1d with the ruler multiplied by ruler_scale and the block centre moved by centre_shift cells."""
    Tsubeff, hop, Nzp = SS.subaperture_sizing(cfg.scene.r0, cfg.PRF_op, cfg.system.vs, WL, OSF)
    _, azpos = SS.subaperture_axis(cfg.PRF_op, Nzp, cfg.system.vs, WL, cfg.scene.r0)
    return SS.sata_kernel(data, M, Tsubeff, hop, Nzp, ruler_scale * azpos + centre_shift, WL, True, "cola")


STEP = lambda x: np.where(np.asarray(x) < 0, 0.03, 0.11)          # two plateaus [m]: 115 degrees apart
DIST = (30, 60, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 4000, 6000)


def part_a():
    print("A. the kernel alone, monostatic line at the full PRF (no folds)")
    cfg, tr = mono_setup(1)
    Na, ds = cfg.Na, cfg.system.vs / cfg.PRF_op
    xcell = (np.arange(Na) - Na / 2) * ds - cfg.scene.x0            # x of every cell, relative to the reference point
    kern = kernel_args(cfg)
    foc = focuser(S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]))
    Tsubeff, hop, Nzp = SS.subaperture_sizing(cfg.scene.r0, cfg.PRF_op, cfg.system.vs, WL, OSF)
    res = dict(PRF_op=cfg.PRF_op, Na=Na, cell_m=ds, block=Tsubeff, hop=hop, Nzp=Nzp,
               block_s=Tsubeff / cfg.PRF_op, sata_cell_m=float(np.sqrt(WL * cfg.scene.r0 / 2)),
               ruler_m_per_Hz=float(WL * cfg.scene.r0 / (2 * cfg.system.vs)))
    whole = lambda s, M, scheme="cola": SS.sata_1d(s, M, inverse=True, sata_osf=OSF, scheme=scheme, **kern)
    sub = lambda s, M: sata_1d_subband(s, M, f_k=0.0, inverse=True, sata_osf=OSF, **kern)

    def line(xs, Dfun):
        pts = cfg.scene.ptg + np.array([[x, 0.0, 0.0] for x in xs])
        ref = S.generate_reference(cfg, tr, pts, np.ones(len(xs)))
        return ref, S.generate_reference(cfg, tr, pts, np.exp(-2j * np.pi * Dfun(np.asarray(xs, float)) / WL))

    xs = np.array([-6000.0, -3000.0, 0.0, 2500.0, 5500.0])
    # A1, A2
    for name, Dfun in (("A1 zero map", lambda x: 0.0 * x), ("A2 constant map (0.07 m = 100.8 deg)", lambda x: 0.0 * x + 0.07)):
        ref, s_in = line(xs, Dfun)
        M = Dfun(xcell)
        e_w, e_s = rel_db(whole(s_in, M), ref), rel_db(sub(s_in, M), ref)
        res[name[:2]] = dict(before_dB=rel_db(s_in, ref), whole_dB=e_w, sub_dB=e_s,
                             schemes={sc: rel_db(whole(s_in, M, sc), ref) for sc in ("cola", "crossfade", "margin")})
        check("A", f"{name}: output - reference, whole band", e_w, -250, unit="dB")
        check("A", f"{name}: output - reference, sub-band kernel", e_s, -250, unit="dB")
    # A3: one target at a distance d from the step, on either side
    rows = []
    for d in DIST:
        for side in (-1, 1):
            ref, s_in = line([side * d], STEP)
            out = whole(s_in, STEP(xcell))
            c = int(np.argmax(np.abs(foc(ref))))
            r = target_numbers(foc(out), foc(ref), c, ds)
            r.update(x=float(side * d), line_dB=rel_db(out, ref), before_deg=float(np.degrees(np.angle(foc(s_in)[c] / foc(ref)[c]))))
            rows.append(r)
    res["A3"] = rows
    far = [r for r in rows if abs(r["x"]) >= 600]
    check("A", "A3 two plateaus, target >= 600 m from the step: largest |phase| left", max(abs(r["phase_deg"]) for r in far), 0.5, unit="deg")
    check("A", "A3 two plateaus, target >= 600 m from the step: largest |amplitude change|", max(abs(r["amp_dB"]) for r in far), 0.05, unit="dB")
    check("A", "A3 two plateaus, target >= 600 m from the step: largest shift of the peak", max(abs(r["shift_m"]) for r in far), 0.2, unit="m")
    # A4: the kernel broken on purpose must be caught by A3
    mut = {}
    for name, fn in (("ruler with the sign flipped", lambda s, M: broken_kernel(cfg, s, M, ruler_scale=-1.0)),
                     ("ruler times 0.5", lambda s, M: broken_kernel(cfg, s, M, ruler_scale=0.5)),
                     ("ruler times 2", lambda s, M: broken_kernel(cfg, s, M, ruler_scale=2.0)),
                     ("block centre 200 cells (769 m) ahead", lambda s, M: broken_kernel(cfg, s, M, centre_shift=200.0)),
                     ("phase with the sign flipped (inverse=False)", lambda s, M: SS.sata_1d(s, M, inverse=False, sata_osf=OSF, **kern))):
        rr = []
        for d in (600, 1000, 2000, 4000):
            for side in (-1, 1):
                ref, s_in = line([side * d], STEP)
                c = int(np.argmax(np.abs(foc(ref))))
                r = target_numbers(foc(fn(s_in, STEP(xcell))), foc(ref), c, ds)
                r["x"] = float(side * d)
                rr.append(r)
        mut[name] = rr
        check("A", f"A4 broken kernel is caught ({name}): largest |phase| or loss", max(max(abs(r["phase_deg"]), 10 * abs(r["amp_dB"])) for r in rr), 5.0, larger=True, unit="deg")
    res["A4"] = mut
    # A5: linear maps D = g x (a slope): no step, the phase changes inside the cell of the kernel
    rows = []
    for g in (1e-5, 2e-5, 5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4):
        ref, s_in = line([500.0], lambda x: g * x)
        c = int(np.argmax(np.abs(foc(ref))))
        r = dict(g=g, rad_per_cell=float(2 * np.pi * g * res["sata_cell_m"] / WL))
        for sc in ("cola", "crossfade", "margin"):
            out = whole(s_in, g * xcell, sc)
            r[sc] = dict(target_numbers(foc(out), foc(ref), c, ds), line_dB=rel_db(out, ref))
        rows.append(r)
    res["A5"] = rows
    check("A", "A5 linear map, 0.08 rad per SATA cell: |phase| left", abs(rows[0]["cola"]["phase_deg"]), 0.2, unit="deg")
    check("A", "A5 linear map: the error grows with the gradient (largest difference outside the main lobe, last / first)",
          rows[-1]["cola"]["side_dB"] - rows[0]["cola"]["side_dB"], 20, larger=True, unit="dB")
    # A6: with one channel the sub-band kernel is the whole-band kernel
    ref, s_in = line(xs, STEP)
    res["A6_whole_vs_sub_dB"] = rel_db(whole(s_in, STEP(xcell)), sub(s_in, STEP(xcell)))
    check("A", "A6 one channel: whole band - sub-band kernel", res["A6_whole_vs_sub_dB"], -60, unit="dB")
    return res


# ================================================================================================ part S
def part_s():
    print("S. the kernels of the 4-channel system (PRF_op = 500 Hz) on lines without folds")
    cfg, tr = mono_setup(4)
    Na, Nch, P, ds = cfg.Na, cfg.Na_ch, cfg.PRF_op, cfg.system.vs / cfg.PRF_op
    xcell = (np.arange(Nch) - Nch / 2) * ds - cfg.scene.x0
    fa = np.fft.fftfreq(Na, 1 / cfg.prf)
    kern = kernel_args(cfg)
    ref_pt = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    B = 4.0 * cfg.system.vs * np.sin(cfg.theta_tx / 2.0) / WL

    def band(s, lo, hi):
        """Full-PRF line -> only the Doppler band [lo, hi) (at most PRF_op wide) -> sampled at PRF_op: no folds."""
        F = np.fft.fft(s)
        F[~((fa >= lo) & (fa < hi))] = 0
        return np.fft.ifft(F)[::cfg.Nrx]

    def one(x, lo, hi, kernel):
        p = cfg.scene.ptg + np.array([x, 0.0, 0.0])
        ref = band(S.generate_reference(cfg, tr, p[None], [1.0]), lo, hi)
        s_in = ref * np.exp(-2j * np.pi * STEP(x) / WL)
        out = kernel(s_in, STEP(xcell))
        foc = focuser(band(ref_pt, lo, hi))
        I0, I = foc(ref), foc(out)
        c = int(np.argmax(np.abs(I0)))
        return dict(x=float(x), amp_dB=float(20 * np.log10(abs(I[c] / I0[c]))), phase_deg=float(np.degrees(np.angle(I[c] / I0[c]))), line_dB=rel_db(out, ref))

    whole = lambda s, M: SS.sata_1d(s, M, inverse=True, sata_osf=OSF, **kern)
    xs = (-4000.0, -2000.0, -1000.0, -600.0, 600.0, 1000.0, 2000.0, 4000.0)
    res = dict(PRF_op=P, cell_m=ds, sizing=list(SS.subaperture_sizing(cfg.scene.r0, P, cfg.system.vs, WL, OSF)), B=B, kernels=[])
    cases = [("whole band", 0.0, whole)]
    for k in range(cfg.Nsb):
        f_k = subband_frequency_beam(cfg, k)[0]
        cases.append((f"sub-band {k}", f_k, lambda s, M, f_k=f_k: sata_1d_subband(s, M, f_k=f_k, inverse=True, sata_osf=OSF, **kern)))
    for name, f_k, kernel in cases:
        lo, hi = f_k - P / 2, f_k + P / 2
        # the part of the window that holds signal, and the same without 100 Hz next to each edge of the window
        occ = (max(lo, -B / 2), min(hi, B / 2))
        inner = (max(lo + 100, -B / 2), min(hi - 100, B / 2))
        r = dict(name=name, f_k=f_k, window=[lo, hi], occupied=list(occ), inner=list(inner),
                 full=[one(x, lo, hi, kernel) for x in xs], guard=[one(x, inner[0], inner[1], kernel) for x in xs])
        if name != "whole band":
            r["wrong_labels"] = [one(x, lo, hi, whole) for x in xs]            # the whole-band kernel on the line of this window
        res["kernels"].append(r)
        check("S", f"{name} kernel (window {lo:+.0f} .. {hi:+.0f} Hz), line 100 Hz away from the edges: largest |phase| left",
              max(abs(q["phase_deg"]) for q in r["guard"]), 1.0, unit="deg")
        check("S", f"{name} kernel, line filling the window: largest |phase| left", max(abs(q["phase_deg"]) for q in r["full"]), 3.0, unit="deg")
        if name != "whole band":
            check("S", f"{name}: the whole-band labels on the same line are caught (largest |phase| left)",
                  max(abs(q["phase_deg"]) for q in r["wrong_labels"]), 30.0, larger=True, unit="deg")
    # how wide the map has to be around a target: the table of mode="lut" (target cell +- 2), the footprint, the whole line
    hw = SS.sata_footprint_halfwidth(cfg)
    x, D = 1000.0, 0.07
    p = cfg.scene.ptg + np.array([x, 0.0, 0.0])
    c = int(round(Nch / 2 + (cfg.scene.x0 + x) / ds))
    ref = band(S.generate_reference(cfg, tr, p[None], [1.0]), -150.0, 150.0)
    foc = focuser(band(ref_pt, -150.0, 150.0))
    I0 = foc(ref)
    cp = int(np.argmax(np.abs(I0)))
    res["map_support"] = dict(put_deg=float(np.degrees(np.angle(np.exp(-2j * np.pi * D / WL)))), footprint_halfwidth_cells=hw)
    for h, name in ((2, "lut"), (hw, "footprint"), (Nch, "whole_line")):
        M = np.zeros(Nch)
        M[max(c - h, 0):c + h + 1] = D
        q = target_numbers(foc(whole(ref * np.exp(-2j * np.pi * D / WL), M)), I0, cp, ds)
        res["map_support"][name] = q
        print(f"  (info) map non-zero only at the target cell +- {min(h, Nch)} cells ({name}): phase put {res['map_support']['put_deg']:.1f} deg, left {q['phase_deg']:+.2f} deg, amplitude {q['amp_dB']:+.2f} dB", flush=True)
    check("S", "whole band kernel, map of the footprint width (+- 82 cells) around the target: |phase| left", abs(res["map_support"]["footprint"]["phase_deg"]), 1.0, unit="deg")
    return res


# ================================================================================================ part B
def tanh_step(x, x0, width=1.0):
    return 0.5 * (1.0 + np.tanh((np.asarray(x, float) - x0) / width))


TERRAINS = {   # name -> (h(x) [m], description, target positions [m])
    "flat0": (lambda x: 0.0 * np.asarray(x, float), "flat terrain at h = 0", (-5000.0, -2000.0, 0.0, 2500.0, 5500.0)),
    "flat500": (lambda x: 0.0 * np.asarray(x, float) + 500.0, "flat terrain at h = 500 m", (-5000.0, -2000.0, 0.0, 2500.0, 5500.0)),
    "plateaus": (lambda x: 200.0 + 400.0 * tanh_step(x, -3500.0) - 300.0 * tanh_step(x, 1000.0) + 500.0 * tanh_step(x, 4000.0),
                 "plateaus of 200, 600, 300 and 800 m (steps at x = -3.5, 1 and 4 km)", (-6000.0, -5000.0, -2000.0, -500.0, 2500.0, 5500.0, 7000.0)),
    "ramp2": (lambda x: np.tan(np.radians(2.0)) * np.clip(np.asarray(x, float) + 8000.0, 0.0, 16000.0), "ramp of 2 deg (h 0 -> 559 m from -8 to +8 km)", (-5000.0, -2000.0, 0.0, 2500.0, 5500.0)),
    "ramp8": (lambda x: np.tan(np.radians(8.0)) * np.clip(np.asarray(x, float) + 8000.0, 0.0, 16000.0), "ramp of 8 deg (h 0 -> 2249 m from -8 to +8 km)", (-5000.0, -2000.0, 0.0, 2500.0, 5500.0)),
    "sine": (lambda x: 100.0 + 100.0 * np.sin(2 * np.pi * np.asarray(x, float) / 20e3), "h = 100 + 100 sin(2 pi x / 20 km)", (-5000.0, -2000.0, 0.0, 2500.0, 5500.0)),
    "mountain": (lambda x: 1000.0 * np.exp(-0.5 * (np.asarray(x, float) / 1500.0) ** 2), "isolated mountain 1000 m high (Gaussian, sigma 1.5 km)", (-5200.0, -2600.0, 0.0, 2600.0, 5200.0)),
}
B_CASES = [("flat0", 0.0), ("plateaus", 0.0), ("flat0", 300.0), ("flat500", 300.0), ("plateaus", 300.0), ("plateaus", 100.0),
           ("ramp2", 300.0), ("ramp8", 300.0), ("ramp8", 100.0), ("sine", 300.0), ("mountain", 300.0), ("mountain", 100.0)]


def cut_band(cfg, s):
    """A full-PRF line with the Doppler spectrum cut to the processed bandwidth (what reconstruct does to its output)."""
    F = np.fft.fft(s)
    F[np.abs(np.fft.fftfreq(cfg.Na, 1 / cfg.prf)) > cfg.abw / 2] = 0
    return np.fft.ifft(F)


def run_b(tname, bxt, step=16, high_level=False):
    terrain, desc, xs = TERRAINS[tname]
    cfg = S.make_config([(x, float(terrain(x))) for x in xs], nrx=1, bxt_max=0.0)
    cfg = replace(cfg, array=ArrayGeometry(bat=np.zeros(1), bxt=np.array([float(bxt)])))
    tr = S.build_platform_tracks(cfg)
    ds = cfg.system.vs / cfg.PRF_op
    ch = S.generate_channels(cfg, tr)                               # [1, Na]: one receiver, sampled at the full PRF
    ref = cut_band(cfg, S.generate_reference(cfg, tr))
    M = S.terrain_lookup_table(cfg, tr, 0, terrain, step=step) if bxt else np.zeros(cfg.Na)      # the DEM table (zero without baseline)
    dC0 = S.delta_C0_per_target(cfg, tr)[:, 0]
    out = {"none": S.reconstruct(cfg, tr, ch)}
    corr = SS.sata_1d(ch[0], M, inverse=True, sata_osf=OSF, **kernel_args(cfg))
    out["whole"] = S.reconstruct(cfg, tr, corr[None])
    corr = sata_channels_subband(cfg, tr, ch, 0, sata_osf=OSF, maps=[M])
    out["sub"] = reconstruct_from_spectra(cfg, tr, [[channel_spectrum(cfg, corr[0])]], True)
    if high_level:                                                  # the entry points of the package with the maps made from the targets
        out["whole_footprint"] = S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=OSF, mode="footprint"))
        out["sub_footprint"] = S.reconstruct_subband(cfg, tr, ch, sata_osf=OSF, mode="footprint")
        out["whole_lut"] = S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=OSF, mode="lut"))
    foc = focuser(S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]))
    I0 = foc(ref)
    img = {k: foc(v) for k, v in out.items()}
    cells = [int(round(cfg.Na / 2 + x / ds)) for x in xs]                       # in the focused image (positions relative to the reference point)
    map_cells = [int(round(cfg.Na / 2 + (cfg.scene.x0 + x) / ds)) for x in xs]  # in the table of the channel (bat = 0)
    g = np.gradient(M, ds)
    r = dict(terrain=tname, desc=desc, bxt=float(bxt), xs=list(xs), hs=[float(terrain(x)) for x in xs], dC0=[float(d) for d in dC0],
             expected_deg=[float(np.degrees(np.angle(np.exp(-2j * np.pi * d / WL)))) for d in dC0],
             rad_per_cell=[float(2 * np.pi * abs(g[c]) * np.sqrt(WL * cfg.scene.r0 / 2) / WL) for c in map_cells],
             line_dB={k: rel_db(v, ref) for k, v in out.items()},
             targets={k: [target_numbers(v, I0, c, ds) for c in cells] for k, v in img.items()},
             whole_vs_sub_dB=float(20 * np.log10(float(np.linalg.norm(out["whole"] - out["sub"])) / float(np.linalg.norm(ref)) + 1e-300)),
             sata_minus_none_dB=max(rel_db(out["whole"], out["none"]), rel_db(out["sub"], out["none"])))
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    plot = dict(t=t.astype(np.float32), x=(cfg.system.vs * t).astype(np.float32), M=M.astype(np.float32), ref=np.abs(I0).astype(np.float32),
                **{k: np.abs(v).astype(np.float32) for k, v in img.items()},
                **{"err_" + k: np.abs(img[k] - I0).astype(np.float32) for k in ("none", "whole", "sub")})    # |IRF - Reference|, complex difference
    return r, plot


def part_b(step=16, fresh=False):
    print("B. one receiver with a cross-track baseline at the full PRF, DEM table, against the monostatic reference")
    res, plots = [], {}
    os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
    for tname, bxt in B_CASES:
        t0 = time.time()
        cache = os.path.join(OUT, "cache", f"B_{tname}_{bxt:g}_step{step}")          # a case already computed is read back
        if not fresh and os.path.exists(cache + ".json") and os.path.exists(cache + ".npz") and "err_none" in np.load(cache + ".npz").files:
            r, plot = json.load(open(cache + ".json")), dict(np.load(cache + ".npz"))
        else:
            r, plot = run_b(tname, bxt, step, high_level=(tname, bxt) in (("flat500", 300.0), ("plateaus", 300.0)))
            np.savez_compressed(cache + ".npz", **plot)
            json.dump(r, open(cache + ".json", "w"), indent=1)
        res.append(r)
        plots[f"{tname}_{bxt:g}"] = plot
        ph = lambda k: max(abs(q["phase_deg"]) for q in r["targets"][k])
        print(f"  {r['desc']}, bxt = {bxt:g} m: line error no SATA {r['line_dB']['none']:.1f}, whole {r['line_dB']['whole']:.1f}, sub {r['line_dB']['sub']:.1f} dB | "
              f"largest |phase| no SATA {ph('none'):.1f}, whole {ph('whole'):.2f}, sub {ph('sub'):.2f} deg ({time.time() - t0:.0f} s)", flush=True)
    by = {(r["terrain"], r["bxt"]): r for r in res}
    ph = lambda r, k: max(abs(q["phase_deg"]) for q in r["targets"][k])
    floor = ph(by[("flat0", 300.0)], "none")
    for tname in ("flat0", "plateaus"):
        r = by[(tname, 0.0)]
        check("B", f"monostatic channel (bxt = 0), {r['desc']}: SATA - no SATA", r["sata_minus_none_dB"], -200, unit="dB")
        check("B", f"monostatic channel (bxt = 0), {r['desc']}: output - reference", r["line_dB"]["whole"], -100, unit="dB")
    r = by[("flat0", 300.0)]
    check("B", "flat terrain at h = 0, bxt = 300 m: SATA changes nothing (whole - no SATA)", abs(r["line_dB"]["whole"] - r["line_dB"]["none"]), 1e-6, unit="dB")
    check("B", "flat terrain at h = 0, bxt = 300 m: largest |phase| (the floor of one channel + its reference filter)", floor, 0.5, unit="deg")
    r = by[("flat500", 300.0)]
    check("B", "flat at 500 m: phase without SATA = -2 pi dC0 / wl (largest difference)",
          max(abs(np.degrees(np.angle(np.exp(1j * np.radians(q["phase_deg"] - e))))) for q, e in zip(r["targets"]["none"], r["expected_deg"])), 0.5, unit="deg")
    for k, name in (("whole", "whole band"), ("sub", "sub-band"), ("whole_footprint", "whole band, footprint map"), ("sub_footprint", "sub-band, footprint map")):
        check("B", f"flat at 500 m, {name}: largest |phase| left", ph(r, k), floor + 0.2, unit="deg")
        # the DEM table reaches the level of flat terrain; the footprint map only covers +- 1.2 km around every target
        check("B", f"flat at 500 m, {name}: output - reference", r["line_dB"][k], by[("flat0", 300.0)]["line_dB"]["none"] + 1.0 if "footprint" not in k else -30.0, unit="dB")
    for tname in ("flat500", "plateaus"):                           # not a check: the table of mode="lut" only holds the cells of the targets (+- 2)
        q = by[(tname, 300.0)]
        print(f"  (info) {tname}, map of mode='lut' (targets only, +- 2 cells): largest |phase| left {ph(q, 'whole_lut'):.1f} deg "
              f"(without SATA {ph(q, 'none'):.1f} deg)", flush=True)
    for bxt in (300.0, 100.0):
        r = by[("plateaus", bxt)]
        check("B", f"plateaus, bxt = {bxt:g} m (targets >= 1 km from the steps): largest |phase| left, whole band", ph(r, "whole"), 1.0, unit="deg")
        check("B", f"plateaus, bxt = {bxt:g} m: largest |phase| left, sub-band", ph(r, "sub"), 1.0, unit="deg")
    for r in res:
        if r["bxt"]:
            check("B", f"{r['terrain']}, bxt = {r['bxt']:g} m: whole band - sub-band (one channel: the same kernel)", r["whole_vs_sub_dB"], -55, unit="dB")
    return res, plots


# ================================================================================================ figures
def figures(R, plots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    REF, NON, WHO, SUB = "tab:green", "tab:red", "tab:blue", "tab:orange"
    pages = []

    def save(fig, name):
        f = os.path.join(OUT, name + ".png")
        fig.savefig(f, dpi=110)
        plt.close(fig)
        pages.append(f)

    if "A" in R:
        A = R["A"]
        rows = sorted(A["A3"], key=lambda r: r["x"])
        x = np.array([r["x"] for r in rows])
        fig, axs = plt.subplots(2, 2, figsize=(15, 8.6), gridspec_kw=dict(hspace=0.42, wspace=0.2, left=0.06, right=0.985, top=0.9, bottom=0.08))
        fig.suptitle(f"Kernel alone, one channel without folds (PRF = {A['PRF_op']:.0f} Hz, block of {1e3 * A['block_s']:.0f} ms): map with two plateaus "
                     "(step of 115$^\\circ$ at x = 0), one target at x", fontsize=12)
        ax = axs[0, 0]
        ax.plot(x, [r["before_deg"] for r in rows], "x", color=NON, ms=6, label="before the kernel (the phase put on the target)")
        ax.plot(x, [r["phase_deg"] for r in rows], "o-", color=WHO, ms=4, label="after the kernel")
        ax.set_ylabel("phase of the peak re. the reference [deg]"); ax.set_title("Phase of the focused peak", fontsize=10.5); ax.legend(fontsize=8.5, loc="center right")
        ax = axs[0, 1]
        ax.plot(x, [r["amp_dB"] for r in rows], "o-", color=WHO, ms=4)
        ax.set_ylabel("amplitude of the peak re. the reference [dB]"); ax.set_title("Amplitude of the focused peak after the kernel", fontsize=10.5)
        ax = axs[1, 0]
        d = np.abs(x)
        for side, mk in ((-1, "o"), (1, "s")):
            sel = np.sign(x) == side
            o = np.argsort(d[sel])
            ax.semilogx(d[sel][o], np.abs([r["phase_deg"] for r in rows])[sel][o], mk + "-", color=WHO, ms=4, mfc="w" if side < 0 else WHO,
                        label=f"target on the {'left' if side < 0 else 'right'} plateau")
        ax.axvline(A["sata_cell_m"], color="k", ls=":", lw=1); ax.text(A["sata_cell_m"] * 1.05, 20, "$\\sqrt{\\lambda r_0/2}$ = 310 m", fontsize=8.5)
        ax.axvline(2 * A["PRF_op"] / A["block"] * A["ruler_m_per_Hz"], color="k", ls="--", lw=1)
        ax.text(2 * A["PRF_op"] / A["block"] * A["ruler_m_per_Hz"] * 1.05, 6, "first null of the block\nspectrum: 623 m", fontsize=8.5)
        ax.set_yscale("symlog", linthresh=0.01); ax.set_ylim(0, 60)
        ax.set_xlabel("distance of the target to the step [m]"); ax.set_ylabel("|phase left| [deg]"); ax.set_title("Phase left against the distance to the step", fontsize=10.5)
        ax.legend(fontsize=8.5, loc="lower left")
        ax = axs[1, 1]
        names = list(A["A4"])
        cols = ["tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
        ok = [r for r in rows if abs(r["x"]) in (600, 1000, 2000, 4000)]
        ax.plot([r["x"] for r in ok], [abs(r["phase_deg"]) for r in ok], "o-", color=WHO, ms=5, lw=2, label="kernel of the package")
        for nm, c in zip(names, cols):
            rr = sorted(A["A4"][nm], key=lambda r: r["x"])
            ax.plot([r["x"] for r in rr], [max(abs(r["phase_deg"]), 0.01) for r in rr], "x--", color=c, ms=6, label="broken on purpose: " + nm)
        ax.set_yscale("log"); ax.set_ylim(0.003, 400); ax.set_ylabel("|phase left| [deg]"); ax.set_title("The same test with the kernel broken on purpose", fontsize=10.5)
        ax.legend(fontsize=7.6, loc="center", ncol=1)
        for a in (axs[0, 0], axs[0, 1], axs[1, 1]):
            a.set_xlabel("target position x [m]  (step at x = 0)")
        save(fig, "A_step")

        rows = A["A5"]
        g = np.array([r["rad_per_cell"] for r in rows])
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(wspace=0.24, left=0.06, right=0.985, top=0.84, bottom=0.14))
        fig.suptitle("Kernel alone, one channel without folds: map that is a straight line (a slope), one target", fontsize=12)
        sty = {"cola": ("o-", WHO, "scheme cola (the default)"), "crossfade": ("s--", SUB, "scheme crossfade"), "margin": ("^:", "tab:purple", "scheme margin")}
        for a, key, yl in zip(axs, ("amp_dB", "phase_deg", "side_dB"), ("amplitude of the peak re. the reference [dB]", "phase of the peak re. the reference [deg]",
                                                                         "largest difference outside the main lobe [dB re. the peak]")):
            for sc, (mk, c, lab) in sty.items():
                a.plot(g, [r[sc][key] for r in rows], mk, color=c, ms=5, label=lab)
            a.set_xlabel("change of the map phase over one SATA cell of 310 m [rad]"); a.set_ylabel(yl)
        pts = [(c, q) for r in R.get("B", []) if r["bxt"] and r["terrain"] in ("ramp2", "ramp8", "sine", "mountain") for c, q in zip(r["rad_per_cell"], r["targets"]["whole"])]
        if pts:
            axs[0].plot([c for c, _ in pts], [q["amp_dB"] for _, q in pts], "D", color="k", ms=5, mfc="none", ls="none", label="targets on the terrains of part B (DEM table)")
            axs[1].plot([c for c, _ in pts], [q["phase_deg"] for _, q in pts], "D", color="k", ms=5, mfc="none", ls="none")
        axs[0].legend(fontsize=8.5); axs[2].set_title("(paired echoes at $\\pm$623 m: one hop of 20 ms)", fontsize=9.5)
        save(fig, "A_slope")

    if "S" in R:
        Sx = R["S"]
        fig, axs = plt.subplots(1, 5, figsize=(17, 4.9), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.055, right=0.99, top=0.81, bottom=0.2))
        fig.suptitle("Kernels of the 4-channel system (PRF$_{op}$ = 500 Hz) on lines without folds: map with two plateaus (step of 115$^\\circ$ at x = 0), one target at x",
                     fontsize=12)
        for a, r in zip(axs, Sx["kernels"]):
            xx = [q["x"] for q in r["full"]]
            if "wrong_labels" in r:
                a.plot(xx, [abs(q["phase_deg"]) for q in r["wrong_labels"]], "x--", color=NON, ms=6, label="whole-band labels on this line")
            a.plot(xx, [abs(q["phase_deg"]) for q in r["full"]], "s-", color=SUB, ms=4, label="line filling the window")
            a.plot(xx, [abs(q["phase_deg"]) for q in r["guard"]], "o-", color=WHO, ms=4, label="line 100 Hz away from the edges")
            a.set_yscale("log"); a.set_ylim(0.003, 3000); a.set_xlabel("target position x [m]")
            a.set_title(f"{r['name']} kernel\nwindow {r['window'][0]:+.0f} .. {r['window'][1]:+.0f} Hz", fontsize=10)
            a.text(0.5, 0.965, f"signal in the window: {r['occupied'][0]:+.0f} .. {r['occupied'][1]:+.0f} Hz\naway from the edges: {r['inner'][0]:+.0f} .. {r['inner'][1]:+.0f} Hz",
                   transform=a.transAxes, ha="center", va="top", fontsize=7.6, color="0.3")
        axs[0].set_ylabel("|phase left| [deg]")
        h, l = axs[1].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
        save(fig, "S_subbands")

    for r in R.get("B", []):
        key = f"{r['terrain']}_{r['bxt']:g}"
        if key not in plots or key not in ("plateaus_0", "flat500_300", "plateaus_300", "ramp8_300", "mountain_300"):     # representative pages
            continue
        P = plots[key]
        terrain = TERRAINS[r["terrain"]][0]
        xs = np.array(r["xs"])
        tt = xs / 7688.53706432
        pk = P["ref"].max()
        db = lambda k: 20 * np.log10(P[k] / pk + 1e-12)
        fig = plt.figure(figsize=(17, 12.4))
        gs = GridSpec(3, 2, width_ratios=[1.5, 1.0], hspace=0.42, wspace=0.16, figure=fig, left=0.05, right=0.985, top=0.925, bottom=0.055)
        fig.suptitle(f"One receiver without folds (N$_{{rx}}$ = 1, PRF = 2000 Hz), b$_{{xt}}$ = {r['bxt']:g} m | {r['desc']} | DEM table (terrain_lookup_table)", fontsize=12)
        # ---- amplitude: one channel has no ambiguity, the terrain only turns the phase of every target
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(P["t"], db("ref"), color=REF, lw=2.4, label="Reference")
        ax.plot(P["t"], db("none"), color=NON, lw=1.5, ls=(0, (5, 3)), label="no SATA")
        ax.plot(P["t"], db("whole"), color=WHO, lw=1.2, label="SATA whole band")
        ax.plot(P["t"], db("sub"), color=SUB, lw=1.0, ls=(0, (1, 2)), label="SATA per sub-band")
        ax.set_xlim(tt.min() - 2.2, tt.max() + 2.2); ax.set_ylim(-90, 8); ax.set_xlabel("Azimuth time [s]"); ax.set_ylabel("[dB]")
        ax.set_title("Azimuth IRF, amplitude |IRF|: with one channel the terrain does not change it (it only turns the phase of every target)", fontsize=10.5)
        ax.legend(fontsize=8.5, loc="upper right")
        # ---- the complex difference to the reference: this is where no SATA and SATA separate
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(P["t"], db("err_none"), color=NON, lw=1.6, label="no SATA")
        ax.plot(P["t"], db("err_whole"), color=WHO, lw=1.3, label="SATA whole band")
        ax.plot(P["t"], db("err_sub"), color=SUB, lw=1.0, ls=(0, (1, 2)), label="SATA per sub-band")
        ax.set_xlim(tt.min() - 2.2, tt.max() + 2.2); ax.set_ylim(-90, 8); ax.set_xlabel("Azimuth time [s]"); ax.set_ylabel("[dB re. the peak of the Reference]")
        ax.set_title("Difference to the Reference with the phase, |IRF $-$ Reference| (0 dB = as large as the target itself)", fontsize=10.5)
        ax.legend(fontsize=8.5, loc="upper right")
        # ---- the phase of every target
        ax = fig.add_subplot(gs[2, 0])
        p_none = np.array([q["phase_deg"] for q in r["targets"]["none"]]); p_whole = np.array([q["phase_deg"] for q in r["targets"]["whole"]])
        for x_, a_, b_ in zip(xs, p_none, p_whole):
            if abs(a_ - b_) > 12:
                ax.annotate("", xy=(x_, b_ + np.sign(a_ - b_) * 7), xytext=(x_, a_), arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.3))
        ax.plot(xs, r["expected_deg"], "_", color="k", ms=24, mew=1.6, ls="none", label="$-2\\pi\\,\\Delta C_0/\\lambda$ of the target (wrapped)")
        ax.plot(xs, p_none, "X", color=NON, ms=11, mec="k", mew=0.5, ls="none", label="no SATA")
        ax.plot(xs, p_whole, "o", color=WHO, ms=9, mec="k", mew=0.5, ls="none", label="SATA whole band")
        ax.plot(xs, [q["phase_deg"] for q in r["targets"]["sub"]], "+", color=SUB, ms=12, mew=1.8, ls="none", label="SATA per sub-band")
        ax.axhline(0, color=REF, lw=2.0, alpha=0.6, zorder=0, label="Reference (0$^\\circ$)")
        ax.set_ylim(-200, 245); ax.set_yticks(np.arange(-180, 181, 60))
        ax.set_xlabel("target position x [m]"); ax.set_ylabel("phase of the peak re. the Reference [deg]")
        ax.set_title("Phase of the focused peak of every target (arrows: what SATA removes)", fontsize=10.5)
        ax.legend(fontsize=8.5, loc="upper center", ncol=5)
        # ---- the terrain and the targets
        ax = fig.add_subplot(gs[0, 1])
        xg = np.linspace(-9000, 9000, 1801)
        ax.fill_between(xg / 1e3, 0, terrain(xg), color="#D7C9A7", alpha=0.6, lw=0); ax.plot(xg / 1e3, terrain(xg), color="#6D4C2F", lw=1.2)
        ax.plot(xs / 1e3, r["hs"], "o", ms=7, color="#1F4E79", mec="k", ls="none", label="point targets")
        for j, (x_, h_) in enumerate(zip(xs, r["hs"])):
            ax.annotate(f"{h_:.0f} m", (x_ / 1e3, h_), xytext=(0, 9 if j % 2 == 0 else -16), textcoords="offset points", ha="center", fontsize=8.5, color="#1F4E79")
        ax.set_xlim(-9, 9); ax.set_ylim(top=1.18 * max(max(r["hs"]), 50.0)); ax.set_xlabel("azimuth position x [km]"); ax.set_ylabel("height [m]")
        ax.set_title("Terrain (the DEM) and targets", fontsize=10.5)
        ax.legend(fontsize=8.5, loc="upper left")
        # ---- what the kernel reads
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(P["x"] / 1e3, 1e3 * P["M"], color="k", lw=1.4, label="DEM table of the channel")
        ax.plot(xs / 1e3, 1e3 * np.array(r["dC0"]), "o", ms=6, color="#1F4E79", mec="k", ls="none", label="$\\Delta C_0$ of every target (full range history)")
        inside = np.abs(P["x"]) <= 9000
        lo, hi = 1e3 * min(P["M"][inside].min(), min(r["dC0"])), 1e3 * max(P["M"][inside].max(), max(r["dC0"]))
        pad = max(0.08 * (hi - lo), 20.0)
        ax.set_ylim(lo - pad, hi + pad); ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_xlim(-9, 9); ax.set_xlabel("azimuth position x [km]"); ax.set_ylabel("$\\Delta C_0$ [mm]"); ax.set_title("What the kernel reads", fontsize=10.5)
        ax.legend(fontsize=8.5)
        # ---- the numbers of every target
        ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
        rows = [[f"{x_ / 1e3:+.1f}", f"{h_:.0f}", f"{1e3 * d_:+.0f}", f"{a_['phase_deg']:+.1f}", f"{b_['phase_deg']:+.2f}", f"{c_['phase_deg']:+.2f}", f"{b_['amp_dB']:+.2f}"]
                for x_, h_, d_, a_, b_, c_ in zip(xs, r["hs"], r["dC0"], r["targets"]["none"], r["targets"]["whole"], r["targets"]["sub"])]
        tb = ax.table(cellText=rows, loc="center", cellLoc="center",
                      colLabels=["target\nx [km]", "height\n[m]", "$\\Delta C_0$\n[mm]", "no SATA\nphase\n[deg]", "SATA\nwhole band\nphase [deg]", "SATA per\nsub-band\nphase [deg]",
                                 "SATA\nwhole band\namplitude [dB]"])
        tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1.0, 1.85)
        for (i, j), cell in tb.get_celld().items():
            cell.set_edgecolor("0.75")
            if i == 0:
                cell.set_facecolor("0.93"); cell.set_height(cell.get_height() * 2.1); cell.get_text().set_fontsize(8.2)
            elif j == 3:
                cell.get_text().set_color(NON)
            elif j in (4, 6):
                cell.get_text().set_color(WHO)
            elif j == 5:
                cell.get_text().set_color(SUB)
        ax.set_title("Every target: phase and amplitude of the focused peak re. the Reference", fontsize=10.5)
        save(fig, "B_" + key)

    try:
        from PIL import Image
        ims = [Image.open(q).convert("RGB") for q in pages]
        ims[0].save(os.path.join(OUT, "sata_alone.pdf"), save_all=True, append_images=ims[1:], resolution=110, quality=85)
        print("wrote", os.path.join(OUT, "sata_alone.pdf"), f"({len(pages)} pages)")
    except ImportError:
        print("PIL not available: the PNG pages are in", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["A", "S", "B"])
    ap.add_argument("--step", type=int, default=16, help="cells between computed points of the DEM table (part B)")
    ap.add_argument("--fresh", action="store_true", help="part B: compute every case again instead of reading sata_alone/cache")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    R, plots, t0 = {}, {}, time.time()
    if "A" in a.parts:
        R["A"] = part_a()
    if "S" in a.parts:
        R["S"] = part_s()
    if "B" in a.parts:
        R["B"], plots = part_b(a.step, a.fresh)
    R["checks"] = CHECKS
    n_ok = sum(c["ok"] for c in CHECKS)
    print(f"{n_ok} PASS, {len(CHECKS) - n_ok} FAIL ({time.time() - t0:.0f} s)")
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    if not a.no_figures:
        figures(R, plots)


if __name__ == "__main__":
    main()
