# -*- coding: utf-8 -*-
"""
Case matrix for SATA 2-D: one figure and one metrics record per geometry.

WHAT THIS SCRIPT IS FOR
-----------------------
``run_irf2d.py`` shows ONE geometry at a time.  This driver sweeps a matrix of
geometries and, for each of them, produces the same six-panel figure so that
the cases can be laid side by side.  Two scene families are covered:

``single``   one point target at height ``h`` and azimuth ``x_az`` -- the
             cleanest possible probe of the topographic residual, because the
             only thing the reconstruction filter gets wrong is that target's
             own height.

``ramp``     five iso-range scatterers on a line inclined by ``alpha`` ALONG
             AZIMUTH:

                 h_j = h_ref + x_j * tan(alpha) ,   x_j in {-400 .. +400} m

             Each of them is placed on the iso-range surface
             ``y(h) = sqrt(r0^2 - (H-h)^2)`` so that they all fall in the SAME
             range bin.  That is the whole point of the construction: any
             difference between them cannot be blamed on the range-dependent
             part of the filter, it is purely topographic.  ``alpha`` is the
             free parameter, and LOW alpha is the interesting regime -- it is
             the realistic one (a 3 deg terrain slope is a gentle hill, not a
             cliff) and it is where the azimuth-varying part of the residual
             becomes small compared with its mean, which is exactly the regime
             in which one asks whether the per-sub-band machinery still earns
             its four-fold cost over a single whole-band pass.

             The default scene of the 1-D tests, ((-400,80) ... (400,400)), is
             this construction with ``h_ref = 240 m`` and
             ``alpha = atan(0.4) = 21.8 deg`` -- included here as the
             high-alpha reference case.

WHAT IS MEASURED
----------------
For every case the channels are reconstructed three ways -- no SATA, one
whole-band SATA pass, one SATA pass per output sub-band -- and each result is
focused with the exact 2-D matched filter of ``run_irf2d`` (the ideal
monostatic signal of the target of interest, so the range cell migration is
compensated without any approximate RCMC).  Reported per target:

``peak``   focused peak amplitude in % of the monostatic one -- the number
           that says how much of the target was recovered;
``IRW``    -3 dB impulse response width of the two 1-D cuts;
``PSLR``   peak-to-sidelobe ratio of the two cuts;
``ESR``    in-band error-to-signal ratio,
           ``rms|S_rec - S_ref| / rms|S_ref|`` over ``|f_a| < B_a/2`` [dB],
           a whole-image measure that does not depend on picking one target.

Run from ``sar_reconstruction/``::

    python -m sata2d.run_cases --list                 # show the matrix
    python -m sata2d.run_cases --only A2 B2           # a subset
    python -m sata2d.run_cases                        # everything (~15 min)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)

from .arrays import make_params_dpca, dpca_residual
from .geometry import (build_tracks_3d, generate_reference_3d,
                       generate_channels_3d, CoeffTable3D, residual_C0_3d)
from .reconstruction import (range_compress, build_delta_C0_map_3d,
                             reconstruct_subband_2d, scatterer_range,
                             range_bin_of, METHOD_KW)
from .run_irf2d import focus2d, zoom2d, cut_metrics

C_LIGHT = 3.0e8
HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots", "cases")

METHODS = ("no", "whole", "sub")
STYLE = {"mono":  dict(color="0.45", ls="--", lw=1.0, label="monostatic"),
         "no":    dict(color="#C62828", ls="-",  lw=1.0, label="no SATA"),
         "whole": dict(color="#EF6C00", ls="-",  lw=1.0, label="SATA whole band"),
         "sub":   dict(color="#2E7D32", ls="-",  lw=1.3, label="SATA per sub-band")}

RAMP_X = (-400.0, -200.0, 0.0, 200.0, 400.0)


# ---------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------
def ramp_specs(alpha_deg, h_ref, xs=RAMP_X):
    """Iso-range ramp of slope ``alpha_deg`` along azimuth about ``h_ref``."""
    t = np.tan(np.radians(alpha_deg))
    return tuple((float(x), float(h_ref + x * t)) for x in xs)


def _c(cid, kind, note, **kw):
    d = dict(id=cid, kind=kind, note=note, Nrx=4, bxt_max=100.0, seed=0,
             h=240.0, x_az=0.0, alpha=None, h_ref=240.0)
    d.update(kw)
    return d


CASES = [
    # ---- A: single point target ------------------------------------------
    _c("A1", "single", "reference geometry", h=240.0),
    _c("A2", "single", "low topography", h=100.0),
    _c("A3", "single", "high topography", h=400.0),
    _c("A4", "single", "small cross-track baselines", bxt_max=20.0),
    _c("A5", "single", "large cross-track baselines", bxt_max=300.0),
    _c("A6", "single", "two channels", Nrx=2),
    _c("A7", "single", "six channels", Nrx=6),
    _c("A8", "single", "target off the scene centre", x_az=400.0),
    # ---- B: iso-range ramp along azimuth ----------------------------------
    _c("B1", "ramp", "very gentle slope", alpha=1.0),
    _c("B2", "ramp", "gentle slope", alpha=3.0),
    _c("B3", "ramp", "moderate slope", alpha=6.0),
    _c("B4", "ramp", "steep slope (default scene)", alpha=21.8),
    _c("B5", "ramp", "gentle slope, large baselines", alpha=3.0, bxt_max=300.0),
    _c("B6", "ramp", "gentle slope, high plateau", alpha=3.0, h_ref=400.0),
    _c("B7", "ramp", "gentle slope, low plateau", alpha=3.0, h_ref=80.0),
    _c("B8", "ramp", "gentle slope, six channels", alpha=3.0, Nrx=6),
]
CASE_BY_ID = {c["id"]: c for c in CASES}


# ---------------------------------------------------------------------------
def residual_cycles(p, tr):
    """Topographic residual per scatterer, in wavelengths.

    ``residual_C0_3d`` is ``C0(true target) - C0(flat point at the same slant
    range)``: exactly what the flat-earth filter fails to model.  Divided by
    the wavelength it becomes the phase error the reconstruction carries, in
    cycles -- the single number that predicts how badly a case will behave.
    Returned per scatterer as the largest value over the channels, because
    what the reconstruction sees is the DISAGREEMENT between channels.
    """
    out = []
    for ptg in p.points:
        d = [residual_C0_3d(p, tr, ptg, i) / p.wl for i in range(p.Nrx)]
        out.append(dict(max_cycles=float(np.max(np.abs(d))),
                        spread_cycles=float(np.max(d) - np.min(d)),
                        per_channel=[float(v) for v in d]))
    return out


def build_case(case, res_rg=6.0, swath=300.0, dxt=150.0):
    """Parameters + the (dx_az, dh) list of the scene."""
    if case["kind"] == "single":
        specs = ((case["x_az"], case["h"]),)
    else:
        specs = ramp_specs(case["alpha"], case["h_ref"])
    p = make_params_dpca(Nrx=case["Nrx"], dxt=dxt, specs=specs,
                         res_rg=res_rg, swath=swath, bxt_mode="random",
                         bxt_max=case["bxt_max"], seed=case["seed"])
    return p, specs


def run_case(case, res_rg=6.0, swath=300.0, sata_osf=4, verbose=True):
    """Reconstruct, focus every scatterer, measure, and plot.  Returns a dict."""
    t0 = time.time()
    p, specs = build_case(case, res_rg=res_rg, swath=swath)
    tr = build_tracks_3d(p)

    rho_r = C_LIGHT / (2.0 * p.rsf)
    d_az = p.ve / p.prf
    res_az = p.La / 2.0
    rcm = (p.ve * p.int_time / 2.0) ** 2 / (2.0 * p.r0)

    pts = p.points                                    # [Np, 3]
    if verbose:
        print(f"\n=== {case['id']}  {case['kind']}  ({case['note']}) ===")
        print(f"    Nrx={p.Nrx}  PRF={p.prf:.0f} Hz  bxt_max={case['bxt_max']:.0f} m"
              f"  DPCA residual {dpca_residual(p):.1e}")
        print(f"    bxt = {np.array2string(p.bxt, precision=1)} m")
        print(f"    scatterers: " + ", ".join(
            f"(x={s[0]:+.0f} m, h={s[1]:.0f} m)" for s in specs))

    # ---- data ------------------------------------------------------------
    ref_scene = range_compress(generate_reference_3d(p, tr),
                               p.cd, p.rbw, p.rsf, axis=1)
    ch = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)

    tab = CoeffTable3D(p, tr, n_nodes=16)
    hw = int(np.ceil(rcm / (2.0 * rho_r))) + 1
    maps = [build_delta_C0_map_3d(p, tr, i, range_halfwidth=hw)
            for i in range(p.Nrx)]

    rec = {}
    for m in METHODS:
        rec[m] = reconstruct_subband_2d(p, tr, ch, tab, sata_osf=sata_osf,
                                        maps=maps, verbose=False,
                                        **METHOD_KW[m])

    # ---- whole-image error-to-signal ------------------------------------
    S_ref = np.fft.fftshift(np.fft.fft2(ref_scene))
    fa = np.fft.fftshift(np.fft.fftfreq(p.Na, 1.0 / p.prf))
    inband = (np.abs(fa) < p.abw / 2.0)[:, None] & np.ones((1, p.Nr), bool)
    s = np.abs(S_ref)[inband]
    esr = {}
    for m in METHODS:
        e = np.abs(np.fft.fftshift(np.fft.fft2(rec[m])) - S_ref)[inband]
        esr[m] = float(20.0 * np.log10(np.sqrt(np.mean(e ** 2))
                                       / np.sqrt(np.mean(s ** 2))))

    # ---- per-target focusing --------------------------------------------
    span = 4.5 * max(res_az, C_LIGHT / 2.0 / p.rbw)
    na = max(8, int(np.ceil(3.0 * span / d_az)))
    nr = max(8, int(np.ceil(3.0 * span / rho_r)))
    zpa = max(1, int(np.ceil(24.0 * d_az / span)))
    zpr = max(1, int(np.ceil(24.0 * rho_r / span)))
    x_az = (np.arange(2 * na * zpa) - na * zpa) * d_az / zpa
    x_rg = (np.arange(2 * nr * zpr) - nr * zpr) * rho_r / zpr
    ca, cr = len(x_az) // 2, len(x_rg) // 2

    targets = []
    zoom_of = {}                      # panels of the representative target
    for j, ptg in enumerate(pts):
        ref_one = range_compress(generate_reference_3d(p, tr, ptgs=ptg[None, :]),
                                 p.cd, p.rbw, p.rsf, axis=1)
        img = {"mono": focus2d(ref_scene, ref_one)}
        for m in METHODS:
            img[m] = focus2d(rec[m], ref_one)
        # The zero lag of the matched filter is the centre of the fftshifted
        # array, exactly.  Do NOT locate it with argmax: on an iso-range ramp
        # the four other scatterers are compressed by the same chirp and
        # produce peaks of comparable amplitude in the SAME range bin, 200 m
        # away in azimuth, and argmax happily locks onto a neighbour's.
        ia, ir = img["mono"].shape[0] // 2, img["mono"].shape[1] // 2
        z = {k: zoom2d(v, ia, ir, na, nr, zpa, zpr) for k, v in img.items()}
        pm = float(np.abs(z["mono"]).max())

        entry = dict(j=j, x_az=float(specs[j][0]), h=float(specs[j][1]),
                     range_bin=range_bin_of(p, scatterer_range(p, ptg)))
        for k in ("mono",) + METHODS:
            pk = float(np.abs(z[k]).max())
            irw_a, pslr_a = cut_metrics(z[k][:, cr], x_az)
            irw_r, pslr_r = cut_metrics(z[k][ca, :], x_rg)
            entry[k] = dict(peak=pk, pct=100.0 * pk / pm,
                            irw_az=float(irw_a), pslr_az=float(pslr_a),
                            irw_rg=float(irw_r), pslr_rg=float(pslr_r))
        targets.append(entry)
        zoom_of[j] = z

    # representative target: the one the sub-band pass recovers worst
    jrep = int(np.argmin([t["sub"]["pct"] for t in targets]))

    out = dict(
        case=dict(case), specs=[list(map(float, s)) for s in specs],
        geometry=dict(Nrx=int(p.Nrx), prf=float(p.prf), PRF_op=float(p.PRF_op),
                      abw=float(p.abw), rbw=float(p.rbw), Na=int(p.Na),
                      Nr=int(p.Nr), res_rg=float(C_LIGHT / 2.0 / p.rbw),
                      res_az=float(res_az), rho_r=float(rho_r),
                      d_az=float(d_az), rcm=float(rcm),
                      theta_inc=float(np.degrees(p.theta_inc)),
                      r0=float(p.r0), bat=[float(v) for v in p.bat],
                      bxt=[float(v) for v in p.bxt],
                      dpca_residual=float(dpca_residual(p))),
        esr=esr, targets=targets, jrep=jrep,
        residual=residual_cycles(p, tr),
        runtime=float(time.time() - t0))

    _plot_case(out, zoom_of[jrep], x_az, x_rg, ca, cr, span, targets)

    if verbose:
        print(f"    {'target':>8} {'mono':>8} {'no SATA':>9} {'whole':>9} {'sub':>9}")
        for t in targets:
            print(f"    x={t['x_az']:+6.0f} m {100.0:7.1f}% {t['no']['pct']:8.1f}%"
                  f" {t['whole']['pct']:8.1f}% {t['sub']['pct']:8.1f}%")
        print(f"    ESR in-band: no {esr['no']:+6.2f} dB | whole {esr['whole']:+6.2f}"
              f" dB | sub {esr['sub']:+6.2f} dB     [{out['runtime']:.0f} s]")
    return out


# ---------------------------------------------------------------------------
def _plot_case(out, z, x_az, x_rg, ca, cr, span, targets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    case, g = out["case"], out["geometry"]
    trep = targets[out["jrep"]]
    nrm = float(np.abs(z["mono"]).max())

    def db(v):
        return 20.0 * np.log10(np.abs(v) / nrm + 1e-20)

    levels = [-30, -25, -20, -16, -13, -10, -6, -3]
    ramp = case["kind"] == "ramp"
    fig = plt.figure(figsize=(13.2, 9.1 if ramp else 7.0), dpi=150)
    gs = gridspec.GridSpec(3 if ramp else 2, 3, width_ratios=[1.0, 1.0, 1.25],
                           height_ratios=[1.0, 1.0, 0.62] if ramp else [1.0, 1.0],
                           hspace=0.40 if ramp else 0.36, wspace=0.30)

    for col, key in ((0, "mono"), (1, "no")):
        ax = fig.add_subplot(gs[0, col])
        ax.contour(x_rg, x_az, db(z[key]), levels=levels, cmap="Spectral_r",
                   linewidths=1.0, vmin=-34, vmax=0)
        ax.set_title(f"{STYLE[key]['label']}  ({trep[key]['pct']:.1f} % of mono)",
                     fontsize="small")
        ax.set_xlabel("Range [m]"); ax.set_ylabel("Azimuth [m]")
        ax.set_xlim(-span, span); ax.set_ylim(-span, span)
        ax.set_aspect("equal"); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.contour(x_rg, x_az, db(z["whole"]), levels=levels, cmap="Spectral_r",
               linewidths=1.0, vmin=-34, vmax=0)
    ax.set_title(f"SATA whole band  ({trep['whole']['pct']:.1f} % of mono)",
                 fontsize="small")
    ax.set_xlabel("Range [m]"); ax.set_ylabel("Azimuth [m]")
    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    ax.set_aspect("equal"); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.contour(x_rg, x_az, db(z["sub"]), levels=levels, cmap="Spectral_r",
               linewidths=1.0, vmin=-34, vmax=0)
    ax.set_title(f"SATA per sub-band  ({trep['sub']['pct']:.1f} % of mono)",
                 fontsize="small")
    ax.set_xlabel("Range [m]"); ax.set_ylabel("Azimuth [m]")
    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    ax.set_aspect("equal"); ax.grid(alpha=.3)

    for row, (xax, lab) in enumerate(
            ((x_az, "Azimuth [m]"), (x_rg, "Range [m]"))):
        ax = fig.add_subplot(gs[row, 2])
        for k in ("mono",) + METHODS:
            cut = z[k][:, cr] if row == 0 else z[k][ca, :]
            ax.plot(xax, db(cut), **STYLE[k])
        ax.set_xlabel(lab); ax.set_ylabel("Impulse response [dB]")
        ax.set_xlim(-2 * span, 2 * span); ax.set_ylim(-45, 3)
        ax.grid(alpha=.3)
        if row == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=.9)
        ax.set_title("azimuth cut" if row == 0 else "range cut", fontsize="small")

    if ramp:
        ax = fig.add_subplot(gs[2, :])
        xs = [t["x_az"] for t in targets]
        for k in METHODS:
            st = dict(STYLE[k]); st.pop("ls")
            ax.plot(xs, [t[k]["pct"] for t in targets], marker="o", ms=4.5,
                    **st)
        ax.axhline(100.0, **STYLE["mono"])
        ax.set_xlabel("target along-track position $x$ [m]"
                      "      (annotations: target height $h$ [m])")
        ax.set_ylabel("focused peak\n[% of monostatic]")
        ax.set_xticks(xs)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=4, loc="lower left", framealpha=.9)
        lo = min(min(t[k]["pct"] for k in METHODS) for t in targets)
        ax.set_ylim(max(0.0, lo - 8.0), 106.0)
        for t in targets:
            ax.annotate(f"{t['h']:.0f}", (t["x_az"], 104.0), ha="center",
                        va="top", fontsize=7, color="0.35")
        ax.set_title("peak recovery along the ramp", fontsize="small")

    scene = (f"single target: $h$ = {case['h']:.0f} m, $x$ = {case['x_az']:.0f} m"
             if case["kind"] == "single" else
             f"iso-range ramp: $\\alpha$ = {case['alpha']:.1f}$^\\circ$, "
             f"$h_{{ref}}$ = {case['h_ref']:.0f} m, 5 targets")
    fig.suptitle(
        f"{case['id']} -- {case['note']}  |  {scene}\n"
        f"$N_{{rx}}$={g['Nrx']} | PRF={g['prf']:.0f} Hz | "
        f"$b_{{xt}}\\sim U(0,{case['bxt_max']:.0f}$ m$)$ | "
        f"$\\delta_r$={g['res_rg']:.1f} m, $\\delta_x$={g['res_az']:.1f} m | "
        f"ESR: no {out['esr']['no']:+.1f} / whole {out['esr']['whole']:+.1f} / "
        f"sub {out['esr']['sub']:+.1f} dB"
        + (f"  |  panels show the target at $x$={trep['x_az']:+.0f} m, "
           f"$h$={trep['h']:.0f} m" if case["kind"] == "ramp" else ""),
        fontsize="medium")
    fig.subplots_adjust(left=0.055, right=0.985,
                        top=0.90 if ramp else 0.87,
                        bottom=0.06 if ramp else 0.075)

    os.makedirs(PLOTS, exist_ok=True)
    path = os.path.join(PLOTS, f"case_{case['id']}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out["figure"] = os.path.relpath(path, HERE)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="case ids to run")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    ap.add_argument("--res-rg", type=float, default=6.0, dest="res_rg")
    ap.add_argument("--swath", type=float, default=300.0)
    ap.add_argument("--sata-osf", type=int, default=4, dest="sata_osf")
    ap.add_argument("--out", default=os.path.join(PLOTS, "cases.json"))
    args = ap.parse_args(argv)

    if args.list:
        for c in CASES:
            extra = (f"h={c['h']:.0f} m, x={c['x_az']:.0f} m"
                     if c["kind"] == "single"
                     else f"alpha={c['alpha']:.1f} deg, h_ref={c['h_ref']:.0f} m")
            print(f"  {c['id']:>3}  {c['kind']:<7} Nrx={c['Nrx']} "
                  f"bxt_max={c['bxt_max']:>5.0f} m  {extra:<34} {c['note']}")
        return 0

    ids = args.only or [c["id"] for c in CASES]
    results = []
    for cid in ids:
        results.append(run_case(CASE_BY_ID[cid], res_rg=args.res_rg,
                                swath=args.swath, sata_osf=args.sata_osf))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # merge with anything already there, so --only runs accumulate
    old = {}
    if os.path.exists(args.out):
        try:
            old = {r["case"]["id"]: r for r in json.load(open(args.out))}
        except Exception:
            old = {}
    old.update({r["case"]["id"]: r for r in results})
    ordered = [old[c["id"]] for c in CASES if c["id"] in old]
    with open(args.out, "w") as fh:
        json.dump(ordered, fh, indent=1)
    print(f"\nmetrics written to {args.out}  ({len(ordered)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
