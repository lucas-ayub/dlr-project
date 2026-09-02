# -*- coding: utf-8 -*-
"""
2-D SATA + azimuth reconstruction with a full 3-D geometry.

Geometry and numbers reproduce the existing 1-D SATA tests
(``runs/core/run_sata.py``): wl = 0.25 m, H = 720 km, r0 = 766.21 km
(theta_inc = 20 deg), da = 24*wl, La = 2*da, PRF = 2000 Hz, along-track
spacing 100 m -- so the 2-D results are directly comparable.

What it runs
------------
[1] KERNEL SELF-TEST   dC0 = 0 must reproduce the input (COLA); a constant
                       dC0 must apply exactly -2*pi*dC0/wl.
[2] RESIDUAL TABLE     the exact dC0 of Eq. (1) against the closed-form
                       interferometric estimate -b_perp*dh/(r0 sin theta).
[3] SWEEP              one elevated iso-range target, reconstructed four ways:
                       no-SATA / SATA whole-band / SATA per sub-band / ideal
                       (the filter told the true height).  Peak amplitudes,
                       normalised to the ideal.
[4] AZIMUTH TOPOGRAPHY five targets at different azimuth positions and
                       different heights -- the case a single global
                       correction cannot fix.
[5] PLOTS

Run
---
    cd sar_reconstruction
    python -m sata2d.run_sata2d_topo --plots
    python -m sata2d.run_sata2d_topo --quick        # fewer sweep cases
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "sata2d"

from .params3d import make_params3d, iso_range_offset          # noqa: E402
from .geom3d import (build_tracks_3d, CoeffTable3D,            # noqa: E402
                     residual_C0_3d, dC0_approx)
from .datagen3d import generate_reference_3d, generate_channels_3d  # noqa: E402
from .rangecomp import range_compress                          # noqa: E402
from .sata2d import sata_1d                                    # noqa: E402
from .sata3d import (build_delta_C0_map_3d, reconstruct_subband_2d,  # noqa: E402
                     range_bin_of, scatterer_range)

RULE = "=" * 78


def hdr(t):
    print(f"\n{RULE}\n{t}\n{RULE}")


# ---------------------------------------------------------------------------
def matched_filter(s, ref):
    """Pure correlation of s against ref, centred -- as in sar_recon.analysis."""
    Na = len(ref)
    return np.roll(np.fft.ifft(np.fft.fft(s) * np.conjugate(np.fft.fft(ref))),
                   int(Na / 2))


def target_range_bin(p, ptg):
    return range_bin_of(p, scatterer_range(p, ptg))


# ---------------------------------------------------------------------------
def test1_kernel(p):
    hdr("[1] SATA kernel self-test")
    rng = np.random.default_rng(0)
    naz = p.Na_ch
    t = np.arange(naz)
    x = (np.exp(2j * np.pi * 0.13 * t) + 0.5 * np.exp(-2j * np.pi * 0.07 * t)
         + 0.1 * (rng.standard_normal(naz) + 1j * rng.standard_normal(naz)))

    y0 = sata_1d(x, np.zeros(naz), rref=p.r0, prf_data=p.PRF_op, v=p.vs,
                 wl=p.wl, r=p.r0, f_centre=0.0)
    id_err = float(np.max(np.abs(y0[200:-200] - x[200:-200])))

    D = 0.01
    yC = sata_1d(x, np.full(naz, D), rref=p.r0, prf_data=p.PRF_op, v=p.vs,
                 wl=p.wl, r=p.r0, f_centre=0.0)
    applied = float(np.degrees(np.angle(np.mean((yC / x)[200:-200]))))
    expected = float(np.degrees(np.angle(np.exp(-2j * np.pi / p.wl * D))))
    print(f"  identity (dC0 = 0)      : max |err| = {id_err:.2e}   "
          f"{'PASS' if id_err < 1e-9 else 'FAIL'}")
    print(f"  constant dC0 = {D*1e3:.0f} mm    : applied {applied:+.3f} deg, "
          f"expected {expected:+.3f} deg   "
          f"{'PASS' if abs(applied-expected) < 1e-6 else 'FAIL'}")


def test2_residual(p, tracks):
    hdr("[2] Topographic residual dC0 -- exact vs closed form")
    print("  exact  : dC0 = C0(true target) - C0(flat point at same range)")
    print("  approx : dC0 ~ -b_perp*dh / (r0 sin(theta_inc))\n")
    print(f"  {'ch':>3} {'bat [m]':>8} {'bxt [m]':>8} | "
          f"{'dh [m]':>7} {'exact [mm]':>11} {'approx [mm]':>12} "
          f"{'phase [deg]':>12}")
    for dh in (100.0, 240.0, 400.0):
        off = iso_range_offset(p.r0, p.H, p.h0, dh)
        ptg = p.ptg + np.array(off)
        for i in range(p.Nrx):
            ex = residual_C0_3d(p, tracks, ptg, i)
            ap = dC0_approx(p, i, dh)
            print(f"  {i:3d} {p.bat[i]:8.1f} {p.bxt[i]:8.1f} | "
                  f"{dh:7.1f} {ex*1e3:11.2f} {ap*1e3:12.2f} "
                  f"{360*ex/p.wl:12.1f}")
        print()


# ---------------------------------------------------------------------------
def _run_case(Nrx, dx, dxt, dh, bxt_mode="linear", bxt_max=None, seed=0,
              sata_osf=4, verbose=False, n_nodes=16):
    """One elevated iso-range target, reconstructed four ways."""
    p = make_params3d(Nrx=Nrx, dx=dx, dxt=dxt, bxt_mode=bxt_mode,
                      bxt_max=bxt_max, seed=seed, specs=((0.0, dh),))
    tr = build_tracks_3d(p)
    ptg = p.points[0]
    nb = target_range_bin(p, ptg)

    ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)
    ch = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
    ref_line = ref[:, nb]

    tab_flat = CoeffTable3D(p, tr, n_nodes=n_nodes)            # flat earth
    tab_true = CoeffTable3D(p, tr, n_nodes=n_nodes, height=dh)  # knows the height

    maps = [build_delta_C0_map_3d(p, tr, i) for i in range(p.Nrx)]
    out = {}
    for tag, kw, tab in (("no",   dict(use_sata=False),        tab_flat),
                         ("whole", dict(use_sata="whole"),     tab_flat),
                         ("sub",  dict(use_sata="subband"),    tab_flat),
                         ("ideal", dict(use_sata=False),       tab_true)):
        rec = reconstruct_subband_2d(p, tr, ch, tab, sata_osf=sata_osf,
                                     maps=maps, verbose=verbose, **kw)
        out[tag] = float(np.max(np.abs(matched_filter(rec[:, nb], ref_line))))
    return p, out


def test3_sweep(p0, quick=False):
    hdr("[3] Single elevated iso-range target: no-SATA / SATA / ideal")
    print("  focused peak amplitude, normalised to the ideal reconstruction "
          "(100 % = perfect)\n")
    cases = [
        ("dxt=50   dh=240",  dict(Nrx=4, dx=100, dxt=50.0,  dh=240.0)),
        ("dxt=150  dh=240",  dict(Nrx=4, dx=100, dxt=150.0, dh=240.0)),
        ("dxt=300  dh=240",  dict(Nrx=4, dx=100, dxt=300.0, dh=240.0)),
    ]
    if not quick:
        cases += [
            ("dxt=150  dh=400",  dict(Nrx=4, dx=100, dxt=150.0, dh=400.0)),
            ("bxt rand<=100",    dict(Nrx=4, dx=100, dxt=100.0, dh=240.0,
                                      bxt_mode="random", bxt_max=100.0, seed=0)),
            ("bxt rand<=20",     dict(Nrx=4, dx=100, dxt=20.0,  dh=240.0,
                                      bxt_mode="random", bxt_max=20.0, seed=0)),
            ("Nrx=2   dxt=150",  dict(Nrx=2, dx=100, dxt=150.0, dh=240.0)),
            ("Nrx=6   dxt=150",  dict(Nrx=6, dx=100, dxt=150.0, dh=240.0)),
            ("DPCA dx=11",       dict(Nrx=4, dx=11,  dxt=150.0, dh=240.0)),
        ]
    print(f"  {'case':<18} {'max|dC0| [deg]':>15} {'no-SATA':>9} "
          f"{'SATA whole':>11} {'SATA sub-b':>11} {'ideal':>7}")
    rows = []
    for name, kw in cases:
        t0 = time.time()
        p, r = _run_case(**kw)
        tr = build_tracks_3d(p)
        worst = max(abs(residual_C0_3d(p, tr, p.points[0], i))
                    for i in range(p.Nrx))
        f = lambda k: 100.0 * r[k] / r["ideal"]
        print(f"  {name:<18} {360*worst/p.wl:15.0f} {f('no'):8.1f}% "
              f"{f('whole'):10.1f}% {f('sub'):10.1f}% {100.0:6.1f}%"
              f"   ({time.time()-t0:.0f}s)")
        rows.append((name, 360 * worst / p.wl, f('no'), f('whole'), f('sub')))
    return rows


# ---------------------------------------------------------------------------
def test4_azimuth_topo(sata_osf=4, verbose=False):
    hdr("[4] Azimuth-varying topography (position-dependent correction)")
    specs = ((-400.0, 80.0), (-200.0, 160.0), (0.0, 240.0),
             (200.0, 320.0), (400.0, 400.0))
    p = make_params3d(Nrx=4, dx=100.0, dxt=150.0, specs=specs)
    tr = build_tracks_3d(p)
    print(f"  5 iso-range targets, azimuth {[s[0] for s in specs]} m, "
          f"heights {[s[1] for s in specs]} m")
    nb = target_range_bin(p, p.points[0])
    print(f"  all in range bin {nb}\n")

    ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)
    ch = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
    # Focus with a SINGLE-target reference, otherwise the matched filter returns
    # the autocorrelation of the scene (spurious peaks at target-to-target lags).
    ref1 = range_compress(
        generate_reference_3d(p, tr, ptgs=p.ptg[None, :]), p.cd, p.rbw, p.rsf, axis=1)
    ref_line = ref1[:, nb]

    tab = CoeffTable3D(p, tr, n_nodes=16)
    maps = [build_delta_C0_map_3d(p, tr, i, verbose=True) for i in range(p.Nrx)]

    res = {}
    for tag, kw in (("no", dict(use_sata=False)),
                    ("whole", dict(use_sata="whole")),
                    ("sub", dict(use_sata="subband"))):
        t0 = time.time()
        rec = reconstruct_subband_2d(p, tr, ch, tab, sata_osf=sata_osf,
                                     maps=maps, verbose=verbose, **kw)
        res[tag] = matched_filter(rec[:, nb], ref_line)
        print(f"  {tag:>6}: peak {np.max(np.abs(res[tag])):.3e} "
              f"({time.time()-t0:.0f}s)")
    res["mono"] = matched_filter(ref[:, nb], ref_line)
    pm = np.max(np.abs(res["mono"]))
    print(f"\n  {'':>6}  {'peak':>10}  {'% of monostatic':>16}")
    for tag in ("mono", "no", "whole", "sub"):
        v = float(np.max(np.abs(res[tag])))
        print(f"  {tag:>6}  {v:10.3e}  {100*v/pm:15.1f}%")
    print("\n  With topography that varies ALONG AZIMUTH the per-sub-band\n"
          "  correction is genuinely better than the whole-band one: each\n"
          "  sub-band's frequency window maps onto a different piece of the\n"
          "  ground, so the correction becomes position-selective.")
    return p, res, specs


# ---------------------------------------------------------------------------
def make_plots(outdir, sweep_rows, p4, res4, specs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    # --- (a) sweep bar chart ----------------------------------------------
    if sweep_rows:
        names = [r[0] for r in sweep_rows]
        y = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 2.0))
        ax.barh(y - 0.22, [r[2] for r in sweep_rows], 0.2, label="no SATA",
                color="#C62828")
        ax.barh(y, [r[3] for r in sweep_rows], 0.2, label="SATA whole band",
                color="#1F4E79")
        ax.barh(y + 0.22, [r[4] for r in sweep_rows], 0.2, label="SATA per sub-band",
                color="#2E7D32")
        ax.axvline(100, color="k", ls="--", lw=1, label="ideal")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("focused peak, % of the ideal reconstruction")
        ax.set_xlim(0, 118); ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title("SATA recovers the peak the flat-earth filter loses")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "sata3d_sweep.png"), dpi=140)

    # --- (b) azimuth-topography IRF ---------------------------------------
    Na = p4.Na
    ds = p4.vs / p4.prf
    x = (np.arange(Na) - Na / 2) * ds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    order = (("mono", "monostatic reference", "#444444"),
             ("no", "no SATA", "#C62828"),
             ("whole", "SATA whole band", "#1F4E79"),
             ("sub", "SATA per sub-band", "#2E7D32"))
    pk = np.max(np.abs(res4["mono"]))
    for key, lab, col in order:
        db = 20 * np.log10(np.abs(res4[key]) / pk + 1e-20)
        axes[0].plot(x, db, label=lab, color=col, lw=1.1)
        axes[1].plot(x, db, label=lab, color=col, lw=1.1)
    for dxm, dh in specs:
        for ax in axes:
            ax.axvline(dxm, color="#999999", ls=":", lw=0.8)
    axes[0].set_xlim(-700, 700); axes[0].set_ylim(-40, 3)
    axes[0].set_title("five targets, heights 80..400 m")
    axes[1].set_xlim(-120, 120); axes[1].set_ylim(-40, 3)
    axes[1].set_title("zoom on the central target (dh = 240 m)")
    for ax in axes:
        ax.set_xlabel("azimuth [m]"); ax.set_ylabel("normalised [dB]")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Azimuth-varying topography: what SATA recovers", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sata3d_azimuth_topo.png"), dpi=140)

    # --- (c) residual map --------------------------------------------------
    tr = build_tracks_3d(p4)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i in range(p4.Nrx):
        dmap, act = build_delta_C0_map_3d(p4, tr, i)
        prof = dmap[act[len(act) // 2]] if act else np.zeros(p4.Na_ch)
        xs = (np.arange(p4.Na_ch) - p4.Na_ch / 2) * (p4.vs / p4.PRF_op)
        ax.plot(xs, 360 * prof / p4.wl,
                label=f"ch {i}: bat={p4.bat[i]:.0f} m, bxt={p4.bxt[i]:+.0f} m")
    ax.set_xlim(-700, 700)
    ax.set_xlabel("azimuth position [m]")
    ax.set_ylabel(r"residual phase $-360\,\Delta C_0/\lambda$ [deg]")
    ax.set_title("The residual SATA must remove, per channel")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sata3d_residual.png"), dpi=140)
    print(f"\nfigures written to {outdir}/")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer sweep cases")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--outdir", default="plots/sata2d")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args(argv)

    p = make_params3d()
    print(p.summary())
    tr = build_tracks_3d(p)

    test1_kernel(p)
    test2_residual(p, tr)
    rows = [] if args.skip_sweep else test3_sweep(p, quick=args.quick)
    p4, res4, specs = test4_azimuth_topo()

    if args.plots:
        make_plots(args.outdir, rows, p4, res4, specs)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
