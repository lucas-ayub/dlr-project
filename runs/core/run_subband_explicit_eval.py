# -*- coding: utf-8 -*-
r"""
Evaluate the EXPLICIT per-sub-band SATA reconstruction and compare it against
the other methods, to check whether fixing the issues Nida pointed out (the
per-sub-band f -> beta -> x_i frequency-axis rebuild, i.e. the Nyquist-edge
degeneracy of the old squint-based axis for even Nrx) makes the new sub-band
model actually work, and how it stacks up against whole-band SATA.

Methods compared per (Nrx, bxt):
    no-SATA        : plain multichannel reconstruction, no topography correction
    SATA band      : whole-band SATA (sata_channels + reconstruct)
    sub OLD        : old per-sub-band SATA (squint-based axis)  [the buggy one]
    sub EXPLICIT   : new per-sub-band SATA (explicit fftfreq axis) [the fix]

Metric: worst azimuth-ambiguity level [dB] of the focused point response
(lower = better). Also reports the peak focus [% of ideal].

Outputs:
    - console table
    - <plots_dir>/run_subband_explicit_eval/results.json  (for the LaTeX report)
    - <plots_dir>/run_subband_explicit_eval/gap_vs_nrx.png (if matplotlib avail)

Run (from sar_reconstruction/):
    PYTHONPATH=. python ../runs/core/run_subband_explicit_eval.py
    PYTHONPATH=. python ../runs/core/run_subband_explicit_eval.py --nrx 2 3 4 5 6 7 --bxt 10 30 50
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os

import numpy as np

import sar_recon as sar
from sar_recon.config import (SystemParams, Scene, ArrayGeometry,
                              prf_from_dpca, integration_time, build_time_axis)
from sar_recon.signal_model import getRawData1D
from sar_recon.sata import sata_channels
from sar_recon.subband_recon import reconstruct_subband
from sar_recon.subband_sata_explicit import reconstruct_subband_explicit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots", SCRIPT_NAME)

_DX_DPCA = 11.0
_RDELAY_SCENE = 0.0051115753
# Azimuth-varying topography scene: 5 targets at different azimuth AND height,
# the regime where per-sub-band correction is supposed to matter.
AZIMUTH_SPECS = ((-400, 80), (-200, 160), (0, 240), (200, 320), (400, 400))

METHODS = ["no-SATA", "SATA band", "sub OLD", "sub EXPLICIT"]


def build_scene(Nrx, bxt_max, seed=0):
    system = SystemParams()
    base = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0)
    r0, H, y0 = base.r0, base.H, base.y0
    extra = tuple((float(dx), float(np.sqrt(r0 ** 2 - (H - dh) ** 2) - y0), float(dh))
                  for dx, dh in AZIMUTH_SPECS)
    scene = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0, extra_offsets=extra)
    array = ArrayGeometry.linear(Nrx, _DX_DPCA, dxt=0.0, bxt_mode="random",
                                 bxt_max=bxt_max, rng=seed)
    prf, PRF_op = prf_from_dpca(system, Nrx, _DX_DPCA)
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.0 * integration_time(system, scene))
    cfg = sar.ExperimentConfig(name=f"Nrx{Nrx}_bxt{int(bxt_max)}", system=system,
                               scene=scene, array=array, prf=prf, PRF_op=PRF_op,
                               Na=Na, Na_ch=Nc, ta=ta, plots_dir=None)
    return cfg, sar.build_platform_tracks(cfg)


def channels_and_ref(cfg, tracks):
    s = cfg.system
    ptgs = cfg.scene.points[1:]
    sref1 = getRawData1D(cfg.scene.ptg[None, :], tracks.ptx, tracks.ptx, tracks.vtx,
                         tracks.vtx, cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx,
                         cfg.theta_tx, s.wl, cfg.prf)
    sig_true = getRawData1D(ptgs, tracks.ptx, tracks.ptx, tracks.vtx, tracks.vtx,
                            cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx,
                            s.wl, cfg.prf)
    s_ch = np.zeros([cfg.Nrx, cfg.Na_ch], complex)
    for i in range(cfg.Nrx):
        s_ch[i] = getRawData1D(ptgs, tracks.ptx, tracks.prx[i], tracks.vtx,
                               tracks.vrx[i], cfg.ta, cfg.sq_tx, cfg.sq_tx,
                               cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)[::cfg.Nrx]
    return sref1, sig_true, s_ch


def focus_mag(sig, ref):
    S = np.fft.fft(sig) * np.conj(np.fft.fft(ref))
    return np.abs(np.roll(np.fft.ifft(S), len(ref) // 2))


def ambiguity_db(f, mask=100):
    f = f / f.max()
    i0 = int(np.argmax(f))
    m = np.ones(len(f), bool)
    m[max(0, i0 - mask):i0 + mask] = False
    return 20.0 * np.log10(f[m].max())


def _quiet(fn, *a, **k):
    """Run fn silencing the library's stray debug prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def compare_one(Nrx, bxt):
    cfg, tr = build_scene(Nrx, bxt)
    sref1, sig_true, s_ch = channels_and_ref(cfg, tr)
    p = focus_mag(sig_true, sref1).max()

    recs = {
        "no-SATA": _quiet(sar.reconstruct, cfg, tr, s_ch.copy()),
        "SATA band": _quiet(sar.reconstruct, cfg, tr,
                            _quiet(sata_channels, cfg, tr, s_ch.copy(), verbose=False)),
        "sub OLD": _quiet(reconstruct_subband, cfg, tr, s_ch.copy(), use_sata=True,
                          verbose=False, correct_terms=("C0",)),
        "sub EXPLICIT": _quiet(reconstruct_subband_explicit, cfg, tr, s_ch.copy(),
                              use_sata=True, verbose=False, correct_terms=("C0",)),
    }
    out = {}
    for name, sr in recs.items():
        f = focus_mag(sr, sref1)
        out[name] = {"peak_pct": 100.0 * f.max() / p, "amb_db": ambiguity_db(f)}
    return out


def plot_gap(results, nrx_list, bxt_list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("    (matplotlib unavailable, skipping plot)", e)
        return None
    fig, axes = plt.subplots(1, len(bxt_list), figsize=(4.3 * len(bxt_list), 3.6),
                             sharey=True, squeeze=False)
    for bi, bxt in enumerate(bxt_list):
        ax = axes[0][bi]
        for name, style in [("no-SATA", "o:"), ("SATA band", "s-"),
                            ("sub OLD", "^--"), ("sub EXPLICIT", "D-")]:
            y = [results[f"{n}_{bxt}"][name]["amb_db"] for n in nrx_list]
            ax.plot(nrx_list, y, style, label=name, lw=1.8, ms=6)
        ax.set_title(f"bxt_max = {bxt:.0f} m", fontsize=10)
        ax.set_xlabel(r"$N_{rx}$")
        ax.grid(alpha=0.3)
        if bi == 0:
            ax.set_ylabel("worst azimuth ambiguity [dB]\n(lower = better)")
    axes[0][-1].legend(fontsize=8, loc="best")
    fig.suptitle("Azimuth ambiguity vs. $N_{rx}$: explicit sub-band SATA vs. others",
                 y=1.02)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "gap_vs_nrx.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    plot -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrx", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--bxt", type=float, nargs="+", default=[10, 30, 50])
    args = ap.parse_args()

    results = {}
    hdr = (f"{'Nrx':>4}{'bxt':>5} | {'no-SATA':>9}{'band':>9}{'subOLD':>9}"
           f"{'subEXPL':>9} | {'OLD-band':>9}{'EXPL-band':>10}")
    print(hdr)
    print("-" * len(hdr))
    for Nrx in args.nrx:
        for bxt in args.bxt:
            r = compare_one(Nrx, bxt)
            results[f"{Nrx}_{int(bxt)}"] = r
            b = r["SATA band"]["amb_db"]
            print(f"{Nrx:>4}{bxt:>5.0f} | "
                  f"{r['no-SATA']['amb_db']:>9.1f}{b:>9.1f}"
                  f"{r['sub OLD']['amb_db']:>9.1f}{r['sub EXPLICIT']['amb_db']:>9.1f} | "
                  f"{r['sub OLD']['amb_db']-b:>+9.2f}{r['sub EXPLICIT']['amb_db']-b:>+10.2f}")
    print("\n(worst azimuth ambiguity [dB], lower=better; "
          "last 2 cols = gap vs SATA band)")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    payload = {"nrx": args.nrx, "bxt": [int(b) for b in args.bxt],
               "scene": AZIMUTH_SPECS, "results": results}
    jpath = os.path.join(PLOTS_DIR, "results.json")
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nresults -> {jpath}")
    plot_gap(results, args.nrx, [int(b) for b in args.bxt])


if __name__ == "__main__":
    main()
