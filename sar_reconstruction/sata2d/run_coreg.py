# -*- coding: utf-8 -*-
"""
Range co-registration: the experiment behind ``coreg.py``.

Two stages, run separately because each reconstruction is expensive.

``--stage equiv`` (the core result, ~4 reconstructions)
    The 2x2 matrix that proves the range-frequency-dependent filter and the
    explicit co-registration are the SAME correction.  Run with the oracle
    coefficient table (filter given the true target height) so the phase model
    is exact and the only thing left to go wrong is the range alignment.

``--stage pipeline`` (~3 reconstructions)
    The full pipeline as it is used: no SATA / SATA per sub-band with the
    standard filter / SATA per sub-band with the explicit co-registration.

Both write into ``plots/`` and are read back by ``plot_coreg.py``.

Run from ``sar_reconstruction/``::

    python -m sata2d.run_coreg --stage equiv
    python -m sata2d.run_coreg --stage pipeline
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)

from .geometry import (make_params3d, build_tracks_3d, generate_reference_3d,
                       generate_channels_3d, CoeffTable3D)
from .reconstruction import (range_compress, build_delta_C0_map_3d,
                             reconstruct_subband_2d, scatterer_range,
                             range_bin_of, matched_filter, METHOD_KW)
from .coreg import (coregister_channels, coregistration_shift,
                    swath_shift_variation, monochromatic_filter,
                    reconstruct_explicit_coreg)

C_LIGHT = 3.0e8
PLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


class Bench:
    """Scene, reference signal and the two metrics, built once."""

    def __init__(self, nrx=4, dxt=150.0, azimuth=0.0, height=240.0):
        self.p = p = make_params3d(Nrx=nrx, dxt=dxt, specs=((azimuth, height),))
        self.tr = build_tracks_3d(p)
        self.ptg = np.asarray(p.points[0], float)
        self.nb = range_bin_of(p, scatterer_range(p, self.ptg))
        self.rho_r = C_LIGHT / (2.0 * p.rsf)
        self.height = height

        self.ref = range_compress(generate_reference_3d(p, self.tr),
                                  p.cd, p.rbw, p.rsf, axis=1)
        self.ch = range_compress(generate_channels_3d(p, self.tr),
                                 p.cd, p.rbw, p.rsf, axis=2)
        self.ch_co = coregister_channels(p, self.ch, verbose=False)

        self.ref_line = self.ref[:, self.nb]
        self.irf_mono = matched_filter(self.ref_line, self.ref_line)
        self.pm = float(np.abs(self.irf_mono).max())
        self.S_ref = np.fft.fftshift(np.fft.fft2(self.ref))
        fa = np.fft.fftshift(np.fft.fftfreq(p.Na, 1 / p.prf))
        self.inband = (np.abs(fa) < p.abw / 2)[:, None] & np.ones((1, p.Nr), bool)

    def report_geometry(self):
        p = self.p
        print(p.summary())
        print(f"range cell : {self.rho_r:.2f} m")
        print(f"target     : height {self.height:.0f} m -> range bin "
              f"{self.nb}/{p.Nr}\n")
        print("co-registration shifts, Eq. (1)")
        for i in range(p.Nrx):
            dR = float(coregistration_shift(p, i))
            print(f"  ch{i}: bxt = {p.bxt[i]:+7.1f} m -> dR = {dR:+7.2f} m"
                  f" = {dR / self.rho_r:+6.3f} cells")
        print(f"  swath variation: {swath_shift_variation(self.p):.4f} cells\n")

    def score(self, rec):
        """(focused peak, % of monostatic, in-band error/signal [dB], IRF)."""
        irf = matched_filter(rec[:, self.nb], self.ref_line)
        v = float(np.abs(irf).max())
        S = np.fft.fftshift(np.fft.fft2(rec))
        e = np.abs(S - self.S_ref)[self.inband]
        s = np.abs(self.S_ref)[self.inband]
        esr = 20.0 * np.log10(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(s ** 2)))
        return v, 100.0 * v / self.pm, esr, irf


# ---------------------------------------------------------------------------
def stage_equiv(b: Bench, out):
    """The 2x2: {wl(f_r), wl0} x {no co-reg, explicit co-reg}."""
    tab = CoeffTable3D(b.p, b.tr, n_nodes=16, height=float(b.ptg[2]))
    kw = dict(sata_osf=4, verbose=False, **METHOD_KW["no"])
    labels, pct, esr, irfs = [], [], [], []

    def run(tag, data, mono):
        ctx = monochromatic_filter() if mono else _null()
        with ctx:
            rec = reconstruct_subband_2d(b.p, b.tr, data, tab, **kw)
        _, q, e, irf = b.score(rec)
        labels.append(tag); pct.append(q); esr.append(e); irfs.append(irf)
        print(f"  {tag:<46} {q:7.1f}% {e:8.2f} dB")

    print("2x2 equivalence matrix (oracle coefficients, no SATA)")
    print(f"  {'case':<46} {'peak':>8} {'err/sig':>11}")
    run("wl(f_r) filter, no explicit co-reg  [standard]", b.ch,    False)
    run("wl(f_r) filter + explicit co-reg    [double]",   b.ch_co, False)
    run("wl0 filter, no explicit co-reg      [none]",     b.ch,    True)
    run("wl0 filter + explicit co-reg        [new]",      b.ch_co, True)

    np.savez_compressed(out, labels=np.array(labels), pct=np.array(pct),
                        esr=np.array(esr), irfs=np.array(irfs),
                        irf_mono=b.irf_mono, prf=b.p.prf, Na=b.p.Na)
    print(f"\n  -> {out}")


def stage_pipeline(b: Bench, out):
    """no SATA / SATA standard / SATA with the explicit co-registration."""
    p, tr = b.p, b.tr
    tab = CoeffTable3D(p, tr, n_nodes=16)
    maps = [build_delta_C0_map_3d(p, tr, i) for i in range(p.Nrx)]
    common = dict(sata_osf=4, maps=maps, verbose=False)
    labels, pct, esr, irfs = [], [], [], []

    def add(tag, rec):
        _, q, e, irf = b.score(rec)
        labels.append(tag); pct.append(q); esr.append(e); irfs.append(irf)
        print(f"  {tag:<46} {q:7.1f}% {e:8.2f} dB")

    print("full pipeline")
    print(f"  {'case':<46} {'peak':>8} {'err/sig':>11}")
    add("no SATA",
        reconstruct_subband_2d(p, tr, b.ch, tab, **METHOD_KW["no"], **common))
    add("SATA per sub-band, standard filter",
        reconstruct_subband_2d(p, tr, b.ch, tab, **METHOD_KW["sub"], **common))
    add("SATA per sub-band, explicit co-registration",
        reconstruct_explicit_coreg(p, tr, b.ch, tab, **METHOD_KW["sub"], **common))

    # channel range profiles at beam centre, for the illustration
    prof, prof_co = [], []
    for i in range(p.Nrx):
        m = int(round(p.Na_ch / 2 + p.bat[i] / (2 * p.vs) * p.PRF_op))
        prof.append(b.ch[i, m, :].copy())
        prof_co.append(b.ch_co[i, m, :].copy())

    np.savez_compressed(
        out, labels=np.array(labels), pct=np.array(pct), esr=np.array(esr),
        irfs=np.array(irfs), irf_mono=b.irf_mono,
        prof=np.array(prof), prof_co=np.array(prof_co),
        bxt=p.bxt, rho_r=b.rho_r, r_scan0=p.r_scan[0],
        r_target=scatterer_range(p, b.ptg), prf=p.prf, Na=p.Na, height=b.height)
    print(f"\n  -> {out}")


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="equiv", choices=("equiv", "pipeline"))
    ap.add_argument("--nrx", type=int, default=4)
    ap.add_argument("--dxt", type=float, default=150.0)
    ap.add_argument("--height", type=float, default=240.0)
    ap.add_argument("--azimuth", type=float, default=0.0)
    args = ap.parse_args(argv)

    os.makedirs(PLOTS, exist_ok=True)
    b = Bench(args.nrx, args.dxt, args.azimuth, args.height)
    b.report_geometry()
    if args.stage == "equiv":
        stage_equiv(b, os.path.join(PLOTS, "cache_coreg_equiv.npz"))
    else:
        stage_pipeline(b, os.path.join(PLOTS, "cache_coreg_pipeline.npz"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
