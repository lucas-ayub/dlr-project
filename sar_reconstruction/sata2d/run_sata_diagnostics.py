# -*- coding: utf-8 -*-
"""
Isolate the sub-band SATA bug, exactly along the two checks that were asked for:

  (A) "make sure that the sub-aperture length is defined properly by checking
       the data after the STFT"
  (B) "define the frequency bins for each sub-band properly before doing the
       mapping (finding the azimuth time bins)"

Four tests
----------
TEST 1  sub-aperture grid, side by side:
          * CORRECTED                     (prf_data = PRF_op, f_centre = f_k)
          * LEGACY as currently called    (prf = PRF_op, Nsb = Nrx)
          * LEGACY with the full PRF      (prf = prf,    Nsb = Nrx)  <- the trap
TEST 2  STFT check: the Doppler support MEASURED on real channel data inside one
        sub-aperture, against the support the geometry predicts,
        df = |ka| * Tsubeff/prf_data  with  ka = 2*v^2/(wl*r).
TEST 3  mapping check: the bin -> azimuth-sample map ``azpos`` against the
        analytic truth  azpos_true(f) = wl*r*f*prf_data / (2*v^2).
TEST 4  end-to-end phase transfer: inject a known residual delta_C0 = wl/8 at
        the target's azimuth position, focus, and read the phase of the peak.
        A correct kernel gives exactly -360*delta_C0/wl = -45 deg.

Run
---
    python -m sata2d.run_sata_diagnostics
    python run_sata_diagnostics.py --preset small --nrx 2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "sata2d"

from .params import make_params                      # noqa: E402
from .tracks import build_tracks                     # noqa: E402
from .datagen2d import get_raw_data_2d               # noqa: E402
from .rangecomp import range_compress                # noqa: E402
from .sata2d import (subaperture_grid, legacy_subaperture_grid, sata_1d,
                     legacy_sata_1d, subband_centre_frequency)   # noqa: E402

PI = np.pi
RULE = "=" * 78


def hdr(txt):
    print(f"\n{RULE}\n{txt}\n{RULE}")


# ---------------------------------------------------------------------------
def test1_grids(p):
    hdr("TEST 1 -- sub-aperture grid per sub-band")
    print("A channel line has Na_ch = %d samples at PRF_op = %.1f Hz.\n"
          "Output sub-band k is PRF_op wide and centred on f_k.\n"
          % (p.Na_ch, p.PRF_op))
    for k in range(p.Nrx):
        f_k = subband_centre_frequency(k, p.Nrx, p.prf)
        beta_k = float(np.arcsin(np.clip(p.wl * f_k / (2 * p.ve), -1, 1)))
        print(f"--- sub-band k = {k}:  f_k = {f_k:8.1f} Hz, "
              f"beta_k = {np.degrees(beta_k):7.4f} deg ---")

        gc = subaperture_grid(p.r_ref, p.PRF_op, p.ve, p.wl, p.r_ref,
                              f_centre=f_k, sata_osf=1)
        gl = legacy_subaperture_grid(p.r_ref, p.PRF_op, p.Nrx, p.ve, p.wl,
                                     p.r_ref, squint=beta_k, sata_osf=1)
        gf = legacy_subaperture_grid(p.r_ref, p.prf, p.Nrx, p.ve, p.wl,
                                     p.r_ref, squint=beta_k, sata_osf=1)
        print(" CORRECTED  (prf_data = PRF_op, f_centre = f_k)")
        print(gc.report("   "))
        print(" LEGACY as called today  (prf = PRF_op, Nsb = Nrx)")
        print(gl.report("   "))
        print(" LEGACY with the full PRF  (prf = prf, Nsb = Nrx)  <- the trap")
        print(gf.report("   "))
        print(f"   >> Tsubeff ratio  legacy/correct = "
              f"{gl.Tsubeff/gc.Tsubeff:.2f}  (should be 1.00)")
        print(f"   >> ground extent  legacy = {gl.ground_extent:.1f} m vs "
              f"deltax = {gc.deltax:.1f} m")
    return


# ---------------------------------------------------------------------------
def test2_stft(p, chan_line, n0_ch):
    hdr("TEST 2 -- sub-aperture length / frequency bins checked ON THE STFT")
    ka = 2.0 * p.ve ** 2 / (p.wl * p.r_ref)       # Doppler rate [Hz/s]
    print(f"Doppler rate ka = 2 v^2/(wl r) = {ka:.1f} Hz/s")
    print("Inside ONE sub-aperture a point target sweeps only "
          f"{ka * 20 / p.PRF_op:.0f} Hz, so its\n"
          "sub-aperture spectrum has a single peak even though the FULL channel\n"
          "line is aliased (abw = %.0f Hz > PRF_op = %.0f Hz).  The peak bin must\n"
          "map back onto the target's azimuth pixel through azpos -- that is the\n"
          "check.  Which of the Nrx aliases the peak belongs to is decided by the\n"
          "sub-band centre f_k, i.e. by the frequency-bin definition.\n"
          % (p.abw, p.PRF_op))

    grids_c = [subaperture_grid(p.r_ref, p.PRF_op, p.ve, p.wl, p.r_ref,
                                f_centre=subband_centre_frequency(k, p.Nrx, p.prf))
               for k in range(p.Nrx)]
    grids_l = [legacy_subaperture_grid(
        p.r_ref, p.PRF_op, p.Nrx, p.ve, p.wl, p.r_ref,
        squint=float(np.arcsin(np.clip(
            p.wl * subband_centre_frequency(k, p.Nrx, p.prf) / (2 * p.ve), -1, 1))))
        for k in range(p.Nrx)]

    print(f"{'grid':>10s} {'Tsubeff':>8s} {'Nzp':>5s} {'bin [Hz]':>9s} "
          f"{'bin [samples]':>14s} {'mean |err|':>11s} {'median':>8s} {'p90':>8s}")
    for name, grids in (("corrected", grids_c), ("legacy", grids_l)):
        errs = _mapping_errors(chan_line, grids, n0_ch, ka, p)
        g = grids[0]
        print(f"{name:>10s} {g.Tsubeff:8d} {g.Nzp:5d} "
              f"{g.prf_data/g.Nzp:9.1f} {g.prf_data/g.Nzp/ka*g.prf_data:14.1f} "
              f"{np.mean(np.abs(errs)):11.2f} "
              f"{np.median(np.abs(errs)):8.2f} "
              f"{np.percentile(np.abs(errs), 90):8.2f}")
    print("\nError = (sub-aperture centre + azpos[peak bin]) - true target pixel,\n"
          "in azimuth samples.  It cannot be better than half a bin; halving the\n"
          "sub-aperture doubles the bin and therefore doubles the mapping error.")


def _mapping_errors(x, grids, n0, ka, p):
    """Locate the target from each sub-aperture STFT and compare with n0."""
    g0 = grids[0]
    L, hop, Nzp = g0.Tsubeff, g0.Tsub, g0.Nzp
    ramp = np.arange(hop) / (hop - 1)
    win = np.concatenate((ramp, ramp[::-1]))[:L]
    errs = []
    for s in range(0, len(x) - L, hop):
        c = s + 0.5 * L
        # True instantaneous Doppler of the target at this sub-aperture centre.
        f_true = (n0 - c) * ka / g0.prf_data
        # Keep one bin of margin from the sub-band edges: at the very edge the
        # STFT peak wraps into the neighbouring band and the test is meaningless.
        margin = g0.prf_data / g0.Nzp
        if not (-p.prf / 2 + margin < f_true < p.prf / 2 - margin):
            continue
        k = int(np.clip(np.floor(f_true / p.PRF_op + p.Nrx / 2), 0, p.Nrx - 1))
        g = grids[k]
        buf = np.zeros(Nzp, complex)
        buf[:L] = x[s:s + L] * win
        spec = np.abs(np.fft.fft(buf))
        if spec.max() <= 0:
            continue
        b = int(np.argmax(spec))
        errs.append(c + g.azpos[b] - n0)
    return np.array(errs) if errs else np.array([np.nan])


# ---------------------------------------------------------------------------
def test3_mapping(p):
    hdr("TEST 3 -- frequency bins -> azimuth sample mapping")
    print("Ground truth (exact for the parabolic range history):\n"
          "   azpos_true(f) = wl * r * f * prf_data / (2 v^2)\n")
    print(f"{'sub-band':>9s} {'grid':>10s} {'span [samples]':>18s} "
          f"{'max |error|':>12s}")
    for k in range(p.Nrx):
        f_k = subband_centre_frequency(k, p.Nrx, p.prf)
        beta_k = float(np.arcsin(np.clip(p.wl * f_k / (2 * p.ve), -1, 1)))
        for name, g, rate in (
                ("corrected", subaperture_grid(p.r_ref, p.PRF_op, p.ve, p.wl,
                                               p.r_ref, f_centre=f_k), p.PRF_op),
                ("legacy", legacy_subaperture_grid(p.r_ref, p.PRF_op, p.Nrx,
                                                   p.ve, p.wl, p.r_ref,
                                                   squint=beta_k), p.PRF_op),
                ("legacy-full", legacy_subaperture_grid(p.r_ref, p.prf, p.Nrx,
                                                        p.ve, p.wl, p.r_ref,
                                                        squint=beta_k), p.PRF_op)):
            # Truth for the bins this grid claims, on the real data rate.
            # It is the ABSOLUTE azimuth offset -- the reconstructed image is
            # broadside, so no tan(squint) is subtracted (BUG 3).
            truth = p.wl * p.r_ref * g.fsub * rate / (2 * p.ve ** 2)
            err = np.max(np.abs(g.azpos - truth))
            print(f"{k:>9d} {name:>10s} "
                  f"{g.azpos.min():8.1f} ..{g.azpos.max():7.1f} {err:12.2f}")
    print("\n'legacy-full' shows what happens if the ambiguous (prf, Nsb) pair is\n"
          "'fixed' by passing the full PRF: bins and mapping both blow up by Nrx.")


# ---------------------------------------------------------------------------
def test4_phase_transfer(p, ref_line):
    hdr("TEST 4 -- end-to-end phase transfer of a localised delta_C0")
    naz = len(ref_line)
    A = p.wl / 8.0                       # -> exactly -45 deg at the peak
    expected = -360.0 * A / p.wl

    def focus(x):
        f = np.roll(np.arange(naz) * p.prf / naz - 0.5 * p.prf, int(naz / 2))
        mf = np.exp(4j * PI / p.wl * p.r_ref
                    * (np.sqrt(np.maximum(1 - (p.wl * f / 2 / p.ve) ** 2, 0)) - 1))
        return np.fft.ifft(np.fft.fft(x) * mf)

    img0 = focus(ref_line)
    n0 = int(np.argmax(np.abs(img0)))
    ph0 = np.angle(img0[n0], deg=True)

    for width, tag in ((naz, "constant (insensitive probe)"),
                       (60.0, "Gaussian bump, sigma = 60 samples"),
                       (30.0, "Gaussian bump, sigma = 30 samples")):
        if width >= naz:
            dC0 = np.full(naz, A)
        else:
            dC0 = A * np.exp(-0.5 * ((np.arange(naz) - n0) / width) ** 2)
        out_c = sata_1d(ref_line, dC0, rref=p.r_ref, prf_data=p.prf, v=p.ve,
                        wl=p.wl, r=p.r_ref, f_centre=0.0, sata_osf=1)
        out_l = legacy_sata_1d(ref_line, dC0, rref=p.r_ref, prf=p.prf,
                               Nsb=p.Nrx, v=p.ve, squint=0.0, wl=p.wl,
                               r=p.r_ref, sata_osf=1)
        print(f"\ndelta_C0 = wl/8 = {A*1e3:.3f} mm, {tag}")
        print(f"   expected peak phase {expected:+.2f} deg, no peak loss")
        print(f"   {'kernel':>16s} {'phase change':>14s} {'error':>9s} "
              f"{'peak loss [dB]':>15s}")
        for name, out in (("corrected", out_c), ("legacy(Nsb=Nrx)", out_l)):
            img = focus(out)
            d = float(np.angle(np.exp(1j * np.radians(
                np.angle(img[n0], deg=True) - ph0)), deg=True))
            loss = 20 * np.log10(np.abs(img[n0]) / np.abs(img0[n0]))
            print(f"   {name:>16s} {d:+14.2f} {d-expected:+9.2f} {loss:15.3f}")
    print("\nThe narrower the topography feature, the more the correction depends\n"
          "on the frequency -> azimuth mapping.  The legacy grid, whose\n"
          "sub-aperture is Nrx times too short, reads delta_C0 on a bin grid Nrx\n"
          "times coarser and therefore loses both phase accuracy and peak power.")


# ---------------------------------------------------------------------------
def test5_squint_offset(p):
    hdr("TEST 5 -- the per-sub-band azimuth offset introduced by tan(squint)")
    ds = p.ve / p.PRF_op
    print("Classic SATA subtracts tan(squint) from the mapping because it images\n"
          "at that squint.  The sub-band reconstruction images BROADSIDE, so the\n"
          "subtraction becomes a pure offset of the topography lookup:\n")
    print(f"{'k':>3s} {'f_k [Hz]':>10s} {'beta_k [deg]':>13s} "
          f"{'offset [samples]':>17s} {'offset [m]':>12s}")
    for k in range(p.Nrx):
        f_k = subband_centre_frequency(k, p.Nrx, p.prf)
        beta_k = float(np.arcsin(np.clip(p.wl * f_k / (2 * p.ve), -1, 1)))
        off = p.r_ref * np.tan(beta_k) / p.ve * p.PRF_op
        print(f"{k:>3d} {f_k:10.1f} {np.degrees(beta_k):13.4f} "
              f"{off:17.1f} {off*ds:12.1f}")
    print("\nThe offsets have OPPOSITE signs above and below zero Doppler, so the\n"
          "sub-bands pull the correction apart instead of agreeing.  For k with\n"
          "beta_k = 0 (whole-band SATA) the offset vanishes -- which is exactly\n"
          "why the whole-band SATA works and the per-sub-band version does not.")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="small", choices=["small", "paper"])
    ap.add_argument("--nrx", type=int, default=2)
    args = ap.parse_args(argv)

    p = make_params(args.preset, Nrx=args.nrx)
    print(p.summary())

    ptx, prx, vtx, vrx = build_tracks(p)
    idx = p.idx_dec
    p_tgt = np.array([[p.r_ref, p.tgt_az]])
    sigma = np.ones(1)

    s_ref, _sc, _rh, fmax, fmin = get_raw_data_2d(
        p_tgt, ptx, prx, vtx, vrx, p.sq_tx, p.sq_rx, p.theta_tx, p.theta_rx,
        p.wl, p.prf, sigma, p.rbw, p.cd, p.rsf, p.r_scan, p.abw)
    _s, s_channel, _r, _a, _b = get_raw_data_2d(
        p_tgt, ptx[:, idx], prx[:, :, idx], vtx[idx], vrx[:, idx], p.sq_tx,
        p.sq_rx, p.theta_tx, p.theta_rx, p.wl, p.PRF_op, sigma, p.rbw, p.cd,
        p.rsf, p.r_scan, p.abw)

    s_ref_rc = range_compress(s_ref, p.cd, p.rbw, p.rsf, axis=1)
    s_ch_rc = range_compress(s_channel, p.cd, p.rbw, p.rsf, axis=2)
    n_tgt = int(np.argmax(np.abs(s_ref_rc).sum(axis=0)))

    test1_grids(p)
    test2_stft(p, s_ch_rc[0, :, n_tgt], p.Na_ch / 2)
    test3_mapping(p)
    test4_phase_transfer(p, s_ref_rc[:, n_tgt])
    test5_squint_offset(p)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
