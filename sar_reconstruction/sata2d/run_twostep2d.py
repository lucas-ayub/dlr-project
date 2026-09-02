# -*- coding: utf-8 -*-
"""
Full 2-D pipeline -- port of the reference ``MP2DATBF.py``.

Steps
-----
1. parameters + platform tracks (linear orbit)
2. generate the raw data twice:
     * monostatic, at the FULL equivalent PRF   -> the reference signal
     * bistatic per channel, on the decimated
       azimuth axis (PRF_op = prf/Nrx)          -> the data to reconstruct
3. range compression of both
4. two-step Range-Doppler multichannel azimuth reconstruction
5. azimuth bulk focusing of both, and comparison:
     * 2-D phase difference inside the common (range, Doppler) band
     * azimuth impulse response (peak position, PSLR)

Run
---
    python -m sata2d.run_twostep2d                # from sar_reconstruction/
    python run_twostep2d.py --preset small --nrx 2 --plots

Everything is in RAM at the "small" preset.  Use ``--backend hdf5`` (requires
h5py) for the "paper" preset.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

if __package__ in (None, ""):                       # allow direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "sata2d"

from .params import make_params                                 # noqa: E402
from .tracks import build_tracks                                # noqa: E402
from .datagen2d import get_raw_data_2d                          # noqa: E402
from .rangecomp import range_compress, range_matched_filter     # noqa: E402
from .rd_recons2d import run_two_step                           # noqa: E402

PI = np.pi
C_LIGHT = 3.0e8


# ---------------------------------------------------------------------------
def azimuth_matched_filter(p, r0, with_rc=True):
    """
    2-D bulk focusing filter [Na, Nr] in the (azimuth freq, range freq) domain.

    ``ph_bulk`` is the standard Range-Doppler bulk term written with the
    range-frequency dependence kept explicitly (the ``(1 + fr/f0)`` factor);
    the second half subtracts the range-frequency-only part so that the filter
    focuses in azimuth without moving the range peak.
    """
    f = np.roll(np.arange(p.Na) * p.prf / p.Na - 0.5 * p.prf, int(p.Na / 2))
    fr = p.fr
    scale = (1.0 + fr / p.f0)
    ph_bulk = (4j * PI / p.wl * r0
               * (np.sqrt(scale[None, :] ** 2
                          - (p.wl * f[:, None] / 2.0 / p.ve) ** 2)
                  - scale[None, :]))
    mf = np.exp(ph_bulk)
    if with_rc:
        mf = mf * range_matched_filter(fr, p.cd, p.rbw)[None, :]
    return mf.astype(np.complex64)


def fft2d(x):
    return np.fft.fft(np.fft.fft(x, axis=0), axis=1)


def irf_metrics(line, label=""):
    """Peak index, peak value and PSLR of a 1-D impulse response."""
    a = np.abs(line)
    pk = int(np.argmax(a))
    peak = a[pk]
    # First null on each side, then the highest side lobe outside them.
    d = np.diff(a)
    left = pk
    while left > 1 and d[left - 1] > 0:
        left -= 1
    right = pk
    while right < len(a) - 2 and d[right] < 0:
        right += 1
    side = np.concatenate((a[:max(left - 1, 0)], a[min(right + 2, len(a)):]))
    pslr = 20 * np.log10(side.max() / peak) if side.size else np.nan
    return dict(label=label, peak_idx=pk, peak=float(peak), pslr_db=float(pslr))


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="small", choices=["small", "paper"])
    ap.add_argument("--nrx", type=int, default=2, help="number of channels")
    ap.add_argument("--analytic", action="store_true",
                    help="closed-form 2x2 matrix inverse (Nrx = 2 only)")
    ap.add_argument("--no-rcmc", action="store_true",
                    help="skip the range-cell-migration of the residual filter")
    ap.add_argument("--backend", default="memory", choices=["memory", "hdf5"])
    ap.add_argument("--outdir", default="plots/sata2d")
    ap.add_argument("--plots", action="store_true", help="save the figures")
    args = ap.parse_args(argv)

    p = make_params(args.preset, Nrx=args.nrx)
    print(p.summary())

    # ---- 1) tracks --------------------------------------------------------
    ptx, prx, vtx, vrx = build_tracks(p)
    idx = p.idx_dec
    ptx_d, prx_d = ptx[:, idx], prx[:, :, idx]
    vtx_d, vrx_d = vtx[idx], vrx[:, idx]

    # ---- 2) point target and raw data ------------------------------------
    # (ground range, azimuth) of the scatterer; sigma = 1.
    p_tgt = np.array([[p.r_ref, p.tgt_az]])
    sigma = np.ones(1)

    t0 = time.time()
    s_ref, _s_ch_full, _rh, fmax, fmin = get_raw_data_2d(
        p_tgt, ptx, prx, vtx, vrx, p.sq_tx, p.sq_rx, p.theta_tx, p.theta_rx,
        p.wl, p.prf, sigma, p.rbw, p.cd, p.rsf, p.r_scan, p.abw, verbose=True)
    print(f"raw data (full PRF reference)  : {time.time()-t0:.1f} s")

    t0 = time.time()
    _s, s_channel, _rh_c, fmax_d, fmin_d = get_raw_data_2d(
        p_tgt, ptx_d, prx_d, vtx_d, vrx_d, p.sq_tx, p.sq_rx, p.theta_tx,
        p.theta_rx, p.wl, p.PRF_op, sigma, p.rbw, p.cd, p.rsf, p.r_scan, p.abw)
    print(f"raw data (decimated channels)  : {time.time()-t0:.1f} s")

    # ---- 3) range compression --------------------------------------------
    s_channel_rc = range_compress(s_channel, p.cd, p.rbw, p.rsf, axis=2)

    # ---- 4) two-step reconstruction --------------------------------------
    print("two-step reconstruction:")
    t0 = time.time()
    rec, ds = run_two_step(p, s_channel_rc, fmax, fmin,
                           analytic=args.analytic,
                           apply_rcmc=not args.no_rcmc,
                           backend=args.backend,
                           filename="sata2d_dataset.h5")
    print(f"  total                        : {time.time()-t0:.1f} s")

    # ---- 5) focusing + comparison ----------------------------------------
    mf = azimuth_matched_filter(p, p.r_ref, with_rc=True)     # reference: raw
    mf_worc = azimuth_matched_filter(p, p.r_ref, with_rc=False)  # rec: already RC'd

    REF_F = fft2d(s_ref) * mf
    REC_F = fft2d(rec) * mf_worc

    # In-band mask: common Doppler band x chirp bandwidth.
    fa = np.roll(np.arange(p.Na) * p.prf / p.Na - 0.5 * p.prf, int(p.Na / 2))
    band_a = (fa > fmin) & (fa < fmax)
    band_r = np.abs(p.fr) < 0.5 * p.rbw
    mask = band_a[:, None] & band_r[None, :]

    ph_diff = np.angle(np.conjugate(REC_F) * REF_F, deg=True)
    # Remove the (irrelevant) constant offset before quoting an RMS.
    off = np.angle(np.sum(np.conjugate(REC_F[mask]) * REF_F[mask]), deg=True)
    resid = np.angle(np.exp(1j * np.radians(ph_diff[mask] - off)), deg=True)
    print(f"\nphase difference vs the ideal full-PRF signal (in band):")
    print(f"  constant offset : {off:8.3f} deg")
    print(f"  rms residual    : {np.sqrt(np.mean(resid**2)):8.3f} deg")
    print(f"  max |residual|  : {np.max(np.abs(resid)):8.3f} deg")

    ref_img = np.fft.ifft(np.fft.ifft(REF_F, axis=0), axis=1)
    rec_img = np.fft.ifft(np.fft.ifft(REC_F, axis=0), axis=1)
    nr_pk = int(np.argmax(np.abs(ref_img).max(axis=0)))
    # Centre the impulse response on its own peak before measuring it.
    pk0 = int(np.argmax(np.abs(ref_img[:, nr_pk])))
    roll = p.Na // 2 - pk0
    ref_line = np.roll(ref_img[:, nr_pk], roll)
    rec_line = np.roll(rec_img[:, nr_pk], roll)
    m_ref = irf_metrics(ref_line, "reference")
    m_rec = irf_metrics(rec_line, "reconstructed")
    print("\nazimuth impulse response (range bin %d):" % nr_pk)
    for m in (m_ref, m_rec):
        print(f"  {m['label']:>14s}: peak at {m['peak_idx']:6d}, "
              f"|peak| = {m['peak']:.3e}, PSLR = {m['pslr_db']:6.2f} dB")
    print(f"  peak offset      : {m_rec['peak_idx'] - m_ref['peak_idx']} samples")
    print(f"  amplitude ratio  : {m_rec['peak']/m_ref['peak']:.4f}")

    # ---- 6) plots ---------------------------------------------------------
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.outdir, exist_ok=True)

        pd = np.where(mask, ph_diff - off, np.nan)
        pd = np.roll(np.roll(pd, p.Na // 2, axis=0), p.Nr // 2, axis=1)
        plt.figure(figsize=(8, 5))
        plt.imshow(pd, origin="lower", aspect="auto", cmap="bwr",
                   vmin=-5, vmax=5,
                   extent=[np.min(p.fr) / 1e6, np.max(p.fr) / 1e6,
                           np.min(fa) / 1e3, np.max(fa) / 1e3])
        plt.colorbar(label="phase difference [deg]")
        plt.xlabel("range frequency [MHz]")
        plt.ylabel("azimuth frequency [kHz]")
        plt.title("reconstructed vs ideal (2-D spectrum)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "phase_difference_2d.png"), dpi=140)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        w = 120
        sl = slice(p.Na // 2 - w, p.Na // 2 + w)
        for line, lab in ((ref_line, "reference"), (rec_line, "reconstructed")):
            a = np.abs(line)
            db = 20 * np.log10(a / a.max() + 1e-20)
            axes[0].plot(np.arange(-w, w), db[sl], label=lab)
            axes[1].plot(np.arange(p.Na) - p.Na // 2, db, label=lab, lw=0.7)
        axes[0].set_title("main lobe")
        axes[1].set_title("full line (azimuth ambiguities)")
        for ax in axes:
            ax.set_ylim(-60, 2)
            ax.set_xlabel("azimuth sample (relative to peak)")
            ax.set_ylabel("normalised amplitude [dB]")
            ax.grid(alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "azimuth_irf.png"), dpi=140)

        plt.close("all")
        print(f"\nfigures written to {args.outdir}/")

    ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
