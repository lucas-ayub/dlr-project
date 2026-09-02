# -*- coding: utf-8 -*-
"""
Short no-SATA / SATA-band / SATA-sub comparison -- to check the Nsb fix.

Run (from sar_reconstruction/):
    PYTHONPATH=. python ../runs/core/compare_sata_quick.py
    PYTHONPATH=. python ../runs/core/compare_sata_quick.py --nrx 2 3 4 5 6 7 --bxt 10 30 50
"""
from __future__ import annotations

import argparse

import numpy as np

import sar_recon as sar
from sar_recon.config import (SystemParams, Scene, ArrayGeometry,
                              prf_from_dpca, integration_time, build_time_axis)
from sar_recon.signal_model import getRawData1D
from sar_recon.sata import sata_channels
from sar_recon.subband_recon import reconstruct_subband

_DX_DPCA = 11.0
_RDELAY_SCENE = 0.0051115753
AZIMUTH_SPECS = ((-400, 80), (-200, 160), (0, 240), (200, 320), (400, 400))


def build_scene(Nrx, bxt_max, seed=0):
    system = SystemParams()
    base = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0)
    r0, H, y0 = base.r0, base.H, base.y0
    extra = tuple((float(dx), float(np.sqrt(r0**2 - (H - dh)**2) - y0), float(dh))
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


def compare_one(Nrx, bxt):
    cfg, tr = build_scene(Nrx, bxt)
    sref1, sig_true, s_ch = channels_and_ref(cfg, tr)
    p = focus_mag(sig_true, sref1).max()

    srec_no = sar.reconstruct(cfg, tr, s_ch.copy())
    srec_band = sar.reconstruct(cfg, tr, sata_channels(cfg, tr, s_ch.copy(), verbose=False))
    srec_sub = reconstruct_subband(cfg, tr, s_ch.copy(), use_sata=True, verbose=False,
                                   correct_terms=("C0",))

    out = {}
    for name, sr in [("no-SATA", srec_no), ("SATA band", srec_band), ("SATA sub", srec_sub)]:
        f = focus_mag(sr, sref1)
        out[name] = (100 * f.max() / p, ambiguity_db(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrx", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--bxt", type=float, nargs="+", default=[10, 30, 50])
    args = ap.parse_args()

    hdr = f"{'Nrx':>4}{'bxt':>6} | {'no-SATA':>18}{'SATA band':>18}{'SATA sub':>18}"
    print(hdr); print("-" * len(hdr))
    for Nrx in args.nrx:
        for bxt in args.bxt:
            r = compare_one(Nrx, bxt)
            def cell(k):
                pk, db = r[k]
                return f"{pk:5.1f}% / {db:6.1f} dB"
            print(f"{Nrx:>4}{bxt:>6.0f} | {cell('no-SATA'):>18}{cell('SATA band'):>18}{cell('SATA sub'):>18}"
                  f"   gap sub-band = {r['SATA sub'][1]-r['SATA band'][1]:+.2f} dB")


if __name__ == "__main__":
    main()