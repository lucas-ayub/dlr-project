"""
Why do SATA whole band and SATA per sub-band give the same numbers?

T1  identity      : are the two reconstructions bit-identical, or only close?
T2  kernel read   : bin by bin, which cell each kernel reads and which delta_C0 it applies
                    (azpos_k - azpos_whole in units of X; energy-weighted phase mismatch)
T3  no alias map  : maps with the direct image only (a DEM that knows only the true terrain)
T4  collision     : two targets exactly X apart with different heights (and a 4 km control)

Run from sata1d_code/:  python -m probes.probe_whole_vs_sub   (or python probes/probe_whole_vs_sub.py)
"""
import os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import sata1d as S
from sata1d.sata import (subaperture_sizing, subaperture_axis, sata_footprint_halfwidth,
                         sata_image_offsets, footprint_map, az_pixel_of_scatterer, sata_1d)
from sata1d.subband import (subband_frequency_beam, residual_C0_subband, subband_axis, sata_1d_subband)
from sata1d.reconstruction import channel_spectrum, reconstruct_from_spectra
import run_geometry_sweep as G

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_whole_vs_sub")
OSF, BXT = G.SATA_OSF, G.BXT_MAX


def X_cells(cfg):
    prf, v, wl, r = cfg.PRF_op, cfg.system.vs, cfg.system.wl, cfg.scene.r0
    return r * np.tan(np.arcsin(wl * prf / (2 * v))) / v * prf


def dc0_map(cfg, tr, ch, k=None, images=True):
    """Same map as build_delta_C0_array / build_delta_C0_subband_array; images=False keeps
    only the direct image (offset 0)."""
    f_c = 0.0 if k is None else subband_frequency_beam(cfg, k)[0]
    res = (lambda p: S.residual_C0(cfg, tr, p, ch)) if k is None else \
          (lambda p: residual_C0_subband(cfg, tr, p, ch, k))
    by = defaultdict(list)
    for (dx, dy, dh) in cfg.scene.extra_offsets:
        by[az_pixel_of_scatterer(cfg, dx, ch)].append(res(cfg.scene.ptg + np.array([dx, dy, dh])))
    offs = sata_image_offsets(cfg, f_c) if images else (0.0,)
    return footprint_map(by, cfg.Na_ch, sata_footprint_halfwidth(cfg), offs)


def correct_whole(cfg, tr, chs, images=True):
    out = chs.copy()
    for i in range(cfg.Nrx):
        out[i] = sata_1d(chs[i], dc0_map(cfg, tr, i, None, images), cfg.scene.r0, cfg.PRF_op,
                         cfg.system.vs, cfg.system.wl, cfg.scene.r0, 0.0, True, OSF)
    return out


def correct_sub(cfg, tr, chs, k, images=True):
    f_k = subband_frequency_beam(cfg, k)[0]
    out = chs.copy()
    for i in range(cfg.Nrx):
        out[i] = sata_1d_subband(chs[i], dc0_map(cfg, tr, i, k, images), cfg.scene.r0, cfg.PRF_op,
                                 cfg.system.vs, cfg.system.wl, cfg.scene.r0, f_k, True, OSF)
    return out


def recon_sub(cfg, tr, chs, images=True):
    spec = []
    for k in range(cfg.Nrx):
        c = correct_sub(cfg, tr, chs, k, images)
        spec.append([channel_spectrum(cfg, c[j]) for j in range(cfg.Nrx)])
    return reconstruct_from_spectra(cfg, tr, spec)


def run_scene(targets, images=True, extra=False):
    cfg = S.make_config(targets, bxt_max=BXT, seed=G.BXT_SEED)
    tr = S.build_platform_tracks(cfg)
    ch = S.generate_channels(cfg, tr)
    cfg0 = S.make_config(targets, bxt_max=0.0, seed=G.BXT_SEED)
    tr0 = S.build_platform_tracks(cfg0)
    rec = {"flat": S.reconstruct(cfg0, tr0, S.generate_channels(cfg0, tr0)),
           "none": S.reconstruct(cfg, tr, ch),
           "whole": S.reconstruct(cfg, tr, correct_whole(cfg, tr, ch, images)),
           "sub": recon_sub(cfg, tr, ch, images),
           "oracle": S.reconstruct(cfg, tr, G.oracle_channels(cfg, tr))}
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    H = np.conj(np.fft.fft(sref))
    foc = {k: np.roll(np.fft.ifft(np.fft.fft(v) * H), cfg.Na // 2) for k, v in rec.items()}
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    pk = np.abs(foc["flat"]).max()
    xs = np.array([x for x, _ in targets])
    L = lambda e: G.ambiguity_level(t, e, pk, xs, cfg.system.vs, G.ambiguity_times(cfg))
    m = {k: L(np.abs(foc[k] - foc["flat"])) for k in ("none", "whole", "sub", "oracle")}
    return cfg, tr, ch, rec, foc, m


def fmt(m):
    return "  ".join(f"{k}: {v:6.1f}" for k, v in m.items())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    np.set_printoptions(precision=3, suppress=True)
    one = [(0.0, 240.0)]

    # ------------------------------------------------------------------ T1
    print("=" * 78, "\nT1  identity: 1 target, h = 240 m, bxt ~ U(0,100)")
    cfg, tr, ch, rec, foc, m = run_scene(one)
    # the package functions themselves (what the sweeps call)
    rw = S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=OSF))
    rs = S.reconstruct_subband(cfg, tr, ch, sata_osf=OSF)
    print(f"   package outputs bit-identical?          {np.array_equal(rw, rs)}")
    print(f"   ||sub - whole|| / ||whole||            = {np.linalg.norm(rs - rw) / np.linalg.norm(rw):.3e}")
    print(f"   probe maps == package maps (whole)      "
          f"{all(np.allclose(dc0_map(cfg, tr, i), S.build_delta_C0_array(cfg, tr, i)) for i in range(cfg.Nrx))}")
    print(f"   probe maps == package maps (sub, all k) "
          f"{all(np.allclose(dc0_map(cfg, tr, i, k), S.build_delta_C0_subband_array(cfg, tr, i, k)) for i in range(cfg.Nrx) for k in range(cfg.Nrx))}")
    wholeset = S.sata_channels(cfg, tr, ch, sata_osf=OSF)
    for k in range(cfg.Nrx):
        sk = S.sata_channels_subband(cfg, tr, ch, k, sata_osf=OSF)
        print(f"   channel set k={k}: ||set_k - whole set|| / ||whole set|| = "
              f"{np.linalg.norm(sk - wholeset) / np.linalg.norm(wholeset):.2e}")
    print(f"   topographic ambiguity [dB re flat peak]: {fmt(m)}")

    # ------------------------------------------------------------------ T2
    print("=" * 78, "\nT2  kernel read: which cell and which delta_C0 each bin uses")
    Ts, hop, Nzp = subaperture_sizing(cfg.scene.r0, cfg.PRF_op, cfg.system.vs, cfg.system.wl, OSF)
    Xc = X_cells(cfg)
    fw, aw = subaperture_axis(cfg.PRF_op, Nzp, cfg.system.vs, cfg.system.wl, cfg.scene.r0, 0.0)
    print(f"   Tsubeff = {Ts}, Nzp = {Nzp}, X = {Xc:.1f} cells = {Xc * cfg.system.vs / cfg.PRF_op / 1e3:.3f} km")
    for i in range(cfg.Nrx):
        dC0w = S.residual_C0(cfg, tr, cfg.scene.points[1], i)
        dC0k = [residual_C0_subband(cfg, tr, cfg.scene.points[1], i, k) for k in range(cfg.Nrx)]
        print(f"   ch {i}: dC0 whole = {dC0w * 1e3:+8.4f} mm | per band: " +
              " ".join(f"{d * 1e3:+8.4f}" for d in dC0k) + " mm")
    print("   azpos_k - azpos_whole (per FFT bin, in units of X) -> set of values:")
    for k in range(cfg.Nrx):
        f_k = subband_frequency_beam(cfg, k)[0]
        fk, ak = subband_axis(cfg.PRF_op, Nzp, cfg.system.vs, cfg.system.wl, cfg.scene.r0, f_k)
        d = np.round((fk - fw) / cfg.PRF_op).astype(int)
        print(f"     k={k} (f_k = {f_k:+6.0f} Hz): fold shift m = {sorted(set(d.tolist()))}, "
              f"max |(azpos_k-azpos_w) - m X| = {np.abs((ak - aw) - np.interp(fk, fw[np.argsort(fw)], aw[np.argsort(fw)]) * 0 - (ak - aw)).max():.1f}")
    # energy-weighted mismatch of the applied phase, subaperture by subaperture
    print("   applied phase, set k vs whole, energy-weighted over all sub-apertures and bins:")
    rows = []
    for images in (True, False):
        for k in range(cfg.Nrx):
            f_k = subband_frequency_beam(cfg, k)[0]
            fk, ak = subband_axis(cfg.PRF_op, Nzp, cfg.system.vs, cfg.system.wl, cfg.scene.r0, f_k)
            num = den = same = right_w = right_k = 0.0
            for i in range(cfg.Nrx):
                mw, mk = dc0_map(cfg, tr, i, None, images), dc0_map(cfg, tr, i, k, images)
                truth = S.residual_C0(cfg, tr, cfg.scene.points[1], i)
                for st in range(0, cfg.Na_ch - Ts, hop):
                    seg = np.zeros(Nzp, complex); seg[:Ts] = ch[i, st:st + Ts] * S.sata.triangular_window(Ts)
                    E = np.abs(np.fft.fft(seg)) ** 2
                    c = st + 0.5 * Ts
                    pw = np.clip(np.round(aw + c).astype(int), 0, cfg.Na_ch - 1)
                    pk_ = np.clip(np.round(ak + c).astype(int), 0, cfg.Na_ch - 1)
                    vw, vk = mw[pw], mk[pk_]
                    den += E.sum(); same += E[np.isclose(vw, vk, atol=1e-9)].sum()
                    right_w += E[np.isclose(vw, truth, atol=1e-6)].sum()
                    right_k += E[np.isclose(vk, truth, atol=1e-6)].sum()
            rows.append((images, k, same / den, right_w / den, right_k / den))
            print(f"     map {'with' if images else 'NO  '} alias images, k={k}: energy where set k applies the same "
                  f"dC0 as whole = {100 * same / den:5.1f} %; energy corrected with the true dC0: "
                  f"whole {100 * right_w / den:5.1f} %, set k {100 * right_k / den:5.1f} %")

    # ------------------------------------------------------------------ T3
    print("=" * 78, "\nT3  maps without alias images (direct image only)")
    for name, tg in {"1 target h=240": one,
                     "ramp 5x200 m, 20 deg": [(x, (x + 400) * np.tan(np.radians(20))) for x in np.arange(-400, 401, 200.0)],
                     "2 targets 2.6 km (240/-120)": [(-1300.0, 240.0), (1300.0, -120.0)]}.items():
        _, _, _, _, _, mi = run_scene(tg, images=True)
        _, _, _, _, _, mn = run_scene(tg, images=False)
        print(f"   {name:30s} with images : {fmt(mi)}")
        print(f"   {'':30s} no images   : {fmt(mn)}")

    # ------------------------------------------------------------------ T4
    print("=" * 78, "\nT4  collision: two targets X apart vs a 4 km control")
    Xm = Xc * cfg.system.vs / cfg.PRF_op
    for name, tg in {f"X apart ({Xm / 1e3:.2f} km), 240/-120 m": [(-Xm / 2, 240.0), (Xm / 2, -120.0)],
                     "4.0 km apart,       240/-120 m": [(-2000.0, 240.0), (2000.0, -120.0)]}.items():
        for images in (True, False):
            c, t_, chs, rec, foc, mm = run_scene(tg, images=images)
            print(f"   {name}  {'images' if images else 'no img'}: {fmt(mm)}")
            if images and "X apart" in name:
                np.savez_compressed(os.path.join(OUT, "t4_collision.npz"),
                                    **{k: np.abs(v).astype(np.float32) for k, v in foc.items()},
                                    t=(np.arange(c.Na) - c.Na // 2) / c.prf)
    np.save(os.path.join(OUT, "t2_rows.npy"), np.array(rows, dtype=float))
    print("done")
