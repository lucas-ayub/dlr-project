"""
Q1, part 2: what it takes for "per sub-band" to differ from "whole band".
One target at h = 500 m. On top of its geometric residual (constant delta C0 per channel) a
Doppler-DEPENDENT residual is injected: channel i is delayed by dt_i = dt0 * bxt_i / bxt_0
(the registration residual delta Dt that a formation with converging tracks would give;
phase 2 pi f dt_i, linear in Doppler). Reconstructions:
  none    no SATA
  whole   SATA whole band, delta C0 map (current code)
  sub     SATA per sub-band, delta C0 map per band (current code)
  sub_fk  SATA per sub-band, map value = residual evaluated at the band centre f_k
  lin     SATA whole band with a Doppler-dependent map: M0 (constant, fold-aware images) + M1 f
  exact   the un-delayed channel rotated by delta C0 (what a perfect correction gives)
Metric: highest ambiguity re. reference peak (as in the sweeps).
"""
import csv
import numpy as np

import sata1d as S
import sata1d.sata as SS
from sata1d.subband import subband_frequency_beam, subband_axis, sata_1d_subband
from sata1d.reconstruction import channel_spectrum, reconstruct_from_spectra
import run_geometry_sweep as G

WL = 0.25
H = 500.0
DT0 = [0.0, 0.05e-3, 0.1e-3, 0.2e-3, 0.4e-3]          # delay of channel 0 [s]


def delayed_channels(cfg, tr, p, dts):
    """Bistatic channels of target p, channel i delayed by dts[i] at the full PRF, then subsampled."""
    out = []
    f = np.fft.fftfreq(cfg.Na, 1.0 / cfg.prf)
    for i in range(cfg.Nrx):
        line = S.signal_model.raw(cfg, tr, p[None, :], tr.prx[i], tr.vrx[i])
        line = np.fft.ifft(np.fft.fft(line) * np.exp(-2j * np.pi * f * dts[i]))
        out.append(line[::cfg.Nrx])
    return np.array(out)


def kernel_lin(data, M0, M1, Tsubeff, hop, Nzp, azpos, fsub, wl):
    """sata_kernel (cola) with ph = -2 pi / wl (M0[pos] + wl M1[pos] f_b): a Doppler-dependent map."""
    data = np.asarray(data, dtype=complex); dimx = len(data)
    win = SS.triangular_window(Tsubeff)
    out = np.zeros(dimx, complex); wsum = np.zeros(dimx)
    for start in range(0, dimx, hop):
        seg = data[start:start + Tsubeff]; L = len(seg)
        if L < 2:
            break
        centre = start + 0.5 * Tsubeff
        pos = np.clip(np.round(azpos + centre).astype(int), 0, dimx - 1)
        ph = -2.0 * np.pi / wl * (M0[pos] + wl * M1[pos] * fsub)
        buf = np.zeros(Nzp, complex); buf[:L] = seg * win[:L]
        out[start:start + L] += np.fft.ifft(np.fft.fft(buf) * np.exp(-1j * ph))[:L]
        wsum[start:start + L] += win[:L]
    nz = wsum > 1e-12
    out[nz] /= wsum[nz]; out[~nz] = data[~nz]
    return out


def maps_lin(cfg, ch, dC0, dt):
    """M0: delta C0 with the fold-aware constant lambda dt m Delta at the image read by fold m
    (image offset -m X); M1: dt at every image."""
    hw = SS.sata_footprint_halfwidth(cfg)
    offsets = SS.sata_image_offsets(cfg, 0.0)
    X = offsets.max()
    pix = SS.az_pixel_of_scatterer(cfg, 0.0, ch)
    M0 = np.zeros(cfg.Na_ch); M1 = np.zeros(cfg.Na_ch)
    for o in offsets:
        m = -o / X if X > 0 else 0.0
        M0 += SS.footprint_map({pix: [dC0 + WL * dt * m * cfg.PRF_op]}, cfg.Na_ch, hw, offsets=(o,))
        M1 += SS.footprint_map({pix: [dt]}, cfg.Na_ch, hw, offsets=(o,))
    return M0, M1


def run_case(dt0):
    cfg = S.make_config([(0.0, H)], bxt_max=G.BXT_MAX, seed=G.BXT_SEED); tr = S.build_platform_tracks(cfg)
    p = cfg.scene.points[1]
    dts = dt0 * cfg.array.bxt / cfg.array.bxt[0]
    dC0 = np.array([S.residual_C0(cfg, tr, p, i) for i in range(cfg.Nrx)])
    ch = delayed_channels(cfg, tr, p, dts)
    ch0 = S.generate_channels(cfg, tr, ptgs=[p])
    prf, v, r = cfg.PRF_op, cfg.system.vs, cfg.scene.r0
    Tsub, hop, Nzp = SS.subaperture_sizing(r, prf, v, WL, G.SATA_OSF)
    fsub, azpos = SS.subaperture_axis(prf, Nzp, v, WL, r, 0.0)
    rec = {}
    rec["none"] = S.reconstruct(cfg, tr, ch)
    rec["whole"] = S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=G.SATA_OSF))
    rec["sub"] = S.reconstruct_subband(cfg, tr, ch, sata_osf=G.SATA_OSF)
    # per sub-band with the residual evaluated at f_k
    spec = []
    for k in range(cfg.Nrx):
        f_k = subband_frequency_beam(cfg, k)[0]
        corr = ch.copy()
        for i in range(cfg.Nrx):
            m = SS.build_delta_C0_array(cfg, tr, i, f_centre=f_k,
                                        residual_fn=lambda q, c, i=i, f_k=f_k: dC0[i] + WL * f_k * dts[i])
            corr[i] = sata_1d_subband(ch[i], m, rref=r, prf=prf, v=v, wl=WL, r=r, f_k=f_k,
                                      inverse=True, sata_osf=G.SATA_OSF)
        spec.append([channel_spectrum(cfg, corr[j]) for j in range(cfg.Nrx)])
    rec["sub_fk"] = reconstruct_from_spectra(cfg, tr, spec)
    # whole band with a Doppler-dependent map
    corr = ch.copy()
    for i in range(cfg.Nrx):
        M0, M1 = maps_lin(cfg, i, dC0[i], dts[i])
        corr[i] = kernel_lin(ch[i], M0, M1, Tsub, hop, Nzp, azpos, fsub, WL)
    rec["lin"] = S.reconstruct(cfg, tr, corr)
    rec["exact"] = S.reconstruct(cfg, tr, ch0 * np.exp(2j * np.pi * dC0 / WL)[:, None])
    # focus and measure
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    ref = np.roll(np.fft.ifft(np.fft.fft(S.generate_reference(cfg, tr)) * Hf), cfg.Na // 2)
    pk = np.abs(ref).max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt_amb = G.ambiguity_times(cfg)
    out = {"dt0_ms": dt0 * 1e3, "phi_edge_deg": np.degrees(2 * np.pi * cfg.abw / 2 * dt0)}
    for k, s in rec.items():
        e = np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
        out[k] = G.ambiguity_level(t, e, pk, np.array([0.0]), cfg.system.vs, dt_amb)
        out[k + "_main"] = 20 * np.log10(e[np.abs(t) <= 0.02].max() / pk)
    out["ref"] = G.ambiguity_level(t, np.abs(ref), pk, np.array([0.0]), cfg.system.vs, dt_amb)
    return out


if __name__ == "__main__":
    rows = [run_case(d) for d in DT0]
    keys = ["dt0_ms", "phi_edge_deg", "ref", "none", "whole", "sub", "sub_fk", "lin", "exact"]
    print(" ".join(f"{k:>9s}" for k in keys))
    for r in rows:
        print(" ".join(f"{r[k]:9.2f}" for k in keys) + "   main-lobe: " +
              " ".join(f"{k}={r[k + '_main']:.2f}" for k in ("none", "whole", "sub_fk", "lin", "exact")))
    with open("probes/deep_q1_doppler_residual.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
