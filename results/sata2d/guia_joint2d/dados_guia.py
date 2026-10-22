"""
Numbers and curves of the guides of the 2-D joint (results/sata2d/guia_joint2d). Run from this folder, one step per
call (every step takes one to two minutes):

    python dados_guia.py fidelidade | ponto | relevo | rcm

Writes dados_<step>.npz / .json next to this file. graficos_guia.py makes the figures from them.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "sar_reconstruction")))
import sata2d.run_sweep_joint2d as W
from sata2d.geometry import build_tracks_3d, generate_reference_3d, generate_channels_3d, residual_C0_3d, CoeffTable3D, C0_LIGHT
from sata2d.reconstruction import range_compress, reconstruct_subband_2d
from sata2d.joint2d import dem_maps, reconstruct_joint_2d, Model2D, energy, image_to_data, processed_band, image_positions

FLAT = lambda x, dy: np.zeros(np.broadcast(x, dy).shape)
dB = lambda a, b: 10 * np.log10(energy(a - b) / energy(b))


def base(tag="N4b300"):
    s = W.scene("ramp_az8_one", tag)
    return s["p"]


def on_pixel(p, n_off, m_off, h):
    """Ground point whose closest approach falls on image row Na // 2 + n_off and on range bin Nr // 2 + m_off of the data."""
    r = p.r_scan[p.Nr // 2 + m_off]
    return np.array([n_off * p.vs / p.prf, np.sqrt(r ** 2 - (p.H - h) ** 2), h]), r


def fidelidade():
    p = base()
    tr = build_tracks_3d(p)
    out = dict(bxt=[float(b) for b in p.bxt], cases=[])
    for n_off, m_off, h in ((0, 0, 0.0), (0, 0, 800.0), (300, 20, 0.0), (300, 20, 800.0), (-500, -30, 0.0), (-700, -45, 1000.0)):
        ptg, r = on_pixel(p, n_off, m_off, h)
        d = np.array([residual_C0_3d(p, tr, ptg, i) for i in range(p.Nrx)])
        row = dict(n_off=n_off, m_off=m_off, h=h, dC0_mm=[float(1e3 * v) for v in d])
        sim = range_compress(generate_channels_3d(p, tr, ptgs=ptg[None]), p.cd, p.rbw, p.rsf, axis=2)
        mono = range_compress(generate_reference_3d(p, tr, ptgs=ptg[None]), p.cd, p.rbw, p.rsf, axis=1)
        for order in (0, 1):
            M = Model2D(p, tr, np.broadcast_to(d[:, None, None], (p.Nrx, p.Na, 2 * p.Nr)).copy(), shift_order=order)
            g = np.zeros(M.shape, np.complex64)
            g[p.Na // 2 + n_off, 2 * (p.Nr // 2 + m_off)] = np.exp(-4j * np.pi * (r - M.r_ref) / p.wl)
            row[f"channels_order{order}"] = dB(M.forward(g), M.data(sim))
            row["monostatic"] = dB(M.monostatic(g), mono)
        print(row, flush=True)
        out["cases"].append(row)
    rng = np.random.default_rng(1)
    dC0 = 0.3 * rng.standard_normal((p.Nrx, p.Na, 2 * p.Nr)) * (p.bxt / 300.0)[:, None, None]
    t0 = time.time(); M = Model2D(p, tr, dC0); out["t_build"] = time.time() - t0
    a = (rng.standard_normal(M.shape) + 1j * rng.standard_normal(M.shape)).astype(np.complex64)
    b = (rng.standard_normal((p.Nrx, p.Na_ch, p.Nr)) + 1j * rng.standard_normal((p.Nrx, p.Na_ch, p.Nr))).astype(np.complex64)
    t0 = time.time(); Fa = M.forward(a); out["t_forward"] = time.time() - t0
    t0 = time.time(); Ab = M.adjoint(b); out["t_adjoint"] = time.time() - t0
    l, r_ = np.vdot(b.astype(complex), Fa.astype(complex)), np.vdot(Ab.astype(complex), a.astype(complex))
    out["adjoint_test"] = float(abs(l - r_) / abs(l))
    t0 = time.time(); out["sigma_max"] = M.largest_singular_value(); out["t_sigma"] = time.time() - t0
    out["shapes"] = dict(Na=p.Na, Nr=p.Nr, Na_ch=p.Na_ch, Nrx=p.Nrx)
    print({k: v for k, v in out.items() if k != "cases"})
    json.dump(out, open(os.path.join(HERE, "dados_fidelidade.json"), "w"), indent=1)


def _levels(p, tr, M, g, ref, none, band, xs):
    lo, hi = band
    im = dict(ref_whole=ref, ref_cut=processed_band(p, ref), joint_whole=image_to_data(M, g), none_whole=none,
              ref=processed_band(p, M.to_common_band(ref, lo, hi)), none=M.to_common_band(none, lo, hi), joint=image_to_data(M, g, band))
    env = W.envelopes(p, tr, im)
    lev = {k: W.ambiguity_level(p, env[k], env["ref" if k in ("ref", "none", "joint") else "ref_whole"].max(), xs) for k in env}
    return env, lev


def _solve(p, tr, terrain):
    ch = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
    ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)
    dC0, hmap = dem_maps(p, tr, terrain, osr=2)
    M = Model2D(p, tr, dC0)
    e0 = energy(M.data(ch))
    hist = []
    rec, info = reconstruct_joint_2d(p, tr, ch, dC0, model=M, return_info=True, callback=lambda n, g, r: hist.append(10 * np.log10(energy(r) / e0)))
    none = reconstruct_subband_2d(p, tr, ch, CoeffTable3D(p, tr, n_nodes=16), use_sata=False, verbose=False)
    return ch, ref, dC0, hmap, M, info, none, np.array(hist)


def ponto():
    """One target on flat terrain, on a pixel at the reference range of the model: the range band of every channel."""
    p = base()
    ptg, r = on_pixel(p, 0, 0, 0.0)
    p.extra_offsets = ((0.0, float(ptg[1] - p.y0), 0.0),)
    tr = build_tracks_3d(p)
    ch, ref, dC0, hmap, M, info, none, hist = _solve(p, tr, FLAT)
    g = info["image"]
    xs = p.points[:, 0]
    env, lev = _levels(p, tr, M, g, ref, none, info["band"], xs)
    lo3, hi3 = M.common_band(p.Nrx - 1)
    env3 = W.envelopes(p, tr, dict(ref=processed_band(p, M.to_common_band(ref, lo3, hi3)), joint=image_to_data(M, g, (lo3, hi3))))
    lev["joint_3of4"] = W.ambiguity_level(p, env3["joint"], env3["ref"].max(), xs)
    # error of the output (whole band) against the reference, per range frequency, inside the processed azimuth band
    fa = np.fft.fftfreq(p.Na, 1 / p.prf)
    sel = np.abs(fa) <= p.abw / 2
    U, R = np.fft.fft2(M.monostatic(g))[sel], np.fft.fft2(ref)[sel]
    err = (np.abs(U - R) ** 2).sum(axis=0) / ((np.abs(R) ** 2).sum(axis=0).max())
    sig = (np.abs(R) ** 2).sum(axis=0) / ((np.abs(R) ** 2).sum(axis=0).max())
    shift = np.array([float(M.spectral_shift(i)[p.Na // 2, p.Nr]) for i in range(p.Nrx)])
    print(lev, "| shift [MHz]", shift / 1e6, "| unexplained %.1f dB" % (20 * np.log10(info["unexplained"])))
    np.savez_compressed(os.path.join(HERE, "dados_ponto.npz"), **{f"env_{k}": v.astype(np.float32) for k, v in env.items()},
                        err=err, sig=sig, fr=np.fft.fftfreq(p.Nr, 1 / p.rsf), shift=shift, hist=hist, bxt=p.bxt,
                        levels=json.dumps(lev), unexplained=info["unexplained"], iterations=info["iterations"])


def relevo():
    """Five targets on the isolated mountain, four channels with baselines up to 500 m."""
    s = W.scene("mountain_diag5", "N4b500")
    p, tr, terrain = s["p"], s["tr"], s["terrain"]
    ch, ref, dC0, hmap, M, info, none, hist = _solve(p, tr, terrain)
    g = info["image"]
    xs = p.points[:, 0]
    env, lev = _levels(p, tr, M, g, ref, none, info["band"], xs)
    lo, hi = info["band"]
    i1 = int(np.argmax(np.abs(p.bxt)))
    print(lev, "| unexplained %.1f dB" % (20 * np.log10(info["unexplained"])))
    x, r = image_positions(p, 2)
    np.savez_compressed(os.path.join(HERE, "dados_relevo.npz"), **{f"env_{k}": v.astype(np.float32) for k, v in env.items()},
                        hmap=hmap[::8].astype(np.float32), dC0=dC0[i1, ::8].astype(np.float32), shift=M.spectral_shift(i1)[::8],
                        kept=(np.clip(hi - lo, 0, None) / p.rbw)[::8].astype(np.float32), x=x[::8], r=r, r0=p.r0, hist=hist, bxt=p.bxt,
                        img=np.abs(g[::2]).astype(np.float32), img_common=np.abs(M.band_limit(g, lo, hi)[::2]).astype(np.float32),
                        xy=s["xy"], hs=s["hs"], y0=p.y0, H=p.H,
                        levels=json.dumps(lev), unexplained=info["unexplained"], iterations=info["iterations"])


def rcm():
    """The range-compressed echo of one target (monostatic, full PRF): in time and in the range-Doppler domain."""
    p = base()
    ptg, r = on_pixel(p, 0, 0, 0.0)
    tr = build_tracks_3d(p)
    ref = range_compress(generate_reference_3d(p, tr, ptgs=ptg[None]), p.cd, p.rbw, p.rsf, axis=1)
    m = p.Nr // 2
    up = 8                                                     # range oversampling of the picture
    F = np.fft.fft(ref, axis=1)
    Z = np.zeros((p.Na, up * p.Nr), complex)
    Z[:, :p.Nr // 2] = F[:, :p.Nr // 2]; Z[:, -p.Nr // 2:] = F[:, -p.Nr // 2:]
    fine = np.fft.ifft(Z, axis=1) * up
    c0, c1 = up * (m - 3), up * (m + 6)
    rd = np.fft.fftshift(np.fft.fft(fine[:, c0:c1], axis=0), axes=0)
    np.savez_compressed(os.path.join(HERE, "dados_rcm.npz"), time=np.abs(fine[::4, c0:c1]).astype(np.float32), rd=np.abs(rd[::4]).astype(np.float32),
                        ta=p.ta[::4], fa=np.fft.fftshift(np.fft.fftfreq(p.Na, 1 / p.prf))[::4],
                        dr=(np.arange(c0, c1) - up * m) * (C0_LIGHT / (2 * p.rsf)) / up, r=r, wl=p.wl, vs=p.vs, abw=p.abw)
    print("ok", fine.shape)


if __name__ == "__main__":
    t0 = time.time()
    {"fidelidade": fidelidade, "ponto": ponto, "relevo": relevo, "rcm": rcm}[sys.argv[1]]()
    print("%.0f s" % (time.time() - t0))
