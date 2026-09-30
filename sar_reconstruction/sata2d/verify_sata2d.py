"""
Verification of the SATA 2D chain (residual maps + kernel + two-step reconstruction).

V1  kernel axis: bins carry the absolute Doppler of the band; whole band == 1D kernel
V2  zero map == identity, constant map == one global rotation
V3  scatterer cell: closest approach of the bistatic pair, incl. bat/2
V4  range bin: the compressed energy of each channel sits in the deposited bin (+- hw_r)
V5  map: one target, block +- hw at x_t and at the fold images, zero elsewhere;
    sub-band k uses the one-sided fold set
V6  energy of a target inside its map, per channel (whole band and per sub-band)
V7  corrected channel vs the ideal rotation exp(+j 2 pi dC0 / lambda)
V8  one target, h = 100..400 m: ambiguity of the reconstruction vs the reference
V9  sign: remove / inject / remove twice
V10 two targets: each reads its own dC0 (cross-talk)
V11 cross-track ramp: targets at the same azimuth in different range bins get
    their own dC0 on their own range bins
V12 bxt = 0 -> no correction (identity)
V13 whole band vs per sub-band on iso-range scenes
V14 range-cell-migration lookup: the map a Doppler bin of data range bin m reads holds the
    scatterer only if its energy at that Doppler reaches m (|n + rcm(f) - m| <= gate);
    one target with and without the lookup
V15 signal model of the reconstruction: every channel (at the full PRF) equals the monostatic
    reference times exp(-2 pi j (C0/lambda(f_r) + (Dt - C1) f_a + C2 f_a^2 lambda)); with the
    band-limited pulse of the generator the error is -30 dB or lower (the sampled chirp of the
    reference generator gave -6 dB for b_xt = 64 m: a sub-sample delay changes which samples the
    5.7-sample pulse covers)
"""
from __future__ import annotations

import os
import sys

import numpy as np

if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)

from .arrays import make_params_dpca
from .geometry import (build_tracks_3d, generate_reference_3d, generate_channels_3d,
                       CoeffTable3D, residual_C0_3d, iso_range_offset)
from .reconstruction import (range_compress, ResidualMaps, reconstruct_subband_2d,
                             channels_subband_3d, scatterer_range, scatterer_range_bistatic,
                             range_bin_of, az_pixel_of, sata_footprint_halfwidth,
                             sata_image_offsets, azimuth_envelope, worst_ambiguity)
from .sata import subaperture_grid, sata_1d, subband_centre_frequency

OSF = 4
RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def params(specs=(), offsets=None, bxt_max=100.0):
    p = make_params_dpca(specs=tuple(specs), bxt_mode="random", bxt_max=bxt_max, seed=0)
    if offsets is not None:
        p.extra_offsets = tuple(tuple(float(v) for v in o) for o in offsets)
    return p, build_tracks_3d(p)


def iso(x, h, p):
    _, dy, dh = iso_range_offset(p.r0, p.H, p.h0, h)
    return (x, dy, dh)


def data(p, tr):
    ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)
    ch = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
    return ref, ch


def ambiguities(p, tr, ref, recs, xs):
    e_ref = azimuth_envelope(p, tr, ref)
    out = {"ref": worst_ambiguity(p, e_ref, e_ref, xs)}
    for tag, rec in recs.items():
        out[tag] = worst_ambiguity(p, azimuth_envelope(p, tr, rec), e_ref, xs)
    return out


def oracle_channels(p, tr):
    out = 0
    for ptg in p.points:
        ch = range_compress(generate_channels_3d(p, tr, ptgs=ptg[None, :]), p.cd, p.rbw, p.rsf, axis=2)
        d = np.array([residual_C0_3d(p, tr, ptg, i) for i in range(p.Nrx)])
        out = out + ch * np.exp(2j * np.pi * d / p.wl)[:, None, None]
    return out


# ---------------------------------------------------------------- V1 axis
p0, tr0 = params()
g = subaperture_grid(p0.r0, p0.PRF_op, p0.vs, p0.wl, p0.r0, f_centre=0.0, sata_osf=OSF)
ff = np.fft.fftfreq(g.Nzp, 1 / p0.PRF_op)
ok1 = np.allclose(g.fsub, ff)
lines = []
for k in range(p0.Nrx):
    fk = subband_centre_frequency(k, p0.Nrx, p0.prf)
    gk = subaperture_grid(p0.r0, p0.PRF_op, p0.vs, p0.wl, p0.r0, f_centre=fk, sata_osf=OSF)
    inband = np.all((gk.fsub >= fk - p0.PRF_op / 2) & (gk.fsub < fk + p0.PRF_op / 2))
    alias = np.allclose(np.mod(gk.fsub - ff, p0.PRF_op), 0)
    ok1 &= inband and alias
    lines.append(f"k={k}: [{gk.fsub.min():.0f},{gk.fsub.max():.0f}] Hz")
check("V1 kernel axis", ok1, f"whole = fftfreq; " + ", ".join(lines) + f"; Tsub={g.Tsubeff}, Nzp={g.Nzp}")

# ---------------------------------------------------------------- V2 identity / constant
p, tr = params([(0.0, 240.0)])
ref, ch = data(p, tr)
nb = range_bin_of(p, scatterer_range(p, p.points[0]))
line = ch[0, :, nb]
z = sata_1d(line, np.zeros(p.Na_ch), p.r0, p.PRF_op, p.vs, p.wl, p.r0, inverse=True, sata_osf=OSF)
e_id = np.linalg.norm(z - line) / np.linalg.norm(line)
c = 0.0137
z = sata_1d(line, np.full(p.Na_ch, c), p.r0, p.PRF_op, p.vs, p.wl, p.r0, inverse=True, sata_osf=OSF)
e_c = np.linalg.norm(z - line * np.exp(2j * np.pi * c / p.wl)) / np.linalg.norm(line)
check("V2 zero map = identity, constant map = rotation", e_id < 1e-12 and e_c < 1e-12, f"{e_id:.1e}, {e_c:.1e}")

# ---------------------------------------------------------------- V3 azimuth cell
errs = []
for i in range(p.Nrx):
    ptx, prx, _, _ = tr
    idx = p.idx_dec
    R = np.sqrt(np.sum((ptx[idx] - p.points[0]) ** 2, 1)) + np.sqrt(np.sum((prx[i][idx] - p.points[0]) ** 2, 1))
    errs.append(int(np.argmin(R)) - az_pixel_of(p, p.points[0][0], i))
check("V3 scatterer cell (bat/2)", max(abs(e) for e in errs) <= 0.5, f"cell error per channel {np.round(errs, 2)}")

# ---------------------------------------------------------------- V4 range bin / hw_r
maps = ResidualMaps(p, tr)
ok4, det = True, []
for i in range(p.Nrx):
    n_dep = maps.deposits[i][0][0]
    prof = np.abs(ch[i]) ** 2
    prof = prof.sum(0) / prof.sum()
    inside = prof[max(0, n_dep - 2):n_dep + maps.hw_r + 1].sum()
    cen = float(np.sum(np.arange(p.Nr) * prof)) - n_dep
    ok4 &= inside > 0.9 and 0 <= cen <= 1.5
    det.append(f"ch{i}: bin {n_dep}, centroid {cen:+.2f} (model {maps.centroid:+.2f}), {100 * inside:.1f}% in [-2, +{maps.hw_r}]")
check("V4 range bin and hw_r", ok4, "; ".join(det))

# ---------------------------------------------------------------- V5 map shape
hw = sata_footprint_halfwidth(p)
X = sata_image_offsets(p, 0.0)
xt = int(round(az_pixel_of(p, p.points[0][0], 0)))
dmap, act = maps.get(0, 0.0)
d0 = residual_C0_3d(p, tr, p.points[0], 0)
nb0 = maps.deposits[0][0][0]
row = dmap[nb0]
expect = np.zeros(p.Na_ch)
for off in X:
    lo, hi = int(round(off)) + xt - hw, int(round(off)) + xt + hw
    expect[max(0, lo):hi + 1] = d0
ok5 = np.allclose(row, expect) and set(act) == set(range(nb0 - 2, nb0 + maps.hw_r + 1))
fk = subband_centre_frequency(2, p.Nrx, p.prf)
Xk = sata_image_offsets(p, fk)
rowk = maps.get(0, fk)[0][nb0]
expk = np.zeros(p.Na_ch)
for off in Xk:
    lo, hi = int(round(off)) + xt - hw, int(round(off)) + xt + hw
    expk[max(0, lo):min(p.Na_ch - 1, hi) + 1] = d0
ok5 &= np.allclose(rowk, expk)
check("V5 map: blocks at x_t + mX", ok5,
      f"hw={hw} cells, X={X[-1]:.0f} cells, whole m={np.round(X / X[-1]).astype(int).tolist()}, "
      f"k=2 m={np.round(Xk / X[-1]).astype(int).tolist()}, active bins {act}")

# ---------------------------------------------------------------- V6 energy inside the map
det, ok6 = [], True
for i in range(p.Nrx):
    line = ch[i, :, nb]
    gk = subaperture_grid(p.r0, p.PRF_op, p.vs, p.wl, p.r0, 0.0, OSF)
    hop, L, Nzp = gk.Tsub, gk.Tsubeff, gk.Nzp
    ramp = np.arange(hop) / (hop - 1)
    win = np.concatenate((ramp, ramp[::-1]))[:L]
    e_in = e_tot = 0.0
    dm = maps.get(i, 0.0)[0][maps.deposits[i][0][0]]
    for start in range(0, p.Na_ch - L + 1, hop):
        buf = np.zeros(Nzp, complex); buf[:L] = line[start:start + L] * win
        S = np.abs(np.fft.fft(buf)) ** 2
        posaux = np.clip(np.round(gk.azpos + start + 0.5 * L).astype(int), 0, p.Na_ch - 1)
        e_tot += S.sum(); e_in += S[dm[posaux] != 0].sum()
    fr = e_in / e_tot
    ok6 &= fr > 0.99
    det.append(f"{100 * fr:.2f}%")
check("V6 energy read inside the map", ok6, "per channel " + ", ".join(det))

# ---------------------------------------------------------------- V7 corrected channel vs ideal rotation
cw = channels_subband_3d(p, tr, ch, 0, f_centre=0.0, sata_osf=OSF, maps=maps)
det, ok7 = [], True
for i in range(p.Nrx):
    d = residual_C0_3d(p, tr, p.points[0], i)
    ideal = ch[i, :, nb] * np.exp(2j * np.pi * d / p.wl)
    e = np.linalg.norm(cw[i, :, nb] - ideal) / np.linalg.norm(ideal)
    ok7 &= e < 0.05
    det.append(f"ch{i}: {100 * e:.2f}% (dC0={1e2 * d:+.2f} cm)")
check("V7 corrected channel vs ideal rotation", ok7, "; ".join(det))

# ---------------------------------------------------------------- V8 one target vs height
tab = CoeffTable3D(p, tr, n_nodes=16)
det, ok8 = [], True
for h in (0.0, 100.0, 240.0, 400.0):
    ph, th = params([(0.0, h)] if h else [])
    rf, chh = data(ph, th)
    tb = CoeffTable3D(ph, th, n_nodes=16)
    recs = {}
    for tag, kw in (("none", dict(use_sata=False)), ("whole", dict(use_sata="whole")), ("sub", dict(use_sata="subband"))):
        recs[tag] = reconstruct_subband_2d(ph, th, chh, tb, sata_osf=OSF, verbose=False, **kw)
    a = ambiguities(ph, th, rf, recs, [0.0])
    if h == 0:
        floor = a["none"]
    ok8 &= a["whole"] < floor + 1.0 and a["sub"] < floor + 1.0
    det.append(f"h={h:.0f}: ref {a['ref']:.1f}, none {a['none']:.1f}, whole {a['whole']:.1f}, sub {a['sub']:.1f} dB")
check("V8 one target, ambiguity vs height (h = 0: floor of the 2-D reconstruction)", ok8, "; ".join(det))

# ---------------------------------------------------------------- V9 sign
rem = channels_subband_3d(p, tr, ch, 0, f_centre=0.0, sata_osf=OSF, maps=maps, remove=True)
inj = channels_subband_3d(p, tr, ch, 0, f_centre=0.0, sata_osf=OSF, maps=maps, remove=False)
rem2 = channels_subband_3d(p, tr, rem, 0, f_centre=0.0, sata_osf=OSF, maps=maps, remove=True)
a9 = ambiguities(p, tr, ref, {j: reconstruct_subband_2d(p, tr, c_, tab, use_sata=False, verbose=False)
                              for j, c_ in enumerate((rem, inj, rem2))}, [0.0])
a9 = [a9[0], a9[1], a9[2]]
check("V9 sign: remove / inject / remove twice", a9[0] < a9[1] - 6 and a9[0] < a9[2] - 6,
      f"{a9[0]:.1f} / {a9[1]:.1f} / {a9[2]:.1f} dB")

# ---------------------------------------------------------------- V10 cross-talk, two targets
p2, tr2 = params([(-1500.0, 240.0), (1500.0, -120.0)])
ref2, ch2 = data(p2, tr2)
maps2 = ResidualMaps(p2, tr2)
nb2 = range_bin_of(p2, scatterer_range(p2, p2.points[0]))
det, ok10 = [], True
for j, ptg in enumerate(p2.points):
    one = range_compress(generate_channels_3d(p2, tr2, ptgs=ptg[None, :]), p2.cd, p2.rbw, p2.rsf, axis=2)
    cw = channels_subband_3d(p2, tr2, one, 0, f_centre=0.0, sata_osf=OSF, maps=maps2)
    for i in range(p2.Nrx):
        d = residual_C0_3d(p2, tr2, ptg, i)
        ideal = one[i, :, nb2] * np.exp(2j * np.pi * d / p2.wl)
        e = np.linalg.norm(cw[i, :, nb2] - ideal) / np.linalg.norm(ideal)
        ok10 &= e < 0.05
    det.append(f"target {j}: ch-avg err {100 * e:.2f}%")
check("V10 two targets read their own dC0", ok10, "; ".join(det))

# ---------------------------------------------------------------- V11 cross-track ramp: own range bins
N, alpha = 5, 5.0
dy = np.linspace(0.10, 1.0, N) * 2000.0
off = np.stack([np.zeros(N), dy, dy * np.tan(np.radians(alpha))], 1)
p3, tr3 = params(offsets=off)
maps3 = ResidualMaps(p3, tr3)
det, ok11 = [], True
for i in (0,):
    bins = [n for n, _, _ in maps3.deposits[i]]
    ds = [d for _, _, d in maps3.deposits[i]]
    xt = int(round(az_pixel_of(p3, 0.0, i)))
    got = [maps3.get(i, 0.0)[0][n, xt] for n in bins]
    ok11 &= np.allclose(got, ds) and len(set(bins)) == N
    det.append(f"ch{i}: bins {bins}, dC0 [cm] {np.round(1e2 * np.array(ds), 2).tolist()}, read {np.allclose(got, ds)}")
check("V11 cross-track ramp: each range bin its own dC0", ok11, "; ".join(det))

# ---------------------------------------------------------------- V12 bxt = 0
p4, tr4 = params([(0.0, 240.0)], bxt_max=0.0)
ref4, ch4 = data(p4, tr4)
m4 = ResidualMaps(p4, tr4)
mx = max(np.abs(m4.get(i, 0.0)[0]).max() for i in range(p4.Nrx))
cw4 = channels_subband_3d(p4, tr4, ch4, 0, f_centre=0.0, sata_osf=OSF, maps=m4)
e12 = np.linalg.norm(cw4 - ch4) / np.linalg.norm(ch4)
check("V12 bxt = 0 -> identity", mx < 1e-4 and e12 < 1e-6, f"max|dC0| = {mx * 1e3:.2e} mm, change {e12:.1e}")

# ---------------------------------------------------------------- V13 whole vs sub, iso-range scenes
scenes = {"1 target 240 m": [(0.0, 240.0)],
          "ramp 5 x 200 m, 15 deg": [(x, (x + 400) * np.tan(np.radians(15))) for x in np.arange(-400, 401, 200.0)],
          "random 5 x 200 m": list(zip((-400.0, -200.0, 0.0, 200.0, 400.0), (0, 20, 48, 96, 43)))}
det, ok13 = [], True
for name, tg in scenes.items():
    ps, ts = params(tg)
    rs, cs = data(ps, ts)
    tbs = CoeffTable3D(ps, ts, n_nodes=16)
    xs = [x for x, _ in tg]
    recs = {}
    for tag, kw in (("none", dict(use_sata=False)), ("whole", dict(use_sata="whole")), ("sub", dict(use_sata="subband"))):
        recs[tag] = reconstruct_subband_2d(ps, ts, cs, tbs, sata_osf=OSF, verbose=False, **kw)
    recs["oracle"] = reconstruct_subband_2d(ps, ts, oracle_channels(ps, ts), tbs, use_sata=False, verbose=False)
    a = ambiguities(ps, ts, rs, recs, xs)
    ok13 &= abs(a["whole"] - a["sub"]) < 1.0 and a["whole"] < a["oracle"] + 2.0
    det.append(f"{name}: ref {a['ref']:.1f}, none {a['none']:.1f}, whole {a['whole']:.1f}, sub {a['sub']:.1f}, oracle {a['oracle']:.1f} dB")
check("V13 whole vs sub-band on iso-range scenes", ok13, "; ".join(det))

# ---------------------------------------------------------------- V14 RCM lookup
from .reconstruction import range_cell_migration
rows = maps.rows(0, 0.0)
n0, pix0, d0 = maps.deposits[0][0]
edges = maps.group_edges(0.0)
f_rep = 0.5 * (edges[:-1] + edges[1:])
ok14, det = True, []
for g_, fg in enumerate(f_rep):
    q = float(range_cell_migration(p, fg))
    have = sorted(m for m, D in rows.items() if D[g_, pix0] != 0.0)
    want = [m for m in range(p.Nr) if abs(n0 + q - m) <= maps.gate]
    ok14 &= have == want
    if g_ in (0, maps.n_groups // 2, maps.n_groups - 1):
        det.append(f"group {g_} (f={fg:+.0f} Hz, rcm={q:.2f} bins): bins {have}")
rec_a = reconstruct_subband_2d(p, tr, ch, tab, use_sata="whole", maps=maps, rcm_lookup=True, verbose=False)
rec_b = reconstruct_subband_2d(p, tr, ch, tab, use_sata="whole", maps=maps, rcm_lookup=False, verbose=False)
a14 = ambiguities(p, tr, ref, {"rcm": rec_a, "bin": rec_b}, [0.0])
ok14 &= a14["rcm"] < floor + 1.0 and a14["bin"] < floor + 1.0
check("V14 RCM lookup: gated bins per Doppler group; one target with/without", ok14,
      "; ".join(det) + f"; ambiguity rcm {a14['rcm']:.1f} / bin {a14['bin']:.1f} dB (floor {floor:.1f})")

# ---------------------------------------------------------------- V15 signal model per channel
from .geometry import get_raw_data_3d, get_coeff_nu_3d
ptx_, prx_, vtx_, vrx_ = tr
S_ref = np.fft.fft2(ref)
fa_ = np.fft.fftfreq(p.Na, 1 / p.prf); fr_ = np.fft.fftfreq(p.Nr, 1 / p.rsf)
wl_arr = 3.0e8 / (3.0e8 / p.wl + fr_)
inb = (np.abs(fa_) < p.abw / 2)[:, None] & (np.abs(fr_) < p.rbw / 2)[None, :]
det, ok15 = [], True
for i in range(p.Nrx):
    s_i = range_compress(get_raw_data_3d(p, p.points, ptx_, prx_[i], vtx_, vrx_[i], p.ta, p.prf),
                         p.cd, p.rbw, p.rsf, axis=1)
    C0_, C1_, C2_, Dt_ = get_coeff_nu_3d(p.points[0], ptx_, prx_[i], vtx_, vrx_[i], p.prf, p.wl, p.ta,
                                         p.sq_tx, p.sq_rx[i], p.theta_tx, p.theta_rx[i])
    M = np.exp(-2j * np.pi * (C0_ / wl_arr[None, :] + (Dt_ - C1_) * fa_[:, None] + C2_ * fa_[:, None] ** 2 * wl_arr[None, :]))
    model = S_ref * M
    e15 = 20 * np.log10(np.sqrt(np.mean(np.abs(np.fft.fft2(s_i) - model)[inb] ** 2)) / np.sqrt(np.mean(np.abs(model)[inb] ** 2)))
    ok15 &= e15 < -25.0
    det.append(f"ch{i}: {e15:.1f} dB")
check("V15 channel = monostatic x model, error-to-signal in band", ok15, "; ".join(det))

n_ok = sum(ok for _, ok in RESULTS)
print(f"\n{n_ok}/{len(RESULTS)} checks passed")
