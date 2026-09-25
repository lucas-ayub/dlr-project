"""
Verification of the SATA 1D implementation. Prints a report and writes figures to
./verification/. Each test states what is checked and the pass criterion.

 T1  kernel identity (zero map) ............................ rel. error < 1e-12
 T2  inject / remove round trip, pipeline maps ............ rel. error < 2 %
 T3  constant map = one global phase rotation .............. rel. error < 1e-12
 T4  residual_C0 vs closed form -bxt dh / (r0 tan(theta)) .. ratio within 2 %
 T5  channel pixel (bat/2 phase centre) vs range history ... |error| <= 1 cell
 T6  ruler, whole band: STFT peak -> target cell / alias ... |error| <= 1 bin
 T7  ruler, per sub-band: own-band energy -> target cell ... |error| <= 1 bin
 T8  footprint coverage of the target's STFT energy ........ >= 99 % of energy
 T9  channel correction vs ideal rotation (1 target) ....... error < 5 %
 T10 reconstruction, 1 target, heights 0..400 m ............ ambiguity within 1 dB of reference
 T11 multi-target scenes vs the oracle correction .......... reported (gap to oracle)
 T12 sign of the correction (remove / inject / remove twice)  only "remove" reaches the reference
 T13 cross-talk: 2-target map applied to each target alone .. each reads its own dC0, error < 5 %
 T14 channel-level error vs oracle channels (T11 scenes) .... reported
 T15 footprint half-width: 1st / 2nd / 3rd null ............. reported (2nd null adopted)
 T16 kernel oversampling osf = 1, 2, 4 ....................... ambiguity within 1 dB of osf = 4
 T17a random sparse scenes (no alias collision) vs oracle .. within 1 dB of the oracle
 T17b two targets vs separation: alias-collision bands ... outside |d - mX| < hw within 1 dB of oracle
 T18 bxt = 0: map is zero, SATA returns the input ........... identical to no SATA
 T19 Doppler axes: kernel axis = fftfreq (except Nyquist); sub-band axis wraps into the band
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sata1d as S
from sata1d.sata import subaperture_sizing, subaperture_axis, triangular_window
from sata1d.subband import subband_axis

OUT = "verification"
os.makedirs(OUT, exist_ok=True)
REPORT = []


def log(msg=""):
    print(msg)
    REPORT.append(msg)


def verdict(name, ok, detail):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def rel(a, b, m=None):
    m = np.ones(len(a), bool) if m is None else m
    return np.linalg.norm((a - b)[m]) / np.linalg.norm(b[m])


cfg = S.make_config([(0.0, 240.0)])
tr = S.build_platform_tracks(cfg)
wl, v, r, P = cfg.system.wl, cfg.system.vs, cfg.scene.r0, cfg.PRF_op
ch = S.generate_channels(cfg, tr)
tgt = cfg.scene.points[1]
kw = dict(rref=r, prf=P, v=v, wl=wl, r=r, sata_osf=4)
f_bands = [S.subband_frequency_beam(cfg, k)[0] for k in range(cfg.Nrx)]
results = {}

# ---------------------------------------------------------------- T1-T3 kernel algebra
log("== Kernel algebra ==")
s = ch[0]
rng = np.random.default_rng(1)
rand_map = rng.uniform(-0.3, 0.3, cfg.Na_ch)
e1 = max(rel(S.sata_1d(s, np.zeros(cfg.Na_ch), **kw), s),
         *[rel(S.sata_1d_subband(s, np.zeros(cfg.Na_ch), f_k=f, **kw), s) for f in f_bands])
results["T1"] = verdict("T1 identity (zero map, whole + 4 sub-bands)", e1 < 1e-12, f"max rel. error {e1:.1e}")
maps = [S.build_delta_C0_array(cfg, tr, 0)] + [S.build_delta_C0_subband_array(cfg, tr, 0, k) for k in range(cfg.Nrx)]
e2 = [rel(S.sata_1d(S.sata_1d(s, maps[0], inverse=False, **kw), maps[0], inverse=True, **kw), s, np.abs(s) > 0)]
e2 += [rel(S.sata_1d_subband(S.sata_1d_subband(s, maps[k + 1], f_k=f, inverse=False, **kw),
                             maps[k + 1], f_k=f, inverse=True, **kw), s, np.abs(s) > 0) for k, f in enumerate(f_bands)]
e2r = rel(S.sata_1d(S.sata_1d(s, rand_map, inverse=False, **kw), rand_map, inverse=True, **kw), s, np.abs(s) > 0)
results["T2"] = verdict("T2 inject then remove (pipeline maps, whole + 4 sub-bands)", max(e2) < 0.02,
                        f"max rel. error {max(e2) * 100:.2f} %")
log(f"    info: white-noise map (uniform +-30 cm per cell) does not round-trip: {e2r * 100:.0f} % "
    f"(the phase must vary slowly over one sub-aperture response, ~2 hw cells)")
c = -0.0547
out = S.sata_1d(s, np.full(cfg.Na_ch, c), inverse=True, **kw)
e3 = rel(out, s * np.exp(1j * 2 * np.pi / wl * c))
results["T3"] = verdict("T3 constant map = exp(+j 2pi dC0 / wl) on every sample", e3 < 1e-12, f"rel. error {e3:.1e}")

# ---------------------------------------------------------------- T4 residual vs closed form
log("\n== Residual C0 ==")
ratios, rows = [], []
for h in (40.0, 120.0, 240.0, 400.0):
    c_h = S.make_config([(0.0, h)])
    t_h = S.build_platform_tracks(c_h)
    p = c_h.scene.points[1]
    for i in range(c_h.Nrx):
        num = S.residual_C0(c_h, t_h, p, i)
        tan_th = c_h.scene.y0 / c_h.scene.H          # look angle at the reference point
        closed = -c_h.array.bxt[i] * h / (c_h.scene.r0 * tan_th)
        rows.append((h, c_h.array.bxt[i], num, closed))
        if abs(closed) > 1e-4:
            ratios.append(num / closed)
ratios = np.array(ratios)
results["T4"] = verdict("T4 residual_C0 / closed form", np.all(np.abs(ratios - 1) < 0.02),
                        f"ratio {ratios.min():.4f} .. {ratios.max():.4f}")
fig, ax = plt.subplots(figsize=(6.5, 4))
rows = np.array(rows)
ax.plot(rows[:, 3] * 100, rows[:, 2] * 100, "o", color="tab:blue")
lim = [rows[:, 2:].min() * 110, 1]
ax.plot(lim, lim, "k--", lw=.8)
ax.set_xlabel(r"closed form $-b_{xt}\,\Delta h/(r_0\tan\theta)$ [cm]")
ax.set_ylabel(r"numerical $\Delta C_0$ (residual_C0) [cm]"); ax.grid(alpha=.3)
ax.set_title("T4: numerical residual vs closed form (4 heights x 4 channels)")
plt.tight_layout(); plt.savefig(f"{OUT}/t4_residual.png", dpi=130); plt.close()

# ---------------------------------------------------------------- T5 channel pixel
log("\n== Target position per channel ==")
errs = []
for i in range(cfg.Nrx):
    rh = (np.linalg.norm(tr.ptx - tgt, axis=1) + np.linalg.norm(tr.prx[i] - tgt, axis=1))[::cfg.Nrx]
    measured = int(np.argmin(rh))
    predicted = S.az_pixel_of_scatterer(cfg, 0.0, i)
    errs.append(measured - predicted)
    log(f"    channel {i}: bat/2 = {cfg.array.bat[i] / 2:5.2f} m, closest approach at cell {measured}, "
        f"predicted {predicted}")
results["T5"] = verdict("T5 az_pixel_of_scatterer vs closest approach", max(map(abs, errs)) <= 1,
                        f"errors {errs} cells")

# ---------------------------------------------------------------- T6/T7 ruler
log("\n== Frequency-to-position ruler ==")
Tsubeff, hop, Nzp = subaperture_sizing(r, P, v, wl, 4)
win = triangular_window(Tsubeff)
xt = S.az_pixel_of_scatterer(cfg, 0.0, 0)
X = r * np.tan(np.arcsin(wl * P / (2 * v))) / v * P
rh0 = (np.linalg.norm(tr.ptx - tgt, axis=1) + np.linalg.norm(tr.prx[0] - tgt, axis=1))
f_true_full = -np.gradient(rh0, 1 / cfg.prf) / wl                    # true Doppler, full PRF
bin_cells = np.mean(np.diff(np.sort(subaperture_axis(P, Nzp, v, wl, r)[1])))


def peaks(axis_fn):
    rows = []
    for start in range(0, cfg.Na_ch - Tsubeff, hop):
        seg = s[start:start + Tsubeff]
        if np.abs(seg).min() == 0:
            continue
        spec = np.abs(np.fft.fft(np.r_[seg * win, np.zeros(Nzp - Tsubeff)]))
        fsub, azpos = axis_fn()
        k = int(np.argmax(spec))
        c_w = start + 0.5 * Tsubeff
        rows.append((c_w, f_true_full[int(c_w) * cfg.Nrx], fsub[k], azpos[k] + c_w))
    return np.array(rows)


pk = peaks(lambda: subaperture_axis(P, Nzp, v, wl, r))
m = np.round((pk[:, 2] - pk[:, 1]) / P)          # label = true Doppler + m PRF_op -> cell xt + m X
err6 = pk[:, 3] - (xt + m * X)
results["T6"] = verdict("T6 whole band: peak cell = target cell + m X", np.max(np.abs(err6)) <= bin_cells,
                        f"max |error| {np.max(np.abs(err6)):.2f} cells (1 bin = {bin_cells:.2f} cells), "
                        f"windows direct {np.sum(m == 0)}, folded {np.sum(m != 0)}")
own_err, old_own_err = [], []
for k, f_k in enumerate(f_bands):
    pk_k = peaks(lambda: subband_axis(P, Nzp, v, wl, r, f_k))
    own = np.abs(pk_k[:, 1] - f_k) < P / 2
    own_err += list(pk_k[own, 3] - xt)
    phi = np.fft.fftfreq(Nzp, 1 / P)                                   # previous 1D sub-band axis
    old = lambda: (f_k + phi, r * (np.tan(np.arcsin(wl * (f_k + phi) / (2 * v)))
                                   - np.tan(np.arcsin(wl * f_k / (2 * v)))) / v * P)
    pk_o = peaks(old)
    old_own_err += list(pk_o[own, 3] - xt)
own_err, old_own_err = np.abs(own_err), np.abs(old_own_err)
results["T7"] = verdict("T7 per sub-band: energy of band k -> target cell", own_err.max() <= bin_cells,
                        f"max |error| {own_err.max():.2f} cells over {len(own_err)} windows; "
                        f"previous axis (f_k + fftfreq, measured from beta_k): max {old_own_err.max():.0f} cells, "
                        f"{np.mean(old_own_err > bin_cells) * 100:.0f} % of windows off by >= 1 fold")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(pk[:, 0], pk[:, 3], ".", color="tab:blue", label="STFT peak -> cell (whole band)")
for mm in (-1, 0, 1):
    ax.axhline(xt + mm * X, color="grey", ls=":", lw=1)
ax.set_xlabel("window centre [cell]"); ax.set_ylabel("cell of the peak (posaux)")
ax.set_title("T6: the peak lands on the target cell or on target +- X (folded Doppler)")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/t6_ruler.png", dpi=130); plt.close()

# ---------------------------------------------------------------- T8 footprint coverage
log("\n== Footprint coverage ==")


def coverage(axis_fn, dmap):
    e_in = e_tot = 0.0
    for start in range(0, cfg.Na_ch - Tsubeff, hop):
        seg = s[start:start + Tsubeff]
        spec = np.abs(np.fft.fft(np.r_[seg * win, np.zeros(Nzp - Tsubeff)])) ** 2
        fsub, azpos = axis_fn()
        pos = np.clip(np.round(azpos + start + 0.5 * Tsubeff).astype(int), 0, cfg.Na_ch - 1)
        e_tot += spec.sum()
        e_in += spec[dmap[pos] != 0].sum()
    return e_in / e_tot


cov = [coverage(lambda: subaperture_axis(P, Nzp, v, wl, r), S.build_delta_C0_array(cfg, tr, 0))]
for k, f_k in enumerate(f_bands):
    cov.append(coverage(lambda: subband_axis(P, Nzp, v, wl, r, f_k), S.build_delta_C0_subband_array(cfg, tr, 0, k)))
results["T8"] = verdict("T8 energy of the target read inside the map", min(cov) >= 0.99,
                        "whole " + f"{cov[0] * 100:.2f} %, sub-bands " + ", ".join(f"{c * 100:.2f} %" for c in cov[1:]))

# ---------------------------------------------------------------- T9 channel correction
log("\n== Channel correction vs ideal ==")
ill = np.abs(ch[0]) > 0
errs9 = []
for i in range(cfg.Nrx):
    d = S.residual_C0(cfg, tr, tgt, i)
    ideal = ch[i] * np.exp(1j * 2 * np.pi / wl * d)
    errs9.append(rel(S.sata_1d(ch[i], S.build_delta_C0_array(cfg, tr, i), inverse=True, **kw), ideal, ill))
    for k, f_k in enumerate(f_bands):
        dk = S.residual_C0_subband(cfg, tr, tgt, i, k)
        idk = ch[i] * np.exp(1j * 2 * np.pi / wl * dk)
        errs9.append(rel(S.sata_1d_subband(ch[i], S.build_delta_C0_subband_array(cfg, tr, i, k), f_k=f_k,
                                           inverse=True, **kw), idk, ill))
results["T9"] = verdict("T9 corrected channel vs exp(+j 2pi dC0/wl) * channel", max(errs9) < 0.05,
                        f"max rel. error {max(errs9) * 100:.2f} % (4 channels x whole + 4 sub-bands)")


# ---------------------------------------------------------------- T10 reconstruction
def focus(cfg_, tr_, sig_ref, recs, xs):
    sref = S.generate_reference(cfg_, tr_, cfg_.scene.ptg[None, :])
    F = lambda x: np.roll(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(sref))), cfg_.Na // 2)
    foc = {"ref": F(sig_ref), **{k: F(v_) for k, v_ in recs.items()}}
    ds = cfg_.system.vs / cfg_.prf
    idx = cfg_.Na // 2 + np.round(np.asarray(xs) / ds).astype(int)
    mask = np.ones(cfg_.Na, bool)
    for i in idx:
        mask[max(0, i - 40):i + 41] = False
    pk_ = np.abs(foc["ref"]).max()
    return {k: 20 * np.log10(np.abs(v_)[mask].max() / pk_) for k, v_ in foc.items()}


log("\n== Reconstruction, one target ==")
rows10 = []
for h in (0.0, 80.0, 160.0, 240.0, 320.0, 400.0):
    c_h = S.make_config([(0.0, h)])
    t_h = S.build_platform_tracks(c_h)
    ch_h = S.generate_channels(c_h, t_h)
    a = focus(c_h, t_h, S.generate_reference(c_h, t_h),
              {"none": S.reconstruct(c_h, t_h, ch_h),
               "whole": S.reconstruct(c_h, t_h, S.sata_channels(c_h, t_h, ch_h)),
               "sub": S.reconstruct_subband(c_h, t_h, ch_h)}, [0.0])
    rows10.append((h, a["ref"], a["none"], a["whole"], a["sub"]))
    log(f"    h = {h:5.0f} m: ref {a['ref']:6.1f} dB, no SATA {a['none']:6.1f}, whole {a['whole']:6.1f}, sub {a['sub']:6.1f}")
rows10 = np.array(rows10)
dev = np.max(np.abs(rows10[:, 3:] - rows10[:, [1]]))
results["T10"] = verdict("T10 ambiguity with SATA vs reference", dev <= 1.0, f"max deviation {dev:.2f} dB")
fig, ax = plt.subplots(figsize=(6.5, 4))
for j, (lab, c_) in enumerate((("reference", "0.3"), ("no SATA", "tab:red"), ("SATA whole band", "tab:blue"),
                               ("SATA per sub-band", "tab:green"))):
    ax.plot(rows10[:, 0], rows10[:, 1 + j], "o-" if j != 3 else "s--", color=c_, label=lab)
ax.set_xlabel("target height [m]"); ax.set_ylabel("worst ambiguity [dB]"); ax.grid(alpha=.3); ax.legend()
ax.set_title("T10: one target, reconstruction")
plt.tight_layout(); plt.savefig(f"{OUT}/t10_height.png", dpi=130); plt.close()

# ---------------------------------------------------------------- T11 oracle
log("\n== Multi-target scenes vs oracle ==")
scenes = {
    "2 targets 3 km apart (240 / -120 m)": [(-1500.0, 240.0), (1500.0, -120.0)],
    "3 targets 3 km apart (0 / 300 / 0 m)": [(-3000.0, 0.0), (0.0, 300.0), (3000.0, 0.0)],
    "ramp 5 x 200 m, 3 deg": [(x, (x + 400) * np.tan(np.radians(3))) for x in (-400, -200, 0, 200, 400)],
    "ramp 9 x 200 m, 15 deg": [(x, (x + 800) * np.tan(np.radians(15))) for x in np.arange(-800, 801, 200.0)],
    "random heights 5 x 200 m (0-100 m)": list(zip((-400.0, -200.0, 0.0, 200.0, 400.0), (0, 20, 48, 96, 43))),
    "cliff 5 x 200 m, 100 m jump": list(zip((-400.0, -200.0, 0.0, 200.0, 400.0), (0, 0, 100, 100, 100))),
    "spike 5 x 200 m, 500 m": list(zip((-400.0, -200.0, 0.0, 200.0, 400.0), (0, 0, 500, 0, 0))),
}
rows11 = []
for name, tg in scenes.items():
    c_s = S.make_config(tg)
    t_s = S.build_platform_tracks(c_s)
    pts = c_s.scene.points[1:]
    per = [S.generate_channels(c_s, t_s, p[None, :]) for p in pts]
    ch_s = np.sum(per, axis=0)
    oracle = np.sum([np.array([per_p[i] * np.exp(1j * 2 * np.pi / c_s.system.wl * S.residual_C0(c_s, t_s, p, i))
                               for i in range(c_s.Nrx)]) for per_p, p in zip(per, pts)], axis=0)
    a = focus(c_s, t_s, S.generate_reference(c_s, t_s),
              {"none": S.reconstruct(c_s, t_s, ch_s), "oracle": S.reconstruct(c_s, t_s, oracle),
               "whole": S.reconstruct(c_s, t_s, S.sata_channels(c_s, t_s, ch_s)),
               "sub": S.reconstruct_subband(c_s, t_s, ch_s),
               "hold": S.reconstruct(c_s, t_s, S.sata_channels(c_s, t_s, ch_s, mode="hold"))},
              [t[0] for t in tg])
    rows11.append((name, a))
    log(f"    {name:38s}: ref {a['ref']:6.1f} | oracle {a['oracle']:6.1f} | whole {a['whole']:6.1f} | "
        f"sub {a['sub']:6.1f} | previous map {a['hold']:6.1f} | no SATA {a['none']:6.1f}")
fig, ax = plt.subplots(figsize=(10, 4.5))
keys = (("ref", "reference", "0.3"), ("oracle", "oracle (each target, own dC0)", "tab:purple"),
        ("whole", "SATA whole band", "tab:blue"), ("sub", "SATA per sub-band", "tab:green"),
        ("hold", "previous map", "tab:orange"), ("none", "no SATA", "tab:red"))
for j, (k, lab, c_) in enumerate(keys):
    ax.bar(np.arange(len(rows11)) + (j - 2.5) * 0.13, [a[k] for _, a in rows11], 0.13, color=c_, label=lab)
ax.set_xticks(range(len(rows11)), [n for n, _ in rows11], rotation=20, ha="right", fontsize=8)
ax.invert_yaxis(); ax.set_ylabel("worst ambiguity [dB]"); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8, ncol=3)
ax.set_title("T11: multi-target scenes (lower = better)")
plt.tight_layout(); plt.savefig(f"{OUT}/t11_oracle.png", dpi=130); plt.close()


# ---------------------------------------------------------------- T12 sign of the correction
log("\n== Sign of the correction ==")
c12 = S.make_config([(0.0, 240.0)]); t12 = S.build_platform_tracks(c12); ch12 = S.generate_channels(c12, t12)
maps12 = [S.build_delta_C0_array(c12, t12, i) for i in range(c12.Nrx)]
apply = lambda sig, inv: np.array([S.sata_1d(sig[i], maps12[i], inverse=inv, **kw) for i in range(c12.Nrx)])
a12 = focus(c12, t12, S.generate_reference(c12, t12),
            {"none": S.reconstruct(c12, t12, ch12),
             "remove": S.reconstruct(c12, t12, apply(ch12, True)),
             "inject (wrong sign)": S.reconstruct(c12, t12, apply(ch12, False)),
             "remove twice": S.reconstruct(c12, t12, apply(apply(ch12, True), True))}, [0.0])
for k_, v_ in a12.items():
    log(f"    {k_:22s}: {v_:6.1f} dB")
ok12 = (abs(a12["remove"] - a12["ref"]) < 1.0) and (a12["inject (wrong sign)"] > a12["none"] - 3.0) \
    and (a12["remove twice"] > a12["remove"] + 10.0)
results["T12"] = verdict("T12 only the removal (exp(+j 2pi dC0/wl)) reaches the reference", ok12,
                         f"remove {a12['remove']:.1f} dB, inject {a12['inject (wrong sign)']:.1f} dB, "
                         f"twice {a12['remove twice']:.1f} dB, none {a12['none']:.1f} dB, ref {a12['ref']:.1f} dB")

# ---------------------------------------------------------------- T13 cross-talk
log("\n== Cross-talk between two targets ==")
c13 = S.make_config([(-1500.0, 240.0), (1500.0, -120.0)]); t13 = S.build_platform_tracks(c13)
p1, p2 = c13.scene.points[1], c13.scene.points[2]
e13, rows13 = [], []
for i in range(c13.Nrx):
    m2 = S.build_delta_C0_array(c13, t13, i)                       # map built from BOTH targets
    for name, p in (("P1", p1), ("P2", p2)):
        s_p = S.generate_channels(c13, t13, p[None, :])[i]           # signal of ONE target
        d_p = S.residual_C0(c13, t13, p, i)
        e = rel(S.sata_1d(s_p, m2, inverse=True, **kw), s_p * np.exp(1j * 2 * np.pi / wl * d_p), np.abs(s_p) > 0)
        e13.append(e); rows13.append((i, name, d_p, e))
for i, name, d_p, e in rows13:
    log(f"    channel {i} {name}: own dC0 = {d_p * 100:+.2f} cm, error vs own rotation {e * 100:.2f} %")
results["T13"] = verdict("T13 each target reads its own dC0 from the two-target map", max(e13) < 0.05,
                         f"max error {max(e13) * 100:.2f} % (4 channels x 2 targets)")

# ---------------------------------------------------------------- T14 channel-level oracle
log("\n== Channel-level error vs oracle (T11 scenes) ==")
rows14 = []
for name, tg in scenes.items():
    c_s = S.make_config(tg); t_s = S.build_platform_tracks(c_s); pts = c_s.scene.points[1:]
    per = [S.generate_channels(c_s, t_s, p[None, :]) for p in pts]
    ch_s = np.sum(per, axis=0)
    orc = np.sum([np.array([pp[i] * np.exp(1j * 2 * np.pi / wl * S.residual_C0(c_s, t_s, p, i))
                            for i in range(c_s.Nrx)]) for pp, p in zip(per, pts)], axis=0)
    sat = S.sata_channels(c_s, t_s, ch_s)
    hol = S.sata_channels(c_s, t_s, ch_s, mode="hold")
    m_ = np.abs(ch_s) > 0
    e_s = np.linalg.norm((sat - orc)[m_]) / np.linalg.norm(orc[m_])
    e_h = np.linalg.norm((hol - orc)[m_]) / np.linalg.norm(orc[m_])
    e_n = np.linalg.norm((ch_s - orc)[m_]) / np.linalg.norm(orc[m_])
    rows14.append((name, e_s, e_h, e_n))
    log(f"    {name:38s}: SATA {e_s * 100:5.1f} %  | previous map {e_h * 100:5.1f} %  | no SATA {e_n * 100:5.1f} %")
results["T14"] = verdict("T14 channel-level error vs oracle", True,
                         "reported above (SATA < previous map in every scene: "
                         f"{all(r[1] <= r[2] + 1e-9 for r in rows14)})")

# ---------------------------------------------------------------- T15 footprint half-width
log("\n== Footprint half-width (which null) ==")
from sata1d.sata import footprint_map, sata_image_offsets
c15 = S.make_config([(0.0, 240.0)]); t15 = S.build_platform_tracks(c15); ch15 = S.generate_channels(c15, t15)
hw2 = S.sata_footprint_halfwidth(c15)
rows15 = []
for nn, hw_ in ((1, hw2 // 2), (2, hw2), (3, (3 * hw2) // 2), (4, 2 * hw2)):
    maps = []
    for i in range(c15.Nrx):
        pix = S.az_pixel_of_scatterer(c15, 0.0, i)
        maps.append(footprint_map({pix: [S.residual_C0(c15, t15, c15.scene.points[1], i)]}, c15.Na_ch, hw_,
                                  sata_image_offsets(c15)))
    corr = np.array([S.sata_1d(ch15[i], maps[i], inverse=True, **kw) for i in range(c15.Nrx)])
    e_ch = max(rel(corr[i], ch15[i] * np.exp(1j * 2 * np.pi / wl * S.residual_C0(c15, t15, c15.scene.points[1], i)),
                   np.abs(ch15[i]) > 0) for i in range(c15.Nrx))
    a15 = focus(c15, t15, S.generate_reference(c15, t15), {"sata": S.reconstruct(c15, t15, corr)}, [0.0])
    rows15.append((nn, hw_, e_ch, a15["sata"]))
    log(f"    null {nn}: hw = {hw_:3d} cells ({hw_ * c15.system.vs / c15.PRF_op / 1e3:.2f} km): "
        f"channel error {e_ch * 100:.2f} %, ambiguity {a15['sata']:.1f} dB (ref {a15['ref']:.1f})")
results["T15"] = verdict("T15 footprint half-width", True, f"2nd null adopted (hw = {hw2} cells)")

# ---------------------------------------------------------------- T16 oversampling
log("\n== Kernel oversampling ==")
rows16 = []
for osf in (1, 2, 4, 8):
    a16 = focus(c15, t15, S.generate_reference(c15, t15),
                {"sata": S.reconstruct(c15, t15, S.sata_channels(c15, t15, ch15, sata_osf=osf))}, [0.0])
    rows16.append((osf, a16["sata"]))
    log(f"    osf = {osf}: Nzp = {32 * osf:4d} bins ({500 / (32 * osf):.2f} Hz/bin): ambiguity {a16['sata']:.1f} dB")
dev16 = max(abs(r[1] - rows16[2][1]) for r in rows16)
results["T16"] = verdict("T16 osf = 1..8 within 1 dB of osf = 4", dev16 <= 1.0, f"max deviation {dev16:.2f} dB")

# ---------------------------------------------------------------- T17a random sparse scenes (no collision)
log("\n== Random sparse scenes vs oracle (separations outside the alias-collision bands) ==")
X_km = X * v / P / 1e3                                   # one PRF_ch fold in km (6.23 km)
rows17 = []
for seed in range(6):
    g = np.random.default_rng(100 + seed)
    gaps = g.uniform(3800.0, 4900.0, 2)                  # 3 targets: pairs 3.8-4.9 km and 7.6-9.8 km, both outside X +- 1.3 km
    xs = np.concatenate(([0.0], np.cumsum(gaps))); xs -= xs.mean() + g.uniform(-500, 500)
    hs = g.uniform(-300, 400, 3)
    tg = list(zip(xs, hs))
    c_s = S.make_config(tg, seed=seed); t_s = S.build_platform_tracks(c_s); pts = c_s.scene.points[1:]
    per = [S.generate_channels(c_s, t_s, p[None, :]) for p in pts]
    ch_s = np.sum(per, axis=0)
    orc = np.sum([np.array([pp[i] * np.exp(1j * 2 * np.pi / wl * S.residual_C0(c_s, t_s, p, i))
                            for i in range(c_s.Nrx)]) for pp, p in zip(per, pts)], axis=0)
    a17 = focus(c_s, t_s, S.generate_reference(c_s, t_s),
                {"none": S.reconstruct(c_s, t_s, ch_s), "oracle": S.reconstruct(c_s, t_s, orc),
                 "whole": S.reconstruct(c_s, t_s, S.sata_channels(c_s, t_s, ch_s)),
                 "sub": S.reconstruct_subband(c_s, t_s, ch_s)}, list(xs))
    rows17.append((seed, a17))
    log(f"    seed {seed}: x = [" + ", ".join(f"{x / 1e3:+.1f}" for x in xs) + "] km, h = ["
        + ", ".join(f"{h:+.0f}" for h in hs) + "] m, bxt = [" + ", ".join(f"{b:.0f}" for b in c_s.array.bxt) + "] m")
    log(f"        ref {a17['ref']:6.1f} | oracle {a17['oracle']:6.1f} | whole {a17['whole']:6.1f} | "
        f"sub {a17['sub']:6.1f} | no SATA {a17['none']:6.1f} dB")
dev17 = max(max(a["whole"], a["sub"]) - a["oracle"] for _, a in rows17)
results["T17a"] = verdict("T17a random sparse scenes (no collision) within 1 dB of the oracle", dev17 <= 1.0,
                          f"max (SATA - oracle) = {dev17:.2f} dB over 6 scenes")

# ---------------------------------------------------------------- T17b alias collision
log("\n== Two targets against their separation: the Doppler-alias collision ==")
rows17b = []
for d_km in (2, 3, 4, 4.5, 5, 5.5, 6, X_km, 6.5, 7, 7.5, 8, 9, 10, 11, 2 * X_km, 14):
    d = d_km * 1e3
    c_s = S.make_config([(-d / 2, 240.0), (d / 2, -120.0)]); t_s = S.build_platform_tracks(c_s)
    pts = c_s.scene.points[1:]
    per = [S.generate_channels(c_s, t_s, p[None, :]) for p in pts]
    ch_s = np.sum(per, axis=0)
    orc = np.sum([np.array([pp[i] * np.exp(1j * 2 * np.pi / wl * S.residual_C0(c_s, t_s, p, i))
                            for i in range(c_s.Nrx)]) for pp, p in zip(per, pts)], axis=0)
    a_ = focus(c_s, t_s, S.generate_reference(c_s, t_s),
               {"none": S.reconstruct(c_s, t_s, ch_s), "oracle": S.reconstruct(c_s, t_s, orc),
                "whole": S.reconstruct(c_s, t_s, S.sata_channels(c_s, t_s, ch_s)),
                "sub": S.reconstruct_subband(c_s, t_s, ch_s)}, [-d / 2, d / 2])
    rows17b.append((d_km, a_))
    log(f"    d = {d_km:5.2f} km: ref {a_['ref']:6.1f} | oracle {a_['oracle']:6.1f} | whole {a_['whole']:6.1f} | "
        f"sub {a_['sub']:6.1f} | no SATA {a_['none']:6.1f} dB")
hw_km = S.sata_footprint_halfwidth(cfg) * v / P / 1e3
core_km = hw_km / 2                                     # first null of the window response: main lobes share bins
outside = [a_ for d_km, a_ in rows17b if abs(d_km - X_km) > hw_km and abs(d_km - 2 * X_km) > hw_km]
inside = [a_ for d_km, a_ in rows17b if abs(d_km - X_km) <= core_km or abs(d_km - 2 * X_km) <= core_km]
dev_out = max(max(a_["whole"], a_["sub"]) - a_["oracle"] for a_ in outside)
results["T17b"] = verdict("T17b outside the collision bands |d - mX| > hw (SATA within 1 dB of the oracle)", dev_out <= 1.0,
                          f"max (SATA - oracle) = {dev_out:.2f} dB on {len(outside)} separations; in the core "
                          f"|d - mX| <= {core_km:.2f} km (X = {X_km:.2f} km) SATA is {np.mean([a_['whole'] - a_['oracle'] for a_ in inside]):.0f} dB "
                          f"(whole) / {np.mean([a_['sub'] - a_['oracle'] for a_ in inside]):.0f} dB (sub) above the oracle, "
                          f"{np.mean([a_['none'] - a_['whole'] for a_ in inside]):.0f} dB better than no SATA: inherent limit")
fig, ax = plt.subplots(figsize=(7.5, 3.8))
dd = np.array([d_km for d_km, _ in rows17b])
for k_, lab, c_, mk in (("ref", "reference", "0.3", "-"), ("oracle", "oracle", "tab:purple", "-"),
                        ("whole", "SATA whole band", "tab:blue", "o-"), ("sub", "SATA per sub-band", "tab:green", "s--"),
                        ("none", "no SATA", "tab:red", "-")):
    ax.plot(dd, [a_[k_] for _, a_ in rows17b], mk, color=c_, ms=4, lw=1.3, label=lab)
for m_ in (1, 2):
    ax.axvspan(m_ * X_km - hw_km, m_ * X_km + hw_km, color="0.92", zorder=0)
    ax.axvspan(m_ * X_km - core_km, m_ * X_km + core_km, color="0.80", zorder=0)
    ax.axvline(m_ * X_km, color="0.5", ls=":", lw=.8)
ax.set_xlabel("separation of the two targets [km]"); ax.set_ylabel("worst ambiguity [dB]"); ax.grid(alpha=.3)
ax.legend(fontsize=8, ncol=2); ax.set_title("T17b: two targets (240 / -120 m); shaded: |d - mX| < hw (light), < hw/2 (dark): alias collision")
plt.tight_layout(); plt.savefig(f"{OUT}/t17_collision.png", dpi=130); plt.close()

# ---------------------------------------------------------------- T18 bxt = 0
log("\n== bxt = 0 ==")
c18 = S.make_config([(-1500.0, 240.0), (1500.0, -120.0)], bxt_max=0.0); t18 = S.build_platform_tracks(c18)
ch18 = S.generate_channels(c18, t18)
m18 = max(np.abs(S.build_delta_C0_array(c18, t18, i)).max() for i in range(c18.Nrx))
e18 = rel(S.sata_channels(c18, t18, ch18).ravel(), ch18.ravel())
results["T18"] = verdict("T18 bxt = 0: zero map and identity", m18 < 1e-6 and e18 < 1e-8,
                         f"max |map| = {m18:.1e} m, SATA vs input rel. error {e18:.1e}")

# ---------------------------------------------------------------- T19 axes
log("\n== Doppler axes ==")
fk_ax, _ = subaperture_axis(P, Nzp, v, wl, r)
phi_ = np.fft.fftfreq(Nzp, 1 / P)
nyq = np.argmin(np.abs(np.abs(phi_) - P / 2))
same = np.allclose(np.delete(fk_ax, nyq), np.delete(phi_, nyq))
ok19 = same and abs(abs(fk_ax[nyq]) - P / 2) < 1e-9
for k, f_k in enumerate(f_bands):
    fs_, _ = subband_axis(P, Nzp, v, wl, r, f_k)
    in_band = np.all((fs_ >= f_k - P / 2 - 1e-9) & (fs_ < f_k + P / 2 + 1e-9))
    congruent = np.allclose(np.mod(fs_ - phi_ + P / 2, P) - P / 2, 0.0)
    ok19 &= in_band and congruent
    log(f"    band {k} (f_k = {f_k:+.0f} Hz): axis in [{fs_.min():+.1f}, {fs_.max():+.1f}] Hz, "
        f"inside band {in_band}, congruent to fftfreq mod PRF_ch {congruent}")
results["T19"] = verdict("T19 kernel axis = fftfreq except Nyquist (+PRF/2); sub-band axes wrap into their band",
                         ok19, f"kernel/fftfreq equal on {Nzp - 1} bins, Nyquist labelled {fk_ax[nyq]:+.0f} Hz")

log("\n== Summary ==")
log(", ".join(f"{k}: {'PASS' if v_ else 'FAIL'}" for k, v_ in results.items()))
with open(f"{OUT}/verification_report.txt", "w") as fh:
    fh.write("\n".join(REPORT) + "\n")
np.save(f"{OUT}/t11_rows.npy", np.array([(n, *[a[k] for k in ("ref", "oracle", "whole", "sub", "hold", "none")])
                                         for n, a in rows11], dtype=object), allow_pickle=True)
