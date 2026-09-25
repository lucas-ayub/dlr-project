"""
Whole band vs per sub-band SATA, band by band.
  1) residual coefficients: dC0 (and dC1, dC2) fitted over each band's look angles vs the whole aperture
  2) corrected channels: set k (per sub-band) vs the whole-band set, restricted to the Doppler band k
  3) reconstruction: error of each output sub-band against the reference, for no SATA / whole / sub / oracle
Output: subband_vs_wholeband.png and a printed table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sata1d as S
from sata1d.subband import getcoeff_beam, subband_frequency_beam, sata_channels_subband
from sata1d.reconstruction import GetCoeffNu

WL = 0.25
SCENES = {
    "1 target, h = 240 m": [(0.0, 240.0)],
    "2 targets 3 km (240 / -120 m)": [(-1500.0, 240.0), (1500.0, -120.0)],
    "along-track ramp 9 x 200 m, 15 deg": [(x, (x + 800) * np.tan(np.radians(15))) for x in np.arange(-800, 801, 200.0)],
    "random heights 5 x 200 m": list(zip((-400.0, -200.0, 0.0, 200.0, 400.0), (0, 20, 48, 96, 43))),
}


def band_masks(cfg):
    """Full-PRF spectrum bins of each output sub-band (same order as reconstruct_from_spectra)."""
    Na, prf, P = cfg.Na, cfg.prf, cfg.PRF_op
    fa = np.fft.fftfreq(Na, 1 / prf)
    return [(fa >= -prf / 2 + k * P) & (fa < -prf / 2 + (k + 1) * P) & (np.abs(fa) <= cfg.abw / 2)
            for k in range(cfg.Nrx)]


# ---------------------------------------------------------------- 1) coefficients per band
cfg = S.make_config([(0.0, 240.0)]); tr = S.build_platform_tracks(cfg); p = cfg.scene.points[1]
print("1) residual coefficients, target h = 240 m (whole aperture vs each band's look angles)")
coef = np.zeros((cfg.Nrx, 1 + cfg.Nrx))
for i in range(cfg.Nrx):
    com = (tr.ptx, tr.prx[i], tr.vtx, tr.vrx[i], tr.ptx, tr.vtx, cfg.prf, WL, cfg.ta,
           cfg.sq_tx, cfg.sq_rx[i], cfg.theta_tx, cfg.theta_rx[i])
    w = np.array(GetCoeffNu(p, *com)) - np.array(GetCoeffNu(cfg.scene.ptg, *com))
    coef[i, 0] = w[0]
    line = f"   ch {i}: whole dC0 = {w[0] * 100:+.5f} cm |"
    for k in range(cfg.Nrx):
        _, _, lo, hi = subband_frequency_beam(cfg, k)
        a = np.array(getcoeff_beam(p, tr.ptx, tr.prx[i], tr.vtx, tr.vrx[i], cfg.prf, WL, cfg.ta, lo, hi))
        b = np.array(getcoeff_beam(cfg.scene.ptg, tr.ptx, tr.prx[i], tr.vtx, tr.vrx[i], cfg.prf, WL, cfg.ta, lo, hi))
        coef[i, 1 + k] = (a - b)[0]
        line += f" band {k}: {(a - b)[0] * 100:+.5f}"
    print(line)
print(f"   max |dC0(band) - dC0(whole)| = {np.abs(coef[:, 1:] - coef[:, [0]]).max() * 1e3:.2e} mm")

# ---------------------------------------------------------------- 2) + 3) per scene
rows = []
for name, tg in SCENES.items():
    c = S.make_config(tg); t = S.build_platform_tracks(c); pts = c.scene.points[1:]
    per = [S.generate_channels(c, t, q[None, :]) for q in pts]
    ch = np.sum(per, axis=0)
    orc = np.sum([np.array([pp[i] * np.exp(2j * np.pi / WL * S.residual_C0(c, t, q, i)) for i in range(c.Nrx)])
                  for pp, q in zip(per, pts)], axis=0)
    whole = S.sata_channels(c, t, ch)
    # 2) channel sets: set k vs whole set, in the channel's baseband (all folds), energy-weighted
    d_set = []
    for k in range(c.Nrx):
        sk = sata_channels_subband(c, t, ch, k)
        d_set.append(np.linalg.norm(sk - whole) / np.linalg.norm(whole))
    # 3) reconstruction, error per output band against the reference spectrum
    ref = np.fft.fft(S.generate_reference(c, t))
    recs = {"no SATA": S.reconstruct(c, t, ch), "whole band": S.reconstruct(c, t, whole),
            "per sub-band": S.reconstruct_subband(c, t, ch), "oracle": S.reconstruct(c, t, orc)}
    masks = band_masks(c)
    esr = {}
    for k_, rec in recs.items():
        R = np.fft.fft(rec)
        esr[k_] = [20 * np.log10(np.linalg.norm((R - ref)[m]) / np.linalg.norm(ref[m])) for m in masks]
    rows.append((name, d_set, esr))

print("\n2) corrected channels: || set k - whole-band set || / || whole-band set ||")
for name, d_set, _ in rows:
    print(f"   {name:36s}: " + "  ".join(f"k={k}: {d * 100:5.2f} %" for k, d in enumerate(d_set)))
print("\n3) error of each output sub-band vs reference (ESR, dB)")
for name, _, esr in rows:
    print(f"   {name}")
    for k_, v in esr.items():
        print(f"      {k_:13s}: " + "  ".join(f"band {k}: {x:6.1f}" for k, x in enumerate(v)))

# ---------------------------------------------------------------- figure
fig, A = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.6), sharey=True)
cols = {"no SATA": "#B2182B", "whole band": "#185FA5", "per sub-band": "#1D9E75", "oracle": "#7F4FC9"}
for a, (name, _, esr) in zip(A, rows):
    for j, (k_, v) in enumerate(esr.items()):
        a.bar(np.arange(4) + (j - 1.5) * 0.2, v, 0.2, color=cols[k_], label=k_)
    a.set_xticks(range(4), [f"band {k}\n{(-1.5 + k) * 500:+.0f} Hz" for k in range(4)], fontsize=8)
    a.set_title(name, fontsize=9); a.grid(alpha=.3, axis="y"); a.invert_yaxis()
A[0].set_ylabel("error of the output sub-band vs reference [dB]"); A[0].legend(fontsize=7)
fig.suptitle("Whole band vs per sub-band SATA, band by band (lower = better)", fontsize=11)
plt.tight_layout(); plt.savefig("subband_vs_wholeband.png", dpi=130); plt.close()
print("\nsaved subband_vs_wholeband.png")
