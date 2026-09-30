"""Sub-aperture bookkeeping A/B: triangular analysis window + COLA (ours) vs the
reference cross-fade of corrected rectangular blocks, with zero padding or with the
IDL real-data margin. Worst ambiguity [dB] after reconstruction, whole band and per
sub-band, on a few scenes."""
import numpy as np
import sata1d as S

SCENES = {
    "1 target 240 m": [(0.0, 240.0)],
    "2 targets 3 km (240/-120)": [(-1500.0, 240.0), (1500.0, -120.0)],
    "along-track ramp N9 20 deg": [((i - 4) * 200.0, i * 200 * np.tan(np.radians(20))) for i in range(9)],
    "random heights N9 hmax 500": list(zip((np.arange(9) - 4) * 200.0,
                                           np.r_[0.0, np.random.default_rng(2000).uniform(0, 500, 9)[1:]])),
    "spike N9 500 m": [((i - 4) * 200.0, 500.0 if i == 4 else 0.0) for i in range(9)],
    "cliff N9 250 m": [((i - 4) * 200.0, 250.0 if i >= 4 else 0.0) for i in range(9)],
}
SCHEMES = ("cola", "crossfade", "margin")


def amb(cfg, foc, ref, xs, w=40):
    ds = cfg.system.vs / cfg.prf
    m = np.ones(cfg.Na, bool)
    for x in xs:
        i = cfg.Na // 2 + int(round(x / ds)); m[max(0, i - w):i + w + 1] = False
    return 20 * np.log10(np.abs(foc)[m].max() / np.abs(ref).max())


print(f"{'scene':28s} {'none':>6s} " + " ".join(f"{'W-' + s:>9s} {'S-' + s:>9s}" for s in SCHEMES) + f" {'oracle':>7s}")
for name, tg in SCENES.items():
    c = S.make_config(tg); t = S.build_platform_tracks(c)
    sref = S.generate_reference(c, t, c.scene.ptg[None, :])
    F = lambda x: np.roll(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(sref))), c.Na // 2)
    ref = F(S.generate_reference(c, t))
    ch = S.generate_channels(c, t)
    xs = [x for x, _ in tg]
    row = [amb(c, F(S.reconstruct(c, t, ch)), ref, xs)]
    for sc in SCHEMES:
        row.append(amb(c, F(S.reconstruct(c, t, S.sata_channels(c, t, ch, sata_osf=4, scheme=sc))), ref, xs))
        row.append(amb(c, F(S.reconstruct_subband(c, t, ch, sata_osf=4, scheme=sc)), ref, xs))
    orc = 0
    for p in c.scene.points[1:]:
        d = np.array([S.residual_C0(c, t, p, i) for i in range(c.Nrx)])
        orc = orc + S.generate_channels(c, t, ptgs=[p]) * np.exp(2j * np.pi * d / c.system.wl)[:, None]
    row.append(amb(c, F(S.reconstruct(c, t, orc)), ref, xs))
    print(f"{name:28s} " + " ".join(f"{v:9.2f}" for v in row), flush=True)
