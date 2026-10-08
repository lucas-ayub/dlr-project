"""
Why is the per-sub-band SATA worse than the whole-band one with the terrain look-up table?
For a few terrains: whole band, per sub-band, per sub-band with the table filled over the
whole line (x_max = 30 km instead of 15 km), a hybrid (sub-band kernels for the central bands
1-2, whole-band kernel for the outer bands 0 and 3), and the ambiguity carried by each output
sub-band of the whole-band and per-sub-band reconstructions.

    python probes/probe_subband_lut.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import sata1d as S
import sata1d.sata as SS
from sata1d.subband import sata_channels_subband, channel_spectrum
from sata1d.reconstruction import reconstruct_from_spectra
from run_one_case import ambiguity_level, oracle_channels
from run_geometry_sweep import SATA_OSF, BXT_MAX, BXT_SEED
from run_terrain_lut_cases import terrain_from, target_positions, ALL_PRESETS, X_FOLD

CASES = [("SIN-B600-A400-P25", None), ("COS-B800-A600-P60", None), ("RAMP-H1000-L12", None),
         ("SIN-A250-PX", None), ("FLAT-500", "500+0*x"), ("SIN-B600-A400-P25 (N=1)", None)]


def run(name, expr, n_targets):
    t0 = time.time()
    terrain = terrain_from(expr)
    xs = target_positions(n_targets); hs = terrain(xs)
    cfg = S.make_config(list(zip(xs, hs)), bxt_max=BXT_MAX, seed=BXT_SEED)
    tr = S.build_platform_tracks(cfg)
    dC0 = S.delta_C0_per_target(cfg, tr)
    tab15 = [S.terrain_lookup_table(cfg, tr, i, terrain, step=4, x_max=15000.0) for i in range(cfg.Nrx)]
    tab30 = [S.terrain_lookup_table(cfg, tr, i, terrain, step=4, x_max=30000.0) for i in range(cfg.Nrx)]
    ch = S.generate_channels(cfg, tr)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    Na = cfg.Na
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), Na // 2))
    t = (np.arange(Na) - Na // 2) / cfg.prf
    dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    pk = F(S.generate_reference(cfg, tr)).max()
    amb = lambda s: ambiguity_level(t, F(s), pk, xs, cfg.system.vs, dt)

    def whole_channels(M):
        out = ch.copy()
        for i in range(cfg.Nrx):
            out[i] = SS.sata_1d(out[i], M[i], rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=cfg.system.wl,
                                r=cfg.scene.r0, squint=cfg.sq_tx, inverse=True, sata_osf=SATA_OSF)
        return out

    def recon(spec_sets):                       # spec_sets[k] = channels used for output band k
        spec = [[channel_spectrum(cfg, sk[j]) for j in range(cfg.Nrx)] for sk in spec_sets]
        return reconstruct_from_spectra(cfg, tr, spec, True)

    w15 = whole_channels(tab15)
    sub15 = [sata_channels_subband(cfg, tr, ch, k, sata_osf=SATA_OSF, maps=tab15) for k in range(cfg.Nsb)]
    sub30 = [sata_channels_subband(cfg, tr, ch, k, sata_osf=SATA_OSF, maps=tab30) for k in range(cfg.Nsb)]
    rec = {"none": S.reconstruct(cfg, tr, ch),
           "whole (table 15 km)": S.reconstruct(cfg, tr, w15),
           "sub (table 15 km)": recon(sub15),
           "sub (table 30 km)": recon(sub30),
           "hybrid: bands 1,2 sub / 0,3 whole (15 km)": recon([w15, sub15[1], sub15[2], w15]),
           "hybrid: bands 1,2 sub / 0,3 whole (30 km)": recon([w15, sub30[1], sub30[2], w15]),
           "oracle": S.reconstruct(cfg, tr, oracle_channels(cfg, tr, dC0))}
    print(f"\n=== {name}: {expr}, {n_targets} targets at {np.round(xs[:3]).astype(int)}... ({time.time() - t0:.0f} s)")
    for k, v in rec.items():
        print(f"  {k:44s} {amb(v):7.1f} dB")

    # ambiguity carried by each output sub-band (band-limited reconstruction before the correlation)
    prfF = cfg.prf; fa = np.fft.fftfreq(Na, d=1.0 / prfF)
    print("  per output band k (f in [f_k - 250, f_k + 250)):   " + "   ".join(f"k={k} ({(-cfg.Nsb/2 + k + 0.5) * cfg.PRF_op:+.0f} Hz)" for k in range(cfg.Nsb)))
    for key in ("none", "whole (table 15 km)", "sub (table 15 km)", "sub (table 30 km)", "oracle"):
        U = np.fft.fft(rec[key]); vals = []
        for k in range(cfg.Nsb):
            f_k = (-cfg.Nsb / 2 + k + 0.5) * cfg.PRF_op
            m = (fa >= f_k - cfg.PRF_op / 2) & (fa < f_k + cfg.PRF_op / 2)
            vals.append(amb(np.fft.ifft(U * m)))
        print(f"  {key:44s} " + "   ".join(f"{v:12.1f}" for v in vals))

    # table mismatch at the fold distances the kernels read: |dC0(x_j + m X) - dC0(x_j)| (channel 0)
    ds = cfg.system.vs / cfg.PRF_op
    cell = lambda x: int(np.clip(round(cfg.Na_ch / 2 + (cfg.scene.x0 + x + 0.5 * cfg.array.bat[0]) / ds), 0, cfg.Na_ch - 1))
    own = np.array([tab15[0][cell(x)] for x in xs])
    print("  table error read m folds away (ch 0, mean over targets, mm):  " +
          "  ".join(f"m={m:+d}: {1e3 * np.mean([abs(tab15[0][cell(x + m * X_FOLD)] - o) for x, o in zip(xs, own)]):6.1f}"
                    for m in (-3, -2, -1, 1, 2, 3)) + f"   (|own| mean {1e3 * np.abs(own).mean():.1f} mm; targets' own dC0 vs table: {1e3 * np.abs(own - dC0[:, 0]).max():.3f} mm)")


if __name__ == "__main__":
    for name, expr in CASES:
        n = 1 if "(N=1)" in name else 7
        cid = name.split(" ")[0]
        run(name, expr or ALL_PRESETS[cid][0], n)
