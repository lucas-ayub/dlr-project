"""
Q2, part 1: one target at x = 0, height h from 0 to 6 km, no SATA.
  measured: highest ambiguity in the windows m = +-1, +-2, in dB re. the reference peak
            (the metric of the sweeps), and the same re. the target's OWN reconstructed peak
  predicted: |c(d)| beta_|d| from the channel phases, c(d) = (1/N) sum_i e^{j phi_i} e^{-j 2 pi d i / N}
  main lobe |c(0)| and the energy check sum_d |c(d)|^2 = 1
"""
import csv
import numpy as np
import sata1d as S
import run_geometry_sweep as G

WL = 0.25
B1, B2 = 0.595, 0.190


def leak(g):
    N = len(g)
    return np.array([np.mean(g * np.exp(-2j * np.pi * d * np.arange(N) / N)) for d in range(N)])


def run(heights):
    cfg0 = S.make_config([(0.0, 0.0)]); tr0 = S.build_platform_tracks(cfg0)
    sref = S.generate_reference(cfg0, tr0, cfg0.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    foc = lambda cfg, x: np.roll(np.fft.ifft(np.fft.fft(x) * Hf), cfg.Na // 2)
    ref = foc(cfg0, S.generate_reference(cfg0, tr0)); pk = np.abs(ref).max()
    t = (np.arange(cfg0.Na) - cfg0.Na // 2) / cfg0.prf
    dt = G.ambiguity_times(cfg0)
    rows = []
    for h in heights:
        cfg = S.make_config([(0.0, h)], bxt_max=G.BXT_MAX, seed=G.BXT_SEED); tr = S.build_platform_tracks(cfg)
        p = cfg.scene.points[1]
        phi = np.array([2 * np.pi * S.residual_C0(cfg, tr, p, i) / WL for i in range(cfg.Nrx)])
        c = leak(np.exp(-1j * phi))
        e = np.abs(foc(cfg, S.reconstruct(cfg, tr, S.generate_channels(cfg, tr))))
        main = e[np.abs(t) <= 0.02].max()
        amb = {}
        for m in (-2, -1, 1, 2):
            sel = np.abs(t - m * dt) <= 0.02
            amb[m] = e[sel].max()
        worst1 = max(amb[-1], amb[1]); worst2 = max(amb[-2], amb[2])
        rows.append(dict(h=h, phi0=np.degrees(phi[0]), phi1=np.degrees(phi[1]), phi2=np.degrees(phi[2]), phi3=np.degrees(phi[3]),
                         c0=abs(c[0]), c1=abs(c[1]), c2=abs(c[2]), c3=abs(c[3]), energy=float(np.sum(np.abs(c) ** 2)),
                         main_db=20 * np.log10(main / pk),
                         amb1_ref_db=20 * np.log10(worst1 / pk), amb2_ref_db=20 * np.log10(worst2 / pk),
                         amb1_own_db=20 * np.log10(worst1 / main), amb2_own_db=20 * np.log10(worst2 / main),
                         pred1_db=20 * np.log10(max(abs(c[1]), abs(c[3])) * B1 + 1e-12),
                         pred2_db=20 * np.log10(abs(c[2]) * B2 + 1e-12),
                         pred_main_db=20 * np.log10(abs(c[0]) + 1e-12)))
        print(f"h={h:6.0f} main {rows[-1]['main_db']:6.1f} amb1 {rows[-1]['amb1_ref_db']:6.1f} (pred {rows[-1]['pred1_db']:6.1f}) "
              f"own {rows[-1]['amb1_own_db']:6.1f} energy {rows[-1]['energy']:.4f}", flush=True)
    return rows


if __name__ == "__main__":
    hs = np.concatenate((np.arange(0, 600, 10.0), np.arange(600, 6001, 25.0)))
    rows = run(hs)
    with open("probes/deep_q2_single.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
