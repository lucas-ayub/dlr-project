"""
Why the ambiguity saturates.
Part 1 (no SATA): one target at x = 0, height 0 ... 6 km. Measured ambiguity peak in the windows
m = -2, -1, +1, +2 against the prediction |c(d)| beta(d), c(d) = DFT over the channels of exp(j phi_i),
phi_i = 2 pi dC0_i / wl, beta(d) = (B_a - |d| PRF_op) / B_a.
"""
import numpy as np
import sata1d as S
import run_geometry_sweep as G

WL = 0.25


def leak(phases_or_gains):
    g = np.asarray(phases_or_gains)
    N = len(g)
    return np.array([np.mean(g * np.exp(-2j * np.pi * d * np.arange(N) / N)) for d in range(N)])


def windows(cfg, x, env, pk):
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt = G.ambiguity_times(cfg)
    out = {}
    for m in (-2, -1, 1, 2):
        sel = np.abs(t - (x / cfg.system.vs + m * dt)) <= 0.02
        out[m] = 20 * np.log10(env[sel].max() / pk)
    return out


def single_target(heights):
    rows = []
    c0 = S.make_config([(0.0, 0.0)], bxt_max=0.0); t0 = S.build_platform_tracks(c0)
    sref = S.generate_reference(c0, t0, c0.scene.ptg[None, :]); H = np.conj(np.fft.fft(sref))
    foc = lambda cfg, x: np.roll(np.fft.ifft(np.fft.fft(x) * H), cfg.Na // 2)
    flat = foc(c0, S.reconstruct(c0, t0, S.generate_channels(c0, t0)))
    pk = np.abs(flat).max()
    for h in heights:
        cfg = S.make_config([(0.0, h)], bxt_max=G.BXT_MAX, seed=G.BXT_SEED); tr = S.build_platform_tracks(cfg)
        p = cfg.scene.points[1]
        dC0 = np.array([S.residual_C0(cfg, tr, p, i) for i in range(cfg.Nrx)])
        c = leak(np.exp(-2j * np.pi * dC0 / WL))
        e = foc(cfg, S.reconstruct(cfg, tr, S.generate_channels(cfg, tr)))
        w = windows(cfg, 0.0, np.abs(e - flat), pk)
        main = 20 * np.log10(np.abs(e).max() / pk)
        rows.append(dict(h=h, **{f"meas_m{m}": v for m, v in w.items()}, main=main,
                         **{f"c{d}": float(np.abs(c[d])) for d in range(4)},
                         **{f"phi{i}": float(np.angle(np.exp(-2j * np.pi * dC0[i] / WL))) for i in range(4)},
                         **{f"dC0_{i}": float(dC0[i]) for i in range(4)}))
    return rows, cfg


if __name__ == "__main__":
    import csv
    hs = np.arange(0, 6001, 20.0)
    rows, cfg = single_target(hs)
    with open("probes/saturation_single.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    Ba, P = cfg.abw, cfg.PRF_op
    print("beta1", (Ba - P) / Ba, "beta2", (Ba - 2 * P) / Ba)
