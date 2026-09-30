"""
Why the ambiguity after SATA levels off for close targets with large height steps.
SATA is linear in the data for a fixed map, so the channels of each target can be corrected
separately. For target j and channel i the corrected signal is written as
    SATA(s_ij) = g_ij * s_ij^ideal + d_ij ,
s_ij^ideal = s_ij exp(+j 2 pi dC0_ij / wl) (the exactly corrected channel), g_ij = projection gain
(ideal: 1), d_ij = what is left (distortion). The leakage of target j into the Doppler band shifted
by d is then c_j(d) = (1/Nrx) sum_i g_ij exp(-j 2 pi d i / Nrx), exactly as without SATA with
exp(j phi_i) replaced by g_ij.  Prediction: max_j max_d |c_j(d)| beta(d); compare with the measured
topographic ambiguity of SATA whole band.
"""
import csv
import numpy as np
import sata1d as S
import run_geometry_sweep as G
import sweep_v3 as V

WL = 0.25
B1, B2 = 0.595, 0.190


def leak(g):
    N = len(g)
    return np.array([np.mean(g * np.exp(-2j * np.pi * d * np.arange(N) / N)) for d in range(N)])


def analyse(case):
    cfg = G.make_cfg(case, G.BXT_MAX); tr = S.build_platform_tracks(cfg)
    pts = cfg.scene.points[1:]
    g = np.zeros((len(pts), cfg.Nrx), complex); dist = np.zeros((len(pts), cfg.Nrx))
    for j, p in enumerate(pts):
        s = S.generate_channels(cfg, tr, ptgs=[p])
        d = np.array([S.residual_C0(cfg, tr, p, i) for i in range(cfg.Nrx)])
        ideal = s * np.exp(2j * np.pi * d / WL)[:, None]
        out = S.sata_channels(cfg, tr, s, sata_osf=G.SATA_OSF)
        for i in range(cfg.Nrx):
            g[j, i] = np.vdot(ideal[i], out[i]) / np.vdot(ideal[i], ideal[i])
            dist[j, i] = np.linalg.norm(out[i] - g[j, i] * ideal[i]) / np.linalg.norm(ideal[i])
    pred = max(max(abs(leak(g[j])[1]) * B1, abs(leak(g[j])[3]) * B1, abs(leak(g[j])[2]) * B2) for j in range(len(pts)))
    cfg2, foc = V.simulate(case)
    m, _, _ = V.metrics(cfg2, foc, case)
    return g, dist, 20 * np.log10(pred), m["ambt_db_whole"], m["ambt_db_none"]


if __name__ == "__main__":
    rows = []
    for step, N, a in [(2600, 5, 1), (2600, 5, 3), (2600, 5, 5), (2600, 5, 10), (2600, 5, 20), (2600, 5, 30), (2600, 5, 50),
                       (2600, 2, 20), (2600, 13, 20), (200, 5, 5), (200, 5, 20), (200, 5, 50), (200, 13, 50)]:
        c = G.Case("ramp", f"N{N}_a{a}", "", G.heights_from_slopes([a] * (N - 1), step), 200.0, {})
        g, dist, pred, meas, none = analyse(c)
        dh = step * np.tan(np.radians(a))
        r = dict(step=step, N=N, alpha=a, dh_neighbour=dh, pred_db=pred, meas_whole_db=meas, meas_none_db=none,
                 **{f"absg_ch{i}": float(np.abs(g[:, i]).mean()) for i in range(4)},
                 **{f"dist_ch{i}": float(dist[:, i].mean()) for i in range(4)})
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    with open("probes/sata_gain.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
