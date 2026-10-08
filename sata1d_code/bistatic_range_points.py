"""
Topographic scene with cross-track baselines only (b_at = 0, b_xt != 0): a few points of the
terrain, their bistatic range histories R_T(t) + R_R,i(t) on one plot, and C0, C1, C2 of every
point and channel (GetCoeffNu, the coefficients of the reconstruction filters).

    python bistatic_range_points.py
    python bistatic_range_points.py --channel 0 --bxt 100
"""
import argparse
import csv
import os
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sata1d as S

# chosen points: (azimuth x [m], ground-range offset dy [m]) relative to the reference point
POINTS = [(-3000.0, -1500.0), (-1500.0, 500.0), (0.0, 0.0), (1000.0, 1500.0), (2500.0, -500.0), (4000.0, 2500.0)]


def terrain(x, y):
    """Topography h(x, y) [m]: a 900 m hill, a 400 m ridge and a gentle slope in ground range."""
    hill = 900.0 * np.exp(-0.5 * (((x - 1000.0) / 1800.0) ** 2 + ((y - 1000.0) / 1500.0) ** 2))
    ridge = 400.0 * np.exp(-0.5 * ((x + 2500.0) / 900.0) ** 2) * np.exp(-0.5 * ((y + 800.0) / 2500.0) ** 2)
    return hill + ridge + 0.05 * (y + 4000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bxt", type=float, default=100.0, help="b_xt ~ U(0, bxt) [m]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channel", type=int, default=0, help="channel of the range-history plot")
    ap.add_argument("--out", default="bistatic_range_points")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = S.make_config([], bxt_max=a.bxt, seed=a.seed)
    cfg = replace(cfg, array=replace(cfg.array, bat=np.zeros(cfg.Nrx)))          # b_at = 0
    tr = S.build_platform_tracks(cfg)
    ptg0, wl, ta = cfg.scene.ptg, cfg.system.wl, cfg.ta
    pts = np.array([ptg0 + np.array([x, y, terrain(x, y)]) for x, y in POINTS])

    def coeffs(p, i):
        return S.GetCoeffNu(p, tr.ptx, tr.prx[i], tr.vtx, tr.vrx[i], tr.ptx, tr.vtx, cfg.prf, wl, ta,
                            cfg.sq_tx, cfg.sq_rx[i], cfg.theta_tx, cfg.theta_rx[i])
    ref = np.array([coeffs(ptg0, i) for i in range(cfg.Nrx)])
    C = np.array([[coeffs(p, i) for i in range(cfg.Nrx)] for p in pts])            # [point, channel, (C0, C1, C2, Dt)]

    rows = []
    print(f"b_at = {cfg.array.bat} m, b_xt = {np.round(cfg.array.bxt, 2)} m, reference point {np.round(ptg0, 1)}")
    print(f"{'point':>5} {'x':>7} {'dy':>7} {'h':>6} {'ch':>3} {'C0 [m]':>12} {'C1 [s]':>12} {'C2 [s^2/m]':>12} "
          f"{'dC0 [mm]':>10} {'dC1 [us]':>10} {'dC2':>10}")
    for j, ((x, y), p) in enumerate(zip(POINTS, pts)):
        for i in range(cfg.Nrx):
            c0, c1, c2, dt = C[j, i]
            r = dict(point=f"P{j + 1}", x_m=x, dy_m=y, h_m=round(p[2] - ptg0[2], 2), channel=i, bxt_m=round(cfg.array.bxt[i], 2),
                     C0_m=c0, C1_s=c1, C2_s2_per_m=c2, Dt_s=dt,
                     dC0_mm=1e3 * (c0 - ref[i, 0]), dC1_us=1e6 * (c1 - ref[i, 1]), dC2=c2 - ref[i, 2])
            rows.append(r)
            print(f"{r['point']:>5} {x:7.0f} {y:7.0f} {r['h_m']:6.0f} {i:3d} {c0:12.6f} {c1:12.4e} {c2:12.4e} "
                  f"{r['dC0_mm']:10.3f} {r['dC1_us']:10.4f} {r['dC2']:10.3e}")
    for i in range(cfg.Nrx):
        print(f"  ref    0       0      0 {i:3d} {ref[i, 0]:12.6f} {ref[i, 1]:12.4e} {ref[i, 2]:12.4e}")
    with open(os.path.join(a.out, "coefficients.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- figure: terrain + points | bistatic range histories | bistatic - monostatic ----
    i = a.channel
    cols = plt.cm.tab10(np.arange(len(pts)))
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.25, 1.25], hspace=0.32, wspace=0.28)
    a0 = fig.add_subplot(gs[:, 0])
    xg, yg = np.meshgrid(np.linspace(-6000, 6000, 241), np.linspace(-4000, 4000, 161))
    im = a0.contourf(xg / 1e3, yg / 1e3, terrain(xg, yg), levels=20, cmap="terrain")
    fig.colorbar(im, ax=a0, label="height h [m]", shrink=0.8)
    for j, ((x, y), p) in enumerate(zip(POINTS, pts)):
        a0.plot(x / 1e3, y / 1e3, "o", ms=9, mfc=cols[j], mec="k")
        a0.annotate(f"P{j + 1} ({p[2] - ptg0[2]:.0f} m)", (x / 1e3, y / 1e3), xytext=(6, 6) if x < 3000 else (-6, 8), ha="left" if x < 3000 else "right",
                    textcoords="offset points", fontsize=9)
    a0.plot(0, 0, "k+", ms=12, mew=2)
    a0.set_xlabel("azimuth x [km]"); a0.set_ylabel("ground range offset $\\Delta y$ [km]")
    a0.set_title("Topography and chosen points (+ = reference point, h = 0)", fontsize=10)

    a1 = fig.add_subplot(gs[0, 1:]); a2 = fig.add_subplot(gs[1, 1:])
    for j, p in enumerate(pts):
        RT = np.linalg.norm(tr.ptx - p, axis=1)
        RR = np.linalg.norm(tr.prx[i] - p, axis=1)
        a1.plot(ta, (RT + RR) / 1e3, color=cols[j], lw=1.8,
                label=f"P{j + 1}: $C_0$ = {C[j, i, 0]:.4f} m, $C_1$ = {C[j, i, 1]:.3e} s, $C_2$ = {C[j, i, 2]:.3e} s$^2$/m")
        a2.plot(ta, 1e3 * (RT + RR - 2 * RT), color=cols[j], lw=1.8, label=f"P{j + 1}")
    RT0 = np.linalg.norm(tr.ptx - ptg0, axis=1); RR0 = np.linalg.norm(tr.prx[i] - ptg0, axis=1)
    a1.plot(ta, (RT0 + RR0) / 1e3, "k--", lw=1.2, label=f"reference: $C_0$ = {ref[i, 0]:.4f} m")
    a2.plot(ta, 1e3 * (RR0 - RT0), "k--", lw=1.2, label="reference")
    a1.set_ylabel("bistatic range $R_T + R_{R}$ [km]"); a1.set_title(f"Bistatic range histories, channel {i} "
                 f"($b_{{xt}}$ = {cfg.array.bxt[i]:.1f} m, $b_{{at}}$ = 0)", fontsize=10)
    a2.set_ylabel("$R_T + R_R - 2R_T = R_R - R_T$ [mm]")
    a2.set_title("Bistatic minus monostatic range (what C0, C1, C2 describe)", fontsize=10)
    for ax in (a1, a2):
        ax.set_xlabel("azimuth time [s]"); ax.grid(alpha=0.3); ax.set_xlim(ta[0], ta[-1])
    a1.legend(fontsize=8, loc="upper center"); a2.legend(fontsize=8, ncol=4, loc="best")
    fig.text(0.5, 0.01, f"b_xt = [" + ", ".join(f"{b:.1f}" for b in cfg.array.bxt) + "] m, b_at = 0. "
             "C0, C1, C2 from GetCoeffNu (parabolas fitted over the common illuminated aperture).",
             ha="center", fontsize=9, color="0.35")
    png = os.path.join(a.out, f"bistatic_range_points_ch{i}.png")
    fig.savefig(png, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {png} and {os.path.join(a.out, 'coefficients.csv')}")


if __name__ == "__main__":
    main()
