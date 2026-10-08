"""
Reference point and one topographic target, b_xt != 0 and b_at = 0: bistatic range histories of
the pair, the bistatic-minus-monostatic residual R_R - R_T of each (the curve that C0, C1, C2
describe), its fitted parabola, and the analytic prediction

    R_R(t) - R_T(t) = C0 (r0 + rR) / (R_T(t) + R_R(t)),   C0 = rR - r0   (exact for b_at = 0),

with the coefficients of GetCoeffNu next to the analytic ones.

    python c0_time_dependence.py                        # target x = 1500 m, h = 500 m, channel 0 in the figure
    python c0_time_dependence.py --x 0 --h 1000 --channel 1
"""
import argparse
import os
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sata1d as S


def coeffs(cfg, tr, p, i):
    return S.GetCoeffNu(p, tr.ptx, tr.prx[i], tr.vtx, tr.vrx[i], tr.ptx, tr.vtx, cfg.prf, cfg.system.wl,
                        cfg.ta, cfg.sq_tx, cfg.sq_rx[i], cfg.theta_tx, cfg.theta_rx[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=1500.0, help="azimuth of the target [m]")
    ap.add_argument("--h", type=float, default=500.0, help="height of the target [m] (on the iso-range surface)")
    ap.add_argument("--bxt", type=float, default=100.0, help="b_xt ~ U(0, bxt)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--out", default="c0_time_dependence")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = S.make_config([(a.x, a.h)], bxt_max=a.bxt, seed=a.seed)
    cfg = replace(cfg, array=replace(cfg.array, bat=np.zeros(cfg.Nrx)))
    tr = S.build_platform_tracks(cfg)
    v, H, ta, wl = cfg.system.vs, cfg.scene.H, cfg.ta, cfg.system.wl
    pts = {"reference": cfg.scene.ptg, "target": cfg.scene.points[1]}
    print(f"b_at = {cfg.array.bat}, b_xt = {np.round(cfg.array.bxt, 2)} m")
    for name, p in pts.items():
        print(f"{name:9s}: x = {p[0] - cfg.scene.ptg[0]:7.1f} m, y = {p[1]:9.1f} m, h = {p[2]:6.1f} m")

    # numerical (GetCoeffNu) and analytic coefficients, every channel
    print(f"\n{'point':9s} {'ch':>2} {'bxt':>6} | {'C0 num [m]':>12} {'C0 an [m]':>12} {'diff [um]':>9} | "
          f"{'C1 [s]':>9} | {'C2 num':>11} {'C2 an':>11} | {'Dt [s]':>9}")
    rows = {}
    for i in range(cfg.Nrx):
        for name, p in pts.items():
            C0, C1, C2, Dt = coeffs(cfg, tr, p, i)
            y, z = p[1], H - p[2]
            r0, rR = np.hypot(y, z), np.hypot(y - cfg.array.bxt[i], z)
            c2, d2 = v ** 2 / r0, 0.5 * v ** 2 * (1.0 / rR - 1.0 / r0)
            C0a, C2a = rR - r0, d2 / (4.0 * c2 * (c2 + d2))
            rows[(name, i)] = (C0, C1, C2, Dt, C0a, C2a, r0, rR)
            print(f"{name:9s} {i:2d} {cfg.array.bxt[i]:6.1f} | {C0:12.6f} {C0a:12.6f} {1e6 * (C0 - C0a):9.2f} | "
                  f"{C1:9.1e} | {C2:11.4e} {C2a:11.4e} | {Dt:9.1e}")
    for i in range(cfg.Nrx):
        d = rows[("target", i)][0] - rows[("reference", i)][0]
        da = rows[("target", i)][4] - rows[("reference", i)][4]
        print(f"channel {i}: delta_C0 numeric {1e3 * d:9.3f} mm, analytic {1e3 * da:9.3f} mm, "
              f"-b_xt dy / r0 = {-1e3 * cfg.array.bxt[i] * (pts['target'][1] - pts['reference'][1]) / rows[('reference', i)][6]:9.3f} mm")

    # figure: channel a.channel
    i = a.channel
    cols = {"reference": "k", "target": "tab:red"}
    fig, axs = plt.subplots(4, 1, figsize=(11, 15))
    ax0, ax1, ax3, ax2 = axs
    t_ill = rows[("reference", i)][6] * np.tan(cfg.theta_tx / 2.0) / v          # half illumination time [s]
    hist = {}
    for name, p in pts.items():
        RT = np.linalg.norm(tr.ptx - p, axis=1); RR = np.linalg.norm(tr.prx[i] - p, axis=1)
        k = np.argmin(RT + RR)
        C0, C1, C2, Dt, C0a, C2a, r0, rR = rows[(name, i)]
        hist[name] = (RT, RR, k, C0, C2, r0, rR)
        ax0.plot(ta, (RT + RR) / 1e3, color=cols[name], lw=1.8, label=f"{name}: $R_T + R_R$ (bistatic)")
        ax0.plot(ta, 2 * RT / 1e3, color=cols[name], lw=1.2, ls="--", label=f"{name}: $2R_T$ (monostatic)")
        ax0.plot(ta[k], (RT[k] + RR[k]) / 1e3, "v", ms=9, color=cols[name], mec="k")
        tau = ta - ta[k]
        dR = RR - RT
        pred = C0 * (r0 + rR) / (RT + RR)
        ax1.plot(tau, 1e3 * dR, color=cols[name], lw=2.0, label=f"{name}: $R_R - R_T$ (exact geometry)")
        ax1.plot(tau, 1e3 * pred, color=cols[name], lw=1.2, ls=":", label=f"{name}: $C_0\\,(r_0 + r_R)/(R_T + R_R)$")
        ax1.axhline(1e3 * C0, color=cols[name], lw=0.8, ls="--")
        ax1.annotate(f"$C_0$ = {1e3 * C0:.3f} mm", (tau[0], 1e3 * C0), xytext=(4, 4 if name == "reference" else -12),
                     textcoords="offset points", fontsize=9, color=cols[name])
        d2 = 0.5 * v ** 2 * (1.0 / rR - 1.0 / r0)
        ax3.plot(tau, 1e3 * (dR - C0), color=cols[name], lw=2.0, label=f"{name}: $R_R - R_T - C_0$ (exact)")
        ax3.plot(tau, 1e3 * d2 * tau ** 2, color=cols[name], lw=1.2, ls=":", label=f"{name}: parabola $d_2\\tau^2$, $d_2$ = {d2:.3e} m/s$^2$")
        print(f"channel {i}, {name}: max |(R_R - R_T) - C0 (r0 + rR)/(R_T + R_R)| = "
              f"{1e6 * np.abs(dR - pred).max():.3f} um; R_R - R_T spans {1e3 * (dR.max() - dR.min()):.2f} mm over the line")
    RTr, RRr, kr, C0r, _, r0r, rRr = hist["reference"]; RTt, RRt, kt, C0t, _, r0t, rRt = hist["target"]
    n = min(kr, kt, len(ta) - kr - 1, len(ta) - kt - 1)
    sl_r, sl_t = slice(kr - n, kr + n + 1), slice(kt - n, kt + n + 1)
    tau = ta[sl_r] - ta[kr]
    diff = (RRt - RTt)[sl_t] - (RRr - RTr)[sl_r]
    ax2.plot(tau, 1e6 * (diff - (C0t - C0r)), color="tab:purple", lw=2.0,
             label=f"$[(R_R - R_T)_{{target}} - (R_R - R_T)_{{reference}}] - \\Delta C_0$, each at its own $\\tau$; $\\Delta C_0$ = {1e3 * (C0t - C0r):.3f} mm")
    ax2.axhline(0, color="tab:purple", lw=0.8, ls="--")
    print(f"channel {i}: (R_R - R_T)_t - (R_R - R_T)_ref varies by {1e6 * (diff.max() - diff.min()):.2f} um over +-{tau[-1]:.2f} s "
          f"(phase {360 * (diff.max() - diff.min()) / wl:.4f} deg)")
    T = tau[-1]
    ax0.set_ylabel("range [km]"); ax0.set_title(f"Range histories, channel {i} ($b_{{xt}}$ = {cfg.array.bxt[i]:.1f} m, $b_{{at}}$ = 0); "
                                                 "$\\blacktriangledown$ = minimum", fontsize=10)
    ax0.set_xlabel("azimuth time t [s]")
    ax1.set_ylabel("$R_R - R_T$ [mm]"); ax1.set_xlabel("$\\tau$ = time from the minimum of each history [s]")
    ax1.set_title("Bistatic minus monostatic range: the curve whose parabola gives $C_0$ (vertex), $C_1$ (slope), $C_2$ (curvature)", fontsize=10)
    ax2.set_ylabel("[$\\mu$m]"); ax2.set_xlabel("$\\tau$ [s]")
    ax2.set_title("What the SATA corrects, minus its constant part: residual of the target minus residual of the reference, minus $\\Delta C_0$", fontsize=10)
    ax3.set_ylabel("$R_R - R_T - C_0$ [mm]"); ax3.set_xlabel("$\\tau$ [s]")
    ax3.set_title("The time-varying part of each residual: a parabola in $\\tau$ (this is what $C_2$ carries, not $C_0$)", fontsize=10)
    for ax in (ax1, ax3, ax2):
        ax.axvspan(-t_ill, t_ill, color="gold", alpha=0.15, lw=0)
    for ax in axs:
        ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="best")
    ax3.legend(fontsize=8.5, loc="upper right")
    ax3.text(0, 0.92 * ax3.get_ylim()[1], f"illuminated: $|\\tau| \\leq$ {t_ill:.2f} s\n(window of the parabola fit)", ha="center", va="top",
             fontsize=8.5, color="0.35")
    for ax in (ax1, ax3, ax2):
        ax.set_xlim(-T, T)
    fig.suptitle(f"Reference (h = 0) and target (x = {a.x:.0f} m, h = {a.h:.0f} m, same $r_0$) | "
                 f"$b_{{xt}}$ = [" + ", ".join(f"{b:.1f}" for b in cfg.array.bxt) + "] m, $b_{at}$ = 0", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(a.out, f"c0_time_dependence_ch{i}.png")
    fig.savefig(png, dpi=100); plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
