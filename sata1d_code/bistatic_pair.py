"""
Reference point and one topographic target, b_xt != 0 and b_at = 0: the bistatic-minus-monostatic
range (R_T + R_R) - 2 R_T = R_R - R_T of the two against time (its minimum is C0), one panel per
channel, and C0, C1, C2 of each (GetCoeffNu). --quantity bistatic plots R_T + R_R itself instead.

    python bistatic_pair.py                   # target x = 1500 m, h = 500 m, b_xt ~ U(0, 100 m)
    python bistatic_pair.py --x 0 --h 1000
    python bistatic_pair.py --quantity bistatic
    python bistatic_pair.py --x 0 --h-list 100 300 500 1000 2000 3000 --bxt-sets 10,30,60,100 120,180,240,300 --pdf
                                              # one figure per (h, set of 4 b_xt), all in <out>.pdf

bxt-sets : explicit b_xt of the Nrx channels [m], comma-separated, one set per figure (instead of U(0, bxt))
"""
import argparse
import os
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sata1d as S


def run_case(x, h, bxt, seed, quantity, out, bxt_values=None, tag=""):
    os.makedirs(out, exist_ok=True)
    cfg = S.make_config([(x, h)], bxt_max=bxt, seed=seed)
    cfg = replace(cfg, array=replace(cfg.array, bat=np.zeros(cfg.Nrx),
                                     bxt=cfg.array.bxt if bxt_values is None else np.asarray(bxt_values, float)))
    tr = S.build_platform_tracks(cfg)
    ta = cfg.ta
    pts = {"reference (h = 0)": cfg.scene.ptg, f"target (x = {x:.0f} m, h = {h:.0f} m)": cfg.scene.points[1]}
    cols = ["k", "tab:red"]
    print(f"b_at = {cfg.array.bat} m, b_xt = {np.round(cfg.array.bxt, 2)} m")
    print(f"{'point':34s} {'ch':>2} {'C0 [m]':>12} {'C1 [s]':>10} {'C2 [s^2/m]':>12} {'Dt [s]':>8}")

    fig, axs = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for i, ax in enumerate(axs.ravel()):
        curves, Cs, ds, ks = [], [], [], []
        for (name, p), c in zip(pts.items(), cols):
            C0, C1, C2, Dt = S.GetCoeffNu(p, tr.ptx, tr.prx[i], tr.vtx, tr.vrx[i], tr.ptx, tr.vtx, cfg.prf, cfg.system.wl,
                                          ta, cfg.sq_tx, cfg.sq_rx[i], cfg.theta_tx, cfg.theta_rx[i])
            print(f"{name:34s} {i:2d} {C0:12.6f} {C1:10.1e} {C2:12.4e} {Dt:8.1e}")
            RT = np.linalg.norm(tr.ptx - p, axis=1); RR = np.linalg.norm(tr.prx[i] - p, axis=1)
            curves.append(RT + RR); Cs.append((C0, C1, C2)); ds.append(RT + RR - 2 * RT); ks.append(np.argmin(RT + RR))
            if quantity in ("pair", "aligned_pair"):
                continue
            y = (RT + RR) / 1e3 if quantity == "bistatic" else 1e3 * (RT + RR - 2 * RT)
            k = np.argmin(RT + RR)
            t_axis = ta if quantity == "bistatic" else ta - ta[k]      # difference: time from the minimum of R_T + R_R, as in the fit
            ax.plot(t_axis, y, color=c, lw=1.8,
                    label=f"{name}\n$C_0$ = {C0:.4f} m, $C_1$ = {C1:.1e} s, $C_2$ = {C2:.4e} s$^2$/m")
            ax.plot(t_axis[k], y[k], "v", ms=9, color=c, mec="k")
        if quantity == "pair":
            ax.plot(ta, curves[1] - curves[0], color="tab:purple", lw=1.8,
                    label=f"target $-$ reference; $\\Delta C_0$ = {1e3 * (Cs[1][0] - Cs[0][0]):.3f} mm")
            ax.axhline(0, color="0.5", lw=0.7)
        if quantity == "aligned_pair":
            n = min(ks[0], ks[1], len(ta) - 1 - ks[0], len(ta) - 1 - ks[1])
            tau = ta[ks[0] - n:ks[0] + n + 1] - ta[ks[0]]
            diff = ds[1][ks[1] - n:ks[1] + n + 1] - ds[0][ks[0] - n:ks[0] + n + 1]
            dC0, dC1, dC2 = (Cs[1][j] - Cs[0][j] for j in range(3))
            ax.plot(tau, 1e3 * diff, color="tab:purple", lw=1.8,
                    label=f"target $-$ reference; at $\\tau$ = 0: {1e3 * diff[n]:.3f} mm\n"
                          f"$\\Delta C_0$ = {1e3 * dC0:.3f} mm, $\\Delta C_1$ = {dC1:.1e} s, $\\Delta C_2$ = {dC2:.3e} s$^2$/m")
            ax.plot(0, 1e3 * diff[n], "v", ms=9, color="tab:purple", mec="k")
        ax.set_title(f"channel {i}: $b_{{xt}}$ = {cfg.array.bxt[i]:.1f} m, $b_{{at}}$ = 0 | target h = {h:.0f} m", fontsize=10)
        ax.set_ylabel({"bistatic": "$R_T + R_R$ [km]", "difference": "$(R_T + R_R) - 2R_T$ [mm]",
                       "pair": "$(R_T + R_R)_{target} - (R_T + R_R)_{reference}$ [m]",
                       "aligned_pair": "$[(R_T + R_R) - 2R_T]_{target} - [(R_T + R_R) - 2R_T]_{reference}$ [mm]"}[quantity])
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper center" if quantity == "bistatic" else "best")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    for ax in axs[1]:
        ax.set_xlabel("azimuth time [s]" if quantity not in ("difference", "aligned_pair") else
                      "$\\tau$ = time from the minimum of $R_T + R_R$ of each point [s]")
    fig.suptitle({"bistatic": "Bistatic range histories $R_T + R_R$ of the reference point and of the target ($\\blacktriangledown$ = minimum)",
                  "pair": "Difference between the bistatic range histories of the target and of the reference point, "
                          "$(R_T + R_R)_{target}(t) - (R_T + R_R)_{reference}(t)$",
                  "aligned_pair": "Bistatic minus monostatic range of the target minus that of the reference point, "
                                  "both minima at $\\tau$ = 0 ($\\blacktriangledown$ = $\\tau$ = 0, where it equals $\\Delta C_0$)"}[quantity]
                 if quantity != "difference" else
                 "Bistatic minus monostatic range, $(R_T + R_R) - 2R_T = R_R - R_T$, of the reference point and of the target, "
                 "both minima at $\\tau$ = 0 ($\\blacktriangledown$ = vertex, where it equals $C_0$)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    stem = {"bistatic": "bistatic_pair", "difference": "bistatic_minus_monostatic_pair",
            "pair": "bistatic_pair_difference", "aligned_pair": "bistatic_minus_monostatic_pair_difference"}[quantity]
    png = os.path.join(out, f"{stem}{tag}.png")
    fig.savefig(png, dpi=100); plt.close(fig)
    print(f"wrote {png}")
    return png


def bundle_pdf(pngs, out):
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out) as pdf:
        for png in pngs:
            im = mpimg.imread(png)
            hh, ww = im.shape[:2]
            fig = plt.figure(figsize=(ww / 100.0, hh / 100.0))
            ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(im); ax.axis("off")
            pdf.savefig(fig); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=1500.0, help="azimuth of the target [m]")
    ap.add_argument("--h", type=float, default=500.0, help="height of the target [m] (on the iso-range surface)")
    ap.add_argument("--bxt", type=float, default=100.0, help="b_xt ~ U(0, bxt)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quantity", default="aligned_pair", choices=["aligned_pair", "pair", "difference", "bistatic"],
                    help="aligned_pair: [(R_T + R_R) - 2 R_T]_target - [(R_T + R_R) - 2 R_T]_reference, each at its own tau [mm]; "
                         "pair: (R_T + R_R)_target - (R_T + R_R)_reference at the same t [m]; "
                         "difference: (R_T + R_R) - 2 R_T of each at its own tau [mm]; bistatic: R_T + R_R of each [km]")
    ap.add_argument("--h-list", type=float, nargs="*", default=None, help="sweep: heights of the target [m]")
    ap.add_argument("--bxt-sets", nargs="*", default=None,
                    help="sweep: explicit b_xt of the channels [m], comma-separated, one set per figure")
    ap.add_argument("--pdf", action="store_true", help="sweep: bundle the figures into <out>.pdf")
    ap.add_argument("--out", default=None, help="default bistatic_pair (single case) or bistatic_pair_sweep")
    a = ap.parse_args()
    if a.h_list is None and a.bxt_sets is None:
        run_case(a.x, a.h, a.bxt, a.seed, a.quantity, a.out or "bistatic_pair")
        return
    out = a.out or "bistatic_pair_sweep"
    hs = a.h_list or [a.h]
    sets = [[float(v) for v in s.split(",")] for s in a.bxt_sets] if a.bxt_sets else [None]
    pngs = []
    for h in hs:
        for bx in sets:
            tag = f"_h{h:04.0f}" + ("" if bx is None else "_bxt" + "-".join(f"{v:g}" for v in bx))
            pngs.append(run_case(a.x, h, a.bxt, a.seed, a.quantity, out, bxt_values=bx, tag=tag))
    if a.pdf:
        print("wrote", bundle_pdf(pngs, out + ".pdf"))


if __name__ == "__main__":
    main()
