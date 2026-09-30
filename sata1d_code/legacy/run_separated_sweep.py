"""
Separated targets, SATA 1D: the cases of the geometry sweep with the SAME heights,
but the targets 2.6 km apart instead of 200 m.
2.6 km > 2 hw = 2.52 km (each target has its own block in the delta_C0 map) and every
pair distance k * 2.6 km stays >= 1 km away from a multiple of X = 6.23 km (Doppler-alias
collision core +-0.63 km), which limits N to 5.
Also a spacing sweep for two scenes (along-track ramp N = 5, 20 deg; random heights N = 5).

Outputs (default ./geometry_sweep_separated/):
    bxt100/<family>/<id>_<slug>.png|.npz, sweep_results.csv
    spacing_sweep.png, spacing_sweep.csv
    close_vs_separated.png
Run:  python run_separated_sweep.py [--workers 2]
"""
import argparse
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_geometry_sweep as G

SEP = 2600.0
FAMS = ["ramp", "random_heights", "piecewise"]
N_MAX = 5
X_KM = 6.2287
SPACINGS = [200, 400, 600, 800, 1000, 1200, 1600, 2000, 2600]


def separated_cases():
    out = []
    for c in G.build_cases(FAMS):
        if c.N > N_MAX:
            continue
        c.spacing = SEP
        c.desc = c.desc + r", same heights as the 200-m case, targets 2.6 km apart"
        out.append(c)
    return out


def collides(spacing, N):
    d = np.arange(1, N) * spacing / 1e3
    m = np.maximum(1, np.round(d / X_KM))
    return bool(np.any(np.abs(d - X_KM * m) < 0.63))


def spacing_sweep(out_dir):
    scenes = {"along-track ramp N=5, 20 deg": [c for c in G.build_cases(["ramp"]) if c.slug == "N5_a20"][0],
              "random heights N=5, hmax 500 m (seed 0)":
                  [c for c in G.build_cases(["random_heights"]) if c.slug == "N5_hmax500_seed0"][0]}
    rows = []
    for name, base in scenes.items():
        for s in SPACINGS:
            c = G.Case(base.family, base.slug, base.desc, base.heights.copy(), float(s), dict(base.params))
            cfg, foc = G.simulate(c, G.BXT_MAX)
            m = G.metrics(cfg, foc, c)
            rows.append(dict(scene=name, spacing=s, collision=collides(s, c.N),
                             **{k: round(float(v), 2) for k, v in m.items() if k.startswith("amb")}))
            print(name, s, {k: rows[-1][k] for k in ("ambpk_db_none", "ambpk_db_whole", "ambpk_db_sub", "ambpk_db_oracle")}, flush=True)
    with open(os.path.join(out_dir, "spacing_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for tag, name in zip(("ramp", "random"), scenes):
        r = [x for x in rows if x["scene"] == name]
        s_ = np.array([x["spacing"] for x in r]) / 1e3
        fig, ax = plt.subplots(figsize=(10, 5))
        for k, col, lab, ls in (("ambpk_db_ref", "tab:green", "reference", "-"), ("ambpk_db_none", "tab:red", "no SATA", "--"),
                                ("ambpk_db_whole", "tab:blue", "SATA whole band", "-"),
                                ("ambpk_db_sub", "tab:orange", "SATA per sub-band", ":")):
            ax.plot(s_, [x[k] for x in r], ls, marker="o", color=col, label=lab, lw=1.6)
        for x in r:
            if x["collision"]:
                ax.axvspan(x["spacing"] / 1e3 - 0.1, x["spacing"] / 1e3 + 0.1, color="0.85", zorder=0)
        ax.axvline(2.52, color="k", lw=0.8, ls=":"); ax.text(2.53, -63, r"$2h_w$ = 2.52 km", fontsize=8)
        ax.axvline(0.309, color="k", lw=0.8, ls=":"); ax.text(0.32, -63, r"$\delta x$ = 309 m", fontsize=8)
        ax.set_xlabel("spacing between neighbouring targets [km]")
        ax.set_ylabel("ambiguity peak at $x_j/v \\pm m\\,0.81$ s [dB]")
        ax.set_title(f"SATA 1D | {name} | same heights, increasing spacing\n"
                     "grey: a pair of targets within 0.63 km of a multiple of 6.23 km (Doppler-alias collision)", fontsize=10)
        ax.grid(alpha=0.3); ax.set_ylim(-65, -5); ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"spacing_sweep_{tag}.png"), dpi=130); plt.close(fig)


def close_vs_separated(out_dir, close_csv):
    sep = {(r["family"], r["slug"]): r for r in csv.DictReader(open(os.path.join(out_dir, "sweep_results.csv")))}
    close = {(r["family"], r["slug"]): r for r in csv.DictReader(open(close_csv)) if float(r["bxt_max"]) > 0}
    ramp = sorted([k for k in sep if k[0] == "ramp"], key=lambda k: (int(k[1].split("_")[0][1:]), float(k[1].split("_a")[1])))
    Ns = sorted({int(k[1].split("_")[0][1:]) for k in ramp}); As = sorted({float(k[1].split("_a")[1]) for k in ramp})
    f = lambda d, k, key: float(d[k][key])
    panels = [("improvement over no SATA, 200 m apart", close, "imp"), ("improvement over no SATA, 2.6 km apart", sep, "imp"),
              ("gap SATA - oracle, 200 m apart", close, "gap"), ("gap SATA - oracle, 2.6 km apart", sep, "gap")]
    fig, axs = plt.subplots(2, 2, figsize=(16, 7.5), gridspec_kw=dict(hspace=0.5, wspace=0.22))
    for ax, (title, d, kind) in zip(axs.ravel(), panels):
        M = np.full((len(Ns), len(As)), np.nan)
        for k in ramp:
            i, j = Ns.index(int(k[1].split("_")[0][1:])), As.index(float(k[1].split("_a")[1]))
            M[i, j] = (f(d, k, "amb_db_none") - f(d, k, "amb_db_whole")) if kind == "imp" else \
                      (f(d, k, "amb_db_whole") - f(d, k, "amb_db_oracle"))
        cmap, vmin, vmax = (G_CMAP, -40, 40) if kind == "imp" else ("Reds", 0, 20)
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(As)), [f"{a:g}" for a in As]); ax.set_yticks(range(len(Ns)), Ns)
        ax.set_xlabel(r"inclination $\alpha$ of the 200-m ramp [deg] (heights $j\cdot200\tan\alpha$)"); ax.set_ylabel("N targets")
        for i in range(len(Ns)):
            for j in range(len(As)):
                ax.text(j, i, f"{M[i, j]:.1f}".replace("-0.0", "0.0"), ha="center", va="center", fontsize=8,
                        color="white" if (kind == "imp" and abs(M[i, j]) > 17) or (kind == "gap" and M[i, j] > 12) else "#222222")
        ax.set_title(f"SATA 1D whole band, along-track ramp: {title} [dB]", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.savefig(os.path.join(out_dir, "close_vs_separated.png"), dpi=130, bbox_inches="tight"); plt.close(fig)


from matplotlib.colors import LinearSegmentedColormap
G_CMAP = LinearSegmentedColormap.from_list("improv", ["#B2182B", "#E8A48C", "#E4E4E4", "#8FB8DB", "#1F5FA8", "#0B2E5C"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="geometry_sweep_separated")
    ap.add_argument("--close-csv", default="geometry_sweep_v2/sweep_results.csv")
    a = ap.parse_args()
    cases = separated_cases()
    jobs, counter = [], {}
    for c in cases:
        counter[c.family] = counter.get(c.family, 0) + 1
        cid = f"{G.PREFIX[c.family]}S-{counter[c.family]:03d}"
        d = os.path.join(a.out, "bxt100", c.family); os.makedirs(d, exist_ok=True)
        jobs.append((c, G.BXT_MAX, cid, os.path.join(d, f"{cid}_{c.slug}_sep2600.png")))
    print(len(jobs), "cases", flush=True)
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        rows = pool.map(G._job, jobs, chunksize=1)
    with open(os.path.join(a.out, "sweep_results.csv"), "w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"cases done in {time.time() - t0:.0f} s", flush=True)
    spacing_sweep(a.out)


if __name__ == "__main__":
    main()
