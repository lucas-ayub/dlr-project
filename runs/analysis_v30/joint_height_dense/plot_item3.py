"""Figures of item 3 (from cache/dense_*.npz, cache/item3.json):
  fig/dense_irf.(pdf|png)      focused lines of scenes full of targets (two terrains, two densities)
  fig/dense_summary.(pdf|png)  image error and highest-ambiguity gap vs number of targets, all terrains
    python3 -I plot_item3.py"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from style import plt, COL, NAME, INK, MUTED, TERRAIN, FILL, AMBER, B300, db, mismatch, save
from common import CACHE, FIG
import item3_dense as I

EX = [("mountain", 20), ("mountain", 200), ("foothills", 20), ("foothills", 200)]
TN = {"mountain": "isolated mountain, 1000 m", "mountains": "random mountains", "alpine": "fractal alpine relief",
      "foothills": "SRTM: western foothills", "escarpment": "SRTM: eastern escarpment", "highsierra": "SRTM: High Sierra"}
MK = {"mountain": "o", "mountains": "s", "alpine": "^", "foothills": "D", "escarpment": "v", "highsierra": "P"}


def load(name, n, s):
    return dict(np.load(os.path.join(CACHE, f"dense_{name}_N{n:03d}_s{s}.npz"), allow_pickle=True))


def fig_irf():
    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(3 * len(EX), 2, height_ratios=[0.32, 0.3, 1] * len(EX), hspace=0.06)
    for r, (name, n) in enumerate(EX):
        d = load(name, n, 0)
        m = json.loads(str(d["metrics"]))
        x = d["t"] * float(d["vs"]) / 1e3
        pk = np.abs(d["ref"]).max()
        for c, k in enumerate(("std", "joint")):
            ta = fig.add_subplot(gs[3 * r + 1, c])
            lo = d["hgrid"].min() - 50
            ta.fill_between(d["tgrid"] / 1e3, lo, d["hgrid"], color=FILL, lw=0)
            ta.plot(d["tgrid"] / 1e3, d["hgrid"], color=TERRAIN, lw=0.8)
            ta.plot(d["xs"] / 1e3, d["hs"], "o", ms=1.6, color=INK)
            ta.set_xlim(-20, 20); ta.set_ylim(lo, d["hgrid"].max() + 0.15 * np.ptp(d["hgrid"]))
            ta.set_xticklabels([]); ta.grid(False); ta.set_yticks([]); ta.spines["left"].set_visible(False)
            if c == 0:
                ta.set_title(f"{TN[name]}, {n} targets", fontsize=8.8, pad=2)
            a = fig.add_subplot(gs[3 * r + 2, c])
            a.plot(x, db(d["ref"], pk), color=COL["ref"], lw=0.5)
            a.plot(x, db(d[k], pk), color=COL[k], lw=0.5, alpha=0.85)
            a.set_ylim(-70, 4); a.set_xlim(-20, 20); a.set_yticks([-60, -30, 0])
            a.text(0.99, 0.97, ("image error " + f"{m[k]['err']:.1f}".replace("-", "\u2212") + " dB"), transform=a.transAxes, ha="right", va="top",
                   fontsize=8, color=INK, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
            if c:
                a.set_yticklabels([])
            else:
                a.set_ylabel("[dB]")
            if r < len(EX) - 1:
                a.set_xticklabels([])
                a.tick_params(axis="x", length=0)
            else:
                a.set_xlabel("position along the line [km]")
    fig.text(0.13, 0.955, "standard (red) vs reference (green)", fontsize=9.2, va="bottom", color=INK)
    fig.text(0.56, 0.955, "joint (purple) vs reference (green)", fontsize=9.2, va="bottom", color=INK)
    fig.text(0.13, 0.025, "brown: terrain (relative to the line mean), dots: the targets; 0 dB = reference peak",
             fontsize=8, color=MUTED, va="top")
    fig.subplots_adjust(hspace=0.06, top=0.95, bottom=0.08)
    for r in range(len(EX)):
        pass
    save(fig, FIG, "dense_irf")


def blind_min(name, xs, X):
    ter, _ = I.terrain(name)
    return min(mismatch(B300, float(ter(x + m * X) - ter(x)), m) for x in xs for m in (-2, -1, 1, 2)
               if abs(x + m * X) < 23000)


def fig_summary():
    rows = json.load(open(os.path.join(CACHE, "item3.json")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    xs_n = np.array(I.NS)
    jit = {k: f for k, f in zip(I.TERRAINS, np.linspace(0.9, 1.1, len(I.TERRAINS)))}
    table = []
    for name in I.TERRAINS:
        R = [r for r in rows if r["terrain"] == name]
        for k in ("std", "joint"):
            med = [np.median([r["metrics"][k]["err"] for r in R if r["N"] == n]) for n in I.NS]
            lo = [np.min([r["metrics"][k]["err"] for r in R if r["N"] == n]) for n in I.NS]
            hi = [np.max([r["metrics"][k]["err"] for r in R if r["N"] == n]) for n in I.NS]
            a.errorbar(xs_n * jit[name], med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], color=COL[k],
                       marker=MK[name], ms=3.2, lw=0.8, elinewidth=0.6, capsize=0, alpha=0.9,
                       ls="-" if name in ("mountain", "mountains", "alpine") else "--")
        for r in R:
            gap_j = r["metrics"]["joint"]["amb"] - r["metrics"]["ref"]["amb"]
            gap_s = r["metrics"]["std"]["amb"] - r["metrics"]["ref"]["amb"]
            d = load(name, r["N"], r["seed"])
            mm = blind_min(name, d["xs"], float(d["X"]))
            table.append(dict(terrain=name, N=r["N"], seed=r["seed"], err_std=r["metrics"]["std"]["err"],
                              err_joint=r["metrics"]["joint"]["err"], amb_std=r["metrics"]["std"]["amb"],
                              amb_joint=r["metrics"]["joint"]["amb"], amb_ref=r["metrics"]["ref"]["amb"],
                              gap_joint=gap_j, gap_std=gap_s, mismatch=mm))
            b.plot(r["N"] * jit[name], gap_s, MK[name], ms=3, color=COL["std"], alpha=0.75, mew=0)
            b.plot(r["N"] * jit[name], gap_j, MK[name], ms=3.4, color=(AMBER if mm < 0.4 else COL["joint"]),
                   alpha=0.9, mew=0)
    a.set_xscale("log"); b.set_xscale("log")
    for ax in (a, b):
        ax.set_xticks(I.NS); ax.set_xticklabels([str(n) for n in I.NS]); ax.minorticks_off()
        ax.set_xlabel("number of point targets on the line")
    a.set_ylabel("image error vs reference [dB]"); a.set_ylim(-40, 8)
    a.set_title("(a) image error, median and range of 3 scenes", fontsize=9)
    b.set_ylabel("highest ambiguity − reference [dB]"); b.set_ylim(-3, 45)
    b.axhline(3, color=MUTED, ls="--", lw=0.7, zorder=0)
    b.text(0.985, 3.6, "3 dB", transform=b.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.8, color=MUTED,
           bbox=dict(fc="white", ec="none", pad=0.4), zorder=5)
    b.set_title("(b) highest ambiguity above the reference", fontsize=9)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL["std"], lw=1.2, label="standard"), Line2D([], [], color=COL["joint"], lw=1.2,
         label="joint"), Line2D([], [], color=AMBER, marker="o", ls="", ms=4, label="joint, near a blind height (mismatch < 0.4 rad)")]
    h += [Line2D([], [], color=INK, marker=MK[k], ls="-" if i < 3 else "--", lw=0.7, ms=3.5, label=TN[k])
          for i, k in enumerate(I.TERRAINS)]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=7.8, handlelength=1.8,
               columnspacing=1.0)
    fig.tight_layout(w_pad=1.5)
    save(fig, FIG, "dense_summary")
    json.dump(table, open(os.path.join(CACHE, "item3_table.json"), "w"), indent=1)
    return table


if __name__ == "__main__":
    fig_irf()
    T = fig_summary()
    for name in I.TERRAINS:
        for n in I.NS:
            R = [t for t in T if t["terrain"] == name and t["N"] == n]
            print(f"{name:11s} N={n:3d} err std {np.median([t['err_std'] for t in R]):5.1f} joint "
                  f"{np.median([t['err_joint'] for t in R]):6.1f} | gap joint "
                  + " ".join(f"{t['gap_joint']:5.1f}({t['mismatch']:.2f})" for t in R))
