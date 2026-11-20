"""Figures of item 1 (from cache/height_*.npz):
  fig/height_irf.(pdf|png)    focused lines for 5 mountain heights: standard vs reference | joint vs reference
  fig/height_amb.(pdf|png)    highest ambiguity, weakest true peak and 3-dB width vs mountain height
    python3 -I plot_item1.py"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from style import plt, COL, NAME, INK, MUTED, TERRAIN, FILL, AMBER, B300, db, mismatch, save
from common import CACHE, FIG, relative, gauss_mountain

SHOW = [0, 500, 1000, 2000, 3000]


def load(H):
    return dict(np.load(os.path.join(CACHE, f"height_{int(H):04d}.npz"), allow_pickle=True))


def blind(d):
    ter, _ = relative(gauss_mountain(float(d["H"])))
    X = float(d["X"])
    best = (9.0, None)
    for x in d["xs"]:
        for m in (-2, -1, 1, 2):
            mm = mismatch(B300, float(ter(x + m * X) - ter(x)), m)
            if mm < best[0]:
                best = (mm, (float(x), m))
    return best


def fig_irf():
    fig = plt.figure(figsize=(7.2, 1.25 * len(SHOW) + 1.3))
    gs = fig.add_gridspec(len(SHOW) + 1, 2, height_ratios=[0.9] + [1] * len(SHOW))
    top = fig.add_subplot(gs[0, :])
    shades = plt.cm.copper(np.linspace(0.15, 0.85, len(SHOW)))
    for H, c in zip(SHOW, shades):
        d = load(H)
        top.plot(d["tgrid"] / 1e3, d["hgrid"] + float(d["href"]), color=c, lw=1.0)
        top.plot(d["xs"] / 1e3, d["hs"] + float(d["href"]), "o", ms=3, color=c)
        top.plot([], [], color=c, lw=1.4, label=f"{H} m")
    top.legend(loc="upper left", ncol=len(SHOW), fontsize=7.8, handlelength=1.2, columnspacing=1.0, title="mountain height",
               title_fontsize=7.8)
    top.set_xlim(-20, 20); top.set_ylim(-150, 4300); top.set_ylabel("[m]")
    top.set_title("the scene: Gaussian mountain ($\\sigma$ = 1.5 km), five targets on the surface", fontsize=9)
    axs = np.array([[fig.add_subplot(gs[r + 1, c]) for c in range(2)] for r in range(len(SHOW))])
    for r, H in enumerate(SHOW):
        d = load(H)
        m = json.loads(str(d["metrics"]))
        x = d["t"] * float(d["vs"]) / 1e3
        pk = np.abs(d["ref"]).max()
        for c, k in enumerate(("std", "joint")):
            a = axs[r, c]
            a.plot(x, db(d["ref"], pk), color=COL["ref"], lw=0.6)
            a.plot(x, db(d[k], pk), color=COL[k], lw=0.6, alpha=0.9)
            a.set_ylim(-70, 4); a.set_xlim(-20, 20); a.set_yticks([-60, -30, 0])
            a.text(0.99, 0.97, ("highest ambiguity " + f"{m[k]['amb']:.1f}".replace("-", "\u2212") + " dB"), transform=a.transAxes, ha="right",
                   va="top", fontsize=8.2, color=INK, bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
            if c == 0:
                a.set_ylabel(f"{H} m\n[dB]")
            else:
                a.set_yticklabels([])
            if r < len(SHOW) - 1:
                a.set_xticklabels([])
        if r == 0:
            axs[r, 0].set_title("standard reconstruction (red) vs reference (green)", fontsize=9)
            axs[r, 1].set_title("joint reconstruction (purple) vs reference (green)", fontsize=9)
    for a in axs[-1]:
        a.set_xlabel("position along the line [km]")
    fig.text(0.01, 0.0, "rows: mountain height; 0 dB = reference peak", fontsize=8.2, color=MUTED, va="top")
    fig.tight_layout(h_pad=0.15, w_pad=0.8)
    save(fig, FIG, "height_irf")


def fig_amb():
    files = sorted(glob.glob(os.path.join(CACHE, "height_*.npz")))
    D = [dict(np.load(p, allow_pickle=True)) for p in files]
    H = np.array([float(d["H"]) for d in D])
    M = [json.loads(str(d["metrics"])) for d in D]
    bl = [blind(d) for d in D]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for k in ("std", "joint", "ref"):
        a.plot(H, [m[k]["amb"] for m in M], "o-", ms=2.6, color=COL[k], label=NAME[k])
    for h, (mm, w), m in zip(H, bl, M):
        if m["joint"]["amb"] - m["ref"]["amb"] > 5:
            dx = {2500: -420, 2600: 0, 2900: 120}.get(int(h), 0)
            a.annotate(f"{mm:.2f} rad", xy=(h, m["joint"]["amb"]), xytext=(h + dx, -19 if int(h) == 2600 else -24),
                       fontsize=7.6, color=AMBER, ha="center", va="bottom",
                       bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9),
                       arrowprops=dict(arrowstyle="-", color=AMBER, lw=0.6, shrinkA=0, shrinkB=2))
    a.set_xlabel("mountain height [m]"); a.set_ylabel("highest ambiguity [dB]")
    a.set_ylim(-45, 0)
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=8.2, handlelength=1.4)
    a.set_title("(a) highest ambiguity (amber: blind-height mismatch)", fontsize=9)
    for k in ("std", "joint"):
        b.plot(H, [m[k]["peak_loss"] for m in M], "o-", ms=2.6, color=COL[k])
    b.set_xlabel("mountain height [m]"); b.set_ylabel("weakest target peak [dB]")
    b.set_ylim(-30, 2)
    b.set_title("(b) weakest of the five target peaks", fontsize=9)
    w = [float(np.max(m["joint"]["w3db"])) for m in M] + [float(np.max(m["ref"]["w3db"])) for m in M]
    b.text(0.03, 0.03, f"3-dB width of every target:\njoint = reference = {np.median(w):.2f} m\nat every height",
           transform=b.transAxes, ha="left", va="bottom", fontsize=8, color=INK)
    fig.tight_layout(w_pad=1.2)
    save(fig, FIG, "height_amb")
    rows = []
    for h, d, m, (mm, w) in zip(H, D, M, bl):
        rows.append(dict(H=h, href=float(d["href"]), mismatch=mm, worst_pair=w,
                         **{f"{k}_{q}": (m[k][q] if q != "w3db" else float(np.mean(m[k][q])))
                            for k in ("ref", "std", "joint") for q in ("amb", "amb_old", "floor", "peak_loss", "w3db")},
                         w3db_joint_max=float(np.max(m["joint"]["w3db"])), w3db_ref_max=float(np.max(m["ref"]["w3db"]))))
    json.dump(rows, open(os.path.join(CACHE, "item1_table.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    fig_irf()
    rows = fig_amb()
    for r in rows:
        print(f"{r['H']:5.0f}  std {r['std_amb']:6.1f}  joint {r['joint_amb']:6.1f}  ref {r['ref_amb']:6.1f}  "
              f"mism {r['mismatch']:.2f} {r['worst_pair']}  w3 joint {r['w3db_joint_max']:.2f}")
