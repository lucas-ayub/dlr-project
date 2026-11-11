"""Report figures of the real-terrain results (report/figures/realdem_*.pdf), from the results of run_real_dem.py.

    python real_dem_figures.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import run_real_dem as R

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid": True,
                     "grid.color": "0.88", "grid.linewidth": 0.6, "lines.linewidth": 1.0, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.03, "legend.framealpha": 0.9})
W_FULL = 6.3
C_REF, C_STD, C_JNT = "#2ca02c", "#d62728", "#9467bd"
TERRAIN_C, FILL_C = "#6D4C2F", "#D7C9A7"
OUT = os.path.normpath(R.FIG)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".pdf"))


def db(v, peak):
    return 20 * np.log10(np.maximum(np.abs(v), 1e-12) / peak)


def legend_above(ax, ncol=3):
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=ncol, frameon=False, borderaxespad=0.0,
              handlelength=1.6, columnspacing=1.2, fontsize=7.0)


# --------------------------------------------------------------------------------------------- map
def fig_map():
    dem = R.DEM()
    b = dem.bounds
    ls = LightSource(azdeg=315, altdeg=40)
    rgb = ls.shade(dem.h, cmap=plt.get_cmap("gist_earth"), vert_exag=0.05, blend_mode="soft",
                   vmin=0, vmax=4200)
    fig, ax = plt.subplots(figsize=(W_FULL * 0.62, W_FULL * 0.62 * (b.top - b.bottom) / (b.right - b.left)
                                    / np.cos(np.deg2rad(37.6))))
    ax.imshow(rgb, extent=[b.left, b.right, b.bottom, b.top], aspect=1 / np.cos(np.deg2rad(37.6)), interpolation="bilinear")
    for k, (key, L) in enumerate(R.LINES.items(), 1):
        _, _, la, lo = dem.line(L["lat"], L["lon"], L["az"], step=500.0)
        ax.plot(lo, la, color="#d62728", lw=1.8, solid_capstyle="butt")
        dx, dy = (0.06, 0.0) if L["az"] == 0 else (0.0, 0.05)
        ax.text(L["lon"] + dx, L["lat"] + dy, str(k), color="white", fontsize=8, weight="bold", ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="#d62728", ec="none"))
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.grid(False)
    sm = plt.cm.ScalarMappable(cmap="gist_earth", norm=plt.Normalize(0, 4200))
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("height [m]")
    save(fig, "realdem_map")


# --------------------------------------------------------------------------------------------- one line
def line_figure(tag, key, name):
    z = np.load(os.path.join(R.OUT, f"{tag}_{key}.npz"))
    with open(os.path.join(R.OUT, f"{tag}.json")) as fh:
        res = json.load(fh)[key]
    pos = z["pos"] / 1e3
    peak = float(z["peak"])
    lev = res["levels"]
    fig, axs = plt.subplots(3, 1, figsize=(W_FULL, 4.3), sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.2, 1.2], hspace=0.12))
    a = axs[0]
    x, h = z["x"] / 1e3, z["h"]
    a.fill_between(x, h.min() - 200, h, color=FILL_C, lw=0)
    a.plot(x, h, color=TERRAIN_C, lw=0.9, label="terrain (SRTM)")
    a.axhline(res["hmean"], color="0.35", ls="--", lw=0.8, label="reference height (mean of the line)")
    tg = np.array(res["targets"])
    a.plot(tg[:, 0] / 1e3, tg[:, 1] + res["hmean"], "v", color="k", ms=4, label="targets")
    a.set_ylim(h.min() - 200, h.max() + 300)
    a.set_ylabel("height [m]")
    legend_above(a)
    for ax, k, c, lab in ((axs[1], "std", C_STD, "no correction"), (axs[2], "joint", C_JNT, "joint reconstruction")):
        ax.plot(pos, db(z["ref"], peak), color=C_REF, lw=0.7, alpha=0.9, label=f"reference ({lev['ref']:.1f} dB)")
        ax.plot(pos, db(z[k], peak), color=c, lw=0.8, label=f"{lab}: strongest ghost {lev[k]:.1f} dB")
        ax.set_ylim(-80, 5)
        ax.set_ylabel("[dB]")
        ax.legend(loc="upper right", fontsize=6.6, ncol=2)
    axs[2].set_xlabel("azimuth position [km]")
    axs[2].set_xlim(-20, 20)
    save(fig, f"realdem_{tag}_{key}")


# --------------------------------------------------------------------------------------------- zoom on one target
def fig_zoom():
    dem = R.DEM()
    L = R.LINES["yosemite"]
    x, h, _, _ = dem.line(L["lat"], L["lon"], L["az"])
    r = R.run_line(x, h, R.TARGETS5, R.BXT_A, keep=True)
    tg = np.array(r["targets"])
    it = int(np.argmax(np.abs(tg[:, 1])))                     # the target farthest from the reference height
    t, ht = tg[it]
    up = 8
    Na = len(r["pos"])

    def upsample(s):
        S_ = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(s)))
        P = np.zeros(Na * up, complex)
        P[(Na * up - Na) // 2:(Na * up + Na) // 2] = S_
        return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(P))) * up

    d = r["pos"][1] - r["pos"][0]
    pos = (np.arange(Na * up) - Na * up // 2) * d / up
    foc = {k: upsample(v) for k, v in r["foc"].items()}
    peak = np.abs(foc["ref"]).max()
    X = r["X"]
    fig = plt.figure(figsize=(W_FULL, 2.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.75, 1.0], wspace=0.35)
    a = fig.add_subplot(gs[0])
    m = np.abs(pos - t) <= 30
    i0 = np.argmin(np.abs(pos - t))
    for k, c, lab, ls in (("ref", C_REF, "reference", "-"), ("joint", C_JNT, "joint", "--"), ("std", C_STD, "no correction", "-")):
        a.plot(pos[m] - t, db(foc[k][m], peak), color=c, ls=ls, label=lab, lw=1.2 if k == "ref" else 1.0)
    a.set_ylim(-45, 3)
    a.set_xlabel("distance from the target [m]")
    a.set_ylabel("[dB]")
    a.set_title("(a) the target: main lobe", fontsize=8)
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               fontsize=7.0)

    b = fig.add_subplot(gs[1], projection="polar")
    pr = foc["ref"][i0]
    for k, c, lab in (("ref", C_REF, "reference"), ("joint", C_JNT, "joint"), ("std", C_STD, "no correction")):
        v = foc[k][i0] / pr
        b.annotate("", xy=(np.angle(v), abs(v)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=1.4))
        print(f"zoom phasor {lab:14s} {20 * np.log10(abs(v)):6.2f} dB {np.degrees(np.angle(v)):7.2f} deg")
    b.set_rmax(1.1)
    b.set_rticks([0.5, 1.0])
    b.tick_params(labelsize=6.5)
    b.set_title("(b) the peak as a phasor\n(relative to the reference)", fontsize=8)

    c_ = fig.add_subplot(gs[2])
    m2 = np.abs(pos - (t + X)) <= 300
    for k, c, lab in (("ref", C_REF, "reference"), ("joint", C_JNT, "joint"), ("std", C_STD, "no correction")):
        c_.plot((pos[m2] - t) / 1e3, db(foc[k][m2], peak), color=c, label=lab)
    lv = {k: db(np.abs(foc[k][m2]).max(), peak) for k in foc}
    c_.text(0.03, 0.95, f"no correction {lv['std']:.1f} dB\njoint {lv['joint']:.1f} dB", transform=c_.transAxes,
            fontsize=6.8, va="top")
    c_.set_ylim(-85, 3)
    c_.set_xlabel("distance from the target [km]")
    c_.set_title(f"(c) its ghost at +X = {X / 1e3:.2f} km", fontsize=8)
    save(fig, "realdem_zoom")
    vs = foc["std"][i0] / pr
    vj = foc["joint"][i0] / pr
    return dict(h_rel=float(ht), std_db=float(20 * np.log10(abs(vs))), std_deg=float(np.degrees(np.angle(vs))),
                joint_db=float(20 * np.log10(abs(vj))), joint_deg=float(np.degrees(np.angle(vj))),
                ghost_std=float(lv["std"]), ghost_joint=float(lv["joint"]), X=float(X))


# --------------------------------------------------------------------------------------------- 20 targets: summary
def fig_bars20():
    with open(os.path.join(R.OUT, "lines20.json")) as fh:
        res = json.load(fh)
    keys = list(R.LINES)
    fig, ax = plt.subplots(figsize=(W_FULL * 0.75, 2.2))
    y = np.arange(len(keys))[::-1]
    std = [res[k]["levels"]["std"] for k in keys]
    jnt = [res[k]["levels"]["joint"] for k in keys]
    ref = np.median([res[k]["levels"]["ref"] for k in keys])
    ax.barh(y + 0.18, std, height=0.34, color=C_STD, label="no correction")
    ax.barh(y - 0.18, jnt, height=0.34, color=C_JNT, label="joint")
    for yy, s, j in zip(y, std, jnt):
        ax.text(s - 0.5, yy + 0.18, f"{s:.1f}", va="center", ha="right", fontsize=6.5)
        ax.text(j - 0.5, yy - 0.18, f"{j:.1f}", va="center", ha="right", fontsize=6.5)
    ax.axvline(ref, color=C_REF, ls="--", lw=1.0, label=f"reference ({ref:.1f} dB)")
    ax.set_yticks(y)
    ax.set_yticklabels([R.LINES[k]["name"] for k in keys])
    ax.set_xlim(-40, 0)
    ax.set_xlabel("strongest ghost [dB]")
    legend_above(ax)
    save(fig, "realdem_lines20_summary")


# --------------------------------------------------------------------------------------------- 200 random lines
def fig_cdf():
    a = np.loadtxt(os.path.join(R.OUT, "random_lines.csv"), delimiter=",")
    ref, std, jnt = a[:, 6], a[:, 7], a[:, 8]
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 2.4), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    for v, c, lab in ((std, C_STD, "no correction"), (jnt, C_JNT, "joint")):
        s = np.sort(v)
        ax.step(s, 100 * np.arange(1, len(s) + 1) / len(s), where="post", color=c, label=f"{lab} (median {np.median(v):.1f} dB)")
    ax.axvline(np.median(ref), color=C_REF, ls="--", lw=1.0, label=f"reference ({np.median(ref):.1f} dB)")
    ax.set_xlabel("strongest ghost [dB]")
    ax.set_ylabel("lines with a ghost below [%]")
    ax.set_xlim(-50, 0)
    ax.set_ylim(0, 102)
    legend_above(ax, ncol=1)
    ax = axs[1]
    rel = a[:, 4] - a[:, 3]
    for bm, mk in ((300, "o"), (135, "s")):
        m = np.isclose(a[:, 5], bm, atol=1)
        ax.scatter(rel[m], (std - jnt)[m], s=9, marker=mk, facecolor="none", edgecolor=C_JNT if bm == 300 else "0.35",
                   lw=0.7, label=rf"largest $b_{{xt}}$ = {bm} m")
    ax.set_xlabel("relief of the line (max - min) [m]")
    ax.set_ylabel("gain of the joint [dB]")
    legend_above(ax, ncol=2)
    save(fig, "realdem_random200")


def numbers():
    """The numbers quoted in the report text (written to results/summary.json)."""
    a = np.loadtxt(os.path.join(R.OUT, "random_lines.csv"), delimiter=",")
    ref, std, jnt = a[:, 6], a[:, 7], a[:, 8]
    rel = a[:, 4] - a[:, 3]
    gain = std - jnt
    m = rel > 200
    with open(os.path.join(R.OUT, "lines5.json")) as fh:
        l5 = json.load(fh)
    with open(os.path.join(R.OUT, "lines20.json")) as fh:
        l20 = json.load(fh)
    return dict(n=len(a), med_ref=float(np.median(ref)), med_std=float(np.median(std)), med_joint=float(np.median(jnt)),
                med_gain=float(np.median(gain)), relief_max=float(rel.max()), n_relief200=int(m.sum()),
                gain_p10_relief200=float(np.percentile(gain[m], 10)), frac_gain28_relief200=float(np.mean(gain[m] >= 28)),
                within10=int(np.sum(jnt - ref <= 10)), worse=int(np.sum(jnt > std)),
                lines5={k: v["levels"] | dict(hmin=v["hmin"], hmax=v["hmax"]) for k, v in l5.items()},
                lines20={k: v["levels"] for k, v in l20.items()})


def main():
    fig_map()
    for key in ("yosemite", "escarpment", "highsierra"):
        line_figure("lines5", key, R.LINES[key]["name"])
    line_figure("lines20", "yosemite", R.LINES["yosemite"]["name"])
    fig_bars20()
    fig_cdf()
    s = numbers()
    s["zoom"] = fig_zoom()
    with open(os.path.join(R.OUT, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
