"""
Summary pages of sweep_v4 on physical axes.
  collapse_<set>_<d>.png : no SATA vs highest target (with the single-target prediction),
                           SATA whole / per sub-band and improvement vs the largest neighbour step (200 m)
                           or vs the highest target (2.6 and 10 km)
  sub_vs_whole_<set>.png : SATA per sub-band vs SATA whole band
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "report_v5"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "<"]
FAMS = {"RH": "random heights", "PW": "piecewise ramp", "ZI": "zig-zag", "ST": "steepening", "FL": "flattening",
        "CL": "cliff", "SK": "spike", "ME": "mesa"}
HS = {"hS": "small heights ($h_j = j\\cdot20\\,\\mathrm{m}\\cdot\\tan\\alpha$)",
      "hN": "normal heights ($h_j = j\\cdot200\\,\\mathrm{m}\\cdot\\tan\\alpha$)",
      "hL": "large heights ($h_j = j\\cdot2600\\,\\mathrm{m}\\cdot\\tan\\alpha$)"}
DIST = {"dS": "targets 200 m apart", "dN": "targets 2.6 km apart", "dL": "targets 10 km apart"}
B1, B2 = 0.595, 0.190
INK, MUTED = "#222222", "#8a8a85"
LVL = "highest ambiguity [dB re. reference peak]"
FOOT = ("Each point is one case. Highest ambiguity of the image, in dB relative to the reference peak. "
        "$N_{rx}$ = 4, PRF = 2000 Hz, $b_{xt}$ = [63.7, 27.0, 4.1, 1.7] m.")

def build_phase_table(path="probes/phase_table.npz", h_min=-8000.0, h_max=45000.0):
    import sata1d as S
    import run_geometry_sweep as G
    hs = np.concatenate((-np.logspace(-1, np.log10(-h_min), 150)[::-1], [0.0], np.logspace(-1, np.log10(h_max), 260)))
    phi = np.zeros((len(hs), 4))
    for k, h in enumerate(hs):
        cfg = S.make_config([(0.0, h)], bxt_max=G.BXT_MAX, seed=G.BXT_SEED); tr = S.build_platform_tracks(cfg)
        phi[k] = [2 * np.pi * S.residual_C0(cfg, tr, cfg.scene.points[1], i) / cfg.system.wl for i in range(4)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, h=hs, phi=phi)


if not os.path.exists("probes/phase_table.npz"):
    build_phase_table()
_tab = np.load("probes/phase_table.npz")


def phases(h):
    return np.array([np.interp(h, _tab["h"], _tab["phi"][:, i]) for i in range(4)])


def single_level(h):
    e = np.exp(1j * phases(h))
    c = [abs(np.mean(e * np.exp(-2j * np.pi * d * np.arange(4) / 4))) for d in range(4)]
    return max(max(c[1], c[3]) * B1, c[2] * B2)


def predicted_db(hs):
    return 20 * np.log10(max(single_level(h) for h in hs) + 1e-12)


def load():
    rows = list(csv.DictReader(open("sweep_v4/results.csv")))
    for r in rows:
        h = np.array([float(x) for x in r["heights"].split()])
        r["_hmax"] = max(np.abs(h).max(), 0.1)
        r["_step"] = max(np.abs(np.diff(h)).max(), 0.1)
        r["_h"] = h
        pre, dk, hk = r["config"].split("_")
        r["_pre"], r["_dk"], r["_hk"] = pre, dk, hk
    return rows


def style(ax, xl, yl, title, log=True):
    if log:
        ax.set_xscale("log")
    ax.grid(alpha=0.25, which="both", lw=0.6)
    ax.set_xlabel(xl, fontsize=10, color=INK); ax.set_ylabel(yl, fontsize=10, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def groups(rows, mode):
    if mode == "ramp":
        return [(hk, HS[hk], PAL[i], MARK[i], [r for r in rows if r["_hk"] == hk]) for i, hk in enumerate(HS)]
    return [(p, FAMS[p], PAL[i], MARK[i], [r for r in rows if r["_pre"] == p]) for i, p in enumerate(FAMS)]


ZORDER = {PAL[0]: 6, PAL[1]: 5, PAL[2]: 4}


def dots(ax, x, y, col, mk, label=None):
    ax.scatter(x, y, s=26, marker=mk, facecolor=col, edgecolor="white", linewidth=0.6, alpha=0.9, label=label,
               zorder=ZORDER.get(col, 3))


def legend_and_foot(fig, ax, ncol):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=ncol, fontsize=9, frameon=False)
    fig.text(0.5, 0.008, FOOT, ha="center", va="bottom", fontsize=8.8, color="0.35")


def collapse(rows, mode, dk, out_png, head):
    sel = [r for r in rows if r["_dk"] == dk]
    close = dk == "dS"
    hx = "highest target [m]"
    sx = "largest height difference between neighbouring targets [m]"
    xk, xl = ("_step", sx) if close else ("_hmax", hx)
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    hgrid = np.logspace(-1, np.log10(45000), 400)
    pred = [20 * np.log10(single_level(h)) for h in hgrid]
    panels = [(axs[0, 0], "_hmax", hx, "none", "no SATA"),
              (axs[0, 1], xk, xl, "whole", "SATA whole band"),
              (axs[1, 0], xk, xl, "sub", "SATA per sub-band"),
              (axs[1, 1], None, None, "scatter", "no SATA vs SATA whole band")]
    for ax, xkey, xlab, k, lab in panels:
        if k == "scatter":
            for _, name, col, mk, rr in groups(sel, mode):
                if rr:
                    dots(ax, [float(r["ambpk_db_none"]) for r in rr], [float(r["ambpk_db_whole"]) for r in rr], col, mk)
            for g in (0, 10, 20, 30, 40):
                ax.plot([-66, 0], [-66 - g, -g], color=MUTED, lw=0.8, ls="-" if g == 0 else "--", zorder=1)
                if g:
                    ax.text(-62 + g, -63.5, f"{g} dB", fontsize=8.5, color=MUTED, ha="center")
            ax.text(-64, -57, "dashed: improvement of\n10, 20, 30, 40 dB", fontsize=8.5, color=MUTED)
            ax.set_xlim(-66, 0); ax.set_ylim(-66, 0)
            style(ax, "no SATA [dB]", "SATA whole band [dB]", lab, log=False)
            continue
        for _, name, col, mk, rr in groups(sel, mode):
            if rr:
                dots(ax, [r[xkey] for r in rr], [float(r[f"ambpk_db_{k}"]) for r in rr], col, mk, name)
        ax.set_ylim(-66, 0)
        style(ax, xlab, LVL, lab)
    legend_and_foot(fig, axs[0, 0], 2 if mode == "ramp" else 3)
    fig.suptitle(f"{head}, {DIST[dk]}", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.16 if mode != "ramp" else 0.13, 1, 0.96))
    fig.savefig(out_png, dpi=120); plt.close(fig)


def sub_vs_whole(rows, mode, out_png, head):
    fig, axs = plt.subplots(1, 3, figsize=(16, 6.6), sharey=True)
    for ax, dk in zip(axs, DIST):
        sel = [r for r in rows if r["_dk"] == dk]
        for _, name, col, mk, rr in groups(sel, mode):
            if rr:
                dots(ax, [float(r["ambpk_db_whole"]) for r in rr], [float(r["ambpk_db_sub"]) for r in rr], col, mk, name)
        d = [abs(float(r["ambpk_db_sub"]) - float(r["ambpk_db_whole"])) for r in sel]
        ax.plot([-66, 0], [-66, 0], color=INK, lw=1)
        ax.set_xlim(-66, 0); ax.set_ylim(-66, 0)
        style(ax, "SATA whole band [dB]", "SATA per sub-band [dB]",
              f"{DIST[dk]} | largest difference {max(d):.2f} dB", log=False)
    legend_and_foot(fig, axs[0], 3)
    fig.suptitle(f"{head}: SATA per sub-band vs SATA whole band, highest ambiguity of every case", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.2 if mode != "ramp" else 0.15, 1, 0.93))
    fig.savefig(out_png, dpi=120); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    sets = {"ramps": ("ramp", [r for r in rows if r["_pre"] == "RA"], "Along-track ramps (all cases)"),
            "random": ("fam", [r for r in rows if r["_pre"] in FAMS], "Random geometries (all cases)")}
    for name, (mode, rr, head) in sets.items():
        for dk in DIST:
            collapse(rr, mode, dk, os.path.join(OUT, f"collapse_{name}_{dk}.png"), head)
        sub_vs_whole(rr, mode, os.path.join(OUT, f"sub_vs_whole_{name}.png"), head.split(" (")[0])
        print(name, len(rr))
