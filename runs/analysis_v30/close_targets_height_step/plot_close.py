"""Figures and tables of close_targets_height_step (run item_close.py first).
    python3 -I plot_close.py                    talk style  -> fig/close_*.pdf|png
    FIG_STYLE=report python3 -I plot_close.py   report style (serif)
Outputs: fig/close_maps (highest ambiguity, d x dh, cliff and bump, standard vs joint), fig/close_spread (joint
highest ambiguity against the height spread of the terrain under a target), fig/close_irf (three example lines),
cache/close_table.json|md."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from style import plt, COL, NAME, INK, MUTED, TERRAIN, FILL, AMBER, db, save
from subsample import terrain_abs, DX, dh_sample, phase
from close_common import local_profile

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, FIG = os.path.join(HERE, "cache"), os.path.join(HERE, "fig")
R = json.load(open(os.path.join(CACHE, "close.json")))
DS = [6.0, 12.0, 18.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]
DH = [0.0, 10.0, 50.0, 100.0, 300.0, 1000.0]
CELL = 5.77                                    # 3-dB width of the main lobe [m] (joint_height_dense)
REPORT = os.environ.get("FIG_STYLE") == "report"
if REPORT:
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})


def spread(r, half=2 * CELL):
    """Height spread of the terrain under a target: max over targets of max |h(x_k) - h(x_t)| over the line samples
    x_k within +-half of the target (one DEM height per sample, dx = 3.84 m)."""
    h = terrain_abs(r["case"], r["dh"], 0.5 * DX, r["w"])
    out = 0.0
    for x in r["xs"]:
        ks = np.arange(np.ceil((x - half) / DX), np.floor((x + half) / DX) + 1)
        out = max(out, float(np.max(np.abs(h(ks * DX) - h(x)))))
    return out


def get(case, d, dh, w=10.0):
    for r in R:
        if r["case"] == case and r["d"] == d and r["dh"] == dh and r["w"] == w:
            return r


# ---------------------------------------------------------------- table
rows = []
for r in R:
    rows.append(dict(case=r["case"], w=r["w"], d=r["d"], dh=r["dh"], spread=spread(r), jump=dh_sample(r),
                     mismatch=r["mismatch"],
                     **{f"amb_{k}": r[k]["amb"] for k in ("ref", "std", "joint")},
                     pk_err_joint=float(np.max(np.abs(np.subtract(r["joint"]["pk"], r["ref"]["pk"])))),
                     pos_err_joint=float(np.nanmax(np.abs(r["joint"]["pos"]))) if np.isfinite(r["joint"]["pos"]).any() else None,
                     dip_ref=r["ref"]["dip"][0], dip_joint=r["joint"]["dip"][0],
                     lerr_std=r["std"]["lerr"], lerr_joint=r["joint"]["lerr"]))
json.dump(rows, open(os.path.join(CACHE, "close_table.json"), "w"), indent=1)
with open(os.path.join(CACHE, "close_table.md"), "w") as f:
    f.write("| case | w [m] | d [m] | dh [m] | jump [m] | amb ref | amb std | amb joint | peak err joint [dB] | "
            "dip ref | dip joint | local err std | local err joint |\n|" + "---|" * 13 + "\n")
    for q in rows:
        f.write(f"| {q['case']} | {q['w']:g} | {q['d']:g} | {q['dh']:g} | {q['jump']:.1f} | {q['amb_ref']:.1f} | "
                f"{q['amb_std']:.1f} | {q['amb_joint']:.1f} | {q['pk_err_joint']:.2f} | {q['dip_ref']:.1f} | "
                f"{q['dip_joint']:.1f} | {q['lerr_std']:.1f} | {q['lerr_joint']:.1f} |\n")

# ---------------------------------------------------------------- maps
def maps(cases, name, figsize):
    fig, ax = plt.subplots(len(cases), 2, figsize=figsize, squeeze=False, sharex=True, sharey=True)
    for i, (case, title) in enumerate(cases):
        for j, k in enumerate(("std", "joint")):
            a = ax[i][j]
            M = np.array([[get(case, d, dh)[k]["amb"] for d in DS] for dh in DH])
            im = a.imshow(M, origin="lower", cmap="Reds", vmin=-62, vmax=0, aspect="auto")
            for p in range(len(DH)):
                for q in range(len(DS)):
                    v = M[p, q]
                    a.text(q, p, f"{v:.0f}", ha="center", va="center", fontsize=8.8 if REPORT else 8.4,
                           color="white" if v > -25 else INK)
            a.set_xticks(range(len(DS))); a.set_xticklabels([f"{d:g}" for d in DS])
            a.set_yticks(range(len(DH))); a.set_yticklabels([f"{h:g}" for h in DH])
            a.grid(False)
            a.set_title(f"{title}: {NAME[k]}" if len(cases) > 1 else NAME[k], fontsize=plt.rcParams["axes.titlesize"])
            if i == len(cases) - 1:
                a.set_xlabel("distance between the two targets d [m]")
            if j == 0:
                a.set_ylabel("height change $\\Delta h$ [m]")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("highest ambiguity [dB]")
    save(fig, FIG, name)


maps([("step", "cliff"), ("bump", "bump")], "close_maps", (7.0, 4.6) if REPORT else (8.2, 5.2))
if not REPORT:
    maps([("step", "cliff")], "close_maps_step", (8.2, 2.7))

# ---------------------------------------------------------------- joint against the height jump under a target
fig, a = plt.subplots(figsize=(5.6, 3.3) if REPORT else (6.0, 3.3))
MK = {"step": ("o", "cliff, w = 10 m"), "width": ("s", "cliff, w = 1 m or 50 m"), "bump": ("^", "bump, 10 m wide"),
      "three": ("D", "three targets, middle one on the cliff")}
jp = np.array([dh_sample(r) for r in R])
for c, (m, lab) in MK.items():
    sel = [i for i, r in enumerate(R) if r["case"] == c]
    a.scatter(np.maximum(jp[sel], 0.1), [R[i]["joint"]["amb"] for i in sel], marker=m, s=17, facecolor="none",
              edgecolor=COL["joint"], lw=0.9, label=f"joint: {lab}")
a.axhline(np.median([r["ref"]["amb"] for r in R]), color=COL["ref"], lw=1.2, label="reference (median)")
a.axhline(-40, color=MUTED, lw=0.7, ls="--")
a.set_xscale("log"); a.set_xlim(0.07, 600); a.set_ylim(-65, 2)
a.set_xticks([0.1, 1, 10, 100]); a.set_xticklabels(["$\\leq$0.1", "1", "10", "100"])
a.set_xlabel("height change between the two DEM samples around a target, $\\delta h$ [m]")
a.set_ylabel("highest ambiguity [dB]")
t = a.secondary_xaxis("top", functions=(phase, lambda p: p / phase(1.0)))
t.set_xlabel("phase on the 300 m baseline, $2\\pi b_{xt}\\,\\delta h/(\\lambda r_0\\tan\\theta)$ [rad]")
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=plt.rcParams["legend.fontsize"] - 0.3, ncol=2,
         handletextpad=0.3, columnspacing=0.9)
save(fig, FIG, "close_jump")

# ---------------------------------------------------------------- example lines
EX = [("step", 50.0, 300.0, "d = 50 m, $\\Delta h$ = 300 m"), ("step", 1000.0, 300.0, "d = 1000 m, $\\Delta h$ = 300 m"),
      ("step", 6.0, 1000.0, "d = 6 m, $\\Delta h$ = 1000 m: targets on the cliff face")]
fig, ax = plt.subplots(3, 3, figsize=(7.0, 5.0) if REPORT else (8.6, 5.6),
                       gridspec_kw=dict(height_ratios=[0.8, 1.3, 1.3], hspace=0.6, wspace=0.28))
for j, (case, d, dh, title) in enumerate(EX):
    z = np.load(os.path.join(CACHE, f"close_{case}_w10_d{d:g}_h{dh:g}.npz"), allow_pickle=True)
    m = json.loads(str(z["metrics"]))
    x = z["t"] * float(z["vs"])
    pk = np.abs(z["ref"]).max()
    xs = np.array(m["xs"])
    lo, hi = xs.min() - max(25.0, 0.4 * d), xs.max() + max(25.0, 0.4 * d)
    a = ax[0][j]
    xc = np.linspace(lo, hi, 800)
    hc = terrain_abs(case, dh, 0.5 * DX, 10.0)(xc) - m["href"]
    a.fill_between(xc, hc, hc.min() - 0.08 * dh - 1, color=FILL, lw=0); a.plot(xc, hc, color=TERRAIN, lw=1.1)
    a.plot(xs, np.array(m["hs"]), "o", color=INK, ms=3.5)
    a.set_xlim(lo, hi); a.set_title(title, fontsize=plt.rcParams["axes.titlesize"])
    a.set_ylabel("h [m]" if j == 0 else "")
    rr = dict(t=z["t"].astype(float), vs=float(z["vs"]), ref=z["ref"], std=z["std"], joint=z["joint"])
    for row, (l0, l1, xl) in ((1, (lo, hi, "position [m]")), (2, (-15000, 15000, "position [km]"))):
        a = ax[row][j]
        sel = (x >= l0) & (x <= l1)
        for k in ("ref", "std", "joint"):
            if row == 1:
                xu, su = local_profile(rr, k, lo, hi)
                q = (xu >= lo) & (xu <= hi)
                a.plot(xu[q], 20 * np.log10(su[q] + 1e-12), color=COL[k], lw=1.1, label=NAME[k], alpha=0.95)
            else:
                a.plot(x[sel] / 1000.0, db(z[k][sel], pk), color=COL[k], lw=0.7, label=NAME[k], alpha=0.9)
        a.set_xlim(l0 / (1000.0 if row == 2 else 1.0), l1 / (1000.0 if row == 2 else 1.0))
        a.set_ylim(-70, 3)
        a.set_xlabel(xl)
        a.set_ylabel("[dB]" if j == 0 else "")
fig.legend(*ax[1][0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
save(fig, FIG, "close_irf")
print("ok")
