"""Figures of item 5 (cache/item5.json, cache/item5_lines_*.npz from item5_regularisation.py).
    python3 -I plot_item5.py        -> fig/item5_summary, fig/item5_lcurve, fig/item5_lines; cache/item5_table.md"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common_m as M
import common_r as RC
import item5_regularisation as I

COL = dict(standard=M.RED, current=M.PURPLE, lcurve="#0072B2", lcurve2="#56B4E9", discrepancy="#009E73", bmr="#E69F00",
           maria="#000000", wise="#CC79A7")
LAB = dict(standard="standard", current="joint now (Tikhonov, $\\mu=0.03\\sigma_{max}$, 2 passes)",
           lcurve="L-curve corner (RelSea + Menger)", lcurve2="L-curve corner, 2 passes",
           discrepancy="discrepancy principle", bmr="statistical (BMR, $\\mu^2=N_0/S_0$)",
           maria="MARIA (ML-inspired, adaptive)", wise="WISE (WCF, adaptive)")
METH = ["standard", "current", "lcurve", "lcurve2", "discrepancy", "bmr", "maria", "wise"]
SCN = dict(mountain="1000 m mountain, 5 targets", dense40="mountain line, 40 targets",
           foothills20="SRTM western foothills, 20 targets", clutter1500="mountain line, 1500 scatterers")
SNRS = ["inf", "30", "20", "10"]


def ref_levels():
    p = os.path.join(M.CACHE, "item5_ref.json")
    if os.path.exists(p):
        return json.load(open(p))
    out = {}
    for k, (xs, ter, amps) in I.scenes().items():
        if k == "clutter1500":
            continue
        r = I.setup(xs, ter, amps)[-1]
        out[k] = I.metrics(dict(r, refcopy=r["ref"]), "refcopy")
    json.dump(out, open(p, "w"), indent=1)
    return out


def med(res, sc, snr, m, q):
    v = [res[k][m][q] for k in res if k.startswith(f"{sc}|{snr}|") and m in res[k] and q in res[k][m]]
    return float(np.median(v)) if v else np.nan


def summary(res, ref):
    plt = M.style()
    REP = os.environ.get("FIG_STYLE") == "report"
    if REP:
        fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
        axs = axs.ravel()
    else:
        fig, axs = plt.subplots(1, 4, figsize=(12.6, 3.3))
    x = np.arange(4)
    for ax, sc in zip(axs, SCN):
        q = "nmse" if sc == "clutter1500" else "amb"
        for m in METH:
            if m == "standard" and q == "nmse":
                continue
            y = [med(res, sc, s, m, q) for s in SNRS]
            ax.plot(x, y, color=COL[m], marker="o", ms=3.5, lw=1.2 if m not in ("current", "maria") else 1.9, label=LAB[m])
        if sc in ref:
            ax.axhline(ref[sc]["amb"], color=M.GREEN, lw=1.0, label="reference (no noise)")
        ax.set_xticks(x)
        ax.set_xticklabels(["no noise", "30 dB", "20 dB", "10 dB"])
        ax.set_xlabel("SNR of the channels")
        ax.set_title(SCN[sc], fontsize=9.5 if REP else 8.4)
        ax.set_ylabel("error of the focused line [dB]" if q == "nmse" else "highest ambiguity [dB]")
        if q == "amb":
            ax.set_ylim(-45, 0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4 if REP else 5, fontsize=8.6 if REP else 7.2,
               bbox_to_anchor=(0.5, 0.0 if REP else -0.09))
    fig.tight_layout(rect=(0, 0.12 if REP else 0.08, 1, 1))
    M.save(fig, "item5_summary")
    plt.close(fig)


def lcurve(res):
    plt = M.style()
    REP = os.environ.get("FIG_STYLE") == "report"
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4) if REP else (7.6, 3.0))
    for ax, key in zip(axs, ["mountain|inf|0", "mountain|20|0"]):
        e = res[key]["lcurve"]
        cv = np.array(e["curve"])
        o = np.argsort(cv[:, 0])
        ax.plot(cv[o, 1] / np.log(10) / 2, cv[o, 2] / np.log(10) / 2, color="#0072B2", marker=".", ms=4, lw=0.8)
        sm = e["sigma_max"]
        for mu, col, lab in ((e["mu"], "#0072B2", "L-curve corner"), (0.03 * sm, M.PURPLE, "joint now, $0.03\\sigma_{max}$")):
            j = int(np.argmin(np.abs(np.log(cv[:, 0]) - np.log(mu))))
            ax.plot(cv[j, 1] / np.log(10) / 2, cv[j, 2] / np.log(10) / 2, "o", color=col, ms=7, mfc="none", mew=1.6,
                    label=f"{lab}: $\\mu/\\sigma_{{max}}$ = {mu / sm:.1e}")
        a, b = e["region_rel"]
        ax.set_title(("no noise" if "inf" in key else "SNR 20 dB") + f" - RelSea region $\\mu/\\sigma_{{max}}\\in$ [{a:.0e}, {b:.0e}]",
                     fontsize=9 if REP else 8)
        ax.set_xlabel("$\\log_{10}\\|A g - s\\|$")
        ax.set_ylabel("$\\log_{10}\\|g\\|$")
        ax.legend(fontsize=8.2 if REP else 7, loc="lower left")
    fig.tight_layout()
    M.save(fig, "item5_lcurve")
    plt.close(fig)


def lines():
    plt = M.style()
    d = np.load(os.path.join(M.CACHE, "item5_lines_mountain_20_0.npz"))
    t, X, xs = d["t"], float(d["X"]), d["xs"]
    v = I.M.S.make_config([(0.0, 0.0)]).system.vs
    x = t * v / 1e3
    fig, axs = plt.subplots(3, 1, figsize=(8.0, 5.0), sharex=True)
    for ax, m in zip(axs, ["current", "lcurve", "maria"]):
        ax.plot(x, d["ref"], color=M.GREEN, lw=0.7, label="reference (no noise)")
        ax.plot(x, d[m], color=COL[m], lw=0.7, label=LAB[m] + ", SNR 20 dB")
        for p in xs:
            for k in (-2, -1, 1, 2):
                ax.axvline((p + k * X) / 1e3, color=M.AMBER, lw=0.5, ls=":")
        ax.set_ylim(-70, 3)
        ax.set_ylabel("[dB]")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    axs[-1].set_xlim(-20, 20)
    axs[-1].set_xlabel("position along the line [km]   (dotted: ambiguity positions $x_p \\pm X, \\pm 2X$)")
    fig.tight_layout()
    M.save(fig, "item5_lines")
    plt.close(fig)


def table(res, ref):
    L = ["| scene | SNR | " + " | ".join(METH) + " | reference |", "|---|---|" + "---|" * (len(METH) + 1)]
    for sc in SCN:
        q = "nmse" if sc == "clutter1500" else "amb"
        for s in SNRS:
            row = [f"{med(res, sc, s, m, q):.1f}" if not np.isnan(med(res, sc, s, m, q)) else "-" for m in METH]
            L.append(f"| {sc} ({q}) | {s} | " + " | ".join(row) + f" | {ref.get(sc, {}).get('amb', float('nan')):.1f} |")
    L.append("")
    L.append("| scene, no noise | " + " | ".join(METH[1:]) + " |")
    L.append("|---|" + "---|" * (len(METH) - 1))
    for sc in SCN:
        k = f"{sc}|inf|0"
        L.append(f"| {sc}: mu/sigma_max | " + " | ".join(f"{res[k][m].get('mu_rel', np.nan):.2g}" if m in res[k] else "-" for m in METH[1:]) + " |")
        L.append(f"| {sc}: A + A^H calls | " + " | ".join(str(res[k][m]['calls_A'] + res[k][m]['calls_AH']) if m in res[k] else "-" for m in METH[1:]) + " |")
        L.append(f"| {sc}: time [s] | " + " | ".join(f"{res[k][m]['sec']:.1f}" if m in res[k] else "-" for m in METH[1:]) + " |")
    L.append("")
    L.append("| scene, SNR 20 dB (seed 0) | " + " | ".join(METH) + " |")
    L.append("|---|" + "---|" * len(METH))
    for sc in SCN:
        k = f"{sc}|20|0"
        for q in ("floor", "peak_loss", "nmse"):
            if q != "nmse" and sc == "clutter1500":
                continue
            L.append(f"| {sc}: {q} | " + " | ".join(f"{res[k][m][q]:.1f}" if m in res[k] and q in res[k][m] else "-" for m in METH) + " |")
    s = "\n".join(L) + "\n"
    open(os.path.join(M.CACHE, "item5_table.md"), "w").write(s)
    return s


if __name__ == "__main__":
    res = json.load(open(os.path.join(M.CACHE, "item5.json")))
    ref = ref_levels()
    summary(res, ref)
    lcurve(res)
    lines()
    print(ref)
    print(table(res, ref))
