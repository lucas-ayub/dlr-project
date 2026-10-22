"""PASS / FAIL and the PDF of verify_sata_recon.py (sata_recon/checks.json, sata_recon/sata_recon.pdf)."""
import glob
import json
import os

import numpy as np

from verify_sata_recon import OUT, SCENES, NRX, BMAX

CHECKS = []


def check(part, name, value, limit, larger=False, unit="dB"):
    ok = bool(value >= limit) if larger else bool(value <= limit)
    CHECKS.append(dict(part=part, name=name, value=float(value), limit=float(limit), larger=larger, unit=unit, ok=ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {part} {name}: {value:.4g} {unit} ({'>=' if larger else '<='} {limit:g})", flush=True)


def load():
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "runs", "*.json")))]
    order = {s: i for i, s in enumerate(SCENES)}
    return sorted(R, key=lambda r: (order[r["scene"]], r["nrx"], r["bmax"]))


EXACT = ("flat500", "periodic")          # scenes where theory says SATA before the reconstruction is exact
MUT_CHECKED = ("ruler_sign", "whole_labels_in_sub", "sub_windows_reversed", "phase_sign", "dem_moved_1km")
MUT_INFO = ("dem_moved_31m", "dem_moved_123m")   # shifts of 0.1 and 0.4 SATA cell (310 m): below the resolution of the method


def checks(zero, cells):
    R = load()
    print(len(R), "runs")
    # ---- M: the model of this script (M1, M2 share raw() with the package: sanity only)
    check("M", "sum of the per-target echoes - generate_channels (worst run; same raw(): sanity only)", max(r["model_raw_vs_generate"] for r in R), -250)
    check("M", "ideal model with a zero DEM - standard reconstruction (worst run)", max(r["model_flat_vs_none"] for r in R), -250)
    check("M", "oracle of this script - oracle_channels of the package (worst run)", max(r["model_oracle_vs_package_oracle"] for r in R), -90)
    check("M", "height table of this script - dC0 of the targets (worst run)", max(r["table_vs_targets_mm"] for r in R), 0.01, unit="mm")
    # ---- C: cells of the targets in every channel (what the DEM table indexes)
    check("C", "cell of a target in channel i: formula of az_pixel_of_scatterer - closest approach of the bistatic history (worst)",
          max(abs(c["formula_err_cells"]) for c in cells), 0.01, unit="cells")
    check("C", "az_pixel_of_scatterer after rounding to an integer cell (worst)", max(abs(c["err_cells"]) for c in cells), 0.5 + 1e-6, unit="cells")
    check("C", "x that terrain_lookup_table evaluates at that cell - x of the target (worst)", max(abs(c["table_x_err_m"]) for c in cells), 1e-6, unit="m")
    # ---- Z: zero DEM
    for z in zero:
        check("Z", f"zero DEM, Nrx = {z['nrx']}: SATA whole band - standard reconstruction", z["whole"], -120)
        check("Z", f"zero DEM, Nrx = {z['nrx']}: SATA per sub-band - standard reconstruction", z["sub"], -120)
    # ---- E: exact scenes
    for sc in EXACT:
        rr = [r for r in R if r["scene"] == sc]
        for k in ("whole", "sub"):
            check("E", f"{sc} ({len(rr)} runs): |ideal SATA {k} - oracle| (worst)", max(abs(r["levels"]["ideal_" + k] - r["levels"]["oracle"]) for r in rr), 0.1)
        rs = [r for r in rr if r["max_rad_per_sata_cell"] <= 0.6]
        for k in ("whole", "sub"):
            check("E", f"{sc} ({len(rs)} runs with <= 0.6 rad per SATA cell): |package SATA {k} - oracle| (worst)",
                  max(abs(r["levels"]["pkg_" + k] - r["levels"]["oracle"]) for r in rs), 1.0)
            check("E", f"{sc} (same runs): package - ideal SATA {k} inside the ambiguity windows (worst)",
                  max(r["amb_diff"][f"pkg_vs_ideal_{k}"] for r in rs), -45.0)
        for r in rr:
            if r["max_rad_per_sata_cell"] > 0.6:
                print(f"  (info) {sc} N{r['nrx']} b{r['bmax']:g}, {r['max_rad_per_sata_cell']:.2f} rad per SATA cell: package "
                      f"{r['levels']['pkg_whole']:.1f} / {r['levels']['pkg_sub']:.1f} dB, oracle {r['levels']['oracle']:.1f} dB (resolution of the sub-apertures)")
    # ---- P: package against the ideal SATA, complex difference inside the ambiguity windows, against what the method leaves
    smooth = [r for r in R if r["max_rad_per_sata_cell"] <= 0.5 and r["scene"] not in EXACT]
    for k in ("whole", "sub"):
        margin = min(r["amb_diff"][f"ideal_vs_oracle_{k}"] - r["amb_diff"][f"pkg_vs_ideal_{k}"] for r in smooth)
        check("P", f"<= 0.5 rad per SATA cell ({len(smooth)} runs), SATA {k}: (ideal - oracle) minus (package - ideal), inside the ambiguity windows (smallest)",
              -margin, -15.0)
        check("P", f"same runs, SATA {k}: |package - ideal| highest ambiguity (worst)", max(abs(r["levels"]["pkg_" + k] - r["levels"]["ideal_" + k]) for r in smooth), 1.0)
    # ---- X: the package broken on purpose must be seen
    for r in R:
        for mu, m in r.get("mutations", {}).items():
            if mu in MUT_INFO and r["scene"] != "periodic":
                print(f"  (info) {r['scene']} N{r['nrx']} b{r['bmax']:g}, {mu} (below one SATA cell): "
                      f"{m['diff_whole'] - r['amb_diff']['pkg_vs_ideal_whole']:+.1f} / {m['diff_sub'] - r['amb_diff']['pkg_vs_ideal_sub']:+.1f} dB")
            if mu not in MUT_CHECKED:
                continue
            for k in ("whole", "sub"):
                if k == "whole" and mu in ("whole_labels_in_sub", "sub_windows_reversed"):
                    continue
                if r["scene"] == "periodic" and mu != "phase_sign":
                    continue                      # the fold-periodic terrain is blind to positions one fold apart by construction
                check("X", f"{r['scene']} N{r['nrx']} b{r['bmax']:g}, {mu}: (package broken - ideal) minus (package - ideal), SATA {k}",
                      m[f"diff_{k}"] - r["amb_diff"][f"pkg_vs_ideal_{k}"], 10.0, larger=True)
    # ---- S: per-channel correction, even with the positions of the targets (footprint maps, not allowed in the method).
    # Largest complex difference to the oracle over the whole image (peaks included): two targets one fold apart (pairX)
    # against the same two targets half a fold apart (pairX2), same ramp, same Nrx and baselines.
    near = {(r["nrx"], r["bmax"]): r for r in R if r["scene"] == "pairX2"}
    for r in R:
        if r["scene"] == "pairX" and (r["nrx"], r["bmax"]) in near:
            q = near[(r["nrx"], r["bmax"])]
            for k in ("whole", "sub"):
                check("S", f"N{r['nrx']} b{r['bmax']:g}, SATA {k} WITH the target positions: |IRF - oracle| max, one fold apart "
                      f"({r['full_diff']['footprint_' + k]:.1f} dB) minus half a fold apart ({q['full_diff']['footprint_' + k]:.1f} dB)",
                      r["full_diff"]["footprint_" + k] - q["full_diff"]["footprint_" + k], 10.0, larger=True)
    n = sum(c["ok"] for c in CHECKS)
    print(f"{n} PASS, {len(CHECKS) - n} FAIL")
    json.dump(dict(checks=CHECKS, zero=zero, cells=cells), open(os.path.join(OUT, "checks.json"), "w"), indent=1)


# ================================================================================================ PDF
def pdf():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    R = load()
    REF, NON, ORA, WHO, SUB = "tab:green", "tab:red", "k", "tab:blue", "tab:orange"
    names = list(SCENES)
    out = os.path.join(OUT, "sata_recon.pdf")
    with PdfPages(out) as P:
        # ---- 0) the checks (written by --checks)
        C = json.load(open(os.path.join(OUT, "checks.json")))["checks"] if os.path.exists(os.path.join(OUT, "checks.json")) else []
        head = {"M": "M  the model of this script against the package", "C": "C  azimuth cell of every place (closest approach, bistatic)",
                "Z": "Z  zero DEM: SATA must be the identity", "E": "E  exact scenes: SATA = oracle",
                "P": "P  the package against the ideal SATA (method)", "X": "X  the package broken on purpose must be seen",
                "S": "S  per-channel correction WITH the target positions, targets one fold apart"}
        per = 46
        for p0 in range(0, max(len(C), 1), per):
            fig = plt.figure(figsize=(15, 9.5)); fig.text(0.03, 0.965, f"Checks: {sum(c['ok'] for c in C)} PASS, {sum(not c['ok'] for c in C)} FAIL "
                                                          f"(page {p0 // per + 1})", fontsize=13, weight="bold")
            yy, last = 0.93, None
            for c in C[p0:p0 + per]:
                if c["part"] != last:
                    fig.text(0.03, yy, head.get(c["part"], c["part"]), fontsize=9.5, weight="bold"); yy -= 0.019; last = c["part"]
                fig.text(0.04, yy, ("PASS" if c["ok"] else "FAIL"), fontsize=7.6, color="tab:green" if c["ok"] else "tab:red", weight="bold", family="monospace")
                fig.text(0.075, yy, f"{c['name']}: {c['value']:.4g} {c.get('unit', '')} ({'>=' if c.get('larger') else '<='} {c['limit']:g})"[:205], fontsize=7.4)
                yy -= 0.0172
            P.savefig(fig); plt.close(fig)
        # ---- 1) the implementation against the method
        fig, axs = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw=dict(wspace=0.22, left=0.06, right=0.88, top=0.86, bottom=0.11))
        fig.suptitle("Is the package doing what SATA does? Inside the ambiguity windows: |package - ideal SATA| against |ideal SATA - oracle| "
                     "(what the method itself leaves), every run", fontsize=12)
        g = np.array([r["max_rad_per_sata_cell"] for r in R])
        for ax, key, name in ((axs[0], "whole", "SATA whole band"), (axs[1], "sub", "SATA per sub-band")):
            x = np.array([r["amb_diff"][f"ideal_vs_oracle_{key}"] for r in R]); y = np.array([r["amb_diff"][f"pkg_vs_ideal_{key}"] for r in R])
            sc = ax.scatter(np.maximum(x, -95), y, c=np.clip(g, 0, 3), cmap="viridis", vmin=0, vmax=3, s=36, edgecolor="k", lw=0.4)
            ax.plot([-95, 5], [-95, 5], color="0.4", lw=1, ls="--"); ax.plot([-95, 5], [-110, -10], color="0.4", lw=0.8, ls=":")
            ax.text(-20, -42, "15 dB below", fontsize=8.5, color="0.35", rotation=33)
            ax.set_xlim(-97, 5); ax.set_ylim(-97, 5)
            ax.set_xlabel(f"|ideal - oracle| [dB re. the reference peak]  (exact scenes: below -95, drawn at -95)"); ax.set_ylabel("|package - ideal| [dB re. the reference peak]")
            ax.set_title(name, fontsize=11)
        cb = fig.colorbar(sc, cax=fig.add_axes([0.9, 0.11, 0.015, 0.75]))
        cb.set_label("phase change of the DEM correction over one SATA cell of 310 m [rad]")
        P.savefig(fig); plt.close(fig)
        # ---- 2) the method against the oracle, per scene
        fig, axs = plt.subplots(2, 1, figsize=(15, 9.6), sharex=True, gridspec_kw=dict(hspace=0.14, left=0.06, right=0.985, top=0.91, bottom=0.12))
        fig.suptitle("What SATA before the reconstruction leaves, with the exact DEM: highest ambiguity of every run "
                     "(5 numbers of channels x 2 baseline sets per scene)", fontsize=12)
        k = np.arange(len(R))
        for ax, key, name in ((axs[0], "whole", "SATA whole band"), (axs[1], "sub", "SATA per sub-band")):
            ax.plot(k, [r["levels"]["ref"] for r in R], "o", color=REF, ms=6, ls="none", label="Reference")
            ax.plot(k, [r["levels"]["none"] for r in R], "x", color=NON, ms=6, mew=1.6, ls="none", label="no SATA")
            ax.plot(k, [r["levels"]["oracle"] for r in R], "_", color=ORA, ms=11, mew=2, ls="none", label="oracle (each target with its own dC0)")
            ax.plot(k, [r["levels"]["ideal_" + key] for r in R], "s", color=WHO if key == "whole" else SUB, mfc="none", ms=8, ls="none",
                    label=f"ideal {name} (infinitely fine sub-apertures)")
            ax.plot(k, [r["levels"]["pkg_" + key] for r in R], ".", color=WHO if key == "whole" else SUB, ms=8, ls="none", label=f"{name}, package")
            ax.set_ylabel("highest ambiguity [dB]"); ax.set_ylim(-65, 8); ax.legend(fontsize=8.5, loc="lower right", ncol=3)
            for i in range(1, len(R)):
                if R[i]["scene"] != R[i - 1]["scene"]:
                    ax.axvline(i - 0.5, color="0.5", lw=0.8)
        for s in names:
            idx = [i for i, r in enumerate(R) if r["scene"] == s]
            if idx:
                axs[0].text(np.mean(idx), 3, s, ha="center", fontsize=8.5)
        axs[1].set_xticks(k); axs[1].set_xticklabels([f"{r['nrx']}/{r['bmax']:g}" for r in R], rotation=90, fontsize=6.5)
        axs[1].set_xlabel("run: number of channels / largest cross-track baseline [m]")
        P.savefig(fig); plt.close(fig)
        # ---- 3) package - ideal against the phase gradient, and the broken versions
        fig, axs = plt.subplots(1, 2, figsize=(15, 6.0), gridspec_kw=dict(wspace=0.25, left=0.06, right=0.87, top=0.86, bottom=0.22))
        fig.suptitle("Where the package departs from the ideal SATA, and what a wrong implementation would look like (inside the ambiguity windows)", fontsize=12)
        ax = axs[0]
        for key, c, name in (("whole", WHO, "SATA whole band"), ("sub", SUB, "SATA per sub-band")):
            ax.semilogx(np.maximum(g, 1e-3), [r["amb_diff"][f"pkg_vs_ideal_{key}"] for r in R], "o", color=c, ms=5, mfc="none" if key == "sub" else c, label=name)
        ax.set_xlabel("phase change of the DEM correction over one SATA cell of 310 m [rad]")
        ax.set_ylabel("|package - ideal| [dB re. the reference peak]"); ax.legend(fontsize=9)
        ax.set_ylim(-100, 0); ax.text(1.1e-3, -97, "constant DEM: below the axis", fontsize=8.5)
        ax.set_title("Difference to the ideal: resolution of the sub-apertures", fontsize=10.5)
        ax = axs[1]
        mu = [r for r in R if "mutations" in r]
        names_m = ["ruler_sign", "whole_labels_in_sub", "sub_windows_reversed", "phase_sign", "dem_moved_31m", "dem_moved_123m", "dem_moved_1km"]
        labels = ["package\nas it is", "ruler with the\nsign flipped", "whole-band labels\nin the sub-bands", "sub-band windows\nin reverse order",
                  "phase with the\nsign flipped", "DEM moved\nby 31 m", "DEM moved\nby 123 m", "DEM moved\nby 1 km"]
        x = np.arange(len(labels))
        mk = ["o", "s", "^", "D", "v"]
        for j, r in enumerate(mu):
            for key, c, off in (("whole", WHO, -0.15), ("sub", SUB, 0.15)):
                vals = [r["amb_diff"][f"pkg_vs_ideal_{key}"]] + [r["mutations"][n][f"diff_{key}"] if not (key == "whole" and n in ("whole_labels_in_sub", "sub_windows_reversed")) else np.nan for n in names_m]
                ax.plot(x + off + (j - 2) * 0.03, vals, mk[j % 5], color=c, ms=5, mfc="none" if key == "sub" else c, ls="none",
                        label=f"{r['scene']} N{r['nrx']} b{r['bmax']:g}" if key == "whole" else None)
        ax.axvspan(-0.5, 0.5, color="0.92", zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
        ax.set_ylabel("|variant - ideal| [dB re. the reference peak]"); ax.legend(fontsize=7.8, loc="upper left", bbox_to_anchor=(1.01, 1), title="filled: whole band\nopen: per sub-band", title_fontsize=7.8)
        ax.set_title("The same comparison with the package broken on purpose", fontsize=10.5)
        P.savefig(fig); plt.close(fig)
        # ---- 4) when is SATA before the reconstruction enough: ramps, phase difference between places one fold apart
        from verify_sata_recon import fold
        lam_r_tan = 0.25 * 766205.9 * np.tan(np.radians(20.0))
        fig, ax = plt.subplots(figsize=(15, 6.2), gridspec_kw=dict(left=0.06, right=0.985, top=0.86, bottom=0.11))
        fig.suptitle("Ramps: what SATA before the reconstruction leaves, against the phase difference that the DEM puts between two places one fold apart",
                     fontsize=12)
        for key, c, name in (("ideal_whole", WHO, "ideal SATA whole band"), ("ideal_sub", SUB, "ideal SATA per sub-band"), ("none", NON, "no SATA")):
            xs, ys = [], []
            for r in R:
                if r["scene"].startswith("ramp"):
                    a = {"ramp1": 1.0, "ramp3": 3.0, "ramp8": 8.0}[r["scene"]]
                    xs.append(2 * np.pi * max(r["bxt"]) * fold(r["nrx"]) * np.tan(np.radians(a)) / lam_r_tan)
                    ys.append(r["levels"][key] - r["levels"]["oracle"])
            ax.semilogx(xs, ys, "o" if key != "none" else "x", color=c, ms=6, mfc="none" if key == "ideal_sub" else c, ls="none", label=name)
        xx = np.logspace(-2, 2, 50)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("2 pi b$_{xt,max}$ $\\Delta h_X$ / ($\\lambda$ r$_0$ tan$\\theta$) [rad]  ($\\Delta h_X$: height difference between x and x + X, X = one fold)")
        ax.set_ylabel("highest ambiguity minus that of the oracle [dB]"); ax.legend(fontsize=9, loc="upper left")
        ax.set_xlim(0.3, 100)
        P.savefig(fig); plt.close(fig)
        # ---- 5) IRF pages
        for sc, n, b in (("flat500", 4, 500.0), ("periodic", 3, 500.0), ("ramp1", 4, 100.0), ("ramp3", 4, 300.0), ("mountain", 4, 300.0),
                         ("mountain1", 4, 100.0), ("hills25", 4, 100.0), ("hills25", 8, 500.0), ("pairX", 4, 100.0), ("pairX2", 4, 100.0)):
            f = os.path.join(OUT, "runs", f"{sc}_N{n}_b{b:g}")
            if not os.path.exists(f + ".npz"):
                continue
            z, r = np.load(f + ".npz"), json.load(open(f + ".json"))
            L = r["levels"]
            fig, axs = plt.subplots(2, 1, figsize=(15, 9.4), sharex=True, gridspec_kw=dict(hspace=0.22, left=0.06, right=0.985, top=0.9, bottom=0.07))
            fig.suptitle(f"{r['desc']} | {r['n_targets']} target(s) | N$_{{rx}}$ = {n}, b$_{{xt}}$ = {r['bxt']} m | DEM table "
                         f"(terrain_lookup_table) | {r['max_rad_per_sata_cell']:.2f} rad per SATA cell", fontsize=11)
            for ax, key, name, c in ((axs[0], "whole", "SATA whole band", WHO), (axs[1], "sub", "SATA per sub-band", SUB)):
                ax.plot(z["t"], z["none"], color=NON, lw=0.8, alpha=0.7, zorder=1, label=f"no SATA ({L['none']:.1f} dB)")
                ax.plot(z["t"], z["pkg_" + key], color=c, lw=0.9, zorder=2, label=f"{name}, package ({L['pkg_' + key]:.1f} dB)")
                ax.plot(z["t"], z["ideal_" + key], color="tab:purple", lw=0.9, ls=(0, (2, 2)), zorder=3, label=f"ideal {name} ({L['ideal_' + key]:.1f} dB)")
                ax.plot(z["t"], z["oracle"], color=ORA, lw=0.8, zorder=4, label=f"oracle ({L['oracle']:.1f} dB)")
                ax.plot(z["t"], z["ref"], color=REF, lw=1.6, zorder=5, label=f"Reference ({L['ref']:.1f} dB)")
                from verify_sata_recon import targets
                xt = targets(r["scene"], r["nrx"]) / 7688.53706432
                ax.set_xlim(xt.min() - 1.8, xt.max() + 1.8)
                ax.set_ylim(-90, 5); ax.set_ylabel("[dB]"); ax.legend(fontsize=8.5, loc="upper right", ncol=2)
                ax.set_title(f"{name}: highest ambiguity in brackets", fontsize=10.5)
            axs[1].set_xlabel("Azimuth time [s]")
            P.savefig(fig); plt.close(fig)
    print("wrote", out)
