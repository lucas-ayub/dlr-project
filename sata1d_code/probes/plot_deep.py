"""Figures of the deep analysis (Q1: sub-band vs whole band; Q2: saturation). Writes deep_figs/."""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(0, "probes")

OUT = "deep_figs"
INK, MUTED = "#222222", "#8a8a85"
C = dict(none="#C62828", whole="#1565C0", sub="#EF6C00", sub_fk="#7B1FA2", lin="#2E7D32", exact="#555555")
LAB = dict(none="no SATA", whole="SATA whole band (current)", sub="SATA per sub-band (current)",
           sub_fk="per sub-band, residual at the band centre $f_k$", lin="whole band, Doppler-dependent map (const + linear)",
           exact="exact correction")


def q1_inject():
    R = list(csv.DictReader(open("probes/deep_q1_doppler_residual.csv")))
    x = [float(r["phi_edge_deg"]) for r in R]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    for k, mk in (("none", "s"), ("whole", "o"), ("sub", "x"), ("sub_fk", "D"), ("lin", "^"), ("exact", "_")):
        ax.plot(x, [float(r[k]) for r in R], marker=mk, ms=8 if k != "sub" else 11, color=C[k], lw=1.6, label=LAB[k],
                ls="--" if k == "sub" else "-", mew=2 if k == "sub" else 1)
    ax.axhline(float(R[0]["ref"]), color=MUTED, lw=0.8, ls=":"); ax.text(1, float(R[0]["ref"]) + 0.8, "reference", fontsize=8.5, color=MUTED)
    ax.set_xlabel("injected Doppler-dependent residual: phase of channel 0 at the band edge ($f = \\pm617$ Hz) [deg]")
    ax.set_ylabel("highest ambiguity [dB re. reference peak]"); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="center right")
    ax.set_title("One target at h = 500 m: ambiguity after each correction", fontsize=10.5)
    ax = axs[1]
    for k, mk in (("none", "s"), ("whole", "o"), ("sub", "x"), ("sub_fk", "D"), ("lin", "^"), ("exact", "_")):
        ax.plot(x, [float(r[k + "_main"]) for r in R], marker=mk, ms=8 if k != "sub" else 11, color=C[k], lw=1.6,
                ls="--" if k == "sub" else "-", mew=2 if k == "sub" else 1)
    ax.set_xlabel("same, [deg]"); ax.set_ylabel("main-lobe peak of the target [dB re. reference peak]"); ax.grid(alpha=.3)
    ax.set_title("Main lobe", fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/q1_injected_residual.png", dpi=130); plt.close(fig)


def q2_single():
    R = list(csv.DictReader(open("probes/deep_q2_single.csv")))
    f = lambda k: np.array([float(r[k]) for r in R]); h = f("h") / 1e3
    fig, axs = plt.subplots(4, 1, figsize=(14, 14), sharex=True, gridspec_kw=dict(height_ratios=[0.9, 1.3, 1, 1.2]))
    ax = axs[0]
    for i, (col, b) in enumerate(zip(("#6A1B9A", "#1565C0", "#00897B", "#9E9D24"), (63.7, 27.0, 4.1, 1.7))):
        ax.plot(h, np.degrees(np.angle(np.exp(1j * np.radians(f(f"phi{i}"))))), ".", ms=3, color=col, label=f"channel {i}, $b_{{xt}}$ = {b} m")
    ax.set_ylabel("phase error $\\varphi_i$, wrapped [deg]"); ax.set_ylim(-190, 190); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=4, loc="lower left")
    ax.set_title("Phase error of each channel, $\\varphi_i = 2\\pi\\,\\Delta\\rho_i/\\lambda$ (one cycle per 1.09 / 2.59 / 17.1 / 42.4 km of height)", fontsize=10)
    ax = axs[1]
    ax.plot(h, f("amb1_ref_db"), color="#C62828", lw=1.8, label="measured, ambiguities at $\\pm$0.81 s")
    ax.plot(h, f("pred1_db"), "k--", lw=1, label="predicted $\\max(|c(1)|, |c(3)|)\\,\\beta_1$")
    ax.plot(h, f("amb2_ref_db"), color="#F9A825", lw=1.5, label="measured, ambiguities at $\\pm$1.62 s")
    ax.plot(h, f("pred2_db"), "k:", lw=1, label="predicted $|c(2)|\\,\\beta_2$")
    ax.axhline(20 * np.log10(0.595), color="k", lw=1.1); ax.text(5.95, 20 * np.log10(0.595) + 0.7, "bound $20\\log\\beta_1 = -4.5$ dB (all energy in one neighbouring band)", ha="right", fontsize=8.5)
    ax.axhline(20 * np.log10(0.5 * 0.595), color="0.4", lw=0.9, ls="-."); ax.text(5.95, 20 * np.log10(0.5 * 0.595) + 0.7, "random phases, rms: $-10.5$ dB", ha="right", fontsize=8.5, color="0.3")
    hl = np.logspace(-2, np.log10(0.6), 50); ax.plot(hl, 20 * np.log10(hl / 0.1) - 20.87, color="#C62828", lw=0.8, ls=":", alpha=.8)
    ax.text(0.62, -20, "small-error regime:\n20 dB per decade", fontsize=8.5, color="#C62828")
    ax.set_ylim(-60, 2); ax.set_ylabel("[dB re. reference peak]"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.set_title("Highest ambiguity relative to the REFERENCE peak (the metric of the sweeps): bounded", fontsize=10)
    ax = axs[2]
    ax.plot(h, f("main_db"), color="#2E7D32", lw=1.6, label="measured main-lobe peak")
    ax.plot(h, f("pred_main_db"), "k--", lw=1, label="predicted $|c(0)|$")
    ax.set_ylim(-42, 2); ax.set_ylabel("[dB re. reference peak]"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Main lobe of the target: what the ambiguities take, the main lobe loses ($\\sum_d |c(d)|^2 = 1$)", fontsize=10)
    ax = axs[3]
    ax.plot(h, f("amb1_own_db"), color="#C62828", lw=1.8, label="ambiguity relative to the target's OWN reconstructed peak")
    ax.axhline(0, color="k", lw=0.8); ax.text(0.05, 1.5, "0 dB: ambiguity as strong as the target", fontsize=8.5)
    ax.set_ylim(-45, 36); ax.set_ylabel("[dB re. own peak]"); ax.set_xlabel("target height h [km]"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("The same ambiguity relative to the target's own peak: unbounded (+33 dB at h = 4.2 km, where the main lobe is at $-38$ dB)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/q2_single_target.png", dpi=120); plt.close(fig)


def q2_sweep():
    import plot_v5_summary as P5
    rows = P5.load()
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    for ax, dk in zip(axs, P5.DIST):
        sel = [r for r in rows if r["_dk"] == dk]
        ramps = [r for r in sel if r["_pre"] == "RA"]; others = [r for r in sel if r["_pre"] != "RA"]
        ax.scatter([P5.predicted_db(r["_h"]) for r in ramps], [float(r["ambpk_db_none"]) for r in ramps], s=16, color="#1565C0", alpha=.8, label=f"ramps ({len(ramps)} cases)")
        ax.scatter([P5.predicted_db(r["_h"]) for r in others], [float(r["ambpk_db_none"]) for r in others], s=16, color="#EF6C00", alpha=.8, marker="^", label=f"random geometries ({len(others)} cases)")
        ax.plot([-70, 0], [-70, 0], color=INK, lw=1); ax.set_xlim(-70, 0); ax.set_ylim(-66, 0); ax.grid(alpha=.3)
        ax.set_xlabel("predicted from the channel phases, $\\max_j\\max_d |c_j(d)|\\,\\beta_{|d|}$ [dB]"); ax.set_title(P5.DIST[dk], fontsize=10.5)
        d = [float(r["ambpk_db_none"]) - P5.predicted_db(r["_h"]) for r in sel if float(r["ambpk_db_none"]) > -40]
        ax.text(0.03, 0.93, f"cases above $-40$ dB: |measured $-$ predicted| $\\leq$ {max(abs(x) for x in d):.2f} dB", transform=ax.transAxes, fontsize=8.5)
    axs[0].set_ylabel("simulated, no SATA [dB re. reference peak]"); axs[0].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("All 1053 cases of the sweep, no SATA: the ambiguity is fully explained by the four channel phases of each target", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{OUT}/q2_sweep_predicted.png", dpi=120); plt.close(fig)


def q2_plateau():
    import deep_q2_sata_plateau as PL
    G = list(csv.DictReader(open("probes/sata_gain.csv")))
    G = [r for r in G if r["step"] == "2600" and r["N"] == "5"]
    V = {r["slug"]: r for r in csv.DictReader(open("sweep_v4/results.csv")) if r["config"] == "RA_dS_hL" and r["N"] == "5"}
    fig, axs = plt.subplots(2, 2, figsize=(15, 9.6))
    for ax, a in zip(axs[0], (5, 20)):
        r, b = PL.case(a)
        cells, E, ap, nd = b["cells"], b["E"], b["applied_deg"], b["needed_deg"]
        err = np.degrees(np.angle(np.exp(1j * np.radians(ap - nd))))
        ax2 = ax.twinx()
        ax.fill_between(cells, 0, E / E.max(), color="#90CAF9", alpha=.7, label="energy of the target per bin (normalised)")
        ax2.plot(cells, err, ".", ms=4, color="#C62828", label="applied $-$ needed phase [deg]")
        for k in range(-2, 3):
            ax.axvline(k * 13, color=MUTED, lw=0.7, ls=":")
        ax.set_xlim(-60, 60); ax.set_ylim(0, 1.05); ax2.set_ylim(-190, 190)
        ax.set_xlabel("cell the bin points to, relative to the target [cells of 15.4 m]  (dotted: the 5 targets, 200 m apart)")
        ax.set_ylabel("energy per bin"); ax2.set_ylabel("phase error after the correction [deg]", color="#C62828")
        ax.set_title(f"channel 0, sub-aperture on the central target: $\\alpha$ = {a}$^\\circ$, step between neighbours "
                     f"{r['dh']:.0f} m = {abs(r['phi_step_deg']):.0f}$^\\circ$ per neighbour\n"
                     f"energy within 1 rad of the right phase: {100 * r['E_within_1rad']:.0f} %", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    ax = axs[1, 0]
    dh = [float(r["dh_neighbour"]) for r in G]
    for i, (col, b) in enumerate(zip(("#6A1B9A", "#1565C0", "#00897B", "#9E9D24"), (63.7, 27.0, 4.1, 1.7))):
        g = np.array([float(r[f"absg_ch{i}"]) for r in G]); d = np.array([float(r[f"dist_ch{i}"]) for r in G])
        ax.plot(dh, g, "o-", color=col, label=f"$|g_{i}|$, $b_{{xt}}$ = {b} m")
        ax.plot(dh, g ** 2 + d ** 2, "s:", color=col, ms=4, alpha=.7)
    ax.set_xscale("log"); ax.set_xlabel("height step between neighbours 200 m apart [m]"); ax.set_ylabel("$|g_i|$ (solid), energy kept (dotted)")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_title("After SATA, per channel: correlation with the ideally corrected signal $|g_i|$\nand fraction of the target's energy that survives the correction", fontsize=9.5)
    ax = axs[1, 1]
    meas = [float(V[f"N5_a{float(r['alpha']):g}"]["ambpk_db_whole"]) for r in G]
    ax.plot(dh, meas, "o-", color="#1565C0", label="SATA whole band, simulated (sweep)")
    ax.plot(dh, [float(r["pred_db"]) for r in G], "k--", label="predicted from the gains $g_i$")
    ax.plot(dh, [float(V[f"N5_a{float(r['alpha']):g}"]["ambpk_db_none"]) for r in G], "s-", color="#C62828", label="no SATA, simulated")
    ax.axhline(20 * np.log10(np.sqrt(2) / 4 * 0.595), color="0.4", ls="-.", lw=1)
    ax.text(dh[0], 20 * np.log10(np.sqrt(2) / 4 * 0.595) + 0.7, "two good channels out of four: $-13.5$ dB", fontsize=8.5, color="0.3")
    ax.set_xscale("log"); ax.set_xlabel("height step between neighbours 200 m apart [m]"); ax.set_ylabel("highest ambiguity [dB re. reference peak]")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8); ax.set_title("Ramp, N = 5, targets 200 m apart, large heights", fontsize=9.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/q2_sata_plateau.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    q1_inject(); print("q1")
    q2_single(); print("q2 single")
    q2_sweep(); print("q2 sweep")
    q2_plateau(); print("q2 plateau")
