"""Figure for probe_whole_vs_sub.py (numbers copied from its printed output, 2026-09-28)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_whole_vs_sub")
C = {"no SATA": "#B2182B", "SATA whole band": "#185FA5", "SATA per sub-band": "#1D9E75"}
scen = ["1 target\nh = 240 m", "ramp 5×200 m\n20°", "2 targets\n2.6 km apart", "2 targets\nX = 6.23 km apart"]
img = {"no SATA": [-12.8, -11.3, -12.5, -14.7], "SATA whole band": [-66.2, -36.3, -64.3, -24.4],
       "SATA per sub-band": [-66.2, -36.3, -62.6, -24.3]}
noi = {"no SATA": [-12.8, -11.3, -12.5, -14.7], "SATA whole band": [-18.4, -16.8, -18.3, -17.1],
       "SATA per sub-band": [-12.9, -11.5, -12.6, -14.7]}
cov = {"with alias images": ([99.9] * 4, [99.9] * 4), "direct image only": ([39.0] * 4, [10.7, 39.0, 39.0, 11.1])}

fig = plt.figure(figsize=(13, 8.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.55, wspace=0.22)
for j, (title, d) in enumerate((("(a) map WITH alias images (current code)", img),
                                ("(b) map with the direct image only (DEM-like)", noi))):
    ax = fig.add_subplot(gs[0, j]); x = np.arange(len(scen)); w = 0.26
    for i, (k, v) in enumerate(d.items()):
        b = ax.bar(x + (i - 1) * w, np.array(v) - 0, w - 0.03, color=C[k], label=k)
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, vi - 1.2, f"{vi:.0f}", ha="center", va="top", fontsize=7.5, color="#333")
    ax.set_xticks(x, scen, fontsize=8.5); ax.set_ylim(-75, 0); ax.invert_yaxis() if False else None
    ax.set_title(title, fontsize=10.5, loc="left"); ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
    ax.set_ylabel("topographic ambiguity [dB re. flat peak]\n(more negative = better)", fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if j == 0: fig.legend(*ax.get_legend_handles_labels(), fontsize=9, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.955))
    ax.text(0.99, 0.02, "oracle ≈ −100 dB in all cases", transform=ax.transAxes, ha="right", fontsize=8, color="#555")

ax = fig.add_subplot(gs[1, :])
labels, vals, cols = [], [], []
for mode, (wv, sv) in cov.items():
    labels.append("whole\nband"); vals.append(wv[0]); cols.append(C["SATA whole band"])
    for k in range(4):
        labels.append(f"set\nk={k}"); vals.append(sv[k]); cols.append(C["SATA per sub-band"])
xx = np.arange(len(labels)) + np.repeat([0, 0.8], 5)
ax.bar(xx, vals, 0.7, color=cols)
for xi, v in zip(xx, vals):
    ax.text(xi, v + 1.5, f"{v:.1f} %", ha="center", fontsize=8)
ax.set_xticks(xx, labels, fontsize=8.5); ax.set_ylim(0, 112)
for g, mode in enumerate(cov):
    ax.text(xx[5 * g:5 * g + 5].mean(), -24, mode, ha="center", fontsize=9.5, fontweight="bold")
ax.set_ylabel("target energy corrected\nwith its true ΔC0 [%]", fontsize=9)
ax.set_title("(c) T2 — bin-by-bin read of the kernels (1 target): which fraction of the target's energy receives the right ΔC0",
             fontsize=10.5, loc="left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
fig.suptitle("SATA whole band vs per sub-band (C0 term): why the numbers coincide", fontsize=12.5, x=0.01, ha="left")
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "whole_vs_sub_tests.png"), dpi=140, bbox_inches="tight")
print("saved")
