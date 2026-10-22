"""Heatmaps of run_joint_sweep.py (highest ambiguity of every case x baseline, one page per method, plus joint minus
SATA whole band) and one PDF: heatmaps first, then the pages of every case (ramps, sines and cosines, random
terrains), one page per baseline. Run from sata1d_code after run_joint_sweep.py:
    python joint_cases/joint_sweep_summary.py            # -> joint_cases/joint_sweep.pdf
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import run_joint_sweep as W

METHODS = [("none", "no SATA"), ("whole", "SATA whole band (block map)"), ("sub", "SATA per sub-band (block map)"),
           ("joint_full", "joint, full system"), ("joint_occ", "joint, occupied parcels only")]
GROUP_NAME = dict(W.GROUPS)


def load(out):
    res = {}
    for p in glob.glob(os.path.join(out, "*", "*.json")):
        r = json.load(open(p))
        res[(r["case"], r["baseline"])] = r
    return res


def row_labels(cases, res):
    lab = {}
    for c in cases:
        r = next((res[(c, b)] for b, _, _ in W.BASELINES if (c, b) in res), None)
        lab[c] = f"{int(c[1:])}. " + (r["label"] if r else W.case_info(c)["label"])
    return lab


def col_labels():
    import textwrap
    return ["\n".join(textwrap.fill(p, 14) for p in k.split(", ", 1)) + "\n" + W.bxt_text(W.BXT[t]).replace(" m", "")
            for t, _, k in W.BASELINES]


def heatmap(ax, M, cmap, vmin, vmax, fmt, text_dark):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                v = M[i, j]
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7.5,
                        color="0.1" if text_dark(v) else "white")
    return im


def group_lines(ax, groups, ncols):
    """Separators and the group names on the right of the heatmap."""
    k0 = 0
    for g, n in groups:
        if k0:
            ax.axhline(k0 - 0.5, color="k", lw=1.6)
        ax.text(ncols - 0.4, k0 + (n - 1) / 2.0, GROUP_NAME[g], rotation=270, ha="left", va="center", fontsize=10,
                fontweight="bold", transform=ax.transData, clip_on=False)
        k0 += n


def page(cases, res, key, title, out_png, diff=None):
    labs = row_labels(cases, res)
    nb = len(W.BASELINES)
    M = np.full((len(cases), nb), np.nan)
    ref = np.full((len(cases), 1), np.nan)
    for i, c in enumerate(cases):
        for j, (b, _, _) in enumerate(W.BASELINES):
            r = res.get((c, b))
            if r:
                M[i, j] = r[key] - (r[diff] if diff else 0.0)
                ref[i, 0] = r["ref"]
    groups = []
    for c in cases:
        g = W.CASES[c][0]
        if groups and groups[-1][0] == g:
            groups[-1][1] += 1
        else:
            groups.append([g, 1])
    fig = plt.figure(figsize=(19, 0.34 * len(cases) + 3.4))
    if diff is None:
        gs = fig.add_gridspec(1, 3, width_ratios=[0.55, nb, 0.25], wspace=0.03, left=0.27, right=0.93, top=0.86, bottom=0.03)
        a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[0, 1]); cb = fig.add_subplot(gs[0, 2])
        cmap = plt.get_cmap("Reds")
        vmin, vmax = -70.0, 0.0
        dark = lambda v: (v - vmin) / (vmax - vmin) < 0.55
        heatmap(a0, ref, cmap, vmin, vmax, lambda v: f"{v:.1f}", dark)
        im = heatmap(a1, M, cmap, vmin, vmax, lambda v: f"{v:.1f}", dark)
        a0.set_xticks([0]); a0.set_xticklabels(["reference\n(no baselines)"], fontsize=8.5)
        a0.set_yticks(range(len(cases))); a0.set_yticklabels([labs[c] for c in cases], fontsize=8.5)
        a1.set_yticks([])
        for k0 in np.cumsum([n for _, n in groups])[:-1]:
            a0.axhline(k0 - 0.5, color="k", lw=1.6)
        cbar = fig.colorbar(im, cax=cb); cbar.set_label("highest ambiguity [dB re. the reference peak]  (lower is better)")
        cb.set_position([cb.get_position().x0 + 0.035, cb.get_position().y0, 0.012, cb.get_position().height])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[nb, 0.25], wspace=0.03, left=0.27, right=0.93, top=0.86, bottom=0.03)
        a1 = fig.add_subplot(gs[0, 0]); cb = fig.add_subplot(gs[0, 1])
        lim = 20.0
        im = heatmap(a1, np.clip(M, -lim, lim), plt.get_cmap("RdBu_r"), -lim, lim, lambda v: f"{v:+.1f}", lambda v: abs(v) < 0.6 * lim)
        for i in range(M.shape[0]):           # write the unclipped value
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]) and abs(M[i, j]) > lim:
                    for t in a1.texts:
                        if t.get_position() == (j, i):
                            t.set_text(f"{M[i, j]:+.1f}")
        a1.set_yticks(range(len(cases))); a1.set_yticklabels([labs[c] for c in cases], fontsize=8.5)
        cbar = fig.colorbar(im, cax=cb)
        cbar.set_label("difference [dB]   (blue: joint lower = better,  red: joint higher = worse)")
        cb.set_position([cb.get_position().x0 + 0.035, cb.get_position().y0, 0.012, cb.get_position().height])
    a1.set_xticks(range(nb)); a1.set_xticklabels(col_labels(), fontsize=8.5)
    for a_ in fig.axes[:-1]:
        a_.xaxis.tick_top()
    group_lines(a1, groups, nb)
    a1.set_title("cross-track baselines $b_{xt}$ [m] of channels 0, 1, 2, 3 (channel 0 is the transmitter: $b_{at}$ = $b_{xt}$ = 0)",
                 fontsize=10, pad=46)
    fig.suptitle(title, fontsize=13, y=0.985)
    fig.savefig(out_png, dpi=90, bbox_inches="tight"); plt.close(fig)
    return out_png


def bundle(pngs, out, outline):
    """Multi-page PDF (JPEG pages) with bookmarks: outline = [(title, page index, level)]."""
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in pngs]
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "joint_sweep_pages.pdf")   # outside the output folder
    ims[0].save(tmp, save_all=True, append_images=ims[1:], resolution=90, quality=75)
    try:
        from pypdf import PdfReader, PdfWriter
        w = PdfWriter(clone_from=PdfReader(tmp))
        parent = None
        for title, k, lvl in outline:
            it = w.add_outline_item(title, k, parent=parent if lvl else None)
            if lvl == 0:
                parent = it
        w.write(out)
        os.remove(tmp)
    except ImportError:
        import shutil
        shutil.move(tmp, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=W.OUT)
    ap.add_argument("--pdf", default=os.path.join(HERE, "joint_sweep.pdf"))
    a = ap.parse_args()
    res = load(a.out)
    cases = [c for c in W.CASES if any((c, b) in res for b, _, _ in W.BASELINES)]
    hm = os.path.join(a.out, "heatmaps"); os.makedirs(hm, exist_ok=True)
    pngs = [page(cases, res, k, f"Highest ambiguity: {name}", os.path.join(hm, f"{i + 1}_{k}.png"))
            for i, (k, name) in enumerate(METHODS)]
    pngs.append(page(cases, res, "joint_occ", "Joint (occupied parcels only) minus SATA whole band: highest ambiguity difference",
                     os.path.join(hm, "6_joint_occ_minus_whole.png"), diff="whole"))
    outline = [("Heatmaps", 0, 0)] + [(t, i, 1) for i, t in enumerate([n for _, n in METHODS] + ["joint (occupied) minus SATA whole band"])]
    labs = row_labels(cases, res)
    g_prev = None
    for c in cases:
        g = W.CASES[c][0]
        if g != g_prev:
            outline.append((GROUP_NAME[g], len(pngs), 0)); g_prev = g
        outline.append((labs[c].replace("$\\pi$", "\u03c0"), len(pngs), 1))
        pngs += [os.path.join(a.out, b, f"{c}.png") for b, _, _ in W.BASELINES if (c, b) in res]
    print("wrote", bundle(pngs, a.pdf, outline), len(pngs), "pages")


if __name__ == "__main__":
    main()
