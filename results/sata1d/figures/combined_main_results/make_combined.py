"""
Combined PDF of the main SATA 1D results with the block map with copies (footprint):
along-track ramps, random geometries and the sine / cosine terrains.

  heatmaps      : drawn here from sweep_csv/sweep_distance_height_results.csv (ramps, random geometries)
                  and from the results*.json of the terrain sweeps (sine / cosine terrains)
  summary pages : existing figures (summary_*.png, subband_vs_wholeband_*.png, summary_vs_n_targets_footprint.png)
  case pages    : cases/ (run_sweep_lut_cases.py --map footprint) and the terrain sweep folders

    python make_combined.py            # writes ../combined_main_results.pdf
"""
import csv
import glob
import json
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(HERE)                                   # results/sata1d/figures
CSV = os.path.join(FIG, "..", "sweep_csv", "sweep_distance_height_results.csv")
TERRAIN = [("sweep_terrain_footprint_realistic", ["B100-A100-P50", "B50-A50-P30", "B80-A80-P20", "B60-A60-P8", "B40-A40-P5"]),
           ("sweep_terrain_footprint_complete", ["B500-A100-P50", "B300-A50-P30", "B400-A80-P20", "B250-A60-P8", "B200-A40-P5"])]
COUNTS = [1, 2, 3, 4, 5, 6, 7, 9, 11, 15, 31]

CMAP = LinearSegmentedColormap.from_list("improv", ["#B2182B", "#E8A48C", "#E4E4E4", "#8FB8DB", "#1F5FA8", "#0B2E5C"])
LEVEL = dict(cmap="viridis_r", vmin=-65, vmax=-5, dark=lambda v: v > -32, cb="highest ambiguity [dB]")
IMPROV = dict(cmap=CMAP, vmin=-55, vmax=55, dark=lambda v: abs(v) > 24, cb="improvement [dB]")
WHAT = {"level": "highest ambiguity with SATA whole band [dB relative to the reference peak]",
        "improv": "improvement = highest ambiguity (no SATA) $-$ highest ambiguity (SATA whole band) [dB]"}
FOOT = "$N_{rx}$ = 4, PRF = 2000 Hz, $B_a$ = 1235 Hz, $b_{xt}$ = [63.7, 27.0, 4.1, 1.7] m. $\\Delta C_0$ map: block map with copies."

ALPHAS = [0.1, 1, 1.5, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
DIST = {"dS": "targets 200 m apart", "dN": "targets 2.6 km apart", "dL": "targets 10 km apart"}
HRAMP = {"hS": "small heights (target j at height j$\\cdot$20 m$\\cdot$tan$\\alpha$)",
         "hN": "normal heights (target j at height j$\\cdot$200 m$\\cdot$tan$\\alpha$)",
         "hL": "large heights (target j at height j$\\cdot$2600 m$\\cdot$tan$\\alpha$)"}
FAMS = {"RH": "random heights", "PW": "piecewise ramp", "ZI": "zig-zag ramp", "ST": "steepening ramp",
        "FL": "flattening ramp", "CL": "cliff", "SK": "spike", "ME": "mesa"}


def heat(ax, M, xt, yt, style, fs=7.5):
    im = ax.imshow(M, cmap=style["cmap"], vmin=style["vmin"], vmax=style["vmax"], aspect="auto")
    ax.set_xticks(range(len(xt)), xt, fontsize=8.5); ax.set_yticks(range(len(yt)), yt, fontsize=8.5)
    ax.set_xticks(np.arange(-.5, len(xt)), minor=True); ax.set_yticks(np.arange(-.5, len(yt)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    ax.tick_params(length=2.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}".replace("-0.0", "0.0"), ha="center", va="center", fontsize=fs,
                        color="white" if style["dark"](M[i, j]) else "#222222")
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def value(r, what):
    w, n = float(r["ambpk_db_whole"]), float(r["ambpk_db_none"])
    return w if what == "level" else n - w


def ramps_page(rows, what):
    style = LEVEL if what == "level" else IMPROV
    fig = plt.figure(figsize=(19, 9.8))
    gs = GridSpec(3, 4, width_ratios=[1, 1, 1, 0.03], height_ratios=[6, 3, 3], hspace=0.30, wspace=0.05, figure=fig,
                  left=0.075, right=0.955, top=0.895, bottom=0.085)
    for i, dk in enumerate(DIST):
        for j, hk in enumerate(HRAMP):
            rs = [r for r in rows if r["config"] == f"RA_{dk}_{hk}"]
            Ns = sorted({int(r["N"]) for r in rs})
            M = np.full((len(Ns), len(ALPHAS)), np.nan)
            for r in rs:
                M[Ns.index(int(r["N"])), ALPHAS.index(float(r["slug"].split("_a")[1]))] = value(r, what)
            ax = fig.add_subplot(gs[i, j])
            im = heat(ax, M, [f"{a:g}" for a in ALPHAS], Ns, style)
            if i == 0:
                ax.set_title(HRAMP[hk], fontsize=10.5)
            if j == 0:
                ax.set_ylabel(f"{DIST[dk]}\nnumber of targets", fontsize=10)
            else:
                ax.set_yticklabels([]); ax.tick_params(left=False)
            if i == 2:
                ax.set_xlabel("ramp slope $\\alpha$ [deg]", fontsize=10)
    fig.colorbar(im, cax=fig.add_subplot(gs[:, 3])).set_label(style["cb"], fontsize=10)
    fig.suptitle(f"Along-track ramps: {WHAT[what]}", fontsize=13, y=0.965)
    fig.text(0.5, 0.018, "Each cell is one case. " + FOOT, ha="center", fontsize=9, color="0.35")
    return fig


def random_page(rows):
    fig = plt.figure(figsize=(19, 7.6))
    gs = GridSpec(1, 6, width_ratios=[1, 1, 0.03, 0.16, 1, 0.03], wspace=0.06, figure=fig, left=0.075, right=0.955, top=0.79, bottom=0.17)
    cols = [(dk, hk) for dk in DIST for hk in ("hS", "hN", "hL")]
    xt = [{"hS": "small", "hN": "normal", "hL": "large"}[hk] for _, hk in cols]
    worst = lambda key: np.array([[max([float(r[key]) for r in rows if r["config"] == f"{pre}_{dk}_{hk}"], default=np.nan)
                                   for dk, hk in cols] for pre in FAMS])
    none, whole = worst("ambpk_db_none"), worst("ambpk_db_whole")
    panels = ((0, none, LEVEL, "no SATA: highest ambiguity [dB]", None),
              (1, whole, LEVEL, "SATA whole band: highest ambiguity [dB]", 2),
              (4, none - whole, IMPROV, "improvement = no SATA $-$ SATA whole band [dB]", 5))
    for col, M, style, title, cb in panels:
        ax = fig.add_subplot(gs[0, col])
        im = heat(ax, M, xt, list(FAMS.values()), style, fs=9)
        if col:
            ax.set_yticklabels([]); ax.tick_params(left=False)
        for k, dk in enumerate(DIST):
            ax.text(3 * k + 1, -0.72, DIST[dk], ha="center", va="bottom", fontsize=9.5)
            if k:
                ax.axvline(3 * k - 0.5, color="white", lw=5)
        ax.set_xlabel("heights of the scene", fontsize=10)
        ax.set_title(title, fontsize=10.5, pad=28)
        if cb is not None:
            fig.colorbar(im, cax=fig.add_subplot(gs[0, cb])).set_label(style["cb"], fontsize=10)
    fig.suptitle("Random geometries: highest ambiguity [dB relative to the reference peak] without SATA and with SATA whole band", fontsize=13, y=0.975)
    fig.text(0.5, 0.03, "Each cell collects the cases of one kind of geometry (5 to 13 targets when 200 m apart, 5 targets otherwise; several heights "
             "or slopes) and shows the highest of them.\nHeights of the scene: small = 0.1 x, normal = 1 x, large = 13 x the base heights. " + FOOT,
             ha="center", fontsize=9, color="0.35")
    return fig


def terrain_results():
    """[(label, {n: amb dict})] for the 20 sine / cosine terrains, A + A first."""
    out = []
    for folder, shapes in TERRAIN:
        res = {}
        for path in sorted(glob.glob(os.path.join(FIG, folder, "results*.json"))):
            res.update(json.load(open(path)))
        for shape in shapes:
            b, a, p = (s[1:] for s in shape.split("-"))
            for fn in ("SIN", "COS"):
                out.append((f"h = {b} + {a} {fn.lower()}(2$\\pi$x / {p} km)", {n: res[f"{fn}-{shape}|{n}"] for n in COUNTS}))
    return out


def terrain_page():
    data = terrain_results()
    fig = plt.figure(figsize=(19, 9.8))
    gs = GridSpec(1, 5, width_ratios=[1, 0.025, 0.11, 1, 0.025], wspace=0.06, figure=fig, left=0.125, right=0.955, top=0.885, bottom=0.10)
    for p, (what, style) in enumerate((("level", LEVEL), ("improv", IMPROV))):
        M = np.array([[d[n]["whole"] if what == "level" else d[n]["none"] - d[n]["whole"] for n in COUNTS] for _, d in data])
        ax = fig.add_subplot(gs[0, 3 * p])
        im = heat(ax, M, COUNTS, [lab for lab, _ in data], style, fs=8.5)
        if p == 1:
            ax.set_yticklabels([]); ax.tick_params(left=False)
        ax.axhline(len(data) / 2 - 0.5, color="white", lw=5)
        ax.set_xlabel("number of point targets on the terrain", fontsize=10)
        ax.set_title(WHAT[what].replace(" [dB", "\n[dB") if what == "level" else WHAT[what].replace(" $-$", "\n$-$"), fontsize=10.5)
        fig.colorbar(im, cax=fig.add_subplot(gs[0, 3 * p + 1])).set_label(style["cb"], fontsize=10)
    fig.suptitle("Sine and cosine terrains with point targets on them, SATA whole band", fontsize=13, y=0.975)
    fig.text(0.5, 0.022, "Each cell is one case; the terrain is the scene and the point targets lie on it, evenly spaced in azimuth over at most 10.8 km. "
             + FOOT, ha="center", fontsize=9, color="0.35")
    return fig


def case(name):
    return os.path.join(HERE, "cases", name)


def terrain_case(folder, cid, n):
    return os.path.join(FIG, folder, f"{cid}_N{n:02d}_footprint.png")


def pages(heatmaps):
    """Ordered list of the pages: a matplotlib figure (heatmap) or the path of an existing PNG."""
    real, comp = "sweep_terrain_footprint_realistic", "sweep_terrain_footprint_complete"
    f = lambda n: os.path.join(FIG, n)
    return [
        # along-track ramps
        heatmaps["ramps_level"], heatmaps["ramps_improv"],
        f("summary_ramps_200m.png"), f("summary_ramps_2600m.png"), f("summary_ramps_10km.png"), f("subband_vs_wholeband_ramps.png"),
        case("RA-SS-012_ramp_N2_a40_footprint.png"), case("RA-012_ramp_N2_a40_footprint.png"), case("RA-049_ramp_N7_a20_footprint.png"),
        case("RA-NN-036_ramp_N5_a20_footprint.png"), case("RA-NN-038_ramp_N5_a40_footprint.png"), case("RA-LN-038_ramp_N5_a40_footprint.png"),
        case("MP-R13_ramp_MP-R13_footprint.png"), case("MP-R25_ramp_MP-R25_footprint.png"), case("MP-R49_ramp_MP-R49_footprint.png"),
        # random geometries
        heatmaps["random"],
        f("summary_random_200m.png"), f("summary_random_2600m.png"), f("summary_random_10km.png"), f("subband_vs_wholeband_random.png"),
        case("RH-017_random_heights_N13_hmax1000_seed0_footprint.png"), case("CL-012_cliff_N9_dh500_footprint.png"),
        case("SK-007_spike_N9_dh1000_footprint.png"), case("ME-005_mesa_N9_dh500_footprint.png"),
        case("RH-NN-005_random_heights_N5_hmax1000_seed0_footprint.png"), case("PW-NN-002_piecewise_N5_seed1_footprint.png"),
        case("ZI-NN-012_zigzag_N5_a40_footprint.png"), case("CL-NN-005_cliff_N5_dh500_footprint.png"),
        case("SK-NN-003_spike_N5_dh1000_footprint.png"), case("ME-NN-002_mesa_N5_dh500_footprint.png"),
        case("RH-LN-005_random_heights_N5_hmax1000_seed0_footprint.png"),
        case("MP-H13_ramp_MP-H13_footprint.png"), case("MP-H25_ramp_MP-H25_footprint.png"),
        # sine / cosine terrains
        heatmaps["terrain"],
        os.path.join(FIG, real, "summary_vs_n_targets_footprint.png"), os.path.join(FIG, comp, "summary_vs_n_targets_footprint.png"),
        terrain_case(real, "SIN-B100-A100-P50", 1), terrain_case(real, "SIN-B100-A100-P50", 5),
        terrain_case(real, "SIN-B100-A100-P50", 6), terrain_case(real, "SIN-B100-A100-P50", 31),
        terrain_case(comp, "SIN-B500-A100-P50", 6),
        terrain_case(real, "COS-B80-A80-P20", 5), terrain_case(real, "COS-B80-A80-P20", 6),
        terrain_case(real, "COS-B40-A40-P5", 6), terrain_case(real, "COS-B40-A40-P5", 7),
        terrain_case(real, "SIN-B40-A40-P5", 31),
        # point target on the reference point (h(0) = 0)
        f("sweep_terrain_footprint_refpoint/REF-A100-P50_N01_footprint.png"),
    ]


def main():
    import img2pdf
    from PIL import Image
    from pypdf import PdfWriter
    rows = list(csv.DictReader(open(CSV)))
    heatmaps = {"ramps_level": ramps_page(rows, "level"), "ramps_improv": ramps_page(rows, "improv"),
                "random": random_page(rows), "terrain": terrain_page()}
    for k, fig in heatmaps.items():
        fig.savefig(os.path.join(HERE, f"heatmap_{k}.png"), dpi=120)
    tmp = tempfile.mkdtemp()
    out = PdfWriter()
    for n, page in enumerate(pages(heatmaps)):
        pdf = os.path.join(tmp, f"p{n:03d}.pdf")
        if isinstance(page, str):                               # existing PNG, embedded as it is (no recompression)
            im = Image.open(page); dpi = im.info.get("dpi", (95, 95))
            rgb = os.path.join(tmp, f"p{n:03d}.png")
            im.convert("RGB").save(rgb, dpi=dpi, optimize=True)
            with open(pdf, "wb") as fh:
                fh.write(img2pdf.convert(rgb))
        else:                                                   # heatmap, as vector graphics
            page.savefig(pdf)
        out.append(pdf)
    dst = os.path.join(FIG, "combined_main_results.pdf")
    out.write(dst)
    print("wrote", dst, len(out.pages), "pages", round(os.path.getsize(dst) / 2 ** 20, 1), "MiB")


if __name__ == "__main__":
    main()
