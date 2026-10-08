"""
Cases with channel 0 as the transmitter (b_at = b_xt = 0): one figure per case with four blocks,

    IRF after reconstruction (whole azimuth line; with one   |   target geometry (2D and 3D)
    target also the zoom on its main lobe)                    |
    quadratic fit of the range difference of every channel     |   channel geometry

The cases are every combination of a target height (--h-list, one target at azimuth --x) and a set of
cross-track baselines (--bxt-sets: one b_xt per channel, comma-separated; the first is channel 0, the
transmitter, and must be 0), plus the scenes with several targets of --multi (x:h,x:h,... each) with every
baseline set. --targets runs one scene only: that list of targets x:h with every baseline set.
The default grid separates the two effects, and the figures come in this order:

    no topography and no cross-track baseline     h = 0,  b_xt = 0          (control)
    cross-track baseline only                     h = 0,  b_xt != 0
    topography only                               h != 0, b_xt = 0
    topography and cross-track baseline together  h != 0, b_xt != 0

    python tx_rx_channel_cases.py                                      # (3 heights + 2 scenes of 3 targets) x 3 baseline sets
    python tx_rx_channel_cases.py --multi                              # single-target cases only
    python tx_rx_channel_cases.py --h-list 100 500 --bxt-sets 0,20,40,60
    python tx_rx_channel_cases.py --targets=-1500:200,0:500,1500:1000 --bxt-sets 0,10,18,30
    python tx_rx_channel_cases.py --pdf                                # also every case in <out>.pdf

The range fit is the one of the reference point (h = 0), which is what the reconstruction filters use;
it depends on the baselines, not on the targets.
"""
import argparse
import os
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import sata1d as S
import tx_rx_channel as T
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, coordinates_text

H_LIST = [0.0, 200.0, 1000.0]                       # target heights [m]
BXT_SETS = ["0,0,0,0", "0,10,18,30", "0,100,180,300"]   # b_xt of the channels [m]; channel 0 is the transmitter
MULTI = ["-2600:0,0:0,2600:0", "-2600:0,0:500,2600:1000"]   # scenes with several targets x:h (flat; rising A, B, C)
GROUPS = ["No topography and no cross-track baseline", "Cross-track baseline only", "Topography only",
          "Topography and cross-track baseline together"]


def group_of(targets, bxt):
    """Index in GROUPS: which of the two effects (target height, cross-track baseline) the case has."""
    topo = any(h != 0 for _, h in targets)
    xt = any(b != 0 for b in bxt)
    return (2 if topo else 0) + (1 if xt else 0) if not (topo and xt) else 3


def make_case(targets, bxt):
    """Configuration with explicit cross-track baselines; channel 0 on the transmitter."""
    bxt = np.asarray(bxt, float)
    if bxt[0] != 0:
        raise SystemExit(f"b_xt set {bxt}: the first value is channel 0 (the transmitter) and must be 0")
    cfg = S.make_config(targets, nrx=len(bxt), bxt_max=0.0, tx_rx_channel0=True)
    return replace(cfg, array=replace(cfg.array, bxt=bxt))


def block_title(fig, spec, text, dy=0.012):
    """Heading centred above one block of the figure."""
    box = spec.get_position(fig)
    fig.text(0.5 * (box.x0 + box.x1), box.y1 + dy, text, ha="center", va="bottom", fontsize=12.5, fontweight="bold")


def case_figure(cfg, tr, targets, png, pdf=None, group=None):
    xs = np.array([x for x, _ in targets]); hs = np.array([h for _, h in targets])
    irf = T.compute_irf(cfg, tr)
    fits = T.range_fits(cfg, tr)
    Nrx = cfg.Nrx

    fig = plt.figure(figsize=(22, 13))
    outer = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], hspace=0.32, wspace=0.11,
                     left=0.055, right=0.985, top=0.885, bottom=0.115)
    n = len(targets)
    tgt = (f"one target at x = {xs[0]:.0f} m, h = {hs[0]:.0f} m" if n == 1 else
           f"{n} targets at x = [" + ", ".join(f"{x:.0f}" for x in xs) + "] m, h = [" + ", ".join(f"{h:.0f}" for h in hs) + "] m"
           if n <= 6 else f"{n} targets, h from {hs.min():.0f} to {hs.max():.0f} m")
    fig.suptitle((f"{group}:  " if group else "") + f"{tgt}  |  $b_{{xt}}$ = [" + ", ".join(f"{b:g}" for b in cfg.array.bxt) + "] m  |  channel 0 is the transmitter\n"
                 f"$N_{{rx}}$ = {Nrx}, PRF = {cfg.prf:.0f} Hz ({cfg.PRF_op:.0f} Hz per channel), $B_a$ = {cfg.abw:.0f} Hz, "
                 f"$b_{{at}}$ = [" + ", ".join(f"{b:.2f}" for b in cfg.array.bat) + "] m (DPCA)", fontsize=14, y=0.985)

    # IRF after reconstruction
    if n == 1:                                                            # one target: whole line and zoom on its main lobe
        g = outer[0, 0].subgridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.14)
        T.draw_irf(fig.add_subplot(g[0]), fig.add_subplot(g[1]), cfg, irf, small=True)
    else:                                                                 # several targets: the whole line only
        T.draw_irf_line(fig.add_subplot(outer[0, 0]), cfg, irf, small=True)
    block_title(fig, outer[0, 0], "IRF after reconstruction", dy=0.022)

    # target geometry, 2D and 3D
    g = outer[0, 1].subgridspec(1, 3, width_ratios=[1.0, 1.2, 0.07], wspace=0.02)    # last column: room for the 3D labels
    g2 = fig.add_subplot(g[0]); a3 = fig.add_subplot(g[1], projection="3d")
    xs_, dys, hs_ = scene_coordinates(xs, hs)
    dense = len(xs) > 13
    geometry_2d(g2, xs_, dys, hs_, False, title="height against azimuth", annotate=not dense)
    lo, top = min(0.0, hs_.min()), max(0.0, hs_.max())
    pad = 0.15 * max(top - lo, 40.0)
    g2.set_ylim(lo - pad, top + pad)                                      # room around the targets, legend under the axes
    g2.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False)
    geometry_3d(a3, xs_, dys, hs_, False, title="3D (azimuth, ground range, height)")
    a3.set_box_aspect((1.6, 1.3, 0.8), zoom=0.98)                         # keep the 3D view inside its block
    if not dense:
        a3.text2D(0.0, 0.0, coordinates_text(xs_, dys, hs_), transform=a3.transAxes, fontsize=7, family="monospace",
                  ha="left", va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    block_title(fig, outer[0, 1], "Target geometry", dy=0.022)

    # quadratic fit of the range difference, one small panel per channel
    nrow = int(np.ceil(Nrx / 2))
    g = outer[1, 0].subgridspec(nrow, 2, hspace=0.42, wspace=0.24)
    axs = [fig.add_subplot(g[i // 2, i % 2]) for i in range(Nrx)]
    for i, ax in enumerate(axs):
        T.draw_range_fit(ax, cfg, i, fits[i], small=True)
        if i // 2 == nrow - 1 or i + 2 >= Nrx:
            ax.set_xlabel("$\\tau$ = time from the closest approach [s]", fontsize=9)
        else:
            ax.tick_params(labelbottom=False)
    box = outer[1, 0].get_position(fig)
    fig.legend(handles=T.range_fit_legend_handles(), loc="upper center", bbox_to_anchor=(0.5 * (box.x0 + box.x1), box.y0 - 0.04),
               ncol=2, fontsize=9, frameon=False)
    block_title(fig, outer[1, 0], "Quadratic fit of the range difference (bistatic $-$ monostatic) of the reference point, per channel", dy=0.022)

    # channel geometry
    ax = fig.add_subplot(outer[1, 1])
    T.draw_geometry(ax, cfg, small=True)
    block_title(fig, outer[1, 1], "Channel geometry", dy=0.022)

    fig.text(0.5, 0.008, T.irf_note(cfg, irf), ha="center", va="bottom", fontsize=9.5, color="0.35")
    fig.savefig(png, dpi=100)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)
    return irf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=float, default=0.0, help="azimuth position of the single target [m]")
    ap.add_argument("--h-list", type=float, nargs="+", default=H_LIST, help="target heights [m], one case per height")
    ap.add_argument("--bxt-sets", nargs="+", default=BXT_SETS, help="b_xt of the channels [m], comma-separated, one set per case")
    ap.add_argument("--multi", nargs="*", default=MULTI, help="scenes with several targets, each x:h,x:h,... (no value: none)")
    ap.add_argument("--targets", nargs="+", default=None, help="x:h [x:h ...] (or one comma-separated list): this scene only")
    ap.add_argument("--out", default="tx_rx_channel_cases")
    ap.add_argument("--pdf", action="store_true", help="also write every case in <out>.pdf")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sets = [[float(v) for v in s.split(",")] for s in a.bxt_sets]
    if a.targets:
        toks = [t for s in a.targets for t in s.split(",") if t]
        scenes = [("targets", [tuple(float(v) for v in t.split(":")) for t in toks])]
    else:
        scenes = [(f"h{h:04.0f}", [(a.x, h)]) for h in a.h_list]
        for s in a.multi:
            tg = [tuple(float(v) for v in t.split(":")) for t in s.split(",") if t]
            scenes.append((f"targets{len(tg)}_h" + "-".join(f"{h:.0f}" for _, h in tg), tg))
    pdf = PdfPages(a.out.rstrip("/") + ".pdf") if a.pdf else None
    cases = sorted(((group_of(targets, bxt), ib, it, bxt, tag, targets) for ib, bxt in enumerate(sets)
                    for it, (tag, targets) in enumerate(scenes)), key=lambda c: c[:3])
    print(f"{'group':>45} {'target height(s) [m]':>22} {'b_xt [m]':>20}   main-lobe peak of each target [dB]")
    for g, _, _, bxt, tag, targets in cases:
        cfg = make_case(targets, bxt); tr = S.build_platform_tracks(cfg)
        png = os.path.join(a.out, f"{tag}_bxt{max(bxt):03.0f}.png")
        irf = case_figure(cfg, tr, targets, png, pdf, GROUPS[g])
        desc = ", ".join(f"{h:.0f}" for _, h in targets)
        print(f"{GROUPS[g]:>45} {desc:>22} {str([f'{b:g}' for b in bxt]).replace(chr(39), ''):>20}   "
              + "  ".join(f"{q['peak']:+7.2f}" for q in irf["targets"]), flush=True)
    if pdf is not None:
        pdf.close(); print("wrote", a.out.rstrip("/") + ".pdf")


if __name__ == "__main__":
    main()
