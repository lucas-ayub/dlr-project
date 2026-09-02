# -*- coding: utf-8 -*-
r"""
delta_C0 vs. sub-band frequency (beta_k), with one star per target, compared
against the old (whole-band) SATA's delta_C0.

WHY THIS PLOT
-------------
`residual_C0_subband` computes delta_C0 for ONE scatterer, ONE channel, ONE
output sub-band k -- i.e. one number. This script sweeps k = 0..Nrx-1 for
EVERY scatterer in a multi-target scene, on a chosen channel, and plots:

  * SATA-sub:  delta_C0(target, k) vs. beta_k [deg]  -- a star per (target, k),
    one line per target (thin, connecting its own stars). Per the C0-invariance
    finding, each target's line should be flat: same delta_C0 regardless of
    sub-band.
  * SATA-band (old, whole-band): delta_C0(target) -- a single number per
    target (no k-dependence), drawn as a horizontal dashed reference line, in
    the same colour as that target's star series, spanning the same beta_k
    axis, so you can see directly whether the per-sub-band value sits on top
    of the whole-band value (expected: yes, since C0 is angle-invariant) or
    diverges from it.

The x-axis (beta_k, the sub-band's centre beam angle in degrees) is a
monotonic function of the sub-band's Doppler frequency, beta_k =
asin(wl f_k / 2 v_s) -- see `subband_frequency_beam` -- so it plays the role
of a frequency axis while staying readable in degrees.

A companion table (printed to console and saved as CSV) gives the exact
numbers for a direct comparison: max|delta_C0_sub(k) - delta_C0_band| per
target, across all channels.

Run (from sar_reconstruction/):
    PYTHONPATH=. python ../runs/core/plot_delta_c0_vs_frequency.py
    PYTHONPATH=. python ../runs/core/plot_delta_c0_vs_frequency.py --nrx 4 --bxt 50
    PYTHONPATH=. python ../runs/core/plot_delta_c0_vs_frequency.py --channel 0
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import sar_recon as sar
from sar_recon.config import (SystemParams, Scene, ArrayGeometry,
                              prf_from_dpca, integration_time, build_time_axis)
from sar_recon.sata import residual_C0
from sar_recon.subband_recon import residual_term_subband, subband_frequency_beam

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots", SCRIPT_NAME)

# Same DPCA/scene conventions as run_subband_c0_c1_c2.py, kept self-contained
# here so this script has no dependency on the driver module.
_DX_DPCA = 11.0
_RDELAY_SCENE = 0.0051115753
_SEED = 0
# (dx [m], dh [m]) for the multi-target, azimuth-varying-topography scene --
# same as AZIMUTH_SPECS in run_subband_c0_c1_c2.py.
AZIMUTH_SPECS = ((-400, 80), (-200, 160), (0, 240), (200, 320), (400, 400))

# A "stretched" version of the same scene (5x the azimuth spread and height of
# AZIMUTH_SPECS) -- use with --large to push dx/dh into a regime where a
# sub-band-dependent effect (C1) would have the best chance of being visible,
# if it is there at all.
AZIMUTH_SPECS_LARGE = tuple((5 * dx, 5 * dh) for dx, dh in AZIMUTH_SPECS)


def build_azimuth_topo(Nrx, bxt_max, specs=AZIMUTH_SPECS, seed=_SEED):
    system = SystemParams()
    base = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0)
    r0, H, y0 = base.r0, base.H, base.y0
    extra = tuple((float(dx), float(np.sqrt(r0 ** 2 - (H - dh) ** 2) - y0), float(dh))
                  for dx, dh in specs)
    scene = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0, extra_offsets=extra)
    array = ArrayGeometry.linear(Nrx, _DX_DPCA, dxt=0.0, bxt_mode="random",
                                 bxt_max=bxt_max, rng=seed)
    prf, PRF_op = prf_from_dpca(system, Nrx, _DX_DPCA)
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.0 * integration_time(system, scene))
    cfg = sar.ExperimentConfig(name=f"aztopo_dpca_Nrx{Nrx}_bxt{int(bxt_max)}", system=system,
                               scene=scene, array=array, prf=prf, PRF_op=PRF_op,
                               Na=Na, Na_ch=Nc, ta=ta, plots_dir=None)
    return cfg, sar.build_platform_tracks(cfg)


def collect_data(cfg, tracks, channel: int, include_center: bool = True):
    """
    For the given channel, return:
      betas     : (Nrx,) beta_k [deg] for each output sub-band
      sub_vals  : (Ntargets, Nrx) delta_C0 per target per sub-band (SATA-sub)
      band_vals : (Ntargets,) delta_C0 per target (SATA-band, whole-band)
      labels    : (Ntargets,) "dx=.., dh=.." strings for the legend
      center_idx: index of the synthetic "scene centre" entry in the arrays
                  above (or None if include_center=False)

    include_center : if True (default), prepend a SYNTHETIC target at
        (dx, dy, dh) = (0, 0, 0) -- i.e. the scene centre itself, the point
        the reconstruction filter assumes. Its residual is delta_C0 =
        C0(centre) - C0(centre) = 0 by construction for EVERY sub-band and
        for the whole-band case -- a sanity check: if this is not (numerically)
        zero, something upstream is broken.
    """
    Nrx = cfg.Nrx
    betas = np.array([np.degrees(subband_frequency_beam(cfg, k)[1]) for k in range(Nrx)])

    center = cfg.scene.ptg
    offsets = list(cfg.scene.extra_offsets)
    center_idx = None
    if include_center:
        offsets = [(0.0, 0.0, 0.0)] + offsets
        center_idx = 0

    sub_vals = np.zeros((len(offsets), Nrx))
    band_vals = np.zeros(len(offsets))
    labels = []
    for i, (dx, dy, dh) in enumerate(offsets):
        ptg_real = center + np.array([dx, dy, dh], dtype=np.float64)
        for k in range(Nrx):
            sub_vals[i, k] = residual_term_subband(cfg, tracks, ptg_real, channel, k, "C0")
        band_vals[i] = residual_C0(cfg, tracks, ptg_real, channel)
        if i == center_idx:
            labels.append("scene centre (sanity check, expect 0)")
        else:
            labels.append(f"dx={dx:.0f} m, $\\Delta h$={dh:.0f} m")
    return betas, sub_vals, band_vals, labels, center_idx


def plot_channel(cfg, tracks, channel: int, ax=None, legend=True, short_title=False, tag=""):
    plt = _mpl()
    if plt is None:
        return None
    betas, sub_vals, band_vals, labels, center_idx = collect_data(cfg, tracks, channel)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    n_ramp = len(labels) - (1 if center_idx is not None else 0)
    ramp_colors = iter(plt.cm.viridis(np.linspace(0, 0.9, n_ramp)))
    colors = []
    for i in range(len(labels)):
        colors.append("0.35" if i == center_idx else next(ramp_colors))
    handles = []
    for i, label in enumerate(labels):
        is_center = (i == center_idx)
        h, = ax.plot(betas, sub_vals[i], "-", color=colors[i],
                     markerfacecolor=colors[i], markeredgecolor="k", markeredgewidth=0.6,
                     marker=("X" if is_center else "*"),
                     markersize=(11 if is_center else 14),
                     lw=(1.6 if is_center else 1.2), label=label,
                     zorder=(5 if is_center else 3))
        handles.append(h)
        ax.axhline(band_vals[i], color=colors[i], ls="--", lw=1.3, alpha=0.85)
    ax.set_xlabel(r"$\beta_k$ [deg]")
    ax.set_ylabel(r"$\delta C_0$ [m]")
    ax.grid(alpha=0.3)
    bxt_ch = cfg.array.bxt[channel]
    if short_title:
        ax.set_title(rf"ch {channel} ($b_{{xt}}$={bxt_ch:.1f} m)", fontsize=10)
    else:
        ax.set_title(rf"$\delta C_0$ vs. sub-band frequency, channel {channel} "
                     rf"($b_{{xt}}$={bxt_ch:.1f} m)" "\n"
                     r"stars+solid = SATA-sub, dashed = SATA-band (old)", fontsize=10)
    if legend:
        ax.legend(fontsize=7, loc="best", ncol=1)
    if own_fig:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        out = os.path.join(PLOTS_DIR, f"delta_c0_vs_freq_ch{channel}_{tag}.png")
        fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"    plot -> {out}")
    return betas, sub_vals, band_vals, labels, handles


def plot_all_channels(cfg, tracks, tag=""):
    """One panel per channel, all on one figure, sharing the y-axis. A SINGLE
    shared legend (targets are the same across channels) sits below the whole
    figure instead of one legend per panel, to avoid overlap."""
    plt = _mpl()
    if plt is None:
        return
    Nrx = cfg.Nrx
    fig, axes = plt.subplots(1, Nrx, figsize=(4.6 * Nrx, 4.8), sharey=True, squeeze=False)
    axes = axes[0]
    handles = labels = None
    for ch, ax in enumerate(axes):
        _betas, _sub, _band, labels, handles = plot_channel(
            cfg, tracks, ch, ax=ax, legend=False, short_title=True)
        if ch > 0:
            ax.set_ylabel("")
    fig.suptitle(rf"$\delta C_0$: SATA-sub (stars, per sub-band) vs. SATA-band (dashed) "
                 rf"-- $N_{{rx}}={Nrx}$, $b_{{xt}}^{{\max}}={cfg.array.bxt.max():.0f}$ m",
                 y=1.06, fontsize=13)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
              bbox_to_anchor=(0.5, -0.06), fontsize=9, frameon=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"delta_c0_vs_freq_all_channels_{tag}.png")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    plot -> {out}")


def plot_scene_geometry(cfg, tag=""):
    """Two-panel view of the scene: (a) side view (height vs. azimuth -- the
    topography 'ramp' the targets sit on), (b) top view (range vs. azimuth --
    shows the iso-range placement, i.e. why range grows with height).

    Reference point (0, 0): the FIRST target of the ramp (offsets[0]), not the
    scene centre -- easier to read the ramp's progression against its own
    starting point. The true scene centre (dx=dy=dh=0, the point the
    reconstruction filter actually assumes) is drawn separately, at whatever
    position that puts it relative to this new reference, and is generally
    NOT at (0, 0) here."""
    plt = _mpl()
    if plt is None:
        return
    offsets = cfg.scene.extra_offsets
    if not offsets:
        print("    (scene has no extra_offsets, skipping geometry plot)")
        return
    dx0, dy0, dh0 = offsets[0]                 # first ramp point -- new origin
    dx = np.array([o[0] for o in offsets]) - dx0
    dy = np.array([o[1] for o in offsets]) - dy0
    dh = np.array([o[2] for o in offsets]) - dh0
    # true scene centre (originally dx=dy=dh=0), expressed in the new frame:
    cx, cy, ch = -dx0, -dy0, -dh0
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(offsets)))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.8))

    axa.scatter(dx, dh, c=colors, s=160, marker="*", edgecolor="k", linewidth=0.6, zorder=3)
    axa.plot(dx, dh, "-", color="0.6", lw=1, zorder=2)
    axa.scatter([cx], [ch], c="red", marker="o", s=70,
               label="scene centre\n(reconstruction pt.)", zorder=4)
    axa.scatter([0], [0], facecolor="none", edgecolor="k", marker="*", s=260,
               linewidth=1.4, label="reference: first ramp point", zorder=5)
    for xi, hi in zip(dx, dh):
        axa.annotate(f"{hi + dh0:.0f} m", (xi, hi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    axa.set_xlabel("Azimuth [m] (relative to first ramp point)")
    axa.set_ylabel("Height [m] (relative to first ramp point)")
    axa.set_title("Side view: topography ramp (targets)")
    axa.grid(alpha=0.3); axa.legend(fontsize=8, loc="upper left")

    axb.scatter(dx, dy, c=colors, s=160, marker="*", edgecolor="k", linewidth=0.6, zorder=3)
    axb.plot(dx, dy, "-", color="0.6", lw=1, zorder=2)
    axb.scatter([cx], [cy], c="red", marker="o", s=70, label="scene centre", zorder=4)
    axb.scatter([0], [0], facecolor="none", edgecolor="k", marker="*", s=260,
               linewidth=1.4, label="reference: first ramp point", zorder=5)
    for xi, yi, hi in zip(dx, dy, dh):
        axb.annotate(f"$\\Delta h$={hi + dh0:.0f} m", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    axb.set_xlabel("Azimuth [m] (relative to first ramp point)")
    axb.set_ylabel("Range [m] (relative to first ramp point)")
    axb.set_title("Top view: ground track (why range grows with height)")
    axb.grid(alpha=0.3); axb.legend(fontsize=8, loc="upper left")

    bxt_str = ", ".join(f"ch{ii}={b:.0f}m" for ii, b in enumerate(cfg.array.bxt))
    fig.suptitle(rf"Scene geometry -- $N_{{rx}}={cfg.Nrx}$, array $b_{{xt}}$: {bxt_str}", y=1.03)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"scene_geometry_{tag}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    plot -> {out}")


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["text.usetex"] = False
        matplotlib.rcParams["axes.formatter.use_mathtext"] = True
        return plt
    except Exception as e:                                # pragma: no cover
        print("    (matplotlib unavailable, skipping plots)", e)
        return None


def write_comparison_table(cfg, tracks, csv_path):
    """
    For every channel and target: delta_C0 per sub-band (SATA-sub), the
    whole-band delta_C0 (SATA-band), and max|sub(k) - band| across k -- the
    direct numeric comparison between the two methods.
    """
    Nrx = cfg.Nrx
    rows = []
    for ch in range(Nrx):
        betas, sub_vals, band_vals, labels, _center_idx = collect_data(cfg, tracks, ch)
        for i, label in enumerate(labels):
            gap = float(np.max(np.abs(sub_vals[i] - band_vals[i])))
            rows.append({
                "channel": ch, "bxt_m": float(cfg.array.bxt[ch]),
                "target": label.replace("$\\Delta h$", "dh").replace("$", ""),
                "delta_C0_band_m": band_vals[i],
                **{f"delta_C0_sub_k{k}_m": sub_vals[i, k] for k in range(Nrx)},
                "max_abs_gap_sub_vs_band_m": gap,
            })
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"    table -> {csv_path}")

    print("\n[delta_C0 comparison: SATA-sub vs. SATA-band]")
    hdr = f"{'ch':>3}{'bxt[m]':>9}  {'target':<28}{'band[m]':>12}{'max|sub-band|[m]':>20}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['channel']:>3}{r['bxt_m']:>9.1f}  {r['target']:<28}"
              f"{r['delta_C0_band_m']:>12.4g}{r['max_abs_gap_sub_vs_band_m']:>20.4g}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--nrx", type=int, default=4)
    ap.add_argument("--bxt", type=float, default=100.0, help="bxt_max [m]")
    ap.add_argument("--channel", type=int, default=None,
                    help="plot only this channel (default: all channels, one panel each)")
    ap.add_argument("--seed", type=int, default=_SEED)
    ap.add_argument("--large", action="store_true",
                    help="use a 5x-stretched scene (AZIMUTH_SPECS_LARGE: dx up to "
                         "+/-2000 m, dh up to 2000 m) and bxt_max=500 m by default, "
                         "to push the geometry into a regime where a sub-band-"
                         "dependent effect (C1) has the best chance of showing up")
    args = ap.parse_args()

    specs = AZIMUTH_SPECS_LARGE if args.large else AZIMUTH_SPECS
    bxt = args.bxt if args.bxt != 100.0 or not args.large else 500.0

    cfg, tracks = build_azimuth_topo(args.nrx, bxt, specs=specs, seed=args.seed)
    print(f"[scene] Nrx={args.nrx}  bxt_max={bxt:.0f} m  large={args.large}  "
          f"bxt={np.round(cfg.array.bxt, 1)}")
    print(f"[targets] {cfg.scene.extra_offsets}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    tag = f"Nrx{args.nrx}_bxt{int(bxt)}" + ("_large" if args.large else "")
    csv_path = os.path.join(PLOTS_DIR, f"delta_c0_comparison_{tag}.csv")
    write_comparison_table(cfg, tracks, csv_path)

    plot_scene_geometry(cfg, tag=tag)

    if args.channel is not None:
        plot_channel(cfg, tracks, args.channel, tag=tag)
    else:
        plot_all_channels(cfg, tracks, tag=tag)
        # also save one full-size figure per channel for closer inspection
        for ch in range(cfg.Nrx):
            plot_channel(cfg, tracks, ch, tag=tag)


if __name__ == "__main__":
    main()
    
    