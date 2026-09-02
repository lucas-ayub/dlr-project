# -*- coding: utf-8 -*-
r"""
Acquisition-geometry plot: who is the REFERENCE and who is the SCENE CENTRE.

Answers three questions visually, straight from the code:

1) Which channel is the reference?
   The reconstruction does NOT use a physical receiver as reference. In
   GetCoeffNu the per-channel coefficients are the residual between
       - the MONOSTATIC round-trip on the transmitter track   rhMS = 2*rhT
         (a virtual sensor co-located with the TX: b_at=0, b_xt=0), and
       - the BISTATIC path of receiver kk                     rhBS = rhA+rhR.
   The matched-filter reference signal (sref) is likewise generated
   TX -> target -> TX (getRawData1D(ptg, ptx, ptx, ...)). So the REFERENCE is
   the ideal monostatic phase centre at the transmitter (marked "REF (mono @ TX)").

2) Who is the scene centre?
   Scene.ptg = [x0, y0, h0] (default x0=20 m, h0=2 m) -- the single point the
   reconstruction fits its coefficients to (sceneMid is always [3,1]). The
   physical scatterers used to GENERATE the signal are ptg + extra_offsets.

3) Does Nrx count only receivers?
   Yes. ArrayGeometry.Nrx = len(bat) = number of RECEIVER channels. There is
   one separate transmitter. So Nrx=3 means 3 receive channels (+ 1 TX).
   Each receiver i sits at along-track -b_at[i] and cross-track b_xt[i]; its
   two-way (DPCA) phase centre is at the midpoint TX--RX, i.e. -b_at[i]/2.

Run (from sar_reconstruction/):
    PYTHONPATH=. python ../runs/core/plot_acquisition_geometry.py --nrx 3
    PYTHONPATH=. python ../runs/core/plot_acquisition_geometry.py --nrx 4 --bxt 50 \
        --ssamp 300 --alpha 5 --ntargets 5
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import sar_recon as sar
from sar_recon.config import (SystemParams, Scene, ArrayGeometry,
                              prf_from_dpca, integration_time, build_time_axis)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots", SCRIPT_NAME)

_DX_DPCA = 11.0
_RDELAY_SCENE = 0.0051115753
_SEED = 0
_V_OVER_PRF = SystemParams().vs / 2000.0


def build_scene(Nrx, bxt_max, S_samp, alpha_deg, n_targets, seed=_SEED):
    system = SystemParams()
    S_m = S_samp * _V_OVER_PRF
    tan_a = np.tan(np.radians(alpha_deg))
    base = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0)
    r0, H, y0 = base.r0, base.H, base.y0
    ks = np.arange(n_targets) - (n_targets - 1) // 2
    k_min = ks.min()
    offs = []
    for k in ks:
        dx = k * S_m
        dh = (k - k_min) * S_m * tan_a
        y = np.sqrt(r0 ** 2 - (H - dh) ** 2) - y0
        offs.append((float(dx), float(y), float(dh)))
    scene = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0,
                  extra_offsets=tuple(offs))
    array = ArrayGeometry.linear(Nrx, _DX_DPCA, dxt=0.0, bxt_mode="random",
                                 bxt_max=bxt_max, rng=seed)
    prf, PRF_op = prf_from_dpca(system, Nrx, _DX_DPCA)
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.4 * integration_time(system, scene))
    cfg = sar.ExperimentConfig(name=f"geom_Nrx{Nrx}", system=system, scene=scene,
                               array=array, prf=prf, PRF_op=PRF_op, Na=Na,
                               Na_ch=Nc, ta=ta, plots_dir=None)
    return cfg


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


def plot_geometry(cfg, tag=""):
    plt = _mpl()
    if plt is None:
        return None
    Nrx = cfg.Nrx
    bat = cfg.array.bat                     # along-track receiver offsets (trail TX)
    bxt = cfg.array.bxt                     # cross-track receiver baselines
    r0 = cfg.scene.r0
    H = cfg.scene.H
    ptg = cfg.scene.ptg                     # scene centre used by reconstruction
    pts = cfg.scene.points                  # [center, center+offsets...]
    tgts = pts[1:]                          # physical scatterers generating signal

    fig = plt.figure(figsize=(13.5, 5.2))
    # --------------------------------------------------------------- panel 1
    # Array layout in the along-track / cross-track plane (platform-local, m).
    ax1 = fig.add_subplot(1, 2, 1)
    # transmitter at origin
    ax1.plot(0, 0, marker="*", ms=18, color="k", label="TX (transmitter)", zorder=5)
    # receivers: along-track = -bat (they trail TX), cross-track = bxt
    ax1.scatter(-bat, bxt, s=70, color="tab:blue", zorder=4,
                label=f"RX channels (Nrx={Nrx})")
    for i in range(Nrx):
        ax1.annotate(f"RX{i}", (-bat[i], bxt[i]), textcoords="offset points",
                     xytext=(4, 6), fontsize=8, color="tab:blue")
    # two-way (DPCA) effective phase centres = midpoint TX-RX
    ax1.scatter(-bat / 2.0, bxt / 2.0, s=45, marker="D", color="tab:green",
                zorder=4, label="effective phase centre (TX--RX midpoint)")
    # reference = monostatic virtual sensor at TX
    ax1.scatter([0], [0], s=180, facecolors="none", edgecolors="tab:red",
                linewidths=2.0, zorder=6, label="REF (monostatic @ TX)")
    ax1.set_xlabel("Along-track [m]  (negative = trailing TX)")
    ax1.set_ylabel("Cross-track baseline $b_{xt}$ [m]")
    ax1.set_title("Array layout (platform-local)")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8, loc="best")
    ax1.axhline(0, color="0.7", lw=0.8, zorder=0)

    # --------------------------------------------------------------- panel 2
    # Imaging geometry in the cross-track (ground range) / height plane (km).
    ax2 = fig.add_subplot(1, 2, 2)
    # platform
    ax2.plot(0, H / 1e3, marker="*", ms=18, color="k", zorder=5,
             label="platform (TX/RX @ H)")
    # line of sight to scene centre
    ax2.plot([0, ptg[1] / 1e3], [H / 1e3, ptg[2] / 1e3], color="tab:red",
             ls="--", lw=1.2, zorder=2, label=f"slant range $r_0$={r0/1e3:.1f} km")
    # scene centre (reconstruction anchor)
    ax2.scatter([ptg[1] / 1e3], [ptg[2] / 1e3], s=130, marker="P",
                color="tab:red", zorder=6, label="scene centre ptg (recon anchor)")
    # physical targets
    ax2.scatter(tgts[:, 1] / 1e3, tgts[:, 2] / 1e3, s=55, color="tab:orange",
                zorder=5, label=f"targets ({len(tgts)}, iso-range)")
    ax2.set_xlabel("Cross-track / ground range [km]")
    ax2.set_ylabel("Height [km]")
    ax2.set_title("Imaging geometry (range--height)")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8, loc="best")

    fig.suptitle(rf"Acquisition geometry | $N_{{rx}}={Nrx}$ (receiver channels) | "
                 rf"1 TX | reference = monostatic phase centre at TX", y=1.02)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"acquisition_geometry_{tag}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    plot -> {out}")

    # also print the numeric layout so it's unambiguous
    print(f"  Nrx = {Nrx} receiver channels (+ 1 transmitter)")
    print(f"  scene centre ptg = [x0={ptg[0]:.1f} m, y0={ptg[1]:.1f} m, h0={ptg[2]:.1f} m]")
    print(f"  reference        = monostatic phase centre at TX (b_at=0, b_xt=0)")
    for i in range(Nrx):
        print(f"    RX{i}: b_at={bat[i]:6.2f} m  b_xt={bxt[i]:6.2f} m  "
              f"-> phase centre along-track {-bat[i]/2:6.2f} m")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrx", type=int, default=3)
    ap.add_argument("--bxt", type=float, default=50.0)
    ap.add_argument("--ssamp", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--ntargets", type=int, default=5)
    args = ap.parse_args()
    cfg = build_scene(args.nrx, args.bxt, args.ssamp, args.alpha, args.ntargets)
    tag = f"Nrx{args.nrx}_bxt{int(args.bxt)}_S{args.ssamp}_a{args.alpha:g}"
    plot_geometry(cfg, tag=tag)


if __name__ == "__main__":
    main()
