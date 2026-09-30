"""
Report figures of single sweep cases, from the .npz files the sweep scripts write:
  <case>_A.png  azimuth IRFs (reference / SATA whole band / SATA per sub-band / no SATA)
                with the target geometry in 2D and in 3D;
  <case>_B.png  (2-D sweep only) 2-D IRF contour of the representative target and the
                phase of the reconstructed spectrum vs the reference along f_r and f_a.

Usage: python -m sata2d.plot_case_pages <npz ...> --out DIR [--label "SATA 2D"]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from .sweep_geometry_plots import geometry_2d, geometry_3d, coordinates_text
except ImportError:
    from sweep_geometry_plots import geometry_2d, geometry_3d, coordinates_text

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})


def page_a(z, title, out_png, topo):
    t = z["t"]
    env = {k[4:]: z[k] for k in z.files if k.startswith("env_")}
    m = z["metrics"][0]
    pk = env["ref"].max()
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    xs, dys, hs = z["xs"], z["dys"], z["hs"]
    fig = plt.figure(figsize=(11.5, 5.6))
    gs = GridSpec(2, 2, width_ratios=[1.6, 1.0], hspace=0.5, wspace=0.16, figure=fig)
    fig.suptitle(title, fontsize=10, y=1.0)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=1.8, label="Reference")
    ax.plot(t, db["whole"], color="tab:blue", lw=1.2, ls=(0, (5, 3)), label="SATA whole band")
    ax.plot(t, db["sub"], color="tab:orange", lw=1.4, ls=(0, (1, 2)), label="SATA per sub-band")
    ax.set_title(f"worst ambiguity: reference {m['amb_db_ref']:.1f} dB | SATA whole {m['amb_db_whole']:.1f} dB | "
                 f"SATA sub {m['amb_db_sub']:.1f} dB | oracle {m['amb_db_oracle']:.1f} dB")
    az = fig.add_subplot(gs[1, 0])
    az.plot(t, db["ref"], color="tab:green", lw=1.8, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.2, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"worst ambiguity: reference {m['amb_db_ref']:.1f} dB | no SATA {m['amb_db_none']:.1f} dB")
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(loc="upper right")
    geometry_2d(fig.add_subplot(gs[0, 1]), xs, dys, hs, topo, title="target geometry: height profile")
    a3 = fig.add_subplot(gs[1, 1], projection="3d")
    geometry_3d(a3, xs, dys, hs, topo, title="target geometry in 3D")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def page_b(z, title, out_png):
    xs, dys, hs = z["xs"], z["dys"], z["hs"]
    j = int(z["j"])
    fig, (ac, ar, aa) = plt.subplots(1, 3, figsize=(11.5, 3.5), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.0], wspace=0.32))
    fig.suptitle(title + f" | representative target {j + 1}/{len(xs)}: x = {xs[j]:.0f} m, "
                 f"$\\Delta y$ = {dys[j]:.0f} m, h = {hs[j]:.0f} m", fontsize=10, y=1.02)
    levels = [-30, -20, -13, -10, -6, -3]
    cs = ac.contour(z["x_rg"], z["x_az"], z["contour_sub"], levels=levels, cmap="RdYlBu_r", linewidths=0.9)
    ac.clabel(cs, levels, fmt="%g", fontsize=6)
    ac.set_xlabel("range [m]"); ac.set_ylabel("azimuth [m]"); ac.grid(alpha=0.3)
    ac.set_title("2-D IRF, SATA per sub-band [dB re. reference peak]")
    for k, col, lab in (("none", "tab:red", "no SATA"), ("sub", "tab:orange", "SATA per sub-band"),
                        ("oracle", "tab:purple", "oracle")):
        ar.plot(z["fr"] / 1e6, z[f"phase_r_{k}"], color=col, lw=0.8, label=lab)
        aa.plot(z["fa"] / 1e3, z[f"phase_a_{k}"], color=col, lw=0.8, label=lab)
    ar.set_xlabel("range frequency [MHz]"); ar.set_ylabel("phase [deg]")
    ar.set_title("phase, reconstructed vs reference, along $f_r$ ($f_a=0$)")
    aa.set_xlabel("azimuth frequency [kHz]"); aa.set_ylabel("phase [deg]")
    aa.set_title("phase, reconstructed vs reference, along $f_a$ ($f_r=0$)")
    for a_ in (ar, aa):
        a_.set_ylim(-180, 180); a_.grid(alpha=0.3); a_.legend(loc="upper right", fontsize=7)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", default=".")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in a.npz:
        z = np.load(f, allow_pickle=True)
        name = os.path.splitext(os.path.basename(f))[0]
        topo = bool(np.all(z["xs"] == 0.0)) and len(z["xs"]) > 1
        title = f"{a.label} | {name}" if a.label else name
        page_a(z, title, os.path.join(a.out, name + "_A.png"), topo)
        if "contour_sub" in z.files:
            page_b(z, title, os.path.join(a.out, name + "_B.png"))
        print("done", name)


if __name__ == "__main__":
    main()
