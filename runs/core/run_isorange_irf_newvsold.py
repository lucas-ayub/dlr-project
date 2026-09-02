# -*- coding: utf-8 -*-
r"""
IRF / spectrum comparison for MULTIPLE ISO-RANGE TARGETS, focused on the
new explicit sub-band reconstruction vs. the OLD SATA (whole band).

Follows the same "Spectrum comparison, ramp (DPCA)" style already used in
run_subband_recon.py (plot_ramp_topright_*), but:

  * the reconstruction methods compared are
        ref
        no-SATA
        SATA-band          (= OLD SATA, whole band)
        new sub-band (C0)        (subband_sata_explicit, explicit freq axis)
        new sub-band (C0+C1+C2)  (same, all terms)
  * it sweeps the DISTANCE between the iso-range targets (close / medium / far),
  * it draws the target GEOMETRY (side view + top view) for every case,
  * for each case it produces BOTH a side-by-side figure (one panel per method)
    AND a combined figure (all methods on one axes).

Scene: N iso-range targets forming a ramp -- all at the same slant range r0,
spaced in azimuth by S_m = S_samp * (v/PRF), with height growing along the ramp
(dh_k = (k - k_min) * S_m * tan(alpha)), the left-most target on the ground.
Varying S_samp changes only how far apart the targets sit in azimuth.

Run (from sar_reconstruction/):
    PYTHONPATH=. python ../runs/core/run_isorange_irf_newvsold.py
    PYTHONPATH=. python ../runs/core/run_isorange_irf_newvsold.py --nrx 4 --bxt 50 \
        --ssamp 50 300 1000 --alpha 5 --ntargets 5
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os

import numpy as np

import sar_recon as sar
from sar_recon.config import (SystemParams, Scene, ArrayGeometry,
                              prf_from_dpca, integration_time, build_time_axis)
from sar_recon.signal_model import getRawData1D
from sar_recon.sata import sata_channels
from sar_recon.subband_sata_explicit import reconstruct_subband_explicit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots", SCRIPT_NAME)

_DX_DPCA = 11.0
_RDELAY_SCENE = 0.0051115753
_SEED = 0
_V_OVER_PRF = SystemParams().vs / 2000.0          # ~3.844 m/sample (spec sampling)

# Distance cases: label -> S_samp (azimuth spacing in focused samples).
DISTANCE_CASES = {"close": 50, "medium": 300, "far": 1000}


# ---------------------------------------------------------------------------
# scene: N iso-range targets, azimuth spacing S_samp, ramp height
# ---------------------------------------------------------------------------
def build_isorange_ramp(Nrx, bxt_max, S_samp, alpha_deg=5.0, n_targets=5,
                        seed=_SEED, v_over_prf=_V_OVER_PRF):
    """Return (cfg, tracks, S_m, dh_max). n_targets iso-range targets centred on
    k=0, at dx_k = k*S_m, height dh_k = (k - k_min)*S_m*tan(alpha)."""
    system = SystemParams()
    S_m = S_samp * v_over_prf
    tan_a = np.tan(np.radians(alpha_deg))
    base = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0)
    r0, H, y0 = base.r0, base.H, base.y0

    ks = np.arange(n_targets) - (n_targets - 1) // 2       # centred integers
    k_min = ks.min()
    offs = []
    for k in ks:
        dx = k * S_m
        dh = (k - k_min) * S_m * tan_a                     # left-most on ground
        y = np.sqrt(r0 ** 2 - (H - dh) ** 2) - y0          # iso-range placement
        offs.append((float(dx), float(y), float(dh)))
    scene = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0,
                  extra_offsets=tuple(offs))
    array = ArrayGeometry.linear(Nrx, _DX_DPCA, dxt=0.0, bxt_mode="random",
                                 bxt_max=bxt_max, rng=seed)
    prf, PRF_op = prf_from_dpca(system, Nrx, _DX_DPCA)
    # widen the aperture so the outermost targets fit comfortably
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.4 * integration_time(system, scene))
    cfg = sar.ExperimentConfig(name=f"iso_Nrx{Nrx}_S{S_samp}_a{alpha_deg:g}",
                               system=system, scene=scene, array=array, prf=prf,
                               PRF_op=PRF_op, Na=Na, Na_ch=Nc, ta=ta, plots_dir=None)
    return cfg, sar.build_platform_tracks(cfg), S_m, (n_targets - 1) * S_m * tan_a


def channels_and_refs(cfg, tracks):
    s = cfg.system
    ptgs = cfg.scene.points[1:]
    sref1 = getRawData1D(cfg.scene.ptg[None, :], tracks.ptx, tracks.ptx, tracks.vtx,
                         tracks.vtx, cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx,
                         cfg.theta_tx, s.wl, cfg.prf)
    s_ch = np.zeros([cfg.Nrx, cfg.Na_ch], complex)
    for i in range(cfg.Nrx):
        s_ch[i] = getRawData1D(ptgs, tracks.ptx, tracks.prx[i], tracks.vtx,
                               tracks.vrx[i], cfg.ta, cfg.sq_tx, cfg.sq_tx,
                               cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)[::cfg.Nrx]
    return sref1, s_ch


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ---------------------------------------------------------------------------
# reconstruction methods (focus: OLD SATA vs NEW explicit sub-band)
# ---------------------------------------------------------------------------
def build_methods(cfg, tr, s_ch, correct_terms=("C0",)):
    """dict: method-name -> zero-arg callable. Focused comparison:
    OLD SATA (whole band) vs the corrected explicit sub-band."""
    terms_label = "+".join(correct_terms)
    return {
        "SATA-band (old)": lambda: _quiet(
            sar.reconstruct, cfg, tr,
            _quiet(sata_channels, cfg, tr, s_ch.copy(), verbose=False)),
        f"new sub-band ({terms_label})": lambda: _quiet(
            reconstruct_subband_explicit, cfg, tr, s_ch.copy(),
            use_sata=True, verbose=False, correct_terms=correct_terms),
    }


_STYLE = {
    "SATA-band (old)": ("C1", "dashdot"),
    "new sub-band (C0)": ("C0", "dotted"),
}
_EXTRA = [("C0", "dotted"), ("C4", (0, (1, 1))), ("C2", (0, (3, 1, 1, 1)))]


def _style_for(name, idx):
    return _STYLE.get(name, _EXTRA[idx % len(_EXTRA)])


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


# ---------------------------------------------------------------------------
# geometry plot
# ---------------------------------------------------------------------------
def plot_geometry(cases, Nrx, bxt_max, alpha_deg, n_targets):
    """One figure: side view (height vs azimuth) + top view (range vs azimuth),
    overlaying every distance case, azimuth measured from the first target."""
    plt = _mpl()
    if plt is None:
        return None
    fig, (ax_side, ax_top) = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(cases)))
    for (label, S_samp), col in zip(cases.items(), colors):
        cfg, _tr, S_m, dh_max = build_isorange_ramp(Nrx, bxt_max, S_samp,
                                                    alpha_deg, n_targets)
        offs = np.array(cfg.scene.extra_offsets)          # [N,3] = (dx, dy, dh)
        dx = offs[:, 0] - offs[0, 0]                      # azimuth from first tgt
        dh = offs[:, 2]
        r = np.sqrt((cfg.scene.r0 ** 2))                  # all iso-range = r0
        rng = np.full_like(dx, cfg.scene.r0)
        ax_side.plot(dx, dh, "o-", color=col, lw=1.6, ms=7,
                     label=f"{label}: S={S_samp} samp ($S_m$={S_m:.0f} m)")
        ax_top.plot(dx, rng, "o-", color=col, lw=1.6, ms=7,
                    label=f"{label} ($\\Delta h_{{max}}$={dh_max:.0f} m)")
    ax_side.set_xlabel("Azimuth [m]"); ax_side.set_ylabel("Height [m]")
    ax_side.set_title("Side view (iso-range ramp)"); ax_side.grid(alpha=0.3)
    ax_side.legend(fontsize=8)
    ax_top.set_xlabel("Azimuth [m]"); ax_top.set_ylabel("Slant range [m]")
    ax_top.set_title("Top view (all targets on iso-range r$_0$)")
    ax_top.grid(alpha=0.3); ax_top.legend(fontsize=8)
    fig.suptitle(rf"Iso-range target geometry | $N_{{rx}}={Nrx}$ | "
                 rf"$\alpha={alpha_deg:g}^\circ$ | {n_targets} targets", y=1.02)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"geometry_Nrx{Nrx}_a{alpha_deg:g}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    plot -> {out}")
    return out


# ---------------------------------------------------------------------------
# IRF spectrum: combined (all methods, one axes) + side-by-side (one per method)
# ---------------------------------------------------------------------------
def _title(Nrx, bxt_max, S_samp, S_m, alpha_deg, dh_max, label):
    return (rf"IRF spectrum, iso-range ({label}, DPCA) | $N_{{rx}}={Nrx}$ | "
            rf"$b_{{xt}}^{{\max}}={bxt_max:.0f}$ m | $S$={S_samp} samp "
            rf"($S_m$={S_m:.0f} m) | $\alpha={alpha_deg:g}^\circ$ "
            rf"($\Delta h_{{\max}}$={dh_max:.0f} m)")


def plot_irf_case(Nrx, bxt_max, S_samp, label, alpha_deg=5.0, n_targets=5,
                  correct_terms=("C0",)):
    plt = _mpl()
    if plt is None:
        return None, None
    cfg, tr, S_m, dh_max = build_isorange_ramp(Nrx, bxt_max, S_samp,
                                              alpha_deg, n_targets)
    sref1, s_ch = channels_and_refs(cfg, tr)
    methods = build_methods(cfg, tr, s_ch, correct_terms=correct_terms)
    ta = cfg.ta
    tag = f"Nrx{Nrx}_bxt{int(bxt_max)}_S{S_samp}_a{alpha_deg:g}_{label}"

    # precompute spectra once
    specs = {}
    ref_db = None
    idx = 0
    for name, fn in methods.items():
        res = sar.analyze(cfg, sref1, fn())
        if ref_db is None:
            ref_db = 20 * np.log10(np.abs(res.srefF) / np.max(np.abs(res.srefF)))
        specs[name] = 20 * np.log10(np.abs(res.srecNF) / np.max(np.abs(res.srecNF)))

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # -- combined --
    figc, axc = plt.subplots(1, 1, figsize=(8.0, 4.4))
    axc.plot(ta, ref_db, color="k", lw=1.3, ls="solid", label="ref")
    for i, (name, sp) in enumerate(specs.items()):
        col, ls = _style_for(name, i)
        axc.plot(ta, sp, color=col, lw=1.3, ls=ls, label=name)
    axc.set_xlabel("Time [s]"); axc.set_ylabel("[dB]"); axc.set_ylim([-100, 0])
    axc.grid(alpha=0.3); axc.legend(fontsize="small", loc="best")
    axc.set_title(_title(Nrx, bxt_max, S_samp, S_m, alpha_deg, dh_max, label),
                  fontsize=9)
    outc = os.path.join(PLOTS_DIR, f"irf_combined_{tag}.png")
    figc.tight_layout(); figc.savefig(outc, dpi=150, bbox_inches="tight"); plt.close(figc)
    print(f"    plot -> {outc}")

    # -- side by side --
    n = len(specs)
    figs, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    for i, (ax, (name, sp)) in enumerate(zip(axes, specs.items())):
        col, _ls = _style_for(name, i)
        ax.plot(ta, ref_db, color="k", lw=1.2, ls="solid", label="ref")
        ax.plot(ta, sp, color=col, lw=1.3, ls="dashed", label=name)
        ax.set_xlabel("Time [s]"); ax.set_ylim([-100, 0]); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best"); ax.set_title(name, fontsize=9)
    axes[0].set_ylabel("[dB]")
    figs.suptitle(_title(Nrx, bxt_max, S_samp, S_m, alpha_deg, dh_max, label),
                  y=1.03, fontsize=9)
    outs = os.path.join(PLOTS_DIR, f"irf_sidebyside_{tag}.png")
    figs.tight_layout(); figs.savefig(outs, dpi=150, bbox_inches="tight"); plt.close(figs)
    print(f"    plot -> {outs}")
    return outc, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrx", type=int, default=4)
    ap.add_argument("--bxt", type=float, default=50.0)
    ap.add_argument("--alpha", type=float, default=5.0, help="ramp inclination [deg]")
    ap.add_argument("--ntargets", type=int, default=5)
    ap.add_argument("--ssamp", type=int, nargs="+", default=None,
                    help="azimuth spacings in focused samples; default close/medium/far")
    ap.add_argument("--terms", type=str, default="C0",
                    help="comma-separated sub-band terms for the corrected method "
                         "(e.g. 'C0' or 'C0,C1,C2')")
    args = ap.parse_args()

    if args.ssamp is None:
        cases = DISTANCE_CASES
    else:
        cases = {f"S{s}": s for s in args.ssamp}
    correct_terms = tuple(t.strip() for t in args.terms.split(","))

    print(f"[iso-range IRF] Nrx={args.nrx} bxt={args.bxt:.0f} alpha={args.alpha:g} "
          f"ntargets={args.ntargets} terms={correct_terms} cases={cases}")
    plot_geometry(cases, args.nrx, args.bxt, args.alpha, args.ntargets)
    for label, S_samp in cases.items():
        print(f"  case '{label}' (S={S_samp} samp)")
        plot_irf_case(args.nrx, args.bxt, S_samp, label,
                      alpha_deg=args.alpha, n_targets=args.ntargets,
                      correct_terms=correct_terms)


if __name__ == "__main__":
    main()
