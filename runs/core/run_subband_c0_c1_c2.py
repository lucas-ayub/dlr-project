# -*- coding: utf-8 -*-
r"""
Driver / test for the per-sub-band SATA reconstruction (subband_recon).

Compares, for each (Nrx, bxt_max) case, up to five reconstructions of the same
channel data:

    no-SATA                : flat-earth reconstruction (ignores topography)
    SATA band               : sar_recon.sata.sata_channels -- one broadside
                               pass per channel for the whole band
    SATA sub (C0)            : subband_recon.reconstruct_subband, per-sub-band,
                               correcting only the C0 (bulk range) residual --
                               the validated, whole-band-equivalent term
    SATA sub (C0+C1)         : same, ALSO correcting the C1 (linear / azimuth
                               registration) residual -- EXPERIMENTAL
    SATA sub (C0+C1+C2)      : same, ALSO correcting the C2 (quadratic /
                               focus-rate) residual -- EXPERIMENTAL

C1/C2 correction is opt-in via `correct_terms` on `reconstruct_subband` /
`sata_channels_subband` (see sar_recon/subband_recon.py and sar_recon/sata.py
for the exact phase model). Use `--terms C0,C1` (etc.) to restrict every
sweep/plot in this script to a single explicit combination instead of the
default three-way comparison.

It does four things:

  1. SELF-TEST: reconstruct_subband(use_sata=False) must equal sar.reconstruct
     bit-for-bit (the sub-band plumbing reduces to the baseline).

  2. SWEEP over (Nrx, bxt_max) [DPCA mode, random cross-track baselines] on an
     azimuth-varying topography scene, reporting focused-peak recovery and
     worst azimuth ambiguity for every method above.

  3. PLOTS (on by default; --no-plots to skip). Figures go to
     runs/core/plots/<this-file's-name>/ (png) -- e.g. if this file is renamed
     to run_subband_c0_c1_c2.py, figures land in
     runs/core/plots/run_subband_c0_c1_c2/ automatically (see SCRIPT_NAME):
       (a) residual_per_subband  -- dC0/dC1/dC2 vs sub-band k for one elevated
           target: dC0 flat, dC1 antisymmetric in beta_k (the key finding);
       (b) peak_vs_dxt           -- focused-peak recovery vs bxt_max, per method;
       (c) combined_4panel_*     -- the STANDARD reconstruction 4-panel (as
           run_experiment.plot_combined: amplitude, spectrum [dB], zoomed IRF,
           spectral phase with the B_a lines), one per method;
       (d) sata_grid_*           -- the Nrx x Nrx 'Output of SATA' grid;
       (e) topright_spectrum_*   -- single-panel overlay of the "spectrum [dB]"
           (top-right) panel for the ramp scene, all methods on one axes,
           swept over alpha (ramp inclination) as well as bxt/S_samp.

Plots use matplotlib mathtext ($...$), so no LaTeX toolchain is needed
(consistent with USE_LATEX=0). Console output is ASCII, copy-paste-safe.

Place under runs/core/ in the repo. Run:
    cd sar_reconstruction
    PYTHONPATH=. python ../runs/core/run_subband_recon.py
    PYTHONPATH=. python ../runs/core/run_subband_recon.py --nrx 4 --bxt 100
    PYTHONPATH=. python ../runs/core/run_subband_recon.py --alpha 10
    PYTHONPATH=. python ../runs/core/run_subband_recon.py --terms C0,C1
    PYTHONPATH=. python ../runs/core/run_subband_recon.py --no-plots
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import sar_recon as sar
from sar_recon.config import (SystemParams, Scene, ArrayGeometry,
                              prf_from_dpca, integration_time, build_time_axis)
from sar_recon.signal_model import getRawData1D
from sar_recon.sata import sata_channels
from sar_recon.subband_recon import (reconstruct_subband, subband_frequency_beam,
                                      getcoeff_beam, sata_channels_subband)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Derived from this script's own filename (not hardcoded) so the plots folder
# always matches whatever this file is actually called -- e.g. renamed to
# run_subband_c0_c1_c2.py -> plots/run_subband_c0_c1_c2/.
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots", SCRIPT_NAME)

# --- DPCA is the default operating mode (small along-track spacing; the PRF is
#     set by the DPCA condition PRF_op = 2*vs/(Nrx*dx)). Cross-track baselines are
#     drawn in RANDOM mode, uniform on (0, bxt_max). ------------------------------
_DX_DPCA = 11.0                       # DPCA along-track spacing [m] -> bat = 11*i
_RDELAY_SCENE = 0.0051115753          # valid off-nadir scene geometry (r0 > H) for
                                      # iso-range targets; DPCA timing is set by the
                                      # ARRAY (dx) and PRF, not by the scene range
_SEED = 0                             # RNG seed for the random bxt draw

DEFAULT_NRX = [2, 3, 4, 5]
DEFAULT_BXT = [10, 20, 30, 50]        # bxt_max values swept (all in the working regime)
# cases for the standard 4-panel showcase (single elevated target):
PANEL_CASES = [(4, 20), (4, 50)]      # single-target 4-panel cases (low bxt)
# (dx [m], dh [m]) for the azimuth-varying topography scene: a ramp of targets
# spread along azimuth, each at a growing iso-range height.
AZIMUTH_SPECS = ((-400, 80), (-200, 160), (0, 240), (200, 320), (400, 400))

# ---------------------------------------------------------------------------
# Which residual term(s) SATA-sub corrects. Every sweep/table/plot in this
# script compares "SATA sub" for EACH entry of TERM_SETS side by side, so you
# can see directly what each extra term buys you. Override the whole list at
# once with --terms (restricts every method to a single explicit combination).
#
#   C0        bulk range / phase offset          -- validated (see sata.py)
#   C0 + C1   + azimuth registration / linear     -- EXPERIMENTAL (see below)
#   C0+C1+C2  + focus / chirp-rate (quadratic)     -- EXPERIMENTAL (see below)
#
# "EXPERIMENTAL" = the phase-correction mechanism for C1/C2 is a first-order
# extension of the classic (C0-only) SATA kernel (see sata.sata_1d), reusing
# the same position->Doppler-bin map; it has not been derived/validated with
# the same rigour as the C0 term. Use these columns to see the DIRECTION and
# MAGNITUDE of the effect, not as a finished result.
# ---------------------------------------------------------------------------
TERM_SETS = [("C0",), ("C0", "C1"), ("C0", "C1", "C2")]


def _term_label(terms: tuple) -> str:
    return "SATA sub (" + "+".join(terms) + ")"


def _method_order(term_sets=None):
    term_sets = term_sets if term_sets is not None else TERM_SETS
    return ["no-SATA", "SATA band"] + [_term_label(t) for t in term_sets]


_BASE_COLORS = {"no-SATA": "C3", "SATA band": "C1"}
_SUB_COLORS = ["C0", "C2", "C4", "C5", "C6"]          # one per extra term-set
_SUB_LS = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (1, 1))]


def _method_style(name, term_sets=None):
    """Return (color, linestyle) for a method name, stable across plots."""
    if name in _BASE_COLORS:
        return _BASE_COLORS[name], "-"
    term_sets = term_sets if term_sets is not None else TERM_SETS
    labels = [_term_label(t) for t in term_sets]
    i = labels.index(name) if name in labels else 0
    return _SUB_COLORS[i % len(_SUB_COLORS)], _SUB_LS[i % len(_SUB_LS)]


# ---------------------------------------------------------------------------
# scene + focusing helpers -- DPCA timing + random cross-track baselines
# ---------------------------------------------------------------------------
def _dpca_array(Nrx, bxt_max, seed=_SEED):
    """Linear array with DPCA along-track spacing and RANDOM bxt (uniform 0..max)."""
    return ArrayGeometry.linear(Nrx, _DX_DPCA, dxt=0.0,
                                bxt_mode="random", bxt_max=bxt_max, rng=seed)


def build_azimuth_topo(Nrx, bxt_max, specs=AZIMUTH_SPECS, seed=_SEED):
    system = SystemParams()
    base = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0)
    r0, H, y0 = base.r0, base.H, base.y0
    extra = tuple((float(dx), float(np.sqrt(r0 ** 2 - (H - dh) ** 2) - y0), float(dh))
                  for dx, dh in specs)
    scene = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0, extra_offsets=extra)
    array = _dpca_array(Nrx, bxt_max, seed)
    prf, PRF_op = prf_from_dpca(system, Nrx, _DX_DPCA)
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.0 * integration_time(system, scene))
    cfg = sar.ExperimentConfig(name=f"aztopo_dpca_Nrx{Nrx}_bxt{int(bxt_max)}", system=system,
                               scene=scene, array=array, prf=prf, PRF_op=PRF_op,
                               Na=Na, Na_ch=Nc, ta=ta, plots_dir=None)
    return cfg, sar.build_platform_tracks(cfg)


def build_single_target(Nrx, bxt_max, dh, seed=_SEED):
    """One elevated iso-range target (DPCA timing, random bxt)."""
    system = SystemParams()
    base = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0)
    r0, H, y0 = base.r0, base.H, base.y0
    off = (0.0, float(np.sqrt(r0 ** 2 - (H - dh) ** 2) - y0), float(dh))
    scene = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0, extra_offsets=(off,))
    array = _dpca_array(Nrx, bxt_max, seed)
    prf, PRF_op = prf_from_dpca(system, Nrx, _DX_DPCA)
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.0 * integration_time(system, scene))
    cfg = sar.ExperimentConfig(name=f"one_dpca_Nrx{Nrx}_bxt{int(bxt_max)}", system=system,
                               scene=scene, array=array, prf=prf, PRF_op=PRF_op,
                               Na=Na, Na_ch=Nc, ta=ta, plots_dir=None)
    return cfg, sar.build_platform_tracks(cfg), cfg.scene.ptg + np.array(off)


# v/PRF used to express the ramp spacing in focused samples. Defined from the
# reference sampling PRF = 2000 Hz (v_s/2000 ~ 3.844 m/sample), per the spec; the
# reconstruction itself still runs in DPCA mode.
_V_OVER_PRF = SystemParams().vs / 2000.0          # ~ 3.844 m/sample
_RAMP_SSAMP = [50, 100, 300, 600, 1000]           # focused-sample spacings swept
_RAMP_ALPHA_DEG = 2.0                             # default ramp inclination [deg]
RAMP_ALPHA_SWEEP = [5.0, 10.0, 15.0, 20.0]        # alpha values swept for the ramp scene


def build_ramp(Nrx, bxt_max, S_samp, alpha_deg=_RAMP_ALPHA_DEG, seed=_SEED,
               v_over_prf=_V_OVER_PRF):
    r"""Five iso-range targets forming a genuine ramp (DPCA timing, random bxt).

    Targets at dx_k = k * S_m, k in {-2,-1,0,1,2}, with the spacing set in focused
    samples S_m = S_samp * (v/PRF). The height varies with dx (the ramp), with the
    left-most target on the ground:
        dh_k = (dx_k - dx_min) tan(alpha) = (k+2) S_m tan(alpha),
    so dh grows with both the spacing and the inclination (dh_max = 4 S_m tan a).
    Each target is placed on the iso-range surface (same slant range r0).
    Returns (cfg, tracks, S_m, dh_max).
    """
    system = SystemParams()
    S_m = S_samp * v_over_prf
    tan_a = np.tan(np.radians(alpha_deg))
    base = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0)
    r0, H, y0 = base.r0, base.H, base.y0
    offs = []
    for k in (-2, -1, 0, 1, 2):
        dx = k * S_m
        dh = (k + 2) * S_m * tan_a                     # k=-2 -> 0 (on the ground)
        y = np.sqrt(r0 ** 2 - (H - dh) ** 2) - y0      # iso-range placement
        offs.append((float(dx), float(y), float(dh)))
    scene = Scene(rDelay=_RDELAY_SCENE, c0=system.c0, h0=0.0, extra_offsets=tuple(offs))
    array = _dpca_array(Nrx, bxt_max, seed)
    prf, PRF_op = prf_from_dpca(system, Nrx, _DX_DPCA)
    # widen the simulated aperture so the outermost targets (dx = +/-2 S_m) fit
    Na, Nc, ta = build_time_axis(prf, Nrx, 2.4 * integration_time(system, scene))
    cfg = sar.ExperimentConfig(name=f"ramp_Nrx{Nrx}_S{S_samp}_a{alpha_deg:g}",
                               system=system, scene=scene, array=array, prf=prf,
                               PRF_op=PRF_op, Na=Na, Na_ch=Nc, ta=ta, plots_dir=None)
    return cfg, sar.build_platform_tracks(cfg), S_m, (4 * S_m * tan_a)


def channels_and_refs(cfg, tracks):
    s = cfg.system
    ptgs = cfg.scene.points[1:]                         # elevated targets
    sref1 = getRawData1D(cfg.scene.ptg[None, :], tracks.ptx, tracks.ptx, tracks.vtx,
                         tracks.vtx, cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx,
                         cfg.theta_tx, s.wl, cfg.prf)
    sig_true = getRawData1D(ptgs, tracks.ptx, tracks.ptx, tracks.vtx, tracks.vtx,
                            cfg.ta, cfg.sq_tx, cfg.sq_tx, cfg.theta_tx, cfg.theta_tx,
                            s.wl, cfg.prf)
    s_ch = np.zeros([cfg.Nrx, cfg.Na_ch], complex)
    for i in range(cfg.Nrx):
        s_ch[i] = getRawData1D(ptgs, tracks.ptx, tracks.prx[i], tracks.vtx,
                               tracks.vrx[i], cfg.ta, cfg.sq_tx, cfg.sq_tx,
                               cfg.theta_tx, cfg.theta_tx, s.wl, cfg.prf)[::cfg.Nrx]
    return sref1, sig_true, s_ch


def focus_mag(sig, ref):
    S = np.fft.fft(sig) * np.conj(np.fft.fft(ref))
    return np.abs(np.roll(np.fft.ifft(S), len(ref) // 2))


def ambiguity_db(f, mask=100):
    f = f / f.max()
    i0 = int(np.argmax(f))
    m = np.ones(len(f), bool)
    m[max(0, i0 - mask):i0 + mask] = False
    return 20.0 * np.log10(f[m].max())


def subband_residuals(Nrx, bxt_max, dh):
    """dC0/dC1/dC2 vs sub-band k for one elevated iso-range target. Since bxt is
    random per channel, use the channel with the largest |bxt| so the residual is
    clearly visible."""
    cfg, tr, ptg_real = build_single_target(Nrx, bxt_max, dh)
    ch = int(np.argmax(np.abs(cfg.array.bxt)))
    ref = cfg.scene.ptg
    ks = np.arange(Nrx)
    beta = np.zeros(Nrx); dC = np.full((3, Nrx), np.nan)
    for k in ks:
        _f, bk, blo, bhi = subband_frequency_beam(cfg, k)
        beta[k] = np.degrees(bk)
        common = (tr.ptx, tr.prx[ch], tr.vtx, tr.vrx[ch], tr.ptx, tr.vtx,
                  cfg.prf, cfg.system.wl, cfg.ta, blo, bhi)
        R = getcoeff_beam(ptg_real, *common)
        C = getcoeff_beam(ref, *common)
        if R and C:
            dC[:, k] = (R[0] - C[0], R[1] - C[1], R[2] - C[2])
    return ks, beta, dC, ch, float(cfg.array.bxt[ch])


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def self_test():
    print("[self-test] reconstruct_subband(use_sata=False) vs sar.reconstruct")
    ok = True
    for Nrx in (2, 3, 4, 5):
        cfg = sar.CONFIG_FACTORIES["topo_dpca_rand_bxt20"](Nrx=Nrx, base_dir=".",
                                                           scene_name="topo_ramp")
        tr = sar.build_platform_tracks(cfg)
        s_ch = sar.generate_channels(cfg, tr)
        ref = sar.reconstruct(cfg, tr, s_ch.copy())
        fb = reconstruct_subband(cfg, tr, s_ch.copy(), use_sata=False)
        d = float(np.max(np.abs(ref - fb)))
        ok = ok and d < 1e-6
        print(f"    Nrx={Nrx}: max|diff| = {d:.2e}")
    print("    ->", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def _reconstruct_all(cfg, tr, verbose_sata=False, term_sets=None):
    """Reconstruct one scene with no-SATA, SATA-band, and one SATA-sub pass PER
    entry of `term_sets` (default: TERM_SETS -- C0 / C0+C1 / C0+C1+C2), and
    return per-method metrics + signals."""
    term_sets = term_sets if term_sets is not None else TERM_SETS
    sref1, sig_true, s_ch = channels_and_refs(cfg, tr)
    srec = {
        "no-SATA": sar.reconstruct(cfg, tr, s_ch.copy()),
        "SATA band": sar.reconstruct(cfg, tr, sata_channels(cfg, tr, s_ch.copy(),
                                                             verbose=verbose_sata)),
    }
    for terms in term_sets:
        srec[_term_label(terms)] = reconstruct_subband(
            cfg, tr, s_ch.copy(), use_sata=True, verbose=verbose_sata,
            correct_terms=terms)

    p = focus_mag(sig_true, sref1).max()
    out = {"cfg": cfg, "sref1": sref1, "sig_true": sig_true, "srec": srec}
    for name, sr in srec.items():
        f = focus_mag(sr, sref1)
        out[name] = {"peak_pct": 100 * f.max() / p, "ambig_db": ambiguity_db(f)}
    return out


def run_one(Nrx, bxt_max, verbose_sata=False, term_sets=None):
    """Azimuth-varying topography scene (ramp of targets spread in azimuth)."""
    cfg, tr = build_azimuth_topo(Nrx, bxt_max)
    return _reconstruct_all(cfg, tr, verbose_sata, term_sets=term_sets)


def run_one_single(Nrx, bxt_max, dh, verbose_sata=False, term_sets=None):
    """Single elevated iso-range target at height dh (clean ambiguity metric)."""
    cfg, tr, _ = build_single_target(Nrx, bxt_max, dh)
    return _reconstruct_all(cfg, tr, verbose_sata, term_sets=term_sets)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _mpl():
    """Import matplotlib with Agg + mathtext (no LaTeX). Returns plt or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["text.usetex"] = False        # respect USE_LATEX=0
        matplotlib.rcParams["axes.formatter.use_mathtext"] = True
        return plt
    except Exception as e:                                # pragma: no cover
        print("    (matplotlib unavailable, skipping plots)", e)
        return None


def plot_residual_per_subband(Nrx=4, bxt_max=100.0, dh=200.0):
    """delta_C0 / delta_C1 / delta_C2 vs sub-band k -- the terms available to the
    SATA-sub correction. dC0 is flat across sub-bands (the whole-band SATA
    already applies it, per-sub-band is a no-op there); dC1 is the one that is
    genuinely sub-band-dependent (antisymmetric in beta_k); dC2 is plotted for
    completeness but is expected to be weak in the broadside regime."""
    plt = _mpl()
    if plt is None:
        return
    ks, beta, dC, ch, bxt_ch = subband_residuals(Nrx, bxt_max, dh)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    labels = [r"$\delta C_0^{(k)}$ [m]", r"$\delta C_1^{(k)}$ [s]", r"$\delta C_2^{(k)}$ [s$^2$/m]"]
    colors = ["C0", "C2", "C4"]
    for i, ax in enumerate(axes):
        ax.plot(ks, dC[i], "o-", color=colors[i], lw=1.6, ms=7)
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.3)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.set_xlabel(r"sub-band $k$")
        ax.set_xticks(ks)
        ax.set_xticklabels([f"{k}\n{beta[k]:+.2f}$^\\circ$" for k in ks], fontsize=8)
    fig.suptitle(rf"Per-sub-band residuals (DPCA)  "
                 rf"($N_{{rx}}={Nrx}$, ch {ch}: $b_{{xt}}={bxt_ch:.1f}$ m, "
                 rf"$\Delta h={dh:.0f}$ m):  $C_0$ invariant, $C_1$ antisymmetric in "
                 rf"$\beta_k$, $C_2$ weak", y=1.05)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"residual_per_subband_Nrx{Nrx}_bxt{int(bxt_max)}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    plot -> {out}")


def plot_peak_vs_dxt(results, nrxs, bxts, term_sets=None):
    """Focused-peak recovery vs bxt_max, one panel per Nrx, all methods."""
    plt = _mpl()
    if plt is None or len(bxts) < 2:
        if plt is not None:
            print("    (peak_vs_dxt needs >=2 bxt_max values, skipping)")
        return
    methods = _method_order(term_sets)
    n = len(nrxs)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, Nrx in zip(axes, nrxs):
        for m in methods:
            y = [results[(Nrx, b)][m]["peak_pct"] for b in bxts]
            c, ls = _method_style(m, term_sets)
            ax.plot(bxts, y, "o", color=c, ls=ls, lw=1.4, ms=5, label=m)
        ax.set_title(rf"$N_{{rx}}={Nrx}$"); ax.set_xlabel(r"$b_{xt}^{\max}$ [m]")
        ax.grid(alpha=0.3); ax.set_ylim(0, 105)
    axes[0].set_ylabel("focused peak [% of ideal]")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Peak recovery vs cross-track baseline (DPCA, random bxt)", y=1.02)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "peak_vs_bxt.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    plot -> {out}")


def _mpl_imshow_annot(ax, M, rows, cols, fmt="{:+.1f}", cmap="RdBu_r", vlim=None):
    import numpy as _np
    v = vlim if vlim else float(_np.nanmax(_np.abs(M))) or 1.0
    im = ax.imshow(M, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not _np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=8, color="black")
    return im


def plot_method_bars(results, nrxs, bxts, term_sets=None):
    """Grouped bars: worst azimuth ambiguity [dB] for every method, per case."""
    plt = _mpl()
    if plt is None:
        return
    methods = _method_order(term_sets)
    cases = [(N, b) for N in nrxs for b in bxts]
    labels = [f"N{N}\n{int(b)}m" for (N, b) in cases]
    nmet = len(methods)
    x = np.arange(len(cases)); w = 0.8 / nmet
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(cases)), 4.2))
    for j, m in enumerate(methods):
        y = [results[c][m]["ambig_db"] for c in cases]
        c, _ls = _method_style(m, term_sets)
        ax.bar(x + (j - (nmet - 1) / 2) * w, y, w, color=c, label=m)
    ax.set_ylabel("worst azimuth ambiguity [dB]  (lower = better)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel(r"case  ($N_{rx}$, $b_{xt}^{\max}$)")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, ncol=nmet, loc="upper center")
    ax.set_title("Ambiguity suppression per method, across cases (DPCA)")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "cases_method_bars.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    plot -> {out}")


def plot_sub_vs_band_diff(results, nrxs, bxts, term_sets=None):
    """Heatmaps of (SATA-sub, most complete term set) - (SATA-band) over the
    (Nrx, bxt) grid, for both the ambiguity [dB] and the peak [%]. Positive
    ambiguity difference = sub worse; negative peak difference = sub worse."""
    plt = _mpl()
    if plt is None or len(nrxs) < 2 and len(bxts) < 2:
        return
    term_sets = term_sets if term_sets is not None else TERM_SETS
    sub_label = _term_label(term_sets[-1])
    dA = np.full((len(nrxs), len(bxts)), np.nan)
    dP = np.full((len(nrxs), len(bxts)), np.nan)
    for i, N in enumerate(nrxs):
        for j, b in enumerate(bxts):
            r = results[(N, b)]
            dA[i, j] = r[sub_label]["ambig_db"] - r["SATA band"]["ambig_db"]
            dP[i, j] = r[sub_label]["peak_pct"] - r["SATA band"]["peak_pct"]
    cols = [f"{int(b)}" for b in bxts]; rows = [f"{N}" for N in nrxs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    im0 = _mpl_imshow_annot(axes[0], dA, rows, cols, fmt="{:+.1f}")
    axes[0].set_title(r"$\Delta$ ambiguity: sub $-$ band [dB]")
    im1 = _mpl_imshow_annot(axes[1], dP, rows, cols, fmt="{:+.0f}")
    axes[1].set_title(r"$\Delta$ peak: sub $-$ band [\%]")
    for ax in axes:
        ax.set_xlabel(r"$b_{xt}^{\max}$ [m]"); ax.set_ylabel(r"$N_{rx}$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046); fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle(f"{sub_label} minus SATA band "
                 "(positive dB / negative % = sub worse)", y=1.03)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "cases_sub_vs_band_diff.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    plot -> {out}")


def plot_dh_sweep(Nrx=4, bxt_max=100.0, dhs=(50, 100, 200, 300, 400), term_sets=None):
    """Peak recovery and worst ambiguity vs target height dh, all methods
    (single elevated target)."""
    plt = _mpl()
    if plt is None:
        return
    methods = _method_order(term_sets)
    peak = {m: [] for m in methods}; amb = {m: [] for m in methods}
    for dh in dhs:
        r = run_one_single(Nrx, bxt_max, float(dh), term_sets=term_sets)
        for m in methods:
            peak[m].append(r[m]["peak_pct"]); amb[m].append(r[m]["ambig_db"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in methods:
        c, ls = _method_style(m, term_sets)
        ax[0].plot(dhs, peak[m], "o", ls=ls, color=c, label=m)
        ax[1].plot(dhs, amb[m], "o", ls=ls, color=c, label=m)
    ax[0].set_ylabel("focused peak [% of ideal]"); ax[0].set_ylim(0, 105)
    ax[1].set_ylabel("worst ambiguity [dB]")
    for a in ax:
        a.set_xlabel(r"target height $\Delta h$ [m]"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(rf"Effect of topography height ($N_{{rx}}={Nrx}$, "
                 rf"$b_{{xt}}^{{\max}}={bxt_max:.0f}$ m, single target)", y=1.03)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "cases_dh_sweep.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    plot -> {out}")


def _draw_combined(plt, cfg, res, method, tag=""):
    """Replica of sar_recon.plotting.plot_combined (2x2), saved to our PLOTS_DIR
    with mathtext. ref vs rec: amplitude, spectrum [dB], zoomed IRF, phase."""
    Nrx, prf, abw = cfg.Nrx, cfg.prf, cfg.abw
    dbat = cfg.array.bat[1] - cfg.array.bat[0] if Nrx > 1 else 0.0
    bxt_max_actual = float(np.max(np.abs(cfg.array.bxt))) if Nrx > 0 else 0.0
    ta = cfg.ta
    dph = np.angle(np.fft.fft(res.srecNF) * np.conj(np.fft.fft(res.srefF)), deg=True)
    dph[res.abw_idx] = 0
    tk = dict(fontsize=9, va="top",
              bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.85))

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(rf"Numerical Reconstruction ({method}, DPCA) | $N_{{rx}}={Nrx}$ | "
                 rf"$\mathrm{{PRF}}={prf:.1f}$ Hz | $B_a={abw:.1f}$ Hz | "
                 rf"$\Delta b_{{at}}={dbat:.1f}$ m | $b_{{xt}}^{{\max}}={bxt_max_actual:.1f}$ m (rand)")

    ax[0, 0].plot(ta, abs(res.sref), label="ref")
    ax[0, 0].plot(ta, abs(res.srecN), label="rec")
    ax[0, 0].set_xlabel("Time [s]"); ax[0, 0].set_ylabel("Amplitude")
    ax[0, 0].grid(); ax[0, 0].legend(fontsize="small", loc="best")
    ax[0, 0].text(0.02, 0.95, f"Nrx={Nrx}", transform=ax[0, 0].transAxes, **tk)

    ax[0, 1].plot(ta, 20 * np.log10(abs(res.srefF) / np.max(abs(res.srefF))), label="ref")
    ax[0, 1].plot(ta, 20 * np.log10(abs(res.srecNF) / np.max(abs(res.srecNF))), label="rec")
    ax[0, 1].set_xlabel("Time [s]"); ax[0, 1].set_ylabel("[dB]")
    ax[0, 1].set_ylim([-100, 0]); ax[0, 1].grid(); ax[0, 1].legend(fontsize="small", loc="best")
    ax[0, 1].text(0.02, 0.95, f"Nrx={Nrx}", transform=ax[0, 1].transAxes, **tk)

    ax[1, 0].plot(res.taz * 1e3,
                  20 * np.log10(abs(res.u_refFocC) / np.max(abs(res.u_refFocC))), label="ref")
    ax[1, 0].plot(res.taz * 1e3,
                  20 * np.log10(abs(res.u_interpFocCN) / np.max(abs(res.u_interpFocCN))), label="rec")
    ax[1, 0].set_xlabel("Time [ms]"); ax[1, 0].set_ylabel("[dB]")
    ax[1, 0].grid(); ax[1, 0].legend(fontsize="small", loc="best")
    ax[1, 0].text(0.02, 0.95, f"Nrx={Nrx}", transform=ax[1, 0].transAxes, **tk)

    ax[1, 1].plot(res.fa, dph)
    ax[1, 1].axvline(x=abw / 2, color="r", linestyle="-.")
    ax[1, 1].axvline(x=-abw / 2, color="r", linestyle="-.")
    ax[1, 1].set_xlabel("Doppler freq [Hz]"); ax[1, 1].set_ylabel("[deg]"); ax[1, 1].grid()
    ax[1, 1].text(0.02, 0.95, f"Nrx={Nrx}", transform=ax[1, 1].transAxes, **tk)

    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"combined_4panel_{method}_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    plot -> {out}")


def _build_recons(cfg, tr, s_ch, term_sets=None):
    """no-SATA / SATA-band / one SATA-sub(...) entry per term_sets combination,
    as a dict of zero-arg callables (lazy -- only computed if requested)."""
    term_sets = term_sets if term_sets is not None else TERM_SETS
    recons = {
        "no-SATA": lambda: sar.reconstruct(cfg, tr, s_ch.copy()),
        "SATA-band": lambda: sar.reconstruct(cfg, tr, sata_channels(cfg, tr, s_ch.copy(), verbose=False)),
    }
    for terms in term_sets:
        label = "SATA-sub (" + "+".join(terms) + ")"
        recons[label] = (lambda terms=terms: reconstruct_subband(
            cfg, tr, s_ch.copy(), use_sata=True, verbose=False, correct_terms=terms))
    return recons


def plot_standard_4panel(Nrx, bxt_max, dh=200.0, methods=None, term_sets=None):
    """The standard reconstruction 4-panel (as run_experiment.plot_combined), one
    per selected method. DPCA timing (dx=11 m -> PRF by DPCA) with a single
    elevated iso-range target and random cross-track baselines (uniform
    0..bxt_max), so the topographic azimuth ambiguities are visible. Files are
    tagged by (Nrx, bxt_max, dh) so multiple cases do not overwrite."""
    plt = _mpl()
    if plt is None:
        return
    cfg, tr, _ = build_single_target(Nrx, bxt_max, dh)
    sref1, _sig_true, s_ch = channels_and_refs(cfg, tr)   # ref = clean single point
    tag = f"Nrx{Nrx}_bxt{int(bxt_max)}_dh{int(dh)}"
    recons = _build_recons(cfg, tr, s_ch, term_sets)
    methods = methods if methods is not None else tuple(recons.keys())
    for name in methods:
        srecN = recons[name]()
        _draw_combined(plt, cfg, sar.analyze(cfg, sref1, srecN), name, tag=tag)


def plot_ramp_topright_combined(Nrx, bxt_max, S_samp, alpha_deg=_RAMP_ALPHA_DEG,
                                methods=None, term_sets=None):
    """Single-panel comparison for the 5-target iso-range RAMP scene: overlays
    just the 'spectrum [dB]' panel (top-right of the standard 4-panel) for all
    selected methods on one set of axes. Files tagged by (Nrx, bxt_max, S_samp,
    alpha)."""
    plt = _mpl()
    if plt is None:
        return
    cfg, tr, S_m, dh_max = build_ramp(Nrx, bxt_max, S_samp, alpha_deg)
    sref1, _sig, s_ch = channels_and_refs(cfg, tr)
    tag = f"ramp_Nrx{Nrx}_bxt{int(bxt_max)}_S{S_samp}_a{alpha_deg:g}"

    recons = _build_recons(cfg, tr, s_ch, term_sets)
    methods = methods if methods is not None else tuple(recons.keys())
    colors = {"no-SATA": "C3", "SATA-band": "C1"}
    linestyles = {"no-SATA": "dashed", "SATA-band": "dashdot"}
    extra_ls = ["dotted", (0, (3, 1, 1, 1)), (0, (1, 1))]
    extra_colors = ["C0", "C2", "C4"]
    k = 0
    for name in methods:
        if name not in colors:
            colors[name] = extra_colors[k % len(extra_colors)]
            linestyles[name] = extra_ls[k % len(extra_ls)]
            k += 1
    ta = cfg.ta

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
    ref_plotted = False
    for name in methods:
        srecN = recons[name]()
        res = sar.analyze(cfg, sref1, srecN)
        if not ref_plotted:
            ax.plot(ta, 20 * np.log10(abs(res.srefF) / np.max(abs(res.srefF))),
                    color="k", lw=1.3, ls="solid", label="ref")
            ref_plotted = True
        ax.plot(ta, 20 * np.log10(abs(res.srecNF) / np.max(abs(res.srecNF))),
                color=colors[name], lw=1.3, ls=linestyles[name], label=name)

    ax.set_xlabel("Time [s]"); ax.set_ylabel("[dB]")
    ax.set_ylim([-100, 0]); ax.grid(alpha=0.3); ax.legend(fontsize="small", loc="best")
    ax.set_title(rf"Spectrum comparison, ramp (DPCA) | $N_{{rx}}={Nrx}$ | "
                 rf"$b_{{xt}}^{{\max}}={bxt_max:.0f}$ m | $S$={S_samp} samp "
                 rf"($S_m$={S_m:.0f} m) | $\alpha={alpha_deg:g}^\circ$ "
                 rf"($\Delta h_{{\max}}$={dh_max:.0f} m)")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"topright_spectrum_{tag}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    plot -> {out}")


def plot_ramp_topright_sidebyside(Nrx, bxt_max, S_samp, alpha_deg=_RAMP_ALPHA_DEG,
                                  methods=None, term_sets=None):
    """Same 'spectrum [dB]' panel as plot_ramp_topright_combined, but drawn as
    side-by-side subplots (one per method, ref vs rec). Files tagged by
    (Nrx, bxt_max, S_samp, alpha)."""
    plt = _mpl()
    if plt is None:
        return
    cfg, tr, S_m, dh_max = build_ramp(Nrx, bxt_max, S_samp, alpha_deg)
    sref1, _sig, s_ch = channels_and_refs(cfg, tr)
    tag = f"ramp_Nrx{Nrx}_bxt{int(bxt_max)}_S{S_samp}_a{alpha_deg:g}"

    recons = _build_recons(cfg, tr, s_ch, term_sets)
    methods = methods if methods is not None else tuple(recons.keys())
    palette = ["C3", "C1", "C0", "C2", "C4"]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(methods)}
    ta = cfg.ta

    fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.2),
                             sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, name in zip(axes, methods):
        srecN = recons[name]()
        res = sar.analyze(cfg, sref1, srecN)
        ax.plot(ta, 20 * np.log10(abs(res.srefF) / np.max(abs(res.srefF))),
                color="k", lw=1.3, ls="solid", label="ref")
        ax.plot(ta, 20 * np.log10(abs(res.srecNF) / np.max(abs(res.srecNF))),
                color=colors[name], lw=1.3, ls="dashed", label=name)
        ax.set_xlabel("Time [s]"); ax.set_ylim([-100, 0]); ax.grid(alpha=0.3)
        ax.legend(fontsize="small", loc="best")
        ax.set_title(name)
    axes[0].set_ylabel("[dB]")
    fig.suptitle(rf"Spectrum comparison, ramp (DPCA) | $N_{{rx}}={Nrx}$ | "
                 rf"$b_{{xt}}^{{\max}}={bxt_max:.0f}$ m | $S$={S_samp} samp "
                 rf"($S_m$={S_m:.0f} m) | $\alpha={alpha_deg:g}^\circ$ "
                 rf"($\Delta h_{{\max}}$={dh_max:.0f} m)", y=1.03)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"topright_spectrum_sidebyside_{tag}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    plot -> {out}")


def plot_sata_grid(Nrx, bxt_max, correct_terms=("C0",)):
    """The N x N 'Output of SATA' grid (the whiteboard figure, generalised).

    Rows = channels i, columns = output sub-bands k. Each panel shows the phase
    SATA imprinted on channel i to prepare sub-band k, Dphi = angle(X_i^{(k)} /
    x_i), as a function of azimuth sample, for the given `correct_terms`.
    Reading it:
      * down a column (fixed sub-band, varying channel): the correction tracks
        the channel's cross-track baseline (delta_C0_i propto B_perp,i);
      * across a row (fixed channel, varying sub-band): nearly identical when
        correct_terms=("C0",) (C0 is angle-invariant); adding "C1" breaks that
        symmetry (C1 is antisymmetric in beta_k), which is visible as a
        left-right tilt appearing across each row.
    For Nrx=2 this is exactly the 2x2 = 4-panel sketch.
    """
    plt = _mpl()
    if plt is None:
        return
    cfg, tr = build_azimuth_topo(Nrx, bxt_max)
    _, _, s_ch = channels_and_refs(cfg, tr)
    base = s_ch.astype(complex)

    # SATA output for every sub-band (all channels at once), for each k.
    corr_by_k = [sata_channels_subband(cfg, tr, base, k, verbose=False,
                                       correct_terms=correct_terms)
                 for k in range(Nrx)]

    naz = base.shape[1]
    x = np.arange(naz)
    fig, axes = plt.subplots(Nrx, Nrx, figsize=(2.7 * Nrx, 2.1 * Nrx),
                             sharex=True, sharey=True, squeeze=False)
    for i in range(Nrx):            # rows = channels
        thr = 0.03 * np.abs(base[i]).max()
        mask = np.abs(base[i]) > thr
        for k in range(Nrx):        # cols = sub-bands
            ax = axes[i][k]
            dphi = np.degrees(np.angle(corr_by_k[k][i] * np.conj(base[i])))
            ax.plot(x, np.where(mask, dphi, np.nan), lw=0.8, color="C0")
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(rf"sub-band {k}", fontsize=9)
            if k == 0:
                ax.set_ylabel(rf"ch {i}" + "\n" + r"$\Delta\varphi$ [deg]", fontsize=8)
            if i == Nrx - 1:
                ax.set_xlabel("azimuth sample", fontsize=8)
    terms_tag = "+".join(correct_terms)
    fig.suptitle(r"Output of SATA: channel $i$ data prepared for sub-band $k$   "
                 rf"(DPCA, $N_{{rx}}={Nrx}$, $b_{{xt}}^{{\max}}={bxt_max:.0f}$ m, "
                 rf"terms={terms_tag})", y=1.005)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR,
                       f"sata_grid_Nrx{Nrx}_bxt{int(bxt_max)}_{terms_tag}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    plot -> {out}")


def main():
    global _SEED, TERM_SETS
    ap = argparse.ArgumentParser(
        description="no-SATA vs whole-band vs sub-band SATA (DPCA mode, random bxt).")
    ap.add_argument("--nrx", type=int, default=None)
    ap.add_argument("--bxt", "--dxt", dest="bxt", type=float, default=None,
                    help="bxt_max: upper bound of the random cross-track baselines [m]")
    ap.add_argument("--alpha", type=float, default=None,
                    help="ramp inclination angle [deg]; default sweeps RAMP_ALPHA_SWEEP "
                         f"={RAMP_ALPHA_SWEEP}")
    ap.add_argument("--seed", type=int, default=_SEED, help="RNG seed for random bxt")
    ap.add_argument("--terms", type=str, default=None,
                    help="restrict every sweep/plot to ONE explicit SATA-sub term "
                         "combination, comma-separated, e.g. 'C0', 'C0,C1', "
                         "'C0,C1,C2'. Default: compare all three (TERM_SETS).")
    ap.add_argument("--no-plots", action="store_true", help="disable figure output")
    args = ap.parse_args()
    do_plots = not args.no_plots
    _SEED = args.seed
    if args.terms:
        TERM_SETS = [tuple(t.strip() for t in args.terms.split(","))]

    print(f"[mode] DPCA (dx={_DX_DPCA:.0f} m -> PRF by DPCA condition), "
          f"random bxt (uniform 0..bxt_max, seed={_SEED})")
    print(f"[terms] SATA-sub combinations compared: "
          f"{[('+'.join(t)) for t in TERM_SETS]}")
    self_test()

    nrxs = [args.nrx] if args.nrx else DEFAULT_NRX
    bxts = [args.bxt] if args.bxt else DEFAULT_BXT
    alphas = [args.alpha] if args.alpha else RAMP_ALPHA_SWEEP

    methods = _method_order()
    print("\n[sweep] azimuth-varying topography  (focused peak %% of ideal | worst ambiguity dB)")
    col_w = 20
    hdr = f"{'Nrx':>4}{'bxt_max':>8} | " + "".join(f"{m:>{col_w}}" for m in methods)
    print(hdr); print("-" * len(hdr))
    results = {}
    for Nrx in nrxs:
        for bxt in bxts:
            r = run_one(Nrx, bxt)
            results[(Nrx, bxt)] = r
            def cell(k): return f"{r[k]['peak_pct']:5.0f}% / {r[k]['ambig_db']:6.1f}"
            row = "".join(f"{cell(m):>{col_w}}" for m in methods)
            print(f"{Nrx:>4}{bxt:>8.0f} | {row}")

    if do_plots:
        print("\n[plots]")
        rep_nrx = args.nrx if args.nrx else 4
        rep_bxt = args.bxt if args.bxt else 50.0
        # (a) the terms available for correction: dC0/dC1/dC2 per sub-band
        plot_residual_per_subband(Nrx=rep_nrx, bxt_max=rep_bxt, dh=200.0)
        # (b) summary: peak recovery vs bxt_max per Nrx
        plot_peak_vs_dxt(results, nrxs, bxts)
        # --- case visualisations (the effect across different cases) -----------
        # (c) method comparison bars across all (Nrx, bxt) cases
        plot_method_bars(results, nrxs, bxts)
        # (d) sub - band difference heatmaps over the (Nrx, bxt) grid
        plot_sub_vs_band_diff(results, nrxs, bxts)
        # (e) effect of topography height dh (single target)
        plot_dh_sweep(Nrx=rep_nrx, bxt_max=rep_bxt)
        # ----------------------------------------------------------------------
        # (f) the standard reconstruction 4-panel, for several single-target cases
        if args.nrx or args.bxt:
            plot_standard_4panel(Nrx=rep_nrx, bxt_max=rep_bxt)
        else:
            for (N, b) in PANEL_CASES:
                print(f"    4-panel case Nrx={N}, bxt_max={b}")
                plot_standard_4panel(Nrx=N, bxt_max=float(b))
        # (g) single-panel spectrum comparison (no-SATA / SATA band / SATA sub x N)
        #     for the 5-target iso-range RAMP, swept over alpha, bxt, S_samp.
        print("    ramp spectrum comparisons (5 iso-range targets, alpha sweep "
              f"{alphas}):")
        for alpha in alphas:
            for b in (20, 50):
                for S in _RAMP_SSAMP:
                    plot_ramp_topright_combined(Nrx=rep_nrx, bxt_max=float(b),
                                                S_samp=S, alpha_deg=alpha)
                    plot_ramp_topright_sidebyside(Nrx=rep_nrx, bxt_max=float(b),
                                                  S_samp=S, alpha_deg=alpha)
        # (h) the N x N 'Output of SATA' grid per configuration, for the most
        #     complete term set requested
        for (Nrx, bxt) in results:
            plot_sata_grid(Nrx, bxt, correct_terms=TERM_SETS[-1])

    print("\nnote: DPCA mode (small along-track spacing, random cross-track "
          "baselines). 'SATA band' is the current one-pass-per-channel correction; "
          "'SATA sub (...)' is the per-sub-band model for the given term "
          "combination. The residual_per_subband figure shows the finding: dC0 "
          "is flat across sub-bands, dC1 is not (dC2 is weak).")


if __name__ == "__main__":
    main()