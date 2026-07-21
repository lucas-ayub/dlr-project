# -*- coding: utf-8 -*-
r"""
Exhaustive baseline stress tests for the 1D SATA topography correction.

Goal: visualise whether SATA 1D behaves correctly across *all sorts* of
baselines -- small, medium, large cross-track; along-track / DPCA spacings;
many channels; and non-uniform (arbitrary) arrays.

It reuses the reconstruction building blocks of run_sata.py:
    _recon_elevated(Nrx, dxt, dh, dx=100.0, bat_offset=0.0, array=None)
        -> (cfg, ref, srec_no, srec_sa, srec_ideal)
    _focus_mag, _ambiguity_db

For a single elevated iso-range target it reconstructs three ways
(no-SATA / +SATA / ideal) and reports, per configuration:
    peak recovery  = 100 * |peak(recon)| / |peak(ideal)|      [%]
    worst ambiguity level                                     [dB below peak]

Figures are written to  runs/core/plots/run_sata/baselines/  as png + pdf:
    base_xtrack_sweep      cross-track baseline sweep (peak + ambiguity)
    base_large_stress      large-baseline stress test (where it breaks)
    base_atrack_sweep      along-track / DPCA spacing sweep
    base_height_family     peak vs height for a family of baselines
    base_irf_gallery       IRF (dB) small-multiples across baseline types
    base_nonuniform        non-uniform / arbitrary arrays (bar chart)

Run:
    cd sar_reconstruction
    PYTHONPATH=. python ../runs/core/run_baselines.py
"""
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # so we can import run_sata as a module

import run_sata as R                       # reuse the reconstruction helpers
from sar_recon.config import ArrayGeometry
from sar_recon.sata import residual_C0

PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots", "run_sata", "baselines")


# ---------------------------------------------------------------------------
# small helpers on top of run_sata
# ---------------------------------------------------------------------------
def _metrics(Nrx, dxt, dh, dx=100.0, bat_offset=0.0, array=None):
    """Return (peak_recovery_no, peak_recovery_sa, amb_no, amb_sa) in %,%,dB,dB.

    peak recovery is normalised to the IDEAL reconstruction (filter told the
    true height), so 100 % means SATA is as good as knowing the topography."""
    cfg, ref, no, sa, ideal = R._recon_elevated(Nrx, dxt, dh, dx=dx,
                                                bat_offset=bat_offset, array=array)
    foc = lambda s: R._focus_mag(s, ref, cfg.prf, cfg.abw, taper=False)
    f_no, f_sa, f_id = foc(no), foc(sa), foc(ideal)
    pk_id = f_id.max()
    rec_no = 100.0 * f_no.max() / pk_id
    rec_sa = 100.0 * f_sa.max() / pk_id
    return rec_no, rec_sa, R._ambiguity_db(f_no), R._ambiguity_db(f_sa), (cfg, ref, no, sa, ideal, foc)


def _annot(ax, xs, ys, dy, color, fmt="{:.1f}", fs=5.5):
    """Print each point's value just above/below its marker (small font)."""
    for xi, yi in zip(xs, ys):
        ax.annotate(fmt.format(yi), (xi, yi), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=fs, color=color,
                    clip_on=True)


def _mpl():
    import matplotlib
    matplotlib.use("pgf")                       # real LaTeX text via pdflatex
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsmath,amssymb}",
        "axes.titlesize": 11,
        "font.size": 11,
    })
    return plt


def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOTS_DIR, f"{name}.{ext}"),
                    dpi=150, bbox_inches="tight")
    # also drop a png copy at the outputs root so the LaTeX report finds it
    root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    fig.savefig(os.path.join(root, f"{name}.png"), dpi=150, bbox_inches="tight")
    print(f"    saved {name}.png")


# ---------------------------------------------------------------------------
# 1. cross-track baseline sweep (fine)
# ---------------------------------------------------------------------------
def plot_xtrack_sweep(Nrx_list=(3, 4, 6), dh=200.0,
                      dxt_arr=np.linspace(0, 500, 21), save=True):
    """Peak recovery and worst ambiguity vs cross-track baseline b_xt, for a
    few channel counts. no-SATA degrades as b_xt grows (residual ~ dh*b_xt);
    +SATA stays flat near 100 % / low ambiguity."""
    print("\n[B1] cross-track baseline sweep")
    data = {}
    for Nrx in Nrx_list:
        rn, rs, an, as_ = [], [], [], []
        for dxt in dxt_arr:
            a, b, c, d, _ = _metrics(Nrx, float(dxt), dh)
            rn.append(a); rs.append(b); an.append(c); as_.append(d)
        data[Nrx] = (np.array(rn), np.array(rs), np.array(an), np.array(as_))
        print(f"    Nrx={Nrx}: no-SATA {rn[0]:.0f}->{rn[-1]:.0f}%  "
              f"+SATA {min(rs):.0f}-{max(rs):.0f}%")
    if not save:
        return data
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = {3: "#1f77b4", 4: "#2ca02c", 6: "#d62728"}
    for Nrx in Nrx_list:
        rn, rs, an, as_ = data[Nrx]
        c = cmap.get(Nrx, "#555")
        ax[0].plot(dxt_arr, rn, "--", color=c, lw=1.6,
                   label=f"no-SATA, Nrx={Nrx}")
        ax[0].plot(dxt_arr, rs, "-", color=c, lw=2.2,
                   label=f"+SATA, Nrx={Nrx}")
        ax[1].plot(dxt_arr, an, "--", color=c, lw=1.6)
        ax[1].plot(dxt_arr, as_, "-", color=c, lw=2.2)
    ax[0].axhline(100, color="k", lw=0.7, ls=":")
    ax[0].set_xlabel(r"cross-track baseline $b_{xt}$ [m]")
    ax[0].set_ylabel("peak recovery [\% of ideal]")
    ax[0].set_title("Peak recovery vs cross-track baseline")
    ax[0].set_ylim(0, 115); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7, ncol=1)
    ax[1].set_xlabel(r"cross-track baseline $b_{xt}$ [m]")
    ax[1].set_ylabel("worst ambiguity [dB]")
    ax[1].set_title("Ambiguity level vs cross-track baseline")
    ax[1].grid(alpha=0.3)
    ax[1].plot([], [], "k--", label="no-SATA"); ax[1].plot([], [], "k-", label="+SATA")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Cross-track baseline sweep  (single iso-range target, dh={dh:.0f} m)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "base_xtrack_sweep"); plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# 2. large-baseline stress test
# ---------------------------------------------------------------------------
def plot_large_stress(Nrx=4, dh=200.0,
                      dxt_arr=np.array([0, 50, 100, 200, 300, 400, 600, 800, 1000, 1400]),
                      save=True):
    """Push b_xt to large values to find where SATA / the reconstruction break.
    Plots peak recovery for no-SATA, +SATA and ideal; the point where +SATA and
    ideal *both* fall is a reconstruction limit (ill-conditioning), not a SATA
    limit."""
    print("\n[B2] large-baseline stress test")
    rn, rs, ri = [], [], []
    for dxt in dxt_arr:
        cfg, ref, no, sa, ideal = R._recon_elevated(Nrx, float(dxt), dh)
        foc = lambda s: R._focus_mag(s, ref, cfg.prf, cfg.abw, taper=False)
        pk_ref = foc(ref).max()
        rn.append(100 * foc(no).max() / pk_ref)
        rs.append(100 * foc(sa).max() / pk_ref)
        ri.append(100 * foc(ideal).max() / pk_ref)
        print(f"    b_xt={dxt:5.0f} m  no={rn[-1]:5.0f}%  +SATA={rs[-1]:5.0f}%  ideal={ri[-1]:5.0f}%")
    if not save:
        return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(dxt_arr, rn, "o--", color="#d62728", lw=1.8, label="no-SATA")
    ax.plot(dxt_arr, rs, "s-", color="#2ca02c", lw=2.4, label="+SATA")
    ax.plot(dxt_arr, ri, "^:", color="#555", lw=1.6, label="ideal (knows height)")
    _annot(ax, dxt_arr, rn, -10, "#d62728", fmt="{:.0f}")
    _annot(ax, dxt_arr, rs, 6, "#2ca02c", fmt="{:.1f}")
    ax.axhline(100, color="k", lw=0.7, ls=":")
    # shade the region where ideal itself drops (reconstruction ill-conditioned)
    ri = np.array(ri)
    bad = dxt_arr[ri < 90]
    if bad.size:
        ax.axvspan(bad.min(), dxt_arr.max(), color="orange", alpha=0.10)
        ax.text(bad.min(), 20, " reconstruction\n ill-conditioned",
                fontsize=8, color="#a15c00")
    ax.set_xlabel(r"cross-track baseline $b_{xt}$ [m]  (normal operating range is $\lesssim$ 300 m)")
    ax.set_ylabel("peak recovery [\% of reference]")
    ax.set_title(f"Large-baseline stress test  (Nrx={Nrx}, dh={dh:.0f} m)")
    ax.set_ylim(0, 120); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); _save(fig, "base_large_stress"); plt.close(fig)


# ---------------------------------------------------------------------------
# 3. along-track / DPCA spacing sweep
# ---------------------------------------------------------------------------
def plot_atrack_sweep(Nrx=4, dxt=100.0, dh=200.0,
                      dx_arr=np.array([8, 11, 15, 20, 30, 50, 80, 120, 160, 200]),
                      save=True):
    """Vary the along-track spacing dx (DPCA-type baselines) at fixed cross-track
    baseline. SATA is a phase correction and should be insensitive to dx; the
    reconstruction, however, becomes ill-conditioned for very large dx."""
    print("\n[B3] along-track / DPCA spacing sweep")
    rn, rs, ri = [], [], []
    for dx in dx_arr:
        cfg, ref, no, sa, ideal = R._recon_elevated(Nrx, dxt, dh, dx=float(dx))
        foc = lambda s: R._focus_mag(s, ref, cfg.prf, cfg.abw, taper=False)
        pk = foc(ideal).max()
        rn.append(100 * foc(no).max() / pk)
        rs.append(100 * foc(sa).max() / pk)
        ri.append(100 * foc(ideal).max() / pk)
        print(f"    dx={dx:5.0f} m  no={rn[-1]:5.0f}%  +SATA={rs[-1]:5.0f}%")
    if not save:
        return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(dx_arr, rn, "o--", color="#d62728", lw=1.8, label="no-SATA")
    ax.plot(dx_arr, rs, "s-", color="#2ca02c", lw=2.4, label="+SATA")
    _annot(ax, dx_arr, rn, -10, "#d62728", fmt="{:.0f}")
    _annot(ax, dx_arr, rs, 6, "#2ca02c", fmt="{:.1f}")
    ax.axhline(100, color="k", lw=0.7, ls=":")
    ax.axvline(11, color="#1f77b4", lw=1.0, ls="--")
    ax.text(11.5, 10, "DPCA\nspacing", fontsize=8, color="#1f77b4")
    ax.set_xlabel(r"along-track spacing $dx$ [m]")
    ax.set_ylabel("peak recovery [\% of ideal]")
    ax.set_title(f"Along-track (DPCA) spacing sweep  (Nrx={Nrx}, "
                 rf"$b_{{xt}}$={dxt:.0f} m, dh={dh:.0f} m)")
    ax.set_ylim(0, 120); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); _save(fig, "base_atrack_sweep"); plt.close(fig)


# ---------------------------------------------------------------------------
# 4. height x baseline family
# ---------------------------------------------------------------------------
def plot_height_family(Nrx=4, dxt_list=(50, 100, 200, 300),
                       dh_arr=np.linspace(0, 300, 13), save=True):
    """Peak recovery vs target height for several cross-track baselines.
    Without SATA each baseline gives a different fall-off (the residual scales
    as dh*b_xt, so bigger b_xt falls faster); with SATA every curve collapses
    onto 100 %. This is the clearest picture of the dh*b_xt scaling."""
    print("\n[B4] height x baseline family")
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    cols = ["#1f77b4", "#ff7f0e", "#9467bd", "#d62728"]
    for k, dxt in enumerate(dxt_list):
        rn, rs = [], []
        for dh in dh_arr:
            a, b, _, _, _ = _metrics(Nrx, float(dxt), float(dh))
            rn.append(a); rs.append(b)
        ax.plot(dh_arr, rn, "--", color=cols[k], lw=1.8,
                label=rf"no-SATA, $b_{{xt}}$={dxt} m")
        ax.plot(dh_arr, rs, "-", color=cols[k], lw=1.2, alpha=0.7)
        print(f"    b_xt={dxt}: no-SATA {rn[0]:.0f}->{rn[-1]:.0f}%  +SATA~{np.mean(rs):.0f}%")
    ax.plot([], [], "k-", lw=1.2, label="+SATA (all baselines)")
    ax.axhline(100, color="k", lw=0.7, ls=":")
    ax.set_xlabel(r"target height $\Delta h$ [m]")
    ax.set_ylabel("peak recovery [\% of ideal]")
    ax.set_title(f"Peak recovery vs height, family of cross-track baselines  (Nrx={Nrx})")
    ax.set_ylim(0, 115); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); _save(fig, "base_height_family"); plt.close(fig)


# ---------------------------------------------------------------------------
# 5. IRF gallery across baseline types
# ---------------------------------------------------------------------------
def plot_irf_gallery(dh=200.0, save=True):
    """Small-multiples of the focused IRF (dB) for a menu of baseline types.
    Each panel: reference / no-SATA / +SATA. Visual confirmation that SATA
    restores the IRF across every baseline kind."""
    print("\n[B5] IRF gallery across baseline types")
    # (title, kwargs for _recon_elevated)
    cases = [
        ("small  $b_{xt}$=30 m (Nrx=4)",      dict(Nrx=4, dxt=30.0,  dh=dh)),
        ("medium $b_{xt}$=150 m (Nrx=4)",     dict(Nrx=4, dxt=150.0, dh=dh)),
        ("large  $b_{xt}$=300 m (Nrx=4)",     dict(Nrx=4, dxt=300.0, dh=dh)),
        ("DPCA  dx=11 m (Nrx=4)",             dict(Nrx=4, dxt=100.0, dh=dh, dx=11.0)),
        ("many channels (Nrx=6)",             dict(Nrx=6, dxt=120.0, dh=dh)),
        ("non-uniform array (Nrx=4)",         dict(array=ArrayGeometry(
                                                    bat=np.array([0., 70., 190., 360.]),
                                                    bxt=np.array([-120., -30., 40., 130.])),
                                                Nrx=4, dxt=0.0, dh=dh)),
    ]
    plt = _mpl()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (title, kw) in zip(axes.flat, cases):
        cfg, ref, no, sa, ideal = R._recon_elevated(**kw)
        foc = lambda s: R._focus_mag(s, ref, cfg.prf, cfg.abw, taper=False)
        f_ref, f_no, f_sa = foc(ref), foc(no), foc(sa)
        p = f_ref.max()
        db = lambda f: 20 * np.log10(np.maximum(f / p, 1e-6))
        i0 = int(np.argmax(f_ref)); n = len(f_ref)
        x = (np.arange(n) - i0)
        w = 1800
        sl = slice(max(0, i0 - w), min(n, i0 + w))
        ax.plot(x[sl], db(f_ref)[sl], color="#1f77b4", lw=1.3, label="reference")
        ax.plot(x[sl], db(f_no)[sl], color="#d62728", lw=1.1, alpha=0.8, label="no-SATA")
        ax.plot(x[sl], db(f_sa)[sl], color="#2ca02c", lw=1.3, label="+SATA")
        rec = 100 * f_sa.max() / foc(ideal).max()
        ax.set_title(f"{title}\n+SATA peak {rec:.0f}\% of ideal", fontsize=9)
        ax.set_ylim(-60, 3); ax.grid(alpha=0.3)
        ax.set_xlabel("azimuth [samples]", fontsize=8)
        ax.set_ylabel("[dB]", fontsize=8)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Focused IRF across baseline types  (single iso-range target, dh={dh:.0f} m)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "base_irf_gallery"); plt.close(fig)


# ---------------------------------------------------------------------------
# 6. non-uniform / arbitrary arrays
# ---------------------------------------------------------------------------
def plot_nonuniform(dh=200.0, save=True):
    """Bar chart of peak recovery for a set of NON-uniform / arbitrary arrays
    (irregular along- and cross-track baselines). SATA should recover ~100 %
    regardless of the array layout, since it corrects each channel by its own
    geometry."""
    print("\n[B6] non-uniform / arbitrary arrays")
    arrays = [
        ("uniform ref\n(Nrx=4)",   ArrayGeometry.linear(4, 100.0, 100.0)),
        ("mild non-unif.",         ArrayGeometry(bat=np.array([0., 90., 210., 300.]),
                                                 bxt=np.array([-140., -40., 50., 150.]))),
        ("strong non-unif.",       ArrayGeometry(bat=np.array([0., 40., 260., 330.]),
                                                 bxt=np.array([-200., -20., 60., 240.]))),
        ("clustered",              ArrayGeometry(bat=np.array([0., 20., 40., 300.]),
                                                 bxt=np.array([-60., -30., 0., 260.]))),
        ("wide sparse",            ArrayGeometry(bat=np.array([0., 150., 320., 520.]),
                                                 bxt=np.array([-260., -80., 90., 280.]))),
        ("Nrx=5 irregular",        ArrayGeometry(bat=np.array([0., 60., 150., 250., 380.]),
                                                 bxt=np.array([-180., -70., 10., 90., 200.]))),
    ]
    labels, rec_no, rec_sa = [], [], []
    for name, arr in arrays:
        a, b, _, _, _ = _metrics(arr.Nrx, 0.0, dh, array=arr)
        labels.append(name); rec_no.append(a); rec_sa.append(b)
        print(f"    {name.splitlines()[0]:22s} no={a:5.0f}%  +SATA={b:5.0f}%")
    if not save:
        return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, np.clip(rec_no, 0, 140), w, color="#d62728", label="no-SATA")
    ax.bar(x + w/2, np.clip(rec_sa, 0, 140), w, color="#2ca02c", label="+SATA")
    for i, (a, b) in enumerate(zip(rec_no, rec_sa)):
        ax.text(i - w/2, min(a, 140) + 2, f"{a:.0f}", ha="center", fontsize=7)
        ax.text(i + w/2, min(b, 140) + 2, f"{b:.0f}", ha="center", fontsize=7)
    ax.axhline(100, color="k", lw=0.7, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("peak recovery [\% of ideal]")
    ax.set_ylim(0, 130)
    ax.set_title(f"Non-uniform / arbitrary arrays  (dh={dh:.0f} m)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); _save(fig, "base_nonuniform"); plt.close(fig)


# ---------------------------------------------------------------------------
# 7. phase-error analysis  (the root cause, since SATA is a phase correction)
# ---------------------------------------------------------------------------
def _band_residual_phase(sig, ref, prf, abw):
    """Residual spectral phase of `sig` relative to `ref`, within the processed
    Doppler band, as (fa_band [Hz], phase [rad] wrapped to [-pi,pi])."""
    S = np.fft.fft(sig) * np.conj(np.fft.fft(ref))
    n = len(S)
    fa = np.fft.fftfreq(n, d=1.0 / prf)
    order = np.argsort(fa)
    fa, S = fa[order], S[order]
    band = np.abs(fa) <= abw / 2.0
    return fa[band], np.angle(S[band])


def _circ_std(sig, ref, prf, abw):
    """Circular standard deviation [rad] of the residual band phase -- a bounded,
    wrap-safe measure of how non-flat the phase is (0 = perfectly focused)."""
    _, ph = _band_residual_phase(sig, ref, prf, abw)
    Rlen = np.abs(np.mean(np.exp(1j * ph)))
    return float(np.sqrt(-2.0 * np.log(max(Rlen, 1e-12))))


def _analytic_residual_phase(Nrx, dxt, dh):
    """Worst-channel analytic residual phase 2*pi/lambda*(C0(h)-C0(h0)) [rad]
    -- the quantity SATA is designed to cancel (from GetCoeffNu, the same model
    the reconstruction uses)."""
    cfg, tracks, off = R._build_single_target_cfg(Nrx, dxt, dh)
    ptg = cfg.scene.ptg + off
    vals = [abs(2.0 * np.pi / cfg.system.wl * residual_C0(cfg, tracks, ptg, k))
            for k in range(Nrx)]
    return max(vals)


def plot_phase_error(Nrx=4, save=True):
    """Phase-domain validation: SATA is a phase correction, so the most direct
    check is the residual spectral phase itself.
      (a) residual spectral phase vs Doppler, no-SATA vs +SATA (mild case);
      (b) analytic residual phase 2*pi/lambda*dC0 vs height, family of baselines
          -- the linear ~ dh*b_xt law SATA must cancel;
      (c) measured residual phase (circular std) vs height, no-SATA vs +SATA."""
    print("\n[B7] phase-error analysis (SATA vs no-SATA)")
    plt = _mpl()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    # (a) spectral phase for a mild case (small enough not to wrap)
    dh_a, dxt_a = 70.0, 60.0
    cfg, ref, no, sa, ideal = R._recon_elevated(Nrx, dxt_a, dh_a)
    fa_no, ph_no = _band_residual_phase(no, ref, cfg.prf, cfg.abw)
    fa_sa, ph_sa = _band_residual_phase(sa, ref, cfg.prf, cfg.abw)
    mild_no = _circ_std(no, ref, cfg.prf, cfg.abw)
    mild_sa = _circ_std(sa, ref, cfg.prf, cfg.abw)
    ph_no = np.unwrap(ph_no); ph_no -= ph_no.mean()
    ph_sa = np.unwrap(ph_sa); ph_sa -= ph_sa.mean()
    fghz = fa_no / 1e3
    ax[0].plot(fghz, ph_no, color="#d62728", lw=1.6, label="no-SATA")
    ax[0].plot(fa_sa / 1e3, ph_sa, color="#2ca02c", lw=1.8, label="+SATA")
    ax[0].axhline(0, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel("Doppler frequency [kHz]")
    ax[0].set_ylabel("residual phase [rad]")
    ax[0].set_title(f"(a) Residual spectral phase\n(mild case: $\\Delta h$={dh_a:.0f} m, "
                    rf"$b_{{xt}}$={dxt_a:.0f} m)")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)

    # (b) analytic residual phase vs height, family of cross-track baselines
    dh_arr = np.linspace(0, 300, 13)
    cols = ["#1f77b4", "#ff7f0e", "#9467bd", "#d62728"]
    for k, dxt in enumerate([50, 100, 200, 300]):
        y = [_analytic_residual_phase(Nrx, float(dxt), float(d)) for d in dh_arr]
        ax[1].plot(dh_arr, y, "-", color=cols[k], lw=1.8,
                   label=rf"$b_{{xt}}$={dxt} m")
    ax[1].axhline(2 * np.pi, color="grey", lw=0.8, ls="--")
    ax[1].text(5, 2 * np.pi + 0.2, r"$2\pi$ (wrap)", fontsize=8, color="grey")
    ax[1].set_xlabel(r"target height $\Delta h$ [m]")
    ax[1].set_ylabel(r"analytic residual phase $\frac{2\pi}{\lambda}\Delta C_0$ [rad]")
    ax[1].set_title("(b) Phase error SATA must cancel\n"
                    r"(analytic, $\propto\Delta h\,b_{xt}$)")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)

    # (c) measured residual phase (circular std) vs height, no-SATA vs +SATA
    rn, rs = [], []
    for d in dh_arr:
        cfg, ref, no, sa, ideal = R._recon_elevated(Nrx, 150.0, float(d))
        rn.append(_circ_std(no, ref, cfg.prf, cfg.abw))
        rs.append(_circ_std(sa, ref, cfg.prf, cfg.abw))
    ax[2].plot(dh_arr, rn, "o-", color="#d62728", lw=1.8, ms=4, label="no-SATA")
    ax[2].plot(dh_arr, rs, "s-", color="#2ca02c", lw=2.0, ms=4, label="+SATA")
    _annot(ax[2], dh_arr, rn, 6, "#d62728", fmt="{:.2f}")
    _annot(ax[2], dh_arr, rs, 7, "#2ca02c", fmt="{:.2f}")
    ax[2].set_xlabel(r"target height $\Delta h$ [m]")
    ax[2].set_ylabel("measured residual phase [rad, circular std]")
    ax[2].set_title(r"(c) Phase error left after SATA"
                    "\n" rf"($N_{{rx}}$={Nrx}, $b_{{xt}}$=150 m)")
    ax[2].grid(alpha=0.3); ax[2].legend(fontsize=9)
    ax[2].set_ylim(bottom=0)

    print(f"    (a) mild case circ-std: no-SATA={mild_no:.2f} rad  +SATA={mild_sa:.2f} rad")
    print(f"    (c) circ-std no-SATA {rn[0]:.2f}->{max(rn):.2f} rad ; +SATA ~{np.mean(rs):.2f} rad")
    fig.suptitle("Phase-error analysis: SATA removes the topographic residual phase "
                 r"($\propto\Delta h\,b_{xt}$)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "base_phase_error"); plt.close(fig)


def _robust_peak_phase(sig, ref, prf, abw, mag_frac=0.2, pct=95.0):
    """Peak residual phase [rad] over the band, made robust: keep only bins whose
    magnitude exceeds `mag_frac` x band-peak (drops the low-energy, numerically
    noisy band edges), remove the circular-mean constant, and report the `pct`
    percentile of |phase| (not the raw max, to reject single-bin outliers)."""
    S = np.fft.fft(sig) * np.conj(np.fft.fft(ref))
    n = len(S)
    fa = np.fft.fftfreq(n, d=1.0 / prf)
    band = np.abs(fa) <= abw / 2.0
    Sb = S[band]
    mag = np.abs(Sb)
    keep = mag > mag_frac * mag.max()          # discard low-energy edge bins
    ph = np.angle(Sb[keep])
    # remove the constant (circular mean) -> keep only the focus-relevant spread
    mean_ang = np.angle(np.mean(np.exp(1j * ph)))
    dev = np.abs(np.angle(np.exp(1j * (ph - mean_ang))))
    return float(np.percentile(dev, pct))


def plot_phase_vs_baseline(Nrx=4, dxt_arr=np.linspace(0, 400, 11), save=True):
    """Max residual phase vs cross-track baseline (the b_xt axis complement of
    the height sweep). Left: analytic 2*pi/lambda*dC0 for a family of heights
    (linear ~ b_xt, no wrap). Right: robust peak measured residual phase,
    no-SATA vs +SATA, with band-edge bins masked out."""
    print("\n[B8] max residual phase vs baseline")
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

    # (left) analytic residual phase vs b_xt, family of heights
    cols = {100: "#1f77b4", 200: "#ff7f0e", 300: "#d62728"}
    for dh, c in cols.items():
        y = [_analytic_residual_phase(Nrx, float(b), float(dh)) for b in dxt_arr]
        ax[0].plot(dxt_arr, y, "-", color=c, lw=1.9, label=rf"$\Delta h$={dh} m")
    ax[0].axhline(2 * np.pi, color="grey", lw=0.8, ls="--")
    ax[0].text(5, 2 * np.pi + 0.3, r"$2\pi$ (wrap)", fontsize=8, color="grey")
    ax[0].set_xlabel(r"cross-track baseline $b_{xt}$ [m]")
    ax[0].set_ylabel(r"analytic residual phase $\frac{2\pi}{\lambda}\Delta C_0$ [rad]")
    ax[0].set_title(r"(a) Analytic residual phase vs baseline ($\propto b_{xt}$)")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    # (right) robust measured peak residual phase vs b_xt, no-SATA vs +SATA
    rn, rs = [], []
    for b in dxt_arr:
        cfg, ref, no, sa, ideal = R._recon_elevated(Nrx, float(b), 200.0)
        rn.append(_robust_peak_phase(no, ref, cfg.prf, cfg.abw))
        rs.append(_robust_peak_phase(sa, ref, cfg.prf, cfg.abw))
    ax[1].plot(dxt_arr, rn, "o-", color="#d62728", lw=1.8, ms=4, label="no-SATA")
    ax[1].plot(dxt_arr, rs, "s-", color="#2ca02c", lw=2.0, ms=4, label="+SATA")
    _annot(ax[1], dxt_arr, rn, 6, "#d62728", fmt="{:.2f}")
    _annot(ax[1], dxt_arr, rs, 7, "#2ca02c", fmt="{:.2f}")
    ax[1].axhline(np.pi, color="grey", lw=0.8, ls="--")
    ax[1].text(5, np.pi - 0.28, r"$\pi$ (wrap ceiling)", fontsize=8, color="grey")
    ax[1].set_xlabel(r"cross-track baseline $b_{xt}$ [m]")
    ax[1].set_ylabel("measured peak residual phase [rad]")
    ax[1].set_title(r"(b) Measured peak residual phase ($\Delta h$=200 m)"
                    "\n(edge bins masked, 95th pct)")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9); ax[1].set_ylim(0, np.pi + 0.3)

    print(f"    measured peak phase: no-SATA {rn[0]:.2f}->{max(rn):.2f} rad ; "
          f"+SATA ~{np.mean(rs):.2f} rad")
    fig.suptitle("Max residual phase vs baseline "
                 r"(analytic $\propto b_{xt}$; SATA holds it at the floor)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "base_phase_vs_baseline"); plt.close(fig)


def main():
    print("=" * 70)
    print("SATA 1D -- exhaustive baseline stress tests")
    print("=" * 70)
    plot_xtrack_sweep()
    plot_large_stress()
    plot_atrack_sweep()
    plot_height_family()
    plot_irf_gallery()
    plot_nonuniform()
    plot_phase_error()
    plot_phase_vs_baseline()
    print("\nAll baseline figures written to", PLOTS_DIR)


if __name__ == "__main__":
    main()