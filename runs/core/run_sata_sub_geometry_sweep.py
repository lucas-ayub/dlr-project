#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Large geometry sweep for the 1-D per-sub-band SATA (explicit kernel).

For every scene in the sweep the script reconstructs the azimuth line with

    SATA whole band: whole-band SATA (sata_channels) + sar.reconstruct
    SATA sub-band  : reconstruct_subband_explicit (per-sub-band SATA, C0 term)

and writes ONE figure per case:

    +-------------------------------+-------------------+
    | IRF, full azimuth axis (dB)   |                   |
    |  ref / SATA whole / SATA sub  |  target geometry  |
    +-------------------------------+  (azimuth x h)    |
    | IRF, zoom on the targets [m]  |                   |
    +-------------------------------+-------------------+

with a title stating exactly which geometry the case is.

Sweep dimensions
----------------
* cross-track baselines : bxt ~ U(0, BXT_MAX) (seed fixed, same draw for every
                          case) AND bxt = 0 (no cross-track baseline)
* ramp inclination      : ALPHAS_DEG
* number of targets     : N_TARGETS
* terrain profile family (what happens BETWEEN two neighbouring targets):
    ramp           uniform slope alpha between every pair
    zigzag         slope alternates +alpha / -alpha (peaks and valleys)
    piecewise      every segment has its own slope, drawn from ALPHAS_DEG with
                   random sign (seeded)
    steepening     slope grows along the ramp, 0.1 deg -> alpha_max
    flattening     slope shrinks along the ramp, alpha_max -> 0.1 deg
    cliff          flat, one abrupt height jump dh between the two middle targets
    spike          flat, the middle target alone is raised by dh
    mesa           0 -> dh plateau -> 0 (two abrupt jumps)
    random_heights every target at an independent height U(0, h_max) (seeded)

All targets are point targets on the iso-range surface of the reference
(slant range r0), spaced SPACING metres in azimuth, the first one at h = 0 (the
reconstruction reference height). Array: Nrx = 4, PRF = 2000 Hz, along-track
baselines on the DPCA condition (dx = 2 vs / PRF), exactly as the sata2d studies.

Outputs (default runs/core/plots/sata_sub_geometry_sweep/):
    bxt100/<family>/<case_id>_<slug>.png     one figure per case
    bxt0/<family>/...                        idem, without cross-track baseline
    sweep_results.csv                        metrics of every case
    summary_ramp_heatmaps.png                uniform ramps: alpha x N maps
    all_cases_bxt100.pdf / all_cases_bxt0.pdf  every figure, one per page

Run
---
    python runs/core/run_sata_sub_geometry_sweep.py                 # full sweep
    python runs/core/run_sata_sub_geometry_sweep.py --quick         # few cases
    python runs/core/run_sata_sub_geometry_sweep.py --families ramp cliff --workers 4
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, "..", "..", "sar_reconstruction"))   # runs/core/ -> repo root
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import matplotlib  
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
from matplotlib.gridspec import GridSpec  

import sar_recon as sar  
from sar_recon.config import (SystemParams, Scene, ArrayGeometry,  
                              integration_time, build_time_axis)
from sar_recon.signal_model import getRawData1D  
from sar_recon.subband_sata_explicit import reconstruct_subband_explicit  
from sar_recon.sata import sata_channels  

# ===========================================================================
# SWEEP DEFINITION -- edit here
# ===========================================================================
ALPHAS_DEG = [0.1, 1, 1.5, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
N_TARGETS = [2, 3, 5, 7, 9, 13]          # uniform ramps
N_TARGETS_SHAPES = [5, 9]                # zigzag / steepening / cliff / ...
N_TARGETS_RANDOM = [5, 9, 13]            # piecewise / random_heights
ALPHA_MAX_CURVED = [10, 20, 30, 50]      # steepening / flattening
CLIFF_DH = [10, 50, 100, 250, 500, 1000, 2000]
SPIKE_DH = [100, 500, 1000, 2000]
MESA_DH = [100, 500, 1000]
RANDOM_HMAX = [100, 500, 1000]
PIECEWISE_SEEDS = [0, 1, 2, 3, 4]
RANDOM_HEIGHT_SEEDS = [0, 1]

SPACING = 200.0          # azimuth spacing between neighbouring targets [m]
NRX = 4
PRF = 2000.0             # full (reconstructed) PRF [Hz]
BXT_MAX = 100.0          # bxt ~ U(0, BXT_MAX) [m]
BXT_SEED = 0
SATA_OSF = 4
RDELAY = 0.0051115753    # reference range delay (r0 = 766.2 km)
AMB_MASK = 40            # samples excluded around each target for the ambiguity metric
PEAK_WIN = 3             # +/- samples searched for each target's peak

FAMILIES = ["ramp", "zigzag", "piecewise", "steepening", "flattening",
            "cliff", "spike", "mesa", "random_heights"]
FAMILY_LABEL = {
    "ramp": "Uniform ramp", "zigzag": "Zig-zag ramp (+/- alpha)",
    "piecewise": "Piecewise ramp (random slope per segment)",
    "steepening": "Steepening ramp", "flattening": "Flattening ramp",
    "cliff": "Cliff (abrupt height jump)", "spike": "Spike (one raised target)",
    "mesa": "Mesa (up-plateau-down)", "random_heights": "Random heights",
}
# ===========================================================================

C_REF, C_WHOLE, C_SUB = "tab:green", "tab:blue", "tab:orange"


@dataclass
class Case:
    family: str
    slug: str
    desc: str                    # readable geometry description (title)
    heights: np.ndarray          # [N] heights above the reference [m]
    spacing: float = SPACING
    params: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return len(self.heights)

    @property
    def xs(self) -> np.ndarray:
        return (np.arange(self.N) - (self.N - 1) / 2.0) * self.spacing

    @property
    def slopes_deg(self) -> np.ndarray:
        return np.degrees(np.arctan2(np.diff(self.heights), self.spacing))


def _fmt(a) -> str:
    return f"{a:g}"


def heights_from_slopes(slopes_deg, spacing=SPACING) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(spacing * np.tan(np.radians(slopes_deg)))))


def build_cases(families, spacing=SPACING) -> list[Case]:
    cases = []
    add = lambda fam, slug, desc, h, **p: cases.append(
        Case(fam, slug, desc, np.asarray(h, float), spacing, p))

    if "ramp" in families:
        for N in N_TARGETS:
            for a in ALPHAS_DEG:
                add("ramp", f"N{N}_a{_fmt(a)}",
                    rf"$N={N}$ targets, $\alpha={_fmt(a)}^\circ$",
                    heights_from_slopes([a] * (N - 1), spacing), N=N, alpha=a)
    if "zigzag" in families:
        for N in N_TARGETS_SHAPES:
            for a in ALPHAS_DEG:
                sl = [a if i % 2 == 0 else -a for i in range(N - 1)]
                add("zigzag", f"N{N}_a{_fmt(a)}",
                    rf"$N={N}$ targets, slopes $\pm{_fmt(a)}^\circ$ alternating",
                    heights_from_slopes(sl, spacing), N=N, alpha=a)
    if "piecewise" in families:
        for N in N_TARGETS_RANDOM:
            for sd in PIECEWISE_SEEDS:
                g = np.random.default_rng(1000 + sd)
                sl = g.choice(ALPHAS_DEG, N - 1) * g.choice([-1, 1], N - 1)
                sl_txt = ", ".join(f"{s:+g}" for s in sl)
                add("piecewise", f"N{N}_seed{sd}",
                    rf"$N={N}$ targets, segment slopes [{sl_txt}]$^\circ$ (seed {sd})",
                    heights_from_slopes(sl, spacing), N=N, seed=sd)
    for fam, rev in (("steepening", False), ("flattening", True)):
        if fam in families:
            for N in N_TARGETS_SHAPES:
                for am in ALPHA_MAX_CURVED:
                    sl = np.linspace(0.1, am, N - 1)
                    sl = sl[::-1] if rev else sl
                    a0, a1 = sl[0], sl[-1]
                    add(fam, f"N{N}_amax{am}",
                        rf"$N={N}$ targets, slope ${a0:.3g}^\circ \rightarrow {a1:.3g}^\circ$",
                        heights_from_slopes(sl, spacing), N=N, alpha_max=am)
    if "cliff" in families:
        for N in N_TARGETS_SHAPES:
            for dh in CLIFF_DH:
                h = np.zeros(N); h[N // 2:] = dh
                eq = np.degrees(np.arctan2(dh, spacing))
                add("cliff", f"N{N}_dh{dh}",
                    rf"$N={N}$ targets, jump $\Delta h={dh}$ m between targets "
                    rf"{N // 2} and {N // 2 + 1} (equiv. ${eq:.1f}^\circ$)",
                    h, N=N, dh=dh)
    if "spike" in families:
        for N in N_TARGETS_SHAPES:
            for dh in SPIKE_DH:
                h = np.zeros(N); h[N // 2] = dh
                add("spike", f"N{N}_dh{dh}",
                    rf"$N={N}$ targets, middle target raised by $\Delta h={dh}$ m",
                    h, N=N, dh=dh)
    if "mesa" in families:
        for N in N_TARGETS_SHAPES:
            for dh in MESA_DH:
                h = np.zeros(N); h[1:-1] = dh
                add("mesa", f"N{N}_dh{dh}",
                    rf"$N={N}$ targets, plateau $\Delta h={dh}$ m (edges at 0)",
                    h, N=N, dh=dh)
    if "random_heights" in families:
        for N in N_TARGETS_RANDOM:
            for hm in RANDOM_HMAX:
                for sd in RANDOM_HEIGHT_SEEDS:
                    g = np.random.default_rng(2000 + sd)
                    h = g.uniform(0.0, hm, N); h[0] = 0.0
                    add("random_heights", f"N{N}_hmax{hm}_seed{sd}",
                        rf"$N={N}$ targets, heights $\sim U(0,{hm})$ m (seed {sd})",
                        h, N=N, h_max=hm, seed=sd)
    return cases


# ---------------------------------------------------------------------------
# Simulation of one case
# ---------------------------------------------------------------------------
def build_cfg(case: Case, bxt_max: float, nrx=NRX, prf=PRF, seed=BXT_SEED):
    sysp = SystemParams()
    base = Scene(rDelay=RDELAY, c0=sysp.c0, h0=0.0)
    r0, H, y0 = base.r0, base.H, base.y0
    extra = tuple((float(x), float(np.sqrt(r0 ** 2 - (H - h) ** 2) - y0), float(h))
                  for x, h in zip(case.xs, case.heights))
    scene = Scene(rDelay=RDELAY, c0=sysp.c0, h0=0.0, extra_offsets=extra)
    dx = 2.0 * sysp.vs / prf                               # DPCA along-track step
    if bxt_max > 0:
        array = ArrayGeometry.linear(nrx, dx, 0.0, bxt_mode="random",
                                     bxt_max=bxt_max, rng=seed)
    else:
        array = ArrayGeometry(bat=dx * np.arange(nrx), bxt=np.zeros(nrx))
    Na, Nc, ta = build_time_axis(prf, nrx, 2.0 * integration_time(sysp, scene))
    cfg = sar.ExperimentConfig(name=case.slug, system=sysp, scene=scene, array=array,
                               prf=prf, PRF_op=prf / nrx, Na=Na, Na_ch=Nc, ta=ta,
                               plots_dir=None)
    return cfg, sar.build_platform_tracks(cfg)


def simulate(case: Case, bxt_max: float):
    cfg, tr = build_cfg(case, bxt_max)
    s, Nrx, prf, ta = cfg.system, cfg.Nrx, cfg.prf, cfg.ta
    raw = lambda ptgs, prx, vrx: getRawData1D(
        ptgs, tr.ptx, prx, tr.vtx, vrx, ta, cfg.sq_tx, cfg.sq_tx,
        cfg.theta_tx, cfg.theta_tx, s.wl, prf)
    ptgs = cfg.scene.points[1:]                            # the N targets
    sref1 = raw(cfg.scene.ptg[None, :], tr.ptx, tr.vtx)    # single-point matched filter
    sig = raw(ptgs, tr.ptx, tr.vtx)                        # ideal monostatic full-PRF
    s_ch = np.array([raw(ptgs, tr.prx[i], tr.vrx[i])[::Nrx] for i in range(Nrx)])
    with contextlib.redirect_stdout(io.StringIO()):
        rec_whole = sar.reconstruct(cfg, tr, sata_channels(cfg, tr, s_ch.copy(),
                                                           sata_osf=SATA_OSF))
        rec_sub = reconstruct_subband_explicit(cfg, tr, s_ch.copy(), use_sata=True,
                                               sata_osf=SATA_OSF, correct_terms=("C0",))
    Na = cfg.Na
    F = lambda x: np.roll(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(sref1))), Na // 2)
    return cfg, {"ref": F(sig), "whole": F(rec_whole), "sub": F(rec_sub)}


def metrics(cfg, foc, case: Case):
    Na, ds = cfg.Na, cfg.system.vs / cfg.prf
    pk = np.abs(foc["ref"]).max()
    idx = Na // 2 + np.round(case.xs / ds).astype(int)
    mask = np.ones(Na, bool)
    for i in idx:
        mask[max(0, i - AMB_MASK):i + AMB_MASK + 1] = False
    out = {}
    for k, v in foc.items():
        a = np.abs(v)
        out[f"amb_db_{k}"] = 20 * np.log10(a[mask].max() / pk)
        out[f"_tpk_{k}"] = np.array([a[max(0, i - PEAK_WIN):i + PEAK_WIN + 1].max()
                                     for i in idx])
    for k in ("whole", "sub"):
        loss = 20 * np.log10(out[f"_tpk_{k}"] / out["_tpk_ref"])
        out[f"peak_loss_mean_db_{k}"] = float(loss.mean())
        out[f"peak_loss_worst_db_{k}"] = float(loss.min())
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(case: Case, cfg, foc, m, bxt_max, case_id, out_png):
    Na, prf, vs = cfg.Na, cfg.prf, cfg.system.vs
    pk = np.abs(foc["ref"]).max()
    db = {k: 20 * np.log10(np.abs(v) / pk + 1e-12) for k, v in foc.items()}
    t = (np.arange(Na) - Na // 2) / prf
    x = t * vs
    dbat = cfg.array.bat[1] - cfg.array.bat[0]
    bxt_txt = (rf"$b_{{xt}}\sim U(0,{bxt_max:g})$ m = "
               + "[" + ", ".join(f"{b:.1f}" for b in cfg.array.bxt) + "] m"
               if bxt_max > 0 else r"no cross-track baseline ($b_{xt}=0$)")

    fig = plt.figure(figsize=(17, 8.6))
    gs = GridSpec(2, 2, width_ratios=[1.75, 1.0], height_ratios=[1, 1],
                  hspace=0.34, wspace=0.16, figure=fig)
    fig.suptitle(
        rf"{case_id} | {FAMILY_LABEL[case.family]}: {case.desc}, "
        rf"spacing {case.spacing:g} m" + "\n"
        + bxt_txt + rf" | $N_{{rx}}={cfg.Nrx}$ | PRF$={prf:.0f}$ Hz | "
        rf"$B_a={cfg.abw:.0f}$ Hz | $\Delta b_{{at}}={dbat:.2f}$ m (DPCA) | "
        r"SATA per sub-band (explicit kernel, $C_0$)",
        fontsize=11.5, y=0.995)

    # drawn in this order: reference underneath (thick solid), then whole band
    # (dashed) and per sub-band (dotted) on top so both stay visible when equal
    series = (("ref", "Reference", C_REF, 2.2, "-"),
              ("whole", "SATA whole band", C_WHOLE, 1.6, (0, (5, 3))),
              ("sub", "SATA per sub-band", C_SUB, 1.8, (0, (1, 2))))

    ax = fig.add_subplot(gs[0, 0])
    for k, lab, c, lw, ls in series:
        ax.plot(t, db[k], color=c, lw=lw, ls=ls, label=lab)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(-90, 5)
    ax.set_xlabel("Azimuth time [s]"); ax.set_ylabel("[dB]")
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="upper right", handlelength=3)
    ax.set_title(f"IRF, full azimuth axis | worst ambiguity: ref {m['amb_db_ref']:.1f} dB, "
                 f"SATA whole {m['amb_db_whole']:.1f} dB, SATA sub {m['amb_db_sub']:.1f} dB",
                 fontsize=9.5)

    xlo = case.xs.min() - 1.5 * case.spacing
    xhi = case.xs.max() + 1.5 * case.spacing
    az = fig.add_subplot(gs[1, 0])
    for k, lab, c, lw, ls in series:
        az.plot(x, db[k], color=c, lw=lw, ls=ls, label=lab)
    for xi in case.xs:
        az.axvline(xi, color="0.6", ls=":", lw=0.8)
    az.set_xlim(xlo, xhi); az.set_ylim(-60, 5)
    az.set_xlabel("Azimuth position [m]"); az.set_ylabel("[dB]")
    az.grid(alpha=0.3)
    az.set_title("IRF, zoom on the targets (dotted = target positions)\nper-target peak "
                 f"loss mean/worst: SATA whole {m['peak_loss_mean_db_whole']:.2f}/"
                 f"{m['peak_loss_worst_db_whole']:.2f} dB, SATA sub "
                 f"{m['peak_loss_mean_db_sub']:.2f}/{m['peak_loss_worst_db_sub']:.2f} dB",
                 fontsize=9)

    ag = fig.add_subplot(gs[:, 1])
    xs, hs = case.xs, case.heights
    lo = min(0.0, hs.min())
    ag.fill_between(xs, lo - 0.05 * (np.ptp(hs) + 1), hs, color="#D7C9A7", alpha=0.45, lw=0)
    ag.plot(xs, hs, "-", color="#6D4C2F", lw=1.6, zorder=2)
    ag.scatter(xs, hs, s=55, color="#1F4E79", edgecolor="k", zorder=3, label="point targets")
    ag.axhline(0.0, color="k", ls="--", lw=0.9, label=r"reconstruction reference $h_0=0$")
    span = np.ptp(hs)
    fs = 8 if case.N <= 9 else 6.5
    for i, sl in enumerate(case.slopes_deg):
        xm, hm = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (hs[i] + hs[i + 1])
        ag.annotate(f"{sl:+.3g}°", (xm, hm), textcoords="offset points", xytext=(-4, 7),
                    ha="right", va="bottom", fontsize=fs, color="#6D4C2F",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    for xi, hi in zip(xs, hs):
        ag.annotate(f"{hi:.0f} m", (xi, hi), textcoords="offset points", xytext=(7, -4),
                    ha="left", va="top", fontsize=fs, color="#1F4E79")
    pad = 0.25 * span if span > 0 else 10.0
    ag.set_ylim(lo - pad, hs.max() + pad)
    ag.set_xlim(xlo, xhi)
    ag.set_xlabel("Azimuth position [m]"); ag.set_ylabel("Height above reference [m]")
    ag.grid(alpha=0.3); ag.legend(fontsize=8, loc="upper left")
    jump = np.max(np.abs(np.diff(hs))) if case.N > 1 else 0.0
    ag.set_title(f"Target geometry | $\\Delta h_{{max}}$={span:.0f} m, largest jump "
                 f"{jump:.0f} m, steepest |slope| {np.max(np.abs(case.slopes_deg)):.3g}°",
                 fontsize=9.5)
    ag.text(0.99, 0.02, f"targets on the iso-range surface\n$r_0$ = {cfg.scene.r0/1e3:.1f} km",
            transform=ag.transAxes, ha="right", va="bottom", fontsize=8, color="0.3")

    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _job(args):
    case, bxt_max, case_id, out_png = args
    t0 = time.time()
    cfg, foc = simulate(case, bxt_max)
    m = metrics(cfg, foc, case)
    make_figure(case, cfg, foc, m, bxt_max, case_id, out_png)
    row = {"case_id": case_id, "family": case.family, "slug": case.slug,
           "bxt_max": bxt_max, "N": case.N, "spacing": case.spacing,
           "alpha": case.params.get("alpha", ""),
           "dh_span": float(np.ptp(case.heights)),
           "max_jump": float(np.max(np.abs(np.diff(case.heights)))),
           "max_abs_slope": float(np.max(np.abs(case.slopes_deg))),
           "heights": " ".join(f"{h:.2f}" for h in case.heights),
           "png": out_png, "runtime_s": round(time.time() - t0, 2)}
    row.update({k: round(float(v), 3) for k, v in m.items() if not k.startswith("_")})
    return row


def heatmaps(rows, out_png):
    r = [x for x in rows if x["family"] == "ramp" and x["bxt_max"] > 0]
    if not r:
        return
    Ns = sorted({x["N"] for x in r}); As = sorted({float(x["alpha"]) for x in r})
    get = lambda key: np.array([[next((x[key] for x in r if x["N"] == n
                                       and float(x["alpha"]) == a), np.nan)
                                 for a in As] for n in Ns], float)
    no, sub = get("amb_db_whole"), get("amb_db_sub")
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.2))
    for ax, M, ttl, cmap, vr in ((axs[0], no, "SATA whole band: worst ambiguity [dB]", "magma_r", (-45, 0)),
                                 (axs[1], sub, "SATA per sub-band: worst ambiguity [dB]", "magma_r", (-45, 0)),
                                 (axs[2], sub - no, "SATA sub - SATA whole [dB] (negative = sub better)",
                                  "RdBu_r", (-15, 15))):
        im = ax.imshow(M, cmap=cmap, vmin=vr[0], vmax=vr[1], aspect="auto")
        ax.set_xticks(range(len(As)), [f"{a:g}" for a in As])
        ax.set_yticks(range(len(Ns)), Ns)
        ax.set_xlabel(r"ramp inclination $\alpha$ [deg]"); ax.set_ylabel("N targets")
        ax.set_title(ttl, fontsize=10)
        for i in range(len(Ns)):
            for j in range(len(As)):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="k" if ax is axs[2] or M[i, j] > -25 else "w")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(rf"Uniform ramps, $b_{{xt}}\sim U(0,{r[0]['bxt_max']:g})$ m, "
                 f"spacing {r[0]['spacing']:g} m", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def pngs_to_pdf(pngs, out_pdf):
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow missing -- skipping", out_pdf, ")")
        return
    ims = [Image.open(p).convert("RGB") for p in pngs]
    if ims:
        ims[0].save(out_pdf, save_all=True, append_images=ims[1:], resolution=110)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--bxt", nargs="+", type=float, default=[BXT_MAX, 0.0],
                    help="bxt_max values; 0 = no cross-track baseline")
    ap.add_argument("--spacing", type=float, default=SPACING)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=os.path.join(_HERE, "plots", "sata_sub_geometry_sweep"))
    ap.add_argument("--quick", action="store_true", help="only a handful of cases")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args(argv)

    cases = build_cases(args.families, args.spacing)
    if args.quick:
        cases = cases[::max(1, len(cases) // 6)]
    prefix = {f: f[:2].upper() for f in FAMILIES}
    prefix.update({"flattening": "FL", "spike": "SK", "random_heights": "RH",
                   "steepening": "ST", "piecewise": "PW", "mesa": "ME"})
    jobs, counter = [], {}
    for bxt in args.bxt:
        tag = f"bxt{bxt:g}"
        for c in cases:
            counter[(tag, c.family)] = counter.get((tag, c.family), 0) + 1
            cid = f"{prefix[c.family]}-{counter[(tag, c.family)]:03d}"
            d = os.path.join(args.out, tag, c.family)
            os.makedirs(d, exist_ok=True)
            jobs.append((c, bxt, cid, os.path.join(d, f"{cid}_{c.slug}.png")))
    print(f"{len(cases)} geometries x {len(args.bxt)} bxt settings = {len(jobs)} cases, "
          f"{args.workers} workers -> {os.path.abspath(args.out)}")

    t0 = time.time(); rows = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap(_job, jobs, chunksize=2), 1):
                rows.append(row)
                if i % 20 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f} s)", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            rows.append(_job(j))
            print(f"  {i}/{len(jobs)} {j[2]} {j[0].slug}", flush=True)

    csv_path = os.path.join(args.out, "sweep_results.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    heatmaps(rows, os.path.join(args.out, "summary_ramp_heatmaps.png"))
    if not args.no_pdf:
        for bxt in args.bxt:
            pngs = [r["png"] for r in rows if r["bxt_max"] == bxt]
            pngs_to_pdf(pngs, os.path.join(args.out, f"all_cases_bxt{bxt:g}.pdf"))
    print(f"done in {time.time() - t0:.0f} s -> {csv_path}")


if __name__ == "__main__":
    main()