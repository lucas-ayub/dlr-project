"""
Geometry sweep with the 2-D processing (two-step reconstruction + SATA per range bin):
the same cases as the 1-D sweep. Iso-range targets spaced 200 m in azimuth with
different height profiles (ramps, zig-zag, cliffs, ...) and the cross-track ramps
(targets at the same azimuth, dh = dy tan(alpha), each in its own range bin).
For every case: reference, no SATA, SATA whole band, SATA per sub-band and the oracle
(each scatterer corrected with its own dC0 before the channels are summed).

Outputs (default ./geometry_sweep_2d/):
    bxt<..>/<family>/<id>_<slug>.png   one figure per case: azimuth IRFs, geometry in 2D and 3D,
                                       2-D IRF contour of the representative target and the phase
                                       of the reconstructed spectrum vs the reference along range
                                       frequency (f_a = 0) and along azimuth frequency (f_r = 0)
    bxt<..>/<family>/<id>_<slug>.npz   envelopes, contour and phase cuts of that case
    sweep_results.csv                  metrics of every case (same columns as the 1-D sweep)
Run from sar_reconstruction/:
    python -m sata2d.run_geometry_sweep_2d
    python -m sata2d.run_geometry_sweep_2d --quick
    python -m sata2d.run_geometry_sweep_2d --families ramp topo_ramp --workers 2
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)

from .arrays import make_params_dpca
from .geometry import (build_tracks_3d, generate_reference_3d, generate_channels_3d,
                       CoeffTable3D, residual_C0_3d)
from .reconstruction import (range_compress, reconstruct_subband_2d, azimuth_envelope,
                             worst_ambiguity, focus2d)
from .sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, coordinates_text
from .run_irf2d import zoom2d

ALPHAS_DEG = [0.1, 1, 1.5, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
N_TARGETS = [2, 3, 5, 7, 9, 13]
N_TARGETS_SHAPES = [5, 9]
N_TARGETS_RANDOM = [5, 9, 13]
ALPHA_MAX_CURVED = [10, 20, 30, 50]
CLIFF_DH = [10, 50, 100, 250, 500, 1000, 2000]
SPIKE_DH = [100, 500, 1000, 2000]
MESA_DH = [100, 500, 1000]
RANDOM_HMAX = [100, 500, 1000]
PIECEWISE_SEEDS = [0, 1, 2, 3, 4]
RANDOM_HEIGHT_SEEDS = [0, 1]
TOPO_LENGTH = 2000.0
SPACING = 200.0
BXT_MAX = 100.0
BXT_SEED = 0
SATA_OSF = 4
AMB_MASK = 40
PEAK_WIN = 3
R0_FRAC = 0.5            # reference range at the middle of the swath (cross-track ramps reach +-1.6 km)

FAMILIES = ["ramp", "zigzag", "piecewise", "steepening", "flattening",
            "cliff", "spike", "mesa", "random_heights", "topo_ramp"]
LABEL = {"ramp": "Along-track ramp", "zigzag": "Zig-zag ramp", "piecewise": "Piecewise ramp",
         "steepening": "Steepening ramp", "flattening": "Flattening ramp", "cliff": "Cliff",
         "spike": "Spike", "mesa": "Mesa", "random_heights": "Random heights",
         "topo_ramp": "Cross-track ramp"}
PREFIX = {"ramp": "RA", "zigzag": "ZI", "piecewise": "PW", "steepening": "ST", "flattening": "FL",
          "cliff": "CL", "spike": "SK", "mesa": "ME", "random_heights": "RH", "topo_ramp": "TR"}


@dataclass
class Case:
    family: str
    slug: str
    desc: str
    heights: np.ndarray
    spacing: float = SPACING
    params: dict = field(default_factory=dict)
    offsets: np.ndarray | None = None      # (N, 3) dx, dy, dh; None = iso-range targets spaced in azimuth

    @property
    def N(self):
        return len(self.heights)

    @property
    def xs(self):
        if self.offsets is not None:
            return self.offsets[:, 0]
        return (np.arange(self.N) - (self.N - 1) / 2.0) * self.spacing

    @property
    def slopes_deg(self):
        return np.degrees(np.arctan2(np.diff(self.heights), self.spacing))


def heights_from_slopes(slopes_deg, spacing=SPACING):
    return np.concatenate(([0.0], np.cumsum(spacing * np.tan(np.radians(slopes_deg)))))


def build_cases(families, spacing=SPACING):
    cases = []
    add = lambda fam, slug, desc, h, **p: cases.append(Case(fam, slug, desc, np.asarray(h, float), spacing, p))
    if "ramp" in families:
        for N in N_TARGETS:
            for a in ALPHAS_DEG:
                add("ramp", f"N{N}_a{a:g}", rf"$N={N}$, $\alpha={a:g}^\circ$",
                    heights_from_slopes([a] * (N - 1), spacing), alpha=a)
    if "zigzag" in families:
        for N in N_TARGETS_SHAPES:
            for a in ALPHAS_DEG:
                add("zigzag", f"N{N}_a{a:g}", rf"$N={N}$, slopes $\pm{a:g}^\circ$",
                    heights_from_slopes([a if i % 2 == 0 else -a for i in range(N - 1)], spacing), alpha=a)
    if "piecewise" in families:
        for N in N_TARGETS_RANDOM:
            for sd in PIECEWISE_SEEDS:
                g = np.random.default_rng(1000 + sd)
                sl = g.choice(ALPHAS_DEG, N - 1) * g.choice([-1, 1], N - 1)
                add("piecewise", f"N{N}_seed{sd}", rf"$N={N}$, random slopes (seed {sd})",
                    heights_from_slopes(sl, spacing))
    for fam, rev in (("steepening", False), ("flattening", True)):
        if fam in families:
            for N in N_TARGETS_SHAPES:
                for am in ALPHA_MAX_CURVED:
                    sl = np.linspace(0.1, am, N - 1)
                    sl = sl[::-1] if rev else sl
                    add(fam, f"N{N}_amax{am}", rf"$N={N}$, slope ${sl[0]:.3g}^\circ \to {sl[-1]:.3g}^\circ$",
                        heights_from_slopes(sl, spacing))
    if "cliff" in families:
        for N in N_TARGETS_SHAPES:
            for dh in CLIFF_DH:
                h = np.zeros(N); h[N // 2:] = dh
                add("cliff", f"N{N}_dh{dh}", rf"$N={N}$, jump $\Delta h={dh}$ m", h)
    if "spike" in families:
        for N in N_TARGETS_SHAPES:
            for dh in SPIKE_DH:
                h = np.zeros(N); h[N // 2] = dh
                add("spike", f"N{N}_dh{dh}", rf"$N={N}$, middle target raised by {dh} m", h)
    if "mesa" in families:
        for N in N_TARGETS_SHAPES:
            for dh in MESA_DH:
                h = np.zeros(N); h[1:-1] = dh
                add("mesa", f"N{N}_dh{dh}", rf"$N={N}$, plateau {dh} m", h)
    if "random_heights" in families:
        for N in N_TARGETS_RANDOM:
            for hm in RANDOM_HMAX:
                for sd in RANDOM_HEIGHT_SEEDS:
                    h = np.random.default_rng(2000 + sd).uniform(0.0, hm, N); h[0] = 0.0
                    add("random_heights", f"N{N}_hmax{hm}_seed{sd}",
                        rf"$N={N}$, heights $\sim U(0,{hm})$ m (seed {sd})", h)
    if "topo_ramp" in families:
        for N in N_TARGETS:
            for a in ALPHAS_DEG:
                dy = np.linspace(0.10, 1.0, N) * TOPO_LENGTH
                off = np.stack([np.zeros(N), dy, dy * np.tan(np.radians(a))], axis=1)
                cases.append(Case("topo_ramp", f"N{N}_a{a:g}",
                                  rf"$N={N}$ targets at the same azimuth, $\Delta h=\Delta y\tan{a:g}^\circ$, "
                                  rf"$\Delta y$ up to {TOPO_LENGTH:g} m", off[:, 2].copy(), spacing,
                                  dict(alpha=a), off))
    return cases


def make_params(case, bxt_max):
    specs = () if case.offsets is not None else tuple(zip(case.xs, case.heights))
    p = make_params_dpca(specs=specs, bxt_mode="random", bxt_max=bxt_max, seed=BXT_SEED, r0_frac=R0_FRAC)
    if case.offsets is not None:
        p.extra_offsets = tuple(tuple(float(v) for v in o) for o in case.offsets)
    return p


def oracle_channels(p, tr):
    out = 0
    for ptg in p.points:
        ch = range_compress(generate_channels_3d(p, tr, ptgs=ptg[None, :]), p.cd, p.rbw, p.rsf, axis=2)
        d = np.array([residual_C0_3d(p, tr, ptg, i) for i in range(p.Nrx)])
        out = out + ch * np.exp(2j * np.pi * d / p.wl)[:, None, None]
    return out


def simulate(case, bxt_max):
    p = make_params(case, bxt_max)
    tr = build_tracks_3d(p)
    ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)
    ch = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
    tab = CoeffTable3D(p, tr, n_nodes=16)
    rec = {"none": reconstruct_subband_2d(p, tr, ch, tab, use_sata=False, verbose=False),
           "whole": reconstruct_subband_2d(p, tr, ch, tab, use_sata="whole", sata_osf=SATA_OSF, verbose=False),
           "sub": reconstruct_subband_2d(p, tr, ch, tab, use_sata="subband", sata_osf=SATA_OSF, verbose=False),
           "oracle": reconstruct_subband_2d(p, tr, oracle_channels(p, tr), tab, use_sata=False, verbose=False)}
    env = {"ref": azimuth_envelope(p, tr, ref), **{k: azimuth_envelope(p, tr, v) for k, v in rec.items()}}
    return p, env, detail(p, tr, ref, rec, case)


def detail(p, tr, ref, rec, case, half_az=45.0, half_rg=150.0):
    """2-D IRF contour and spectral phase cuts of the representative target (the one the
    per-sub-band pass recovers worst): the images focused with that target's own 2-D
    matched filter, zoomed around it, and the phase of the reconstructed spectrum
    relative to the reference along f_r (at f_a = 0) and along f_a (at f_r = 0)."""
    ds = p.vs / p.prf
    rho_r = 299792458.0 / (2.0 * p.rsf)
    # representative target: worst peak of the per-sub-band result
    pk = []
    for ptg in p.points:
        i = p.Na // 2 + int(round((ptg[0] - p.x0) / ds))
        ref_one = range_compress(generate_reference_3d(p, tr, ptgs=ptg[None, :]), p.cd, p.rbw, p.rsf, axis=1)
        pk.append(np.abs(focus2d(rec["sub"], ref_one)[p.Na // 2, p.Nr // 2]) /
                  np.abs(focus2d(ref, ref_one)[p.Na // 2, p.Nr // 2]))
    j = int(np.argmin(pk))
    ptg = p.points[j]
    ref_one = range_compress(generate_reference_3d(p, tr, ptgs=ptg[None, :]), p.cd, p.rbw, p.rsf, axis=1)
    img = {"ref": focus2d(ref, ref_one), **{k: focus2d(v, ref_one) for k, v in rec.items()}}
    na, nr = int(np.ceil(half_az / ds)) + 2, int(np.ceil(half_rg / rho_r)) + 2
    zpa, zpr = 4, 8
    ia, ir = p.Na // 2, p.Nr // 2
    zoom = {k: zoom2d(v, ia, ir, na, nr, zpa, zpr) for k, v in img.items()}
    x_az = (np.arange(2 * na * zpa) - na * zpa) * ds / zpa
    x_rg = (np.arange(2 * nr * zpr) - nr * zpr) * rho_r / zpr
    pk_ref = np.abs(zoom["ref"]).max()
    contour = {k: (20 * np.log10(np.abs(v) / pk_ref + 1e-12)).astype(np.float32) for k, v in zoom.items()}
    # spectral phase cuts (2-D spectrum of the focused images = S_data * conj(S_target))
    S = {k: np.fft.fftshift(np.fft.fft2(v)) for k, v in img.items()}
    fa = np.fft.fftshift(np.fft.fftfreq(p.Na, 1.0 / p.prf))
    fr = np.fft.fftshift(np.fft.fftfreq(p.Nr, 1.0 / p.rsf))
    ia0, ir0 = int(np.argmin(np.abs(fa))), int(np.argmin(np.abs(fr)))
    amp_r = np.abs(S["ref"][ia0, :]); amp_a = np.abs(S["ref"][:, ir0])
    ok_r = amp_r > 0.1 * amp_r.max(); ok_a = (amp_a > 0.1 * amp_a.max()) & (np.abs(fa) <= p.abw / 2)
    phase = {}
    for k in ("none", "whole", "sub", "oracle"):
        pr = np.degrees(np.angle(S[k][ia0, :] * np.conj(S["ref"][ia0, :])))
        pa = np.degrees(np.angle(S[k][:, ir0] * np.conj(S["ref"][:, ir0])))
        phase[k] = (np.where(ok_r, pr, np.nan).astype(np.float32), np.where(ok_a, pa, np.nan).astype(np.float32))
    return dict(j=j, target=ptg.copy(), x_az=x_az.astype(np.float32), x_rg=x_rg.astype(np.float32),
                contour=contour, fr=fr.astype(np.float32), fa=fa.astype(np.float32), phase=phase,
                peak_pct={k: float(v) for k, v in zip(range(len(pk)), pk)})


def metrics(p, env, case):
    Na, ds = p.Na, p.vs / p.prf
    idx = Na // 2 + np.round(case.xs / ds).astype(int)
    out, tpk = {}, {}
    for k, v in env.items():
        out[f"amb_db_{k}"] = worst_ambiguity(p, v, env["ref"], case.xs, AMB_MASK)
        tpk[k] = np.array([v[max(0, i - PEAK_WIN):i + PEAK_WIN + 1].max() for i in idx])
    for k in ("whole", "sub"):
        loss = 20 * np.log10(tpk[k] / tpk["ref"])
        out[f"peak_loss_mean_db_{k}"], out[f"peak_loss_worst_db_{k}"] = float(loss.mean()), float(loss.min())
    return out


def make_figure(case, p, env, det, m, bxt_max, case_id, out_png):
    pk = env["ref"].max()
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    t = (np.arange(p.Na) - p.Na // 2) / p.prf
    bxt_txt = (rf"$b_{{xt}}\sim U(0,{bxt_max:g})$ m = [" + ", ".join(f"{b:.1f}" for b in p.bxt) + "] m"
               if bxt_max > 0 else r"$b_{xt}=0$")
    topo = case.offsets is not None
    xs, dys, hs = scene_coordinates(case.xs, case.heights, case.offsets)
    fig = plt.figure(figsize=(17, 14.5))
    gs = GridSpec(3, 3, width_ratios=[1.0, 0.75, 1.0], height_ratios=[1, 1, 1.05],
                  hspace=0.42, wspace=0.28, figure=fig)
    spacing_txt = "" if topo else f", spacing {case.spacing:g} m"
    fig.suptitle(rf"{case_id} | SATA 2D | {LABEL[case.family]}: {case.desc}{spacing_txt}" + "\n"
                 + bxt_txt + rf" | $N_{{rx}}={p.Nrx}$ | PRF$={p.prf:.0f}$ Hz | $B_a={p.abw:.0f}$ Hz | "
                 rf"$\Delta b_{{at}}={p.bat[1]:.2f}$ m (DPCA) | swath {p.r_scan[0]/1e3:.1f}..{p.r_scan[-1]/1e3:.1f} km, "
                 rf"$\rho_r$ = {299792458.0/2/p.rsf:.0f} m", fontsize=11.5, y=0.995)
    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["whole"], color="tab:blue", lw=1.6, ls=(0, (5, 3)), label="SATA whole band")
    ax.plot(t, db["sub"], color="tab:orange", lw=1.8, ls=(0, (1, 2)), label="SATA per sub-band")
    ax.set_title(f"Azimuth IRF (envelope over targets) | worst ambiguity: ref {m['amb_db_ref']:.1f} dB, "
                 f"SATA whole {m['amb_db_whole']:.1f} dB, SATA sub {m['amb_db_sub']:.1f} dB "
                 f"(oracle {m['amb_db_oracle']:.1f} dB)", fontsize=9.5)
    az = fig.add_subplot(gs[1, 0:2])
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"Azimuth IRF | worst ambiguity: ref {m['amb_db_ref']:.1f} dB, no SATA {m['amb_db_none']:.1f} dB",
                 fontsize=9.5)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=10, loc="upper right")
    slope = case.params["alpha"] if topo else np.max(np.abs(case.slopes_deg))
    span = np.ptp(hs)
    geometry_2d(fig.add_subplot(gs[0, 2]), xs, dys, hs, topo,
                title=f"Target geometry | $\\Delta h_{{max}}$={span:.0f} m, steepest |slope| {slope:.3g}\u00b0")
    a3 = fig.add_subplot(gs[1, 2], projection="3d")
    geometry_3d(a3, xs, dys, hs, topo, title="Target geometry in 3D (x, $\\Delta y$, h)")
    a3.text2D(0.99, 0.02, coordinates_text(xs, dys, hs), transform=a3.transAxes, fontsize=5.8,
              family="monospace", ha="right", va="bottom",
              bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    # --- representative target: 2-D IRF contour + spectral phase cuts -------
    tg = det["target"]; j = det["j"]
    tgt_txt = f"target {j + 1}/{len(xs)}: x = {xs[j]:.0f} m, $\\Delta y$ = {dys[j]:.0f} m, h = {hs[j]:.0f} m"
    ac = fig.add_subplot(gs[2, 0])
    Z = det["contour"]["sub"]
    levels = [-30, -20, -13, -10, -6, -3]
    cs = ac.contour(det["x_rg"], det["x_az"], Z, levels=levels, cmap="RdYlBu_r", linewidths=1.0)
    ac.clabel(cs, levels, fmt="%g", fontsize=6)
    ac.set_xlabel("range [m]"); ac.set_ylabel("azimuth [m]")
    ac.set_title(f"2-D IRF, SATA per sub-band [dB re. reference peak]\n{tgt_txt}", fontsize=9)
    ac.grid(alpha=0.3)
    ar = fig.add_subplot(gs[2, 1])
    aa = fig.add_subplot(gs[2, 2])
    for k, col, lab in (("none", "tab:red", "no SATA"), ("sub", "tab:orange", "SATA per sub-band"),
                        ("oracle", "tab:purple", "oracle")):
        pr, pa = det["phase"][k]
        ar.plot(det["fr"] / 1e6, pr, color=col, lw=0.9, label=lab)
        aa.plot(det["fa"] / 1e3, pa, color=col, lw=0.9, label=lab)
    ar.set_xlabel("range frequency [MHz]"); ar.set_ylabel("phase [deg]")
    ar.set_title("Phase of the reconstructed spectrum vs reference\nalong $f_r$ at $f_a = 0$", fontsize=9)
    aa.set_xlabel("azimuth frequency [kHz]"); aa.set_ylabel("phase [deg]")
    aa.set_title("Phase of the reconstructed spectrum vs reference\nalong $f_a$ at $f_r = 0$ (processed band)", fontsize=9)
    for a_ in (ar, aa):
        a_.set_ylim(-180, 180); a_.grid(alpha=0.3); a_.legend(fontsize=7.5, loc="upper right")
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _job(args):
    case, bxt_max, case_id, out_png = args
    t0 = time.time()
    p, env, det = simulate(case, bxt_max)
    m = metrics(p, env, case)
    make_figure(case, p, env, det, m, bxt_max, case_id, out_png)
    xs, dys, hs = scene_coordinates(case.xs, case.heights, case.offsets)
    np.savez_compressed(out_png[:-4] + ".npz", t=(np.arange(p.Na) - p.Na // 2) / p.prf,
                        **{f"env_{k}": v.astype(np.float32) for k, v in env.items()},
                        **{f"contour_{k}": v for k, v in det["contour"].items()},
                        **{f"phase_r_{k}": v[0] for k, v in det["phase"].items()},
                        **{f"phase_a_{k}": v[1] for k, v in det["phase"].items()},
                        x_az=det["x_az"], x_rg=det["x_rg"], fr=det["fr"], fa=det["fa"],
                        j=det["j"], target=det["target"], xs=xs, dys=dys, hs=hs, bxt=p.bxt,
                        metrics=np.array([m], dtype=object))
    row = {"case_id": case_id, "family": case.family, "slug": case.slug, "bxt_max": bxt_max, "N": case.N,
           "dh_span": float(np.ptp(case.heights)), "max_jump": float(np.max(np.abs(np.diff(case.heights)))),
           "max_abs_slope": float(np.max(np.abs(case.slopes_deg))),
           "heights": " ".join(f"{h:.2f}" for h in case.heights), "png": out_png}
    row.update({k: round(float(v), 3) for k, v in m.items()})
    print(f"{case_id} {case.family:15s} {case.slug:22s} ref {m['amb_db_ref']:6.1f} none {m['amb_db_none']:6.1f} "
          f"whole {m['amb_db_whole']:6.1f} sub {m['amb_db_sub']:6.1f} oracle {m['amb_db_oracle']:6.1f}  "
          f"[{time.time() - t0:.0f} s]", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--bxt", nargs="+", type=float, default=[BXT_MAX])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="geometry_sweep_2d")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cases = build_cases(args.families)
    if args.quick:
        cases = cases[::max(1, len(cases) // 6)]
    jobs, counter = [], {}
    for bxt in args.bxt:
        for c in cases:
            counter[(bxt, c.family)] = counter.get((bxt, c.family), 0) + 1
            cid = f"{PREFIX[c.family]}-{counter[(bxt, c.family)]:03d}"
            d = os.path.join(args.out, f"bxt{bxt:g}", c.family)
            os.makedirs(d, exist_ok=True)
            jobs.append((c, bxt, cid, os.path.join(d, f"{cid}_{c.slug}.png")))
    print(f"{len(jobs)} cases, {args.workers} workers", flush=True)
    t0 = time.time()
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            rows = pool.map(_job, jobs, chunksize=1)
    else:
        rows = [_job(j) for j in jobs]
    out_csv = os.path.join(args.out, "sweep_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} cases in {(time.time() - t0) / 60:.1f} min -> {out_csv}")


if __name__ == "__main__":
    main()
