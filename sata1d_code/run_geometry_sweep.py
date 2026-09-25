"""
Geometry sweep: point targets on the iso-range surface, spaced in azimuth, with
different height profiles; plus the topographic ramp of the reference scenes
(targets at the same azimuth on a line dh = dy tan(alpha) through the scene centre). For every case: reference, no SATA, SATA whole band and
SATA per sub-band (new delta_C0 map), plus both SATA versions with the previous
("hold") map for comparison.

Outputs (default ./geometry_sweep/):
    bxt<..>/<family>/<id>_<slug>.png   one figure per case
    all_cases_bxt<..>.pdf              all figures of one bxt setting
    sweep_results.csv                  metrics of every case
    summary_new_vs_hold.png            new map vs previous map
Run:
    python run_geometry_sweep.py                  # full sweep
    python run_geometry_sweep.py --quick          # a few cases
    python run_geometry_sweep.py --families ramp cliff --workers 4
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sata1d as S

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
TOPO_ALPHAS_DEG = ALPHAS_DEG
TOPO_N_TARGETS = N_TARGETS
TOPO_LENGTH = 2000.0     # ground-range extent of the topographic ramp [m]
SPACING = 200.0          # azimuth spacing between targets [m]
BXT_MAX = 100.0          # bxt ~ U(0, BXT_MAX)
BXT_SEED = 0
SATA_OSF = 4
AMB_MASK = 40            # samples excluded around each target for the ambiguity metric
PEAK_WIN = 3

FAMILIES = ["ramp", "zigzag", "piecewise", "steepening", "flattening",
            "cliff", "spike", "mesa", "random_heights", "topo_ramp"]
LABEL = {"ramp": "Uniform ramp", "zigzag": "Zig-zag ramp", "piecewise": "Piecewise ramp",
         "steepening": "Steepening ramp", "flattening": "Flattening ramp", "cliff": "Cliff",
         "spike": "Spike", "mesa": "Mesa", "random_heights": "Random heights",
         "topo_ramp": "Topographic ramp"}
PREFIX = {"ramp": "RA", "zigzag": "ZI", "piecewise": "PW", "steepening": "ST", "flattening": "FL",
          "cliff": "CL", "spike": "SK", "mesa": "ME", "random_heights": "RH",
          "topo_ramp": "TR"}


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
        for N in TOPO_N_TARGETS:
            for a in TOPO_ALPHAS_DEG:
                dy = np.linspace(0.10, 1.0, N) * TOPO_LENGTH
                off = np.stack([np.zeros(N), dy, dy * np.tan(np.radians(a))], axis=1)
                cases.append(Case("topo_ramp", f"N{N}_a{a:g}",
                                  rf"$N={N}$ targets at the same azimuth, $\Delta h=\Delta y\tan{a:g}^\circ$, "
                                  rf"$\Delta y$ up to {TOPO_LENGTH:g} m", off[:, 2].copy(), spacing,
                                  dict(alpha=a), off))
    return cases


def make_cfg(case, bxt_max):
    if case.offsets is None:
        return S.make_config(list(zip(case.xs, case.heights)), bxt_max=bxt_max, seed=BXT_SEED)
    base = S.make_config([], bxt_max=bxt_max, seed=BXT_SEED)
    scene = replace(base.scene, extra_offsets=tuple(tuple(float(v) for v in o) for o in case.offsets))
    return replace(base, scene=scene)


def oracle_channels(cfg, tr):
    """Each scatterer corrected with its own delta_C0 before the channels are summed."""
    out = 0
    for p in cfg.scene.points[1:]:
        ch = S.generate_channels(cfg, tr, ptgs=[p])
        d = np.array([S.residual_C0(cfg, tr, p, i) for i in range(cfg.Nrx)])
        out = out + ch * np.exp(2j * np.pi * d / cfg.system.wl)[:, None]
    return out


def rangeline_channels(cfg, tr):
    """Topographic ramp: every scatterer sits in its own range bin, so each is SATA-corrected
    on its own line (map built from that scatterer only) before the lines are summed."""
    out = 0
    for o in cfg.scene.extra_offsets:
        c1 = replace(cfg, scene=replace(cfg.scene, extra_offsets=(o,)))
        out = out + S.sata_channels(c1, tr, S.generate_channels(c1, tr), sata_osf=SATA_OSF)
    return out


def simulate(case, bxt_max):
    cfg = make_cfg(case, bxt_max)
    tr = S.build_platform_tracks(cfg)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    sig = S.generate_reference(cfg, tr)
    ch = S.generate_channels(cfg, tr)
    rec = {"none": S.reconstruct(cfg, tr, ch), "oracle": S.reconstruct(cfg, tr, oracle_channels(cfg, tr))}
    if case.offsets is not None:
        rec["rangeline"] = S.reconstruct(cfg, tr, rangeline_channels(cfg, tr))
    for mode, tag in (("footprint", ""), ("hold", "_hold")):
        rec["whole" + tag] = S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=SATA_OSF, mode=mode))
        rec["sub" + tag] = S.reconstruct_subband(cfg, tr, ch, sata_osf=SATA_OSF, mode=mode)
    F = lambda x: np.roll(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(sref))), cfg.Na // 2)
    return cfg, {"ref": F(sig), **{k: F(v) for k, v in rec.items()}}


def metrics(cfg, foc, case):
    Na, ds = cfg.Na, cfg.system.vs / cfg.prf
    pk = np.abs(foc["ref"]).max()
    idx = Na // 2 + np.round(case.xs / ds).astype(int)
    mask = np.ones(Na, bool)
    for i in idx:
        mask[max(0, i - AMB_MASK):i + AMB_MASK + 1] = False
    out, tpk = {}, {}
    for k, v in foc.items():
        a = np.abs(v)
        out[f"amb_db_{k}"] = 20 * np.log10(a[mask].max() / pk)
        tpk[k] = np.array([a[max(0, i - PEAK_WIN):i + PEAK_WIN + 1].max() for i in idx])
    for k in ("whole", "sub"):
        loss = 20 * np.log10(tpk[k] / tpk["ref"])
        out[f"peak_loss_mean_db_{k}"], out[f"peak_loss_worst_db_{k}"] = float(loss.mean()), float(loss.min())
    return out


def make_figure(case, cfg, foc, m, bxt_max, case_id, out_png):
    pk = np.abs(foc["ref"]).max()
    db = {k: 20 * np.log10(np.abs(v) / pk + 1e-12) for k, v in foc.items()}
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    bxt_txt = (rf"$b_{{xt}}\sim U(0,{bxt_max:g})$ m = [" + ", ".join(f"{b:.1f}" for b in cfg.array.bxt) + "] m"
               if bxt_max > 0 else r"$b_{xt}=0$")
    fig = plt.figure(figsize=(17, 8.6))
    gs = GridSpec(2, 2, width_ratios=[1.75, 1.0], hspace=0.34, wspace=0.16, figure=fig)
    spacing_txt = "" if case.offsets is not None else f", spacing {case.spacing:g} m"
    fig.suptitle(rf"{case_id} | {LABEL[case.family]}: {case.desc}{spacing_txt}" + "\n"
                 + bxt_txt + rf" | $N_{{rx}}={cfg.Nrx}$ | PRF$={cfg.prf:.0f}$ Hz | $B_a={cfg.abw:.0f}$ Hz | "
                 rf"$\Delta b_{{at}}={cfg.array.bat[1]:.2f}$ m (DPCA)", fontsize=11.5, y=0.995)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["whole"], color="tab:blue", lw=1.6, ls=(0, (5, 3)), label="SATA whole band")
    ax.plot(t, db["sub"], color="tab:orange", lw=1.8, ls=(0, (1, 2)), label="SATA per sub-band")
    ax.set_title(f"IRF | worst ambiguity: ref {m['amb_db_ref']:.1f} dB, SATA whole {m['amb_db_whole']:.1f} dB, "
                 f"SATA sub {m['amb_db_sub']:.1f} dB\n(previous map: whole {m['amb_db_whole_hold']:.1f} dB, "
                 f"sub {m['amb_db_sub_hold']:.1f} dB | oracle {m['amb_db_oracle']:.1f} dB"
                 + (f" | SATA per range line {m['amb_db_rangeline']:.1f} dB" if "amb_db_rangeline" in m else "")
                 + ")", fontsize=9.5)
    az = fig.add_subplot(gs[1, 0])
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"IRF | worst ambiguity: ref {m['amb_db_ref']:.1f} dB, no SATA {m['amb_db_none']:.1f} dB",
                 fontsize=9.5)
    for a in (ax, az):
        a.set_xlim(t[0], t[-1]); a.set_ylim(-90, 5); a.grid(alpha=0.3)
        a.set_xlabel("Azimuth time [s]"); a.set_ylabel("[dB]"); a.legend(fontsize=10, loc="upper right")
    ag = fig.add_subplot(gs[:, 1])
    topo = case.offsets is not None
    xs, hs = (case.offsets[:, 1], case.heights) if topo else (case.xs, case.heights)
    step = np.min(np.diff(xs)) if topo else case.spacing
    lo, span = min(0.0, hs.min()), np.ptp(hs)
    ag.fill_between(xs, lo - 0.05 * (span + 1), hs, color="#D7C9A7", alpha=0.45, lw=0)
    ag.plot(xs, hs, "-", color="#6D4C2F", lw=1.6)
    ag.scatter(xs, hs, s=55, color="#1F4E79", edgecolor="k", zorder=3, label="point targets")
    ag.axhline(0.0, color="k", ls="--", lw=0.9, label=r"reconstruction reference $h_0=0$")
    for xi, hi in zip(xs, hs):
        ag.annotate(f"{hi:.0f} m", (xi, hi), textcoords="offset points", xytext=(7, -4), fontsize=8, color="#1F4E79")
    pad = 0.25 * span if span > 0 else 10.0
    ag.set_ylim(lo - pad, hs.max() + pad); ag.set_xlim(min(0.0, xs.min()) - 1.5 * step, xs.max() + 1.5 * step)
    ag.set_xlabel("Ground-range offset $\\Delta y$ [m] (all targets at azimuth 0)" if topo else "Azimuth position [m]"); ag.set_ylabel("Height above reference [m]")
    ag.grid(alpha=0.3); ag.legend(fontsize=8, loc="upper left")
    slope = case.params["alpha"] if topo else np.max(np.abs(case.slopes_deg))
    ag.set_title(f"Target geometry | $\\Delta h_{{max}}$={span:.0f} m, steepest |slope| {slope:.3g}°", fontsize=9.5)
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _job(args):
    case, bxt_max, case_id, out_png = args
    cfg, foc = simulate(case, bxt_max)
    m = metrics(cfg, foc, case)
    make_figure(case, cfg, foc, m, bxt_max, case_id, out_png)
    row = {"case_id": case_id, "family": case.family, "slug": case.slug, "bxt_max": bxt_max, "N": case.N,
           "dh_span": float(np.ptp(case.heights)), "max_jump": float(np.max(np.abs(np.diff(case.heights)))),
           "max_abs_slope": float(np.max(np.abs(case.slopes_deg))),
           "heights": " ".join(f"{h:.2f}" for h in case.heights), "png": out_png}
    row.update({k: round(float(v), 3) for k, v in m.items()})
    return row


def summary(rows, out_png):
    R = [r for r in rows if r["bxt_max"] > 0]
    if not R:
        return
    fams = [f for f in FAMILIES if any(r["family"] == f for r in R)]
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, (kn, ko, lab) in zip(axs[:2], (("amb_db_whole", "amb_db_whole_hold", "SATA whole band"),
                                         ("amb_db_sub", "amb_db_sub_hold", "SATA per sub-band"))):
        for f in fams:
            ax.scatter([r[ko] for r in R if r["family"] == f], [r[kn] for r in R if r["family"] == f], s=16, label=f)
        ax.plot((-60, 0), (-60, 0), "k--", lw=.8); ax.set_xlim(-60, 0); ax.set_ylim(-60, 0)
        ax.set_xlabel("worst ambiguity, previous map [dB]"); ax.set_ylabel("worst ambiguity, new map [dB]")
        ax.set_title(f"{lab} (below the diagonal = new map better)"); ax.grid(alpha=.3)
    axs[0].legend(fontsize=8, ncol=2)
    keys = (("amb_db_ref", "reference"), ("amb_db_none", "no SATA"), ("amb_db_whole_hold", "whole, previous"),
            ("amb_db_whole", "whole, new"), ("amb_db_sub_hold", "sub, previous"), ("amb_db_sub", "sub, new"))
    for j, (k, lab) in enumerate(keys):
        med = [np.median([r[k] for r in R if r["family"] == f]) for f in fams]
        axs[2].bar(np.arange(len(fams)) + (j - 2.5) * 0.13, med, 0.13, label=lab)
    axs[2].set_xticks(range(len(fams)), fams, rotation=30, ha="right"); axs[2].invert_yaxis()
    axs[2].set_ylabel("median worst ambiguity [dB]"); axs[2].set_title("Per family (lower = better)")
    axs[2].legend(fontsize=8, ncol=3); axs[2].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)


def pngs_to_pdf(pngs, out_pdf):
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in pngs]
    if ims:
        ims[0].save(out_pdf, save_all=True, append_images=ims[1:], resolution=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--bxt", nargs="+", type=float, default=[BXT_MAX, 0.0])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="geometry_sweep")
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
    print(f"{len(jobs)} cases, {args.workers} workers -> {args.out}", flush=True)
    t0, rows = time.time(), []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap(_job, jobs, chunksize=2), 1):
                rows.append(row)
                if i % 20 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            rows.append(_job(j)); print(f"  {i}/{len(jobs)} {j[2]}", flush=True)
    with open(os.path.join(args.out, "sweep_results.csv"), "w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    summary(rows, os.path.join(args.out, "summary_new_vs_hold.png"))
    for bxt in args.bxt:
        pngs_to_pdf([r["png"] for r in rows if r["bxt_max"] == bxt],
                    os.path.join(args.out, f"all_cases_bxt{bxt:g}.pdf"))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
