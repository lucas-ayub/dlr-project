"""
Geometry sweep, SATA 1D: point targets on the iso-range surface, spaced 200 m in
azimuth, with different height profiles (along-track ramps, zig-zag, cliffs, ...);
plus the cross-track ramps (targets at the same azimuth, dh = dy tan(alpha)).
For every case: reference, no SATA, SATA whole band, SATA per sub-band and the
oracle (each scatterer corrected with its own delta_C0 before the channels are summed).

Outputs (default ./geometry_sweep/):
    bxt<..>/<family>/<id>_<slug>.png   one figure per case (IRFs, geometry in 2D and 3D)
    bxt<..>/<family>/<id>_<slug>.npz   the focused envelopes of that case
    all_cases_bxt<..>.pdf              all figures of one bxt setting
    sweep_results.csv                  metrics of every case
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
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, coordinates_text

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
LABEL = {"ramp": "Along-track ramp", "zigzag": "Zig-zag ramp", "piecewise": "Piecewise ramp",
         "steepening": "Steepening ramp", "flattening": "Flattening ramp", "cliff": "Cliff",
         "spike": "Spike", "mesa": "Mesa", "random_heights": "Random heights",
         "topo_ramp": "Cross-track ramp"}
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


def simulate(case, bxt_max):
    cfg = make_cfg(case, bxt_max)
    tr = S.build_platform_tracks(cfg)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    sig = S.generate_reference(cfg, tr)
    ch = S.generate_channels(cfg, tr)
    rec = {"none": S.reconstruct(cfg, tr, ch),
           "whole": S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=SATA_OSF)),
           "sub": S.reconstruct_subband(cfg, tr, ch, sata_osf=SATA_OSF),
           "oracle": S.reconstruct(cfg, tr, oracle_channels(cfg, tr))}
    F = lambda x: np.roll(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(sref))), cfg.Na // 2)
    return cfg, {"ref": F(sig), **{k: F(v) for k, v in rec.items()}}


def ambiguity_times(cfg):
    """Azimuth-time offset of the m-th azimuth ambiguity of a target: a Doppler shift of
    m PRF_op moves the focused copy by m PRF_op / K_a, K_a = 2 vs^2 / (wl r0)."""
    return cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)


def ambiguity_level(t, env, pk, xs, vs, dt_amb, orders=(-2, -1, 1, 2), win=0.02, guard=0.02):
    """Largest |IRF| inside +-win s of every ambiguity position x_j/vs + m dt_amb, in dB re. pk,
    leaving out +-guard s around every target's own main lobe."""
    lobe = np.zeros(len(t), bool)
    for x in xs:
        lobe |= np.abs(t - x / vs) <= guard
    best = 0.0
    for x in xs:
        for m in orders:
            sel = (np.abs(t - (x / vs + m * dt_amb)) <= win) & ~lobe
            if sel.any():
                best = max(best, float(env[sel].max()))
    return 20 * np.log10(best / pk + 1e-15)


def metrics(cfg, foc, case):
    Na, ds = cfg.Na, cfg.system.vs / cfg.prf
    pk = np.abs(foc["ref"]).max()
    idx = Na // 2 + np.round(case.xs / ds).astype(int)
    mask = np.ones(Na, bool)
    for i in idx:
        mask[max(0, i - AMB_MASK):i + AMB_MASK + 1] = False
    t = (np.arange(Na) - Na // 2) / cfg.prf
    dt_amb = ambiguity_times(cfg)
    out, tpk = {}, {}
    for k, v in foc.items():
        a = np.abs(v)
        out[f"amb_db_{k}"] = 20 * np.log10(a[mask].max() / pk)
        out[f"ambpk_db_{k}"] = ambiguity_level(t, a, pk, case.xs, cfg.system.vs, dt_amb)
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
    topo = case.offsets is not None
    xs, dys, hs = scene_coordinates(case.xs, case.heights, case.offsets)
    fig = plt.figure(figsize=(17, 9.2))
    gs = GridSpec(2, 2, width_ratios=[1.75, 1.0], hspace=0.36, wspace=0.14, figure=fig)
    spacing_txt = "" if topo else f", spacing {case.spacing:g} m"
    fig.suptitle(rf"{case_id} | SATA 1D | {LABEL[case.family]}: {case.desc}{spacing_txt}" + "\n"
                 + bxt_txt + rf" | $N_{{rx}}={cfg.Nrx}$ | PRF$={cfg.prf:.0f}$ Hz | $B_a={cfg.abw:.0f}$ Hz | "
                 rf"$\Delta b_{{at}}={cfg.array.bat[1]:.2f}$ m (DPCA)", fontsize=11.5, y=0.995)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["whole"], color="tab:blue", lw=1.6, ls=(0, (5, 3)), label="SATA whole band")
    ax.plot(t, db["sub"], color="tab:orange", lw=1.8, ls=(0, (1, 2)), label="SATA per sub-band")
    ax.set_title(f"IRF | worst ambiguity: ref {m['amb_db_ref']:.1f} dB, SATA whole {m['amb_db_whole']:.1f} dB, "
                 f"SATA sub {m['amb_db_sub']:.1f} dB (oracle {m['amb_db_oracle']:.1f} dB)", fontsize=9.5)
    az = fig.add_subplot(gs[1, 0])
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"IRF | worst ambiguity: ref {m['amb_db_ref']:.1f} dB, no SATA {m['amb_db_none']:.1f} dB",
                 fontsize=9.5)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=10, loc="upper right")
    slope = case.params["alpha"] if topo else np.max(np.abs(case.slopes_deg))
    span = np.ptp(hs)
    geometry_2d(fig.add_subplot(gs[0, 1]), xs, dys, hs, topo,
                title=f"Target geometry | $\\Delta h_{{max}}$={span:.0f} m, steepest |slope| {slope:.3g}\u00b0")
    a3 = fig.add_subplot(gs[1, 1], projection="3d")
    geometry_3d(a3, xs, dys, hs, topo, title="Target geometry in 3D (x, $\\Delta y$, h)")
    a3.text2D(0.99, 0.02, coordinates_text(xs, dys, hs), transform=a3.transAxes, fontsize=5.8,
              family="monospace", ha="right", va="bottom",
              bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _job(args):
    case, bxt_max, case_id, out_png = args
    cfg, foc = simulate(case, bxt_max)
    m = metrics(cfg, foc, case)
    make_figure(case, cfg, foc, m, bxt_max, case_id, out_png)
    xs, dys, hs = scene_coordinates(case.xs, case.heights, case.offsets)
    np.savez_compressed(out_png[:-4] + ".npz", t=(np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf,
                        **{f"env_{k}": np.abs(v).astype(np.float32) for k, v in foc.items()},
                        xs=xs, dys=dys, hs=hs, bxt=cfg.array.bxt, metrics=np.array([m], dtype=object))
    row = {"case_id": case_id, "family": case.family, "slug": case.slug, "bxt_max": bxt_max, "N": case.N,
           "dh_span": float(np.ptp(case.heights)), "max_jump": float(np.max(np.abs(np.diff(case.heights)))),
           "max_abs_slope": float(np.max(np.abs(case.slopes_deg))),
           "heights": " ".join(f"{h:.2f}" for h in case.heights), "png": out_png}
    row.update({k: round(float(v), 3) for k, v in m.items()})
    return row


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
    for bxt in args.bxt:
        pngs_to_pdf([r["png"] for r in rows if r["bxt_max"] == bxt],
                    os.path.join(args.out, f"all_cases_bxt{bxt:g}.pdf"))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
