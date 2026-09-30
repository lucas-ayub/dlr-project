#!/usr/bin/env python3
"""
Geometry sweep with the NEW SATA (delta_C0 map only where each target appears
in the STFT), same cases as runs/core/run_sata_sub_geometry_sweep.py.

For every case: reference (monostatic), no SATA, SATA whole band, SATA per
sub-band (explicit kernel, C0) with the new map, and -- for comparison only --
SATA whole / per sub-band with the LEGACY "hold" map.

One figure per case (layout of the report figures):
    top    : IRF, full azimuth axis -- reference / SATA whole / SATA sub (new map)
    bottom : IRF, full azimuth axis -- reference / no SATA
    right  : target geometry
Outputs (default ./geometry_sweep_new_sata/):
    bxt<..>/<family>/<id>_<slug>.png, all_cases_bxt<..>.pdf,
    sweep_results.csv, summary_new_vs_legacy.png

Run (from this folder):
    python run_geometry_sweep_new_sata.py --workers 2
    python run_geometry_sweep_new_sata.py --quick
"""
from __future__ import annotations
import argparse, contextlib, csv, io, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "runs", "core"))
import run_sata_sub_geometry_sweep as S                        # cases, cfg, metrics (repo)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sar_recon as sar
import sar_recon.sata as sata_mod
import sar_recon.subband_sata_explicit as ex
from sar_recon.signal_model import getRawData1D
from sar_recon.subband_sata_explicit import reconstruct_subband_explicit

C_REF, C_WHOLE, C_SUB, C_NONE = "tab:green", "tab:blue", "tab:orange", "tab:red"


@contextlib.contextmanager
def legacy_maps():
    """Temporarily switch both map builders to the legacy 'hold' mode."""
    a, b = sata_mod.build_delta_C0_array, ex.build_delta_term_subband_array
    sata_mod.build_delta_C0_array = lambda *p, **k: a(*p, **{**k, "mode": "hold"})
    ex.build_delta_term_subband_array = lambda *p, **k: b(*p, **{**k, "mode": "hold"})
    try:
        yield
    finally:
        sata_mod.build_delta_C0_array, ex.build_delta_term_subband_array = a, b


def simulate(case, bxt_max):
    cfg, tr = S.build_cfg(case, bxt_max)
    s, Nrx, prf, ta = cfg.system, cfg.Nrx, cfg.prf, cfg.ta
    raw = lambda ptgs, prx, vrx: getRawData1D(ptgs, tr.ptx, prx, tr.vtx, vrx, ta, cfg.sq_tx, cfg.sq_tx,
                                              cfg.theta_tx, cfg.theta_tx, s.wl, prf)
    ptgs = cfg.scene.points[1:]
    sref1 = raw(cfg.scene.ptg[None, :], tr.ptx, tr.vtx)
    sig = raw(ptgs, tr.ptx, tr.vtx)
    s_ch = np.array([raw(ptgs, tr.prx[i], tr.vrx[i])[::Nrx] for i in range(Nrx)])

    def whole_sub():
        return (sar.reconstruct(cfg, tr, sata_mod.sata_channels(cfg, tr, s_ch.copy(), sata_osf=S.SATA_OSF)),
                reconstruct_subband_explicit(cfg, tr, s_ch.copy(), use_sata=True,
                                             sata_osf=S.SATA_OSF, correct_terms=("C0",)))
    with contextlib.redirect_stdout(io.StringIO()):
        rec_none = sar.reconstruct(cfg, tr, s_ch.copy())
        rec_whole, rec_sub = whole_sub()
        with legacy_maps():
            old_whole, old_sub = whole_sub()
    Na = cfg.Na
    F = lambda x: np.roll(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(sref1))), Na // 2)
    return cfg, {"ref": F(sig), "none": F(rec_none), "whole": F(rec_whole), "sub": F(rec_sub),
                 "whole_old": F(old_whole), "sub_old": F(old_sub)}


def make_figure(case, cfg, foc, m, bxt_max, case_id, out_png):
    Na, prf = cfg.Na, cfg.prf
    pk = np.abs(foc["ref"]).max()
    db = {k: 20 * np.log10(np.abs(v) / pk + 1e-12) for k, v in foc.items()}
    t = (np.arange(Na) - Na // 2) / prf
    dbat = cfg.array.bat[1] - cfg.array.bat[0]
    bxt_txt = (rf"$b_{{xt}}\sim U(0,{bxt_max:g})$ m = [" + ", ".join(f"{b:.1f}" for b in cfg.array.bxt)
               + "] m" if bxt_max > 0 else r"no cross-track baseline ($b_{xt}=0$)")
    fig = plt.figure(figsize=(17, 8.6))
    gs = GridSpec(2, 2, width_ratios=[1.75, 1.0], hspace=0.34, wspace=0.16, figure=fig)
    fig.suptitle(rf"{case_id} | {S.FAMILY_LABEL[case.family]}: {case.desc}, spacing {case.spacing:g} m" + "\n"
                 + bxt_txt + rf" | $N_{{rx}}={cfg.Nrx}$ | PRF$={prf:.0f}$ Hz | $B_a={cfg.abw:.0f}$ Hz | "
                 rf"$\Delta b_{{at}}={dbat:.2f}$ m (DPCA) | NEW SATA map (only where each target appears)",
                 fontsize=11.5, y=0.995)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, db["ref"], color=C_REF, lw=2.2, label="Reference")
    ax.plot(t, db["whole"], color=C_WHOLE, lw=1.6, ls=(0, (5, 3)), label="SATA whole band")
    ax.plot(t, db["sub"], color=C_SUB, lw=1.8, ls=(0, (1, 2)), label="SATA per sub-band")
    ax.set_title(f"IRF, full azimuth axis | worst ambiguity: ref {m['amb_db_ref']:.1f} dB, "
                 f"SATA whole {m['amb_db_whole']:.1f} dB, SATA sub {m['amb_db_sub']:.1f} dB\n"
                 f"(legacy map: SATA whole {m['amb_db_whole_old']:.1f} dB, SATA sub {m['amb_db_sub_old']:.1f} dB)",
                 fontsize=9.5)
    az = fig.add_subplot(gs[1, 0])
    az.plot(t, db["ref"], color=C_REF, lw=2.2, label="Reference")
    az.plot(t, db["none"], color=C_NONE, lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"IRF, full azimuth axis | worst ambiguity: ref {m['amb_db_ref']:.1f} dB, "
                 f"no SATA {m['amb_db_none']:.1f} dB", fontsize=9.5)
    for a in (ax, az):
        a.set_xlim(t[0], t[-1]); a.set_ylim(-90, 5); a.grid(alpha=0.3)
        a.set_xlabel("Azimuth time [s]"); a.set_ylabel("[dB]")
        a.legend(fontsize=10, loc="upper right", handlelength=3)
    ag = fig.add_subplot(gs[:, 1])
    xs, hs = case.xs, case.heights
    lo = min(0.0, hs.min()); span = np.ptp(hs)
    ag.fill_between(xs, lo - 0.05 * (span + 1), hs, color="#D7C9A7", alpha=0.45, lw=0)
    ag.plot(xs, hs, "-", color="#6D4C2F", lw=1.6, zorder=2)
    ag.scatter(xs, hs, s=55, color="#1F4E79", edgecolor="k", zorder=3, label="point targets")
    ag.axhline(0.0, color="k", ls="--", lw=0.9, label=r"reconstruction reference $h_0=0$")
    fs = 8 if case.N <= 9 else 6.5
    for i, sl in enumerate(case.slopes_deg):
        ag.annotate(f"{sl:+.3g}°", (0.5*(xs[i]+xs[i+1]), 0.5*(hs[i]+hs[i+1])), textcoords="offset points",
                    xytext=(-4, 7), ha="right", va="bottom", fontsize=fs, color="#6D4C2F",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    for xi, hi in zip(xs, hs):
        ag.annotate(f"{hi:.0f} m", (xi, hi), textcoords="offset points", xytext=(7, -4),
                    ha="left", va="top", fontsize=fs, color="#1F4E79")
    pad = 0.25 * span if span > 0 else 10.0
    ag.set_ylim(lo - pad, hs.max() + pad)
    ag.set_xlim(xs.min() - 1.5 * case.spacing, xs.max() + 1.5 * case.spacing)
    ag.set_xlabel("Azimuth position [m]"); ag.set_ylabel("Height above reference [m]")
    ag.grid(alpha=0.3); ag.legend(fontsize=8, loc="upper left")
    jump = np.max(np.abs(np.diff(hs))) if case.N > 1 else 0.0
    ag.set_title(f"Target geometry | $\\Delta h_{{max}}$={span:.0f} m, largest jump {jump:.0f} m, "
                 f"steepest |slope| {np.max(np.abs(case.slopes_deg)):.3g}°", fontsize=9.5)
    ag.text(0.99, 0.02, f"targets on the iso-range surface\n$r_0$ = {cfg.scene.r0/1e3:.1f} km",
            transform=ag.transAxes, ha="right", va="bottom", fontsize=8, color="0.3")
    fig.savefig(out_png, dpi=100, bbox_inches="tight"); plt.close(fig)


def _job(args):
    case, bxt_max, case_id, out_png = args
    t0 = time.time()
    cfg, foc = simulate(case, bxt_max)
    m = S.metrics(cfg, foc, case)
    make_figure(case, cfg, foc, m, bxt_max, case_id, out_png)
    row = {"case_id": case_id, "family": case.family, "slug": case.slug, "bxt_max": bxt_max,
           "N": case.N, "alpha": case.params.get("alpha", ""), "dh_span": float(np.ptp(case.heights)),
           "max_jump": float(np.max(np.abs(np.diff(case.heights)))),
           "max_abs_slope": float(np.max(np.abs(case.slopes_deg))),
           "heights": " ".join(f"{h:.2f}" for h in case.heights), "png": out_png,
           "runtime_s": round(time.time() - t0, 2)}
    row.update({k: round(float(v), 3) for k, v in m.items() if not k.startswith("_")})
    return row


def summary(rows, out_png):
    fams = [f for f in S.FAMILIES if any(r["family"] == f for r in rows)]
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5))
    for bi, bxt in enumerate(sorted({r["bxt_max"] for r in rows}, reverse=True)[:1]):
        R = [r for r in rows if r["bxt_max"] == bxt]
        for ax, (kn, ko, lab) in zip(axs[:2], (("amb_db_whole", "amb_db_whole_old", "SATA whole band"),
                                             ("amb_db_sub", "amb_db_sub_old", "SATA per sub-band"))):
            for f in fams:
                x = [r[ko] for r in R if r["family"] == f]; y = [r[kn] for r in R if r["family"] == f]
                ax.scatter(x, y, s=16, label=f)
            lim = (-60, 0); ax.plot(lim, lim, "k--", lw=.8); ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("worst ambiguity, LEGACY map [dB]"); ax.set_ylabel("worst ambiguity, NEW map [dB]")
            ax.set_title(f"{lab}, bxt ~ U(0,{bxt:g}) m  (below the diagonal = new map better)")
            ax.grid(alpha=.3)
        axs[0].legend(fontsize=8, ncol=2)
        keys = (("amb_db_ref", "reference", C_REF), ("amb_db_none", "no SATA", C_NONE),
                ("amb_db_whole_old", "whole, legacy", "#9ecae1"), ("amb_db_whole", "whole, NEW", C_WHOLE),
                ("amb_db_sub_old", "sub, legacy", "#fdd0a2"), ("amb_db_sub", "sub, NEW", C_SUB))
        w = 0.13
        for j, (k, lab, c) in enumerate(keys):
            med = [np.median([r[k] for r in R if r["family"] == f]) for f in fams]
            axs[2].bar(np.arange(len(fams)) + (j - 2.5) * w, med, w, color=c, label=lab)
        axs[2].set_xticks(range(len(fams)), fams, rotation=30, ha="right")
        axs[2].set_ylabel("median worst ambiguity [dB]"); axs[2].invert_yaxis()
        axs[2].set_title(f"Per family, bxt ~ U(0,{bxt:g}) m (lower = better)")
        axs[2].legend(fontsize=8, ncol=3); axs[2].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=S.FAMILIES, choices=S.FAMILIES)
    ap.add_argument("--bxt", nargs="+", type=float, default=[S.BXT_MAX, 0.0])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=os.path.join(HERE, "geometry_sweep_new_sata"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    cases = S.build_cases(args.families, S.SPACING)
    if args.quick:
        cases = cases[::max(1, len(cases) // 6)]
    prefix = {"ramp": "RA", "zigzag": "ZI", "piecewise": "PW", "steepening": "ST", "flattening": "FL",
              "cliff": "CL", "spike": "SK", "mesa": "ME", "random_heights": "RH"}
    jobs, counter = [], {}
    for bxt in args.bxt:
        tag = f"bxt{bxt:g}"
        for c in cases:
            counter[(tag, c.family)] = counter.get((tag, c.family), 0) + 1
            cid = f"{prefix[c.family]}-{counter[(tag, c.family)]:03d}"
            d = os.path.join(args.out, tag, c.family); os.makedirs(d, exist_ok=True)
            jobs.append((c, bxt, cid, os.path.join(d, f"{cid}_{c.slug}.png")))
    print(f"{len(jobs)} cases, {args.workers} workers -> {args.out}", flush=True)
    t0 = time.time(); rows = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap(_job, jobs, chunksize=2), 1):
                rows.append(row)
                if i % 20 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f} s)", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            rows.append(_job(j)); print(f"  {i}/{len(jobs)} {j[2]}", flush=True)
    with open(os.path.join(args.out, "sweep_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary(rows, os.path.join(args.out, "summary_new_vs_legacy.png"))
    for bxt in args.bxt:
        S.pngs_to_pdf([r["png"] for r in rows if r["bxt_max"] == bxt],
                      os.path.join(args.out, f"all_cases_bxt{bxt:g}.pdf"))
    print(f"done in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
