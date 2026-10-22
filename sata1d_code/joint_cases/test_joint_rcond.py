"""
Test of option 1 for the cases where the joint reconstruction breaks: truncate the small singular values of the
per-bin system (np.linalg.pinv(..., rcond=r) in reconstruct_joint, occupied=True). sata1d/joint.py is used unchanged;
only the rcond of its pinv is changed during the call.
Cases of run_joint_sweep (C05, C09, C27, C29, C31) x bxt [0, 50, 100, 150], [0, 100, 200, 300] and, as a control,
U(0, 100 m). Output: joint_cases/rcond_test/ (summary + one page per case and baseline) and joint_rcond_test.pdf.
    python joint_cases/test_joint_rcond.py --workers 4          # from sata1d_code
"""
import argparse
import json
import os
import sys
import time
from functools import partial
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
import sata1d.joint as J
from sata1d.joint import reconstruct_joint, joint_maps
from sata1d.sata import subaperture_sizing
import run_geometry_sweep as G
import run_joint_sweep as W
from run_one_case import ambiguity_level
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy

OUT = os.path.join(HERE, "rcond_test")
CASES = ["C05", "C09", "C27", "C29", "C31"]
BASES = ["rnd100", "s50", "s100"]
RCONDS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]     # 0 = pinv as it is now (cut-off ~1e-15)
SHOW = 0.1                                           # rcond shown on the case pages
_PINV = np.linalg.pinv


def joint_rcond(cfg, tr, ch, maps, rcond):
    """reconstruct_joint(occupied=True) with np.linalg.pinv(..., rcond=rcond)."""
    np.linalg.pinv = partial(_PINV, rcond=max(rcond, 1e-15))
    try:
        return reconstruct_joint(cfg, tr, ch, maps=maps, sata_osf=W.SATA_OSF, occupied=True)
    finally:
        np.linalg.pinv = _PINV


def min_singular(cfg, tr, maps):
    """Smallest / largest singular value of the occupied-parcel system of every block (worst bin), as joint.py builds it."""
    Nrx, Nsb, P = cfg.Nrx, cfg.Nsb, cfg.PRF_op
    wl, v, r0 = cfg.system.wl, cfg.system.vs, cfg.scene.r0
    coef = J.reference_coefficients(cfg, tr)
    Tsub, hop, Nzp = subaperture_sizing(r0, P, v, wl, W.SATA_OSF)
    phi = np.arange(Nzp) * P / Nzp
    f = (-Nsb * P / 2.0 + np.arange(Nsb) * P)[:, None] + phi[None, :]
    a = r0 * np.tan(np.arcsin(np.clip(wl * f / (2.0 * v), -1, 1))) / v * P
    Hq = np.zeros((Nzp, Nrx, Nsb), complex)
    for i in range(Nrx):
        C0, C1, C2, Dt = coef[i]
        Hq[:, i, :] = np.exp(-2j * np.pi * (C0 / wl + (-C1 + Dt) * f.T + C2 * wl * f.T ** 2))
    B = 4.0 * v * np.sin(cfg.theta_tx / 2.0) / wl
    occ = np.abs(f.T) <= B / 2.0 + 4.0 * P / Tsub
    tc, smin = [], []
    for start in range(0, cfg.Na_ch, hop):
        c = start + 0.5 * Tsub
        pos = np.clip(np.round(c + a).astype(int), 0, cfg.Na_ch - 1)
        Gm = Hq * np.exp(-2j * np.pi * maps[:, pos].transpose(2, 0, 1) / wl) * occ[:, None, :]
        sv = np.linalg.svd(Gm, compute_uv=False)                     # [Nzp, Nsb], descending
        nocc = occ.sum(axis=1)
        r = np.array([sv[k, nocc[k] - 1] / sv[k, 0] for k in range(Nzp)])
        tc.append((c - cfg.Na_ch / 2) / P); smin.append(r.min())
    return np.array(tc), np.array(smin)


def run(cid, btag, out=OUT):
    t0 = time.time()
    c = W.case_info(cid)
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    cfg = W.make_cfg(c, btag)
    tr = S.build_platform_tracks(cfg)
    dC0 = S.delta_C0_per_target(cfg, tr)
    ch = S.generate_channels(cfg, tr)
    if terrain is None:
        maps = np.array(joint_maps(cfg, tr, dC0))
    else:
        maps = np.array([S.terrain_lookup_table(cfg, tr, i, terrain, step=4) for i in range(cfg.Nrx)])
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    env = {"ref": F(S.generate_reference(cfg, tr)), "none": F(S.reconstruct(cfg, tr, ch)),
           "whole": F(S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=W.SATA_OSF, mode="footprint", dC0=dC0)))}
    for r in RCONDS:
        env[f"joint_{r:g}"] = F(joint_rcond(cfg, tr, ch, maps, r))
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    amb = {k: float(ambiguity_level(t, v, pk, xs, cfg.system.vs, G.ambiguity_times(cfg))) for k, v in env.items()}
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    tb, smin = min_singular(cfg, tr, maps)

    fig = plt.figure(figsize=(18, 9.6))
    gs = GridSpec(2, 2, width_ratios=[1.45, 1.25], hspace=0.36, wspace=0.14, figure=fig)
    fig.suptitle(f"Case {c['number']}: {c['label']} | $b_{{xt}}$ = {W.bxt_text(W.BXT[btag])} ({W.BKIND[btag]}) | "
                 f"joint reconstruction, occupied parcels only: pinv as now vs. pinv with rcond = {SHOW:g}", fontsize=12, y=0.995)
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["joint_0"], color="tab:purple", lw=1.3, ls=(0, (5, 3)), label="joint, occupied, pinv as now")
    ax.plot(t, db[f"joint_{SHOW:g}"], color="tab:blue", lw=1.2, label=f"joint, occupied, pinv with rcond = {SHOW:g}")
    ax.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | joint as now {amb['joint_0']:.1f} dB | "
                 f"joint, rcond = {SHOW:g}: {amb[f'joint_{SHOW:g}']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | "
                 f"SATA whole band {amb['whole']:.1f} dB", fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 25 if max(amb.values()) > 0 else 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")

    # geometry (as the sweep pages)
    xs_, dys, hs_ = scene_coordinates(xs, hs)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    gg = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.18, width_ratios=[1.0, 1.1])
    g2 = fig.add_subplot(gg[0, 0]); g3 = fig.add_subplot(gg[0, 1], projection="3d")
    lo2, hi2 = min(xs_.min(), 0.0), max(xs_.max(), 0.0); pad2 = max(0.12 * (hi2 - lo2), 500.0)
    if terrain is None:
        geometry_2d(g2, xs_, dys, hs_, False, title=None, annotate=len(xs_) <= 13)
        geometry_3d(g3, xs_, dys, hs_, False, title=None)
    else:
        xg = np.linspace(-half, half, 4801); hg = terrain(xg)
        hz = hg[(xg >= lo2 - pad2) & (xg <= hi2 + pad2)]
        geometry_2d(g2, xs_, dys, hs_, False, title=None, connect=False, annotate=len(xs_) <= 13)
        g2.fill_between(xg, min(hg.min(), 0) - 0.05 * np.ptp(hg), hg, color="#D7C9A7", alpha=0.45, lw=0)
        g2.plot(xg, hg, color="#6D4C2F", lw=1.2, zorder=2, label="terrain (the scene)")
        ylo, yhi = min(hz.min(), 0.0), max(hz.max(), hs_.max(), 1.0)
        g2.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))
        geometry_3d(g3, xs_, dys, hs_, False, connect=False, title=None)
        g3.plot(xg, iso_range_dy(hg), hg, color="#6D4C2F", lw=1.0)
    g2.plot(0.0, 0.0, marker="*", ms=13, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $h$ = 0)")
    g2.set_xlim(lo2 - pad2, hi2 + pad2)
    g2.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.1)
    g2.set_title("Zoom on the point targets and the reference point", fontsize=9.5, pad=36)
    g2.xaxis.set_major_formatter(km); g2.set_xlabel("azimuth position $x$ [km]")
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=6)
    g3.set_xlim(-half, half); g3.xaxis.set_major_formatter(km); g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4)
    g3.set_title(f"Whole azimuth line ({2 * half / 1e3:.0f} km) in 3D (x, $\\Delta y$, h)", fontsize=9.5)
    for t_ in list(g3.texts):
        t_.remove()

    # where the system degenerates
    am = fig.add_subplot(gs[1, 1])
    am.semilogy(tb, smin, color="0.25", lw=1.2, label="smallest / largest singular value (worst bin of the block)")
    am.axhline(SHOW, color="tab:blue", ls="--", lw=1.2, label=f"rcond = {SHOW:g}: smaller singular values are cut")
    am.set_xlim(t[0], t[-1]); am.set_ylim(1e-4, 2); am.grid(alpha=0.3, which="both")
    am.set_xlabel("Azimuth time of the block centre [s]"); am.set_ylabel("singular value ratio")
    am.set_title("Conditioning of the joint system per block (occupied parcels)", fontsize=10)
    am.legend(fontsize=8.5, loc="lower left")
    fig.text(0.5, 0.012, f"Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions of every target. "
             "Joint with rcond: " + ", ".join(f"{r:g} -> {amb[f'joint_{r:g}']:.1f} dB" for r in RCONDS)
             + f". Channel 0 is the transmitter ($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)}.",
             ha="center", fontsize=8.5, color="0.35")
    fig.subplots_adjust(bottom=0.1)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{cid}_{btag}.png"), dpi=90, bbox_inches="tight"); plt.close(fig)
    res = dict(amb, case=cid, baseline=btag, label=c["label"], smin=float(smin.min()), seconds=time.time() - t0)
    json.dump(res, open(os.path.join(out, f"{cid}_{btag}.json"), "w"), indent=1)
    return res


def summary(out=OUT):
    res = {(r["case"], r["baseline"]): r for r in (json.load(open(os.path.join(out, f"{c}_{b}.json"))) for c in CASES for b in BASES)}
    fig, axs = plt.subplots(len(CASES), len(BASES), figsize=(16, 3.0 * len(CASES)), sharex=True)
    xk = np.arange(len(RCONDS))
    for i, c in enumerate(CASES):
        for j, b in enumerate(BASES):
            r, a = res[(c, b)], axs[i, j]
            a.axhline(r["ref"], color="tab:green", lw=2, label="reference")
            a.axhline(r["none"], color="tab:red", lw=1.4, ls=(0, (5, 3)), label="no SATA")
            a.axhline(r["whole"], color="tab:orange", lw=1.4, ls=(0, (2, 2)), label="SATA whole band")
            y = [r[f"joint_{x:g}"] for x in RCONDS]
            a.plot(xk, y, "o-", color="tab:blue", lw=1.6, ms=5, label="joint, occupied parcels")
            for k, v in zip(xk, y):
                a.annotate(f"{v:.1f}", (k, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5)
            lo = min(y + [r["ref"], r["whole"]]); hi = max(y + [r["none"]])
            a.set_ylim(lo - 6, hi + 9); a.grid(alpha=0.3)
            if i == 0:
                a.set_title(f"$b_{{xt}}$ = {W.bxt_text(W.BXT[b])}\n({W.BKIND[b]})", fontsize=10)
            if j == 0:
                a.set_ylabel(f"case {int(c[1:])}\nhighest ambiguity [dB]", fontsize=9)
            a.text(0.01, 0.97, r["label"], transform=a.transAxes, fontsize=7, va="top", color="0.3")
    for a in axs[-1]:
        a.set_xticks(xk); a.set_xticklabels(["as now"] + [f"{x:g}" for x in RCONDS[1:]])
        a.set_xlabel("rcond of the pinv (singular values below rcond x the largest are cut)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    png = os.path.join(out, "00_summary.png")
    fig.savefig(png, dpi=90); plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pdf", default=os.path.join(HERE, "joint_rcond_test.pdf"))
    a = ap.parse_args()
    todo = [(c, b) for c in CASES for b in BASES if not os.path.exists(os.path.join(OUT, f"{c}_{b}.json"))]
    from multiprocessing import Pool
    with Pool(a.workers) as p:
        for r in p.starmap(run, todo):
            print(r["case"], r["baseline"], " ".join(f"{x:g}:{r[f'joint_{x:g}']:.1f}" for x in RCONDS), f"({r['seconds']:.0f} s)", flush=True)
    if all(os.path.exists(os.path.join(OUT, f"{c}_{b}.json")) for c in CASES for b in BASES):
        import joint_sweep_summary as JS
        pngs = [summary()] + [os.path.join(OUT, f"{c}_{b}.png") for c in CASES for b in BASES]
        print("wrote", JS.bundle(pngs, a.pdf, [("rcond test: summary", 0, 0)]
                                 + [(f"case {int(c[1:])}", 1 + k * len(BASES), 0) for k, c in enumerate(CASES)]))


if __name__ == "__main__":
    main()
