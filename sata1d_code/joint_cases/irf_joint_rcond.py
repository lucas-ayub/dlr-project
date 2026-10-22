"""
IRFs of the rcond test (test_joint_rcond.py): for every case and baseline, the focused azimuth line of each method
against the reference, one panel per method, same axes: joint (occupied parcels) with pinv as now, joint with
pinv rcond = 0.1, SATA whole band, no SATA.
Output: joint_cases/rcond_test/irf_<case>_<baseline>.png and joint_cases/joint_rcond_irf.pdf
    python joint_cases/irf_joint_rcond.py --workers 4          # from sata1d_code
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
from sata1d.joint import joint_maps
import run_geometry_sweep as G
import run_joint_sweep as W
import test_joint_rcond as TR
from run_one_case import ambiguity_level

OUT = TR.OUT
RC = 0.1
PANELS = [("joint_now", "joint, occupied parcels, pinv as now", "tab:purple"),
          ("joint_rc", f"joint, occupied parcels, pinv with rcond = {RC:g}", "tab:blue"),
          ("whole", "SATA whole band", "tab:orange"),
          ("none", "no SATA", "tab:red")]


def run(cid, btag):
    c = W.case_info(cid)
    xs, terrain = c["xs"], c["terrain"]
    cfg = W.make_cfg(c, btag)
    tr = S.build_platform_tracks(cfg)
    dC0 = S.delta_C0_per_target(cfg, tr)
    ch = S.generate_channels(cfg, tr)
    maps = (np.array(joint_maps(cfg, tr, dC0)) if terrain is None else
            np.array([S.terrain_lookup_table(cfg, tr, i, terrain, step=4) for i in range(cfg.Nrx)]))
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    env = {"ref": F(S.generate_reference(cfg, tr)), "none": F(S.reconstruct(cfg, tr, ch)),
           "whole": F(S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=W.SATA_OSF, mode="footprint", dC0=dC0))),
           "joint_now": F(TR.joint_rcond(cfg, tr, ch, maps, 0.0)),
           "joint_rc": F(TR.joint_rcond(cfg, tr, ch, maps, RC))}
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt = G.ambiguity_times(cfg)
    amb = {k: float(ambiguity_level(t, v, pk, xs, cfg.system.vs, dt)) for k, v in env.items()}
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    ttg = xs / cfg.system.vs                                       # target times

    fig, axs = plt.subplots(len(PANELS), 1, figsize=(16, 13), sharex=True, sharey=True)
    for a, (k, lab, col) in zip(axs, PANELS):
        a.plot(t, db["ref"], color="tab:green", lw=1.8, label="Reference")
        a.plot(t, db[k], color=col, lw=1.0, label=lab)
        for tt in (ttg if len(ttg) <= 7 else []):                  # ambiguity positions of the targets (+-1, +-2 folds)
            for m in (-2, -1, 1, 2):
                a.axvline(tt + m * dt, color="0.6", lw=0.6, ls=":", zorder=0)
        a.set_ylim(-90, max(8.0, max(amb.values()) + 5)); a.grid(alpha=0.3)
        a.set_ylabel("[dB]")
        a.set_title(f"{lab}: highest ambiguity {amb[k]:.1f} dB (reference {amb['ref']:.1f} dB)", fontsize=10.5, loc="left")
        a.legend(fontsize=9, loc="upper right")
    axs[-1].set_xlim(t[0], t[-1]); axs[-1].set_xlabel("Azimuth time [s]")
    fig.suptitle(f"IRF, case {c['number']}: {c['label']} | $b_{{xt}}$ = {W.bxt_text(W.BXT[btag])} ({W.BKIND[btag]})\n"
                 + (f"dotted lines: ambiguity positions of the targets ($\\pm$1 and $\\pm$2 folds, $\\pm${dt:.2f} s and $\\pm${2 * dt:.2f} s)" if len(ttg) <= 7 else f"ambiguities of each target at $\\pm${dt:.2f} s and $\\pm${2 * dt:.2f} s from it"),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = os.path.join(OUT, f"irf_{cid}_{btag}.png")
    fig.savefig(png, dpi=90); plt.close(fig)
    print(cid, btag, {k: round(v, 1) for k, v in amb.items()}, flush=True)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--baselines", nargs="+", default=["s50", "s100"])
    ap.add_argument("--pdf", default=os.path.join(HERE, "joint_rcond_irf.pdf"))
    a = ap.parse_args()
    jobs = [(c, b) for c in TR.CASES for b in a.baselines]
    todo = [j for j in jobs if not os.path.exists(os.path.join(OUT, f"irf_{j[0]}_{j[1]}.png"))]
    from multiprocessing import Pool
    with Pool(a.workers) as p:
        p.starmap(run, todo)
    import joint_sweep_summary as JS
    pngs = [os.path.join(OUT, f"irf_{c}_{b}.png") for c, b in jobs]
    if all(os.path.exists(p) for p in pngs):
        print("wrote", JS.bundle(pngs, a.pdf, [(f"case {int(c[1:])}, bxt {W.bxt_text(W.BXT[b])}", k, 0)
                                                for k, (c, b) in enumerate(jobs)]))


if __name__ == "__main__":
    main()
