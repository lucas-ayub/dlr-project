"""
Why some cross-track baseline sets break the joint reconstruction: conditioning of the per-bin system of
reconstruct_joint (occupied parcels, as sata1d/joint.py builds it) against the height difference dh_X between
points one fold X = 6.2 km apart (terrain that rises dh_X per fold, i.e. a ramp across the parcels of a bin).
The filters are the real reference-point filters of joint.py; the topographic phase of channel i is
2 pi dC0_i(h) / wl with dC0_i(h) from residual_C0 (linear in h to 0.6 %).
Output: joint_cases/baseline_degeneracy.png (+ .json with the curves)
    python joint_cases/baseline_degeneracy.py          # from sata1d_code
"""
import json
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
import sata1d.joint as J
from sata1d.sata import subaperture_sizing, residual_C0
import run_joint_sweep as W

SHOW = [("s50", "tab:red"), ("dec50", "tab:orange"), ("perm50", "tab:purple"), ("uneven", "tab:blue"), ("rnd100", "tab:green")]
DH = np.arange(-1200.0, 1200.1, 2.0)


def system(btag, c):
    """Reference filters on the bin grid [bin, channel, parcel], occupied parcels [bin, parcel], d dC0 / dh per channel."""
    cfg = W.make_cfg(c, btag); tr = S.build_platform_tracks(cfg)
    Nrx, Nsb, P = cfg.Nrx, cfg.Nsb, cfg.PRF_op
    wl, v, r0 = cfg.system.wl, cfg.system.vs, cfg.scene.r0
    coef = J.reference_coefficients(cfg, tr)
    Tsub, hop, Nzp = subaperture_sizing(r0, P, v, wl, W.SATA_OSF)
    f = (-Nsb * P / 2.0 + np.arange(Nsb) * P)[:, None] + (np.arange(Nzp) * P / Nzp)[None, :]
    Hq = np.zeros((Nzp, Nrx, Nsb), complex)
    for i in range(Nrx):
        C0, C1, C2, Dt = coef[i]
        Hq[:, i, :] = np.exp(-2j * np.pi * (C0 / wl + (-C1 + Dt) * f.T + C2 * wl * f.T ** 2))
    occ = np.abs(f.T) <= 2.0 * v * np.sin(cfg.theta_tx / 2.0) / wl + 4.0 * P / Tsub
    H, y0 = cfg.scene.H, cfg.scene.y0
    dc0 = lambda i, h: residual_C0(cfg, tr, cfg.scene.ptg + np.array([0.0, np.sqrt(r0 ** 2 - (H - h) ** 2) - y0, h]), i)
    k = np.array([(dc0(i, 500.0) - dc0(i, -500.0)) / 1000.0 for i in range(Nrx)])
    return cfg, Hq, occ, k


def conditioning(btag, c, dh=DH):
    """min over the bins of smallest / largest singular value of the occupied-parcel system, for every dh_X."""
    cfg, Hq, occ, k = system(btag, c)
    q = np.arange(cfg.Nsb)
    E = np.exp(-2j * np.pi * k[None, :, None] * q[None, None, :] * dh[:, None, None] / cfg.system.wl)   # [dh, channel, parcel]
    G = Hq[None] * E[:, None] * occ[None, :, None, :]                                                   # [dh, bin, channel, parcel]
    sv = np.linalg.svd(G, compute_uv=False)
    n = occ.sum(1)
    return (np.take_along_axis(sv, (n - 1)[None, :, None], axis=2)[..., 0] / sv[..., 0]).min(1)


def fold_dh(c, X):
    """Largest |h(x + X) - h(x)| of a terrain case along the line."""
    x = np.arange(-25e3, 25e3, 10.0)
    return float(np.abs(c["terrain"](x + X) - c["terrain"](x)).max())


def main():
    c33 = W.case_info("C33")
    cfg = W.make_cfg(c33, "rnd100")
    X = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs)      # fold distance: one PRF_op of Doppler [m]
    curves = {b: conditioning(b, c33) for b in [t for t, _, _ in W.BASELINES + W.EXTRA_BASELINES]}
    groups = {}
    for cid, (grp, _) in W.CASES.items():
        c = W.case_info(cid)
        if c["terrain"] is not None:
            key = "sine and cosine terrains" if grp == "sincos" else ("random mountains, isolated mountain" if "mountain" in c["label"] else
                                                                     "hills, escarpment, plateaus")
            groups[key] = max(groups.get(key, 0.0), fold_dh(c, X))
    json.dump({"dh": DH.tolist(), "X": X, "groups": groups, **{b: v.tolist() for b, v in curves.items()}},
              open(os.path.join(HERE, "baseline_degeneracy.json"), "w"))

    fig, ax = plt.subplots(figsize=(15, 7.5))
    band_col = {"sine and cosine terrains": "#9ecae1", "hills, escarpment, plateaus": "#c7e9c0", "random mountains, isolated mountain": "#fdd0a2"}
    for k, (name, m) in enumerate(sorted(groups.items(), key=lambda kv: -kv[1])):
        ax.axvspan(-m, m, color=band_col[name], alpha=0.45, lw=0, zorder=0)
        ax.text(m - 8, 1.25e-6 * (2.6 if k == 1 else 1.0), f"{name}: |$\\Delta h_X$| up to {m:.0f} m", ha="right", va="bottom", fontsize=9, color="0.25")
    for b, col in SHOW:
        ax.semilogy(DH, curves[b], color=col, lw=1.6, label=f"$b_{{xt}}$ = {W.bxt_text(W.BXT[b])} ({W.BKIND[b]})")
    ax.axhline(0.05, color="0.4", ls=":", lw=1.0)
    ax.text(-1190, 0.058, "0.05", fontsize=8.5, color="0.4")
    ax.set_xlim(DH[0], DH[-1]); ax.set_ylim(1e-6, 2.0); ax.grid(alpha=0.3, which="both")
    ax.set_xlabel(f"height difference between two points one fold apart (X = {X / 1e3:.2f} km), $\\Delta h_X$ [m]")
    ax.set_ylabel("smallest / largest singular value\n(joint system, occupied parcels, worst bin)")
    ax.set_title("Conditioning of the joint reconstruction against the terrain height difference, for five cross-track baseline sets\n"
                 "(close to 0: the effective positions of the channels collapse and the parcels of a bin cannot be separated)", fontsize=11)
    ax.legend(fontsize=9.5, loc="lower left", bbox_to_anchor=(0.0, 0.08), framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(HERE, "baseline_degeneracy.png")
    fig.savefig(out, dpi=100); plt.close(fig)
    print("wrote", out, groups)


if __name__ == "__main__":
    main()
