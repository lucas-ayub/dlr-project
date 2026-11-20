"""Item 2, part 2 (v30): what a time-varying baseline does to the joint reconstruction itself.

Slide-8 scene (1000 m Gaussian mountain, 5 targets), default system of the report (N_rx = 4, b_xt = 0/300/119.8/100.5 m). The
receiver with b_xt = 300 m moves cross-track by eps(t) (shapes of item2_baseline_variation.py: linear drift, two
oscillations per aperture, three bumps); the channels are simulated with the exact distances of the moving receiver.
Joint reconstructions (same solver as the repository, mu = 0.03 sigma_max, 2 passes):
  nominal   the model of the repository built for straight tracks (the motion is ignored)
  moco      first-order motion compensation to the reference height (h = 0, line of sight of the beam centre), then the
            repository's model: what is left is the topography-dependent part of the motion, which a model with
            Delta C0 only cannot describe (item 2: "all but Delta C0", about 2 pi/lambda * eps * h / (r0 tan theta))
  std       the standard reconstruction of the motion-compensated channels
Metric: the deck's highest ambiguity (common_r.amb_old), dB re. the reference peak.
    python3 -I item2_joint.py           -> cache/item2_joint.json, fig/item2_joint.pdf/png
"""
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common_m as M
import common_r as RC
import item2_baseline_variation as I2
from sata1d.geometry import PlatformTracks
from sata1d.joint_line import LineModel, height_table, line_maps
from sata1d.signal_model import raw
import solvers as V

XS = np.array([-8000.0, -550.0, 650.0, 2900.0, 4100.0])
EPS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
KINDS = ["drift", "sine", "bumps"]
MOVING = 1                                                              # the receiver with b_xt = 300 m


def tracks(cfg, kind, eps):
    tr = M.S.build_platform_tracks(cfg)
    T = I2.aperture(cfg)
    e = I2.eps_shape(kind, cfg.ta - cfg.scene.x0 / cfg.system.vs, T, eps)
    prx, vrx = tr.prx.copy(), tr.vrx.copy()
    prx[MOVING, :, 1] += e
    vrx[MOVING] = np.sqrt(cfg.system.vs ** 2 + np.gradient(prx[MOVING, :, 1], cfg.ta) ** 2)
    return PlatformTracks(ptx=tr.ptx, vtx=tr.vtx, prx=prx, vrx=vrx)


def phase_error(cfg, tr, tr0, ter):
    """Largest phase [rad] that the MoCo to h = 0 leaves on a target over its aperture (moving receiver only):
    2 pi / lambda * max_t |(|Rx(t) - p| - |Rx0(t) - p|) - (|Rx(t) - c(t)| - |Rx0(t) - c(t)|)|, p = target, c(t) = point of
    the reference height at the beam centre, Rx / Rx0 = true / nominal receiver track."""
    T = I2.aperture(cfg)
    out = 0.0
    for x, h in zip(XS, ter(XS)):
        sel = np.abs(cfg.ta - x / cfg.system.vs) <= 0.48 * T
        t = cfg.ta[sel]
        p = np.array([x, np.sqrt(cfg.scene.r0 ** 2 - (cfg.scene.H - h) ** 2), h])
        c = np.column_stack([cfg.system.vs * t, np.full(t.size, cfg.scene.y0), np.zeros(t.size)])
        a, b = tr.prx[MOVING][sel], tr0.prx[MOVING][sel]
        e = (np.linalg.norm(a - p, axis=1) - np.linalg.norm(b - p, axis=1)) - (np.linalg.norm(a - c, axis=1) - np.linalg.norm(b - c, axis=1))
        out = max(out, float(2 * np.pi / cfg.system.wl * np.max(np.abs(e))))
    return out


def compute():
    p = os.path.join(M.CACHE, "item2_joint.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    ter, _ = RC.relative(RC.gauss_mountain(1000.0))
    cfg = M.S.make_config(list(zip(XS, ter(XS))), nrx=4, bxt_max=0.0, tx_rx_channel0=True)
    cfg = replace(cfg, array=M.ArrayGeometry(bat=cfg.array.bat, bxt=np.array(M.B300)))
    tr0 = M.S.build_platform_tracks(cfg)
    F = RC.focuser(cfg, tr0)
    r = dict(xs=XS, prf_op=cfg.PRF_op, vs=cfg.system.vs, t=(np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf)
    r["dt_amb"] = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    r["X"] = r["dt_amb"] * cfg.system.vs
    r["ref"] = F(RC.band(cfg, M.S.generate_reference(cfg, tr0)))
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    hg = ter(np.linspace(-half - 100.0, half + 100.0, 8001))
    for kind in KINDS:
        for eps in EPS:
            key = f"{kind}|{eps:g}"
            if eps == 0 and kind != KINDS[0]:
                continue
            tr = tracks(cfg, kind, eps)
            if key in res:
                res[key]["phi_err"] = phase_error(cfg, tr, tr0, ter)
                continue
            ch = M.S.generate_channels(cfg, tr)
            out = {}
            # nominal model (straight tracks)
            tab0 = height_table(cfg, tr0, hg.min() - 60.0, hg.max() + 60.0)
            Mn = LineModel(cfg, tr0, line_maps(cfg, tr0, ter, tab0))
            g, _ = V.current(Mn, ch)
            r["nominal"] = F(V.to_line(Mn, g))
            # first-order motion compensation to the reference height (h = 0): every channel sample at time t is
            # multiplied by exp(+j 2 pi dr(t) / lambda), dr(t) = range of the true receiver minus range of the nominal one
            # to the point of the reference height at the beam centre (x = v t); then the repository's model (straight
            # tracks) is used. What is left: the topography-dependent part of the motion (and its small aperture part).
            tch = cfg.ta[::cfg.Nrx]
            pc = np.column_stack([cfg.system.vs * tch, np.full(tch.size, cfg.scene.y0), np.zeros(tch.size)])
            chm = ch.copy()
            for i in range(cfg.Nrx):
                dr = (np.linalg.norm(tr.prx[i][::cfg.Nrx] - pc, axis=1) - np.linalg.norm(tr0.prx[i][::cfg.Nrx] - pc, axis=1))
                chm[i] = ch[i] * np.exp(2j * np.pi * dr / cfg.system.wl)
            g, _ = V.current(Mn, chm)
            r["moco"] = F(V.to_line(Mn, g))
            r["std"] = F(M.S.reconstruct(cfg, tr0, chm))
            for k in ("nominal", "moco", "std"):
                out[k] = float(RC.amb_old(r, k))
            out["ref"] = float(RC.amb_old(dict(r, x=r["ref"]), "x"))
            out["h_targets"] = [float(v) for v in ter(XS)]
            out["phi_err"] = phase_error(cfg, tr, tr0, ter)
            res[key] = out
            print(key, {k: round(v, 1) for k, v in out.items() if isinstance(v, float)}, flush=True)
            json.dump(res, open(p, "w"), indent=1)
    json.dump(res, open(p, "w"), indent=1)
    return res


def figure(res):
    plt = M.style()
    fig, axs = plt.subplots(1, 4, figsize=M.fs(12.6, 2.9, 7.2, 3.1), sharey=True)
    for ax, kind in zip(axs[:3], KINDS):
        e = [x for x in EPS if x > 0]
        for k, col, lab in (("std", M.RED, "standard after MoCo"), ("nominal", "#999999", "joint, straight-track model"),
                            ("moco", M.PURPLE, "joint after MoCo to h = 0")):
            ax.semilogx(e, [res[f"{kind}|{x:g}"][k] for x in e], color=col, marker="o", ms=3.5, label=lab)
            ax.axhline(res[f"{KINDS[0]}|0"][k], color=col, lw=0.7, ls=":")
        ax.axhline(res[f"{KINDS[0]}|0"]["ref"], color=M.GREEN, lw=1.0, label="reference")
        ax.set_title(f"{kind}" if os.environ.get("FIG_STYLE") == "report" else
                     f"{kind}: receiver $b_{{xt}}$ = 300 m moves by $\\varepsilon$", fontsize=8.4)
        ax.set_xlabel("baseline change $\\varepsilon$ [m]")
    ax = axs[3]
    mk = dict(drift="^", sine="D", bumps="o")
    for kind in KINDS:
        e = [x for x in EPS if x > 0]
        ax.semilogx([res[f"{kind}|{x:g}"]["phi_err"] for x in e], [res[f"{kind}|{x:g}"]["moco"] for x in e], color=M.PURPLE,
                    marker=mk[kind], ls="", ms=4, label=kind)
    ph = np.logspace(-3, 0.5, 50)
    ax.semilogx(ph, 20 * np.log10(ph) - 9.5, color=M.MUTED, lw=0.8, ls="--", label="$20\\log_{10}\\varphi - 9.5$ dB (fit)")
    ax.axvline(np.pi / 4, color=M.AMBER, lw=0.8, ls=(0, (3, 2)))
    ax.text(np.pi / 4 * 0.9, 2, "$\\pi/4$", color=M.AMBER, ha="right", fontsize=7.5)
    ax.axhline(res[f"{KINDS[0]}|0"]["ref"], color=M.GREEN, lw=1.0)
    ax.set_xlabel("residual phase $\\varphi$ [rad]" if os.environ.get("FIG_STYLE") == "report" else
                  "phase left by the MoCo on a target, $\\varphi$ [rad]")
    ax.set_title("joint vs residual phase" if os.environ.get("FIG_STYLE") == "report" else
                 "joint after MoCo vs the phase that $\\Delta C_0$ ignores", fontsize=8.4)
    REP = os.environ.get("FIG_STYLE") == "report"
    if not REP:
        ax.legend(loc="lower right", fontsize=6.8)
    ax.set_xlim(8e-3, 1.5)
    for a in axs:
        a.set_ylim(-46, 8)
    axs[0].set_ylabel("highest ambiguity [dB]")
    h, l = axs[0].get_legend_handles_labels()
    if REP:
        h2, l2 = axs[3].get_legend_handles_labels()
        h, l = h + h2, l + [f"joint, {x} (right panel)" if x in KINDS else x for x in l2]
    fig.legend(h, l, loc="lower center", ncol=3 if REP else 4, fontsize=8.4 if REP else 7.2,
               bbox_to_anchor=(0.5 if REP else 0.38, 0.0 if REP else -0.06))
    fig.tight_layout(rect=(0, 0.22 if REP else 0.06, 1, 1))
    M.save(fig, "item2_joint")
    plt.close(fig)


if __name__ == "__main__":
    figure(compute())
