"""Item 2 (v30): how C0, C1, C2 of the channel filter change with the height of the scatterer when the cross-track
baseline of the receiver varies along the aperture, b_xt(t) = b_xt,0 + eps(t).

Same computation as the deck's slide 15 (report/scripts/make_figures.py, figure of the coefficients against height): GetCoeffNu of sata1d
(parabola fits of the bistatic-minus-monostatic range history over the common illuminated aperture), scatterer moved in
height on the iso-range of the reference point, differences to the reference point (h = 0). Here the receiver track is
not a straight line: its cross-track position is b_xt,0 + eps(t), exact distances.

Also computed (what a parabola cannot describe): the differential range history of the raised point minus that of the
reference point, d(tau) = [r_Tx + r_Rx - 2 r_Tx](h) - [same](0), on the common aperture:
  phase_ignored_C0only = 2 pi / lambda * max |d(tau) - d(0)|   (all that a model with Delta C0 only ignores)
  phase_resid_parabola = 2 pi / lambda * max |d(tau) - parabola fit|   (what even C0, C1, C2 cannot describe)

Systems: the deck's spaceborne default (b_xt,0 = 300 m), and an airborne X-band and L-band geometry (common_m.airborne_config).
eps(t) shapes (t relative to the closest approach of the reference point, T = one aperture):
  bumps  three smooth bumps of height eps (raised cosine, width 0.08 T) at -0.30 T, +0.05 T, +0.35 T  ("a few instants")
  step   smooth step to eps at +0.20 T (width 0.02 T)
  drift  linear, -eps at -T/2 to +eps at +T/2
  sine   eps sin(2 pi t / (T/2))  (two oscillations per aperture, airborne-like roll / turbulence)
    python3 -I item2_baseline_variation.py        -> fig/item2_*.pdf/png, cache/item2.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common_m as M


def eps_shape(kind, t, T, eps):
    if eps == 0 or kind == "none":
        return np.zeros_like(t)
    if kind == "bumps":
        e = np.zeros_like(t)
        for c in (-0.30, 0.05, 0.35):
            u = np.abs(t - c * T) / (0.04 * T)
            e += np.where(u < 1, 0.5 * (1 + np.cos(np.pi * u)), 0.0)
        return eps * e
    if kind == "step":
        u = np.clip((t - 0.20 * T) / (0.02 * T) + 0.5, 0, 1)
        return eps * (0.5 - 0.5 * np.cos(np.pi * u))
    if kind == "drift":
        return eps * t / (T / 2)
    if kind == "sine":
        return eps * np.sin(2 * np.pi * t / (T / 2))
    raise ValueError(kind)


def aperture(cfg):
    return cfg.system.wl * cfg.scene.r0 / cfg.system.La / cfg.system.vs


def one_case(cfg, kind, eps, heights):
    T = aperture(cfg)
    tc = cfg.scene.x0 / cfg.system.vs
    tr = M.tracks_with_eps(cfg, eps_shape(kind, cfg.ta - tc, T, eps))
    wl, fe = cfg.system.wl, cfg.abw / 2
    cr = M.coeffs(cfg, tr, cfg.scene.ptg, fit=True)
    c_ref, fr = np.array(cr[:4]), cr[4]
    rows = []
    sel = np.abs(cfg.ta - tc) <= 0.48 * T                     # inside the aperture of the reference point (and of the raised one)
    tau = cfg.ta[sel] - tc

    def drange(p):                                            # bistatic minus monostatic range on the absolute time axis
        rt = np.linalg.norm(tr.ptx[sel] - p[None, :], axis=1)
        rr = np.linalg.norm(tr.prx[1][sel] - p[None, :], axis=1)
        return rr - rt

    d_ref = drange(cfg.scene.ptg)
    for h in heights:
        c = M.coeffs(cfg, tr, M.iso_point(cfg, h))
        d = np.array(c[:4]) - c_ref
        dd = drange(M.iso_point(cfg, h)) - d_ref                # what the raised point adds, at the same instants
        d0 = np.interp(0.0, tau, dd)
        p = np.polyfit(tau, dd, 2)
        shift = d[3] - d[1]                                   # the filter uses (Dt - C1) f: Dt and C1 together
        rows.append(dict(h=float(h), dC0=float(d[0]), dC1=float(d[1]), dC2=float(d[2]), dDt=float(d[3]),
                         dshift=float(shift),
                         ph_C0=float(2 * np.pi * abs(d[0]) / wl), ph_C1=float(2 * np.pi * abs(shift) * fe),
                         ph_C2=float(2 * np.pi * abs(d[2]) * wl * fe ** 2),
                         ph_ign=float(2 * np.pi / wl * np.max(np.abs(dd - d0))),
                         ph_lin=float(2 * np.pi / wl * abs(p[1]) * T / 2),
                         ph_quad=float(2 * np.pi / wl * abs(p[0]) * (T / 2) ** 2),
                         ph_res=float(2 * np.pi / wl * np.max(np.abs(dd - np.polyval(p, tau)))),
                         n_tau=int(tau.size)))
    # how well a parabola describes the reference point's own range history (motion error, removed by a MoCo)
    pr = np.polyfit(fr["tau"], fr["d_range"], 2)
    ref_res = float(2 * np.pi / wl * np.max(np.abs(fr["d_range"] - np.polyval(pr, fr["tau"]))))
    return dict(kind=kind, eps=float(eps), rows=rows, ref_resid_phase=ref_res, T=float(T))


SYSTEMS = {
    "space": dict(make=lambda: M.spaceborne_config(300.0), heights=np.arange(0.0, 3001.0, 250.0), h_panel=1000.0,
                  eps=[0.0, 0.1, 1.0, 5.0, 20.0],
                  label="spaceborne (default system): $\\lambda$ = 0.25 m, $r_0$ = 766 km, $v$ = 7689 m/s, $b_{xt,0}$ = 300 m"),
    "airX": dict(make=lambda: M.airborne_config(wl=0.031, La=0.5, bxt=2.0), heights=np.arange(0.0, 601.0, 50.0),
                 h_panel=300.0, eps=[0.0, 0.01, 0.1, 0.5, 1.0],
                 label="airborne X band: $\\lambda$ = 3.1 cm, $r_0$ = 4.2 km, $v$ = 90 m/s, $b_{xt,0}$ = 2 m"),
    "airL": dict(make=lambda: M.airborne_config(wl=0.23, La=1.0, bxt=2.0), heights=np.arange(0.0, 601.0, 50.0),
                 h_panel=300.0, eps=[0.0, 0.01, 0.1, 0.5, 1.0],
                 label="airborne L band: $\\lambda$ = 23 cm, $r_0$ = 4.2 km, $v$ = 90 m/s, $b_{xt,0}$ = 2 m"),
}
KINDS = ["bumps", "step", "drift", "sine"]


def compute():
    p = os.path.join(M.CACHE, "item2.json")
    if os.path.exists(p):
        return json.load(open(p))
    out = {}
    for sk, sd in SYSTEMS.items():
        cfg = sd["make"]()
        info = dict(wl=cfg.system.wl, r0=cfg.scene.r0, v=cfg.system.vs, Ba=cfg.abw, prf=cfg.prf, T=aperture(cfg),
                    La=cfg.system.La, H=cfg.scene.H, Na=cfg.Na, bxt0=float(cfg.array.bxt[1]))
        cases = [one_case(cfg, "none", 0.0, sd["heights"])]
        for k in KINDS:
            for e in sd["eps"][1:]:
                cases.append(one_case(cfg, k, e, sd["heights"]))
                print(sk, k, e, "h=max: C1 %.2e C2 %.2e res %.2e ign %.2e rad" % tuple(
                    cases[-1]["rows"][-1][q] for q in ("ph_C1", "ph_C2", "ph_res", "ph_ign")), flush=True)
        out[sk] = dict(info=info, cases=cases)
    json.dump(out, open(p, "w"), indent=1)
    return out


def figures(res):
    plt = M.style()
    from matplotlib.lines import Line2D
    YT = [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2]
    YTL = ["$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "1", "$10^{2}$"]
    REP = os.environ.get("FIG_STYLE") == "report"
    PI4 = (0, (1, 0.8, 4, 0.8))                     # pi/4 line: dash-dot, distinct from the dashed eps curves
    for sk, sd in SYSTEMS.items():
        R = res[sk]
        fe = R["info"]["Ba"] / 2
        # --- (1) slide-15 style panels, per eps shape: the constant-baseline curves + the largest eps of each shape
        fig, axs = plt.subplots(2, 4, figsize=M.fs(11.0, 4.6, 7.2, 5.4), sharey=True)
        for j, kind in enumerate(KINDS):
            sel = [c for c in R["cases"] if c["kind"] in ("none", kind)]
            a, b = axs[0, j], axs[1, j]
            for c in sel:
                ls = "-" if c["kind"] == "none" else (0, (4, 1.5)) if c["eps"] == sd["eps"][-1] else (0, (1, 1.2))
                if c["kind"] != "none" and c["eps"] not in (sd["eps"][2], sd["eps"][-1]):
                    continue
                hs = np.array([r["h"] for r in c["rows"]])
                k1 = int(np.argmin(np.abs(hs - sd["h_panel"])))
                r1 = c["rows"][k1]
                f = np.linspace(-fe, fe, 801)
                a.semilogy(f, np.full_like(f, r1["ph_C0"]), color=M.C_C0, ls=ls, lw=1.3)
                a.semilogy(f, np.maximum(2 * np.pi * abs(r1["dshift"]) * np.abs(f), 1e-12), color=M.C_C1, ls=ls, lw=1.3)
                a.semilogy(f, np.maximum(2 * np.pi * abs(r1["dC2"]) * R["info"]["wl"] * f ** 2, 1e-12), color=M.C_C2, ls=ls,
                           lw=1.3)
                for q, col in (("ph_C0", M.C_C0), ("ph_C1", M.C_C1), ("ph_C2", M.C_C2), ("ph_res", "#444444")):
                    b.semilogy(hs[1:], np.maximum([r[q] for r in c["rows"][1:]], 1e-12), color=col, ls=ls, lw=1.3)
            for ax in (a, b):
                ax.axhline(np.pi / 4, color=M.MUTED, lw=0.8, ls=PI4)
                ax.set_ylim(1e-9, 5e2)
                ax.set_yticks(YT)
                ax.set_yticklabels(YTL)
            a.set_title(f"{kind}: $h$ = {sd['h_panel']:.0f} m")
            a.set_xlabel("Doppler frequency $f$ [Hz]")
            b.set_xlabel("height $h$ [m]")
            b.set_title(f"{kind}: band edge $f=B_a/2$")
        axs[0, 0].set_ylabel("phase change [rad]")
        axs[1, 0].set_ylabel("phase change [rad]")
        hl = [Line2D([], [], color=M.C_C0, label="offset $2\\pi|\\Delta C_0|/\\lambda$"),
              Line2D([], [], color=M.C_C1, label="shift $2\\pi|\\Delta(\\Delta t - C_1)| f$"),
              Line2D([], [], color=M.C_C2, label="bend $2\\pi|\\Delta C_2|\\lambda f^2$"),
              Line2D([], [], color="#444444", label="not a parabola (residual)"),
              Line2D([], [], color=M.MUTED, ls=PI4, label="$\\pi/4$ (for scale)"),
              Line2D([], [], color=M.INK, ls="-", label="constant baseline"),
              Line2D([], [], color=M.INK, ls=(0, (1, 1.2)), label=f"$\\varepsilon$ = {sd['eps'][2]:g} m"),
              Line2D([], [], color=M.INK, ls=(0, (4, 1.5)), label=f"$\\varepsilon$ = {sd['eps'][-1]:g} m")]
        fig.legend(handles=hl, loc="lower center", ncol=4 if REP else 8, bbox_to_anchor=(0.5, 0.0 if REP else -0.04),
                   fontsize=8.4 if REP else 7.2, handlelength=2.6, columnspacing=1.2)
        fig.suptitle(sd["label"], x=0.01, ha="left", fontsize=9.5 if REP else 9)
        fig.tight_layout(rect=(0, 0.1 if REP else 0.04, 1, 0.97))
        M.save(fig, f"item2_panels_{sk}")
        plt.close(fig)
    # --- (2) summary: worst phase at the top height vs eps, every shape, every system
    fig, axs = plt.subplots(1, 3, figsize=M.fs(11.0, 3.0, 7.2, 3.0), sharey=True)
    mk = dict(bumps="o", step="s", drift="^", sine="D")
    for ax, (sk, sd) in zip(axs, SYSTEMS.items()):
        R = res[sk]
        c0 = R["cases"][0]["rows"][-1]
        for q, col, lab in (("ph_C1", M.C_C1, "shift ($\\Delta C_1$)"), ("ph_C2", M.C_C2, "bend ($\\Delta C_2$)"),
                            ("ph_res", "#444444", "not a parabola"), ("ph_ign", M.RED, "all but $\\Delta C_0$")):
            for kind in KINDS:
                cs = [c for c in R["cases"] if c["kind"] == kind]
                ax.loglog([c["eps"] for c in cs], [max(c["rows"][-1][q], 1e-12) for c in cs], color=col, marker=mk[kind],
                          ms=3.5, lw=0.9, alpha=0.9)
            ax.axhline(max(c0[q], 1e-12), color=col, lw=0.6, ls=":")
        ax.axhline(np.pi / 4, color=M.MUTED, lw=0.9, ls=PI4)
        ax.axhline(c0["ph_C0"], color=M.C_C0, lw=1.0)
        ax.set_title(sd["label"].split(":")[0] + f", $h$ = {SYSTEMS[sk]['heights'][-1]:.0f} m", fontsize=9 if REP else 8)
        ax.set_xlabel("baseline change $\\varepsilon$ [m]")
        ax.set_ylim(1e-9, 1e3)
    axs[0].set_ylabel("phase at the band edge [rad]")
    hl = [Line2D([], [], color=M.C_C1, label="shift $\\Delta(\\Delta t - C_1)$"), Line2D([], [], color=M.C_C2, label="bend $\\Delta C_2$"),
          Line2D([], [], color="#444444", label="not a parabola"), Line2D([], [], color=M.RED, label="all but $\\Delta C_0$"),
          Line2D([], [], color=M.C_C0, label="offset $\\Delta C_0$"),
          Line2D([], [], color=M.MUTED, ls=PI4, label="$\\pi/4$"),
          Line2D([], [], color=M.MUTED, ls=":", label="constant baseline")] + \
         [Line2D([], [], color=M.INK, marker=mk[k], ls="", ms=4, label=k) for k in KINDS]
    fig.legend(handles=hl, loc="lower center", ncol=4 if REP else 11, bbox_to_anchor=(0.5, 0.0 if REP else -0.07),
               fontsize=8.4 if REP else 7.0, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.27 if REP else 0.06, 1, 1))
    M.save(fig, "item2_summary")
    plt.close(fig)


def table(res):
    lines = ["| system | shape | eps [m] | dC0 phase | shift d(Dt-C1) phase (band edge) | dC2 phase | not-a-parabola | all but dC0 | ref. point not-a-parabola |",
             "|---|---|---|---|---|---|---|---|---|"]
    for sk, sd in SYSTEMS.items():
        for c in res[sk]["cases"]:
            r = c["rows"][-1]
            lines.append(f"| {sk} (h={r['h']:.0f} m) | {c['kind']} | {c['eps']:g} | {r['ph_C0']:.3g} | {r['ph_C1']:.2e} | "
                         f"{r['ph_C2']:.2e} | {r['ph_res']:.2e} | {r['ph_ign']:.2e} | {c['ref_resid_phase']:.2e} |")
    open(os.path.join(M.CACHE, "item2_table.md"), "w").write("\n".join(lines) + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    res = compute()
    for sk in SYSTEMS:
        print(sk, res[sk]["info"])
    figures(res)
    print(table(res))
