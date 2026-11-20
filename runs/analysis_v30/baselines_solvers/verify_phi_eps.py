"""Check of the residual-phase law of a cross-track baseline that varies during the aperture.

Geometry of the simulator (flat earth): transmitter at (v t, 0, H), receiver i at (v t - b_at,i, b_xt,i + eps(t), H), a place
raised by h on the iso-range of the reference point at (x, sqrt(r0^2 - (H - h)^2), h); look angle theta0 at h = 0.

The receiver range to a point P at ground range y_P changes with the cross-track position b of the receiver as
    d r_Rx / d b = (b - y_P) / r_Rx  ~  -sin theta_P                         (cross-track component of the line of sight)
so a change eps(t) moves it by  -eps(t) sin theta_P  (first order in eps / r0).
A motion compensation to the reference height h = 0 removes this change for the point c(t) of h = 0 at the beam centre,
-eps(t) sin theta_0. What is left on a place raised by h is
    delta r(t; h) = -eps(t) (sin theta_h - sin theta_0).
On the iso-range, cos theta_h = (H - h) / r0, so d theta / d h = 1 / (r0 sin theta) and d(sin theta)/dh = 1 / (r0 tan theta):
    delta r(t; h) ~ -eps(t) h / (r0 tan theta_0),     phi_eps(t) = (2 pi / lambda) eps(t) h / (r0 tan theta_0)   (one way).

This script compares the law with the exact distances (the same expression as phase_error() of item2_joint.py), for a
constant eps and for the four shapes of item2_baseline_variation.py, spaceborne and airborne.
    python3 -I verify_phi_eps.py      -> cache/phi_eps_check.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common_m as M
import item2_baseline_variation as I2


def exact_max_phase(cfg, kind, eps, h, x=None):
    """2 pi / lambda * max_t |residual range| left by the MoCo to h = 0 on a place raised by h (receiver 1 moves)."""
    T = I2.aperture(cfg)
    x = cfg.scene.x0 if x is None else x
    sel = np.abs(cfg.ta - x / cfg.system.vs) <= 0.48 * T
    t = cfg.ta[sel]
    e = I2.eps_shape(kind if kind != "const" else "none", t - x / cfg.system.vs, T, eps)
    if kind == "const":
        e = np.full_like(t, eps)
    vs, H = cfg.system.vs, cfg.scene.H
    b0 = cfg.array.bxt[1]
    rx_true = np.column_stack([vs * t - cfg.array.bat[1], b0 + e, np.full(t.size, H)])
    rx_nom = np.column_stack([vs * t - cfg.array.bat[1], np.full(t.size, b0), np.full(t.size, H)])
    p = M.iso_point(cfg, h, x)
    c = np.column_stack([vs * t, np.full(t.size, cfg.scene.y0), np.zeros(t.size)])
    nr = lambda a, b: np.linalg.norm(a - b, axis=1)
    d = (nr(rx_true, p) - nr(rx_nom, p)) - (nr(rx_true, c) - nr(rx_nom, c))
    return float(2 * np.pi / cfg.system.wl * np.max(np.abs(d))), float(np.max(np.abs(e)))


def law(cfg, eps_max, h):
    th0 = np.arccos(cfg.scene.H / cfg.scene.r0)
    return 2 * np.pi / cfg.system.wl * eps_max * h / (cfg.scene.r0 * np.tan(th0)), np.degrees(th0)


def main():
    rows = ["| system | shape | eps [m] | h [m] | exact max phase [rad] | law (2pi/lambda) eps h/(r0 tan theta0) [rad] | ratio |",
            "|---|---|---|---|---|---|---|"]
    cases = [("space", M.spaceborne_config(300.0), [0.1, 1.0, 5.0], [100.0, 1000.0, 3000.0]),
             ("airX", M.airborne_config(wl=0.031, La=0.5, bxt=2.0), [0.004, 0.01, 0.1], [50.0, 100.0, 300.0]),
             ("airL", M.airborne_config(wl=0.23, La=1.0, bxt=2.0), [0.01, 0.03, 0.1], [100.0, 300.0])]
    worst = 0.0
    for name, cfg, epss, hs in cases:
        for kind in ("const", "drift", "sine", "step", "bumps"):
            for eps in epss:
                for h in hs:
                    ex, em = exact_max_phase(cfg, kind, eps, h)
                    lw, th = law(cfg, em, h)
                    worst = max(worst, abs(ex / lw - 1))
                    rows.append(f"| {name} (theta0 = {th:.1f} deg) | {kind} | {eps:g} | {h:g} | {ex:.4g} | {lw:.4g} | {ex / lw:.3f} |")
    # allowed eps for 0.02 rad
    rows += ["", "Allowed eps for phi = 0.02 rad: eps = 0.02 lambda r0 tan theta0 / (2 pi h)", ""]
    for name, cfg, h in (("space", M.spaceborne_config(300.0), 1000.0), ("airX", M.airborne_config(wl=0.031, La=0.5, bxt=2.0), 100.0),
                         ("airL", M.airborne_config(wl=0.23, La=1.0, bxt=2.0), 100.0)):
        th0 = np.arccos(cfg.scene.H / cfg.scene.r0)
        rows.append(f"- {name}: h = {h:g} m, lambda = {cfg.system.wl:g} m, r0 = {cfg.scene.r0:.0f} m, theta0 = {np.degrees(th0):.1f} deg -> "
                    f"eps = {0.02 * cfg.system.wl * cfg.scene.r0 * np.tan(th0) / (2 * np.pi * h) * 1e3:.2f} mm")
    rows.append("")
    rows.append(f"Largest relative deviation of the law from the exact distances: {worst * 100:.2f} %")
    out = "\n".join(rows) + "\n"
    open(os.path.join(M.CACHE, "phi_eps_check.md"), "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
