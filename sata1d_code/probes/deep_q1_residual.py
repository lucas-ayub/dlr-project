"""
Q1, part 1: the residual range of an elevated iso-range target, exactly.
  Transmitter at (v t, 0, H), receiver i at (v t - bat_i, bxt_i, H); target (x, y, h) with
  y chosen so that the monostatic slant range is r0 for every height (iso-range).
  Monostatic range is then identical for all heights, so the residual w.r.t. the flat target is
      dR_i(t) = |prx_i(t) - p(h)| - |prx_i(t) - p(0)|,
  a pure change of the receiver's closest-approach range rho_i: rho_i(h)^2 = r0^2 - 2 y(h) bxt_i + bxt_i^2.
  Along the aperture dR_i(t) = d_rho_i / sqrt(1 + (v tau / rho)^2) ~ d_rho_i (1 - beta^2 / 2):
  the Doppler dependence is second order in the azimuth angle beta = lambda f / (2 v).
Outputs a table: analytic d_rho vs GetCoeffNu delta C0, phase, and its variation across the band
(exact from the range history, and from the beta^2/2 formula), plus the sub-band-centre offsets.
"""
import numpy as np
import sata1d as S
from sata1d.reconstruction import GetCoeffNu

WL = 0.25


def table(heights=(100, 500, 1000, 3000, 10000, 37000)):
    rows = []
    for h in heights:
        cfg = S.make_config([(0.0, h)]); tr = S.build_platform_tracks(cfg)
        p, p0 = cfg.scene.points[1], cfg.scene.ptg
        r0, v = cfg.scene.r0, cfg.system.vs
        f_edge = cfg.abw / 2
        beta_edge = np.arcsin(WL * f_edge / (2 * v))
        for i in range(cfg.Nrx):
            b = cfg.array.bxt[i]
            rho_h = np.sqrt(r0 ** 2 - 2 * p[1] * b + b ** 2)
            rho_0 = np.sqrt(r0 ** 2 - 2 * p0[1] * b + b ** 2)
            drho = rho_h - rho_0
            com = (tr.ptx, tr.prx[i], tr.vtx, tr.vrx[i], tr.ptx, tr.vtx, cfg.prf, WL, cfg.ta,
                   cfg.sq_tx, cfg.sq_rx[i], cfg.theta_tx, cfg.theta_rx[i])
            dC0 = GetCoeffNu(p, *com)[0] - GetCoeffNu(p0, *com)[0]
            # exact residual along the aperture
            dR = np.linalg.norm(tr.prx[i] - p, axis=1) - np.linalg.norm(tr.prx[i] - p0, axis=1)
            tau = cfg.ta - cfg.array.bat[i] / v
            beta = np.arctan(v * tau / rho_0)
            sel = np.abs(beta) <= beta_edge
            var_exact = np.ptp(dR[sel])                                  # m, across the processed band
            var_formula = abs(drho) * (1 - np.cos(beta_edge))            # d_rho (1 - cos beta) ~ d_rho beta^2 / 2
            phi = 2 * np.pi * drho / WL
            sub = [phi * (1 - np.cos(np.arcsin(WL * fk / (2 * v)))) for fk in (-750, -250, 250, 750)]
            rows.append(dict(h=h, ch=i, bxt=b, drho_mm=drho * 1e3, dC0_mm=dC0 * 1e3, phi_deg=np.degrees(phi),
                             var_exact_deg=np.degrees(2 * np.pi * var_exact / WL),
                             var_formula_deg=np.degrees(2 * np.pi * var_formula / WL),
                             sub_deg=[np.degrees(s) for s in sub]))
    return rows


if __name__ == "__main__":
    import csv
    rows = table()
    for r in rows:
        print(f"h={r['h']:6.0f} ch{r['ch']} bxt={r['bxt']:5.1f}  d_rho={r['drho_mm']:10.3f} mm  dC0(GetCoeffNu)={r['dC0_mm']:10.3f} mm  "
              f"phi={r['phi_deg']:9.1f} deg  variation across band: exact {r['var_exact_deg']:.4f} deg, "
              f"formula {r['var_formula_deg']:.4f} deg | sub-band centres offset [deg]: "
              + " ".join(f"{s:.4f}" for s in r['sub_deg']))
    with open("probes/deep_q1_residual.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "sub_deg"] + ["sub_m750", "sub_m250", "sub_p250", "sub_p750"])
        w.writeheader()
        for r in rows:
            d = {k: v for k, v in r.items() if k != "sub_deg"}
            d.update(dict(zip(["sub_m750", "sub_m250", "sub_p250", "sub_p750"], r["sub_deg"])))
            w.writerow(d)
