"""Check of the topographic residual Delta C0 against the exact range-history fits of the simulator.

Geometry (flat earth, as the simulator): transmitter at (v t, 0, H), receiver i at (v t - b_at,i, b_xt,i, H), a place of
height h on the iso-range of the reference point at (x, sqrt(r0^2 - (H - h)^2), h).
C0,i is the difference between the shortest bistatic path r_T + r_R,i and the shortest monostatic path 2 r_T
(GetCoeffNu), i.e. r_R,i,min - r_T,min. At closest approach
    r_R,i = sqrt((y_h - b)^2 + (H - h)^2) = r0 - b sin(theta_h) + b^2 cos^2(theta_h) / (2 r0) + ...
so C0,i(h) ~ -b sin(theta_h) and, with cos(theta_h) = (H - h) / r0,
    Delta C0,i(h) = C0,i(h) - C0,i(0) ~ -b (sin theta_h - sin theta_0) ~ -b h / (r0 tan theta_0).
Second order: relative error of the linear law in h ~ h / (2 r0 cos theta sin^2 theta); the b^2 term is b sin(theta) / r0
relative to the first-order term.

    python3 -I verify_dC0.py      -> cache/dC0_check.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common_m as M
from dataclasses import replace
from sata1d.config import ArrayGeometry
from sata1d.sata import residual_C0

S = M.S


def main():
    cfg = S.make_config([(0.0, 0.0)], nrx=4, bxt_max=0.0, tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.array(M.B300)))
    tracks = S.build_platform_tracks(cfg)
    H, r0, y0 = cfg.scene.H, cfg.scene.r0, cfg.scene.y0
    th0 = np.arccos(H / r0)
    heights = [100.0, 500.0, 1000.0, 2000.0, 3000.0]
    rows = []
    for h in heights:
        p = cfg.scene.ptg + np.array([0.0, np.sqrt(r0 ** 2 - (H - h) ** 2) - y0, h])
        for i in (1, 2, 3):
            b = cfg.array.bxt[i]
            exact = residual_C0(cfg, tracks, p, i)
            lin = -b * h / (r0 * np.tan(th0))
            thh = np.arccos((H - h) / r0)
            geo = -b * (np.sin(thh) - np.sin(th0))
            rows.append((h, b, exact, lin, geo))
    lines = ["# Delta C0: exact fit vs formula (deck system, look angle %.2f deg, r0 = %.1f km)" % (np.degrees(th0), r0 / 1e3),
             "",
             "| h [m] | b_xt [m] | exact [m] | -b h/(r0 tan th) [m] | rel. error | -b (sin th_h - sin th_0) [m] | rel. error |",
             "|---|---|---|---|---|---|---|"]
    for h, b, ex, lin, geo in rows:
        lines.append("| %g | %g | %.4f | %.4f | %+.2f %% | %.4f | %+.3f %% |" % (h, b, ex, lin, 100 * (lin - ex) / ex, geo, 100 * (geo - ex) / ex))
    out = os.path.join(M.CACHE, "dC0_check.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
