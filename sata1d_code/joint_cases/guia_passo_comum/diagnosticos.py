"""
Numbers and curves for the guide on common-step baselines (joint_cases/guia_passo_comum/):
  1. the reference filters of joint.py on the parcel grid: per-channel delay tau_i*P (cycles per parcel), how far
     H is from the ideal DFT exp(j pi i q / 2)
  2. dC0 per metre of height for each channel (residual_C0) against -bxt / (r0 tan theta)
  3. conditioning per block along azimuth (worst bin; occupied parcels and full system) for pairs of cases, with
     the map of the sweep and with the nearest-target map
  4. case 33: height difference between neighbouring occupied parcels read in the map (DEM vs nearest target)
  5. sweep pages with the nearest-target map (case 33, [0,20,60,140] and [0,150,100,50]; case 29, U(0,500))
Output: joint_cases/guia_passo_comum/diag.json and paginas_mapa_alvos/<baseline>/<case>.png
    SATA_LUT_CACHE=/tmp/lutcache python joint_cases/guia_passo_comum/diagnosticos.py      # from sata1d_code
"""
import json
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
JC = os.path.dirname(HERE)
sys.path.insert(0, JC)
sys.path.insert(0, os.path.dirname(JC))
import sata1d as S
import sata1d.joint as J
from sata1d.joint import joint_maps
from sata1d.sata import subaperture_sizing, residual_C0
import run_joint_sweep as W

OUT = HERE


def grid(cfg, tr):
    Nrx, Nsb, P = cfg.Nrx, cfg.Nsb, cfg.PRF_op
    wl, v, r0 = cfg.system.wl, cfg.system.vs, cfg.scene.r0
    coef = J.reference_coefficients(cfg, tr)
    Tsub, hop, Nzp = subaperture_sizing(r0, P, v, wl, W.SATA_OSF)
    f = (-Nsb * P / 2.0 + np.arange(Nsb) * P)[:, None] + (np.arange(Nzp) * P / Nzp)[None, :]
    a = r0 * np.tan(np.arcsin(np.clip(wl * f / (2.0 * v), -1, 1))) / v * P
    Hq = np.zeros((Nzp, Nrx, Nsb), complex)
    for i in range(Nrx):
        C0, C1, C2, Dt = coef[i]
        Hq[:, i, :] = np.exp(-2j * np.pi * (C0 / wl + (-C1 + Dt) * f.T + C2 * wl * f.T ** 2))
    occ = np.abs(f.T) <= 2.0 * v * np.sin(cfg.theta_tx / 2.0) / wl + 4.0 * P / Tsub
    return coef, f, a, Hq, occ, Tsub, hop, Nzp


def maps_of(cid, btag, kind):
    c = W.case_info(cid)
    cfg = W.make_cfg(c, btag); tr = S.build_platform_tracks(cfg)
    if c["terrain"] is None or kind == "targets":
        m = np.array(joint_maps(cfg, tr, S.delta_C0_per_target(cfg, tr)))
    else:
        m = np.array([W._lut(cfg, tr, i, c["terrain"], cid) for i in range(cfg.Nrx)])
    return c, cfg, tr, m


def along(cid, btag, kind):
    """Worst-bin smallest/largest singular value per block, occupied parcels and full system."""
    c, cfg, tr, maps = maps_of(cid, btag, kind)
    coef, f, a, Hq, occ, Tsub, hop, Nzp = grid(cfg, tr)
    n = occ.sum(1)
    tc, ro, rf = [], [], []
    for start in range(0, cfg.Na_ch, hop):
        cc = start + 0.5 * Tsub
        pos = np.clip(np.round(cc + a).astype(int), 0, cfg.Na_ch - 1)
        G = Hq * np.exp(-2j * np.pi * maps[:, pos].transpose(2, 0, 1) / cfg.system.wl)
        svf = np.linalg.svd(G, compute_uv=False)
        svo = np.linalg.svd(G * occ[:, None, :], compute_uv=False)
        ro.append(float((svo[np.arange(Nzp), n - 1] / svo[:, 0]).min()))
        rf.append(float((svf[:, -1] / svf[:, 0]).min()))
        tc.append(float((cc - cfg.Na_ch / 2) / cfg.PRF_op))
    return {"t": tc, "occ": ro, "full": rf}


def main():
    if "--pages" in sys.argv:
        return pages()
    out = {}
    # 1. filters on the parcel grid
    c5 = W.case_info("C05")
    for b in ("rnd100", "s50", "s300"):
        cfg = W.make_cfg(c5, b); tr = S.build_platform_tracks(cfg)
        coef, f, a, Hq, occ, Tsub, hop, Nzp = grid(cfg, tr)
        P, wl = cfg.PRF_op, cfg.system.wl
        tauP = (-coef[:, 1] + coef[:, 3]) * P                         # cycles per parcel step
        # deviation from the ideal DFT, after removing row and column phases (bin 0)
        Hb = Hq[Nzp // 2]
        D = np.exp(1j * np.pi * np.outer(np.arange(4), np.arange(4)) / 2.0)
        R = Hb / D
        R = R / R[:, :1] / R[:1, :] * R[0, 0]
        out[f"filters_{b}"] = {"tauP": tauP.tolist(), "C2wlP2": (coef[:, 2] * wl * P ** 2).tolist(),
                               "dev_deg_max": float(np.degrees(np.abs(np.angle(R))).max()),
                               "dev_deg": np.degrees(np.angle(R)).round(2).tolist(),
                               "bat": cfg.array.bat.tolist(), "bxt": cfg.array.bxt.tolist(),
                               "X": float(cfg.PRF_op * wl * cfg.scene.r0 / (2 * cfg.system.vs)),
                               "d": float(cfg.system.vs / cfg.prf), "D": float(cfg.system.vs / cfg.PRF_op)}
        print(b, "tau*P", np.round(tauP, 4), "dev max", out[f"filters_{b}"]["dev_deg_max"], flush=True)
    # 2. dC0 per metre
    cfg = W.make_cfg(c5, "s50"); tr = S.build_platform_tracks(cfg)
    r0, H, y0 = cfg.scene.r0, cfg.scene.H, cfg.scene.y0
    hs = np.array([-1000.0, -500.0, 0.0, 250.0, 500.0, 1000.0])
    dc = [[residual_C0(cfg, tr, cfg.scene.ptg + np.array([0.0, np.sqrt(r0 ** 2 - (H - h) ** 2) - y0, h]), i) for h in hs] for i in range(cfg.Nrx)]
    out["dC0"] = {"h": hs.tolist(), "dC0": dc, "bxt": cfg.array.bxt.tolist(), "r0": r0,
                  "look_deg": float(np.degrees(np.arccos(H / r0)))}
    print("dC0 per m:", [(dc[i][4] - dc[i][1]) / 1000 for i in range(4)], "look", out["dC0"]["look_deg"], flush=True)
    # 3. conditioning along azimuth
    pairs = [("C05", "s50", None), ("C05", "rnd500", None), ("C09", "s100", None), ("C09", "s300", None),
             ("C09", "s50", None), ("C09", "perm50", None),
             ("C31", "s20", None), ("C31", "c50", None),
             ("C33", "uneven", None), ("C33", "uneven", "targets"), ("C33", "dec50", None), ("C33", "dec50", "targets"),
             ("C29", "rnd500", None), ("C29", "rnd500", "targets")]
    for cid, b, kind in pairs:
        out[f"along_{cid}_{b}_{kind or 'sweep'}"] = r = along(cid, b, kind)
        print(cid, b, kind, "min occ", f"{min(r['occ']):.3g}", "min full", f"{min(r['full']):.3g}", flush=True)
    # 4. case 33: height differences between neighbouring occupied parcels
    for kind in (None, "targets"):
        c, cfg, tr, maps = maps_of("C33", "uneven", kind)
        coef, f, a, Hq, occ, Tsub, hop, Nzp = grid(cfg, tr)
        ds = cfg.system.vs / cfg.PRF_op
        hs_t = np.asarray(c["hs"]); xs_t = np.asarray(c["xs"])
        dh = []
        for start in range(0, cfg.Na_ch, hop):
            cc = start + 0.5 * Tsub
            x = (np.round(cc + a) - cfg.Na_ch / 2) * ds                                     # [Nsb, Nzp] parcel positions
            if kind is None:
                h = c["terrain"](x)
            else:
                h = hs_t[np.abs(x[..., None] - xs_t).argmin(-1)]
            for k in range(Nzp):
                qs = np.flatnonzero(occ[k])
                dh += list(np.diff(h[qs, k]))
        out[f"dh_C33_{kind or 'sweep'}"] = np.round(np.array(dh), 1).tolist()
        print("C33", kind, "dh percentiles", np.percentile(np.abs(dh), [10, 50, 90, 99]), flush=True)
    json.dump(out, open(os.path.join(OUT, "diag.json"), "w"))


def pages():
    """5. sweep pages with the nearest-target map (run with --pages)."""
    for cid, b in (("C33", "uneven"), ("C33", "dec50"), ("C29", "rnd500")):
        r = W.run(cid, b, out=os.path.join(OUT, "paginas_mapa_alvos"), map_kind="targets")
        print("page", cid, b, {k: round(r[k], 1) for k in ("joint_occ", "joint_full", "whole", "none", "ref")}, flush=True)


if __name__ == "__main__":
    main()
