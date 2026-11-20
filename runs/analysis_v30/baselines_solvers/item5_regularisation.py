"""Item 5 (v30): the joint reconstruction s = A g solved with the methods of three papers on regularised inversion (RelSea L-curve, statistical regularisation / BMR / WISE, MARIA)
(solvers.py), compared on the slide-8 mountain (5 targets), a dense point-target scene (40 targets on the mountain line)
and a real SRTM line (western Sierra foothills, 20 targets 750 m apart, as the deck's 20-target slide), without noise
and with complex white Gaussian noise added to the channels (SNR = mean channel power / noise power = 30, 20, 10 dB).

Metrics (common.py of ../joint_height_dense, the report's definitions):
  amb      highest ambiguity: the largest level within +-154 m of x_p + m X (0 < |m| <= floor(1281 Hz / PRF_op)) of every
           target p, outside +-154 m of every target, dB re. the reference peak (the report's metric)
  amb_new, floor, peak_loss   R's checked metric (common_r.amb_new): the same ambiguity level recomputed, the highest
           level anywhere outside +-154 m of the targets (noise, sidelobes, artefacts) and the weakest target peak re. the
           reference peak
  nmse     20 log10 ||y - ref|| / ||ref|| of the focused complex line (same band)
  w3       3-dB width [m] of the main lobe of the target closest to the centre
  calls    number of A and A^H applications; sec: run time
    python3 -I item5_regularisation.py [check]      -> cache/item5.json, fig/item5_*.pdf/png
"""
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common_m as M
import solvers as V

RDIR = M.HERE
sys.path.insert(1, RDIR)
import common_r as RC                                                   # noqa: E402  (metrics and terrains, snapshot of R/common.py)
from sata1d.joint_line import LineModel, height_table, line_maps, reconstruct_joint_line  # noqa: E402

MOUNTAIN_X = np.array([-8000.0, -550.0, 650.0, 2900.0, 4100.0])
DENSE_X = -7125.0 + 750.0 * np.arange(20)


def scenes():
    ter, _ = RC.relative(RC.gauss_mountain(1000.0))
    rng = np.random.default_rng(30)
    x40 = np.sort(-19500.0 + 1000.0 * np.arange(40) + rng.uniform(-300, 300, 40))
    a40 = np.exp(2j * np.pi * rng.uniform(size=40))
    real, _ = RC.real_terrain("foothills")
    rng = np.random.default_rng(31)
    xc = np.sort(rng.uniform(-20000.0, 20000.0, 1500))                 # distributed-like: 1500 scatterers, ~27 m apart,
    ac = (rng.standard_normal(1500) + 1j * rng.standard_normal(1500)) / np.sqrt(2)   # complex Gaussian amplitudes
    return {"mountain": (MOUNTAIN_X, ter, None),
            "dense40": (x40, ter, a40),
            "foothills20": (DENSE_X, real, None),
            "clutter1500": (xc, ter, ac)}


def setup(xs, terrain, amps=None, bxt=M.B300):
    xs = np.asarray(xs, float)
    cfg = M.S.make_config(list(zip(xs, terrain(xs))), nrx=len(bxt), bxt_max=0.0, amplitudes=amps, tx_rx_channel0=True)
    cfg = replace(cfg, array=M.ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy()))
    tr = M.S.build_platform_tracks(cfg)
    ch = M.S.generate_channels(cfg, tr)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    hg = terrain(np.linspace(-half - 100.0, half + 100.0, 8001))
    tab = height_table(cfg, tr, hg.min() - 60.0, hg.max() + 60.0)
    dC0 = line_maps(cfg, tr, terrain, tab)
    model = LineModel(cfg, tr, dC0)
    F = RC.focuser(cfg, tr)
    r = dict(xs=xs, Na=cfg.Na, prf=cfg.prf, prf_op=cfg.PRF_op, vs=cfg.system.vs,
             t=(np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf, abw=cfg.abw)
    r["dt_amb"] = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    r["X"] = r["dt_amb"] * cfg.system.vs
    r["ref"] = F(RC.band(cfg, M.S.generate_reference(cfg, tr)))
    return cfg, tr, ch, dC0, model, F, r


def metrics(r, key):
    if len(r["xs"]) > 100:                         # distributed-like scene: no isolated targets, only the error
        return dict(nmse=RC.nmse_db(r, key))
    a_new, floor, ploss = RC.amb_new(r, key)
    xc = r["xs"][int(np.argmin(np.abs(r["xs"])))]
    w3, pk = RC.width3db(r, key, xc)
    return dict(amb=float(RC.amb_old(r, key)), amb_new=a_new, floor=floor, peak_loss=ploss, nmse=RC.nmse_db(r, key), w3=float(w3),
                peak=float(pk))


def check_baseline():
    xs, ter, amps = scenes()["mountain"]
    cfg, tr, ch, dC0, model, F, r = setup(xs, ter, amps)
    y1 = reconstruct_joint_line(cfg, tr, ch, dC0)
    g, info = V.current(model, ch)
    y2 = V.to_line(model, g)
    print("current solver vs reconstruct_joint_line: rel. difference %.2e" % (np.linalg.norm(y1 - y2) / np.linalg.norm(y1)))
    r["joint"] = F(y1)
    print("mountain joint amb %.1f dB (deck: -40.8)" % RC.amb_old(r, "joint"))


METHODS = ["current", "lcurve", "lcurve2", "discrepancy", "bmr", "maria", "wise"]   # "maria_exact": see solvers.diag_ARA


EXACT = False


def run_one(model, F, r, ch, N0_true, cfg, tr):
    out = {}
    N0 = N0_true
    t0 = time.time()
    r["std"] = F(M.S.reconstruct(cfg, tr, ch))
    out["standard"] = dict(metrics(r, "std"), sec=time.time() - t0)
    for m in METHODS:
        if m == "discrepancy" and N0 is None:
            continue
        if m == "current":
            g, info = V.current(model, ch)
        elif m == "lcurve":
            g, info = V.lcurve(model, ch, passes=1)
        elif m == "lcurve2":
            g, info = V.lcurve(model, ch, passes=2)
        elif m == "discrepancy":
            g, info = V.discrepancy(model, ch, N0)
        elif m == "bmr":
            g, info = V.bmr(model, ch, N0 if N0 is not None else 0.0)
        elif m == "maria_exact":
            if not EXACT:
                continue
            g, info = V.adaptive(model, ch, kind="maria", N0=N0, diag="hutch")
        else:
            g, info = V.adaptive(model, ch, kind=m, N0=N0)
        r[m] = F(V.to_line(model, g))
        out[m] = dict(metrics(r, m), **{k: v for k, v in info.items() if k not in ("change",)})
        o = out[m]
        print(f"   {m:12s} amb {o.get('amb', np.nan):6.1f}  floor {o.get('floor', np.nan):6.1f}  "
              f"ploss {o.get('peak_loss', np.nan):5.1f}  nmse {o['nmse']:6.1f}  mu/smax {o.get('mu_rel', np.nan):.2e}  "
              f"calls {o['calls_A']}+{o['calls_AH']}  {o['sec']:.1f}s", flush=True)
    return out


def compute(snrs=(None, 30.0, 20.0, 10.0), seeds=(0, 1)):
    p = os.path.join(M.CACHE, "item5.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    lines = {}
    for name, (xs, ter, amps) in scenes().items():
        cfg, tr, ch0, dC0, model, F, r = setup(xs, ter, amps)
        r_ref = dict(r)
        for snr in snrs:
            for seed in (seeds if snr is not None else (0,)):
                key = f"{name}|{'inf' if snr is None else int(snr)}|{seed}"
                if key in res:
                    continue
                print(key, flush=True)
                if snr is None:
                    ch, N0 = ch0, None
                else:
                    P = float(np.mean(np.abs(ch0) ** 2))
                    N0 = P / 10 ** (snr / 10)
                    rng = np.random.default_rng(1000 + seed)
                    ch = ch0 + np.sqrt(N0 / 2) * (rng.standard_normal(ch0.shape) + 1j * rng.standard_normal(ch0.shape))
                rr = dict(r_ref)
                res[key] = run_one(model, F, rr, ch, N0, cfg, tr)
                res[key]["_N0"] = N0
                json.dump(res, open(p, "w"), indent=1)
                if seed == 0 and snr in (None, 20.0):
                    lines[key] = {k: (20 * np.log10(np.abs(rr[k]) / np.abs(rr["ref"]).max() + 1e-12)).astype(np.float32)
                                  for k in ["ref", "std"] + [m for m in METHODS if m in rr]}
                    lines[key]["t"] = rr["t"].astype(np.float32)
                    lines[key]["xs"] = rr["xs"]
                    lines[key]["X"] = rr["X"]
                    np.savez_compressed(os.path.join(M.CACHE, f"item5_lines_{key.replace('|', '_')}.npz"),
                                        **lines[key])
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_baseline()
    else:
        compute()
