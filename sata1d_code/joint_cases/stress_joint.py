"""
Stress tests of the joint reconstruction (sata1d/joint.py) for several numbers of channels, baselines and scenes,
using only information available in a real case: the system parameters and a DEM of the terrain (here the exact
terrain). No method knows where the targets are or their dC0: every map is the look-up table from the terrain
(terrain_lookup_table), for the joint and for the SATA before the reconstruction alike.
Every job writes joint_cases/stress/jobs/<id>.json; the run is resumable (jobs already done are skipped).
    SATA_LUT_CACHE=/tmp/lutcache python joint_cases/stress_joint.py --workers 4 --budget 160   # from sata1d_code, repeat
    python joint_cases/stress_joint.py --table                                                  # text summary
Groups
  identity : flat terrain (DEM = 0): the joint must equal the standard reconstruction (full system) / be ~1e-3 from it
  single   : one target on a flat terrain at height h (DEM = h everywhere)
  ramp     : along-track ramp terrain (flat, ramp over 12 km, flat on top) with 49 or 121 targets on it
  dense    : ramp terrain over 2 km with a target every 10 m (less than one azimuth cell)
  terrain  : realistic terrains with many targets
Baselines: channel 0 on the transmitter (bat = bxt = 0); the others random, scaled so that the largest is exactly bmax.
"""
import argparse
import glob
import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
import sata1d.sata as SS
from sata1d.config import ArrayGeometry
from sata1d.subband import sata_channels_subband, channel_spectrum
from sata1d.reconstruction import reconstruct_from_spectra
import run_geometry_sweep as G
import run_joint_sweep as W

OUT = os.path.join(HERE, "stress")
VERSION = 3          # 3: DEM everywhere, no knowledge of the targets
NRX = [2, 3, 4, 5, 6, 7, 8]
BMAX = [100.0, 300.0, 500.0]


def baselines(nrx, bmax, seed=0):
    """Channel 0 on the transmitter; the others random in (0, bmax], the largest exactly bmax."""
    if bmax == 0:
        return np.zeros(nrx)
    u = np.random.default_rng(seed).uniform(0.1, 1.0, nrx)
    u[0] = 0.0
    return (bmax * u / u.max()).round(1)


def ramp_terrain(slope_deg, x0, length):
    """Flat at h = 0 before x0, ramp of slope_deg over [x0, x0 + length], flat on top after."""
    t = np.tan(np.radians(slope_deg))
    return lambda x: t * np.clip(np.asarray(x, float) - x0, 0.0, length)


def flat_terrain(h):
    return lambda x: np.full(np.shape(np.asarray(x, float)), float(h))


def terrain_by_name(name):
    """Terrain function from its name (so that worker processes can rebuild it)."""
    kind, *p = name.split("_")
    if kind == "flat":
        return flat_terrain(float(p[0]))
    if kind == "ramp":                                    # ramp_<slope>_<x0>_<length>
        return ramp_terrain(float(p[0]), float(p[1]), float(p[2]))
    return W.TERRAINS[kind][0]


def phases(n, seed=1):
    return np.exp(2j * np.pi * np.random.default_rng(seed).uniform(size=n))


def jobs():
    J = []
    span = lambda half, d: np.arange(-half, half + 1e-6, d)
    xs = span(6000.0, 250.0)
    for n in NRX:
        J.append(dict(id=f"identity_N{n}", group="identity", xs=xs, terrain="flat_0", nrx=n, bmax=300.0))
    for h in (0.0, 500.0, 1500.0):
        for n in NRX:
            J.append(dict(id=f"single_h{h:g}_N{n}", group="single", xs=np.array([1300.0]), terrain=f"flat_{h:g}",
                          nrx=n, bmax=500.0))
    for s, d in ((3.0, 250.0), (8.0, 250.0), (15.0, 250.0), (25.0, 250.0), (8.0, 100.0)):
        for bm in BMAX:
            for n in NRX:
                J.append(dict(id=f"ramp_s{s:g}_d{d:g}_b{bm:g}_N{n}", group="ramp", xs=span(6000.0, d),
                              terrain=f"ramp_{s:g}_-6000_12000", nrx=n, bmax=bm, slope=s))
    for n in NRX:
        J.append(dict(id=f"dense_s8_d10_N{n}", group="dense", xs=span(1000.0, 10.0), terrain="ramp_8_-1000_2000",
                      nrx=n, bmax=300.0, slope=8.0))
    for name, half in (("hills", 6000.0), ("mountains", 6000.0), ("escarpment", 3000.0), ("mountain", 4000.0)):
        for bm in BMAX:
            for n in NRX:
                J.append(dict(id=f"terrain_{name}_d250_b{bm:g}_N{n}", group="terrain", xs=span(half, 250.0),
                              terrain=name, nrx=n, bmax=bm))
    for j in J:
        j["hs"] = terrain_by_name(j["terrain"])(j["xs"])
    return J


def config(job):
    amps = phases(len(job["xs"])) if job["group"] in ("ramp", "dense", "terrain") else None
    cfg = S.make_config(list(zip(job["xs"], job["hs"])), nrx=job["nrx"], bxt_max=0.0, amplitudes=amps, tx_rx_channel0=True)
    return replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=baselines(job["nrx"], job["bmax"])))


def dem_table(cfg, tr, i, name):
    """terrain_lookup_table of channel i for the terrain `name` (step 4 cells), cached on disk."""
    cache = os.environ.get("SATA_LUT_CACHE", "/tmp/lutcache")
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, f"dem_{name}_bat{cfg.array.bat[i]:.3f}_bxt{cfg.array.bxt[i]:g}_n{cfg.Na_ch}.npy")
    if os.path.exists(f):
        return np.load(f)
    m = S.terrain_lookup_table(cfg, tr, i, terrain_by_name(name), step=4)
    np.save(f, m)
    return m


def sata_whole(cfg, tr, ch, maps):
    """SATA of every channel with the given maps, then the standard reconstruction."""
    o = np.array(ch, complex)
    for i in range(cfg.Nrx):
        o[i] = SS.sata_1d(o[i], maps[i], rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=cfg.system.wl,
                          r=cfg.scene.r0, squint=cfg.sq_tx, inverse=True, sata_osf=4)
    return S.reconstruct(cfg, tr, o)


def sata_sub(cfg, tr, ch, maps):
    """SATA per sub-band with the given maps."""
    spec = []
    for k in range(cfg.Nsb):
        corr = sata_channels_subband(cfg, tr, ch, k, sata_osf=4, maps=list(maps))
        spec.append([channel_spectrum(cfg, corr[j]) for j in range(cfg.Nrx)])
    return reconstruct_from_spectra(cfg, tr, spec, True)


def conditioning(cfg, tr, maps, sata_osf=4):
    """Smallest sigma_min / sigma_max of the occupied columns of G over all blocks and bins (as in joint.py)."""
    from sata1d.joint import reference_coefficients
    Nrx, Nsb, Na_ch, P = cfg.Nrx, cfg.Nsb, cfg.Na_ch, cfg.PRF_op
    wl, v, r0 = cfg.system.wl, cfg.system.vs, cfg.scene.r0
    Tsub, hop, Nzp = SS.subaperture_sizing(r0, P, v, wl, sata_osf)
    phi = np.arange(Nzp) * P / Nzp
    f = -Nsb * P / 2.0 + np.mod(phi + Nsb * P / 2.0, P)[None, :] + (np.arange(Nsb) * P)[:, None]
    a = r0 * np.tan(np.arcsin(np.clip(wl * f / (2.0 * v), -1.0, 1.0))) / v * P
    coef = reference_coefficients(cfg, tr)
    Hq = np.zeros((Nzp, Nrx, Nsb), complex)
    for i in range(Nrx):
        C0, C1, C2, Dt = coef[i]
        Hq[:, i, :] = np.exp(-2j * np.pi * (C0 / wl + (-C1 + Dt) * f.T + C2 * wl * f.T ** 2))
    B = 4.0 * v * np.sin(cfg.theta_tx / 2.0) / wl
    occ = np.abs(f.T) <= B / 2.0 + 4.0 * P / Tsub
    nocc = occ.sum(1)
    worst = 1.0
    for start in range(0, Na_ch, hop):
        pos = np.clip(np.round(start + 0.5 * Tsub + a).astype(int), 0, Na_ch - 1)
        G_ = Hq * np.exp(-2j * np.pi * maps[:, pos].transpose(2, 0, 1) / wl)
        sv = np.linalg.svd(G_ * occ[:, None, :], compute_uv=False)
        worst = min(worst, float((sv[np.arange(Nzp), nocc - 1] / sv[:, 0]).min()))
    return worst, int(nocc.min()), int(nocc.max())


def run(job):
    t0 = time.time()
    f = os.path.join(OUT, "jobs", job["id"] + ".json")
    cfg = config(job)
    tr = S.build_platform_tracks(cfg)
    ch = S.generate_channels(cfg, tr)
    maps = np.array([dem_table(cfg, tr, i, job["terrain"]) for i in range(cfg.Nrx)])
    lines = {"ref": S.generate_reference(cfg, tr),
             "joint_occ": S.reconstruct_joint(cfg, tr, ch, maps=maps, sata_osf=4, occupied=True),
             "joint_full": S.reconstruct_joint(cfg, tr, ch, maps=maps, sata_osf=4, occupied=False)}
    res = dict(job, xs=None, hs=None, n_targets=len(job["xs"]), bxt=cfg.array.bxt.tolist(), Na_ch=cfg.Na_ch,
               version=VERSION)
    if job["group"] == "identity":
        std = S.reconstruct(cfg, tr, ch)
        for k in ("joint_occ", "joint_full"):
            res["rel_err_" + k] = float(np.linalg.norm(lines[k] - std) / np.linalg.norm(std))
    else:
        lines["none"] = S.reconstruct(cfg, tr, ch)
        lines["whole"] = sata_whole(cfg, tr, ch, maps)
        lines["sub"] = sata_sub(cfg, tr, ch, maps)
        sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
        env = {k: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2)) for k, s in lines.items()}
        pk = env["ref"].max()
        t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
        res["amb"] = {k: float(G.ambiguity_level(t, e, pk, job["xs"], cfg.system.vs, G.ambiguity_times(cfg)))
                      for k, e in env.items()}
        fa = np.fft.fftfreq(cfg.Na, 1.0 / cfg.prf); bl = np.abs(fa) <= cfg.abw / 2
        R = np.fft.fft(lines["ref"]) * bl
        res["err"] = {k: float(10 * np.log10(np.sum(np.abs(np.fft.fft(s) * bl - R) ** 2) / np.sum(np.abs(R) ** 2)))
                      for k, s in lines.items() if k != "ref"}
        res["rho_min"], res["occ_min"], res["occ_max"] = conditioning(cfg, tr, maps)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(f, "w"), indent=1, default=float)
    return job["id"], res["seconds"]


def load():
    return [r for r in (json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "jobs", "*.json"))))
            if r.get("version") == VERSION]


def table():
    rows = load()
    for r in rows:
        if r["group"] == "identity":
            print(f"{r['id']:34s} rel.err joint full {r['rel_err_joint_full']:.1e}  occupied {r['rel_err_joint_occ']:.1e}")
    for r in rows:
        if r["group"] != "identity":
            a, e = r["amb"], r["err"]
            print(f"{r['id']:34s} amb ref {a['ref']:6.1f} none {a['none']:6.1f} whole {a['whole']:6.1f} sub {a['sub']:6.1f}"
                  f" occ {a['joint_occ']:6.1f} full {a['joint_full']:6.1f} | err none {e['none']:6.1f} whole {e['whole']:6.1f}"
                  f" occ {e['joint_occ']:6.1f} | rho {r['rho_min']:.1e} occ {r['occ_min']}-{r['occ_max']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--group", default=None)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table:
        return table()
    os.makedirs(os.path.join(OUT, "jobs"), exist_ok=True)
    done = {r["id"] for r in load()}
    todo = [j for j in jobs() if j["id"] not in done and (a.group is None or j["group"] == a.group)]
    print(len(todo), "jobs to do", flush=True)
    deadline = time.time() + a.budget
    with Pool(a.workers) as pool:
        for jid, sec in pool.imap_unordered(run, todo):
            print(jid, sec, "s", flush=True)
            if time.time() > deadline:
                pool.terminate()
                return print("budget reached, run again", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
