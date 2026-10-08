"""What limits the joint reconstruction on the dense ramp: sub-aperture length (sata_osf), the
rounding of the parcel positions, and a constant-height check (joint must equal the oracle)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sata1d as S
import sata1d.joint as J
from run_one_case import ambiguity_level, oracle_channels

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_joint_residual.npz")


def scene(kind, hmax=1000.0, extent=8000.0, seed=0):
    cell = S.SystemParams().vs / 500.0
    xs = np.arange(-extent, extent + 0.5 * cell, cell)
    hs = {"ramp": hmax * (xs - xs[0]) / (xs[-1] - xs[0]), "const": np.full(len(xs), hmax)}[kind]
    rng = np.random.default_rng(seed)
    amps = np.exp(2j * np.pi * rng.uniform(0.0, 1.0, len(xs)))
    return list(zip(xs, hs)), amps, xs


def run(kind, hmax):
    key = f"{kind}_{hmax:.0f}"
    targets, amps, xs = scene(kind, hmax)
    cfg = S.make_config(targets, bxt_max=100.0, seed=0, amplitudes=amps)
    tr = S.build_platform_tracks(cfg)
    cache = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    if key + "_ch" not in cache:
        t0 = time.time()
        cache[key + "_ch"] = S.generate_channels(cfg, tr)
        cache[key + "_dC0"] = S.delta_C0_per_target(cfg, tr)
        cache[key + "_orc"] = oracle_channels(cfg, tr, cache[key + "_dC0"])
        np.savez_compressed(CACHE, **cache)
        print(f"[{key}] channels generated ({time.time() - t0:.0f} s)")
    ch, dC0, orc = cache[key + "_ch"], cache[key + "_dC0"], cache[key + "_orc"]
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    pk = F(S.generate_reference(cfg, tr)).max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    A = lambda rec: ambiguity_level(t, F(rec), pk, xs, cfg.system.vs, dt)
    print(f"[{key}] none {A(S.reconstruct(cfg, tr, ch)):.1f}  oracle {A(S.reconstruct(cfg, tr, orc)):.1f}")
    maps = J.joint_maps(cfg, tr, dC0)
    for osf in (2, 4, 8, 16):
        rec = S.reconstruct_joint(cfg, tr, ch, maps=maps, sata_osf=osf)
        print(f"[{key}] joint osf {osf:2d}: {A(rec):.1f}")
    return cfg, tr, ch, dC0, maps, A


if __name__ == "__main__":
    run("const", 500.0)
    run("ramp", 1000.0)
