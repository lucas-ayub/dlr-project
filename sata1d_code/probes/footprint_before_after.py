"""
The block map with copies (footprint), before and after the review points:
  before: per sub-band k, delta_C0 fitted over the look angles of sub-band k (legacy
          getcoeff_beam with beta_lo/beta_hi), recomputed for every sub-band;
  after:  delta_C0 of every target from the full range history, computed once
          (delta_C0_per_target), attributed to every sub-band map; loops over Nsb.
Prints the value differences, the times and the ambiguity levels, and writes the
sweep-style figure of the "after" run (probes/footprint_after/<scene>_footprint.png).

    python probes/footprint_before_after.py --scene abc dense_ramp
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sata1d as S
import sata1d.sata as SS
from sata1d.subband import subband_frequency_beam, channel_spectrum
from sata1d.reconstruction import reconstruct_from_spectra
from legacy.subband_beam_window import residual_C0_subband
from run_one_case import scene_targets, oracle_channels, ambiguity_level, case_figure, SATA_OSF
from sweep_geometry_plots import scene_coordinates

SCENES = {"one": dict(scene="one"), "abc": dict(scene="abc"),
          "dense_ramp": dict(scene="dense_ramp"), "dense_hill": dict(scene="dense_hill"),
          "dense_ramp_h100": dict(scene="dense_ramp", hmax=100.0)}


def run(name, out_dir):
    a = SimpleNamespace(scene="abc", targets=[], extent=8000.0, hmax=1000.0, sigma=2500.0, phase="random", seed=0)
    for k_, v_ in SCENES[name].items():
        setattr(a, k_, v_)
    targets, amps, desc = scene_targets(a)
    xs = np.array([x for x, _ in targets]); hs = np.array([h for _, h in targets])
    cfg = S.make_config(targets, bxt_max=100.0, seed=0, amplitudes=amps)
    tr = S.build_platform_tracks(cfg)
    print(f"\n=== {name}: {desc}")

    # delta_C0 values: after (once, full history) and before (per sub-band, window fit)
    t0 = time.time(); dC0 = S.delta_C0_per_target(cfg, tr); t_after = time.time() - t0
    t0 = time.time()
    dC0_k = [np.array([[residual_C0_subband(cfg, tr, p, i, k) for i in range(cfg.Nrx)] for p in cfg.scene.points[1:]])
             for k in range(cfg.Nsb)]
    t_before = time.time() - t0
    dmax = max(np.abs(d - dC0).max() for d in dC0_k)
    print(f"delta_C0: once from the full history {t_after:.1f} s | per sub-band window fit ({cfg.Nsb} x) {t_before:.1f} s"
          f" | max |difference| = {1e3 * dmax:.4f} mm")

    ch = S.generate_channels(cfg, tr)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    env = {"ref": F(S.generate_reference(cfg, tr))}; pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    A = lambda e: ambiguity_level(t, e, pk, xs, cfg.system.vs, dt)

    env["none"] = F(S.reconstruct(cfg, tr, ch))
    env["whole"] = F(S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=SATA_OSF, mode="footprint", dC0=dC0)))
    # sub-band, before: map of sub-band k from the values fitted over the angles of sub-band k
    spec = []
    for k in range(cfg.Nsb):
        f_k = subband_frequency_beam(cfg, k)[0]
        maps_k = [SS.build_delta_C0_array(cfg, tr, j, mode="footprint", f_centre=f_k, dC0=dC0_k[k]) for j in range(cfg.Nrx)]
        corr = S.sata_channels_subband(cfg, tr, ch, k, sata_osf=SATA_OSF, maps=maps_k)
        spec.append([channel_spectrum(cfg, corr[j]) for j in range(cfg.Nrx)])
    env["sub_before"] = F(reconstruct_from_spectra(cfg, tr, spec, True))
    # sub-band, after: the pipeline (values attributed, Nsb loop)
    env["sub"] = F(S.reconstruct_subband(cfg, tr, ch, sata_osf=SATA_OSF, mode="footprint", dC0=dC0))
    env["oracle"] = F(S.reconstruct(cfg, tr, oracle_channels(cfg, tr, dC0)))
    amb = {k_: A(v_) for k_, v_ in env.items()}
    print("highest ambiguity [dB]: " + " | ".join(f"{k_} {v_:.1f}" for k_, v_ in amb.items()))

    db = {k_: 20 * np.log10(v_ / pk + 1e-12) for k_, v_ in env.items() if k_ != "sub_before"}
    amb_f = {k_: v_ for k_, v_ in amb.items() if k_ != "sub_before"}
    xs_, dys, hs_ = scene_coordinates(xs, hs)
    foot = (f"Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions "
            f"(+-{dt:.2f} s, +-{2 * dt:.2f} s) of every target. $N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, "
            f"$B_a$ = {cfg.abw:.0f} Hz, $b_{{xt}}$ = [" + ", ".join(f"{b:.1f}" for b in cfg.array.bxt) + "] m.")
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{name}_footprint.png")
    case_figure(png, f"SATA 1D, block map with copies | {desc}", t, db, amb_f, xs_, dys, hs_, "block map with copies", foot)
    print(f"wrote {png}")
    return amb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", nargs="+", default=list(SCENES), choices=list(SCENES))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "footprint_after"))
    a = ap.parse_args()
    for name in a.scene:
        run(name, a.out)
