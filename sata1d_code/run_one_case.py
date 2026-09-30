"""
One scene, one or more SATA modes, no multiprocessing: meant to be read and stepped through.

    python run_one_case.py                          # scene A/B/C, all modes
    python run_one_case.py --mode whole             # one mode
    python run_one_case.py --targets 0:500          # one target, h = 500 m
    python run_one_case.py --targets -2600:0 0:500 2600:1000 --mode lut_whole lut_sub

modes:  none        no SATA
        whole       SATA whole band, current map        (sata_channels,     mode="footprint")
        sub         SATA per sub-band, current map      (reconstruct_subband, mode="footprint")
        lut_whole   SATA whole band, look-up table      (sata_channels,     mode="lut")
        lut_sub     SATA per sub-band, look-up table    (reconstruct_subband, mode="lut")
        oracle      every target simulated with the corrected phase, no SATA (best possible C0 correction)
Prints the delta_C0 of every target and channel, what each map holds at the targets and at the
first fold image, and the highest ambiguity of every mode; writes run_one_case.png.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sata1d as S
import sata1d.sata as SS

SATA_OSF = 4
MODES = ["none", "whole", "sub", "lut_whole", "lut_sub", "oracle"]


def oracle_channels(cfg, tr):
    out = np.zeros((cfg.Nrx, cfg.Na_ch), complex)
    for p in cfg.scene.points[1:]:
        ch = S.generate_channels(cfg, tr, ptgs=[p])
        d = np.array([S.residual_C0(cfg, tr, p, i) for i in range(cfg.Nrx)])
        out = out + ch * np.exp(2j * np.pi * d / cfg.system.wl)[:, None]
    return out


def ambiguity_level(t, env, pk, xs, vs, dt_amb, orders=(-2, -1, 1, 2), win=0.02, guard=0.02):
    lobe = np.zeros(len(t), bool)
    for x in xs:
        lobe |= np.abs(t - x / vs) <= guard
    best = 0.0
    for x in xs:
        for m in orders:
            sel = (np.abs(t - (x / vs + m * dt_amb)) <= win) & ~lobe
            if sel.any():
                best = max(best, float(env[sel].max()))
    return 20 * np.log10(best / pk + 1e-15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["-2600:0", "0:500", "2600:1000"], help="x:h pairs [m]")
    ap.add_argument("--mode", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--bxt", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    targets = [tuple(float(v) for v in s.split(":")) for s in a.targets]
    xs = np.array([t[0] for t in targets])

    cfg = S.make_config(targets, bxt_max=a.bxt, seed=a.seed)
    tr = S.build_platform_tracks(cfg)
    print(f"channels: {cfg.Nrx}, PRF_ch = {cfg.PRF_op:.0f} Hz, cells = {cfg.Na_ch}, cell = {cfg.system.vs / cfg.PRF_op:.2f} m")
    print("b_xt [m]:", np.round([tr.prx[i][0][1] - tr.ptx[0][1] for i in range(cfg.Nrx)], 2))
    hw, X = SS.sata_footprint_halfwidth(cfg), SS.sata_image_offsets(cfg, 0.0).max()
    print(f"block half-width h_w = {hw} cells, fold X = {X:.1f} cells")

    print("\ndelta_C0 [mm] per target and channel:")
    for p in cfg.scene.points[1:]:
        d = [1e3 * S.residual_C0(cfg, tr, p, i) for i in range(cfg.Nrx)]
        print(f"  x = {p[0] - cfg.scene.x0:8.0f} m  h = {p[2]:6.0f} m  cell = {SS.az_pixel_of_scatterer(cfg, p[0] - cfg.scene.x0, 0):5d}  ",
              " ".join(f"{v:9.3f}" for v in d))

    print("\nchannel 0 maps at the targets and at the first fold image (+X) [mm]:")
    Mw = SS.build_delta_C0_array(cfg, tr, 0)
    Ml = SS.build_delta_C0_array(cfg, tr, 0, mode="lut")
    for p in cfg.scene.points[1:]:
        n = SS.az_pixel_of_scatterer(cfg, p[0] - cfg.scene.x0, 0); m = int(round(n + X))
        print(f"  target at cell {n}: current {1e3 * Mw[n]:9.3f}  lut {1e3 * Ml[n]:9.3f}   |  cell {m}: current {1e3 * Mw[m]:9.3f}  lut {1e3 * Ml[m]:9.3f}")

    ch = S.generate_channels(cfg, tr)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    ref = F(S.generate_reference(cfg, tr)); pk = ref.max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    amb = lambda rec: ambiguity_level(t, F(rec), pk, xs, cfg.system.vs, dt)
    kw = dict(sata_osf=SATA_OSF)

    rec = {}
    for mode in a.mode:
        if mode == "none":
            rec[mode] = S.reconstruct(cfg, tr, ch)
        elif mode == "whole":
            rec[mode] = S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, mode="footprint", **kw))
        elif mode == "sub":
            rec[mode] = S.reconstruct_subband(cfg, tr, ch, mode="footprint", **kw)
        elif mode == "lut_whole":
            rec[mode] = S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, mode="lut", **kw))
        elif mode == "lut_sub":
            rec[mode] = S.reconstruct_subband(cfg, tr, ch, mode="lut", **kw)
        elif mode == "oracle":
            rec[mode] = S.reconstruct(cfg, tr, oracle_channels(cfg, tr))
        print(f"\n{mode:10s} highest ambiguity {amb(rec[mode]):7.1f} dB", flush=True)
    print(f"{'reference':10s} highest ambiguity {ambiguity_level(t, ref, pk, xs, cfg.system.vs, dt):7.1f} dB")

    db = lambda e: 20 * np.log10(e / pk + 1e-12)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, db(ref), color="tab:green", lw=2, label="Reference")
    for mode, r in rec.items():
        ax.plot(t, db(F(r)), lw=1.2, label=f"{mode} ({amb(r):.1f} dB)")
    ax.set_ylim(-90, 5); ax.set_xlim(t[0], t[-1]); ax.grid(alpha=.3); ax.legend(fontsize=9); ax.set_xlabel("Azimuth time [s]"); ax.set_ylabel("[dB]")
    fig.savefig("run_one_case.png", dpi=110, bbox_inches="tight")
    print("\nwrote run_one_case.png")


if __name__ == "__main__":
    main()
