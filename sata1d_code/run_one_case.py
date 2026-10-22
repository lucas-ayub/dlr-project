"""
One scene, the SATA modes you choose, sequential (no multiprocessing): meant to be read and
stepped through. Writes the case figure in the layout of the sweep pages (IRFs + geometry).

    python run_one_case.py                                 # scene A/B/C, look-up table, all modes
    python run_one_case.py --scene dense_ramp              # dense grid of targets on a ramp
    python run_one_case.py --scene dense_hill --extent 8000 --hmax 1000
    python run_one_case.py --targets 0:500 --mode none whole
    python run_one_case.py --map footprint                 # the previous block-and-copies map

scenes:  abc          A (-2.6 km, 0 m), B (0, 500 m), C (+2.6 km, 1000 m)
         one          one target at x = 0, h = 500 m
         dense_ramp   one target per azimuth cell over +-extent, h rising linearly 0 -> hmax
         dense_hill   one target per azimuth cell over +-extent, h = hmax exp(-x^2 / 2 sigma^2)
         custom       --targets x:h [x:h ...]  (or one comma-separated list; write --targets=-750:200,750:800
                      when the first x is negative, so that it is not read as an option)
         (dense scenes: every target gets a random phase, uniform in [0, 2 pi), --seed)
modes:   none    no SATA
         whole   SATA whole band  (sata_channels + reconstruct)
         sub     SATA per sub-band (reconstruct_subband)
         joint   reconstruction inside the SATA sub-apertures (reconstruct_joint): the delta_C0
                 of every folded parcel of every bin enters the inversion matrix, no copies
         oracle  every target simulated with its phase already corrected, no SATA
map:     lut        look-up table: the targets' delta_C0 at the targets' cells, no fold images
         footprint  the previous map: blocks +- h_w at the targets and at the fold images
"""
import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sata1d as S
import sata1d.sata as SS
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, coordinates_text

SATA_OSF = 4
MODES = ["none", "whole", "sub", "joint", "oracle"]
LABEL = {"lut": "look-up table", "footprint": "block map with copies"}


def scene_targets(a):
    """(list of (x, h), amplitudes or None, description) of the chosen scene."""
    if a.scene == "abc":
        return [(-2600.0, 0.0), (0.0, 500.0), (2600.0, 1000.0)], None, "targets A, B, C 2.6 km apart, h = 0, 500, 1000 m"
    if a.scene == "one":
        return [(0.0, 500.0)], None, "one target, h = 500 m"
    if a.scene == "custom":
        toks = [t for s in a.targets for t in s.split(",") if t]
        return [tuple(float(v) for v in t.split(":")) for t in toks], None, "custom targets"
    cell = S.SystemParams().vs / (2000.0 / 4)                      # azimuth cell of one channel [m]
    xs = np.arange(-a.extent, a.extent + 0.5 * cell, cell)
    if a.scene == "dense_ramp":
        hs = a.hmax * (xs - xs[0]) / (xs[-1] - xs[0])
        desc = f"{len(xs)} targets, one per cell ({cell:.1f} m), ramp 0 -> {a.hmax:.0f} m over +-{a.extent / 1e3:.0f} km"
    else:
        hs = a.hmax * np.exp(-0.5 * (xs / a.sigma) ** 2)
        desc = f"{len(xs)} targets, one per cell ({cell:.1f} m), hill {a.hmax:.0f} m, sigma {a.sigma / 1e3:.1f} km, +-{a.extent / 1e3:.0f} km"
    rng = np.random.default_rng(a.seed)
    amps = np.exp(2j * np.pi * rng.uniform(0.0, 1.0, len(xs))) if a.phase == "random" else None
    return list(zip(xs, hs)), amps, desc + (", random phases" if amps is not None else "")


def oracle_channels(cfg, tr, dC0):
    """Each target corrected with its own delta_C0 before the channels are summed."""
    out = np.zeros((cfg.Nrx, cfg.Na_ch), complex)
    for j, p in enumerate(cfg.scene.points[1:]):
        ch = S.generate_channels(cfg, tr, ptgs=[p], amps=[cfg.scene.amps[j]])
        out += ch * np.exp(2j * np.pi * dC0[j] / cfg.system.wl)[:, None]
    return out


def ambiguity_level(t, env, pk, xs, vs, dt_amb, orders=(-2, -1, 1, 2), win=0.02, guard=0.02):
    """Largest |IRF| inside +-win s of every ambiguity position x_j/vs + m dt_amb, in dB re. pk,
    leaving out +-guard s around every target's own main lobe (as in the sweeps)."""
    lobe = np.zeros(len(t), bool)
    for x in xs:
        lobe |= np.abs(t - x / vs) <= guard
    sel = np.zeros(len(t), bool)
    for x in xs:
        for m in orders:
            sel |= np.abs(t - (x / vs + m * dt_amb)) <= win
    sel &= ~lobe
    return 20 * np.log10(env[sel].max() / pk + 1e-15) if sel.any() else np.nan


def case_figure(out_png, title, t, db, amb, xs, dys, hs, map_label, foot):
    fig = plt.figure(figsize=(17, 9.6))
    gs = GridSpec(2, 2, width_ratios=[1.75, 1.0], hspace=0.36, wspace=0.14, figure=fig)
    fig.suptitle(title, fontsize=12, y=0.995)
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    parts = [f"reference {amb['ref']:.1f} dB"]
    if "whole" in db:
        ax.plot(t, db["whole"], color="tab:blue", lw=1.6, ls=(0, (5, 3)), label=f"SATA whole band ({map_label})")
        parts.append(f"SATA whole band {amb['whole']:.1f} dB")
    if "sub" in db:
        ax.plot(t, db["sub"], color="tab:orange", lw=1.8, ls=(0, (1, 2)), label=f"SATA per sub-band ({map_label})")
        parts.append(f"SATA per sub-band {amb['sub']:.1f} dB")
    if "joint" in db:
        ax.plot(t, db["joint"], color="tab:purple", lw=1.4, ls=(0, (3, 1, 1, 1)), label="joint SATA + reconstruction")
        parts.append(f"joint {amb['joint']:.1f} dB")
    if "oracle" in db:
        parts.append(f"oracle {amb['oracle']:.1f} dB")
    ax.set_title("highest ambiguity: " + " | ".join(parts), fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    parts = [f"reference {amb['ref']:.1f} dB"]
    if "none" in db:
        az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
        parts.append(f"no SATA {amb['none']:.1f} dB")
        imp = [f"{n} {amb['none'] - amb[k]:.1f} dB" for k, n in (("whole", "whole band"), ("sub", "per sub-band"), ("joint", "joint")) if k in amb]
        if imp:
            parts.append("improvement: " + ", ".join(imp))
    az.set_title("highest ambiguity: " + " | ".join(parts), fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")
    dense = len(xs) > 13
    g2 = fig.add_subplot(gs[0, 1])
    geometry_2d(g2, xs, dys, hs, False, title=f"Target geometry | $\\Delta h_{{max}}$ = {np.ptp(hs):.0f} m", annotate=not dense)
    a3 = fig.add_subplot(gs[1, 1], projection="3d")
    geometry_3d(a3, xs, dys, hs, False, title="Target geometry in 3D (x, $\\Delta y$, h)")
    if dense:
        for a_, s_ in ((g2, 4), (a3, 3)):
            for c in a_.collections:
                try:
                    c.set_sizes([s_])
                except Exception:
                    pass
    else:
        a3.text2D(0.99, 0.02, coordinates_text(xs, dys, hs), transform=a3.transAxes, fontsize=5.8, family="monospace",
                  ha="right", va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig.text(0.5, 0.015, foot, ha="center", va="bottom", fontsize=9, color="0.35")
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(out_png, dpi=90, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="abc", choices=["abc", "one", "dense_ramp", "dense_hill", "custom"])
    ap.add_argument("--targets", nargs="*", default=[], help="x:h pairs [m] for --scene custom (space- or comma-separated)")
    ap.add_argument("--extent", type=float, default=8000.0, help="dense scenes: targets over +-extent [m]")
    ap.add_argument("--hmax", type=float, default=1000.0, help="dense scenes: maximum height [m]")
    ap.add_argument("--sigma", type=float, default=2500.0, help="dense_hill: width of the hill [m]")
    ap.add_argument("--phase", default="random", choices=["random", "zero"], help="dense scenes: target phases")
    ap.add_argument("--mode", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--map", default="lut", choices=["lut", "footprint"])
    ap.add_argument("--bxt", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="run_one_case")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    targets, amps, desc = scene_targets(a)
    xs = np.array([x for x, _ in targets]); hs = np.array([h for _, h in targets])

    t0 = time.time()
    cfg = S.make_config(targets, bxt_max=a.bxt, seed=a.seed, amplitudes=amps)
    tr = S.build_platform_tracks(cfg)
    print(f"scene: {desc}")
    print(f"channels {cfg.Nrx}, sub-bands {cfg.Nsb}, PRF_ch = {cfg.PRF_op:.0f} Hz, cells = {cfg.Na_ch}, "
          f"cell = {cfg.system.vs / cfg.PRF_op:.2f} m, b_xt = {np.round(cfg.array.bxt, 2)} m")
    hw, X = SS.sata_footprint_halfwidth(cfg), SS.sata_image_offsets(cfg, 0.0).max()
    print(f"block half-width h_w = {hw} cells, fold X = {X:.1f} cells = {X * cfg.system.vs / cfg.PRF_op / 1e3:.2f} km")

    dC0 = S.delta_C0_per_target(cfg, tr)                   # once, for every target and channel
    print(f"delta_C0 of {len(targets)} targets x {cfg.Nrx} channels computed once ({time.time() - t0:.0f} s)")
    if len(targets) <= 13:
        print("delta_C0 [mm] per target and channel:")
        for (x, h), row in zip(targets, dC0):
            print(f"  x = {x:8.0f} m  h = {h:6.0f} m  cell = {SS.az_pixel_of_scatterer(cfg, x, 0):5d}  " + " ".join(f"{1e3 * v:9.3f}" for v in row))
    else:
        print(f"delta_C0 [mm], channel 0: min {1e3 * dC0[:, 0].min():.1f}, max {1e3 * dC0[:, 0].max():.1f}")
    M0 = SS.build_delta_C0_array(cfg, tr, 0, mode=a.map, dC0=dC0)
    nz = np.flatnonzero(M0)
    print(f"map ({LABEL[a.map]}), channel 0: {len(nz)} non-zero cells" + (f", from cell {nz[0]} to {nz[-1]}" if len(nz) else ""))

    ch = S.generate_channels(cfg, tr)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    env = {"ref": F(S.generate_reference(cfg, tr))}
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    kw = dict(sata_osf=SATA_OSF, mode=a.map, dC0=dC0)
    for mode in a.mode:
        t1 = time.time()
        if mode == "none":
            rec = S.reconstruct(cfg, tr, ch)
        elif mode == "whole":
            rec = S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, **kw))
        elif mode == "sub":
            rec = S.reconstruct_subband(cfg, tr, ch, **kw)
        elif mode == "joint":
            xt, ht = np.array(targets, float).T; o = np.argsort(xt)
            dem = lambda x: np.interp(np.asarray(x, float), xt[o], ht[o])   # DEM of the scene: the terrain through the targets
            rec = S.reconstruct_joint(cfg, tr, ch, maps=S.channel_maps(cfg, tr, dem), sata_osf=SATA_OSF)
        elif mode == "oracle":
            rec = S.reconstruct(cfg, tr, oracle_channels(cfg, tr, dC0))
        env[mode] = F(rec)
        print(f"{mode:7s} done ({time.time() - t1:.0f} s)", flush=True)
    amb = {k: ambiguity_level(t, v, pk, xs, cfg.system.vs, dt) for k, v in env.items()}
    print("\nhighest ambiguity [dB re. reference peak]:")
    for k, v in amb.items():
        print(f"  {k:7s} {v:7.1f}")
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    xs_, dys, hs_ = scene_coordinates(xs, hs)
    foot = (f"Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions "
            f"(+-{dt:.2f} s, +-{2 * dt:.2f} s) of every target. $N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, "
            f"$B_a$ = {cfg.abw:.0f} Hz, $b_{{xt}}$ = [" + ", ".join(f"{b:.1f}" for b in cfg.array.bxt) + "] m.")
    png = os.path.join(a.out, f"{a.scene}_{a.map}.png")
    case_figure(png, f"SATA 1D, {LABEL[a.map]} | {desc}", t, db, amb, xs_, dys, hs_, LABEL[a.map], foot)
    np.savez_compressed(png[:-4] + ".npz", t=t, xs=xs_, dys=dys, hs=hs_, dC0=dC0, map_ch0=M0,
                        **{f"env_{k}": (v / pk).astype(np.float32) for k, v in env.items()})
    print(f"\nwrote {png} (+ .npz), {time.time() - t0:.0f} s in total")


if __name__ == "__main__":
    main()
