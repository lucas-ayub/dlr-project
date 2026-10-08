"""
A few cases of the geometry sweep with the look-up table (--map lut) or the block map with
copies (--map footprint): IRFs as in the sweep pages, target geometry, and the delta_C0 that every
cell of the map receives (all channels; footprint: the whole-band map).

    python run_sweep_lut_cases.py
    python run_sweep_lut_cases.py --map footprint
    python run_sweep_lut_cases.py --cases ramp:N2_a40 cliff:N9_dh500
    python run_sweep_lut_cases.py --v4 RA-NN-036 RH-NN-005 --map footprint     # cases of sweep_v4 by id
    python run_sweep_lut_cases.py --many --map lut       # many points and continuous distributions of points
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.collections import PathCollection
import sata1d as S
import sata1d.sata as SS
from run_geometry_sweep import (Case, BXT_SEED, build_cases, make_cfg, oracle_channels, ambiguity_times, ambiguity_level,
                                FAMILIES, LABEL, PREFIX, SATA_OSF, BXT_MAX)
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d

DEFAULT = ["ramp:N2_a40", "ramp:N7_a20", "cliff:N9_dh500", "spike:N9_dh1000", "mesa:N9_dh500",
           "random_heights:N13_hmax1000_seed0", "topo_ramp:N7_a20"]
CH_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]


NAME = {"lut": "look-up table", "footprint": "block map with copies"}
OTHER = {"lut": "footprint", "footprint": "lut"}


V4_DIST = {"S": "200 m apart", "N": "2.6 km apart", "L": "10 km apart"}
V4_H = {"S": "small heights", "N": "normal heights", "L": "large heights"}
V4_DEFAULT = ["RA-SS-012", "RA-NN-036", "RA-NN-038", "RA-LN-038", "ZI-NN-012", "PW-NN-002",
              "RH-NN-005", "RH-LN-005", "CL-NN-005", "SK-NN-003", "ME-NN-002"]


CELL = S.SystemParams().vs / 500.0                    # azimuth cell of one channel [m]


def many_cases():
    """(id, title, Case): several points (sparse) and continuous distributions of points
    (one per cell, random phase in [0, 2 pi), seed 0)."""
    out = []
    rng = np.random.default_rng(0)
    def add(cid, title, xs, hs, dense=False):
        xs, hs = np.asarray(xs, float), np.asarray(hs, float)
        p = {"xs": xs}
        if dense:
            p["amps"] = np.exp(2j * np.pi * np.random.default_rng(0).uniform(0.0, 1.0, len(xs)))
        out.append((cid, title, Case("ramp", cid, title, hs, float(np.diff(xs).mean()) if len(xs) > 1 else 0.0, p)))
    for N, sp in ((13, 1000.0), (25, 500.0), (49, 250.0)):
        xs = (np.arange(N) - (N - 1) / 2) * sp
        add(f"MP-R{N}", f"{N} targets {sp:g} m apart, ramp 0 -> 1000 m", xs, np.linspace(0, 1000, N))
    xs = (np.arange(13) - 6) * 1000.0
    add("MP-H13", "13 targets 1 km apart, random heights U(0, 1000) m", xs, np.r_[0, rng.uniform(0, 1000, 12)])
    xs = (np.arange(25) - 12) * 500.0
    add("MP-H25", "25 targets 500 m apart, random heights U(0, 1000) m", xs, np.r_[0, rng.uniform(0, 1000, 24)])
    xd = np.arange(-8000.0, 8000.0 + 0.5 * CELL, CELL)
    add("CO-R1000", f"continuous: one target per cell ({CELL:.1f} m) over +-8 km, ramp 0 -> 1000 m", xd,
        1000.0 * (xd - xd[0]) / np.ptp(xd), True)
    add("CO-R100", f"continuous: one target per cell over +-8 km, ramp 0 -> 100 m", xd, 100.0 * (xd - xd[0]) / np.ptp(xd), True)
    add("CO-HILL", "continuous: one target per cell over +-8 km, hill 1000 m, sigma 2.5 km", xd,
        1000.0 * np.exp(-0.5 * (xd / 2500.0) ** 2), True)
    add("CO-WAVE", "continuous: one target per cell over +-8 km, h = 250 (1 - cos(2 pi x / 4 km)) m", xd,
        250.0 * (1 - np.cos(2 * np.pi * xd / 4000.0)), True)
    return out


def run_case(case, case_id, out_dir, mp="lut", title=None, full_line=False, compare=False):
    if "xs" in case.params:                       # many_cases: explicit positions, optional random phases
        amps = case.params.get("amps")
        cfg = S.make_config(list(zip(case.params["xs"], case.heights)), bxt_max=BXT_MAX, seed=BXT_SEED, amplitudes=amps)
    else:
        cfg = make_cfg(case, BXT_MAX)
    tr = S.build_platform_tracks(cfg)
    dC0 = S.delta_C0_per_target(cfg, tr)
    xs_case = case.params["xs"] if "xs" in case.params else case.xs
    ch = S.generate_channels(cfg, tr)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    env = {"ref": F(S.generate_reference(cfg, tr)),
           "none": F(S.reconstruct(cfg, tr, ch)),
           "whole": F(S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=SATA_OSF, mode=mp, dC0=dC0))),
           "sub": F(S.reconstruct_subband(cfg, tr, ch, sata_osf=SATA_OSF, mode=mp, dC0=dC0)),
           "other": F(S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=SATA_OSF, mode=OTHER[mp], dC0=dC0))),
           "oracle": F(S.reconstruct(cfg, tr, oracle_amps(cfg, tr, dC0)))}
    if compare:                                   # the other map per sub-band too: its IRFs replace the no-SATA panel
        env["other_sub"] = F(S.reconstruct_subband(cfg, tr, ch, sata_osf=SATA_OSF, mode=OTHER[mp], dC0=dC0))
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    amb = {k: ambiguity_level(t, v, pk, xs_case, cfg.system.vs, ambiguity_times(cfg)) for k, v in env.items()}
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    maps = np.array([SS.build_delta_C0_array(cfg, tr, i, mode=mp, dC0=dC0) for i in range(cfg.Nrx)])

    fig = plt.figure(figsize=(18, 9.6))
    gs = GridSpec(2, 2, width_ratios=[1.45, 1.25], hspace=0.36, wspace=0.14, figure=fig)
    spacing = "" if case.offsets is not None else f", spacing {case.spacing:g} m"
    title = title or f"{case_id}: {LABEL[case.family]}, {case.desc}{spacing}"
    fig.suptitle(f"{title} | " + (f"{NAME[mp]} (top) and {NAME[OTHER[mp]]} (bottom)" if compare else NAME[mp]), fontsize=12, y=0.995)
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["whole"], color="tab:blue", lw=1.6, ls=(0, (5, 3)), label=f"SATA whole band ({NAME[mp]})")
    ax.plot(t, db["sub"], color="tab:orange", lw=1.8, ls=(0, (1, 2)), label=f"SATA per sub-band ({NAME[mp]})")
    ax.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | SATA whole band {amb['whole']:.1f} dB | "
                 f"SATA per sub-band {amb['sub']:.1f} dB", fontsize=10)
    az.plot(t, db["ref"], color="tab:green", lw=2.2, label="Reference")
    if compare:
        az.plot(t, db["other"], color="tab:blue", lw=1.6, ls=(0, (5, 3)), label=f"SATA whole band ({NAME[OTHER[mp]]})")
        az.plot(t, db["other_sub"], color="tab:orange", lw=1.8, ls=(0, (1, 2)), label=f"SATA per sub-band ({NAME[OTHER[mp]]})")
        az.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | SATA whole band {amb['other']:.1f} dB | "
                     f"SATA per sub-band {amb['other_sub']:.1f} dB", fontsize=10)
    else:
        az.plot(t, db["none"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
        az.set_title(f"highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | improvement: "
                     f"whole band {amb['none'] - amb['whole']:.1f} dB, per sub-band {amb['none'] - amb['sub']:.1f} dB", fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")

    xs, dys, hs = scene_coordinates(xs_case, case.heights, case.offsets)
    topo = case.offsets is not None
    many = len(xs) > 13
    gg = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.18, width_ratios=[1.0, 1.1])
    g2 = fig.add_subplot(gg[0, 0]); g3 = fig.add_subplot(gg[0, 1], projection="3d")
    geometry_2d(g2, xs, dys, hs, topo, title=f"Target geometry | $\\Delta h_{{max}}$ = {np.ptp(hs):.0f} m", annotate=not many)
    geometry_3d(g3, xs, dys, hs, topo, title="in 3D (x, $\\Delta y$, h)")
    if many:
        for a_, s_ in ((g2, 8 if len(xs) < 100 else 3), (g3, 6 if len(xs) < 100 else 2)):
            for c in a_.collections:
                if full_line and not isinstance(c, PathCollection):     # the markers only: the terrain fill is a collection too
                    continue
                try:
                    c.set_sizes([s_])
                except Exception:
                    pass
        for t_ in list(g3.texts):
            t_.remove()
    if full_line and not topo:        # 2D: zoom on the targets and the reference point; 3D: the whole azimuth line (as the IRF and the map)
        from matplotlib.ticker import FuncFormatter
        half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
        km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
        g2.plot(0.0, 0.0, marker="*", ms=13, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $h$ = 0)")
        g2.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.1, handlelength=1.6, columnspacing=1.2)
        g2.set_title(f"Zoom on the point targets and the reference point | $\\Delta h_{{max}}$ = {np.ptp(hs):.0f} m", fontsize=9.5, pad=36)
        g2.xaxis.set_major_formatter(km); g2.set_xlabel("azimuth position $x$ [km]")
        g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=6)
        g3.set_xlim(-half, half); g3.xaxis.set_major_formatter(km); g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4)
        g3.set_title(f"Whole azimuth line ({2 * half / 1e3:.0f} km) in 3D (x, $\\Delta y$, h)", fontsize=9.5)
        for t_ in list(g3.texts):
            t_.remove()

    am = fig.add_subplot(gs[1, 1])
    nz = np.flatnonzero(np.any(maps != 0, axis=0))
    cells_t = np.array([SS.az_pixel_of_scatterer(cfg, dx, 0) for (dx, _, _) in cfg.scene.extra_offsets])
    lo = 0
    hi = cfg.Na_ch - 1
    n = np.arange(lo, hi + 1)
    for i in range(cfg.Nrx):
        am.step(n, 1e3 * maps[i, lo:hi + 1], where="mid", color=CH_COLORS[i], lw=1.3,
                label=f"channel {i} ($b_{{xt}}$ = {cfg.array.bxt[i]:.1f} m)")
    if len(cells_t) <= 60:
        for c in np.unique(cells_t):
            am.axvline(c, color="0.75", lw=0.6, zorder=0)
    if len(cells_t) <= 9:
        for c, v in zip(cells_t, dC0[:, 0]):
            am.annotate(f"{1e3 * v:.1f}", (c, 1e3 * v), textcoords="offset points", xytext=(0, -11 if v < 0 else 4),
                        ha="center", fontsize=7, color=CH_COLORS[0])
    am.set_xlim(lo, hi); am.grid(alpha=0.3)
    am.set_xlabel("azimuth cell of the channel line"); am.set_ylabel("$\\Delta C_0$ in the map [mm]")
    am.set_title("$\\Delta C_0$ per cell", fontsize=10)
    am.legend(fontsize=8, loc="lower left")
    fig.text(0.5, 0.012, f"Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions "
             f"of every target. For comparison: " + (f"no SATA {amb['none']:.1f} dB, " if compare else f"{NAME[OTHER[mp]]} (whole band) {amb['other']:.1f} dB, ") +
             f"oracle {amb['oracle']:.1f} dB. $N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz, "
             f"$b_{{xt}}$ = [" + ", ".join(f"{b:.1f}" for b in cfg.array.bxt) + "] m.",
             ha="center", fontsize=9, color="0.35")
    fig.subplots_adjust(bottom=0.1)
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{case_id}_{case.family}_{case.slug}_{mp}.png")
    fig.savefig(png, dpi=95, bbox_inches="tight"); plt.close(fig)
    return png, amb


def oracle_amps(cfg, tr, dC0):
    """Each target (with its amplitude) corrected with its own delta_C0 before the channels are summed."""
    out = np.zeros((cfg.Nrx, cfg.Na_ch), complex)
    for j, p in enumerate(cfg.scene.points[1:]):
        out += S.generate_channels(cfg, tr, ptgs=[p], amps=[cfg.scene.amps[j]]) * np.exp(2j * np.pi * dC0[j] / cfg.system.wl)[:, None]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=DEFAULT, help="family:slug as in the sweep")
    ap.add_argument("--many", nargs="*", default=None, help="many points / continuous distributions (no ids: all)")
    ap.add_argument("--v4", nargs="*", default=None, help="case ids of sweep_v4 (no ids: the default set)")
    ap.add_argument("--map", default="lut", choices=["lut", "footprint"])
    ap.add_argument("--out", default=None, help="default sweep_<map>_cases")
    ap.add_argument("--full-line", action="store_true",
                    help="2D geometry: zoom on the targets and the reference point; 3D: the whole azimuth line of the acquisition")
    ap.add_argument("--compare-maps", action="store_true",
                    help="lower IRF panel: SATA with the other map (whole band and per sub-band) instead of no SATA")
    a = ap.parse_args()
    out = a.out or f"sweep_{a.map}_cases"
    if a.many is not None:
        out = a.out or f"sweep_many_{a.map}_cases"
        for cid, title, c in many_cases():
            if a.many and cid not in a.many:
                continue
            png, amb = run_case(c, cid, out, a.map, f"{cid}: {title}", full_line=a.full_line, compare=a.compare_maps)
            print(f"{cid:9s} " + " | ".join(f"{k} {v:.1f}" for k, v in amb.items()), flush=True)
        return
    if a.v4 is not None:
        import sweep_v4 as V
        jobs = {cid: c for c, cid, _, _ in V.all_jobs()}
        out = a.out or f"sweep_v4_{a.map}_cases"
        for cid in (a.v4 or V4_DEFAULT):
            c = jobs[cid]
            d, h = cid.split("-")[1]
            title = (f"{cid}: {LABEL[c.family]}, {c.N} targets {V4_DIST[d]}, {V4_H[h]}"
                     + (f", $\\alpha$ = {c.params['alpha']:g}$^\\circ$" if "alpha" in c.params else f" ({c.slug})"))
            png, amb = run_case(c, cid, out, a.map, title, full_line=a.full_line, compare=a.compare_maps)
            print(f"{cid} {c.family}:{c.slug:24s} " + " | ".join(f"{k} {v:.1f}" for k, v in amb.items()), flush=True)
        return
    cases = build_cases(FAMILIES)
    counter, ids = {}, {}
    for c in cases:                                         # same ids as the sweep
        counter[c.family] = counter.get(c.family, 0) + 1
        ids[(c.family, c.slug)] = (f"{PREFIX[c.family]}-{counter[c.family]:03d}", c)
    for key in a.cases:
        fam, slug = key.split(":")
        cid, case = ids[(fam, slug)]
        png, amb = run_case(case, cid, out, a.map, full_line=a.full_line, compare=a.compare_maps)
        print(f"{cid} {fam}:{slug:24s} " + " | ".join(f"{k} {v:.1f}" for k, v in amb.items()), flush=True)


if __name__ == "__main__":
    main()
