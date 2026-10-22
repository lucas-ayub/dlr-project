"""
Distributed scene test: every method with the same map, the DEM look-up table (terrain_lookup_table).
Scene: scatterers at random azimuths over +-L on the iso-range surface at the terrain height, one per ~3 m, complex
Gaussian amplitudes, with three dark gaps (no scatterers, the DEM still has the terrain there) where the
ambiguities of the neighbouring folds land.
Methods: no SATA, SATA whole band (DEM), SATA per sub-band (DEM), joint occupied / full (DEM), oracle (every
scatterer corrected with its own delta_C0, from the same table); reference = monostatic full-PRF line.
Metrics, inside |x| <= EVAL:
    error-to-signal ratio  10 log10( sum |a y - y_ref|^2 / sum |y_ref|^2 ), a = best complex scale
    gap ratio              mean |y|^2 inside the gaps / mean |y|^2 outside them (dark-region ambiguity level)
Every focused line, the reference included, is limited to the processed band B_a with a Hamming weighting
(low sidelobes, so that the gaps are dark in the reference) before the comparison.
    SATA_LUT_CACHE is not used; tables and channels are cached in DIST_CACHE (default /tmp/dist_cache).
    python joint_cases/distributed_test.py --budget 120            # from sata1d_code, repeat until done
Output: joint_cases/distributed/<baseline>/<case>.png/.json, joint_cases/distributed/distributed_test.pdf
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
import sata1d.sata as SS
from sata1d.config import ArrayGeometry
from sata1d.joint import reconstruct_joint
from sata1d.subband import sata_channels_subband, channel_spectrum
from sata1d.reconstruction import reconstruct_from_spectra
import run_joint_sweep as W
import joint_sweep_summary as JS
import sweep_5panel as P5

OUT = os.path.join(HERE, "distributed")
CACHE = os.environ.get("DIST_CACHE", "/tmp/dist_cache")
CASES = ["FLAT", "C15", "C24", "C29", "C31", "C33"]
BASELINES = ["rnd100", "rnd500", "s50"]
L, DENS, SEED = 20000.0, 1.0 / 3.0, 7                    # scene half-length [m], scatterers per m, seed
GAPS = [(-6000.0, 1500.0), (-1500.0, 1500.0), (3000.0, 1500.0)]   # (centre, width) [m]
EVAL, MARGIN, SMOOTH = 7000.0, 150.0, 400.0              # evaluation half-length, gap-edge margin, smoothing [m]
VERSION = 3                                              # every line limited to B_a with a Hamming weighting
METHODS = [("none", "no SATA", "tab:red", (0, (5, 3))), ("whole", "SATA whole band (DEM)", "tab:blue", (0, (5, 3))),
           ("sub", "SATA per sub-band (DEM)", "tab:orange", "-"), ("joint_occ", "joint, occupied parcels (DEM)", "tab:purple", "-"),
           ("joint_full", "joint, full system (DEM)", "tab:pink", (0, (5, 3))), ("oracle", "oracle", "k", ":")]


def case_terrain(cid):
    if cid == "FLAT":
        return (lambda x: np.zeros_like(np.asarray(x, float))), "flat terrain, h = 0 (sanity check: no topographic phase)"
    c = W.case_info(cid)
    lab = ", ".join(p for p in c["label"].split(", ") if "target" not in p)
    return c["terrain"], f"case {int(cid[1:])}: {lab}"


def scene():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-L, L, int(2 * L * DENS))
    keep = np.ones(x.size, bool)
    for c, w in GAPS:
        keep &= np.abs(x - c) > w / 2
    x = np.sort(x[keep])
    a = (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) / np.sqrt(2.0)
    return x, a


def make(cid, btag):
    terrain, label = case_terrain(cid)
    xs, amps = scene()
    cfg = S.make_config(list(zip(xs, terrain(xs))), bxt_max=0.0, amplitudes=amps, tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=W.BXT[btag].copy()))
    return cfg, S.build_platform_tracks(cfg), terrain, label, xs, amps


def lut(cfg, tr, i, terrain, cid):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"lut_{cid}_ch{i}_bat{cfg.array.bat[i]:.3f}_bxt{cfg.array.bxt[i]:g}_n{cfg.Na_ch}.npy")
    if os.path.exists(f):
        return np.load(f)
    m = S.terrain_lookup_table(cfg, tr, i, terrain, step=4)
    np.save(f, m)
    return m


def generate(cfg, tr, maps):
    """Reference (monostatic, full PRF), channels and oracle channels, as getRawData1D but each scatterer only
    on the samples around its beam (identical inside the beam, zero outside)."""
    Na, Nrx, wl, prf = cfg.Na, cfg.Nrx, cfg.system.wl, cfg.prf
    pts, amps = cfg.scene.points[1:], np.asarray(cfg.scene.amps, complex)
    lim = cfg.sq_tx + cfg.theta_tx / 2.0
    half = int(np.ceil(1.2 * cfg.scene.r0 * np.tan(lim) / cfg.system.vs * prf))
    ref = np.zeros(Na, complex); ch = np.zeros((Nrx, Na), complex); och = np.zeros((Nrx, Na), complex)
    ds = cfg.system.vs / cfg.PRF_op
    xs = pts[:, 0] - cfg.scene.x0
    # oracle phase of every scatterer in every channel: the table at its (fractional) cell
    dc = np.array([np.interp(cfg.Na_ch / 2 + (cfg.scene.x0 + xs + 0.5 * cfg.array.bat[i]) / ds, np.arange(cfg.Na_ch), maps[i])
                   for i in range(Nrx)])
    n0 = np.searchsorted(tr.ptx[:, 0], pts[:, 0])
    for p, (pt, a) in enumerate(zip(pts, amps)):
        lo, hi = max(n0[p] - half, 0), min(n0[p] + half, Na)
        sl = slice(lo, hi)

        def win(r, v):
            s = np.empty(hi - lo)
            s[:-1] = np.arcsin(np.diff(r) * prf / v[lo + 1:hi])
            s[-1] = 2 * s[-2] - s[-3]
            return np.abs(s) <= lim
        rt = np.sqrt(np.sum((tr.ptx[sl] - pt) ** 2, axis=1))
        wt = win(rt, tr.vtx)
        ref[sl] += a * (wt & wt) * np.exp(-2j * np.pi * 2.0 * rt / wl)
        for i in range(Nrx):
            rr = np.sqrt(np.sum((tr.prx[i, sl] - pt) ** 2, axis=1))
            e = a * (wt & win(rr, tr.vrx[i])) * np.exp(-2j * np.pi * (rt + rr) / wl)
            ch[i, sl] += e
            och[i, sl] += e * np.exp(2j * np.pi * dc[i, p] / wl)
    return ref, ch[:, ::Nrx], och[:, ::Nrx]


def check_generator():
    """The windowed generator against getRawData1D on 40 scatterers of the C24 / U(0, 500 m) scene."""
    from sata1d.signal_model import getRawData1D
    cfg, tr, terrain, _, _, _ = make("C24", "rnd500")
    sub = slice(6000, 6040)
    c2 = replace(cfg, scene=replace(cfg.scene, extra_offsets=cfg.scene.extra_offsets[sub], amplitudes=tuple(cfg.scene.amps[sub])))
    maps = np.zeros((c2.Nrx, c2.Na_ch))
    ref, ch, _ = generate(c2, tr, maps)
    pts, amps = c2.scene.points[1:], c2.scene.amps
    r0 = getRawData1D(pts, tr.ptx, tr.ptx, tr.vtx, tr.vtx, c2.ta, c2.sq_tx, c2.sq_tx, c2.theta_tx, c2.theta_tx, c2.system.wl, c2.prf, amps)
    c1 = getRawData1D(pts, tr.ptx, tr.prx[1], tr.vtx, tr.vrx[1], c2.ta, c2.sq_tx, c2.sq_tx, c2.theta_tx, c2.theta_tx, c2.system.wl, c2.prf, amps)[::c2.Nrx]
    print("generator check: max |diff| ref", np.abs(ref - r0).max(), "channel 1", np.abs(ch[1] - c1).max(),
          "| max |signal|", np.abs(r0).max(), flush=True)


def stage(cid, btag, deadline):
    """Tables, channels and reconstructions of one (case, baseline); returns the result dict or None (budget)."""
    cfg, tr, terrain, label, xs, amps = make(cid, btag)
    maps = []
    for i in range(cfg.Nrx):
        if time.time() > deadline:
            return None
        maps.append(lut(cfg, tr, i, terrain, cid))
    maps = np.array(maps)
    fg = os.path.join(CACHE, f"gen_{cid}_{btag}.npz")
    if not os.path.exists(fg):
        if time.time() > deadline:
            return None
        ref, ch, och = generate(cfg, tr, maps)
        np.savez(fg, ref=ref, ch=ch, och=och)
        return "generated"
    z = np.load(fg); ref, ch, och = z["ref"], z["ch"], z["och"]
    if time.time() > deadline:
        return None
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    fr = np.fft.fftfreq(cfg.Na, 1.0 / cfg.prf)
    band = (np.abs(fr) <= cfg.abw / 2.0) * (0.54 + 0.46 * np.cos(2 * np.pi * fr / cfg.abw))   # B_a, Hamming, every line
    F = lambda s: np.roll(np.fft.ifft(np.fft.fft(s) * Hf * band), cfg.Na // 2)

    def whole():
        o = ch.copy()
        for i in range(cfg.Nrx):
            o[i] = SS.sata_1d(o[i], maps[i], rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=cfg.system.wl,
                              r=cfg.scene.r0, squint=cfg.sq_tx, inverse=True, sata_osf=W.SATA_OSF)
        return S.reconstruct(cfg, tr, o)

    def sub():
        spec = []
        for k in range(cfg.Nsb):
            corr = sata_channels_subband(cfg, tr, ch, k, sata_osf=W.SATA_OSF, maps=list(maps))
            spec.append([channel_spectrum(cfg, corr[j]) for j in range(cfg.Nrx)])
        return reconstruct_from_spectra(cfg, tr, spec, True)

    y = {"ref": F(ref), "none": F(S.reconstruct(cfg, tr, ch)), "whole": F(whole()), "sub": F(sub()),
         "joint_occ": F(reconstruct_joint(cfg, tr, ch, maps=maps, sata_osf=W.SATA_OSF, occupied=True)),
         "joint_full": F(reconstruct_joint(cfg, tr, ch, maps=maps, sata_osf=W.SATA_OSF, occupied=False)),
         "oracle": F(S.reconstruct(cfg, tr, och))}
    return cfg, terrain, label, y, maps


def metrics(cfg, y):
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    x = t * cfg.system.vs
    ev = np.abs(x) <= EVAL
    gap = np.zeros_like(ev); near = np.zeros_like(ev)
    for c, w in GAPS:
        gap |= np.abs(x - c) <= w / 2 - MARGIN
        near |= np.abs(x - c) <= w / 2 + MARGIN
    lit = ev & ~near
    yr = y["ref"]
    p_ref = np.mean(np.abs(yr[lit]) ** 2)
    m, err = {}, {}
    k = max(int(SMOOTH / (cfg.system.vs / cfg.prf)), 1)
    box = np.ones(k) / k
    for key, v in y.items():
        a = np.vdot(v[ev], yr[ev]) / np.vdot(v[ev], v[ev])
        e = np.abs(a * v - yr) ** 2
        m[key] = {"esr": float(10 * np.log10(e[ev].sum() / (np.abs(yr[ev]) ** 2).sum() + 1e-30)),
                  "gap": float(10 * np.log10(np.mean(np.abs(v[gap]) ** 2) / np.mean(np.abs(v[lit]) ** 2)))}
        err[key] = 10 * np.log10(np.convolve(e, box, "same") / p_ref + 1e-30)
        err[key + "_pow"] = 10 * np.log10(np.convolve(np.abs(v) ** 2, box, "same") / p_ref + 1e-30)
    return t, x, m, err


def page(cid, btag, cfg, terrain, label, m, t, x, err, out):
    fig = plt.figure(figsize=(22, 11.5))
    gs = GridSpec(6, 2, width_ratios=[1.5, 1.0], hspace=1.1, wspace=0.16, left=0.05, right=0.985, top=0.9, bottom=0.08)
    fig.suptitle(f"Distributed scene, {label} | $b_{{xt}}$ = {W.bxt_text(W.BXT[btag])} ({W.BKIND[btag]}) | "
                 f"every method with the DEM look-up table", fontsize=13, y=0.975)
    ap = fig.add_subplot(gs[0:3, 0]); ae = fig.add_subplot(gs[3:6, 0])
    sel = np.abs(x) <= EVAL + 2500
    ap.plot(t[sel], err["ref_pow"][sel], color="tab:green", lw=2.0, label="Reference")
    for key, lab, col, ls in METHODS:
        if key in ("none", "whole", "joint_occ"):
            ap.plot(t[sel], err[key + "_pow"][sel], color=col, ls=ls, lw=1.3, label=lab)
    for key, lab, col, ls in METHODS:
        ae.plot(t[sel], err[key][sel], color=col, ls=ls, lw=1.4 if key != "oracle" else 1.8,
                label=f"{lab}: {m[key]['esr']:.1f} dB")
    for a_ in (ap, ae):
        for c, w in GAPS:
            a_.axvspan((c - w / 2) / cfg.system.vs, (c + w / 2) / cfg.system.vs, color="0.85", zorder=0)
        a_.axvline(-EVAL / cfg.system.vs, color="0.5", lw=0.8, ls=":"); a_.axvline(EVAL / cfg.system.vs, color="0.5", lw=0.8, ls=":")
        a_.set_xlim(t[sel][0], t[sel][-1]); a_.grid(alpha=0.3); a_.set_xlabel("Azimuth time [s]")
    ap.set_ylim(-75, 8); ae.set_ylim(-75, 8)
    ap.set_ylabel(f"power, {SMOOTH:.0f} m average [dB re. scene]")
    ae.set_ylabel(f"error power, {SMOOTH:.0f} m average [dB re. scene]")
    ap.set_title("Focused line (grey: dark gaps without scatterers; dotted: evaluation window)", fontsize=10)
    ae.set_title("Error against the reference, |a y - y_ref|^2 (legend: error-to-signal ratio in the window)", fontsize=10)
    ap.legend(fontsize=9, loc="lower left", ncol=2); ae.legend(fontsize=8.6, loc="lower left", ncol=2)
    # terrain and dh_X
    at = fig.add_subplot(gs[0:2, 1])
    X = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs)
    xg = np.linspace(-L, L, 4001); hg = terrain(xg)
    at.fill_between(xg / 1e3, min(hg.min(), 0) - 20, hg, color="#D7C9A7", alpha=0.6, lw=0)
    at.plot(xg / 1e3, hg, color="#6D4C2F", lw=1.2, label="terrain h(x) (scatterers on it)")
    for c, w in GAPS:
        at.axvspan((c - w / 2) / 1e3, (c + w / 2) / 1e3, color="0.85", zorder=0)
    at.plot(xg / 1e3, terrain(xg + X) - hg, color="tab:cyan", lw=1.1, label="$\\Delta h_X$(x) = h(x + X) - h(x)")
    at.axvline(-EVAL / 1e3, color="0.5", lw=0.8, ls=":"); at.axvline(EVAL / 1e3, color="0.5", lw=0.8, ls=":")
    at.set_xlim(-L / 1e3, L / 1e3); at.grid(alpha=0.3)
    at.set_xlabel("azimuth position x [km]"); at.set_ylabel("[m]")
    at.set_title(f"Scene: {int(cfg.scene.points.shape[0] - 1)} scatterers over $\\pm${L / 1e3:.0f} km, complex Gaussian amplitudes", fontsize=10)
    at.legend(fontsize=8, loc="upper left")
    cg = fig.add_subplot(gs[2:4, 1])
    P5.channel_geometry(cg, cfg)
    ab = fig.add_subplot(gs[4:6, 1])
    keys = ["ref"] + [k for k, _, _, _ in METHODS]
    labs = ["Reference"] + [l for _, l, _, _ in METHODS]
    cols = ["tab:green"] + [c for _, _, c, _ in METHODS]
    yy = np.arange(len(keys))
    ab.barh(yy - 0.2, [m[k]["gap"] for k in keys], height=0.38, color=cols, alpha=0.55, label="gap ratio")
    ab.barh(yy[1:] + 0.2, [m[k]["esr"] for k in keys[1:]], height=0.38, color=cols[1:], label="error-to-signal ratio")
    for j, k in enumerate(keys):
        ab.text(m[k]["gap"] - 0.5, j - 0.2, f"{m[k]['gap']:.1f}", ha="right", va="center", fontsize=8)
        if j:
            ab.text(m[k]["esr"] - 0.5, j + 0.2, f"{m[k]['esr']:.1f}", ha="right", va="center", fontsize=8, fontweight="bold")
    ab.set_yticks(yy); ab.set_yticklabels(labs, fontsize=8.5); ab.invert_yaxis()
    ab.set_xlim(-60, 0); ab.grid(alpha=0.3, axis="x"); ab.set_xlabel("[dB]")
    ab.set_title("Error-to-signal ratio (dark bars, bold) and gap ratio (light bars)", fontsize=10)
    fig.text(0.5, 0.012, f"Error-to-signal ratio: sum |a y - y_ref|$^2$ / sum |y_ref|$^2$ over |x| <= {EVAL / 1e3:.0f} km (a: best complex scale). "
             f"Gap ratio: mean power inside the gaps ({MARGIN:.0f} m from the edges) / mean power outside them. Gaps of "
             f"{GAPS[0][1] / 1e3:.1f} km at x = {', '.join(f'{c / 1e3:g}' for c, _ in GAPS)} km. $N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, "
             f"channel 0 is the transmitter, $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)}.", ha="center", fontsize=8.5, color="0.35")
    d = os.path.join(out, btag); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"{cid}.png"), dpi=85); plt.close(fig)


def summary(out):
    res = {(c, b): json.load(open(os.path.join(out, b, f"{c}.json"))) for c in CASES for b in BASELINES
           if os.path.exists(os.path.join(out, b, f"{c}.json"))}
    rows = [(c, b) for b in BASELINES for c in CASES if (c, b) in res]
    keys = [k for k, _, _, _ in METHODS]
    fig, axs = plt.subplots(1, 2, figsize=(22, 0.42 * len(rows) + 3.2))
    for ax, met, title in ((axs[0], "esr", "error-to-signal ratio [dB]"), (axs[1], "gap", "gap ratio [dB] (reference in the first column)")):
        ks = keys if met == "esr" else ["ref"] + keys
        M = np.array([[res[r]["metrics"][k][met] for k in ks] for r in rows])
        im = ax.imshow(M, cmap="RdYlGn_r", vmin=-50, vmax=0, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8.5)
        ax.set_xticks(range(len(ks)))
        short = {"ref": "Reference", "none": "no SATA", "whole": "SATA whole\nband (DEM)", "sub": "SATA per\nsub-band (DEM)",
                 "joint_occ": "joint, occupied\nparcels (DEM)", "joint_full": "joint, full\nsystem (DEM)", "oracle": "oracle"}
        ax.set_xticklabels([short[k] for k in ks], fontsize=7.4)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{res[r]['label'][:60]} | {W.bxt_text(W.BXT[r[1]])}" for r in rows], fontsize=8)
        ax.set_title(title, fontsize=11, pad=30); fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("Distributed scenes, every method with the DEM look-up table (lower is better)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out, "summary.png"); fig.savefig(p, dpi=85); plt.close(fig)
    pngs = [p] + [os.path.join(out, b, f"{c}.png") for c, b in rows]
    outline = [("Summary", 0, 0)] + [(f"{res[r]['label'][:70]} | b_xt = {W.bxt_text(W.BXT[r[1]])}", k + 1, 0) for k, r in enumerate(rows)]
    return JS.bundle(pngs, os.path.join(out, "distributed_test.pdf"), outline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--baselines", nargs="+", default=BASELINES)
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--check", action="store_true", help="only compare the generator with getRawData1D")
    ap.add_argument("--summary", action="store_true", help="only the summary page and the PDF")
    a = ap.parse_args()
    if a.check:
        return check_generator()
    if a.summary:
        return print("wrote", summary(OUT))
    deadline = time.time() + a.budget
    for b in a.baselines:
        for cid in a.cases:
            fj = os.path.join(OUT, b, f"{cid}.json")
            if os.path.exists(fj) and json.load(open(fj)).get("version") == VERSION:
                continue
            t0 = time.time()
            r = stage(cid, b, deadline)
            if r is None:
                return print("budget reached, run again", flush=True)
            if r == "generated":
                print(cid, b, f"generated ({time.time() - t0:.0f} s)", flush=True)
                r = stage(cid, b, deadline)
                if r is None:
                    return print("budget reached, run again", flush=True)
            cfg, terrain, label, y, maps = r
            t, x, m, err = metrics(cfg, y)
            page(cid, b, cfg, terrain, label, m, t, x, err, OUT)
            json.dump({"case": cid, "baseline": b, "label": label, "metrics": m, "version": VERSION}, open(os.path.join(OUT, b, f"{cid}.json"), "w"), indent=1)
            print(cid, b, {k: (round(v["esr"], 1), round(v["gap"], 1)) for k, v in m.items()}, f"({time.time() - t0:.0f} s)", flush=True)
    print("all done; summary:", summary(OUT), flush=True)


if __name__ == "__main__":
    main()
