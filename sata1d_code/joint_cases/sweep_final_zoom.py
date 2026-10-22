"""
Some cases of the final-joint sweep (sweep_final.py) with zoomed IRFs, one page per case:
  top     the whole IRF: Reference, joint (final version) and no SATA, with the windows where the highest ambiguity is
          measured (grey);
  bottom  (a) zoom on the main lobe of the target whose ambiguity is the highest for the joint,
          (b) zoom on the position of the highest ambiguity of the joint,
          (c) zoom on the position of the highest ambiguity of no SATA.
Same computation and numbers as sweep_final.py (audit_dem.compute, every method reads only the DEM).

    python joint_cases/sweep_final_zoom.py --workers 4       # from sata1d_code
Output: joint_cases/sweep_final/zoom/<job>.png and sweep_final/sweep_final_zoom.pdf
"""
import argparse, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import sweep_final as SF
import run_joint_sweep as W

OUT = os.path.join(SF.OUT, "zoom")
CASES = [("one_refpoint", "u500"), ("one_ramp8", "N4b300"), ("one_flank", "N8b100"), ("few_ramp15_9", "N4b500"),
         ("few_sin50km_5", "s50"), ("few_hills_5", "u100"), ("few_plateau_9", "s50"), ("few_ramp20_7", "N2b500"),
         ("many_mountains_25", "u100"), ("many_hills_49", "N8b100")]
WIN, GUARD = 0.02, 0.02          # as in the ambiguity measure of the sweep
UP = 16                          # oversampling of the zoomed IRFs (zero padding of the spectrum)


def focused_complex(d):
    """The focused lines (complex) of Reference, no SATA and joint, recomputed exactly as in audit_dem.compute
    (same DEM table), so that the zooms can be oversampled. Returns {key: spectrum of the focused line}."""
    import sata1d as S
    from sata1d.joint_line import height_table, line_maps, reconstruct_joint_line
    cfg, tr, terrain = d["cfg"], d["tr"], d["c"]["terrain"]
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    hgrid = terrain(np.linspace(-half - 100.0, half + 100.0, 8001))
    tab = height_table(cfg, tr, hgrid.min(), hgrid.max())
    Mline = line_maps(cfg, tr, terrain, tab)
    ch = S.generate_channels(cfg, tr)
    Hf = np.conj(np.fft.fft(S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])))
    raw = {"ref": S.generate_reference(cfg, tr), "none": S.reconstruct(cfg, tr, ch),
           "joint_line": reconstruct_joint_line(cfg, tr, ch, Mline)}
    return {k: np.fft.fft(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2)) for k, s in raw.items()}


def oversampled(spec, prf, t0, lo, hi):
    """|focused line| on a grid UP times finer, between lo and hi seconds (t = (n - Na/2)/prf)."""
    Na = len(spec)
    Z = np.zeros(Na * UP, complex)
    h = Na // 2
    Z[:h] = spec[:h]; Z[-h:] = spec[-h:]
    z = np.fft.ifft(Z) * UP
    tt = (np.arange(Na * UP) / UP - Na // 2) / prf
    sel = (tt >= lo) & (tt <= hi)
    return tt[sel], np.abs(z[sel])


def where_highest(t, env, pk, xs, vs, dt_amb, orders):
    """(time, level dB, target index, order) of the highest ambiguity, with the same windows as the sweep measure."""
    lobe = np.zeros(len(t), bool)
    for x in xs:
        lobe |= np.abs(t - x / vs) <= GUARD
    best = (None, -np.inf, None, None)
    for j, x in enumerate(xs):
        for m in orders:
            sel = (np.abs(t - (x / vs + m * dt_amb)) <= WIN) & ~lobe
            if sel.any():
                k = np.flatnonzero(sel)[np.argmax(env[sel])]
                lv = 20 * np.log10(env[k] / pk + 1e-15)
                if lv > best[1]:
                    best = (t[k], lv, j, m)
    return best


def page(res, d, out_png):
    c, cfg, env, amb, t, pk = d["c"], d["cfg"], d["env"], d["amb"], d["t"], d["pk"]
    xs, vs = np.asarray(c["xs"], float), cfg.system.vs
    dt_amb = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * vs ** 2)
    orders = res["orders"]
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    specs = focused_complex(d)
    chk = {k: float(np.max(np.abs(np.abs(np.fft.ifft(specs[k])) - env[k])) / pk) for k in specs}
    assert max(chk.values()) < 1e-6, chk                    # the same lines as the sweep
    hj = where_highest(t, env["joint_line"], pk, xs, vs, dt_amb, orders)
    hn = where_highest(t, env["none"], pk, xs, vs, dt_amb, orders)
    curves = [("ref", "Reference", dict(color="tab:green", lw=2.0)),
              ("none", "no SATA", dict(color="tab:red", lw=1.2, ls=(0, (5, 3)))),
              ("joint_line", "joint, final version", dict(color="tab:purple", lw=1.2))]
    fig = plt.figure(figsize=(20, 10.6))
    gs = GridSpec(2, 3, height_ratios=[1, 1.05], hspace=0.36, wspace=0.16, figure=fig, left=0.045, right=0.985, top=0.88, bottom=0.08)
    lab = res["label"] if res["label"].startswith("h =") else res["label"][0].upper() + res["label"][1:]
    fig.suptitle(f"{lab} | $N_{{rx}}$ = {cfg.Nrx}, $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)} ({d['kind']}) | every method reads only the DEM",
                 fontsize=13, y=0.975)

    def windows(ax, lo, hi):
        for x in xs:
            for m in orders:
                tc = x / vs + m * dt_amb
                if hi > tc - WIN and lo < tc + WIN:
                    ax.axvspan(tc - WIN, tc + WIN, color="0.88", zorder=0, lw=0)

    ax = fig.add_subplot(gs[0, :])
    windows(ax, t[0], t[-1])
    for k, name, st in curves:
        ax.plot(t, db[k], label=name, **st)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(-90, 5); ax.grid(alpha=0.3)
    ax.set_xlabel("Azimuth time [s]"); ax.set_ylabel("[dB]"); ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"Whole IRF | highest ambiguity: reference {amb['ref']:.1f} dB | no SATA {amb['none']:.1f} dB | "
                 f"joint {amb['joint_line']:.1f} dB   (grey: the $\\pm$20 ms windows where it is measured)", fontsize=10.5)
    for tt, col in ((hj[0], "tab:purple"), (hn[0], "tab:red")):
        if tt is not None:
            ax.axvline(tt, color=col, lw=0.8, ls=":")

    def zoom(axz, tc, half, title, mark=None, ylim=(-90, 5)):
        lo, hi = tc - half, tc + half
        windows(axz, lo, hi)
        for k, name, st in curves:
            tt, e = oversampled(specs[k], cfg.prf, tc, lo, hi)
            axz.plot(1e3 * (tt - tc), 20 * np.log10(e / pk + 1e-12), label=name, **st)
        if mark is not None:
            axz.plot(0, mark, "o", ms=8, mfc="none", mec="k", mew=1.3, zorder=6)
        axz.set_xlim(-1e3 * half, 1e3 * half); axz.set_ylim(*ylim); axz.grid(alpha=0.3)
        axz.set_xlabel(f"Azimuth time relative to {tc:+.3f} s [ms]"); axz.set_ylabel("[dB]")
        axz.set_title(title, fontsize=10)

    j = hj[2] if hj[2] is not None else 0
    t0 = xs[j] / vs
    a1 = fig.add_subplot(gs[1, 0])
    zoom(a1, t0, 0.008, f"(a) main lobe of target {j + 1} of {len(xs)} ($x$ = {xs[j] / 1e3:.2f} km), oversampled x{UP}",
         ylim=(-70, 3))
    a1.legend(fontsize=8.5, loc="upper right")
    if hj[0] is not None:
        zoom(fig.add_subplot(gs[1, 1]), hj[0], 0.06,
             f"(b) highest ambiguity of the joint: {hj[1]:.1f} dB (target {hj[2] + 1}, order {hj[3]:+d}), oversampled x{UP}", mark=hj[1])
    if hn[0] is not None:
        zoom(fig.add_subplot(gs[1, 2]), hn[0], 0.06,
             f"(c) highest ambiguity of no SATA: {hn[1]:.1f} dB (target {hn[2] + 1}, order {hn[3]:+d}), oversampled x{UP}", mark=hn[1])
    fig.text(0.5, 0.012, "Highest ambiguity in dB relative to the reference peak, inside $\\pm$20 ms of the ambiguity positions of every target, "
             f"outside $\\pm$20 ms of every target's own main lobe (orders {', '.join(f'{m:+d}' for m in orders)}).\n"
             f"PRF = {cfg.prf:.0f} Hz, $B_a$ = {cfg.abw:.0f} Hz, channel 0 is the transmitter. The reference is not cut to $B_a$. Circles: the level measured by the sweep (on the samples).",
             ha="center", fontsize=8.5, color="0.35")
    fig.savefig(out_png, dpi=90); plt.close(fig)


def _job(case):
    sc, tag = case
    t1 = time.time()
    res, d = SF.compute(sc, tag)
    page(res, d, os.path.join(OUT, f"{sc}__{tag}.png"))
    return f"{sc:20s} {tag:7s} ref {res['ref']:6.1f} none {res['none']:6.1f} joint {res['joint_line']:6.1f} ({time.time() - t1:.0f} s)"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.pdf:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for line in pool.imap_unordered(_job, CASES):
                print(line, flush=True)
    from joint_sweep_summary import bundle
    pngs = [os.path.join(OUT, f"{sc}__{tag}.png") for sc, tag in CASES]
    outline = [(f"{sc} | {tag}", i, 0) for i, (sc, tag) in enumerate(CASES)]
    print("wrote", bundle(pngs, os.path.join(SF.OUT, "sweep_final_zoom.pdf"), outline))


if __name__ == "__main__":
    main()
