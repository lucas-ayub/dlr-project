"""Animated GIF of the joint reconstruction on one azimuth line: the geometry, the four receivers passing over a
1000 m Gaussian mountain with five point targets, the channels they record, and at the end the focused line
(reference, standard reconstruction, first estimate A^H s and joint reconstruction).

Everything comes from the repository's own code (sata1d): make_config, build_platform_tracks, generate_channels,
generate_reference, reconstruct, height_table, line_maps, LineModel and reconstruct_joint_line.
Default formation: N_rx = 4 (Rx0 = Tx), 500 Hz per channel (2000 Hz together), cross-track baselines
0 / 300 / 119.8 / 100.5 m, lambda = 0.25 m, r0 = 766 km. Same scene as the presentation's mountain slide and the
report: Gaussian mountain (sigma 1.5 km, 1000 m), targets at x = -8, -0.55, 0.65, 2.9, 4.1 km on the surface,
heights relative to the mean height of the line.

    python sata1d_code/joint_cases/animation/make_gif.py            # writes report/figures/readme_animation.gif
    python sata1d_code/joint_cases/animation/make_gif.py --mp4      # also the .mp4 next to it (needs ffmpeg)
    python sata1d_code/joint_cases/animation/make_gif.py --out demo.gif
"""
import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.abspath(os.path.join(HERE, "..", ".."))            # sata1d_code/
sys.path.insert(0, SC)
import sata1d as S                                                             # noqa: E402
from sata1d.config import ArrayGeometry                                        # noqa: E402
from sata1d.joint_line import LineModel, height_table, line_maps, reconstruct_joint_line  # noqa: E402

# ---------------------------------------------------------------- scene and system
XS = np.array([-8000.0, -550.0, 650.0, 2900.0, 4100.0])      # targets [m]
BXT = np.array([0.0, 300.0, 119.8, 100.5])                   # cross-track baselines [m], Rx0 = Tx
H_MTN, SIGMA, HALF = 1000.0, 1500.0, 23600.0
B_ECHO = 1281.0                                              # Doppler band of one point echo [Hz]

C_REF, C_STD, C_JNT, C_FIRST = "#2ca02c", "#d62728", "#7b52ab", "#b9a3d6"
C_TER, C_TER_FILL, C_TGT, C_LIT = "#7a5c3a", "#efe6d8", "#3a3a3a", "#f0a020"
C_RX = ["#1f77b4", "#ff7f0e", "#17becf", "#e377c2"]


def mountain(x):
    return H_MTN * np.exp(-0.5 * (np.asarray(x, float) / SIGMA) ** 2)


def simulate():
    xg = np.linspace(-HALF, HALF, 9441)
    href = float(np.mean(mountain(xg)))
    ter = lambda q: mountain(q) - href                         # heights relative to the mean of the line
    cfg = S.make_config(list(zip(XS, ter(XS))), nrx=4, prf=2000.0, bxt_max=0.0, tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=BXT.copy()))
    tr = S.build_platform_tracks(cfg)

    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    Hf = np.conj(np.fft.fft(sref))
    focus = lambda s: np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2)

    def band(s):
        U = np.fft.fft(s)
        U[np.abs(np.fft.fftfreq(cfg.Na, 1.0 / cfg.prf)) > cfg.abw / 2] = 0
        return np.fft.ifft(U)

    ch = S.generate_channels(cfg, tr)                          # [Nrx, Na/Nrx] what the receivers record
    ref = focus(band(S.generate_reference(cfg, tr)))           # one radar at the full PRF (the ideal)
    std = focus(S.reconstruct(cfg, tr, ch))                    # standard reconstruction (flat-earth filters)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    hgrid = ter(np.linspace(-half - 100.0, half + 100.0, 8001))
    tab = height_table(cfg, tr, hgrid.min() - 60.0, hgrid.max() + 60.0)
    dC0 = line_maps(cfg, tr, ter, tab)                         # DEM phases of every channel at every place
    # first estimate A^H s: the model's adjoint returns the line in its own (focused, FFT-ordered) form; like
    # reconstruct_joint_line, put the echo of the reference point back on it so that it is focused as the others
    adj = LineModel(cfg, tr, dC0).adjoint(ch)
    first = focus(band(np.fft.ifft(np.fft.fft(adj) * np.fft.fft(sref))))
    joint, info = reconstruct_joint_line(cfg, tr, ch, dC0, return_info=True)
    joint = focus(joint)

    # illumination window of every target (where its echo is non-zero), from the simulator itself
    lit = np.array([np.abs(S.generate_reference(cfg, tr, cfg.scene.points[1 + k][None, :])) > 0
                    for k in range(len(XS))])
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    vs = cfg.system.vs
    dt_amb = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * vs ** 2)
    return dict(cfg=cfg, ch=ch, ref=ref, std=std, first=first, joint=joint, lit=lit, t=t, ta=cfg.ta, vs=vs,
                dt_amb=dt_amb, href=href, iters=info.get("iterations"), Lsa=cfg.system.wl * cfg.scene.r0 / cfg.system.La)


def highest_ambiguity(r, key, win=0.02, guard=0.02):
    """Largest |line| within +-win s of every ambiguity position x_p/v + m dt_amb (0 < |m| <= B_echo/PRF_op), outside
    +-guard s around every target; dB re. the reference peak (the metric of the presentation and the report)."""
    t, env, pk = r["t"], np.abs(r[key]), np.abs(r["ref"]).max()
    M = int(B_ECHO // r["cfg"].PRF_op)
    lobe = np.zeros(t.size, bool)
    sel = np.zeros(t.size, bool)
    for x in XS:
        lobe |= np.abs(t - x / r["vs"]) <= guard
        for m in range(-M, M + 1):
            if m:
                sel |= np.abs(t - (x / r["vs"] + m * r["dt_amb"])) <= win
    sel &= ~lobe
    return 20 * np.log10(env[sel].max() / pk + 1e-15)


def db(line, pk):
    return 20 * np.log10(np.abs(line) / pk + 1e-12)


def make(out, fps=12, mp4=False):
    t0 = time.time()
    r = simulate()
    amb = {k: highest_ambiguity(r, k) for k in ("ref", "std", "first", "joint")}
    amb["first"] += 20 * np.log10(np.abs(r["ref"]).max() / np.abs(r["first"]).max())   # A^H s: peak-normalised
    print("highest ambiguity [dB re. reference peak]: " + ", ".join(f"{k} {v:.1f}" for k, v in amb.items()))
    print(f"joint: {r['iters']} iterations; simulation {time.time() - t0:.0f} s")

    cfg, vs = r["cfg"], r["vs"]
    xkm = r["t"] * vs / 1e3                                    # position along the line [km]
    pk = np.abs(r["ref"]).max()
    pk_first = np.abs(r["first"]).max()
    ta_ch = r["ta"][::cfg.Nrx]                                 # channel sampling instants (PRF / N_rx)
    xch = ta_ch * vs / 1e3
    xs_km = XS / 1e3
    hs_abs = mountain(XS)
    xlim = (-20.0, 20.0)

    # ------------------------------------------------ frames: acquisition, then the result
    n_acq, n_res = 110, 64
    xpl = np.linspace(xlim[0] - 6.0, xlim[1] + 6.0, n_acq)  # platform position (beam centre) [km]
    stages = [("ref", 10), ("std", 14), ("first", 14), ("joint", 26)]

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    gs = fig.add_gridspec(6, 1, height_ratios=[1.55, 0.2, 0.5, 0.5, 0.5, 0.5], hspace=0.1,
                          left=0.075, right=0.985, top=0.9, bottom=0.09)
    ax_top = fig.add_subplot(gs[0])
    ax_ch = [fig.add_subplot(gs[i + 2], sharex=ax_top) for i in range(4)]
    ax_irf = fig.add_subplot(gs[2:], sharex=ax_top)
    ax_irf.set_visible(False)
    title = fig.suptitle("", fontsize=13, x=0.075, ha="left", y=0.975)

    # top panel: terrain, targets, formation and beam (vertical scale broken: satellite not to scale)
    xt = np.linspace(xlim[0], xlim[1], 1200)
    ax_top.fill_between(xt, -150, mountain(xt * 1e3), color=C_TER_FILL, zorder=1)
    ax_top.plot(xt, mountain(xt * 1e3), color=C_TER, lw=1.4, zorder=2)
    tg = ax_top.scatter(xs_km, hs_abs, s=42, color=C_TGT, zorder=5, edgecolor="white", linewidth=0.8)
    ax_top.set_ylim(-150, 3300)
    ax_top.set_yticks([0, 1000])
    ax_top.set_ylabel("h [m]", fontsize=9)
    ax_top.tick_params(labelsize=8, labelbottom=False)
    ax_top.axhline(2850, color="0.75", lw=0.6, ls=(0, (4, 3)), zorder=0)
    beam = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, color=C_LIT, alpha=0.18, lw=0, zorder=0)
    ax_top.add_patch(beam)
    foot, = ax_top.plot([], [], color=C_LIT, lw=3, alpha=0.8, solid_capstyle="butt", zorder=3)
    sats = [ax_top.plot([], [], marker="s", ms=5.5, color=C_RX[i], ls="none", zorder=6)[0] for i in range(4)]
    sat_lab = ax_top.text(0, 3020, "", fontsize=8, ha="left", va="bottom", color="0.25")
    ax_top.text(xlim[0] + 0.2, 2930, "orbit (766 km away, not to scale)", fontsize=7.5, color="0.45", va="bottom")
    ins = ax_top.inset_axes([0.80, 0.40, 0.19, 0.5], zorder=10)
    ins.set_facecolor("white")
    for i in range(4):
        ins.plot(cfg.array.bat[i], BXT[i], "s", color=C_RX[i], ms=5)
        dy = {2: 28, 3: -30}.get(i, 0)
        ins.text(cfg.array.bat[i] + 2.0, BXT[i] + dy, ("Tx = Rx0" if i == 0 else f"Rx{i}") + f": {BXT[i]:g} m",
                 fontsize=7, va="center", color=C_RX[i])
    ins.set_xlim(-3, 55)
    ins.set_ylim(-40, 340)
    ins.set_xlabel("along-track [m]", fontsize=7, labelpad=1)
    ins.set_ylabel("cross-track [m]", fontsize=7, labelpad=1)
    ins.tick_params(labelsize=6.5, length=2)
    ins.set_title("formation, top view", fontsize=7.5, pad=2)
    beam_note = ax_top.text(0, 120, "", fontsize=8, ha="center", va="bottom", color="#9a6400")
    for sp in ("top", "right"):
        ax_top.spines[sp].set_visible(False)

    # channel panels: Re of each recorded channel up to the current platform position (aliased chirps)
    ch_lines = []
    lim = np.abs(r["ch"]).max() * 1.1
    for i, a in enumerate(ax_ch):
        a.set_ylim(-lim, lim)
        a.set_yticks([])
        a.set_ylabel(f"Rx{i}", fontsize=9, rotation=0, ha="right", va="center", color=C_RX[i])
        a.tick_params(labelsize=8, labelbottom=(i == 3))
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        ln, = a.plot([], [], color=C_RX[i], lw=0.35)
        ch_lines.append(ln)
    ax_ch[3].set_xlabel("position along the line [km]", fontsize=9)
    ax_ch[0].text(0.0, 1.12, "what each receiver records: Re s$_i$ at 500 Hz, one sample in four of the full PRF "
                  "(the chirps are aliased)", transform=ax_ch[0].transAxes, fontsize=8, color="0.35", va="bottom")
    ax_top.set_xlim(*xlim)

    # result panel
    ax_irf.set_ylim(-75, 5)
    ax_irf.set_yticks([0, -20, -40, -60])
    ax_irf.set_ylabel("[dB re. reference peak]", fontsize=9)
    ax_irf.set_xlabel("position along the line [km]", fontsize=9)
    ax_irf.tick_params(labelsize=8)
    ax_irf.grid(axis="y", color="0.9", lw=0.6)
    for sp in ("top", "right"):
        ax_irf.spines[sp].set_visible(False)
    lines = {
        "ref": ax_irf.plot(xkm, db(r["ref"], pk), color=C_REF, lw=0.7, zorder=4)[0],
        "std": ax_irf.plot(xkm, db(r["std"], pk), color=C_STD, lw=0.7, zorder=2)[0],
        "first": ax_irf.plot(xkm, db(r["first"], pk_first), color=C_FIRST, lw=0.7, zorder=3)[0],
        "joint": ax_irf.plot(xkm, db(r["joint"], pk), color=C_JNT, lw=0.7, zorder=3)[0],
    }
    for ln in lines.values():
        ln.set_visible(False)
    labels = {"ref": f"reference (one radar at the full PRF): highest ambiguity {amb['ref']:.1f} dB",
              "std": f"standard reconstruction: {amb['std']:.1f} dB",
              "first": f"first estimate A$^H$s (peak-normalised): {amb['first']:.1f} dB",
              "joint": f"joint reconstruction, DEM only: {amb['joint']:.1f} dB"}
    colors = {"ref": C_REF, "std": C_STD, "first": C_FIRST, "joint": C_JNT}
    leg = {k: ax_irf.text(0.01, 0.97 - 0.085 * j, labels[k], transform=ax_irf.transAxes, fontsize=8.6,
                          color=colors[k], va="top", visible=False)
           for j, k in enumerate(["ref", "std", "first", "joint"])}

    def set_acq(visible):
        for a in ax_ch:
            a.set_visible(visible)
        ax_irf.set_visible(not visible)

    def update(f):
        if f < n_acq:
            set_acq(True)
            xp = xpl[f]
            title.set_text("1 · Four receivers fly over the mountain and record the channels")
            half = r["Lsa"] / 2e3
            beam.set_xy([[xp, 2850], [xp - half, 0], [xp + half, 0]])
            foot.set_data([xp - half, xp + half], [-60, -60])
            beam_note.set_text("beam footprint L$_{sa}$ = λ r$_0$/L$_a$ ≈ 16 km")
            beam_note.set_position((min(max(xp, xlim[0] + 6), xlim[1] - 14), 2150))
            for i, s in enumerate(sats):
                s.set_data([xp - 0.18 * i], [2850])

            k = int(np.clip(np.searchsorted(r["t"] * vs / 1e3, xp), 0, cfg.Na - 1))
            tg.set_color([C_LIT if r["lit"][p, k] else C_TGT for p in range(len(XS))])
            tg.set_sizes([80 if r["lit"][p, k] else 42 for p in range(len(XS))])
            m = xch <= xp
            for i, ln in enumerate(ch_lines):
                ln.set_data(xch[m], r["ch"][i, m].real)
            return
        set_acq(False)
        beam.set_xy([[0, 0], [0, 0], [0, 0]])
        foot.set_data([], [])
        beam_note.set_text("")
        sat_lab.set_text("")
        for s in sats:
            s.set_data([], [])
        tg.set_color([C_TGT] * len(XS))
        tg.set_sizes([42] * len(XS))
        g = f - n_acq
        shown, acc = [], 0
        for key, n in stages:
            if g >= acc:
                shown.append(key)
            acc += n
        cur = shown[-1]
        for key, ln in lines.items():
            vis = key == "ref" or key == cur
            ln.set_visible(vis)
            leg[key].set_visible(key in shown)
            leg[key].set_alpha(1.0 if vis else 0.55)
        title.set_text({"ref": "2 · Reference: one radar at the full PRF",
                        "std": "3 · Standard reconstruction: the relief leaves ambiguities",
                        "first": "4 · Joint, first estimate A$^H$s: each channel back on the places, then summed",
                        "joint": "5 · Joint reconstruction after the iterations: at the level of the reference"}[cur])

    frames = n_acq + n_res
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {frames} frames, {frames / fps:.1f} s)")
    if mp4:
        mp = os.path.splitext(out)[0] + ".mp4"
        from matplotlib.animation import FFMpegWriter
        anim.save(mp, writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]), dpi=100)
        print(f"wrote {mp}")
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.normpath(os.path.join(SC, "..", "report", "figures", "readme_animation.gif")))
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--mp4", action="store_true")
    a = ap.parse_args()
    make(a.out, a.fps, a.mp4)
