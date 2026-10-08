"""
3-D animation of an acquisition: transmitter and Nrx receivers flying over a topography,
the antenna footprint sweeping the ground, the highlighted targets lit while they are inside
the beam (lines of sight to TX and every RX), the array formation in true metres, and the
bistatic range histories of the highlighted targets with a moving time cursor.

The geometry is the simulation's: TX at (vs t, 0, H), RX i at (vs t - bat[i], bxt[i], H), the
scene around the reference point (x0, y0, 0), beam width theta = wl / La, broadside. In the 3-D
view the orbit (H ~ 720 km, slant range ~ 766 km) and the baselines (metres) cannot be drawn to
scale together with a scene of kilometres, so the platform is drawn at a reduced height with the
same look angle and the baselines are magnified; the formation panel and the range histories use
the true geometry.

    from sata1d.scene3d import animate_scene
    animate_scene(bat=[0, 0, 0, 0], bxt=[63.7, 27.0, 4.1, 1.7],
                  terrain=lambda x, y: 900 * np.exp(-0.5 * ((x / 1800) ** 2 + (y / 1500) ** 2)),
                  targets=[(-3000, -1500), (0, 0), (2500, 500)], out="scene.gif")
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .config import make_config, ArrayGeometry
from .geometry import build_platform_tracks

RX_COLORS = plt.cm.tab10(np.arange(10))


def scene_config(bat, bxt, prf=2000.0):
    """Simulation configuration with the given baselines (Nrx = len(bat))."""
    bat, bxt = np.asarray(bat, float), np.asarray(bxt, float)
    if bat.shape != bxt.shape:
        raise ValueError("bat and bxt must have the same length (one value per channel)")
    cfg = make_config([], nrx=len(bat), prf=prf, bxt_max=0.0)
    return replace(cfg, array=ArrayGeometry(bat=bat, bxt=bxt))


def target_positions(cfg, terrain, targets):
    """Absolute positions [N, 3] of the highlighted targets: (x, y) relative to the reference
    point with h = terrain(x, y), or (x, y, h) given explicitly."""
    out = []
    for t in targets:
        x, y = float(t[0]), float(t[1])
        h = float(t[2]) if len(t) > 2 else float(terrain(x, y))
        out.append(cfg.scene.ptg + np.array([x, y, h]))
    return np.array(out)


def animate_scene(bat, bxt, terrain, targets, out="scene3d.gif", prf=2000.0, extent=(6000.0, 4000.0),
                  frames=60, fps=4, t_span=None, h_vis=6000.0, magnify=None, channel_ranges=None, range_mode="min",
                  elev=24, azim_start=-70, azim_sweep=40, title=None):
    """
    bat, bxt  : along-track / cross-track baselines [m], one per channel (Nrx = len(bat))
    terrain   : h = terrain(x, y) [m], x azimuth and y ground range relative to the reference point
    targets   : highlighted targets, (x, y) on the terrain or (x, y, h)
    extent    : half-sizes (x, y) of the terrain drawn [m]
    t_span    : (t0, t1) [s] of the animation (default: until the footprint has crossed the scene)
    h_vis     : height of the platform in the 3-D view [m] (the look angle is kept)
    magnify   : factor applied to the baselines in the 3-D view (default: largest baseline -> 600 m)
    channel_ranges : channels whose bistatic range histories are plotted (default: all)
    range_mode: "min"    each history minus its own minimum [m] (the parabolas, all from 0)
                "offset" R_T + R_R - 2 r0 [km] (the targets' ranges side by side)
    Returns the path of the GIF.
    """
    cfg = scene_config(bat, bxt, prf)
    tr = build_platform_tracks(cfg)
    Nrx, vs, H, y0, r0 = cfg.Nrx, cfg.system.vs, cfg.scene.H, cfg.scene.y0, cfg.scene.r0
    x0 = cfg.scene.ptg[0]
    theta = cfg.theta_tx
    foot = r0 * theta                                         # azimuth footprint [m]
    P = target_positions(cfg, terrain, targets)
    rel = P - cfg.scene.ptg                                   # relative coordinates for drawing
    ex, ey = extent
    if t_span is None:
        t_span = ((-ex - 0.5 * foot) / vs, (ex + 0.5 * foot) / vs)
    tf = np.linspace(t_span[0], t_span[1], frames)
    channel_ranges = list(range(Nrx)) if channel_ranges is None else list(channel_ranges)

    # visual geometry: platform at h_vis with the true look angle, baselines magnified
    look = np.arctan2(y0, H)
    d_vis = h_vis * np.tan(look)
    bmax = max(np.max(np.abs(cfg.array.bat)), np.max(np.abs(cfg.array.bxt)), 1e-9)
    mag = (600.0 / bmax) if magnify is None else magnify

    def tx_vis(t):
        return np.array([vs * t, -d_vis, h_vis])

    def rx_vis(t, i):
        return np.array([vs * t - mag * cfg.array.bat[i], -d_vis + mag * cfg.array.bxt[i] * np.cos(look),
                         h_vis + mag * cfg.array.bxt[i] * np.sin(look) * 0.0])

    def lit(t, p_abs):
        """inside the two-way beam of TX (broadside) at time t"""
        dx = p_abs[0] - (vs * t)
        return abs(np.arctan2(dx, np.hypot(p_abs[1], H - p_abs[2]))) <= theta / 2

    # bistatic range histories of the highlighted targets (true geometry)
    ta = cfg.ta
    sel = (ta >= t_span[0]) & (ta <= t_span[1])
    hist = {}
    for i in channel_ranges:
        hist[i] = [np.linalg.norm(tr.ptx[sel] - p, axis=1) + np.linalg.norm(tr.prx[i][sel] - p, axis=1) for p in P]
    TC = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
    tcols = [TC[j % len(TC)] for j in range(len(P))]

    xg, yg = np.meshgrid(np.linspace(-ex, ex, 61), np.linspace(-ey, ey, 41))
    zg = terrain(xg, yg)

    fig = plt.figure(figsize=(15, 9.0))
    nch = len(channel_ranges)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.22)
    gr = gs[0, 1].subgridspec(1 + nch, 1, height_ratios=[1.15] + [1.0] * nch, hspace=0.75)
    ax = fig.add_subplot(gs[0, 0], projection="3d", computed_zorder=False)
    af = fig.add_subplot(gr[0])
    ars = [fig.add_subplot(gr[1 + k]) for k in range(nch)]
    for a_ in ars[1:]:
        a_.sharex(ars[0])
    fig.suptitle(title or f"Acquisition geometry: {Nrx} receivers, b_at = [" + ", ".join(f"{b:.1f}" for b in cfg.array.bat)
                 + "] m, b_xt = [" + ", ".join(f"{b:.1f}" for b in cfg.array.bxt) + "] m", fontsize=11.5)

    # formation panel (true metres, static)
    af.plot(0, 0, "s", ms=11, color="k", label="TX")
    for i in range(Nrx):
        af.plot(-cfg.array.bat[i], cfg.array.bxt[i], "o", ms=9, color=RX_COLORS[i], mec="k", label=f"RX{i}")
        af.annotate(f"RX{i}", (-cfg.array.bat[i], cfg.array.bxt[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    af.set_xlabel("along-track offset to TX [m] (flight ->)", fontsize=7.5); af.set_ylabel("b_xt [m]", fontsize=7.5); af.tick_params(labelsize=7)
    af.set_title("Formation (true metres)", fontsize=8.5); af.grid(alpha=0.3)
    pad = 0.15 * bmax + 1.0
    af.set_xlim(min(-cfg.array.bat.max(), 0) - pad, max(-cfg.array.bat.min(), 0) + pad)
    af.set_ylim(min(cfg.array.bxt.min(), 0) - pad, max(cfg.array.bxt.max(), 0) + pad)
    af.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    # range histories (true geometry, km re. r0 pair)
    cursors = []
    for k, (i, ar) in enumerate(zip(channel_ranges, ars)):
        for j in range(len(P)):
            yv = (hist[i][j] - hist[i][j].min()) if range_mode == "min" else (hist[i][j] - 2 * r0) / 1e3
            ar.plot(ta[sel], yv, color=tcols[j], lw=1.4, label=f"T{j + 1}")
            m = np.argmin(hist[i][j])
            ar.plot(ta[sel][m], yv[m], "v", ms=5, color=tcols[j], mec="k", mew=0.5)
        off = np.mean([np.linalg.norm(tr.prx[i][sel] - p, axis=1)[np.argmin(h)] - np.linalg.norm(tr.ptx[sel] - p, axis=1)[np.argmin(h)]
                       for p, h in zip(P, hist[i])])
        cursors.append(ar.axvline(tf[0], color="k", lw=1.0))
        ar.set_title(f"channel {i}: b_at = {cfg.array.bat[i]:.1f} m, b_xt = {cfg.array.bxt[i]:.1f} m, "
                     f"$R_R - R_T$ at the minimum = {off:.2f} m", fontsize=8)
        ar.set_ylabel("$R_T+R_R-\\min$\n[m]" if range_mode == "min" else "$R_T+R_R-2r_0$\n[km]", fontsize=7.5); ar.tick_params(labelsize=7); ar.grid(alpha=0.3)
        if k < nch - 1:
            ar.tick_params(labelbottom=False)
    ars[-1].set_xlabel("azimuth time [s]", fontsize=8.5)
    ars[-1].legend(fontsize=7, ncol=min(len(P), 8), loc="upper center", bbox_to_anchor=(0.5, -0.55), frameon=False,
                   title="highlighted targets (\u25bc = minimum of each curve)", title_fontsize=7)

    def draw(n):
        t = tf[n]
        ax.cla()
        ax.plot_surface(xg / 1e3, yg / 1e3, zg / 1e3, cmap="terrain", alpha=0.45, linewidth=0, antialiased=False, zorder=1)
        # footprint on the ground
        fx = np.clip([vs * t - x0 - foot / 2, vs * t - x0 + foot / 2], -ex, ex)
        if fx[1] > fx[0]:
            xx, yy = np.meshgrid(np.linspace(fx[0], fx[1], 2), np.linspace(-ey, ey, 2))
            ax.plot_surface(xx / 1e3, yy / 1e3, np.full_like(xx, zg.max() + 20) / 1e3, color="gold", alpha=0.18, linewidth=0, zorder=2)
        # tracks
        xs = vs * tf - x0
        ax.plot(xs / 1e3, np.full_like(xs, -d_vis) / 1e3, np.full_like(xs, h_vis) / 1e3, color="0.5", lw=0.8, ls=":")
        txv = tx_vis(t) - np.array([x0, 0, 0])
        ax.scatter(*(txv / 1e3), s=80, marker="s", color="k", depthshade=False, zorder=12)
        rxv = [rx_vis(t, i) - np.array([x0, 0, 0]) for i in range(Nrx)]
        for i, r in enumerate(rxv):
            ax.scatter(*(r / 1e3), s=45, color=RX_COLORS[i], edgecolor="k", depthshade=False, zorder=12)
        # targets and lines of sight
        for j, (p, q) in enumerate(zip(P, rel)):
            on = lit(t, p)
            ax.scatter(q[0] / 1e3, q[1] / 1e3, (q[2] + 40) / 1e3, s=110 if on else 45, color=tcols[j] if on else "0.45",
                       edgecolor="k", depthshade=False, zorder=10)
            ax.text(q[0] / 1e3, q[1] / 1e3, q[2] / 1e3 + 0.3, f"T{j + 1}", fontsize=9, color=tcols[j] if on else "0.4", zorder=13, weight="bold")
            if on:
                ax.plot([txv[0] / 1e3, q[0] / 1e3], [txv[1] / 1e3, q[1] / 1e3], [txv[2] / 1e3, q[2] / 1e3], color="k", lw=0.9, zorder=8)
                for i, r in enumerate(rxv):
                    ax.plot([r[0] / 1e3, q[0] / 1e3], [r[1] / 1e3, q[1] / 1e3], [r[2] / 1e3, q[2] / 1e3],
                            color=RX_COLORS[i], lw=0.7, alpha=0.8, zorder=8)
        xr = max(ex, np.max(np.abs(vs * tf - x0))) / 1e3 + 1
        ax.set_xlim(-xr, xr); ax.set_ylim(-d_vis / 1e3 - 0.5, ey / 1e3)
        ax.set_zlim(0, h_vis / 1e3 + 1)
        ax.set_xlabel("azimuth x [km]"); ax.set_ylabel("ground range [km]"); ax.set_zlabel("height [km]")
        ax.view_init(elev=elev, azim=azim_start + azim_sweep * n / max(frames - 1, 1))
        ax.set_title(f"t = {t:+.2f} s  |  footprint {foot / 1e3:.1f} km  |  platform drawn at {h_vis / 1e3:.0f} km "
                     f"(true {H / 1e3:.0f} km), baselines x{mag:.0f}", fontsize=9.5)
        for c_ in cursors:
            c_.set_xdata([t, t])
        return []

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return out
