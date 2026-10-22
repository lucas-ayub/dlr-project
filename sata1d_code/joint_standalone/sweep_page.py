"""One page per case for run_sweep.py: four panels as the sweep pages of the project (IRF with Reference, no SATA and the
joint; the scene in 2D with a zoom on the targets; the whole line in 3D; the channels seen from above), with the
parameters of the simulation above and below."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

CH = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive"]


def bxt_text(b):
    return "[" + ", ".join(f"{v:g}" for v in b) + "] m"


def iso_range_dy(cfg, h):
    """Ground-range offset that keeps a point of height h on the slant range r0 of the reference point."""
    return np.sqrt(cfg.scene.r0 ** 2 - (cfg.scene.H - np.asarray(h, float)) ** 2) - cfg.scene.y0


def ambiguity_level(cfg, env, pk, xs, orders=(-2, -1, 1, 2), win=0.02, guard=0.02):
    """Highest ambiguity relative to the reference peak [dB]: maximum of the envelope in windows of +-win s around
    x/v + m * dt_amb (m = +-1, +-2) of every target, outside +-guard s around every target's own main lobe."""
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    vs = cfg.system.vs
    dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * vs ** 2)
    lobe = np.zeros(cfg.Na, bool)
    sel = np.zeros(cfg.Na, bool)
    for x in xs:
        lobe |= np.abs(t - x / vs) <= guard
        for m in orders:
            sel |= np.abs(t - (x / vs + m * dt)) <= win
    sel &= ~lobe
    return float(20 * np.log10(env[sel].max() / pk + 1e-15))


def channel_geometry(ax, cfg):
    """Transmitter and receivers seen from above, in the frame of the transmitter."""
    bat, bxt, n = cfg.array.bat, cfg.array.bxt, cfg.Nrx
    xr, yr = -bat, bxt
    ylo, yhi = min(0.0, yr.min()), max(0.0, yr.max())
    sy, sx = max(yhi - ylo, 1.0), max(np.ptp(np.r_[0.0, xr]), 1.0)
    for i in range(n):
        ax.plot([0.0, xr[i]], [0.0, yr[i]], color=CH[i], lw=1.0, ls=":", zorder=1)
        ax.plot(xr[i], yr[i], "o", ms=13, color=CH[i], mec="k", zorder=4,
                label=f"Rx {i}: $b_{{at}}$ = {bat[i]:.2f} m, $b_{{xt}}$ = {bxt[i]:g} m")
        at_tx = np.hypot(xr[i], yr[i]) < 0.02 * max(sx, sy)
        ax.annotate(f"Rx {i}", (xr[i], yr[i]), xytext=(-12, 10) if at_tx else (9, 5), textcoords="offset points",
                    ha="right" if at_tx else "left", color=CH[i], fontsize=10, fontweight="bold")
    ax.plot(0.0, 0.0, marker="*", ms=17, color="gold", mec="k", ls="none", zorder=5, label="Tx")
    ax.annotate("Tx", (0.0, 0.0), xytext=(9, -14), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.annotate("", xy=(0.10 * sx, yhi + 0.13 * sy), xytext=(-0.28 * sx, yhi + 0.13 * sy),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.4))
    ax.text(-0.29 * sx, yhi + 0.13 * sy, "flight direction ", ha="right", va="center", fontsize=9, color="0.25")
    ax.annotate("", xy=(0.12 * sx, ylo + 0.95 * sy), xytext=(0.12 * sx, ylo + 0.65 * sy),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.4))
    ax.text(0.135 * sx, ylo + 0.80 * sy, "towards\nthe scene", fontsize=8.5, color="0.25", va="center")
    ax.set_xlim(-1.5 * sx, 0.62 * sx); ax.set_ylim(ylo - 0.15 * sy, yhi + 0.22 * sy)
    ax.grid(alpha=0.3)
    ax.set_xlabel("along-track position relative to the Tx [m]")
    ax.set_ylabel("cross-track position (horizontal) [m]")
    ax.set_title("Channels seen from above (all at the same altitude; axes not to scale)", fontsize=10)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    covered = np.any((xr < x0 + 0.5 * (x1 - x0)) & (yr > y1 - (0.14 + 0.07 * n) * (y1 - y0)))
    ax.legend(fontsize=7.6, loc="lower left" if covered else "upper left",
              bbox_to_anchor=None if covered else (0.0, 0.92), framealpha=0.92)


def page(out_png, cfg, terrain, xs, label, kind, env, amb, info):
    """env: focused envelopes {'Reference', 'no SATA', 'joint final version'}; amb: highest ambiguity of each [dB]."""
    pk = env["Reference"].max()
    db = {k: 20 * np.log10(e / pk + 1e-12) for k, e in env.items()}
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    hs = terrain(xs)
    dys = iso_range_dy(cfg, hs)
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    lo2, hi2 = min(xs.min(), 0.0), max(xs.max(), 0.0)
    pad2 = max(0.12 * (hi2 - lo2), 500.0)
    xz = np.linspace(lo2 - pad2, hi2 + pad2, 2001)
    hz = terrain(xz)
    dh = float(np.ptp(hz))
    slope = float(np.degrees(np.arctan(np.abs(np.gradient(hz, xz)).max())))

    fig = plt.figure(figsize=(22, 10.6))
    gs = GridSpec(2, 3, width_ratios=[1.32, 0.86, 1.0], hspace=0.38, wspace=0.2, figure=fig,
                  left=0.04, right=0.985, top=0.86, bottom=0.1)
    head = f"{label} | $N_{{rx}}$ = {cfg.Nrx}, $b_{{xt}}$ = {bxt_text(cfg.array.bxt)} ({kind})"
    tail = "the joint reads only the DEM (h(x) at every position)"
    if len(head) + len(tail) > 185:
        fig.suptitle(head + "\n" + tail, fontsize=12.5, y=0.985)
    else:
        fig.suptitle(head + " | " + tail, fontsize=13, y=0.975)

    # left: IRF, top = Reference + joint, bottom = Reference + no SATA
    ax = fig.add_subplot(gs[0, 0]); az = fig.add_subplot(gs[1, 0])
    ax.plot(t, db["Reference"], color="tab:green", lw=2.2, label="Reference")
    ax.plot(t, db["joint final version"], color="tab:purple", lw=1.3, label="joint final version")
    ax.set_title(f"highest ambiguity: reference {amb['Reference']:.1f} dB | joint {amb['joint final version']:.1f} dB", fontsize=10)
    az.plot(t, db["Reference"], color="tab:green", lw=2.2, label="Reference")
    az.plot(t, db["no SATA"], color="tab:red", lw=1.6, ls=(0, (5, 3)), label="no SATA")
    az.set_title(f"highest ambiguity: reference {amb['Reference']:.1f} dB | no SATA {amb['no SATA']:.1f} dB | "
                 f"improvement of the joint {amb['no SATA'] - amb['joint final version']:.1f} dB", fontsize=10)
    for a_ in (ax, az):
        a_.set_xlim(t[0], t[-1]); a_.set_ylim(-90, 5); a_.grid(alpha=0.3)
        a_.set_xlabel("Azimuth time [s]"); a_.set_ylabel("[dB]"); a_.legend(fontsize=9, loc="upper left")

    g2 = fig.add_subplot(gs[0, 1])
    xg = np.linspace(-half, half, 4801); hg = terrain(xg)
    g2.fill_between(xg, min(hg.min(), 0) - 0.05 * max(np.ptp(hg), 1.0), hg, color="#D7C9A7", alpha=0.45, lw=0)
    g2.plot(xg, hg, color="#6D4C2F", lw=1.2, zorder=2, label="terrain (the scene)")
    g2.scatter(xs, hs, s=55 if len(xs) <= 13 else (25 if len(xs) <= 49 else 12), color="#1F4E79", edgecolor="k", zorder=3, label="point targets")
    if len(xs) <= 13:
        for x_, h_ in zip(xs, hs):
            g2.annotate(f"{h_:.0f} m", (x_, h_), textcoords="offset points", xytext=(7, -4), fontsize=8, color="#1F4E79")
    g2.axhline(0.0, color="k", ls="--", lw=0.9, label=r"reconstruction reference $h_0=0$")
    g2.plot(0.0, 0.0, marker="*", ms=13, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $h$ = 0)")
    ylo, yhi = min(hz.min(), 0.0), max(hz.max(), hs.max(), 1.0)
    g2.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo)); g2.set_xlim(lo2 - pad2, hi2 + pad2)
    g2.grid(alpha=0.3); g2.set_ylabel("height above reference [m]")
    g2.xaxis.set_major_formatter(km); g2.set_xlabel("azimuth position $x$ [km]")
    g2.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.1)
    g2.set_title(f"Scene: zoom on the targets | $\\Delta h_{{max}}$ = {dh:.0f} m | max slope {slope:.0f}$^\\circ$", fontsize=10, pad=34)

    g3 = fig.add_subplot(gs[0, 2], projection="3d")
    yr = (min(dys.min(), iso_range_dy(cfg, hg).min(), 0.0) - 200.0, max(dys.max(), 400.0) + 200.0)
    X_, Y_ = np.meshgrid((-half, half), yr)
    g3.plot_surface(X_, Y_, np.zeros_like(X_), color="0.6", alpha=0.15, linewidth=0, shade=False)
    for x_, y_, h_ in zip(xs, dys, hs):
        g3.plot([x_, x_], [y_, y_], [0.0, h_], color="0.55", lw=0.8)
    g3.scatter(xs, dys, hs, s=28 if len(xs) <= 13 else 12, color="#1F4E79", edgecolor="k", depthshade=False, zorder=5)
    g3.plot(xs, dys, np.zeros_like(hs), "o", ms=3, color="0.4")
    g3.plot(xg, iso_range_dy(cfg, hg), hg, color="#6D4C2F", lw=1.0)
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=6)
    g3.set_xlim(-half, half); g3.xaxis.set_major_formatter(km)
    g3.set_zlim(min(0.0, hg.min()), max(hg.max(), 50.0) * 1.15)
    g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4)
    g3.set_ylabel("ground range $\\Delta y$ [m]", fontsize=8, labelpad=4)
    g3.set_zlabel("height [m]", fontsize=8, labelpad=2)
    g3.tick_params(labelsize=7, pad=0); g3.view_init(elev=22, azim=-55)
    g3.set_box_aspect((1.6, 1.3, 0.8), zoom=1.2)
    g3.set_title(f"Scene: whole azimuth line ({2 * half / 1e3:.0f} km) in 3D (x, $\\Delta y$, h)", fontsize=10)

    channel_geometry(fig.add_subplot(gs[1, 1:]), cfg)

    fig.text(0.5, 0.012, "Highest ambiguity in dB relative to the reference peak, inside +-20 ms of the ambiguity positions of every "
             "target (1 and 2 times the ambiguity distance of the channel PRF, on both sides). The joint uses the DEM only; it never "
             f"uses the positions of the targets.\n$N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz, PRF$_{{op}}$ = {cfg.PRF_op:.0f} Hz, "
             f"$B_a$ = {cfg.abw:.0f} Hz, channel 0 is the transmitter ($b_{{at}}$ = $b_{{xt}}$ = 0), $b_{{xt}}$ = {bxt_text(cfg.array.bxt)}; "
             f"joint: {info['iterations']} CGLS iterations, unexplained part of the channels {info['unexplained_dB']:.1f} dB.",
             ha="center", fontsize=8.5, color="0.35")
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
