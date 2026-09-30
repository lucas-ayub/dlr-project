"""
Geometry panels shared by the sweep figures: the scene of one case in 2D (height
profile) and in 3D (azimuth, ground-range offset, height), and the family overviews.
Coordinates: azimuth x [m] along the flight, ground-range offset dy [m] from the
reference point, height h [m] above the reference height h0 = 0.
"""
import numpy as np
import matplotlib.pyplot as plt

R0 = 299792458.0 * 0.0051115753 / 2.0      # slant range of the reference point [m]
H_SAT = 720e3                              # platform height [m]
Y0 = float(np.sqrt(R0 ** 2 - H_SAT ** 2))  # ground range of the reference point [m]


def iso_range_dy(h):
    """Ground-range offset that keeps a target of height h on the slant range r0."""
    return np.sqrt(R0 ** 2 - (H_SAT - np.asarray(h, float)) ** 2) - Y0


def scene_coordinates(xs, heights, offsets=None):
    """(x, dy, h) of the targets: iso-range targets spaced in azimuth, or explicit offsets."""
    if offsets is not None:
        o = np.asarray(offsets, float)
        return o[:, 0], o[:, 1], o[:, 2]
    xs, hs = np.asarray(xs, float), np.asarray(heights, float)
    return xs, iso_range_dy(hs), hs


def geometry_2d(ax, xs, dys, hs, topo, title=None, annotate=True):
    """Height profile: against azimuth (iso-range scenes) or against ground range (cross-track)."""
    u = dys if topo else xs
    step = np.min(np.diff(u)) if len(u) > 1 else 200.0
    lo, span = min(0.0, hs.min()), np.ptp(hs)
    ax.fill_between(u, lo - 0.05 * (span + 1), hs, color="#D7C9A7", alpha=0.45, lw=0)
    ax.plot(u, hs, "-", color="#6D4C2F", lw=1.6)
    ax.scatter(u, hs, s=55, color="#1F4E79", edgecolor="k", zorder=3, label="point targets")
    ax.axhline(0.0, color="k", ls="--", lw=0.9, label=r"reconstruction reference $h_0=0$")
    if annotate and len(u) <= 13:
        for ui, hi in zip(u, hs):
            ax.annotate(f"{hi:.0f} m", (ui, hi), textcoords="offset points", xytext=(7, -4),
                        fontsize=8, color="#1F4E79")
    pad = 0.25 * span if span > 0 else 10.0
    ax.set_ylim(lo - pad, hs.max() + pad)
    ax.set_xlim(min(0.0, u.min()) - 1.5 * step, u.max() + 1.5 * step)
    ax.set_xlabel("ground-range offset $\\Delta y$ [m] (all targets at azimuth 0)" if topo
                  else "azimuth position $x$ [m] (iso-range targets)")
    ax.set_ylabel("height above reference [m]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    if title:
        ax.set_title(title, fontsize=9.5)


def geometry_3d(ax, xs, dys, hs, topo, title=None):
    """The same targets in 3D: drop lines to the reference plane, a line through the targets."""
    xr = (min(xs.min(), -600.0) - 200.0, max(xs.max(), 600.0) + 200.0)
    yr = (min(dys.min(), 0.0) - 200.0, max(dys.max(), 400.0) + 200.0)
    X_, Y_ = np.meshgrid(xr, yr)
    ax.plot_surface(X_, Y_, np.zeros_like(X_), color="0.6", alpha=0.15, linewidth=0, shade=False)
    for x, y, h in zip(xs, dys, hs):
        ax.plot([x, x], [y, y], [0.0, h], color="0.55", lw=0.8)
    ax.plot(xs, dys, hs, "-", color="#6D4C2F", lw=1.4)
    ax.scatter(xs, dys, hs, s=28, color="#1F4E79", edgecolor="k", depthshade=False, zorder=5)
    ax.plot(xs, dys, np.zeros_like(hs), "o", ms=3, color="0.4")
    if len(xs) <= 9:
        for x, y, h in zip(xs, dys, hs):
            ax.text(x, y, h, f" {h:.0f} m", fontsize=6.5, color="#1F4E79")
    ax.set_xlabel("azimuth $x$ [m]", fontsize=8, labelpad=4)
    ax.set_ylabel("ground range $\\Delta y$ [m]", fontsize=8, labelpad=4)
    ax.set_zlabel("height [m]", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7, pad=0)
    zmax = max(hs.max(), 50.0)
    ax.set_zlim(min(0.0, hs.min()), zmax * 1.15)
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((1.6, 1.3, 0.8), zoom=1.3)
    ax.locator_params(axis="x", nbins=5); ax.locator_params(axis="y", nbins=5); ax.locator_params(axis="z", nbins=5)
    if title:
        ax.set_title(title, fontsize=9.5)


def coordinates_text(xs, dys, hs):
    """Compact listing of the target coordinates for a text box."""
    lines = [f"{'x [m]':>7s} {'dy [m]':>7s} {'h [m]':>6s}"]
    lines += [f"{x:7.0f} {y:7.0f} {h:6.0f}" for x, y, h in zip(xs, dys, hs)]
    return "\n".join(lines)


def family_3d(ax, cases, title, label_of, topo=False, ribbon=True):
    """Several cases of one family in 3D (one colour per case), for the overview figures.
    cases: list of (xs, dys, hs)."""
    cols = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, max(len(cases), 2)))
    xmin = min(c[0].min() for c in cases); xmax = max(c[0].max() for c in cases)
    ymin = min(c[1].min() for c in cases); ymax = max(c[1].max() for c in cases)
    for (xs, dys, hs), c, lab in zip(cases, cols, label_of):
        if ribbon and not topo:
            X_ = np.tile(xs, (2, 1)); YY = np.vstack([dys - 300.0, dys + 300.0])
            ax.plot_surface(X_, YY, np.tile(hs, (2, 1)), color=c, alpha=0.20, linewidth=0, shade=False)
        if topo:
            az = np.array([-800.0, 800.0])
            A_, Y_ = np.meshgrid(az, np.r_[0.0, dys])
            ax.plot_surface(A_, Y_, np.tile(np.r_[0.0, hs][:, None], (1, 2)), color=c, alpha=0.16,
                            linewidth=0, shade=False)
        ax.plot(xs, dys, hs, "-", color=c, lw=1.2)
        ax.scatter(xs, dys, hs, color=c, s=12, depthshade=False, label=lab)
    X_, Y_ = np.meshgrid([min(xmin, -800.0) - 200, max(xmax, 800.0) + 200],
                         [min(ymin, 0.0) - 300, ymax + 400])
    ax.plot_surface(X_, Y_, np.zeros_like(X_, dtype=float), color="0.6", alpha=0.10, linewidth=0, shade=False)
    ax.set_xlabel("azimuth [m]", fontsize=8, labelpad=3); ax.set_ylabel("ground range $\\Delta y$ [m]", fontsize=8, labelpad=3)
    ax.set_zlabel("height [m]", fontsize=8, labelpad=2); ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=20, azim=-60 if topo else -35)
    ax.set_box_aspect((1.5, 1.3, 0.8), zoom=1.35)
    ax.locator_params(axis="x", nbins=5); ax.locator_params(axis="y", nbins=5); ax.locator_params(axis="z", nbins=5)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", handlelength=1)
