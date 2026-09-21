# -*- coding: utf-8 -*-
"""
2-D impulse response of a focused point target -- the WHOLE image, no zoom.

Companion of ``run_irf2d.py``.  Same geometry, same reconstruction, same exact
2-D matched-filter focusing (both call ``run_irf2d.compute``), but instead of
zooming +-4.5 resolution cells around the target and interpolating, this
script plots the focused image on its raw sample grid over the whole scene:

  left         2-D magnitude of the whole image [dB], azimuth (km) x range (m)
  top right    azimuth cut over the full slow-time axis (time axis on top)
  bottom right range cut over the full swath

That is the view in which the residual azimuth ambiguities are visible: they
sit at k*PRF_op/Ka in slow time (about +-0.81 s, i.e. +-6 km, and its double),
far outside the zoom of run_irf2d.py.  Their levels are also printed.

What this view is NOT for: IRW and PSLR.  On the raw grid the main lobe is one
or two samples wide, so read those from run_irf2d.py (interpolated zoom).

Everything is normalised to the monostatic peak, so both the loss of peak and
the energy that went into the ambiguities show.

Run from ``sar_reconstruction/``::

    python -m sata2d.run_irf2d_full                  # monostatic reference
    python -m sata2d.run_irf2d_full --method sub     # SATA per sub-band
    python -m sata2d.run_irf2d_full --method no      # no SATA
    python -m sata2d.run_irf2d_full --method sub --floor -80

Accepts every option of run_irf2d.py (``--height``, ``--bxt-max``, ``--nrx``,
``--coreg``, ...) plus ``--floor``.  Writes ``plots/irf2d_<method>_full.png``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)

from .run_irf2d import make_parser, compute, LABEL, PLOTS          # noqa: E402


# ---------------------------------------------------------------------------
def ambiguities(p, cut_db, x_az, orders=(-2, -1, 1, 2), win=0.1):
    """Level of the residual azimuth ambiguities on an azimuth cut.

    Ambiguity ``k`` of the multichannel reconstruction sits at slow time
    ``t_k = k * PRF_op / Ka``, with ``Ka = 2 vs^2 / (wl r0)`` the Doppler rate
    of the (straight-line, speed ``vs``) tracks.  The strongest sample within
    ``+-win`` s of each ``t_k`` is returned as ``(k, t_k, x [m], level [dB])``.
    """
    ka = 2.0 * p.vs ** 2 / (p.wl * p.r0)
    t = x_az / p.ve
    out = []
    for k in orders:
        tk = k * p.PRF_op / ka
        m = np.abs(t - tk) < win
        if not m.any():
            continue
        i = np.flatnonzero(m)[int(np.argmax(np.asarray(cut_db)[m]))]
        out.append((k, tk, float(x_az[i]), float(cut_db[i])))
    return out


def plot_full(p, imgs, key, ia, ir, d_az, rho_r, nrm, height, floor, coreg, out):
    """The focused image with NO zoom and NO interpolation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    Na, Nr = imgs["mono"].shape
    x_az = (np.arange(Na) - ia) * d_az          # [m]
    x_rg = (np.arange(Nr) - ir) * rho_r         # [m]

    def db(v):
        return np.maximum(20.0 * np.log10(np.abs(v) / nrm + 1e-20), floor)

    fig = plt.figure(figsize=(13, 6.2), dpi=150)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.0, 1.35], hspace=0.45, wspace=0.22)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    im = ax0.imshow(db(imgs[key]), origin="lower", aspect="auto", cmap="magma",
                    vmin=floor, vmax=0,
                    extent=[x_rg[0], x_rg[-1], x_az[0] / 1e3, x_az[-1] / 1e3])
    fig.colorbar(im, ax=ax0, label="[dB]  (re. monostatic peak)")
    ax0.set_xlabel("Range [m]")
    ax0.set_ylabel("Azimuth [km]")
    ax0.set_title(f"2-D IRF, full image -- {LABEL[key]}"
                  + ("\n(explicit co-registration)" if coreg else ""))

    colour = {"no": "#C62828", "whole": "#E07B00", "sub": "#2E7D32", "mono": "k"}
    styles = [("mono", "0.35", "--", 0.8, "mono")]
    if key != "mono":
        styles.append((key, colour[key], "-", 0.7, LABEL[key]))
    else:
        styles = [("mono", "k", "-", 0.8, "mono")]
    for k, c, ls, lw, lab in styles:
        ax1.plot(x_az / 1e3, db(imgs[k][:, ir]), color=c, ls=ls, lw=lw, label=lab)
        ax2.plot(x_rg, db(imgs[k][ia, :]), color=c, ls=ls, lw=lw + 0.3, label=lab)
    ax1.set_xlabel("Azimuth [km]")
    ax1.set_title("azimuth cut, full slow-time axis (no interpolation)")
    sec = ax1.secondary_xaxis("top", functions=(lambda x: x * 1e3 / p.ve,
                                                 lambda t: t * p.ve / 1e3))
    sec.set_xlabel("azimuth time [s]")
    ax2.set_xlabel("Range [m]")
    ax2.set_title("range cut, full swath (no interpolation)")
    for ax in (ax1, ax2):
        ax.set_ylabel("Impulse Response [dB]")
        ax.set_ylim(floor, 3)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"$N_{{rx}}$={p.Nrx} | PRF={p.prf:.0f} Hz | $B_a$={p.abw:.0f} Hz | "
                 f"$B_r$={p.rbw / 1e6:.1f} MHz | "
                 f"$\\delta h$={height:.0f} m, "
                 f"$b_{{xt}}^{{max}}$={np.max(np.abs(p.bxt)):.0f} m | no zoom",
                 fontsize="medium", y=1.03)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.10)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return x_az


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = make_parser(description=__doc__)
    ap.add_argument("--floor", type=float, default=-100.0,
                    help="lower dB limit of the figure (default -100)")
    args = ap.parse_args(argv)

    r = compute(args)
    p, imgs, key = r["p"], r["imgs"], r["key"]

    out = args.out or os.path.join(
        PLOTS, f"irf2d_{key}{'_coreg' if args.coreg else ''}_full.png")
    x_az = plot_full(p, imgs, key, r["ia"], r["ir"], r["d_az"], r["rho_r"],
                     r["nrm"], args.height, args.floor, args.coreg, out)

    pk = 100.0 * np.abs(imgs[key]).max() / r["nrm"]
    cut = 20.0 * np.log10(np.abs(imgs[key][:, r["ir"]]) / r["nrm"] + 1e-20)
    print(f"\n  peak      : {pk:6.1f} % of monostatic")
    print(f"  azimuth ambiguities ({LABEL[key]}), expected at k*PRF_op/Ka:")
    for k, tk, x, lev in ambiguities(p, cut, x_az):
        print(f"      k = {k:+d}: t = {tk:+6.3f} s, x = {x / 1e3:+7.2f} km : "
              f"{lev:7.2f} dB")
    print(f"\nfigure written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())