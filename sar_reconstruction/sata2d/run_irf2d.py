# -*- coding: utf-8 -*-
"""
2-D impulse response of a focused point target -- a range x azimuth contour
plus the two 1-D cuts, in the style of Sakar's Fig. 2.9.

Two things make this figure work, and without either of them it does not:

1.  2-D MATCHED-FILTER FOCUSING.  The focused image is

        img = IFFT2( FFT2(x) * conj(FFT2(ref)) )

    with ``ref`` the ideal monostatic signal of the same target.  This is
    exact: the range cell migration is contained in ``ref``, so it is
    compensated without any approximate RCMC.  Focusing each range bin
    separately against the azimuth chirp of a single bin -- the obvious
    shortcut -- leaves the RCM (~41 m, several cells) uncorrected and
    produces skewed contours.

2.  RANGE RESOLUTION COMPARABLE TO THE AZIMUTH ONE.  The standard preset has
    ``res_rg = 30 m`` against ``La/2 = 6 m`` in azimuth: a 5:1 ratio, and the
    cross comes out flattened.  The default here is ``--res-rg 6``, which
    matches the two and reproduces the isotropic cross of the reference
    figure.  ``--res-rg 30`` shows the standard preset instead, with the axes
    scaling themselves accordingly.

Sanity check: for the monostatic reference this script returns
PSLR = -13.3 / -13.5 dB in azimuth and range -- the theoretical rectangular
window value is -13.26 dB -- and IRW(-3 dB) of about 4-5 m for 6 m resolution.

Run from ``sar_reconstruction/``::

    python -m sata2d.run_irf2d                      # monostatic reference
    python -m sata2d.run_irf2d --method sub         # SATA per sub-band
    python -m sata2d.run_irf2d --res-rg 30          # standard preset (anisotropic)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)

from .geometry import (make_params3d, build_tracks_3d, generate_reference_3d,
                       generate_channels_3d, CoeffTable3D)
from .reconstruction import (range_compress, build_delta_C0_map_3d,
                             reconstruct_subband_2d, scatterer_range,
                             range_bin_of, METHOD_KW)

C_LIGHT = 3.0e8
PLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

LABEL = {"mono": "monostatic reference", "no": "no SATA",
         "whole": "SATA whole band", "sub": "SATA per sub-band"}


# ---------------------------------------------------------------------------
def focus2d(x, ref):
    """2-D matched filter against the ideal monostatic signal.

    Returns the focused image with the target at the centre of the array
    (fftshift applied along both axes).  For ``x is ref`` this is the 2-D
    autocorrelation, i.e. the ideal system IRF.
    """
    R = np.fft.fft2(ref)
    return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(x) * np.conj(R)))


def zoom2d(img, ia, ir, na, nr, zpa, zpr):
    """Frequency-domain zero-padding zoom around ``(ia, ir)``.

    Interpolates between samples, which a smooth contour needs.  The
    ``(2*na, 2*nr)`` window has to be wide enough to contain the relevant
    sidelobes, otherwise FFT wraparound contaminates the edges.
    """
    w = img[ia - na:ia + na, ir - nr:ir + nr]
    W = np.fft.fftshift(np.fft.fft2(w))
    P = np.zeros((2 * na * zpa, 2 * nr * zpr), complex)
    P[na * zpa - na:na * zpa + na, nr * zpr - nr:nr * zpr + nr] = W
    # ifft2 divides by the LARGER (zero-padded) N; rescale to keep the
    # amplitude of the original image -- without this everything drops by
    # 20*log10(zpa*zpr) dB
    return np.fft.ifft2(np.fft.ifftshift(P)) * (zpa * zpr)


def cut_metrics(cut, axis):
    """(IRW at -3 dB, PSLR) of a 1-D cut, both from the interpolated grid."""
    a = np.abs(cut)
    a = a / a.max()
    i0 = int(np.argmax(a))
    half = np.where(a >= 10 ** (-3 / 20))[0]
    irw = (axis[half[-1]] - axis[half[0]]) if half.size > 1 else np.nan
    # exclude the main lobe: walk out to the first null on either side
    lo = hi = i0
    while lo > 0 and a[lo - 1] < a[lo]:
        lo -= 1
    while hi < len(a) - 1 and a[hi + 1] < a[hi]:
        hi += 1
    mask = np.ones_like(a, bool)
    mask[lo:hi + 1] = False
    pslr = 20 * np.log10(a[mask].max()) if mask.any() else np.nan
    return irw, pslr


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="mono", choices=("mono", "no", "whole", "sub"),
                    help="what to focus (default: mono)")
    ap.add_argument("--res-rg", type=float, default=6.0, dest="res_rg",
                    help="range resolution [m] (default 6, matches the azimuth one)")
    ap.add_argument("--swath", type=float, default=300.0,
                    help="swath width [m] -- drives Nr and the cost (default 300)")
    ap.add_argument("--azimuth", type=float, default=0.0, help="target azimuth [m]")
    ap.add_argument("--height", type=float, default=240.0, help="target height [m]")
    ap.add_argument("--nrx", type=int, default=4)
    ap.add_argument("--dxt", type=float, default=150.0)
    ap.add_argument("--sata-osf", type=int, default=4, dest="sata_osf")
    ap.add_argument("--span", type=float, default=None,
                    help="axis half-size [m] (default: 4.5 resolution cells)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    specs = ((args.azimuth, args.height),) if (args.azimuth, args.height) != (0.0, 0.0) else ()
    p = make_params3d(Nrx=args.nrx, dxt=args.dxt, specs=specs,
                      res_rg=args.res_rg, swath=args.swath)
    tr = build_tracks_3d(p)
    ptg = np.asarray(p.points[0], float)
    nb = range_bin_of(p, scatterer_range(p, ptg))

    rho_r = C_LIGHT / (2.0 * p.rsf)
    d_az = p.ve / p.prf
    res_rg = C_LIGHT / 2.0 / p.rbw
    res_az = p.La / 2.0
    rcm = (p.ve * p.int_time / 2.0) ** 2 / (2.0 * p.r0)

    print(p.summary())
    print(f"resolution : range {res_rg:.2f} m | azimuth {res_az:.2f} m")
    print(f"sampling   : range {rho_r:.2f} m | azimuth {d_az:.2f} m")
    print(f"RCM        : {rcm:.1f} m = {rcm / rho_r:.1f} range cells")
    print(f"target     : azimuth {args.azimuth:.0f} m, height {args.height:.0f} m "
          f"-> range bin {nb}/{p.Nr}\n")

    ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)

    imgs = {"mono": focus2d(ref, ref)}
    if args.method != "mono":
        ch = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
        tab = CoeffTable3D(p, tr, n_nodes=16)
        # the RCM has to fit inside the width of the dC0 map, otherwise the
        # correction does not follow the target along the aperture
        hw = int(np.ceil(rcm / (2.0 * rho_r))) + 1
        maps = [build_delta_C0_map_3d(p, tr, i, range_halfwidth=hw)
                for i in range(p.Nrx)]
        print(f"  (range_halfwidth = {hw} cells, to cover the RCM)")
        rec = reconstruct_subband_2d(p, tr, ch, tab, sata_osf=args.sata_osf,
                                     maps=maps, **METHOD_KW[args.method])
        imgs[args.method] = focus2d(rec, ref)

    key = args.method
    ia = int(np.argmax(np.abs(imgs["mono"]).max(axis=1)))
    ir = int(np.argmax(np.abs(imgs["mono"]).max(axis=0)))
    nrm = np.abs(imgs["mono"]).max()

    span = args.span if args.span is not None else 4.5 * max(res_az, res_rg)
    na = max(8, int(np.ceil(3.0 * span / d_az)))
    nr = max(8, int(np.ceil(3.0 * span / rho_r)))
    zpa = max(1, int(np.ceil(24.0 * d_az / span)))
    zpr = max(1, int(np.ceil(24.0 * rho_r / span)))

    z = {k: zoom2d(v, ia, ir, na, nr, zpa, zpr) for k, v in imgs.items()}
    x_az = (np.arange(2 * na * zpa) - na * zpa) * d_az / zpa
    x_rg = (np.arange(2 * nr * zpr) - nr * zpr) * rho_r / zpr
    ca, cr = len(x_az) // 2, len(x_rg) // 2

    def db(v):
        return 20.0 * np.log10(np.abs(v) / nrm + 1e-20)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(figsize=(12.5, 5.6), dpi=160)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    levels = [-30, -25, -20, -16, -13, -10, -6, -3]
    ax0.contour(x_rg, x_az, db(z[key]), levels=levels, cmap="Spectral_r",
                linewidths=1.1, vmin=-34, vmax=0)
    ax0.set_xlabel("Range [m]")
    ax0.set_ylabel("Azimuth [m]")
    ax0.set_xlim(-span, span)
    ax0.set_ylim(-span, span)
    ax0.set_aspect("equal")
    ax0.grid(alpha=.3)
    ax0.set_title(f"2-D IRF -- {LABEL[key]}")

    for ax, cut, xax, lab in ((ax1, z[key][:, cr], x_az, "Azimuth [m]"),
                              (ax2, z[key][ca, :], x_rg, "Range [m]")):
        ax.plot(xax, db(cut), "k", lw=1.1)
        if key != "mono":
            ref_cut = z["mono"][:, cr] if lab.startswith("Azimuth") else z["mono"][ca, :]
            ax.plot(xax, db(ref_cut), color="0.6", lw=0.9, ls="--", label="mono")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel(lab)
        ax.set_ylabel("Impulse Response [dB]")
        ax.set_xlim(-2 * span, 2 * span)
        ax.set_ylim(-70, 3)
        ax.grid(alpha=.3)

    fig.suptitle(f"$N_{{rx}}$={p.Nrx} | PRF={p.prf:.0f} Hz | "
                 f"$B_a$={p.abw:.0f} Hz | $B_r$={p.rbw / 1e6:.1f} MHz | "
                 f"$\\delta_r$={res_rg:.1f} m, $\\delta_x$={res_az:.1f} m | "
                 f"$\\delta h$={args.height:.0f} m, "
                 f"$b_{{xt}}^{{max}}$={np.max(np.abs(p.bxt)):.0f} m",
                 fontsize="medium")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.11)

    out = args.out or os.path.join(PLOTS, f"irf2d_{key}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    for lab, cut, xax in (("azimuth", z[key][:, cr], x_az),
                          ("range", z[key][ca, :], x_rg)):
        irw, pslr = cut_metrics(cut, xax)
        print(f"  {lab:>8}: IRW(-3dB) = {irw:6.2f} m | PSLR = {pslr:6.2f} dB")
    print(f"\nfigure written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
