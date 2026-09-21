# -*- coding: utf-8 -*-
"""
How accurate does the DEM have to be for SATA to work?

SATA corrects ``dC0 = C0(target) - C0(flat point at the same range)``, and that
needs the target HEIGHT.  In the simulation the height is exact -- an oracle.
A real system reads it from an external DEM, which is wrong by some eps.  This
sweeps eps and measures the cost.

THE SPLIT THAT MAKES THIS AN EXPERIMENT
---------------------------------------
The data are generated ONCE from the true parameters: the scene does not move.
Only the SATA CORRECTION is built at ``h + eps`` -- the per-channel ``dC0``
maps.  So the target sits where it always sat and the processor believes
something else, which is what a DEM error is.  Building the data at ``h + eps``
instead would move the target and measure nothing.

The reconstruction filter itself stays FLAT-EARTH (``CoeffTable3D`` with no
``height``), exactly as in run_irf2d.py.  Handing the table the DEM height
would make it the oracle of run_c1c2.py -- a filter that already models the
topography -- and SATA on top of it would then remove the residual a second
time.  That double correction is what an earlier version of this script did:
at eps = 0 it returned ~47 % of the monostatic peak instead of ~100 %.

WHAT COMES OUT
--------------
  peak    focused peak, per cent of the monostatic reference
  IRW     -3 dB width of the azimuth cut [m]
  PSLR    worst azimuth sidelobe [dB]
  Sigma   channel-to-channel SPREAD, in cycles, of the residual SATA fails to
          remove, ``C0_i(h_true) - C0_i(h_true + eps)``

Sigma is the predictor: a residual common to every channel is a global phase
and costs nothing, so only its spread across channels degrades the inversion.
The third panel asks whether peak(eps) collapses onto Sigma(eps).

Focusing and metrics follow run_irf2d.py exactly, so the numbers are
comparable with it.  Sanity check: at eps = 0 the monostatic azimuth PSLR is
about -13.3 dB.

Run from ``sar_reconstruction/``::

    python -m sata2d.run_dem_error --eps 0
    python -m sata2d.run_dem_error --eps 0 2 5 10 20 50 --bxt-max 225
    python -m sata2d.run_dem_error --symmetric --coreg
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

from .arrays import make_params_dpca, dpca_residual
from .geometry import (build_tracks_3d, generate_reference_3d,
                       generate_channels_3d, CoeffTable3D, _coeff)
from .reconstruction import (range_compress, build_delta_C0_map_3d,
                             reconstruct_subband_2d, scatterer_range,
                             range_bin_of, METHOD_KW)

C_LIGHT = 3.0e8
PLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


# ---------------------------------------------------------------------------
# shared with run_irf2d.py
# ---------------------------------------------------------------------------
def focus2d(x, ref):
    """2-D matched filter against the ideal monostatic signal."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(x) * np.conj(np.fft.fft2(ref))))


def zoom2d(img, ia, ir, na, nr, zpa, zpr):
    """Frequency-domain zero-padding zoom around ``(ia, ir)``."""
    w = img[ia - na:ia + na, ir - nr:ir + nr]
    W = np.fft.fftshift(np.fft.fft2(w))
    P = np.zeros((2 * na * zpa, 2 * nr * zpr), complex)
    P[na * zpa - na:na * zpa + na, nr * zpr - nr:nr * zpr + nr] = W
    return np.fft.ifft2(np.fft.ifftshift(P)) * (zpa * zpr)


def cut_metrics(cut, axis):
    """(IRW at -3 dB, PSLR) of a 1-D cut, from the interpolated grid."""
    a = np.abs(cut)
    a = a / a.max()
    i0 = int(np.argmax(a))
    thr = 10 ** (-3 / 20)
    half = np.where(a >= thr)[0]
    if half.size > 1:
        lo_i, hi_i = int(half[0]), int(half[-1])

        def cross(i, j):
            if a[j] == a[i]:
                return axis[i]
            t = (thr - a[i]) / (a[j] - a[i])
            return axis[i] + t * (axis[j] - axis[i])

        left = cross(lo_i - 1, lo_i) if lo_i > 0 else axis[lo_i]
        right = cross(hi_i + 1, hi_i) if hi_i < len(a) - 1 else axis[hi_i]
        irw = float(right - left)
    else:
        irw = np.nan
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
# the DEM
# ---------------------------------------------------------------------------
def params_with_height(mk: dict, azimuth: float, h: float):
    """Parameters identical to the run's, but with the target at height ``h``.

    ``Params3D`` stores no ``specs`` or ``points`` field -- ``points`` is a
    read-only property built by the constructor -- so neither assignment nor
    ``dataclasses.replace`` works.  Calling the constructor again with the same
    keywords is the only supported route, and it is exact by construction.

    Used ONLY to build the dC0 maps, never to generate data.
    """
    return make_params_dpca(specs=((azimuth, h),), **mk)


def sigma_cycles(p, tracks, ptg, h_dem: float) -> float:
    """Channel-to-channel spread, in cycles, of what SATA fails to remove.

    SATA subtracts ``C0_i`` evaluated at the DEM height; the truth is at the
    real height.  Only the spread of the difference across channels matters.
    """
    pt_true = np.asarray(ptg, float)
    pt_dem = pt_true.copy()
    pt_dem[2] = h_dem
    left = np.array([_coeff(p, tracks, pt_true, i)[0]
                     - _coeff(p, tracks, pt_dem, i)[0]
                     for i in range(p.Nrx)], dtype=float)
    return float(np.ptp(left) / p.wl)


def reconstruct_at(p, tr, ch, eps: float, hw: int, args, mk: dict):
    """Reconstruct with the SATA correction built at ``h_true + eps``.

    ``p`` -- the TRUE parameters -- drives the inversion geometry, and the
    coefficient table is the ordinary FLAT-EARTH one.  Only the dC0 maps -- the
    thing SATA subtracts -- are built from the DEM height.

    Do NOT pass ``height=h_dem`` to ``CoeffTable3D``: that turns the filter into
    the oracle of run_c1c2.py, which already absorbs the topography, and SATA
    would then correct it twice (peak ~47 % at eps = 0 instead of ~100 %).
    """
    h_dem = args.height + eps
    p_dem = params_with_height(mk, args.azimuth, h_dem)
    tab = CoeffTable3D(p, tr, n_nodes=16)          # flat earth, as in run_irf2d
    maps = [build_delta_C0_map_3d(p_dem, tr, i, range_halfwidth=hw)
            for i in range(p.Nrx)]
    if args.coreg:
        from .coreg import reconstruct_explicit_coreg
        return reconstruct_explicit_coreg(p, tr, ch, tab, sata_osf=args.sata_osf,
                                          maps=maps, verbose=False,
                                          **METHOD_KW["sub"])
    return reconstruct_subband_2d(p, tr, ch, tab, sata_osf=args.sata_osf,
                                  maps=maps, verbose=False, **METHOD_KW["sub"])


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0, 2, 5, 10, 20, 50, 100],
                    help="DEM errors [m]")
    ap.add_argument("--symmetric", action="store_true",
                    help="also sweep -eps, to check the response is even")
    ap.add_argument("--res-rg", type=float, default=6.0, dest="res_rg")
    ap.add_argument("--swath", type=float, default=300.0)
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=240.0,
                    help="TRUE target height [m]")
    ap.add_argument("--nrx", type=int, default=4)
    ap.add_argument("--dxt", type=float, default=150.0)
    ap.add_argument("--bxt-mode", default="random", dest="bxt_mode",
                    choices=("linear", "random"))
    ap.add_argument("--bxt-max", type=float, default=100.0, dest="bxt_max")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sata-osf", type=int, default=4, dest="sata_osf")
    ap.add_argument("--coreg", action="store_true",
                    help="use the explicit co-registration pipeline of coreg.py")
    ap.add_argument("--span", type=float, default=None)
    ap.add_argument("--floor", type=float, default=-45.0,
                    help="lower limit of the IRF dB axis")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    # every keyword except ``specs``, so the DEM copy can be rebuilt identically
    mk = dict(Nrx=args.nrx, dxt=args.dxt, res_rg=args.res_rg, swath=args.swath,
              bxt_mode=args.bxt_mode, bxt_max=args.bxt_max, seed=args.seed)
    p = make_params_dpca(specs=((args.azimuth, args.height),), **mk)
    tr = build_tracks_3d(p)
    ptg = np.asarray(p.points[0], float)
    nb = range_bin_of(p, scatterer_range(p, ptg))

    rho_r = C_LIGHT / (2.0 * p.rsf)
    d_az = p.ve / p.prf
    res_rg = C_LIGHT / 2.0 / p.rbw
    res_az = p.La / 2.0
    rcm = (p.ve * p.int_time / 2.0) ** 2 / (2.0 * p.r0)
    hw = int(np.ceil(rcm / (2.0 * rho_r))) + 1

    print(p.summary())
    print(f"array      : DPCA (residual {dpca_residual(p):.1e}), "
          f"bxt = {np.array2string(p.bxt, precision=1)} m")
    print(f"resolution : range {res_rg:.2f} m | azimuth {res_az:.2f} m")
    print(f"target     : height {args.height:.0f} m -> range bin {nb}/{p.Nr}")
    print(f"dC0 map    : range_halfwidth {hw} cells (RCM {rcm:.1f} m)\n")

    # ---- data, once: the scene never moves -------------------------------
    ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)
    ch = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
    img_mono = focus2d(ref, ref)
    nrm = np.abs(img_mono).max()

    ia = int(np.argmax(np.abs(img_mono).max(axis=1)))
    ir = int(np.argmax(np.abs(img_mono).max(axis=0)))

    span = args.span if args.span is not None else 4.5 * max(res_az, res_rg)
    na = max(8, int(np.ceil(3.0 * span / d_az)))
    nr = max(8, int(np.ceil(3.0 * span / rho_r)))
    zpa = max(1, int(np.ceil(24.0 * d_az / span)))
    zpr = max(1, int(np.ceil(24.0 * rho_r / span)))
    x_az = (np.arange(2 * na * zpa) - na * zpa) * d_az / zpa
    cr = (2 * nr * zpr) // 2

    z_mono = zoom2d(img_mono, ia, ir, na, nr, zpa, zpr)
    irw0, pslr0 = cut_metrics(z_mono[:, cr], x_az)
    print(f"  monostatic : IRW {irw0:5.2f} m | PSLR {pslr0:6.2f} dB "
          f"(expect ~ -13.3)\n")

    # ---- the sweep --------------------------------------------------------
    eps_list = sorted(set(args.eps) | ({-e for e in args.eps}
                                       if args.symmetric else set()))
    print(f"{'eps [m]':>8} {'peak [%]':>9} {'IRW [m]':>8} {'PSLR [dB]':>10} "
          f"{'Sigma [cyc]':>12}")
    rows, cuts = [], []
    for eps in eps_list:
        rec = reconstruct_at(p, tr, ch, eps, hw, args, mk)
        img = focus2d(rec, ref)
        z = zoom2d(img, ia, ir, na, nr, zpa, zpr)
        cut = z[:, cr]
        cuts.append(cut)
        irw, pslr = cut_metrics(cut, x_az)
        pk = 100.0 * np.abs(img).max() / nrm
        sig = sigma_cycles(p, tr, ptg, args.height + eps)
        rows.append((eps, pk, irw, pslr, sig))
        print(f"{eps:8.1f} {pk:9.1f} {irw:8.2f} {pslr:10.2f} {sig:12.4f}")

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_a = np.array([r[0] for r in rows])
    pk_a = np.array([r[1] for r in rows])
    sg_a = np.array([r[4] for r in rows])
    cmap = plt.get_cmap("viridis")
    scale = max(np.max(np.abs(eps_a)), 1e-9)

    def db(v):
        return 20.0 * np.log10(np.abs(v) / nrm + 1e-20)

    # 2 x 2 grid: the two summary panels side by side on top, the IRF spanning
    # the bottom.  Aspect ~1.5:1 instead of 3.5:1, so the figure can be printed
    # upright at full text width and still be read.
    fig = plt.figure(figsize=(11, 8.2), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25],
                          hspace=0.30, wspace=0.26)
    ax_pk = fig.add_subplot(gs[0, 0])
    ax_sg = fig.add_subplot(gs[0, 1])
    ax_irf = fig.add_subplot(gs[1, :])

    ax_pk.plot(eps_a, pk_a, "o-")
    if 0.0 in eps_a:
        ax_pk.axhline(pk_a[list(eps_a).index(0.0)], ls="--", c="k", lw=0.8,
                      label="oracle (eps = 0)")
        ax_pk.legend(fontsize=8)
    ax_pk.set_xlabel("DEM error [m]")
    ax_pk.set_ylabel("focused peak [% of monostatic]")
    ax_pk.set_title("peak vs DEM error")
    ax_pk.grid(alpha=.3)

    ax_sg.plot(sg_a, pk_a, "o")
    ax_sg.set_xlabel(r"$\Sigma$  [cycles]")
    ax_sg.set_ylabel("focused peak [% of monostatic]")
    ax_sg.set_title("does the spread predict it?")
    ax_sg.grid(alpha=.3)

    # all curves normalised to the MONOSTATIC peak, so the loss of peak is
    # visible; normalising each to its own peak would hide the main effect
    ax_irf.plot(x_az, db(z_mono[:, cr]), "k--", lw=1.4, label="monostatic",
                zorder=5)
    for (e, *_), cut in zip(rows, cuts):
        ax_irf.plot(x_az, db(cut), lw=1.0, color=cmap(abs(e) / scale),
                    label=f"$\\epsilon$ = {e:+.0f} m")
    ax_irf.set_xlabel("Azimuth [m]")
    ax_irf.set_ylabel("Impulse response [dB]")
    ax_irf.set_xlim(-2 * span, 2 * span)
    ax_irf.set_ylim(args.floor, 3)
    ax_irf.set_title("azimuth IRF vs DEM error")
    ax_irf.grid(alpha=.3)
    ax_irf.legend(fontsize=8, ncol=4, loc="lower center", framealpha=.9)

    fig.suptitle(f"$N_{{rx}}$={p.Nrx} | $b_{{xt}}^{{max}}$="
                 f"{np.max(np.abs(p.bxt)):.0f} m | $h$={args.height:.0f} m"
                 + ("  |  explicit co-registration" if args.coreg else ""),
                 fontsize="medium")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.07)

    out = args.out or os.path.join(
        PLOTS, f"dem_error{'_coreg' if args.coreg else ''}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    np.savez(os.path.splitext(out)[0] + ".npz",
             eps=eps_a, peak=pk_a, irw=[r[2] for r in rows],
             pslr=[r[3] for r in rows], sigma=sg_a,
             cuts=np.array(cuts), x_az=x_az, height=args.height)
    print(f"\nfigure written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())