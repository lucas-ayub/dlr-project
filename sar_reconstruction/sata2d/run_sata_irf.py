# -*- coding: utf-8 -*-
"""
SATA IRF playground -- ONE method, ONE IRF plot.

A minimal, single-target script: build the STANDARD 3-D geometry (same
defaults as the rest of this package -- wl=0.25 m, H=720 km, r0=766.21 km,
PRF=2000 Hz, Nrx=4), place ONE target at a chosen azimuth offset and height,
reconstruct it with ONE method (no SATA / SATA whole band / SATA per
sub-band -- pick with --method) and plot ref vs. rec (dB, full azimuth time
axis, sar_recon-style parameter box).

For the three methods together in one plot, see run_sata_irf_all.py.

Edit ONLY the "TARGET GEOMETRY" block below -- azimuth offset and height of
the target -- and re-run to see how the IRF changes.  Everything else (the
platform, the array, the acquisition parameters) stays at the defaults used
throughout the report, so results across runs stay directly comparable.  The
array geometry (Nrx, cross-track baseline step) is also exposed below, in
case you want to see the effect of a different array, but it is not the
point of this script -- for that, see run_sata2d_topo.py's test3_sweep.

Run
---
    cd sar_reconstruction
    python -m sata2d.run_sata_irf                  # SATA per sub-band (default)
    python -m sata2d.run_sata_irf --method whole    # SATA whole band
    python -m sata2d.run_sata_irf --method no       # no SATA

Each run overwrites ``plots/sata_irf.png`` next to this script (or pass
--out to change that) and prints the peak amplitude of ref and rec,
normalised to the monostatic reference.
"""
from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg_dir))
    __package__ = os.path.basename(_pkg_dir)  # works whatever this folder is named

from .reconstruction import run_irf, plot_irf_single  # noqa: E402

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

# ===========================================================================
# TARGET GEOMETRY -- this is the block to edit.  Everything below it is the
# standard system/array geometry from the rest of the report; leave it alone
# unless you specifically want to explore a different platform or array.
# ===========================================================================
TARGET_AZIMUTH = 0.0        # [m] azimuth offset of the target from scene centre
TARGET_HEIGHT = 240.0       # [m] target height above the reference surface
#                             (0.0, 0.0) reproduces the plain monostatic IRF:
#                             no SATA, whole band and per sub-band all coincide.

NRX = 4                     # number of receive channels
DXT = 150.0                 # [m] cross-track baseline step ("linear" ladder)
SATA_OSF = 4                # SATA sub-aperture zero-padding oversampling
# ===========================================================================


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("no", "whole", "sub"), default="sub",
                    help="which reconstruction to plot against the monostatic "
                         "reference (default: sub)")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: plots/sata_irf.png "
                         "next to this script; created if missing)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    out = args.out if args.out is not None else os.path.join(PLOTS_DIR, "sata_irf.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    p, res = run_irf(TARGET_AZIMUTH, TARGET_HEIGHT, NRX, DXT, SATA_OSF,
                     verbose=args.verbose, methods=(args.method,))
    plot_irf_single(p, res, args.method, out=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
