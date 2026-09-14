# -*- coding: utf-8 -*-
"""
SATA IRF playground -- the three cases together, one plot.

Same standard 3-D geometry and single target as run_sata_irf.py, but
reconstructs all three methods (no SATA / SATA whole band / SATA per
sub-band) and overlays all four curves (+ the monostatic reference) in one
plot -- dB, full azimuth time axis, sar_recon-style parameter box.

For just one method at a time (faster, one rec curve), see run_sata_irf.py.

Edit ONLY the "TARGET GEOMETRY" block below and re-run.

Run
---
    cd sar_reconstruction
    python -m sata2d.run_sata_irf_all

Each run overwrites ``plots/sata_irf_all.png`` next to this script (or pass
--out to change that) and prints all four peak amplitudes, normalised to the
monostatic reference.
"""
from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg_dir))
    __package__ = os.path.basename(_pkg_dir)  # works whatever this folder is named

from .reconstruction import run_irf, plot_irf_all  # noqa: E402

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

# ===========================================================================
# TARGET GEOMETRY -- this is the block to edit.  Everything below it is the
# standard system/array geometry from the rest of the report; leave it alone
# unless you specifically want to explore a different platform or array.
# ===========================================================================
# These are DEFAULTS; every one of them is also a command-line flag, so a
# figure can be reproduced from the command that made it rather than by editing
# this file.  (0.0, 0.0) reproduces the plain monostatic IRF: no SATA, whole
# band and per sub-band all coincide.
TARGET_AZIMUTH = 0.0        # [m] azimuth offset of the target from scene centre
TARGET_HEIGHT = 240.0       # [m] target height above the reference surface

NRX = 4                     # number of receive channels
DXT = 100.0                 # [m] cross-track scale (= bxt_max in random mode)
SATA_OSF = 4                # SATA sub-aperture zero-padding oversampling
BXT_MAX = 100.0             # [m] bxt ~ U(0, BXT_MAX); bat is on DPCA
SEED = 0
# ===========================================================================


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: plots/sata_irf_all.png "
                         "next to this script; created if missing)")
    ap.add_argument("--height", type=float, default=TARGET_HEIGHT,
                    help="target height above the reference surface [m]")
    ap.add_argument("--azimuth", type=float, default=TARGET_AZIMUTH,
                    help="target azimuth offset from scene centre [m]")
    ap.add_argument("--nrx", type=int, default=NRX)
    ap.add_argument("--bxt-max", type=float, default=BXT_MAX, dest="bxt_max",
                    help="bxt ~ U(0, bxt_max); bat is always on the DPCA condition")
    ap.add_argument("--bxt-mode", default="random", dest="bxt_mode",
                    choices=("linear", "random"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sata-osf", type=int, default=SATA_OSF, dest="sata_osf")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    out = args.out if args.out is not None else os.path.join(PLOTS_DIR, "sata_irf_all.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    p, res = run_irf(args.azimuth, args.height, args.nrx, args.bxt_max,
                     args.sata_osf, verbose=args.verbose,
                     bxt_mode=args.bxt_mode, bxt_max=args.bxt_max, seed=args.seed)
    plot_irf_all(p, res, out=out, height=args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
