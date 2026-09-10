# -*- coding: utf-8 -*-
"""
Array geometry helpers -- the DPCA condition.

DPCA (displaced phase centre antenna) is the along-track timing that makes the
multichannel sampling UNIFORM.  With one transmitter and ``Nrx`` receivers, the
effective phase centre of channel ``i`` sits halfway between transmitter and
receiver, at ``bat_i/2``.  Uniform sampling requires those phase centres to be
spaced by exactly the distance the platform travels between two samples of the
reconstructed (full-PRF) signal::

    dx/2  =  vs / PRF          <=>          PRF = 2 vs / dx

which is ``prf_from_dpca`` of ``sar_recon.config`` written the other way round
(there: ``PRF_op = 2 vs / (Nrx dx)``, and ``PRF = Nrx PRF_op``).

Off the DPCA condition the phase centres are non-uniformly spaced; the
reconstruction still inverts the sampling exactly, but the inversion is worse
conditioned and residual azimuth ambiguities rise.  Every study in this package
is therefore run ON the DPCA condition, so that whatever is left over is
attributable to the geometry under test and not to the array timing.

Two ways to satisfy it:

* ``dpca_dx(prf)``  -- keep the PRF and derive the along-track step.  This is
  what the studies use: nothing else about the system changes, only ``bat``.
* ``dpca_prf(dx)``  -- keep the step and derive the PRF, as
  ``make_topo_dpca_random_config`` does with ``dx = 11 m``.

NOTE: the ``"DPCA dx=11"`` case of ``run_sata2d_topo.test3_sweep`` passes
``dx = 11`` while leaving ``prf`` at its default 2000 Hz.  That is NOT the DPCA
condition (it would need PRF = 1397.9 Hz); it is a small-baseline case with a
misleading label.
"""
from __future__ import annotations

import numpy as np

from .geometry import make_params3d

__all__ = ["dpca_dx", "dpca_prf", "dpca_residual", "make_params_dpca"]

VS_DEFAULT = 7688.53706432   # make_params3d's default platform speed [m/s]


def dpca_dx(prf: float, vs: float = VS_DEFAULT) -> float:
    """Along-track step ``dx`` that satisfies DPCA at this ``prf`` [m]."""
    return 2.0 * vs / prf


def dpca_prf(dx: float, vs: float = VS_DEFAULT) -> float:
    """PRF that satisfies DPCA for this along-track step ``dx`` [Hz]."""
    return 2.0 * vs / dx


def dpca_residual(p) -> float:
    """How far a parameter set is from DPCA, as a fraction of the ideal step.

    Zero means exactly on the condition.  Uses the actual ``bat`` spacing, so
    it also catches a non-uniform ladder.
    """
    if p.Nrx < 2:
        return 0.0
    step = np.diff(p.bat)
    ideal = dpca_dx(p.prf, p.vs)
    return float(np.max(np.abs(step - ideal)) / ideal)


def make_params_dpca(Nrx: int = 4, prf: float = 2000.0, bxt_mode: str = "random",
                     bxt_max: float = 100.0, seed: int = 0, verbose: bool = False,
                     **kw):
    """``make_params3d`` with ``dx`` forced onto the DPCA condition.

    The PRF is kept and the along-track step derived from it, so the only thing
    that changes with respect to the plain defaults is ``bat``.  Cross-track
    baselines default to the realistic random array, ``bxt ~ U(0, bxt_max)``.
    Any other keyword is forwarded to ``make_params3d``.
    """
    dx = dpca_dx(prf, kw.get("vs", VS_DEFAULT))
    p = make_params3d(Nrx=Nrx, dx=dx, prf=prf, bxt_mode=bxt_mode,
                      bxt_max=bxt_max, seed=seed,
                      dxt=kw.pop("dxt", bxt_max), **kw)
    if verbose:
        print(f"  DPCA: dx = {dx:.4f} m, phase-centre spacing {dx/2:.4f} m "
              f"= vs/PRF = {p.vs/p.prf:.4f} m  (residual {dpca_residual(p):.2e})")
    return p
