# -*- coding: utf-8 -*-
"""
``sata2d`` -- self-contained two-step 2-D reconstruction + sub-band SATA.

Isolated on purpose: this package imports NOTHING from ``sar_recon`` and
nothing proprietary (no ``nusarPython``, ``SignalProcessingLibrary``,
``inverseFilter``, ``sarProccessing``, ``Orbit``, ``rat``, ``h5py``).
Copy the directory anywhere, ``pip install numpy scipy matplotlib``, and run.

Entry points
------------
``params.make_params``            acquisition parameters ("small" / "paper")
``datagen2d.get_raw_data_2d``     2-D raw data for a linear orbit
``rangecomp.range_compress``      range matched filter
``rd_recons2d.run_two_step``      the two-step Range-Doppler reconstruction
``sata2d.sata_1d``                corrected SATA kernel
``sata2d.sata_channels_subband``  per-sub-band SATA of the channel cube

Scripts
-------
``run_twostep2d.py``          full 2-D pipeline (port of MP2DATBF.py)
``run_sata_diagnostics.py``   isolates the sub-band SATA bug
``run_sata2d_topo.py``        2-D SATA with bat + bxt + topography

3-D extension
-------------
``params3d.make_params3d``        geometry with along- AND cross-track baselines
``geom3d.CoeffTable3D``           numeric (C0,C1,C2,Dt) vs range, per channel
``datagen3d.generate_channels_3d`` raw data for a 3-D scene
``sata3d.reconstruct_subband_2d`` two-step reconstruction + SATA
"""
from .params import Params, make_params
from .datastore import DataStore
from .datagen2d import get_raw_data_2d, get_range_hist, get_echo
from .rangecomp import range_compress, range_matched_filter
from .rd_recons2d import (get_coeff_nu, get_inversion_filters,
                          generalized_rd, run_two_step)
from .sata2d import (sata_1d, legacy_sata_1d, subaperture_grid,
                     legacy_subaperture_grid, subband_centre_frequency,
                     build_delta_C0_array, sata_channels_subband)
# --- 3-D extension: along- AND cross-track baselines + topography ----------
from .params3d import Params3D, make_params3d, make_bxt, iso_range_offset
from .geom3d import (build_tracks_3d, get_coeff_nu_3d, residual_C0_3d,
                     CoeffTable3D, dC0_approx)
from .datagen3d import (get_raw_data_3d, generate_reference_3d,
                        generate_channels_3d)
from .sata3d import (build_delta_C0_map_3d, sata_channels_subband_3d,
                     reconstruct_subband_2d, scatterer_range_bistatic)

__all__ = [
    "Params", "make_params", "DataStore",
    "get_raw_data_2d", "get_range_hist", "get_echo",
    "range_compress", "range_matched_filter",
    "get_coeff_nu", "get_inversion_filters", "generalized_rd", "run_two_step",
    "sata_1d", "legacy_sata_1d", "subaperture_grid", "legacy_subaperture_grid",
    "subband_centre_frequency", "build_delta_C0_array", "sata_channels_subband",
    "Params3D", "make_params3d", "make_bxt", "iso_range_offset",
    "build_tracks_3d", "get_coeff_nu_3d", "residual_C0_3d", "CoeffTable3D",
    "dC0_approx", "get_raw_data_3d", "generate_reference_3d",
    "generate_channels_3d", "build_delta_C0_map_3d",
    "sata_channels_subband_3d", "reconstruct_subband_2d",
    "scatterer_range_bistatic",
]
