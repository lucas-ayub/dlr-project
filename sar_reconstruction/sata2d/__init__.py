# -*- coding: utf-8 -*-
"""
``sata2d`` -- SATA (2-D) with 3-D geometry (along- and cross-track baselines
+ topography) and the two-step Range-Doppler reconstruction it sits on.

Three library files, plus the run scripts:

* ``geometry.py``       -- parameters, tracks, coefficients, raw-data generation
* ``sata.py``           -- the SATA kernel itself (unchanged, 2-D)
* ``reconstruction.py`` -- the two-step reconstruction, SATA applied to the
                           3-D geometry, and the IRF-playground machinery

Isolated on purpose: this package imports NOTHING from ``sar_recon`` and
nothing proprietary (no ``nusarPython``, ``SignalProcessingLibrary``,
``inverseFilter``, ``sarProccessing``, ``Orbit``, ``rat``, ``h5py``). Copy the
directory anywhere, ``pip install numpy scipy matplotlib``, and run.

Note: this folder can be renamed freely (e.g. to ``sata_2d``) -- every
run_*.py script derives its own package name from the folder it actually
lives in when run directly, instead of assuming a fixed name.

Scripts
-------
``run_sata2d_topo.py``    the main experiment (self-test, residual table,
                          sweep, azimuth-topography, multi-range)
``run_sata_irf.py``       playground: one target, ONE method, one IRF plot
``run_sata_irf_all.py``   playground: one target, the THREE methods together
"""
from .geometry import (Params3D, make_params3d, make_bxt, iso_range_offset,
                       make_topo_ramp_azimuth, build_tracks_3d, get_coeff_nu,
                       get_coeff_nu_3d, residual_C0_3d, CoeffTable3D,
                       dC0_approx, get_raw_data_3d, generate_reference_3d,
                       generate_channels_3d)
from .sata import (sata_1d, legacy_sata_1d, subaperture_grid,
                   legacy_subaperture_grid, subband_centre_frequency,
                   build_delta_C0_array, sata_channels_subband)
from .reconstruction import (DataStore, range_compress, range_matched_filter,
                             get_inversion_filters, create_ref_dataset,
                             interp_filter, generalized_rd, run_two_step,
                             finish_spectrum, build_delta_C0_map_3d,
                             channels_subband_3d, reconstruct_subband_2d,
                             scatterer_range_bistatic, range_bin_of,
                             scatterer_range)

__all__ = [
    "Params3D", "make_params3d", "make_bxt", "iso_range_offset",
    "make_topo_ramp_azimuth",
    "build_tracks_3d", "get_coeff_nu", "get_coeff_nu_3d", "residual_C0_3d",
    "CoeffTable3D", "dC0_approx", "get_raw_data_3d", "generate_reference_3d",
    "generate_channels_3d",
    "sata_1d", "legacy_sata_1d", "subaperture_grid", "legacy_subaperture_grid",
    "subband_centre_frequency", "build_delta_C0_array", "sata_channels_subband",
    "DataStore", "range_compress", "range_matched_filter",
    "get_inversion_filters", "create_ref_dataset", "interp_filter",
    "generalized_rd", "run_two_step", "finish_spectrum",
    "build_delta_C0_map_3d", "channels_subband_3d", "reconstruct_subband_2d",
    "scatterer_range_bistatic", "range_bin_of", "scatterer_range",
]
