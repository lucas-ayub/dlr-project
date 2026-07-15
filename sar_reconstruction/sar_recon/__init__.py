# -*- coding: utf-8 -*-
"""SAR multichannel HRWS azimuth reconstruction — modular pipeline."""
from .config import (
    SystemParams, Scene, ArrayGeometry, ExperimentConfig,
    integration_time, build_time_axis,
    prf_from_fixed, prf_from_dpca,
    make_large_bat_config, make_dpca_config, make_dpca_offset_config,
    scene_ramp_angle_deg,
    CONFIG_FACTORIES, SCENE_PRESETS,
)
from .geometry import PlatformTracks, build_platform_tracks
from .signal_model import getRawData1D, generate_reference, generate_channels
from .reconstruction import (
    GetCoeffNu, GetInversionFilters, ReconstructSignalNumeri, reconstruct,
)
from .analysis import ReconResult, zoom1Dpeak, matched_filter, analyze
from .sata import (sata_1d, residual_C0, build_delta_C0_array, sata_channels,
                   sata_2d, build_delta_C0_2d, delta_C0_from_height)
from .plotting import (
    plot_combined, plot_polyfit_diagnostic, plot_geometry_3d,
    plot_scene_points, plot_scene_points_3d, enable_latex_fonts, set_font_size,
)

__all__ = [
    "SystemParams", "Scene", "ArrayGeometry", "ExperimentConfig",
    "integration_time", "build_time_axis", "prf_from_fixed", "prf_from_dpca",
    "make_large_bat_config", "make_dpca_config", "make_dpca_offset_config",
    "scene_ramp_angle_deg",
    "CONFIG_FACTORIES", "SCENE_PRESETS",
    "PlatformTracks", "build_platform_tracks",
    "getRawData1D", "generate_reference", "generate_channels",
    "GetCoeffNu", "GetInversionFilters", "ReconstructSignalNumeri", "reconstruct",
    "ReconResult", "zoom1Dpeak", "matched_filter", "analyze",
    "sata_1d", "residual_C0", "build_delta_C0_array", "sata_channels",
    "sata_2d", "build_delta_C0_2d", "delta_C0_from_height",
    "plot_combined", "plot_polyfit_diagnostic", "plot_geometry_3d",
    "plot_scene_points", "plot_scene_points_3d", "enable_latex_fonts", "set_font_size",
]