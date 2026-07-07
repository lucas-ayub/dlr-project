# -*- coding: utf-8 -*-
"""
Driver for the modular SAR reconstruction pipeline.

Run:
    python run_experiment.py

To change the system or geometry, edit the presets in sar_recon/config.py
(make_diff_config / make_dpca_config) or build your own ExperimentConfig and
pass it to run_case().

To run with multiple scattering points, you don't need to touch this file:
add a new named scene to sar_recon.config.SCENE_PRESETS (a tuple of
(dx, dy, dh) offsets in metres relative to the central reconstruction point),
then list that name in SCENE_NAMES below.
"""
import os

import sar_recon as sar

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# What to sweep. Edit these freely.
CHANNEL_NUMBERS = list(range(2, 10))
# CASES = ["large_bat", "dpca", "dpca_offset"]
# "single" reproduces the original one-target behaviour. Add any other name
# defined in sar_recon.config.SCENE_PRESETS to also run multi-point scenes,
# e.g. SCENE_NAMES = ["single", "along_track_line", "varied_heights"].
SCENE_NAMES = ["single", "along_track_line", "cross_track_patch", "varied_heights"]
MAKE_PLOTS = True
# Set True to render all plot text through a real LaTeX install (Computer
# Modern font, etc). Requires LaTeX + dvipng + Ghostscript on PATH -- see
# sar_recon.plotting.enable_latex_fonts docstring. Leave False if you don't
# have that toolchain installed.
USE_LATEX_FONTS = True
SAVE_VECTOR = True


CHANNEL_NUMBERS = [2, 3, 4, 5, 6]
#CASES = ["topo_dxt0", "topo_dxt10", "topo_dxt20", "topo_dxt50", "topo_dxt100"]
#SCENE_NAMES = ["topo_ramp"]

CASES = ["topo_dpca_dxt0", "topo_dpca_dxt10", "topo_dpca_dxt20", "topo_dpca_dxt50", "topo_dpca_dxt100"]
SCENE_NAMES = ["topo_ramp"]

def run_case(cfg: sar.ExperimentConfig, make_plots: bool = True, save_vector: bool = False):
    """Full forward + reconstruction + (optional) diagnostics for one config."""
    print(f"r0 = {cfg.scene.r0}")
    print(f"y0 = {cfg.scene.y0}")
    print(f"scene points: {len(cfg.scene.points)} "
          f"(center + {len(cfg.scene.extra_offsets)} extra)")

    tracks = sar.build_platform_tracks(cfg)

    sref = sar.generate_reference(cfg, tracks)
    s_channel = sar.generate_channels(cfg, tracks)

    srecN = sar.reconstruct(cfg, tracks, s_channel, zeroOutBw=True)

    res = sar.analyze(cfg, sref, srecN)


    if make_plots:
        sar.plot_combined(cfg, res, vector=save_vector)
        sar.plot_polyfit_diagnostic(cfg, tracks, vector=save_vector)
        sar.plot_geometry_3d(cfg, vector=save_vector)
        # Scene-layout plots only add value when there's more than one point;
        # cheap either way, so just gate them on extra_offsets being set.
        if cfg.scene.extra_offsets:
            sar.plot_scene_points(cfg, vector=save_vector)
            sar.plot_scene_points_3d(cfg, vector=save_vector)

    return res


def main():
    if USE_LATEX_FONTS:
        # Must happen before any figure is created -- rcParams are global and
        # only affect plots made after this point.
        sar.enable_latex_fonts()

    for Nrx in CHANNEL_NUMBERS:
        for case in CASES:
            for scene_name in SCENE_NAMES:
                cfg = sar.CONFIG_FACTORIES[case](Nrx, SCRIPT_DIR, scene_name)
                print(f"\n=== case={case} | scene={scene_name} | Nrx={Nrx} | "
                      f"prf={cfg.prf:.1f} | PRF_op={cfg.PRF_op:.1f} ===")
                run_case(cfg, make_plots=MAKE_PLOTS, save_vector=SAVE_VECTOR)
    print("end")


if __name__ == "__main__":
    main()