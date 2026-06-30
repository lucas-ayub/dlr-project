# -*- coding: utf-8 -*-
"""
Driver for the modular SAR reconstruction pipeline.

Run:
    python run_experiment.py

To change the system or geometry, edit the presets in sar_recon/config.py
(make_diff_config / make_dpca_config) or build your own ExperimentConfig and
pass it to run_case().
"""
import os

import sar_recon as sar

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# What to sweep. Edit these freely.
CHANNEL_NUMBERS = list(range(2, 10))
CASES = ["diff", "dpca"]
MAKE_PLOTS = True


def run_case(cfg: sar.ExperimentConfig, make_plots: bool = True):
    """Full forward + reconstruction + (optional) diagnostics for one config."""
    print(f"r0 = {cfg.scene.r0}")
    print(f"y0 = {cfg.scene.y0}")

    tracks = sar.build_platform_tracks(cfg)

    sref = sar.generate_reference(cfg, tracks)
    s_channel = sar.generate_channels(cfg, tracks)

    srecN = sar.reconstruct(cfg, tracks, s_channel, zeroOutBw=True)

    res = sar.analyze(cfg, sref, srecN)

    if make_plots:
        sar.plot_combined(cfg, res)
        sar.plot_polyfit_diagnostic(cfg, tracks)
        sar.plot_geometry_3d(cfg)

    return res


def main():
    for Nrx in CHANNEL_NUMBERS:
        for case in CASES:
            cfg = sar.CONFIG_FACTORIES[case](Nrx, SCRIPT_DIR)
            print(f"\n=== case={case} | Nrx={Nrx} | prf={cfg.prf:.1f} | "
                  f"PRF_op={cfg.PRF_op:.1f} ===")
            run_case(cfg, make_plots=MAKE_PLOTS)
    print("end")


if __name__ == "__main__":
    main()
