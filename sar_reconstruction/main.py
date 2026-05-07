# -*- coding: utf-8 -*-

import numpy as np

from run_pipeline import run_case


def main():
    # ============================================================
    # Scene / system parameters
    # ============================================================

    scene_params = {
        "wl": 0.25,
        "v": 7408.5313923924796,
        "r0": 700e3,
        "H": 514e3,
        "h0": 0.0,
        "prf": 1500.0,
        "prf_margin": 1.2,
        "divfac": 1024,
        "antenna_length_factor": 24,
    }

    # ============================================================
    # Test cases
    # ============================================================

    test_cases = {
        "2ch_mixed_large": {
            "title": "Two-Channel Mixed Baseline",
            "b_at": [0.0, 10.0],
            "b_xt": [0.0, 600.0],
        },

        "2ch_at_medium": {
            "title": "Two-Channel Along-Track Baseline",
            "b_at": [0.0, 30.0],
            "b_xt": [0.0, 0.0],
        },

        "2ch_at_large": {
            "title": "Two-Channel Along-Track Baseline",
            "b_at": [0.0, 90.0],
            "b_xt": [0.0, 0.0],
        },

        "3ch_mixed_uniform": {
            "title": "Three-Channel Mixed Baseline Formation",
            "b_at": [0.0, 30.0, 60.0],
            "b_xt": [200.0, 200.0, 200.0],
        },
    }

    cases_to_run = [
        "2ch_mixed_large",
        # "2ch_at_medium",
        # "2ch_at_large",
        # "3ch_mixed_uniform",
    ]

    for case_name in cases_to_run:
        run_case(
            case_name=case_name,
            case_params=test_cases[case_name],
            scene_params=scene_params,
            save_plots=True,
            show_plots=False,
        )


if __name__ == "__main__":
    main()