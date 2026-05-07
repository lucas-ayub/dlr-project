# -*- coding: utf-8 -*-

import numpy as np
import json
import os 
from run_pipeline import run_case

def load_test_cases(filename="test_cases.json"):

    base_dir = os.path.dirname(os.path.abspath(__file__))

    path = os.path.join(base_dir, filename)

    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)

all_cases = load_test_cases()

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

    test_cases = all_cases

    cases_to_run = all_cases.keys()

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