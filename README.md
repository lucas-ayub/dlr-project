# dlr-project

Numerical investigation of **topography-dependent azimuth reconstruction in
multistatic SAR systems**.

**Research question:** what is the impact of unaccounted scene topography on
multichannel SAR azimuth reconstruction?

---

## Setup

```bash
git clone https://github.com/lucas-ayub/dlr-project.git
cd dlr-project
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` requires the `pyproject.toml` at the repo root (points at
`sar_reconstruction/` as the package source) and makes `import sar_recon`
work from anywhere, without manually setting `PYTHONPATH`.

---

## Architecture

```
dlr-project/
├── sar_reconstruction/
│   ├── sar_recon/                  # the package (import sar_recon)
│   │   ├── config.py
│   │   ├── geometry.py
│   │   ├── signal_model.py
│   │   ├── reconstruction.py
│   │   ├── analysis.py
│   │   └── plotting.py
│   └── CUSTOMIZATION_GUIDE.md
│
├── runs/
│   ├── core/                       # main reconstruction pipeline
│   │   ├── run_experiment.py
│   │   └── run_coeff_sweep.py
│   └── validation/                 # validation against the analytic model
│       ├── run_band_comparison.py
│       ├── run_height_sweep_isorange.py
│       └── run_model_validation_report.py
│
├── archive/
│   └── main_legacy.py              # original prototype, pre-modularization
│
├── plots/                          # generated figures (git-ignored)
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

All scripts run from the repo root:

```bash
python runs/core/run_experiment.py
python runs/validation/run_model_validation_report.py
```

Each writes its figures to `plots/<category>/<script_name>/`.
