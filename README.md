# dlr-project

Numerical investigation of **topography-dependent azimuth reconstruction in multichannel / multistatic SAR systems**
(HRWS), developed at the DLR Microwaves and Radar Institute.

**Research question:** what is the impact of unaccounted scene topography on multichannel SAR azimuth reconstruction,
and how well can it be corrected (SATA and the joint SATA + reconstruction) using only system parameters and a DEM?

---

## Setup

```bash
git clone https://github.com/lucas-ayub/dlr-project.git
cd dlr-project
pip install -r requirements.txt
pip install -e .          # makes `import sar_recon` / `import sata2d` work from anywhere
```

`sar_reconstruction/sata2d` also needs `scipy`.

---

## Structure

```
dlr-project/
├── sata1d_code/                     SATA 1D (azimuth only) — main code, see sata1d_code/README.md
│   ├── sata1d/                      package: config, geometry, signal model, reconstruction,
│   │                                SATA (whole band / per sub-band), joint, joint_line
│   ├── joint_cases/                 joint reconstruction: case runs, sweeps, audits with a DEM only, guides
│   ├── joint_standalone/            minimal standalone copy of the whole-line joint reconstruction (joint1d)
│   ├── legacy/                      previous per-sub-band fit (used by the verification scripts)
│   └── *.py                         run, sweep, plot and verification scripts
│
├── sar_reconstruction/
│   ├── sar_recon/                   original reconstruction package (import sar_recon)
│   ├── sata2d/                      SATA 2D with 3-D geometry and the 2-D joint, see sata2d/README.md
│   └── CUSTOMIZATION_GUIDE.md
│
├── runs/                            scripts of the original sar_recon pipeline
│   ├── core/                        reconstruction, coefficient and sub-band experiments
│   ├── validation/                  validation against the analytic model
│   └── sata/                        reference SATA code (IDL and Python port)
│
├── results/                         scripts that assemble the report PDFs (outputs are not versioned)
├── pyproject.toml
└── requirements.txt
```

Generated figures, PDFs and numbers are written by the scripts to `plots/` or next to them and are git-ignored.

## Quick start

```bash
cd sata1d_code
python run_one_case.py --scene dense_ramp --mode none joint oracle --map lut
cd joint_standalone && python run_example.py && cd ..

cd ../sar_reconstruction
python -m sata2d.run_sata2d_topo --plots
```
