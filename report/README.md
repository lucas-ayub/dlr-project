# Project report

[`report.pdf`](report.pdf) — *Topography-Dependent Azimuth Reconstruction in Multistatic SAR Systems* (draft).

## Build

```bash
cd report
python scripts/make_data.py ~/DLR/dlr-project_resultados   # numbers and tables, when the sweep results change
python scripts/make_figures.py all                         # result figures (figures/*.pdf), runs the sata1d package
latexmk -pdf report.tex                                    # pdflatex + bibtex
```

The figures are TikZ/pgfplots in the `.tex` files; the plotted numbers come from `data/*.dat` and `numbers.tex`,
written by `scripts/make_data.py` from the sweep outputs (`all_numbers.json` of `sweep_joint_all.py`,
`run_sweep_joint2d_all.py` and `audit_dem.py`), which are not versioned. `figures/*.pdf` are made by
`scripts/make_figures.py` with the code of this repository; the three PNG pages are copied from the sweeps.

```
report.tex          main file (chapters in sections/)
preamble.tex        packages, colours and notation macros
references.bib      bibliography
numbers.tex         numbers quoted in the text (generated)
data/               pgfplots tables (generated)
figures/            PNG pages from the sweeps
scripts/            make_data.py, make_figures.py
```
