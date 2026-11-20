# Project report

[`report.pdf`](report.pdf) — *Topography-Dependent Azimuth Reconstruction in Multistatic SAR Systems* (draft).

## Build

```bash
cd report
python scripts/make_figures.py all     # figures of chapters 2-4 (figures/*.pdf), runs the sata1d package
latexmk -pdf report.tex                # pdflatex + bibtex
```

The figures of the results chapter come from three places:

- `figures/pres/`: result figures of the final presentation (cropped from the talk);
- `figures/analysis/`: the analyses of `runs/analysis_v30/` (height of the mountain, scenes full of targets, baselines that
  vary in time, close targets next to a cliff, other solvers). Run the scripts of those folders with `FIG_STYLE=report` to remake them in the fonts of the
  report, then copy their `fig/*.pdf` here; every number of the text is in their `results.md`;
- `figures/diagrams/`: diagrams of the talk, standalone (`pdflatex <name>.tex` in that folder).

The remaining figures are TikZ/pgfplots in `tikz/` and in the `.tex` files.

```
report.tex          main file (chapters in sections/)
preamble.tex        packages, colours and notation macros
references.bib      bibliography
figures/            figures of make_figures.py; sub-folders as above
scripts/            make_figures.py
```
