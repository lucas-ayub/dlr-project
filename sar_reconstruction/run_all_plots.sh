#!/usr/bin/env bash
# Regenerate every figure of the sata2d package (~25-30 min in total).
# Run from sar_reconstruction/:   bash run_all_plots.sh
set -e
export MPLBACKEND=Agg

python -m sata2d.run_sata2d_topo --plots
python -m sata2d.run_sata_irf --method sub
python -m sata2d.run_sata_irf_all

python -m sata2d.run_irf2d
python -m sata2d.run_irf2d --method sub
python -m sata2d.run_irf2d_full
python -m sata2d.run_irf2d_full --method no
python -m sata2d.run_irf2d_full --method sub

python -m sata2d.run_cases
python -m sata2d.plot_cases

python -m sata2d.run_dem_error --eps 0 2 5 10 20 50 100 --bxt-max 225

python -m sata2d.run_check
python -m sata2d.plot_esr
python -m sata2d.run_axes_2d
python -m sata2d.run_c1c2
python -m sata2d.run_bxt_test

python -m sata2d.run_coreg --stage equiv
python -m sata2d.run_coreg --stage pipeline
python -m sata2d.plot_coreg

echo "done: figures in sata2d/plots/ (and plots/sata2d/ for run_sata2d_topo)"