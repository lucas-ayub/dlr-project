#!/usr/bin/env bash
# =============================================================================
# Regenerate EVERY figure of the sata2d package (~35-40 min in total).
#
#   cd sar_reconstruction
#   bash run_all_plots.sh               # everything
#   bash run_all_plots.sh 2>&1 | tee run_all.log
#
# A step that fails does not stop the others; the summary at the end lists
# what failed.  Figures go to sata2d/plots/ (and plots/sata2d/ for
# run_sata2d_topo).
# =============================================================================
cd "$(dirname "$0")"
export MPLBACKEND=Agg

# python or python3, whichever exists
if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi

OK=(); FAIL=()
run() {
    local name="$1"; shift
    echo ""
    echo "==================================================================="
    echo ">>> $name"
    echo "    $PY -m $*"
    echo "==================================================================="
    local t0=$SECONDS
    if $PY -m "$@"; then
        OK+=("$name ($((SECONDS - t0)) s)")
    else
        FAIL+=("$name")
    fi
}

P=sata2d/plots

# --- 1. main experiment: geometry, residual, sweep, azimuth topography -------
run "topo (4 figs + multi-range)"   sata2d.run_sata2d_topo --plots --multi-range

# --- 2. IRF playground (1-D, full slow-time axis) ----------------------------
run "sata_irf  no"                  sata2d.run_sata_irf --method no    --out $P/sata_irf_no.png
run "sata_irf  whole"               sata2d.run_sata_irf --method whole --out $P/sata_irf_whole.png
run "sata_irf  sub"                 sata2d.run_sata_irf --method sub   --out $P/sata_irf_sub.png
run "sata_irf_all"                  sata2d.run_sata_irf_all

# --- 3. 2-D IRF, zoom (contour + cuts) ---------------------------------------
run "irf2d mono"                    sata2d.run_irf2d
run "irf2d no"                      sata2d.run_irf2d --method no
run "irf2d whole"                   sata2d.run_irf2d --method whole
run "irf2d sub"                     sata2d.run_irf2d --method sub
run "irf2d sub coreg"               sata2d.run_irf2d --method sub --coreg

# --- 4. 2-D IRF, whole image (no zoom) ---------------------------------------
run "irf2d_full mono"               sata2d.run_irf2d_full
run "irf2d_full no"                 sata2d.run_irf2d_full --method no
run "irf2d_full whole"              sata2d.run_irf2d_full --method whole
run "irf2d_full sub"                sata2d.run_irf2d_full --method sub

# --- 5. the 16-case matrix + its summary figures ------------------------------
run "cases (16 figs + cases.json)"  sata2d.run_cases
run "plot_cases (3 figs)"           sata2d.plot_cases

# --- 6. DEM error ------------------------------------------------------------
run "dem_error"                     sata2d.run_dem_error --eps 0 2 5 10 20 50 100 --bxt-max 225
run "dem_error coreg"               sata2d.run_dem_error --eps 0 2 5 10 20 50 100 --bxt-max 225 --coreg

# --- 7. diagnostics ----------------------------------------------------------
run "check (cache)"                 sata2d.run_check
run "plot_esr (error_budget)"       sata2d.plot_esr
run "axes_2d"                       sata2d.run_axes_2d
run "c1c2 (table)"                  sata2d.run_c1c2
run "bxt_test random (table)"       sata2d.run_bxt_test --bxt-mode random
run "bxt_test linear (table)"       sata2d.run_bxt_test --bxt-mode linear

# --- 8. range co-registration ------------------------------------------------
run "coreg equiv (cache)"           sata2d.run_coreg --stage equiv
run "coreg pipeline (cache)"        sata2d.run_coreg --stage pipeline
run "plot_coreg (3 figs)"           sata2d.plot_coreg

# --- summary -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "done in $((SECONDS / 60)) min $((SECONDS % 60)) s"
echo "ok     : ${#OK[@]}"
for s in "${OK[@]}"; do echo "   + $s"; done
echo "failed : ${#FAIL[@]}"
for s in "${FAIL[@]}"; do echo "   - $s"; done
echo ""
echo "figures:"
ls -1 sata2d/plots/*.png sata2d/plots/cases/*.png plots/sata2d/*.png 2>/dev/null | sed 's/^/   /'
[ ${#FAIL[@]} -eq 0 ]