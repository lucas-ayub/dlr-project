#!/usr/bin/env bash
# =============================================================================
# Regenerate EVERY figure of the sata2d package and SAVE them -- nothing is
# shown on screen.  ~35-40 min in total.
#
#   cd sar_reconstruction
#   bash run_all_plots.sh
#
# Where things go
#   sata2d/plots/         every PNG (the case matrix in sata2d/plots/cases/)
#   sata2d/plots/logs/    the full text output of each step, one .log per step
#
# The terminal only shows one line per step ([ ok ] / [FAIL]).  A step that
# fails does not stop the others; the summary at the end lists them.
# =============================================================================
cd "$(dirname "$0")"
export MPLBACKEND=Agg          # render to files only, never open a window

if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi

P=sata2d/plots
LOG=$P/logs
mkdir -p "$P" "$LOG"

N_OK=0; FAIL=()
run() {
    local name="$1"; shift
    local slug; slug=$(echo "$name" | tr ' /()+' '_____' | tr -s '_')
    local t0=$SECONDS
    printf "  %-38s ... " "$name"
    if $PY -m "$@" > "$LOG/$slug.log" 2>&1; then
        N_OK=$((N_OK + 1))
        printf "[ ok ]  %4d s\n" $((SECONDS - t0))
    else
        FAIL+=("$name  (see $LOG/$slug.log)")
        printf "[FAIL]  %4d s\n" $((SECONDS - t0))
    fi
}

echo "saving every figure to $P/   (logs in $LOG/)"
echo ""

# --- 1. main experiment ------------------------------------------------------
run "topo + multi-range"      sata2d.run_sata2d_topo --plots --multi-range --outdir $P

# --- 2. IRF playground (1-D, full slow-time axis) ----------------------------
run "sata_irf no"             sata2d.run_sata_irf --method no    --out $P/sata_irf_no.png
run "sata_irf whole"          sata2d.run_sata_irf --method whole --out $P/sata_irf_whole.png
run "sata_irf sub"            sata2d.run_sata_irf --method sub   --out $P/sata_irf_sub.png
run "sata_irf_all"            sata2d.run_sata_irf_all            --out $P/sata_irf_all.png

# --- 3. 2-D IRF, zoom (contour + cuts) ---------------------------------------
run "irf2d mono"              sata2d.run_irf2d
run "irf2d no"                sata2d.run_irf2d --method no
run "irf2d whole"             sata2d.run_irf2d --method whole
run "irf2d sub"               sata2d.run_irf2d --method sub
run "irf2d sub coreg"         sata2d.run_irf2d --method sub --coreg

# --- 4. 2-D IRF, whole image (no zoom) ---------------------------------------
run "irf2d_full mono"         sata2d.run_irf2d_full
run "irf2d_full no"           sata2d.run_irf2d_full --method no
run "irf2d_full whole"        sata2d.run_irf2d_full --method whole
run "irf2d_full sub"          sata2d.run_irf2d_full --method sub

# --- 5. the 16-case matrix + its summary figures ------------------------------
run "cases (16 figs)"         sata2d.run_cases
run "plot_cases"              sata2d.plot_cases

# --- 6. DEM error ------------------------------------------------------------
run "dem_error"               sata2d.run_dem_error --eps 0 2 5 10 20 50 100 --bxt-max 225
run "dem_error coreg"         sata2d.run_dem_error --eps 0 2 5 10 20 50 100 --bxt-max 225 --coreg

# --- 7. diagnostics ----------------------------------------------------------
run "check"                   sata2d.run_check
run "plot_esr"                sata2d.plot_esr
run "axes_2d"                 sata2d.run_axes_2d
run "c1c2 (table -> log)"     sata2d.run_c1c2
run "bxt_test random (table)" sata2d.run_bxt_test --bxt-mode random
run "bxt_test linear (table)" sata2d.run_bxt_test --bxt-mode linear

# --- 8. range co-registration ------------------------------------------------
run "coreg equiv"             sata2d.run_coreg --stage equiv
run "coreg pipeline"          sata2d.run_coreg --stage pipeline
run "plot_coreg"              sata2d.plot_coreg

# --- summary -----------------------------------------------------------------
NPNG=$(find "$P" -name '*.png' | wc -l | tr -d ' ')
echo ""
echo "done in $((SECONDS / 60)) min $((SECONDS % 60)) s  |  $N_OK ok, ${#FAIL[@]} failed  |  $NPNG figures in $P/"
for s in "${FAIL[@]}"; do echo "  FAILED: $s"; done
[ ${#FAIL[@]} -eq 0 ]