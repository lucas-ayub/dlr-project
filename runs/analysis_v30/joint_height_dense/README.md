# joint_height_dense — joint reconstruction: mountain height sweep and scenes full of point targets

Run from this folder (needs the dlr-project repository: found by walking up to a folder with `sata1d_code/`, or set
`DLR_REPO`; SRTM lines need `DEM_DIR` with main_srtm90.tif):

    python3 -I item1_height.py     # 31 heights, ~40 s   -> cache/height_*.npz, cache/item1_height.json
    python3 -I plot_item1.py       # fig/height_irf, fig/height_amb, cache/item1_table.(json|md)
    python3 -I item3_dense.py      # 108 runs, ~4 min    -> cache/dense_*.npz, cache/item3.json
    python3 -I plot_item3.py       # fig/dense_irf, fig/dense_summary, cache/item3_table.json

Files: `common.py` (setup: the same system, generators, focusing, band and metric as the report's results),
`style.py` (talk plot style), `profiles.py` (SRTM profiles).

Metrics (details and the check of the talk metric in results.md):
- highest ambiguity: largest |focused line| within +-154 m of x_p + m X (m = +-1, +-2; X = 6.23 km) of every target p,
  outside +-154 m of every target, dB re. the reference peak (unchanged from the talk; verified, no bug);
- floor: largest level anywhere outside +-154 m of every target (catches artefacts not at x_p + m X);
- weakest peak: smallest upsampled true-target peak, dB re. the reference peak;
- 3-dB width: main-lobe width at -3 dB, upsampled x16;
- image error: 20 log10 ||y - y_ref|| / ||y_ref|| of the focused complex lines (whole line, same band B_a);
- blind-height mismatch: as in `style.mismatch`, min over targets and m.
Dense scenes: the peak-based metric needs ambiguity windows outside every +-154 m guard; when there is none it is NaN
(never happened up to 200 targets on +-10 km). For N >= 20 compare its gap to the reference, not its level.
