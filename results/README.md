# Results

Outputs of `sata1d_code/` and `sar_reconstruction/sata2d/`. Nothing here is needed to run the code.

```
sata1d/
  reports/                    PDFs (current)
    sata1d_ramps_distance_height_part1.pdf     ramps, targets 200 m apart: summary, heatmaps, IRFs
    sata1d_ramps_distance_height_part2.pdf     ramps, targets 2.6 km and 10 km apart
    sata1d_random_distance_height_part1.pdf    random heights, piecewise ramps, zig-zag
    sata1d_random_distance_height_part2.pdf    steepening, flattening, cliff, spike, mesa
    sata1d_open_points.pdf                     sub-band vs whole band, metric, saturation (+ IRFs at 5 and 10 km)
    sata1d_algorithm_and_verification.pdf, sata1d_report.pdf, sata1d_summary.pdf, guia_sata_novo.pdf, ...
    previous_versions/        superseded PDFs (older metric / with the oracle)
  sweep_csv/                  metrics of every case of each sweep (one row per case)
    sweep_distance_height_results.csv          the current sweep (sweep_v4.py, 1053 cases)
    sweep_open_points_*.csv                    sweep_v3.py (flat-reference metric)
    sweep_200m_*, sweep_2600m_*                the earlier ramp sweeps
  figures/                    single figures (summary pages of the current sweep, early heatmaps, ...)
  sweeps/                     older sweep outputs with one PNG/NPZ per case, verification figures
sata2d/                       SATA 2D sweep, review figures, sata_review.pdf
```
