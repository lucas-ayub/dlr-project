# Joint reconstruction on real terrain (SRTM, Sierra Nevada)

Azimuth lines (1-D, 47 km) cut from the SRTM 3'' DEM (about 90 m; Farr et al., 2007) over the Sierra Nevada, file
`raster/srtm90.tif` of [opengeos/data](https://github.com/opengeos/data). Point targets on the terrain, four channels with
`bxt = [0, 300, 119.8, 100.5] m` (channel 0 on the transmitter); the joint reconstruction over the whole line
(`sata1d.reconstruct_joint_line`) receives the same SRTM profile as its DEM and never the targets.

```
pip install rasterio scipy matplotlib
python run_real_dem.py all          # downloads the DEM (16 MB) into data/, runs everything, writes the figures
```

| run | what | output |
|---|---|---|
| `lines` | five named lines (Yosemite, eastern escarpment, High Sierra, western foothills, Mono basin), five targets each | `results/lines5.json`, `results/lines5_*.npz` |
| `lines20` | the same lines with 20 targets every 750 m | `results/lines20.json`, `results/lines20_*.npz` |
| `random --n 200` | 200 random lines across the DEM (seed 1), both baseline sets (largest 300 m / 135 m) | `results/random_lines.csv` |
| `figures` | `report/figures/realdem_*.pdf` and `results/summary.json` | |

Results (highest ambiguity, dB relative to the peak of the reference, metric of the report):

| | no correction | joint | reference |
|---|---|---|---|
| 5 named lines, 5 targets | -8.2 to -5.7 | -41.6 to -41.0 | -41.6 |
| 5 named lines, 20 targets | -5.7 to -5.1 | -31.5 to -28.1 | -32.0 |
| 200 random lines (median) | -8.3 | -41.5 | -41.6 |

Tables of the runs: `report/data/realdem_lines.dat`, `report/data/realdem_random200.dat`. Report: section "Real terrain:
SRTM lines over the Sierra Nevada" of the Results chapter.
