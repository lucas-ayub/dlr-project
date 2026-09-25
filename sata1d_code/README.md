# SATA 1D — multichannel azimuth reconstruction with topography correction (C0)

## Contents

```
sata1d/                     package
  config.py                 system, scene, array; make_config() (DPCA bat, bxt ~ U(0, bxt_max))
  geometry.py               transmitter / receiver tracks
  signal_model.py           point-target raw data, reference and channels
  reconstruction.py         coefficients (GetCoeffNu), inversion filters, reconstruction
  sata.py                   SATA whole band: delta_C0 map + kernel sata_1d + sata_channels
  subband.py                SATA per sub-band + sub-band reconstruction
plot_irf_single_target.py           IRF of one target: no SATA / whole band / per sub-band
plot_sata_windows_single_target.py  SATA window by window for one target (PDF)
run_geometry_sweep.py               sweep over target height profiles (incl. topographic ramp)
verify_sata1d.py                    verification tests T1-T19 (writes verification/)
plot_sweep_improvement_heatmaps.py  heatmaps of the SATA improvement over no SATA from the sweep CSV
```

## Requirements

Python 3.10+, numpy, matplotlib, pillow (`pip install -r requirements.txt`).

## Run

```
python plot_irf_single_target.py
python plot_sata_windows_single_target.py
python run_geometry_sweep.py --quick          # a few cases
python run_geometry_sweep.py --workers 4      # full sweep (362 cases)
python verify_sata1d.py                       # tests T1-T19 (~15 min)
python plot_sweep_improvement_heatmaps.py     # after the full sweep
python run_geometry_sweep.py --families topo_ramp --bxt 100   # topographic ramp only
```

## SATA in short

For each channel:

1. **delta_C0 map** (`build_delta_C0_array`): for each scatterer, `residual_C0` =
   C0(scatterer) − C0(reference point) from the range-history fit. The value is
   written where the scatterer appears in the sub-aperture spectra: its cell
   (`az_pixel_of_scatterer`, including the bat/2 phase-centre shift) ± the
   half-width of one sub-aperture response (`sata_footprint_halfwidth`), and the
   same block at the folded-Doppler offsets (`sata_image_offsets`). Neighbouring
   scatterers whose blocks touch are linearly interpolated; where two images
   overlap the nearest one wins; zero elsewhere.
   `mode="hold"` gives the previous map (value held over the whole line).
2. **kernel** (`sata_1d`): triangular sub-apertures with 50 % overlap; each FFT bin
   f is mapped to the cell `posaux = centre + r tan(asin(wl f / 2v)) / v * PRF_op`
   and multiplied by `exp(-j * (-2 pi / wl) * delta_C0[posaux])`; weighted
   overlap-add.

Per sub-band (`reconstruct_subband`): for every output sub-band k, delta_C0 is
fitted over the look angles of that sub-band, the kernel labels every bin with its
absolute Doppler in [f_k − PRF_op/2, f_k + PRF_op/2) and measures positions on the
broadside grid (`subband_axis`), and sub-band k of the reconstruction uses only the
channels corrected for k.
