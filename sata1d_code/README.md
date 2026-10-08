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
  joint.py                  reconstruction inside the SATA sub-apertures (joint SATA + reconstruction)
plot_irf_single_target.py           IRF of one target: no SATA / whole band / per sub-band
plot_sata_windows_single_target.py  SATA window by window for one target (PDF)
run_geometry_sweep.py               sweep over target height profiles (incl. topographic ramp)
verify_sata1d.py                    verification tests T1-T19 (writes verification/)
run_one_case.py                     one scene, chosen modes and map, sequential (no multiprocessing); sweep-style figure
legacy/subband_beam_window.py       previous per-sub-band coefficient fit (not used by the pipeline)
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
python run_one_case.py --scene dense_ramp     # one scene (abc | one | dense_ramp | dense_hill | custom), sequential
python run_one_case.py --mode none joint oracle --map lut   # modes: none | whole | sub | joint | oracle; map: lut | footprint
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
   `mode="lut"` gives the look-up table (`lookup_table`): the scene is the targets, so
   the table holds each target's delta_C0 (full range history) at the target's cell,
   the nearest target's value in cells at most 2 cells from a target (a dense grid of
   targets fills its whole extent) and zero elsewhere; no fold images. The same table
   is used by the whole-band kernel and by every sub-band kernel.
   The delta_C0 values are computed once per target and channel (`delta_C0_per_target`)
   and shared by every map and every sub-band; every channel observes every target over
   the whole integration time, so there is no per-sub-band fit (the previous fit over
   the sub-band's look angles is kept in `legacy/subband_beam_window.py`).
   Output sub-bands are indexed by `cfg.Nsb` (= `Nrx` for the square reconstruction).
2. **kernel** (`sata_1d`): triangular sub-apertures with 50 % overlap; each FFT bin
   f is mapped to the cell `posaux = centre + r tan(asin(wl f / 2v)) / v * PRF_op`
   and multiplied by `exp(-j * (-2 pi / wl) * delta_C0[posaux])`; weighted
   overlap-add.

Per sub-band (`reconstruct_subband`): for every output sub-band k, the kernel labels
every bin with its absolute Doppler in [f_k − PRF_op/2, f_k + PRF_op/2) and measures
positions on the broadside grid (`subband_axis`), reads the same look-up table, and
sub-band k of the reconstruction uses only the channels corrected for k.

## Joint SATA + reconstruction (`joint.py`)

The per-channel kernels multiply every bin of a channel's sub-aperture spectrum by ONE
phase, but a channel is sampled at PRF_op, so every bin holds Nsb folded parcels coming
from Nsb ground positions X = r tan(asin(wl PRF_op / 2v)) / v · PRF_op cells apart
(6.23 km in the default geometry). On continuous terrain those positions have different
heights, no single phase serves all of them, and the parcels that should cancel in the
reconstruction leak as ambiguities (dense ramp 0 → 1000 m: no SATA −4.6 dB, look-up
table whole band −6.8 dB, per sub-band −4.7 dB, oracle −25.6 dB).

`reconstruct_joint` moves the fold bookkeeping into the reconstruction: for every
sub-aperture (same triangular windows, hop and zero-padding as the kernel) and every
bin phi of the Nrx channel spectra it solves

    s_i(phi) = sum_q H_i(f_q) exp(-j 2 pi dC0_i(x_q) / wl) u_q,   f_q = -Nsb PRF_op/2 + q PRF_op + phi,

for the Nsb parcels u_q, with H_i the reference-point filters of the standard
reconstruction, x_q = c + a(f_q) the ground position of parcel q (the kernel's
f → beta → x ruler) and dC0_i(x_q) read from the channel's look-up table
(`joint_maps`: the targets' values at their cells, the nearest target's value
elsewhere, no fold images). The parcels are placed at their true Doppler, transformed
back and overlap-added at the full PRF. With dC0 = 0 the result equals the standard
reconstruction (flat scene: max |difference| / peak = 1.7e-8); with one target it
equals the oracle (−63.6 dB); dense ramp 0 → 1000 m: −23.8 dB (oracle −25.6),
dense hill 1000 m: −22.0 dB (oracle −25.6), ramp 0 → 100 m: −25.6 dB (= oracle).
The remaining gap on steep terrain is the sub-aperture resolution: the energy of a
target spreads over ± ~20 cells of bins (window width and chirp drift within T_sub)
and the terrain changes within that spread.
