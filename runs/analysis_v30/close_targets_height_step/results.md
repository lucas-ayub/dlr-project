# Close point targets next to an abrupt height change: results

Scripts: `item_close.py` (124 runs), `plot_close.py`; table `cache/close_table.md`. Definitions in README.md.

## Findings

1. **Standard reconstruction**: its highest ambiguity depends on the height difference of the targets, not on their
   distance: cliff of 10 m: -37 to -34 dB; 50 m: -22 to -19 dB; 100 m: -16 to -13 dB; 300 m and 1000 m: -9 to -4 dB,
   for every d from 6 m to 3 km. With the bump (targets on flat ground for d >= 100 m) it stays at the reference level
   when both targets are on flat ground and fails when they are on the flanks (d <= 50 m).
2. **Joint reconstruction, targets on flat ground next to the cliff** (d >= 50 m, i.e. >= 25 m from a 10 m wide
   cliff): highest ambiguity -59 to -43 dB for every cliff height up to 1000 m (reference: -61 to -55 dB; the
   five-target mountain of the talk: -41 dB). A cliff between two neighbouring targets is not a problem in itself:
   the model has one DEM phase per place.
3. **Where it degrades**: when a target sits on the cliff face or on the bump flank, i.e. when the height changes by
   dh_s between the two DEM samples around it (one height per 3.84 m sample). dh_s <= 10 m: <= -39 dB (one exception,
   -33 dB: target 1 m from a 1 m sharp 300 m cliff, the cliff inside its main lobe); dh_s <= 115 m: <= -33 dB;
   all 8 failures (> -30 dB) have dh_s >= 135 m, i.e. a phase of >= 3.6 rad on the 300 m baseline: targets on a
   1000 m cliff face 10 m wide (-3 dB), on the flank of a 1000 m bump (-22 to -8 dB) and the middle target on the edge
   of a 1000 m cliff (-5 to -3 dB). The middle target on a 300 m cliff edge gives -36 to -34 dB, on a 100 m edge
   -47 to -44 dB.
4. **Resolution**: two targets 6 m apart are not separated, by the reference either (dip -0.6 dB; main lobe 5.77 m).
   From 12 m on, the reference dip is -26 dB and the joint's is -33 to -24 dB in every run that does not fail; the
   joint keeps the separation of the reference.
5. **Peaks**: in the 116 runs that do not fail the joint's target peaks are within 0.36 dB of the reference (median
   0.005 dB) and within 0.24 m of their position (d > 2 cells); in the 8 failures up to 3.8 dB off.
6. **Blind heights** are not the cause here: the smallest mismatch over all runs is 0.55 rad (bump, d = 12 m,
   dh = 1000 m), far from the < 0.4 rad of the blind-height failures of joint_height_dense.

Interpretation: the joint reconstruction handles an abrupt height change between close targets as long as every
target sits where the DEM is locally flat at the scale of one sample (3.84 m). The limit is the sampling of the DEM
phase, not the cliff: a target on a face whose height changes by ~100 m within one sample gets a channel phase that no
single DEM sample carries (0.027 rad/m on the 300 m baseline). A finer DEM grid (oversampling the line) or a DEM
height per target would remove it; real DEMs (SRTM: 30 m posting) are smoother than these cliffs.

## Numbers used in the report and the talk
- standard, cliff: -37 to -34 dB (10 m), -22 to -19 dB (50 m), -16 to -13 dB (100 m), -9 to -4 dB (300, 1000 m);
- joint, targets >= 25 m from the cliff: -59 to -43 dB for dh up to 1000 m; reference -61 to -55 dB;
- failures: 8 of 124 runs, all with dh_s >= 135 m (>= 3.6 rad); dh_s <= 115 m: <= -33 dB; dh_s <= 10 m: <= -39 dB
  except one -33 dB case;
- resolution: d = 6 m unresolved also by the reference (dip -0.6 dB); d >= 12 m: reference -26 dB, joint -33 to -24 dB;
- peaks within 0.36 dB of the reference in the 116 runs that do not fail.
