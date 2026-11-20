# close_targets_height_step — close point targets next to an abrupt height change

Run from this folder (needs the dlr-project repository: found by walking up to a folder with `sata1d_code/`, or set
`DLR_REPO`; uses `../joint_height_dense/common.py` for the system, generators, focusing, joint reconstruction and the
ambiguity metric):

    python3 -I item_close.py                    # 124 runs, ~2.5 min -> cache/close_*.npz, cache/close.json
    python3 -I plot_close.py                    # talk style   -> fig/close_maps*, fig/close_jump, fig/close_irf
    FIG_STYLE=report python3 -I plot_close.py   # report style (the figures in report/figures/analysis)
                                                # also cache/close_table.(json|md)

Files: `close_common.py` (local metrics), `subsample.py` (height change between the two DEM samples around a target),
`style.py` (plot style, copy of `../joint_height_dense/style.py`).

Scenes (default system, exact distances, no noise; heights relative to the mean height of the line):
- `step`: two unit targets a distance d apart, one on each side of a cliff of height dh,
  h(x) = dh/2 (1 + tanh(2 (x - xs)/w)), w = 10 m, xs = dx/2 (between two line samples, dx = v/PRF = 3.84 m);
- `bump`: the two targets on either side of a narrow bump of height dh, h(x) = dh exp(-(x - xs)^2/(2 s^2)), s = 10 m;
- `width`: the cliff at dh = 300 m with w = 1 m (sharper than one sample) and w = 50 m;
- `three`: three targets at xs - d, xs, xs + d (the middle one on the cliff edge), w = 10 m.
d = 6, 12, 18, 50, 100, 300, 1000, 3000 m; dh = 0, 10, 50, 100, 300, 1000 m. Target phases 0, pi/2 (, pi).

Metrics (dB re. the reference peak unless stated):
- highest ambiguity: the metric of joint_height_dense (largest level within +-154 m of every predicted ambiguity
  position x_p + m X, outside +-154 m of every target);
- peak error: largest |peak_joint - peak_ref| over the targets (upsampled x16);
- dip: lowest level between two neighbouring targets re. the weaker peak (upsampled); <= -3 dB: resolved;
- local error: 20 log10 ||y - y_ref|| / ||y_ref|| on the stretch from 200 m before the first to 200 m after the last
  target;
- dh_s (`subsample.py`): height change between the two line samples (dx = 3.84 m) around a target, maximum over the
  targets; its phase on the 300 m baseline is 2 pi b_xt dh_s / (lambda r0 tan theta) = 0.027 rad/m x dh_s.
