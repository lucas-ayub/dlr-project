# Joint reconstruction on the whole line — standalone

Only what `reconstruct_joint_line` needs, copied from `../sata1d` (no changes in the code; only the imports point to
this package, and `channel_maps`, which serves the sub-aperture methods, was left out). Needs `numpy` (and
`matplotlib` for the example figure).

```
joint1d/
  config.py          make_config, SystemParams, Scene, ArrayGeometry, ExperimentConfig   (copy of sata1d/config.py)
  geometry.py        build_platform_tracks                                              (copy)
  signal_model.py    getRawData1D, generate_reference, generate_channels               (copy)
  reconstruction.py  GetCoeffNu (C0, C1, C2, Dt) and the standard reconstruction        (copy)
  coefficients.py    reference_coefficients (from sata1d/joint.py), residual_C0 (from sata1d/sata.py)
  joint_line.py      line_maps (DEM -> delta_C0), LineModel (model and adjoint), reconstruct_joint_line (CGLS)
run_example.py       5 targets on a 1000 m mountain: Reference, no SATA, joint final version; saves joint_example.png
check_vs_package.py  same numbers as ../sata1d (run from here, with ../sata1d present)
```

Use:

```python
import joint1d as J
dC0 = J.line_maps(cfg, tracks, terrain)                       # terrain: h(x) [m]; DEM only
line = J.reconstruct_joint_line(cfg, tracks, channels, dC0)   # raw line at the full PRF, as reconstruct()
```

Steps inside (see the guides in `../joint_cases/guia_joint_linha/`): P1 `reference_coefficients` -> H_i;
P2 `generate_reference` -> K, K_i = K H_i; P3 `line_maps` -> screen exp(-j 2 pi dC0 / wl); then CGLS on
min_g sum_i |s_i - D H_i K Psi_i g|^2 + mu^2 |g|^2 (two passes), and the output U = K g.
