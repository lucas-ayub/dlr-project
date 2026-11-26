# baselines_solvers: time-varying baselines (item 2) and regularisation methods (item 5)

Findings and numbers are in `results.md`.

**Running the scripts.** Run them from this folder with `python3 -I <script>`. The repository root comes from `DLR_REPO`, or is found by walking up from this folder until `sata1d_code/` appears. `profiles.py` needs the SRTM GeoTIFF (folder from `DEM_DIR`, default `data/dem/` at the repository root) for the foothills scene.

| file | what it does |
|---|---|
| `common_m.py` | Spaceborne and airborne configs, tracks with a moving receiver, the GetCoeffNu wrapper and plot style. |
| `common_r.py`, `profiles.py` | Copies of the helpers of `../joint_height_dense`: the report's metric, terrains and focusing. |
| `item2_baseline_variation.py` | ΔC0/ΔC1/ΔC2 versus height for b_xt(t) = b_xt,0 + ε(t), plus time-domain residuals. Writes `fig/item2_panels_*` and `fig/item2_summary`. |
| `item2_joint.py` | Joint reconstruction of the slide-8 scene with a moving receiver, with and without MoCo. Writes `fig/item2_joint`. |
| `verify_phi_eps.py` | Derivation check of the residual phase φ_ε = (2π/λ) ε(t) h/(r0 tan θ0) against exact distances, every shape, spaceborne and airborne. Writes `cache/phi_eps_check.md`. |
| `verify_dC0.py` | Check of the topographic residual ΔC0 ≈ −b_xt h/(r0 tan θ) (report §2.4.1) against the exact parabola fits, b_xt = 300, 119.8, 100.5 m, h up to 3000 m. Writes `cache/dC0_check.md` (under a second). |
| `solvers.py` | Matrix-free solvers: current, L-curve (RelSea + Menger), discrepancy, BMR, MARIA, WISE. |
| `item5_regularisation.py` (`check` argument) | Comparison on 4 scenes × 4 SNRs. Check mode verifies that the current solver equals `reconstruct_joint_line`. |
| `plot_item5.py` | Writes `fig/item5_summary`, `fig/item5_lcurve`, `fig/item5_lines` and `cache/item5_table.md`. |

Run order: `item2_baseline_variation.py`, `item2_joint.py`, `item5_regularisation.py`, `plot_item5.py`. Together they take about 20 min on 2 CPUs.
