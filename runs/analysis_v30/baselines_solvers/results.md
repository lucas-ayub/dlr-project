# Time-varying baselines (item 2) and regularisation methods (item 5)

The scripts are in this folder. They import the repository's `sata1d` package read-only. `common_r.py` and `profiles.py` are copies of the helpers of `../joint_height_dense`, so the metric is exactly the one of the report.

| script | what it does | output |
|---|---|---|
| `item2_baseline_variation.py` | C0/C1/C2 versus height, for straight and moving receivers, spaceborne and airborne | `cache/item2.json`, `cache/item2_table.md`, `fig/item2_panels_{space,airX,airL}`, `fig/item2_summary` |
| `item2_joint.py` | Joint reconstruction of the slide-8 scene with one moving receiver | `cache/item2_joint.json`, `fig/item2_joint` |
| `item5_regularisation.py` (methods in `solvers.py`) | Seven ways of solving s = A g, on four scenes, at four noise levels | `cache/item5.json`, `cache/item5_lines_*.npz` |
| `plot_item5.py` | Figures and tables for item 5 | `fig/item5_summary`, `fig/item5_lcurve`, `fig/item5_lines`, `cache/item5_table.md`, `cache/item5_ref.json` |

Every figure is saved as both .pdf and .png. Run time on 2 CPUs: item 2 takes about 3 min, `item2_joint` about 2 min, item 5 about 12 min.

---

## Item 2: when the cross-track baseline changes during the aperture

### Method

The computation is the same as in the deck's slide 15 (`GetCoeffNu`: parabola fits of the bistatic-minus-monostatic range history over the common aperture). The only change is that receiver 1 sits at b_xt(t) = b_xt,0 + ε(t), using exact distances.

**Shapes of ε(t).** Times are relative to the reference point's closest approach, and T is one aperture.
- **bumps**: three smooth bumps of height ε, each 0.08 T wide. This is the "a few instants" case.
- **step**: a smooth step of height ε at +0.2 T.
- **drift**: linear, from −ε at −T/2 to +ε at +T/2.
- **sine**: ε·sin(2π t/(T/2)), two oscillations per aperture.

**Systems.**

| | λ | r0 | v | B_a | aperture | b_xt,0 | heights |
|---|---|---|---|---|---|---|---|
| Spaceborne (the deck) | 0.25 m | 766 km | 7689 m/s | 1235 Hz | 2.08 s | 300 m | 0–3000 m |
| Airborne X band | 3.1 cm | 4243 m (3000 m altitude, 45° look) | 90 m/s | 360 Hz (0.5 m antenna) | 2.9 s | 2 m | 0–600 m |
| Airborne L band | 23 cm | 4243 m | 90 m/s | 180 Hz (1 m antenna) | 10.8 s | 2 m | 0–600 m |

The airborne values are typical of DLR's F-SAR (X and L band, Do 228 platform at about 90 m/s, a few km altitude; Reigber et al., Proc. IEEE 101(3), 2013). The antenna lengths and the 2 m baseline are representative choices, not F-SAR specifications.

**Correction to how the shift is measured.** With a moving receiver, the closest-approach times of the raised point and of the reference point fall on different samples. `Dt` and `C1` then both jump. The filter uses (Δt − C1)·f, so the shift curve shows Δ(Δt − C1). Plotting ΔC1 alone gave non-monotonic, wrong values.

The `GetCoeffNu` fits also become unreliable for large disturbances; for example, the X-band 1 m step returns a ΔC0 phase of 27.8 instead of about 48. For that reason the robust numbers are computed in the time domain from exact distances, on the reference aperture:
- **"all but ΔC0"** = 2π/λ · max|d(t) − d(t_c)|, where d(t) = [r_Rx − r_Tx](h) − [r_Rx − r_Tx](0). This is everything a model with ΔC0 alone ignores.
- **"not a parabola"** = what is left after a parabola fit of d(t). Even a model with all of C0, C1 and C2 cannot describe this part.

### Findings

**Straight tracks (constant baseline).** The deck's statement holds.

| | ΔC0 term | everything else together | ratio C0/C2 |
|---|---|---|---|
| Spaceborne, h = 3000 m | 79.7 rad | 4.0·10⁻³ rad | 2·10⁴ |
| Airborne X, h = 600 m | 47.7 rad | 2.1·10⁻² rad | 2.1·10³ |
| Airborne L, h = 600 m | 6.4 rad | 3.9·10⁻² rad | 1.5·10² |

On straight tracks, airborne geometries make C2 relatively 10–100× more important than spaceborne, but C1 and C2 stay far below π/4.

**A moving receiver.** The part that ΔC0 ignores grows linearly with the change of the baseline inside the aperture:

  φ ≈ (2π/λ) · max_t|ε(t)| · h / (r0 tan θ0)    (derivation and check against exact distances: `verify_phi_eps.py`, table in `cache/phi_eps_check.md`; within 2 % at 1000 m and 5 % up to 3000 m spaceborne, 3–10 % high airborne, 22 % for the drift; earlier check: spaceborne drift of 1 m at h = 3000 m gives 0.27 predicted and 0.26 computed; X-band drift of 1 cm at 600 m gives 0.29 predicted and 0.25 computed)

- **How the shape splits it:**
  - a linear drift goes almost entirely into the shift (C1);
  - a step splits about equally between C1, C2 and "not a parabola";
  - oscillations and bumps are mostly "not a parabola", so adding C1 and C2 to the model would not capture them.
- **When it exceeds π/4 at the band edge:**

  | system | crosses π/4 at | rule |
  |---|---|---|
  | Spaceborne, h = 3000 m | ε ≈ 3 m | ε·h > λ r0 tanθ / 8 ≈ 8.7·10³ m² |
  | Airborne X, h = 600 m | ε ≈ 3 cm | ε·h > 16 m² |
  | Airborne L, h = 600 m | ε ≈ 20 cm | ε·h > 120 m² |

  Airborne motion deviations of decimetres to metres are common (turbulence). So yes: in airborne systems the time variation of the baseline makes terms beyond ΔC0 matter. Strictly speaking, the cause is the baseline varying in time, not airborne geometry itself; a straight airborne track still has negligible C1 and C2. And much of the effect is not a parabola at all.

**What it does to the joint reconstruction** (`item2_joint.py`).

Setup:
- slide-8 scene, deck system;
- the b_xt = 300 m receiver moves;
- channels are simulated with the true track;
- first-order MoCo to the reference height (h = 0, line of sight at the beam centre), then the repository's model and solver.

Checks of the MoCo itself:
- On flat terrain the MoCo is exact: a 2 m drift still gives −41.0 dB.
- Without MoCo, the straight-track model fails: ambiguities rise to between −15 and +5 dB.

On the 1000 m mountain after MoCo:

| ε | 0 | 0.25 m | 0.5 m | 1 m | 2 m | 5 m | 10 m |
|---|---|---|---|---|---|---|---|
| drift | −40.8 | −40.3 | −37.3 | −31.5 | −25.2 | −17.1 | −11.1 dB |
| sine | −40.8 | −40.6 | −37.1 | −31.6 | −25.6 | −17.4 | −11.5 dB |
| bumps | −40.8 | −40.8 | −39.7 | −37.5 | −34.3 | −21.5 | −17.1 dB |

The reference is −41.4 dB. The level follows **highest ambiguity ≈ 20 log10 φ − 9.5 dB**, where φ is the phase the MoCo leaves on the highest target (0.02 rad at 0.25 m, 0.08 rad at 1 m).

So **π/4 is far too loose as a tolerance for −40 dB ambiguity suppression**: φ = π/4 gives about −12 dB, and keeping the joint at the reference level needs φ ≲ 0.02 rad. The deck's conclusion for straight tracks is unaffected, because there C1 and C2 are at most about 4·10⁻³ rad. But the deck's "π/4 tolerance" line should not be read as "anything below π/4 is harmless".

At φ ≈ 0.02 rad, the uncompensated baseline variation allowed for 1000 m of relief is about 0.2 m (spaceborne). For 100 m of relief it is 4 mm in X band and 3 cm in L band (airborne). Airborne use of the joint method would therefore need a topography-dependent MoCo, i.e. a ΔC0 that varies over the aperture. That would replace the per-place diagonal screen with a space-variant filter; this is the problem treated in the literature on topography- and aperture-dependent motion compensation (e.g. P. Prats, K. A. Camara de Macedo, A. Reigber, R. Scheiber, J. J. Mallorqui, "Comparison of topography- and aperture-dependent motion compensation algorithms for airborne SAR", IEEE GRSL 4(3):349–353, 2007, doi 10.1109/LGRS.2007.895712).

---

## Item 5: inversion and regularisation methods from the three papers

### The papers, briefly

1. **RelSea.** D. González-Caboverde, G. D. Martín-del-Campo-Becerra, D. L. Torres-Román, E. Torres-García, "RelSea: A Reliable L-Curve Search Region to Improve Regularization Parameter Selection", IEEE Access 14, 152884+, 2026, doi 10.1109/ACCESS.2026.3739335.
   - The parameter α is chosen at the corner of the L-curve (log‖Ax − y‖², log‖x‖²).
   - RelSea restricts the search to a reliable convex region inside the spectral interval [σ_R, σ_1] (the curvature formula κ(α), eq. 20).
   - Inside that region the corner is found by maximum Menger curvature (eq. 13) with a golden-section search (Algorithms 1 and 3; "EMA" and "MOTA" are the enhanced Menger-curvature and θ-curve variants).
2. **Statistical regularisation.** G. Martín del Campo, M. Nannini, A. Reigber, "Statistical Regularization for Enhanced TomoSAR Imaging", IEEE JSTARS, 2020, vol. 13, pp. 1567–1589, doi 10.1109/JSTARS.2020.2970595.
   - It compares deterministic constrained least squares (CLS = Tikhonov, eq. 15) and its weighted form (WCLS, eq. 22) with statistical estimators.
   - The Bayes-minimum-risk operator is F = R_s A^H (A R_s A^H + R_n)^-1 (eq. 31). For R_s = S0 I and R_n = N0 I it is Tikhonov with α = N0/S0 (eq. 59).
   - It also gives the ML-inspired MARIA (Table II) and the new WCF-based WISE (Table III). Both are iterative re-weightings of the power b with a projector P_γ.
3. **MARIA.** G. Martín del Campo, M. Nannini, A. Reigber, "Towards Feature Enhanced SAR Tomography: A Maximum-Likelihood Inspired Approach", IEEE GRSL 15(11), 1730–1734, 2018, doi 10.1109/LGRS.2018.2858571.
   - Fixed point b ← P_γ{T^-1 (v − w)} (eq. 16), with the sufficient statistics F = D(b) A^H R_y^-1 (eq. 13).
   - It starts from matched-filter (MSF) beamforming, and N0 acts as diagonal loading.

### How each method is applied to s = A g

Everything is matrix-free; only `LineModel.forward` and `adjoint` are used.
- **current**: the repository's solver: μ = 0.03 σ_max, CGLS, 2 passes. It is identical to `reconstruct_joint_line` (difference 0).
- **lcurve / lcurve2**: the RelSea search region is the convex part of a 9-point log grid over [10⁻⁴, 1]·σ_max, followed by Algorithm 1 inside it (Menger curvature + golden section). Each L-curve point is a full Tikhonov solve. lcurve uses 1 pass, lcurve2 uses 2.
- **discrepancy**: Morozov's principle, ‖A g − s‖² = M N0 (bisection on log μ). It is used only with noise.
- **bmr**: μ² = N0/S0, with S0 = (‖s‖² − M N0)/tr(AᴴA) and tr(AᴴA) from 16 Hutchinson probes. Without noise, μ = 0, so it reduces to CGLS with early stopping.
- **maria / wise**: single look (Y = s sᴴ):
  - MARIA: b_m ← |a_mᴴR⁻¹s|² b_m / (a_mᴴR⁻¹a_m);
  - WISE: b_m ← tr(Y)|a_mᴴR⁻¹s|² b_m / (a_mᴴa_m);
  - complex estimate g = D(b)AᴴR⁻¹s, obtained as g = W u with u = argmin‖AWu − s‖² + N0‖u‖² and W = D(b)^½;
  - a_mᴴR⁻¹a_m uses the matched-filter approximation d/(d b_m + N0), with d = a_mᴴa_m (the same for every place);
  - zero step b0 = |Aᴴs|²/d². This is the MSF of the 2018 letter; Capon needs many looks;
  - γ = 0, at most 10 iterations, stop when the relative change is below 10⁻³;
  - N0 is the true noise power in noisy runs. Without noise it is set to (0.03 σ_max)²·mean(b0), the same loading as the current method.

### Scenes and metrics

**Scenes:**
1. the slide-8 mountain (5 targets);
2. 40 random-phase targets about 1 km apart on the mountain line;
3. the SRTM western foothills with 20 targets 750 m apart (the deck's 20-target line);
4. a distributed-like scene: 1500 scatterers with complex Gaussian amplitudes, about 27 m apart, on the mountain line.

**Noise:** white complex Gaussian noise added to the channels, with SNR = mean channel power / N0 = ∞, 30, 20, 10 dB. Seeds 0 and 1 are run; the tables show medians.

**Metrics** (from `common_r.py`, the snapshot of R's helpers):
- **highest ambiguity**: the deck's metric (`amb_old`, ±154 m windows), relative to the noise-free reference peak.
- **floor**: the highest level outside the targets.
- **peak loss**: the weakest target peak.
- **NMSE**: the error of the focused complex line versus the reference. It is the only metric used on the distributed-like scene.

The 3-dB width is the same for every method (5.77 m), because the output is always limited to B_a with the chirp.

### Results

Highest ambiguity in dB (scene 4: NMSE in dB):

| scene | SNR | standard | **current** | L-curve | L-curve, 2 passes | discrepancy | BMR | **MARIA** | WISE | reference |
|---|---|---|---|---|---|---|---|---|---|---|
| mountain | ∞ | −4.4 | −40.8 | −39.6 | −34.7 | – | −39.6 | −41.5 | −41.6 | −41.4 |
| | 30 | −4.4 | −37.6 | −34.6 | −31.0 | −23.1 | −37.9 | −41.5 | −41.0 | |
| | 20 | −4.4 | −30.0 | −25.4 | −26.7 | −14.9 | −26.9 | −41.2 | −34.9 | |
| | 10 | −4.4 | −20.7 | −15.7 | −17.3 | −13.0 | −15.1 | −35.6 | −20.0 | |
| 40 targets | ∞ | −4.7 | −32.4 | −31.0 | −28.5 | – | −30.9 | −34.5 | −34.4 | −34.2 |
| | 20 | −4.7 | −23.2 | −20.4 | −16.2 | −20.2 | −27.5 | −34.5 | −28.9 | |
| | 10 | −4.8 | −13.9 | −10.7 | −6.5 | −15.8 | −20.4 | −27.5 | −12.8 | |
| foothills, 20 targets | ∞ | −6.0 | −34.0 | −34.0 | −34.1 | – | −34.0 | −34.7 | −34.1 | −34.4 |
| | 20 | −6.0 | −30.5 | −30.5 | −30.7 | −24.3 | −30.8 | −34.7 | −30.6 | |
| | 10 | −6.0 | −23.9 | −22.5 | −22.9 | −18.6 | −23.6 | −28.0 | −22.3 | |
| 1500 scatterers (NMSE) | ∞ | −0.5 | −29.8 | −29.6 | −28.9 | – | −29.6 | −25.9 | −31.8 | |
| | 20 | −0.5 | −19.3 | −18.4 | −17.1 | −16.4 | −19.9 | −23.8 | −20.2 | |
| | 10 | −0.2 | −9.7 | −8.5 | −7.4 | −8.8 | −11.2 | −15.1 | −9.5 | |

The full table, with SNR 30, the floors, peak losses, μ/σ_max, numbers of A and Aᴴ calls and run times, is in `cache/item5_table.md`.

**Cost.** For a 12288-place line on the noise-free mountain, counted as A + Aᴴ calls:

| method | calls | time |
|---|---|---|
| current | 266 | 0.5 s |
| BMR | 161 | 0.3 s |
| MARIA | 581–1235 | 1–4 s |
| WISE | 1235 | 2–4 s |
| L-curve | about 2400 | 4–7 s |
| L-curve, 2 passes | about 4800 | 8–14 s |

### Interpretation

1. **Without noise (the setting of the report's other results), the linear methods are within about 1–2 dB of each other, except the two-pass L-curve (up to 6 dB worse); MARIA and WISE are up to 2 dB better on point targets.** The current choice, μ = 0.03 σ_max with 2 passes, is the best of the Tikhonov family.
   - **The L-curve is not useful here.** It has no clear corner: it is almost flat, and the "corner" sits at the small-μ end of the RelSea region. The selected μ (≈ 10⁻³ σ_max) under-regularises, so the line is 1–6 dB worse, and the search costs about 10× more.
   - RelSea's own text says that concave L-curves or L-curves without a corner give no guidance, which is what happens here.
2. **With noise, the parameter-choice rules from the papers do not beat μ = 0.03 σ_max:**
   - **L-curve**: under-regularises (μ ≈ 3·10⁻⁴–1·10⁻³ σ_max) and amplifies noise.
   - **Discrepancy principle**: over-regularises (μ ≈ 0.08–0.4 σ_max). Ambiguities are no longer cancelled (−15 to −30 dB), and targets lose up to 2–5 dB.
   - **BMR (μ² = N0/S0)**: close to the current method (±3 dB, better on 40 targets at SNR 20). It needs the noise power.
3. **MARIA is clearly the best on point-target scenes with noise.**

   | scene, SNR 20 dB | current | MARIA |
   |---|---|---|
   | mountain | −30.0 | −41.2 dB (at the reference level down to SNR 20 dB) |
   | 40 targets | −23.2 | −34.5 dB |
   | foothills | −30.5 | −34.7 dB |

   - Its NMSE is 13–16 dB better at SNR 20 dB.
   - It costs 2–5× the current method.
   - **Why it wins:** its adaptive weights D(b) put the energy back on the places that hold scatterers, so the noise at the empty places, including the ambiguity positions, is suppressed. Its prior is "sparse power".
   - **Limits:**
     - (a) On the noise-free distributed-like scene it is worse than the current method (NMSE −25.9 vs −29.8 dB); the 1500 scatterers are still sparse relative to the 3.84 m place spacing, so a truly distributed scene would be harder. Only at SNR ≤ 20 dB does it win there too.
     - (b) In noise it creates isolated spikes at about −45 dB far from the targets (visible in `fig/item5_lines`): noise samples that get promoted to "scatterers".
     - (c) a_mᴴR⁻¹a_m is the matched-filter approximation. A Hutchinson estimate of the exact diagonal (16 probes, CG on R) was tried; it was too noisy and the iteration diverged to −6 dB. So the exact MARIA was not tested.
     - (d) This is a single-look use of estimators that the papers derive for multi-look covariance matrices, with an MSF start instead of Capon.
4. **WISE** is good without noise or at 30 dB (−41.6 / −41.0 dB on the mountain). It degrades faster than MARIA, presumably because its update scales with tr(Y) = ‖s‖², which includes the noise.

### Recommendation

- **For the report and the deck (noise-free simulations):** keep the current solver, μ = 0.03 σ_max with 2 passes, as the reference method. It is the cheapest, it needs no knowledge of noise or scene, and none of the parameter-choice rules (L-curve with RelSea, discrepancy, BMR) improves on it.
- **As an outlook:** sparsity-promoting statistical regularisation (MARIA) is the one method that gains, by about 4–15 dB, on point-target scenes in noise. It deserves a mention, with the distributed-scene and spurious-spike caveats.
- **Noise:** the deck's numbers are noise-free. At SNR 20 dB even the current joint method loses about 10 dB on the mountain (−40.8 → −30.0), because noise lands at the ambiguity positions. This is worth stating in the report's limitations.
