# Joint reconstruction vs mountain height, and scenes full of point targets

The talk's default system is used everywhere:
- N_rx = 4 with Rx0 = Tx, 500 Hz per channel (2000 Hz together);
- b_xt = 0/300/119.8/100.5 m, λ = 0.25 m, v = 7689 m/s, r0 = 766 km, look angle 20°, B_a = 1235 Hz;
- data from `sata1d.generate_channels` (exact distances, no noise);
- the joint is `sata1d.joint_line.reconstruct_joint_line` (μ = 0.03 σ_max, 2 passes ≤ 60 iterations) with the true DEM.

The reference is one monostatic radar at the full PRF, limited to B_a. All lines are focused with the echo of the h = 0 point. 0 dB is the reference peak.

Code is in this folder (`common.py`, `item1_height.py`, `plot_item1.py`, `item3_dense.py`, `plot_item3.py`). The only repository dependency is `sata1d_code` (sata1d plus two helper modules of `joint_cases`); SRTM lines also need the DEM files (`DEM_DIR`). The run takes ≈ 40 s for item 1 and ≈ 4 min for item 3 on one core.

## Metric check (the talk's "highest ambiguity")

The talk metric (`audit_dem.ambiguity_level`, as used for every result of the report) is the largest |focused line| inside ±0.02 s (±154 m) of every predicted ambiguity position:
- the predicted positions are x_p + m·X with m = ±1, ±2 and X = λ r0 PRF_op /(2v) = 6.23 km;
- ±0.02 s (±154 m) around every true target is left out;
- the result is in dB relative to the peak of the band-limited reference.

Checks done:
1. **Positions.** In the standard reconstruction of the 1000 m mountain, every ambiguity peak above −25 dB sits within 0–2 m of a predicted x_p + m·X, so the ±154 m window is centred correctly. All targets lie on the iso-range of the reference point (`make_config`), so X is the same for every target. |m| ≤ 2 is complete, because 3·PRF_op = 1500 Hz exceeds the echo band (1281 Hz).
2. **Normalisation.** The grid maximum of the reference differs from the upsampled true peaks by ≤ 0.02 dB.
3. **Guard of ±154 m around the targets.** It is needed: without it the metric picks up the target's own sidelobes (the reference reaches −17.7 dB just outside its main lobe). It does not hide anything here, because no predicted ambiguity of one target falls within 154 m of another target in these scenes (placement rule of the talk).
4. **Band.** The reference and both reconstructions are compared in the same band B_a, so the comparison is consistent.
5. **What the metric does not see.** Artefacts that are not at the predicted positions. To cover them I added `floor` = the largest level anywhere on the line outside ±154 m of every target (away from the circular wrap at the ends). On the mountain sweep the joint’s floor stays within 1.6 dB of the reference’s floor (−37.4 dB) except where the ambiguity metric itself already flags a problem (2500, 2600, 2900 m, where floor = ambiguity). So no hidden artefacts were found.
6. **Recomputed slide-8 numbers.** Standard −4.38, joint −40.76, reference −41.44 dB, identical to the talk's −4.4 / −40.8 / −41.4 dB. **No bug found; the metric is kept as is.**

Two complementary numbers are now also reported:
- the **weakest true-target peak** (upsampled, re. reference peak): where the energy of the targets goes;
- the **3-dB width** of every main lobe (resolution).

## Item 1: highest ambiguity and resolution vs mountain height

The scene is slide 8: Gaussian mountain with σ = 1.5 km and peak height H. The five targets sit on the surface at x = −8, −0.55, 0.65, 2.9 and 4.1 km. Heights are taken relative to the mean of the line. H runs from 0 to 3000 m every 100 m (31 runs).

Figures:
- `fig/height_irf.pdf`: the terrains, plus standard and joint focused lines for H = 0, 500, 1000, 2000 and 3000 m.
- `fig/height_amb.pdf`: highest ambiguity vs H, and the weakest target peak vs H.

Results (full table in `cache/item1_table.md` / `.json`):
- **Standard.** −41.1 dB at 0 m (flat: it works), then −9.2 dB at 100 m. Above that it stays between −12.4 and −4.4 dB (median −6.1 dB). Its true peaks lose 2 to 28 dB, because the energy goes into the ambiguities.
- **Joint.**
  - **Agreement with the reference.** Within 3 dB of the reference (−41.4 dB) at 22 of 31 heights, and within 1 dB at 16.
  - **0–1500 m.** −41.7 to −40.4 dB.
  - **1600–2300 m.** −39.2 to −37.2 dB, i.e. 2–4 dB above the reference.
  - **Three bad heights.**

    | Height | Joint | Gap above reference | Blind-height mismatch |
    |---|---|---|---|
    | 2500 m | −33.4 dB | +8.1 dB | 0.27 rad |
    | 2600 m | −25.8 dB | +15.7 dB | 0.26 rad |
    | 2900 m | −28.5 dB | +12.9 dB | 0.36 rad |

    In all three the pair (target 4.1 km, its place −X) or (0.65 km, −X) is close to a blind height (mismatch as in `style.mismatch`).
  - **Mismatch is not sufficient.** It is a necessary-type condition only: 1900 m (0.20 rad) and 2800 m (0.09 rad) are not hurt (−38.0 and −37.9 dB).
  - **True peaks.** The joint keeps its true peaks at −0.03 dB at every height, and at worst −0.46 dB at 2600 m.
- **Resolution.** The 3-dB width of every main lobe is 5.77 m for the joint and for the reference, at every height and every target: **the joint does not widen the IRF.** The 5.77 m agrees with ≈ L_a/2 = 6 m.

Interpretation: the standard reconstruction is ruined by any relief of a few hundred metres. The joint follows the reference up to ≈ 1500 m. Above that it is a few dB worse, and it fails only near blind heights, which become more frequent as the relief grows: a higher mountain gives more height differences between a place and its aliases.

## Item 3: scenes full of point targets

Setup:
- **Targets.** N point targets with unit amplitude and random phase at random positions in |x| ≤ 10 km, on the surface, for N = 1, 5, 20, 50, 100 and 200. Three seeds each (seed = 1000·N + s, so every terrain sees the same scenes).
- **Why |x| ≤ 10 km.** Every ambiguity (±2X = ±12.5 km) then stays on the processed line (±23.6 km).
- **Simulated terrains:** isolated 1000 m mountain, random mountains (relief 1000 m) and fractal alpine relief (0–2000 m).
- **SRTM lines:** western foothills (slide 25 line), eastern escarpment (cut 6) and High Sierra (cut 3).
- In total 108 runs.

Metrics:
- **Image error** = 20 log10 ‖y − y_ref‖ / ‖y_ref‖ of the focused complex line over the whole line. It is valid at any density.
- **Highest ambiguity** (talk metric) and its **gap** to the reference's own value. With many targets the ±154 m guards cover more of the line, and the reference's own level at the ambiguity windows rises from −63 dB (1 target) to −30 dB (200 targets: sidelobes of neighbours). So for N ≥ 20 the gap, not the absolute level, is the meaningful number. The windows were never empty here; the code returns NaN if they are.

Figures:
- `fig/dense_irf.pdf`: mountain and western foothills with 20 and 200 targets.
- `fig/dense_summary.pdf`: image error vs N, and ambiguity gap vs N with near-blind runs in amber.

Results (`cache/item3_table.json`):
- **Image error.**
  - Standard: median +0.6 dB at every N. Its error is as large as the image itself, on simulated and on real terrain.
  - Joint: median −29.2 dB (range −34.5 to −17.5 dB). It is flat with N: −31.1 dB at N = 1 and −28.4 dB at N = 200. Simulated −28.8 dB, SRTM −29.3 dB.
- **Where the −30 dB floor comes from.** On flat ground (one target) both the standard and the joint have −35.3 dB, and lowering the regularisation (0.03 → 0.01 → 0.003 σ_max) makes the joint worse (−30.6, −29.4, −21.7 dB), not better. So the floor is the model error of K·H_i against the exact simulator (the talk's −30 to −35 dB), not the regularisation.
- **Ambiguity gap.**
  - Joint: median +1.5 dB; ≤ 3 dB in 77 of 108 runs; > 10 dB in 9.
  - Standard: median +27.3 dB, never below +19.3 dB.
  - For N = 1 the gap is ≤ 3 dB in only 5 of 18 runs, because the reference's level there is −63 dB. The joint's absolute levels are −55 to −62 dB, with two exceptions on the alpine terrain (−33.6 / −34.1 dB, mismatch 0.35 / 0.31 rad).
- **Failures (gap > 10 dB).**
  - The 9 failures:

    | Terrain | Number of targets (seed) | Gap above reference | Mismatch |
    |---|---|---|---|
    | Mountain | 5 (s1) | +20 dB | 0.09 rad |
    | Mountain | 20 (s2) | +24 dB | 0.11 rad |
    | Alpine | 1 (s0) | +29 dB | 0.35 rad |
    | Alpine | 1 (s1) | +29 dB | 0.31 rad |
    | Alpine | 50 (s2) | +14 dB | 0.06 rad |
    | Escarpment | 50 (s0) | +15 dB | 0.12 rad |
    | Escarpment | 100 (s0) | +13 dB | 0.13 rad |
    | Escarpment | 200 (s1) | +11 dB | 0.02 rad |
    | Escarpment | 200 (s2) | +11 dB | 0.03 rad |

  - All 9 have a target / alias pair within 0.4 rad of a blind height (min mismatch 0.02–0.35 rad).
  - The converse does not hold: 33 runs had mismatch < 0.4 rad and only 9 of them failed.
  - Density helps the failures happen only by putting more targets on the steep parts. The eastern escarpment (largest height jumps over X) fails at N ≥ 50; the foothills (gentle) never fail.
- **The foothills line** (gentle relief): joint gap ≤ 1.8 dB and error −31 to −34 dB at every density.

Interpretation: the joint does not get worse with the number of targets. Its image error is set by the model accuracy (≈ −30 dB), and the residual risk is the blind-height geometry, which depends on where the targets sit on the relief, not on how many there are.

