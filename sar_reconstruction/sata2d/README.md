# `sata2d` — SATA (2-D) with 3-D geometry (along- and cross-track baselines + topography)

Three library files, plus the run scripts — the two-step Range-Doppler
reconstruction, the corrected sub-band SATA kernel, and the 3-D geometry
extension (along-track _and_ cross-track baselines, real target topography),
merged down from ten smaller files so the package reads closer to the
original three-file codebase it started from.

SATA itself is unchanged and is a 2-D (range x azimuth) correction; what is
3-D here is the _geometry_ it corrects for — along-track baseline,
cross-track baseline and target height, rather than a flat along-track-only
orbit.

---

## 1. How to run

```bash
pip install numpy scipy matplotlib

cd sar_reconstruction   # the directory that CONTAINS this package folder
# --- main experiments -----------------------------------------------------
python -m sata2d.run_sata2d_topo --plots   # main experiment, 3-D geometry
python -m sata2d.run_sata_irf              # playground: one target, ONE method
python -m sata2d.run_sata_irf_all          # playground: one target, all THREE

# --- impulse response -----------------------------------------------------
python -m sata2d.run_irf2d                     # 2-D IRF, monostatic reference
python -m sata2d.run_irf2d --method sub        # 2-D IRF, SATA per sub-band

# --- diagnostics ----------------------------------------------------------
python -m sata2d.run_check && python -m sata2d.plot_esr   # error budget
python -m sata2d.run_axes_2d                   # slow/fast time, 2-D spectrum
python -m sata2d.run_c1c2                      # C1/C2 residuals + oracle
python -m sata2d.run_bxt_test                  # bxt sweep

# --- range co-registration study ------------------------------------------
python -m sata2d.run_coreg --stage equiv       # the 2x2 equivalence matrix
python -m sata2d.run_coreg --stage pipeline    # full pipeline comparison
python -m sata2d.plot_coreg                    # its three figures
python -m sata2d.run_irf2d --method sub --coreg
```

All the study scripts default to the realistic array: along-track baselines on
the **DPCA condition** (`arrays.make_params_dpca`, `dx = 2*vs/PRF = 7.6885 m`,
so the effective phase centres are uniformly spaced and the multichannel
sampling is exact) and cross-track baselines drawn as `bxt ~ U(0, bxt_max)`,
set with `--bxt-max` (default 100 m) and `--seed`. Pass `--bxt-mode linear`
for the symmetric ladder `bxt_i = dxt*(i-(Nrx-1)/2)`, a deliberate worst case.
Every script writes its figures into `plots/` next to this file.

Every script also runs directly (`python run_sata_irf_all.py`, no `-m`, from
inside the folder) and works whatever the folder itself is named — e.g.
`sata_2d` — since each script derives its own package name from the
directory it is actually running in, rather than assuming a fixed name.

### `run_sata2d_topo.py` — main experiment

Self-test, residual (`dC0`) table, a sweep over array/baseline/topography
cases, and the azimuth-topography experiment (five targets at different
azimuths and heights, the case a single global correction cannot fix).

| flag            | effect                                                                           |
| --------------- | -------------------------------------------------------------------------------- |
| _(none)_        | full run: self-test, residual table, 9-case sweep, azimuth topography (~4 min)   |
| `--quick`       | three sweep cases instead of nine (~1 min)                                       |
| `--skip-sweep`  | straight to the azimuth-topography experiment (~20 s)                            |
| `--multi-range` | also repeat the azimuth-topography case at three reference ranges (near/mid/far) |
| `--plots`       | write the figures                                                                |
| `--outdir DIR`  | where the figures go (default `plots/sata2d`)                                    |

### `run_sata_irf.py` / `run_sata_irf_all.py` — IRF playground

Two smaller scripts for quick, single-target experimentation rather than a
full run: both build the same standard geometry as above (`wl=0.25 m`,
`H=720 km`, `r0=766.21 km`, `PRF=2000 Hz`, `Nrx=4`), place **one** target at
a chosen azimuth offset and height, and reconstruct it. Only the `TARGET
GEOMETRY` block at the top of each file is meant to be edited — everything
else stays at the same defaults, so runs stay directly comparable to each
other and to `run_sata2d_topo.py`.

```bash
python -m sata2d.run_sata_irf --method sub    # one method, one IRF
python -m sata2d.run_sata_irf_all             # all three, one plot
```

- `run_sata_irf.py` reconstructs with a single method (`--method
no|whole|sub`, default `sub`) and plots _ref_ against _rec_, written to
  `plots/sata_irf.png`.
- `run_sata_irf_all.py` reconstructs all three (no SATA / whole band / per
  sub-band) and overlays all four curves against the monostatic reference,
  written to `plots/sata_irf_all.png`.

Both write into a `plots/` folder next to the script itself, created
automatically if it does not exist yet — regardless of the directory you run
the command from. Pass `--out` to write somewhere else instead. Both plot
the azimuth impulse response in dB over the **full** azimuth time axis, with
a parameter box (`Nrx`, `PRF`, `Ba`, `Δb_at`, `bxt_max`). They share their
build/reconstruct/plot machinery (the last section of `reconstruction.py`),
so a change to one script's plotting style is easy to mirror in the other.

For a single target the residual is constant across azimuth, so whole-band
and per-sub-band SATA agree almost exactly — the per-sub-band gain only
shows up with topography that varies _along azimuth_ (several targets at
different azimuths), which is what `run_sata2d_topo.py`'s azimuth-topography
experiment demonstrates.

---

## 2. What is in here

| file                  | role                                                                                                                                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `geometry.py`         | acquisition parameters and array geometry (`bat`/`bxt`/topography), tracks, coefficients (analytic `get_coeff_nu` and numeric `get_coeff_nu_3d`), and raw-data generation for the scene                            |
| `sata.py`             | the corrected **SATA kernel** (sub-aperture STFT, weighted overlap-add) — 2-D, unchanged by the geometry it is fed                                                                                                 |
| `reconstruction.py`   | the **two-step Range-Doppler reconstruction** (STEP 1 reference filter, STEP 2 per-range-bin residual + RCMC), SATA applied per sub-band to the 3-D geometry, plus the shared IRF-playground machinery — see below |
| `run_sata2d_topo.py`  | the main experiment + plots                                                                                                                                                                                        |
| `run_sata_irf.py`     | **playground**: one target, **one** method, one IRF plot                                                                                                                                                           |
| `run_sata_irf_all.py` | **playground**: one target, the **three** methods together, one IRF plot                                                                                                                                           |
| `run_irf2d.py` | **2-D impulse response**: range x azimuth contour + the two 1-D cuts, Sakar Fig. 2.9 style. `--coreg` uses the explicit co-registration pipeline |
| `arrays.py` | the **DPCA condition**: `dpca_dx`, `dpca_prf`, `dpca_residual` and `make_params_dpca`, used by every study script so the along-track sampling is uniform |
| `coreg.py` | the **range co-registration** term: derivation, the ramp, the monochromatic-filter context manager, `reconstruct_explicit_coreg` |
| `run_coreg.py` | the co-registration experiment (`--stage equiv` / `--stage pipeline`) |
| `plot_coreg.py` | its figures, from the cached results |
| `run_check.py` + `plot_esr.py` | error-budget diagnostics: error-to-signal in 2-D and vs Doppler, azimuth IRF, main-lobe zoom |
| `run_axes_2d.py` | which axis is which: azimuth IRF (slow time), range IRF (fast time), 2-D spectrum |
| `run_c1c2.py` | `dC0/dC1/dC2` residuals as phase, plus the oracle reconstruction |
| `run_bxt_test.py` | `bxt` sweep with the oracle filter |
| `docs/` | `coreg_report_en.pdf` / `coreg_report_pt.pdf`, the co-registration study |

`geometry.py`, `sata.py` and `reconstruction.py` were merged from ten
smaller files (`params3d.py`, `geom3d.py`, `datagen3d.py`, `sata2d.py`,
`rd_recons2d.py`, `rangecomp.py`, `siglib.py`, `datastore.py`, `recon3d.py`,
`irf_common.py`) — same code, fewer files, each numerically verified
identical to the pre-merge version bit-for-bit. Each merged file opens with a
docstring naming exactly what went into it, and internal section banners
(`# === n) ... ===`) mark where each former file's content starts, in case
you want to trace something back to how it used to be split up.

The dependency order is `geometry.py` → `sata.py` → `reconstruction.py`:
`sata.py` only depends on `geometry.py` (the analytic `get_coeff_nu`);
`reconstruction.py` depends on both.

---

## 3. What the pipeline does

1. **Parameters and tracks.** One transmitter on a straight track at speed
   `ve`; `Nrx` receivers displaced along-track by `bat[i]` **and**
   cross-track by `bxt[i]` (`geometry.make_params3d`, `geometry.build_tracks_3d`).
2. **Raw data, twice.** The ideal _monostatic_ signal at the full equivalent
   PRF (the reference to compare against), and the per-channel _bistatic_
   signals on the decimated azimuth axis at `PRF_op = prf/Nrx`
   (`geometry.generate_reference_3d` / `generate_channels_3d`).
3. **Range compression** of both (`reconstruction.range_compress`).
4. **Two-step reconstruction + SATA** (`reconstruction.reconstruct_subband_2d`):
   - **STEP 1 — `create_ref_dataset`, in the 2-D frequency domain.** Build
     the filter **once**, at the mid-swath reference range, but for every
     range frequency. Removes the wavelength dependence exactly.
   - **STEP 2 — `generalized_rd`, in the range-time domain.** For every
     range bin, build the filter at the true `r0` and apply only the phase
     _difference_ with respect to the reference filter. `interp_filter`
     additionally shifts it along range to follow the range-cell migration
     of each Doppler bin.
   - **SATA** (still the same 2-D kernel of `sata.py`) corrects the
     topographic residual `dC0` per sub-band before STEP 2, evaluated at the
     target's true (bistatic) slant range via the numeric
     `geometry.CoeffTable3D` / `get_coeff_nu_3d`.
5. **Azimuth focusing** of both, then comparison: peak amplitude and
   position (matched filter against the monostatic reference).

### The residual SATA removes

```
dC0_i(p) = C0_i(p) − C0_i(p_flat(r_p))          (exact, from get_coeff_nu_3d)
dC0     ≈ −b_perp · dh / (r0 · sin θ_inc)       (closed form, for a sanity check only)
```

`p_flat(r_p)` is the flat-earth point at the **same slant range and
azimuth** — exactly what the filter of that range bin assumes. The
reconstruction always uses the exact numeric value; the closed form
(`geometry.dC0_approx`) is never called by the reconstruction path, only
used as an independent check.

### One thing to know about large `bxt`

A receiver displaced by `b_xt` sees the target at a slant range shorter by
≈ `b_xt·sin θ_inc`, so its half-path moves by `b_xt·sin θ / 2` — for
`b_xt = 450 m`, `θ = 20°` that is 77 m, about three range bins. The
correction is therefore deposited at each channel's **bistatic** range bin
(`reconstruction.scatterer_range_bistatic`), not the monostatic one.

---

## 4. Results — single elevated iso-range target

Focused peak, as a percentage of the _ideal_ reconstruction (the filter told
the true height):

| case            | max&nbsp;\|dC0\| [°] | no SATA | SATA whole band | SATA per sub-band |
| --------------- | -------------------- | ------- | --------------- | ----------------- |
| bxt<=20, dh=240 | 16 | 99.5 % | 100.0 % | 100.0 % |
| bxt<=100, dh=240 | 79 | 86.6 % | 100.5 % | 100.5 % |
| bxt<=300, dh=240 | 236 | 46.6 % | 99.8 % | 99.8 % |
| bxt<=100, dh=400 | 131 | 65.7 % | 100.3 % | 100.3 % |
| bxt<=300, dh=400 | 394 | 56.5 % | 100.0 % | 100.0 % |
| Nrx=2, bxt<=100 | 79 | 91.9 % | 100.9 % | 100.9 % |
| Nrx=6, bxt<=100 | 113 | 72.3 % | 100.4 % | 100.4 % |
| off-DPCA dx=11 | 79 | 89.6 % | 100.4 % | 100.4 % |
| off-DPCA dx=100 | 79 | 86.6 % | 100.4 % | 100.4 % |

SATA recovers **99.8–100.9 %** of the ideal peak in every case.  All cases use
`bxt ~ U(0, bxt_max)` with seed 0 and DPCA along-track timing, except the last
two which break DPCA on purpose. For a
**single** target the residual is constant across azimuth, so whole-band and
per-sub-band SATA agree exactly.

## 5. Results — topography that varies along azimuth

Five iso-range targets at azimuth −400 … +400 m with heights 80 … 400 m —
the case a single global correction _cannot_ fix. Focused peak of the
central target, as a percentage of the monostatic reference:

|                       | peak          | % of monostatic |
| --------------------- | ------------- | --------------- |
| monostatic reference | 1.308e+04 | 100.0 % |
| no SATA | 1.285e+04 | 98.3 % |
| SATA whole band | 1.221e+04 | 93.4 % |
| **SATA per sub-band** | **1.324e+04** | **101.3 %** |

**This is where the per-sub-band scheme earns its cost.** Each sub-band's
frequency window maps onto a different piece of the ground, so the correction
becomes position-selective. The whole-band correction ends up *below* no
correction at all: with a residual that changes sign along azimuth, one value
per azimuth line subtracts the wrong correction over half the scene.

Repeating the same case at three reference ranges (12°/20°/28° incidence,
`--multi-range`): per-sub-band holds 98.6–101.3 %; whole-band stays at 82–97 %, below no
correction throughout — evidence
that the per-sub-band scheme's advantage is not a lucky coincidence of one
particular range.

---

---

## 5b. Diagnostics and the range co-registration study

Three results came out of checking why the focused peak sits at 95–96 % rather
than 100 %.

**`C1` and `C2` are identically zero in broadside.** With
`dCk = Ck(true target) − Ck(flat point at the same slant range)` converted to
phase over `T_int`, the maximum `|phi_0|` is 78.8° while `|phi_1|` and
`|phi_2|` are 3.0e−10 and 2.4e−9 of it — `polyfit` noise. Both `r_ms(t)` and
`r_bs(t)` are even about their own points of closest approach, so the
difference carries no odd term. An **oracle** reconstruction, handed the true
target height so that `C0`, `C1` and `C2` are all exact, reaches 99.5 %
against 100.0 % for SATA with `C0` alone — it does not even match it, because
the oracle table is built at one height for every range. Implementing
`C1`/`C2` would buy nothing here.
(`run_c1c2.py`)

**Inter-channel range co-registration is already handled, implicitly.** A
cross-track baseline shifts each channel's energy by `bxt*sin(theta_inc)/2`.
What matters is the *spread* between channels: 10.6 m, or 0.41 range cells,
for `bxt ~ U(0,100) m`; 2.95 cells for a symmetric ladder with `bxt = ±225 m`. That shift is not missing: the
STEP 1 filter is built at every range frequency, and its `C0/wl_m` term
expands into `C0*f0/c + C0*fr/c`, whose second half is linear in range
frequency and is therefore, by the shift theorem, exactly the co-registration.
Measured with the oracle filter:

| STEP 1 filter | explicit co-reg | peak | err/signal |
|---|---|---|---|
| `wl(f_r)` | no | **99.5 %** | −10.83 dB |
| `wl(f_r)` | yes | 95.9 % | −8.41 dB |
| `wl0` | no | 93.3 % | −8.74 dB |
| `wl0` | yes | **99.5 %** | −10.84 dB |

Exactly one of the two routes must be applied; both, or neither, breaks it. The
practical consequence is that the `wl_arr` loop in `create_ref_dataset` is
load-bearing and must not be simplified away. `coreg.py` implements the second
diagonal entry — the co-registration as an explicit, printable, plottable step
— as a test version; it is equivalent, not better.
(`run_coreg.py`, `plot_coreg.py`, `docs/coreg_report_en.pdf`)

**Do not use phase-difference maps in this regime.** With an error-to-signal
ratio of −3.87 dB (no SATA) or −11.65 dB (with SATA), and much worse on a wide
array, `arg(S_rec * conj(S_ref))` is close to uniform and the map shows noise rather
than structure. Use `|S_rec − S_ref| / |S_ref|` in dB instead.
(`run_check.py`, `plot_esr.py`)

## 6. Known limitations

1. **Only the `C0` term is corrected.** For this geometry that is justified,
   and now measured (Section 5b): the `C1` and `C2` residuals are `polyfit` noise
   and an oracle filter does not even match SATA with `C0` alone. It stops being justified
   under squint.
2. **Range co-registration is implicit.** The channels are misaligned in range
   by `bxt*sin(theta_inc)/2` (0.41 cells of spread for `bxt ~ U(0,100) m`), and
   the
   range-frequency dependence of the STEP 1 filter is what corrects it — see
   Section 5b. It works, but it is invisible in the source; `coreg.py` is the
   version that makes it explicit.
3. `apply_rcm` in STEP 1 is off by default: the true range-cell-migration
   effect is range-dependent and is already handled by STEP 2 + RCMC
   (`interp_filter`); turning both on double-counts it.
4. `interp_filter` requires the filter block size to equal the number of
   range bins it operates on (`Nb == Nr`); this is validated, not fixed for
   the general block case.
