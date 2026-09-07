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
python -m sata2d.run_sata2d_topo --plots   # main experiment, 3-D geometry
python -m sata2d.run_sata_irf              # playground: one target, ONE method
python -m sata2d.run_sata_irf_all          # playground: one target, all THREE
```

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
| dxt=50, dh=240  | 93                   | 48.1 %  | 99.8 %          | 99.8 %            |
| dxt=150, dh=240 | 278                  | 8.3 %   | 99.3 %          | 99.3 %            |
| dxt=300, dh=240 | 557                  | 95.8 %  | 99.9 %          | 99.9 %            |
| dxt=150, dh=400 | 464                  | 61.2 %  | 99.0 %          | 99.0 %            |
| bxt ~ U(0,100)  | 79                   | 86.6 %  | 100.4 %         | 100.4 %           |
| bxt ~ U(0,20)   | 16                   | 99.5 %  | 100.0 %         | 100.0 %           |
| Nrx=2, dxt=150  | 93                   | 8.8 %   | 99.3 %          | 99.3 %            |
| Nrx=6, dxt=150  | 464                  | 10.9 %  | 99.3 %          | 99.3 %            |

SATA recovers **99.0–100.4 %** of the ideal peak in every case. For a
**single** target the residual is constant across azimuth, so whole-band and
per-sub-band SATA agree exactly.

## 5. Results — topography that varies along azimuth

Five iso-range targets at azimuth −400 … +400 m with heights 80 … 400 m —
the case a single global correction _cannot_ fix. Focused peak of the
central target, as a percentage of the monostatic reference:

|                       | peak          | % of monostatic |
| --------------------- | ------------- | --------------- |
| monostatic reference  | 1.308e+04     | 100.0 %         |
| no SATA               | 7.725e+03     | 59.1 %          |
| SATA whole band       | 9.238e+03     | 70.7 %          |
| **SATA per sub-band** | **1.194e+04** | **91.3 %**      |

**This is where the per-sub-band scheme earns its cost.** Each sub-band's
frequency window maps onto a different piece of the ground, so the
correction becomes position-selective; a single whole-band correction can
only apply one value per azimuth line and leaves elevated sidelobes.

Repeating the same case at three reference ranges (12°/20°/28° incidence,
`--multi-range`): per-sub-band stays at 91–95 % throughout; whole-band is
inconsistent (64–71 %, once _worse_ than no correction at all) — evidence
that the per-sub-band scheme's advantage is not a lucky coincidence of one
particular range.

---

## 6. Known limitations

1. **Only the `C0` term is corrected.** For this geometry that is justified
   — in broadside `C1 = 0` exactly and `C2 ~ 1e-11` — but it stops being so
   under squint.
2. **No range co-registration between channels.** The `bxt` correction fixes
   the _phase_; a large `bxt` (77 m ≈ 3 range bins for `b_xt = 450 m`) also
   produces a real physical range misalignment between channels that is not
   resampled/shifted — SATA does not touch it.
3. `apply_rcm` in STEP 1 is off by default: the true range-cell-migration
   effect is range-dependent and is already handled by STEP 2 + RCMC
   (`interp_filter`); turning both on double-counts it.
4. `interp_filter` requires the filter block size to equal the number of
   range bins it operates on (`Nb == Nr`); this is validated, not fixed for
   the general block case.
