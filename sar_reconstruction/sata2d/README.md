# `sata2d` — 2-D two-step azimuth reconstruction + corrected sub-band SATA

Self-contained port of the reference DLR files (`dataGenerator.py`,
`genRDrecons.py`, `MP2DATBF.py`, `SATA1D.txt`) for a **linear orbit**, plus a
corrected SATA kernel, a diagnostics script that measures the three bugs that
kept the sub-band SATA + azimuth reconstruction from working, and a **3-D
extension** that carries along-track _and_ cross-track baselines and real target
topography.

---

## 1. How to run

```bash
pip install numpy scipy matplotlib      # h5py only for the "paper" preset

cd sar_reconstruction
python -m sata2d.run_twostep2d --plots  # full 2-D pipeline   (~2 s, in RAM)
python -m sata2d.run_sata_diagnostics   # the 5 SATA bug tests (~2 s)
python -m sata2d.run_sata2d_topo --plots  # 2-D SATA with bat + bxt + topography
```

`run_sata2d_topo.py` takes ~4 min for the full sweep; add `--quick` for three
cases (~1 min) or `--skip-sweep` to jump straight to the azimuth-topography
experiment (~20 s).

Both scripts also run directly (`python run_twostep2d.py`) — they add the parent
directory to `sys.path` themselves. Figures are written to `plots/sata2d/`.

### Options

| flag             | effect                                                                   |
| ---------------- | ------------------------------------------------------------------------ |
| `--preset small` | default; reduced bandwidth/aperture, `Na × Nr = 2048 × 512`, runs in RAM |
| `--preset paper` | the reference numbers, `90112 × 32768` — use with `--backend hdf5`       |
| `--nrx N`        | number of receive channels (validated for 2, 3, 4)                       |
| `--analytic`     | closed-form 2×2 matrix inverse (`Nrx = 2` only)                          |
| `--no-rcmc`      | skip the range-cell migration of the residual filter                     |
| `--backend hdf5` | out-of-core via HDF5 instead of NumPy arrays in RAM                      |
| `--outdir DIR`   | where the figures go (default `plots/sata2d`)                            |

### Expected output

```
phase difference vs the ideal full-PRF signal (in band):
  constant offset :   -0.022 deg
  rms residual    :    1.101 deg
  max |residual|  :    4.057 deg

azimuth impulse response (range bin 25):
       reference: peak at 1024, |peak| = 2.311e+02, PSLR = -28.42 dB
   reconstructed: peak at 1024, |peak| = 2.294e+02, PSLR = -28.18 dB
  peak offset      : 0 samples
  amplitude ratio  : 0.9927
```

---

## 2. What is in here

| file                      | role                                                   | reference file it replaces    |
| ------------------------- | ------------------------------------------------------ | ----------------------------- |
| `params.py`               | acquisition parameters, two presets                    | header of `MP2DATBF.py`       |
| `tracks.py`               | linear-orbit TX/RX tracks                              | header of `MP2DATBF.py`       |
| `datagen2d.py`            | 2-D raw-data generator                                 | `dataGenerator.py`            |
| `rangecomp.py`            | range matched filter                                   | `sarProccessing.mp_rc`        |
| `rd_recons2d.py`          | **two-step Range-Doppler reconstruction**              | `genRDrecons.py`              |
| `sata2d.py`               | **corrected SATA kernel + sub-band driver**            | `SATA1D.txt`                  |
| `siglib.py`               | `fft2`/`ifft2` shims                                   | `SignalProcessingLibrary`     |
| `datastore.py`            | `ds['CH'][...]` in RAM or HDF5                         | `h5py.File`                   |
| `run_twostep2d.py`        | main pipeline                                          | `MP2DATBF.py`                 |
| `run_sata_diagnostics.py` | the bug hunt                                           | —                             |
| `params3d.py`             | **3-D** geometry: bat + bxt + scene topography         | —                             |
| `geom3d.py`               | 3-D tracks, numeric `(C0,C1,C2,Dt)`, coefficient table | `sar_recon.GetCoeffNu`        |
| `datagen3d.py`            | vectorised 2-D generator for the 3-D scene             | —                             |
| `sata3d.py`               | topographic residual map + per-sub-band 2-D SATA       | —                             |
| `run_sata2d_topo.py`      | the 3-D SATA experiment + plots                        | `runs/core/run_sata.py` (1-D) |
| `reference/`              | the original files, unmodified                         | —                             |
| `doc/`                    | LaTeX documents (theory + function reference)          | —                             |

Function-by-function reference: `doc/sata2d_functions_EN.pdf` (7 pages).
Full derivations and a line-by-line reading of both codebases:
`doc/SATA_2D_teoria_codigo.pdf` (48 pages, in Portuguese).

---

## 3. What the pipeline does

1. **Parameters and tracks.** One transmitter on a straight track at speed `ve`;
   `Nrx` receivers on the same line, displaced along-track by `deltaX[i]`.
2. **Raw data, twice.** The ideal _monostatic_ signal at the full equivalent PRF
   (the reference to compare against), and the per-channel _bistatic_ signals on
   the decimated azimuth axis at `PRF_op = prf/Nrx` (the data to reconstruct).
   Both are restricted to the Doppler band common to the monostatic and the most
   bistatic channel, so the two can be compared bin by bin at the end.
3. **Range compression** of both.
4. **Two-step reconstruction** (below).
5. **Azimuth focusing** of both, then comparison: the 2-D phase difference inside
   the common band, and the azimuth impulse response (peak, PSLR).

### Why two steps

The reconstruction filter depends on the slant range `r0` (through `C0, C1, C2`)
**and** on the range frequency `fr` (through `wl(fr) = c/(f0+fr)`). Doing both at
once is `Nr × Nr` filter cubes — impossible for a real swath. So:

- **STEP 1 — `create_ref_dataset`, in the 2-D frequency domain.** Build the
  filter **once**, at the mid-swath reference range, but for **every range
  frequency**. This removes the wavelength dependence exactly. Result goes to
  `REC_Pref`, back to range time by an IFFT.
- **STEP 2 — `generalized_rd`, in the range-time domain.** For every range bin,
  build the filter at the true `r0` and apply only the **phase difference** with
  respect to the reference filter,
  `iHf(r0) = exp(j·arg P(r0)) · conj(exp(j·arg P(r_ref)))`.
  That residual is smooth in range frequency, so it may be applied in range time.
  `interp_filter` additionally shifts it along range to follow the range cell
  migration of each Doppler bin.

### Layout conventions (same as the 1-D `sar_recon.reconstruction`)

- `CH` `[Nrx, Na_ch, Nr]` — channel data, 2-D **frequency** domain, unshifted FFT
  order on both axes.
- Azimuth frequency of row `kk` of `REC` is `fa[kk] = -prf/2 + kk·prf/Na`;
  sub-band `jj` occupies rows `jj·Na_ch … (jj+1)·Na_ch` and covers
  `fsub + jj·prf/Nrx`.
- Raw FFT bin `n` of a channel carries, **for output sub-band k**, the true
  Doppler `-prf/2 + n·PRF_op/Na_ch + k·PRF_op`.

---

## 4. Validation

Single point target, `small` preset, against the ideal full-PRF monostatic
signal:

|                             | `Nrx = 2`  | `Nrx = 3` | `Nrx = 4` |
| --------------------------- | ---------- | --------- | --------- |
| in-band rms phase residual  | **1.10°**  | 2.46°     | 2.86°     |
| amplitude ratio (rec./ref.) | **0.9927** | 0.9899    | 0.9889    |
| peak offset                 | 0 samples  | 0         | 0         |
| PSLR (reference −28.42 dB)  | −28.18 dB  | —         | —         |
| azimuth ambiguities         | ≈ −55 dB   | —         | —         |

---

## 4b. 2-D SATA with real baselines and topography

The linear-orbit model of §3 has receivers on the **same straight line** as the
transmitter, so the only baseline is along-track. With a purely along-track array
the reconstruction is essentially blind to topography — `dC0/dr ≈ -bat²/(4r²)`,
i.e. **0.008° per 100 m** of range error — and SATA has nothing to correct.
`params3d.py` / `geom3d.py` / `datagen3d.py` / `sata3d.py` add the two missing
ingredients.

### Geometry

```
transmitter   ptx(t) = ( vs·t,           0,       H )
receiver i    prx(t) = ( vs·t - bat[i],  bxt[i],  H )
target        ptg    = ( x0,             y0,      h0 ),  y0 = sqrt(r0² - (H-h0)²)
```

`bat` and `bxt` are generated exactly as in `sar_recon.config.ArrayGeometry.linear`:
`bat[i] = bat_offset + dx·i`, and `bxt[i] = dxt·(i - (Nrx-1)/2)` (`"linear"`) or
`bxt[i] ~ U(0, bxt_max)` with a seed (`"random"`).

Defaults reproduce the 1-D SATA tests of `runs/core/run_sata.py`: `wl = 0.25 m`,
`H = 720 km`, `rDelay = 0.0051115753 s` → `r0 = 766.21 km`, `θ_inc = 20.0°`,
`da = 24·wl`, `La = 2·da`, `PRF = 2000 Hz`, along-track spacing 100 m. So the
2-D results are directly comparable with the 1-D ones.

### The residual SATA removes

```
dC0_i(p) = C0_i(p) − C0_i(p_flat(r_p))          (exact, from get_coeff_nu_3d)
dC0     ≈ −b_perp · dh / (r0 · sin θ_inc)       (closed form, for sanity)
```

`p_flat(r_p)` is the flat-earth point at the **same slant range and azimuth** —
exactly what the filter of that range bin assumes. Measured agreement between the
two, for the default geometry:

| ch  | bat [m] | bxt [m] | dh [m] | exact [mm] | approx [mm] | phase [deg] |
| --- | ------- | ------- | ------ | ---------- | ----------- | ----------- |
| 0   | 0       | −225    | 240    | 193.34     | 206.07      | 278.4       |
| 1   | 100     | −75     | 240    | 64.45      | 68.69       | 92.8        |
| 3   | 300     | +225    | 400    | −322.00    | −343.44     | −463.7      |

6 % apart — the closed form is a good order-of-magnitude check, and the exact
value is what the code uses.

### One thing to know about large `bxt`

A receiver displaced by `b_xt` sees the target at a slant range shorter by
≈ `b_xt·sin θ_inc`, so its **half-path moves by `b_xt·sin θ / 2`**. For
`b_xt = 450 m`, `θ = 20°` that is 77 m — about **three range bins**. The
correction must therefore be deposited at each channel's **bistatic** range bin
(`sata3d.scatterer_range_bistatic`), not at the monostatic one. Using the
monostatic bin makes large-baseline cases fail silently; that was found and fixed
during this work.

### Results — single elevated iso-range target

Focused peak, as a percentage of the _ideal_ reconstruction (the filter told the
true height). `max|dC0|` is the worst per-channel residual in degrees.

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
| DPCA dx=11      | 278                  | 37.4 %  | 99.2 %          | 99.2 %            |

SATA recovers **99.0–100.4 %** of the ideal peak in every case.

For a **single** target the residual is constant across azimuth, so whole-band
and per-sub-band SATA agree exactly — as they should: `C0` is evaluated at
closest approach and is essentially angle-invariant.

### Results — topography that varies along azimuth

Five iso-range targets at azimuth −400 … +400 m with heights 80 … 400 m — the
case a single global correction _cannot_ fix. Focused peak of the central target,
as a percentage of the monostatic reference:

|                       | peak          | % of monostatic |
| --------------------- | ------------- | --------------- |
| monostatic reference  | 1.308e+04     | 100.0 %         |
| no SATA               | 7.725e+03     | 59.1 %          |
| SATA whole band       | 9.238e+03     | 70.7 %          |
| **SATA per sub-band** | **1.194e+04** | **91.3 %**      |

**This is where the per-sub-band scheme earns its cost.** Each sub-band's
frequency window maps onto a different piece of the ground (Eq. `azpos`), so the
correction becomes position-selective; a single whole-band correction can only
apply one value per azimuth line and leaves elevated sidelobes.

Figures: `plots/sata2d/sata3d_sweep.png`, `sata3d_azimuth_topo.png`,
`sata3d_residual.png`.

---

## 5. The sub-band SATA bug

A channel line has `Na_ch` samples at `PRF_op`, so its spectrum is only `PRF_op`
wide. Output sub-band `k` is **`PRF_op` wide** (not `Nrx·PRF_op`) and centred on
`f_k = (-Nrx/2 + k + 1/2)·PRF_op`. A channel line therefore _already is_ "one
sub-band's worth of samples": only the **frequency label** of each bin changes
from sub-band to sub-band.

Three defects, all in **STEP 1** of the SATA kernel:

### BUG 1 — sub-aperture length, `/Nsb` applied twice

`Tsubeff = round(deltax*prf/v*0.5/Nsb)*2` assumes data sampled at `prf/Nsb`. The
call site passed `prf = PRF_op` (**already** the per-channel rate) _and_
`Nsb = Nrx`.

|           | `Tsubeff`  | ground extent                                 |
| --------- | ---------- | --------------------------------------------- |
| correct   | 20 samples | 107.8 m (`deltax = sqrt(wl·r/2)` = 104.2 m ✓) |
| as called | 10 samples | 53.9 m ✗                                      |

`Nzp` shrinks with it, so the STFT Doppler bin doubles (41.3 → 82.6 Hz) and the
frequency→azimuth map is quantised on a grid twice as coarse.

### BUG 2 — the frequency bins were never per-sub-band

```
fsub = Nsb*(arange(Nzp)*prf/Nzp/Nsb - prf*0.5/Nsb) + dfc
     =     arange(Nzp)*prf/Nzp - prf*0.5           + dfc
```

The leading `Nsb` cancels the `/Nsb`: **`Nsb` has no effect on the frequency axis
at all.** It came out `PRF_op` wide only _because of_ BUG 1. Fix BUG 1 alone — by
passing the full `prf` with `Nsb = Nrx`, the natural reading of the parameter
names — and the axis silently becomes `Nrx×` too wide, and `azpos = …·prf` with
it (mapping error 537–560 samples). **Both must be fixed together.**

### BUG 3 — the mapping was measured against a squinted image grid ← decisive

```
azpos = r*(tan(betasub) - tan(squint)) / v * prf
```

The `- tan(squint)` is right for **classic** SATA, where the image is processed
at the same squint as the beam. In the sub-band reconstruction the image is the
ordinary **broadside** image and only the frequency _window_ is squinted
(`squint = beta_k`). Subtracting `tan(beta_k)` shifts the whole topography lookup
by `r·tan(beta_k)/v·PRF_op` = **±186.7 samples (±1006 m)**, with _opposite signs_
for the sub-bands above and below zero Doppler.

This is exactly why **whole-band SATA works** (`beta_0 = 0` → no shift) **and the
per-sub-band version does not.**

### Measured

Locating the target from the STFT peak and mapping it back through `azpos`,
error in azimuth samples (`run_sata_diagnostics.py`, TEST 2):

| grid      | `Tsubeff` | `Nzp` | bin [Hz] | mean   | median     | p90    |
| --------- | --------- | ----- | -------- | ------ | ---------- | ------ |
| corrected | 20        | 32    | 41.3     | 13.27  | **3.00**   | 6.33   |
| legacy    | 10        | 16    | 82.6     | 186.15 | **186.33** | 195.53 |

Half a bin is 5.8 samples, so the corrected kernel is at the theoretical floor.
`azpos` vs the analytic truth `wl·r·f·prf_data/(2·vs²)` (TEST 3): corrected 0.00,
legacy 186.66, legacy-with-full-PRF 537–560.

### The fix

Drop the ambiguous `(prf, Nsb)` pair. The kernel takes instead:

| argument       | meaning                                                                                                                          |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `prf_data`     | the sampling rate of the array actually handed in (`PRF_op`)                                                                     |
| `f_centre`     | the **absolute** Doppler centre of the sub-band (`f_k`)                                                                          |
| `squint_image` | the squint of the image grid `delta_C0_array` lives on — **0** for the sub-band reconstruction; `beta_k` reproduces classic SATA |

`legacy_sata_1d` / `legacy_subaperture_grid` keep the old behaviour so both stay
side-by-side comparable — that is what produces the tables above.

Before/after edits for `sar_recon/sata.py` and `sar_recon/subband_recon.py`:
**`PATCH_sar_recon.md`**.

---

## 6. Fixes applied to the reference code (so it runs at all)

- **[FIX 1]** `dataGenerator`: the band mask was computed on `bsVld_idx` but
  indexed `msVld_idx`. With `deltaX != 0` the two index sets differ and every
  bistatic channel's illumination window came out shifted.
- **[FIX 2]** `interpFilter` received the whole `r_scan` while operating on a
  block of `Nb` samples — silently wrong for `Nb != Nr`. Now validated, and the
  caller passes the block's slice.
- STEP 1's RCM term was multiplied by a literal `0` (dead code). It is now the
  explicit `apply_rcm` argument, off by default — same behaviour, visible.
- `np.float` / `np.complex` removed (NumPy ≥ 1.24); `arcsin` arguments clipped.
- The `SATA1D.txt` overlap-add bookkeeping (duplicated `temp = aux`, off-by-one
  on the final block) replaced by a standard weighted overlap-add with a
  triangular window and explicit weight normalisation, so COLA holds and
  `delta_C0 ≡ 0` reproduces the input exactly. The physics is untouched.
- `MP2DATBF.py` was rewritten rather than ported: it called `generalizedRD` with
  a signature from a different version of `genRDrecons`.

---

## 7. Known limitations and next steps

1. **Only the `C0` term is corrected**, as in the 1-D routine. For this geometry
   that is justified — in broadside `C1 = 0` exactly and `C2 ~ 1e-11` — but it
   stops being so under squint.
2. ~~No cross-track baseline~~ — **done**: see §4b. The 3-D modules carry `bat`,
   `bxt` and target height, and the residual comes from the exact geometry.
   What is _not_ corrected is the **range mis-registration** between channels
   that a large `bxt` produces (77 m ≈ 3 range bins for `b_xt = 450 m`): SATA
   fixes the phase, not the range offset. Co-registering the channels in range
   is the next step for very large baselines.
3. `apply_rcm` in STEP 1 is off by default (as in the original).
4. `interp_filter` requires `Nb == Nr`; blocking is kept for the HDF5 path but is
   now validated.
5. The `intpF=True` branch of `genRDrecons` (interpolating the data rather than
   the filter, via `getDeltaP`/`phUnwrap`) was not ported — it is not needed for
   the linear-orbit case.

---

## 8. Sanity checks after any change to the kernel

1. `delta_C0_array == 0` must reproduce the input bit-for-bit (COLA).
2. `reconstruct_subband(use_sata=False)` must still equal `reconstruct()` exactly.
3. `Tsubeff · vs / PRF_op` must equal `sqrt(wl·r/2)` to within one sample —
   the "check the sub-aperture length after the STFT" test.
4. `azpos` must match `wl · r · fsub · PRF_op / (2 · vs²)` to machine precision.
