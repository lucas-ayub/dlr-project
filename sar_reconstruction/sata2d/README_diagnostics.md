# Diagnostic scripts (Sep 2026)

These complement the `run_sata_*` scripts. Run them from
`sar_reconstruction/`, e.g. `python -m sata2d.run_irf2d`.

| file | what it does |
|---|---|
| `run_irf2d.py`    | **2-D IRF** — range x azimuth contour plus the two 1-D cuts, in the style of Sakar's Fig. 2.9 |
| `run_check.py`    | reconstructs (no SATA / per sub-band), measures the error-to-signal ratio, writes `plots/cache_check.npz` |
| `plot_esr.py`     | 4-panel **error budget** from that cache: 2-D error-to-signal map, cut vs Doppler, azimuth IRF, main-lobe zoom |
| `run_axes_2d.py`  | 4-panel **axis** figure: azimuth IRF (slow time), zoom in metres, range IRF (fast time), 2-D spectrum |
| `run_c1c2.py`     | `dC0/dC1/dC2` residuals as phase, plus an "oracle" reconstruction (filter given the true target height) |
| `run_bxt_test.py` | `bxt` sweep with the oracle filter, isolating the range misalignment |

## `run_irf2d.py` — the two things that make the figure work

**2-D matched-filter focusing.** `img = IFFT2(FFT2(x) * conj(FFT2(ref)))`,
with `ref` the ideal monostatic signal of the same target. This is exact: the
RCM (~41 m, 8 range cells) is contained in `ref` and is therefore compensated
without any approximate RCMC. Focusing each range bin against the azimuth
chirp of a single bin — the obvious shortcut — leaves that RCM uncorrected.

**Range resolution comparable to the azimuth one.** The standard preset has
`res_rg = 30 m` against `La/2 = 6 m` in azimuth: a 5:1 ratio, and the cross
comes out flattened. The script defaults to `--res-rg 6`, which matches the
two and reproduces the isotropic cross of the reference figure; `--swath 300`
keeps `Nr = 128`, i.e. the same cost as the standard preset. `--res-rg 30`
shows the original case, with the axes scaling themselves accordingly.

Sanity check: for the monostatic reference the script returns
PSLR = -13.3 / -13.5 dB in azimuth and range — the theoretical rectangular
window value is -13.26 dB — and IRW(-3 dB) of 3.7-4.2 m for 6 m resolution.

```
python -m sata2d.run_irf2d                  # monostatic reference
python -m sata2d.run_irf2d --method sub     # SATA per sub-band (mono dashed)
python -m sata2d.run_irf2d --res-rg 30      # standard preset, anisotropic
```

## Measurement results

- `dC1` and `dC2` are **identically zero** in broadside (ratios 2.7e-8 and
  2.4e-9 against `dC0`). Implementing them would buy nothing.
- Oracle (exact phase): 96.1 % / -6.27 dB. SATA with `C0` only: 95.4 % /
  -6.39 dB. The phase model is already at its ceiling.
- What is left is the **inter-channel range misalignment** caused by `bxt`:
  1.47 cells in the standard preset. With `bxt` -> 0 the oracle reaches
  99.8 % / -19.50 dB.

## A note on diagnostics

Do not use `arg(S_rec * conj(S_ref))` maps in this regime. With an
error-to-signal ratio of +1.18 dB (no SATA) or -6.39 dB (with SATA), the
phase difference is close to uniform and the map shows noise, not structure.
Use `|S_rec - S_ref| / |S_ref|` in dB instead.

## Documentation

`docs/coreg.pdf` — full derivation of the inter-channel range co-registration
term (`docs/coreg.tex` is the source).
