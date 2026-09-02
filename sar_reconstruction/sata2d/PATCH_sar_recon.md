# Fixes to apply to `sar_recon/sata.py` and `sar_recon/subband_recon.py`

The reference implementation lives in `sata2d/sata2d.py`; these are the minimal
edits that carry the same fixes into the existing 1-D package. Rationale and
measurements: `sata2d/README.md`, section 3.

---

## 1. `sar_recon/sata.py` — `sata_1d` signature

**Before**

```python
def sata_1d(data, delta_C0_array, rref, prf, v, wl, r,
            squint=0.0, Nsb=1, inverse=False, sata_osf=1, verbose=True):
    ...
    Tsubeff = int(np.round(deltax * prf / v * 0.5 / Nsb) * 2)
    ...
    fsub = Nsb * (np.arange(Nzp) * prf / Nzp / Nsb - prf * 0.5 / Nsb) + dfc
    ...
    azpos = r * (np.tan(betasub) - np.tan(squint)) / v * prf
```

**After** — `(prf, Nsb)` is ambiguous; replace it with the sampling rate of the
array actually handed in plus the absolute sub-band centre, and separate the
frequency squint from the image squint:

```python
def sata_1d(data, delta_C0_array, rref, prf_data, v, wl, r,
            f_centre=0.0, squint_image=0.0, inverse=False,
            sata_osf=1, verbose=True):
    ...
    # [FIX 1] prf_data is already the per-channel rate: no extra /Nsb.
    Tsubeff = int(np.round(deltax * prf_data / v * 0.5) * 2)
    Nzp = int(2 ** np.ceil(np.log2(Tsubeff)) * sata_osf)
    ...
    # [FIX 2] one sub-band spans prf_data, centred on f_centre.
    dfc = np.round(f_centre * Nzp / prf_data) * prf_data / Nzp
    pfc = np.round(np.mod(f_centre, prf_data) * Nzp / prf_data)
    fsub = np.arange(Nzp) * prf_data / Nzp - prf_data * 0.5 + dfc
    fsub[0] = prf_data * 0.5 + dfc
    fsub = np.roll(fsub, int(Nzp / 2 + pfc))
    betasub = np.arcsin(np.clip(wl * fsub / (2.0 * v), -1.0, 1.0))
    # [FIX 3] the reconstructed image is BROADSIDE -> squint_image = 0.
    azpos = r * (np.tan(betasub) - np.tan(squint_image)) / v * prf_data
```

## 2. `sar_recon/sata.py` — `sata_channels` (whole-band SATA)

```python
out[kk, :] = sata_1d(out[kk, :], dC0, rref=cfg.scene.r0,
                     prf_data=cfg.PRF_op,          # was prf=cfg.PRF_op
                     v=cfg.system.vs, wl=cfg.system.wl, r=cfg.scene.r0,
                     f_centre=0.0, squint_image=0.0,   # was squint=cfg.sq_tx, Nsb=1
                     inverse=remove, sata_osf=sata_osf, verbose=verbose)
```

Behaviour is unchanged here (`Nsb` was already 1 and `sq_tx = 0`) — this is
just the new signature.

## 3. `sar_recon/subband_recon.py` — `sata_channels_subband`

**Before**

```python
_f, beta_k, _lo, _hi = subband_frequency_beam(cfg, k)
out[kk, :] = sata_1d(out[kk, :], dC0, rref=cfg.scene.r0, prf=cfg.PRF_op,
                     v=cfg.system.vs, wl=cfg.system.wl, r=cfg.scene.r0,
                     squint=beta_k, Nsb=Nrx,          # <-- BUGS 1 + 2 + 3
                     inverse=remove, sata_osf=sata_osf, verbose=verbose)
```

**After**

```python
f_k, beta_k, _lo, _hi = subband_frequency_beam(cfg, k)
out[kk, :] = sata_1d(out[kk, :], dC0, rref=cfg.scene.r0,
                     prf_data=cfg.PRF_op,   # the channel line's own rate
                     v=cfg.system.vs, wl=cfg.system.wl, r=cfg.scene.r0,
                     f_centre=f_k,          # the sub-band's frequency window
                     squint_image=0.0,      # the image is broadside
                     inverse=remove, sata_osf=sata_osf, verbose=verbose)
```

`subband_frequency_beam` already returns `f_k` first, so nothing else changes.

---

## Sanity checks after patching

1. `delta_C0_array == 0` must reproduce the input bit-for-bit (COLA holds).
2. `reconstruct_subband(use_sata=False)` must still equal `reconstruct()`
   exactly.
3. `Tsubeff * v / PRF_op` must equal `deltax = sqrt(wl*r/2)` to within one
   sample — the "check the data after the STFT" test.
4. `azpos` must match `wl * r * fsub * PRF_op / (2 * vs**2)` to machine
   precision (TEST 3 in `run_sata_diagnostics.py`).
