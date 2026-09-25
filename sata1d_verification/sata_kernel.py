"""
user_sata1d.py
==============
Verbatim copy of `sata_for_topography` from the project file SATA1D.txt
(Pau Prats' 1-D SATA), with ONLY the two changes needed to run headless:

  * `matplotlib.use('TkAgg')` / `plt.ion()` removed (no display in the VM);
  * nothing else touched -- STEP 1 (the frequency axis, betasub, azpos) and
    STEP 2 (the sub-aperture FFT and the ph = -2*pi/wl * delta_C0 correction)
    are byte-for-byte the original, so any result below is a statement about
    YOUR code, not a re-implementation.

A thin helper `step1_mapping(...)` reproduces STEP 1 alone (the exact same
lines) and returns fsub / betasub / azpos, so the verification script can look
at the Doppler->angle->azimuth-position map without running the full loop.
"""
import numpy as np


# ---------------------------------------------------------------------------
# THE USER'S CODE -- sata_for_topography, verbatim from SATA1D.txt
# (only the matplotlib display lines at the top of the module were dropped)
# ---------------------------------------------------------------------------
def sata_for_topography(data, delta_C0_array, rref, prf, Nsb, v, squint, wl, r,
                        inverse=False, sata_osf=1):
    dimx = len(data)

    # STEP 1: SATA pre-computations
    deltax = np.sqrt(wl * rref / 2)
    Tsubeff = np.round(deltax * prf / v * 0.5 / Nsb) * 2
    Nzp = int(2 ** np.ceil(np.log10(Tsubeff) / np.log10(2)) * sata_osf)
    Tovl = int(Tsubeff * 0.5)
    Tsub = Tsubeff * 0.5
    Nsub = int(dimx // Tsub)
    marg_az = np.round(0.5 * (1 + Nzp - Tsubeff))

    print('')
    print(f'SATA: resolution in the accommodation of the topography at mid range: {deltax:.2f}')
    print(f'SATA: sub-aperture length: {Tsubeff}, zero padded to {Nzp:.2f} with margin {marg_az:.2f}')

    if (Tsubeff <= 2):
        print('SATA: sub-aperture length very small, no correction needed')
        return

    rightweight = np.arange(Tovl) / (Tovl - 1)
    leftweight = rightweight[::-1]

    # frequency axis of the subaperture
    fc = 2 * v / wl * np.sin(squint)
    pfc = np.round(np.mod(fc, prf) * Nzp / prf)
    dfc = np.round(fc * Nzp / prf) * prf / Nzp
    fsub = Nsb * (np.arange(Nzp) * prf / Nzp / Nsb - prf * 0.5 / Nsb) + dfc
    fsub[0] = prf * 0.5 + dfc
    fsub = np.roll(fsub, +int(Nzp / 2 + pfc))
    betasub = np.arcsin(wl * fsub / (2 * v))

    azpos = r * (np.tan(betasub) - np.tan(squint)) / v * prf

    # STEP 2: loop to perform SATA
    temp = np.zeros(int(2 * Tsub), complex)
    for m in range(Nsub):
        if m == Nsub - 1:
            posix = (m + 0.5) * Tsub
            aux = data[int(m * Tsub):]
            Tsubeff = int(dimx - m * Tsub)
        else:
            posix = (m + 1) * Tsub
            aux = data[int(m * Tsub): int((m + 2) * Tsub)]

        if Nzp != Tsubeff:
            aux = np.concatenate((aux, np.zeros([int(Nzp - Tsubeff)], complex)), axis=0)

        aux = np.fft.fft(aux)

        posaux = (azpos + dimx / 2 + (posix - dimx / 2)).astype(int)
        posaux[posaux < 0] = 0
        posaux[posaux > dimx - 1] = dimx - 1

        ph = -2 * np.pi / wl * delta_C0_array[posaux]
        ph[np.where(np.isnan(ph))] = 0
        ph[np.where(np.isinf(ph))] = 0

        if inverse:
            aux *= np.exp(-1j * ph)
        else:
            aux *= np.exp(1j * ph)
        aux = np.fft.ifft(aux, axis=0)

        if m == 0:
            data[0:int(Tsub)] = aux[:int(Tsub)]
        else:
            aux[:int(Tsub)] = temp[int(Tsub):int(2 * Tsub)] * leftweight + aux[:int(Tsub)] * rightweight
            data[int((m - 1) * Tsub):m * int(Tsub)] = temp[:int(Tsub)]
            temp = aux
        temp = aux

    data[int((m - 1) * Tsub):] = temp[:int(dimx - (m - 1) * Tsub)]
    return data


# ---------------------------------------------------------------------------
# CORRECTED kernel -- same STEP 1 physics, but the buggy overlap-add of
# SATA1D.txt is replaced by the normalised triangular weighted-overlap-add
# (WOLA) that sar_recon/sata.py uses, so that with delta_C0 = 0 the identity
# is reproduced exactly. Use this one for any QUANTITATIVE measurement; the
# original's WOLA (off-by-one + duplicated temp) corrupts the output line.
# ---------------------------------------------------------------------------
def sata_1d_corrected(data, delta_C0_array, rref, prf, Nsb, v, squint, wl, r,
                      inverse=False, sata_osf=1):
    data = np.asarray(data, complex).copy()
    dimx = len(data)
    g = step1_mapping(dimx, rref, prf, Nsb, v, squint, wl, r, sata_osf)
    azpos, Nzp = g["azpos"], g["Nzp"]
    L = int(g["Tsubeff"])
    hop = max(1, L // 2)                         # 50% overlap
    if L <= 2:
        return data
    win = 1.0 - np.abs(np.arange(L) - 0.5 * (L - 1)) / (0.5 * L)   # triangular
    out = np.zeros(dimx, complex)
    wsum = np.zeros(dimx)
    for start in range(0, dimx, hop):
        seg = data[start:start + L]
        Lseg = len(seg)
        buf = seg * win[:Lseg]
        if Lseg < Nzp:
            buf = np.concatenate([buf, np.zeros(Nzp - Lseg, complex)])
        spec = np.fft.fft(buf)
        centre = start + 0.5 * L
        posaux = np.clip(np.round(azpos + centre).astype(int), 0, dimx - 1)
        ph = -2 * np.pi / wl * delta_C0_array[posaux]
        ph[~np.isfinite(ph)] = 0.0
        spec *= np.exp(-1j * ph) if inverse else np.exp(1j * ph)
        rec = np.fft.ifft(spec)[:Lseg]
        out[start:start + Lseg] += rec
        wsum[start:start + Lseg] += win[:Lseg]
    nz = wsum > 1e-12
    out[nz] /= wsum[nz]
    out[~nz] = data[~nz]
    return out


# ---------------------------------------------------------------------------
# STEP 1 alone -- the exact same lines, returned instead of consumed.
# This is the Doppler-bin -> azimuth-cell map SATA uses to look up delta_C0.
# ---------------------------------------------------------------------------
def step1_mapping(dimx, rref, prf, Nsb, v, squint, wl, r, sata_osf=1):
    deltax = np.sqrt(wl * rref / 2)
    Tsubeff = np.round(deltax * prf / v * 0.5 / Nsb) * 2
    Nzp = int(2 ** np.ceil(np.log10(Tsubeff) / np.log10(2)) * sata_osf)
    Tsub = Tsubeff * 0.5

    fc = 2 * v / wl * np.sin(squint)
    pfc = np.round(np.mod(fc, prf) * Nzp / prf)
    dfc = np.round(fc * Nzp / prf) * prf / Nzp
    fsub = Nsb * (np.arange(Nzp) * prf / Nzp / Nsb - prf * 0.5 / Nsb) + dfc
    fsub[0] = prf * 0.5 + dfc
    fsub = np.roll(fsub, +int(Nzp / 2 + pfc))
    betasub = np.arcsin(np.clip(wl * fsub / (2 * v), -1.0, 1.0))
    azpos = r * (np.tan(betasub) - np.tan(squint)) / v * prf
    return dict(fsub=fsub, betasub=betasub, azpos=azpos,
                Tsubeff=Tsubeff, Nzp=Nzp, Tsub=Tsub, deltax=deltax)
