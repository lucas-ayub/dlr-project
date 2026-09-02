# -*- coding: utf-8 -*-
"""
2-D raw-data generator -- port of the reference ``dataGenerator.py``.

What it produces
----------------
For a linear orbit and a list of point scatterers it returns

``data``      [Na, Nr]        the ideal *monostatic* signal at the full
                              equivalent PRF -- the signal a single-channel
                              system with PRF = prf would have recorded.  This
                              is the reference the reconstruction is compared
                              against.
``data_ch``   [Nrx, Na, Nr]   the *bistatic* signal of every receive channel
                              (TX at ptx, RX_i at prx[i]).  In the pipeline
                              this is generated on the decimated time axis, so
                              its azimuth length is Na_ch.
``rh_ch``     [Nrx, Na]       bistatic range history of the reference target
``f_max``, ``f_min``          Doppler limits common to the monostatic and the
                              most-bistatic channel; the reconstruction uses
                              them to null the out-of-band azimuth frequencies.

Signal model (per scatterer, per channel)
-----------------------------------------
Range history (two-way path length, TX -> target -> RX)::

    rh(t) = |ptx(t) - p| + |prx(t) - p|

Echo: a linear FM chirp of duration ``cd`` and bandwidth ``rbw`` delayed by
``rh/c`` and carrying the two-way carrier phase ``exp(-2j*pi*rh/wl)``
(:func:`get_echo` receives ``R = rh/2`` and applies ``-4j*pi*R/wl``).

Aperture selection
------------------
A sample is kept when the instantaneous squint ``asin(d rh/dt / v)`` lies inside
the antenna beam, AND when its instantaneous Doppler lies inside the band that
is common to the monostatic and the extreme bistatic channel.  The second
condition is what makes the monostatic reference and the reconstructed signal
occupy exactly the same Doppler support, so the two can be compared bin by bin.

Differences from the reference file (all bug fixes, flagged inline)
-------------------------------------------------------------------
* ``np.complex`` -> ``complex`` (removed in NumPy >= 1.24).
* The per-channel valid-index list was built by indexing ``msVld_idx`` with a
  mask computed on ``bsVld_idx``.  Fixed to index ``bsVld_idx`` (see [FIX 1]).
* ``arcsin`` arguments are clipped to [-1, 1].
"""
from __future__ import annotations

import numpy as np

C_LIGHT = 3.0e8
PI = np.pi

__all__ = ["get_range_hist", "get_echo", "get_valid_idx", "get_raw_data_2d"]


# ---------------------------------------------------------------------------
def get_range_hist(p, ptx, prx):
    """Two-way range history |ptx-p| + |prx-p| for a 2-D geometry.

    ``p``   : (2,)      target coordinates (x = ground range, y = azimuth)
    ``ptx`` : (2, Na)   transmitter track
    ``prx`` : (2, Na)   receiver track
    Returns (Na,).  Note this is the SUM of the two legs, i.e. twice the
    one-way range in the monostatic case.
    """
    return (np.sqrt((prx[1, :] - p[1]) ** 2 + (prx[0, :] - p[0]) ** 2)
            + np.sqrt((ptx[1, :] - p[1]) ** 2 + (ptx[0, :] - p[0]) ** 2))


def get_echo(rmin, Nr, rbw, cd, rsf, wl, R):
    """
    One range line: a chirp of duration ``cd`` starting at delay ``2R/c``.

    ``R`` is HALF the two-way path length, so that ``2R/c`` is the round-trip
    delay.  The samples outside the recording window are left at zero; the
    function returns early (all zeros) when the echo falls completely outside.
    """
    echo = np.zeros(Nr, dtype=complex)
    nrg_start = int(np.ceil(rsf * 2 * (R - rmin) / C_LIGHT))
    nrg_end = int(np.floor(rsf * (2 * (R - rmin) / C_LIGHT + cd)))
    if 0 <= nrg_start < Nr:
        if nrg_end >= Nr:
            nrg_end = Nr - 1
    else:
        if 0 <= nrg_end < Nr:
            nrg_start = 0
        else:
            return echo

    fast_time = 2 * rmin / C_LIGHT + (nrg_start + np.arange(nrg_end - nrg_start + 1)) / rsf
    echo[nrg_start:nrg_end + 1] = np.exp(
        -1j * PI * rbw / cd * (fast_time - 2 * R / C_LIGHT - 0.5 * cd) ** 2
        - 4j * PI / wl * R
    )
    return echo


def get_valid_idx(ps, p, v, Na, prf, sq, theta, wa_tx):
    """
    Indices where the target ``p`` is inside the beam of the platform ``ps``.

    ``wa_tx`` is the transmit illumination mask; when it is all zeros (i.e. we
    are computing the transmit mask itself) only the beam condition is applied.
    """
    inst_sq = np.zeros(Na, np.float64)
    range_hist = np.sqrt((ps[1, :] - p[1]) ** 2 + (ps[0, :] - p[0]) ** 2)
    inst_sq[0:Na - 1] = np.arcsin(np.clip(np.diff(range_hist) * prf / v[1:Na], -1.0, 1.0))
    inst_sq[Na - 1] = 2 * inst_sq[Na - 2] - inst_sq[Na - 3]
    if len(np.where(wa_tx == 1)[0]) == 0:
        return np.where(abs(inst_sq) < (sq + theta / 2.0))[0]
    return np.array(np.where((abs(inst_sq) < (sq + theta / 2.0)) & (wa_tx == 1))).flatten()


# ---------------------------------------------------------------------------
def get_raw_data_2d(p, ptx, prx, v_tx, v_rx, sq_tx, sq_rx, theta_tx, theta_rx,
                    wl, prf, sigma_tgt, rbw, cd, rsf, r_scan, abw, ref_tgt=0,
                    verbose=False):
    """
    Port of ``dataGenerator.getRawData2D_test``.

    ``p``        [Np, 2]     scatterer coordinates (ground range, azimuth)
    ``ptx``      [2, Na]     transmitter track
    ``prx``      [Nrx, 2, Na] receiver tracks
    ``v_tx``     [Na]        transmitter speed
    ``v_rx``     [Nrx, Na]   receiver speeds
    ``r_scan``   [Nr]        slant range of each range bin

    Returns ``(data, data_ch, rh_ch, f_max, f_min)`` -- see the module docstring.
    """
    Na = len(ptx[0, :])
    Nrx = len(prx[:, 0, 0])
    Np = len(p[:, 0])
    Nr = len(r_scan)

    data = np.zeros([Na, Nr], np.complex64)
    rh_ch = np.zeros([Nrx, Na], np.float64)
    data_ch = np.zeros([Nrx, Na, Nr], np.complex64)
    wa_tx = np.zeros(Na, np.float64)

    # The "most bistatic" channel: the one whose phase centre is furthest from
    # the transmitter.  It sets the narrowest common Doppler band.
    mb_idx = int(np.where((ptx[1, 0] - prx[:, 1, 0])
                          == np.max(ptx[1, 0] - prx[:, 1, 0]))[0][0])

    # --- transmit illumination window of the reference target --------------
    msVld_idx = get_valid_idx(ptx, p[ref_tgt, :], v_tx, Na, prf, sq_tx, theta_tx, wa_tx)
    wa_tx[msVld_idx] = 1.0

    # --- common Doppler band ------------------------------------------------
    f_max = np.max(-1 / wl * np.diff(get_range_hist(p[ref_tgt, :], ptx, ptx)[msVld_idx]) * prf)
    bsVld_ref = get_valid_idx(prx[mb_idx, :, :].reshape([2, Na]), p[ref_tgt, :],
                              v_rx[mb_idx, :].flatten(), Na, prf,
                              sq_rx[mb_idx], theta_rx[mb_idx], wa_tx)
    f_min = np.min(-1 / wl * np.diff(
        get_range_hist(p[ref_tgt, :], ptx, prx[mb_idx, :, :].reshape([2, Na]))[bsVld_ref] * prf))

    if verbose:
        print(f"dataGen: common Doppler band  f_min = {f_min:.1f} Hz, "
              f"f_max = {f_max:.1f} Hz  (abw = {abw:.1f} Hz)")

    # --- monostatic reference at the full equivalent PRF -------------------
    for p_idx in range(Np):
        rh_ms = get_range_hist(p[p_idx, :], ptx, ptx)
        f_ms = -1 / wl * np.diff(rh_ms[msVld_idx] * prf)
        valid_idx = msVld_idx[np.where((f_ms < f_max) & (f_ms > f_min))[0]]
        for ii in range(np.size(valid_idx)):
            data[valid_idx[ii], :] += sigma_tgt[p_idx] * get_echo(
                r_scan[0], Nr, rbw, cd, rsf, wl, rh_ms[valid_idx[ii]] / 2)

        # --- per-channel bistatic signals ----------------------------------
        for kk in range(Nrx):
            bsVld_idx = get_valid_idx(prx[kk, :, :].reshape([2, Na]), p[p_idx, :],
                                      v_rx[kk, :].flatten(), Na, prf,
                                      sq_rx[kk], theta_rx[kk], wa_tx)
            rh_bs = get_range_hist(p[p_idx, :], ptx, prx[kk, :, :].reshape([2, Na]))
            f_bs = -1 / wl * np.diff(rh_bs[bsVld_idx] * prf)
            # [FIX 1] the reference indexed msVld_idx with a mask computed on
            # bsVld_idx.  With deltaX != 0 the two index sets differ, which
            # shifted the illumination window of every bistatic channel.
            valid_idx_ch = bsVld_idx[np.where((f_bs < f_max) & (f_bs > f_min))[0]]
            if p_idx == int(Np / 2):
                rh_ch[kk, valid_idx_ch] = rh_bs[valid_idx_ch]
            for ii in range(np.size(valid_idx_ch)):
                data_ch[kk, valid_idx_ch[ii], :] += sigma_tgt[p_idx] * get_echo(
                    r_scan[0], Nr, rbw, cd, rsf, wl, rh_bs[valid_idx_ch[ii]] / 2.0)

    return data, data_ch, rh_ch, f_max, f_min
