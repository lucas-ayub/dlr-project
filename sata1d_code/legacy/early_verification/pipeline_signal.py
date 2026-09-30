"""
pipeline_signal.py
==================
`getRawData1D` copied VERBATIM from the repo's
`sar_reconstruction/sar_recon/signal_model.py`, plus a tiny broadside
linear-track builder so the verification can feed it the way the pipeline does.
Only numpy is needed (no config/geometry package).
"""
import numpy as np


# --- VERBATIM from sar_recon/signal_model.py -------------------------------
def getRawData1D(ptgs, ptx, prx, vtx, vrx, ta, sq_tx, sq_rx,
                 theta_tx, theta_rx, wl, prf):
    Na = len(ta)
    Np = len(ptgs[:, 0])

    inst_sq_tx = np.zeros(Na, np.float64)
    inst_sq_rx = np.zeros(Na, np.float64)
    datal = np.zeros(Na, np.complex128)

    for p_idx in range(Np):
        wa_tx = np.zeros(Na)
        wa_rx = np.zeros(Na)

        rh_ms = np.sqrt(np.sum((ptx - ptgs[p_idx, :][np.newaxis, :]) ** 2, axis=1))
        inst_sq_tx[0:Na - 1] = np.arcsin(np.diff(rh_ms) * prf / vtx[1:Na])
        inst_sq_tx[Na - 1] = 2 * inst_sq_tx[Na - 2] - inst_sq_tx[Na - 3]
        wa_tx[np.where(abs(inst_sq_tx) <= (sq_tx + theta_tx / 2))] = 1

        rh_bs = np.sqrt(np.sum((prx - ptgs[p_idx, :][np.newaxis, :]) ** 2, axis=1))
        inst_sq_rx[0:Na - 1] = np.arcsin(np.diff(rh_bs) * prf / vrx[1:Na])
        inst_sq_rx[Na - 1] = 2 * inst_sq_rx[Na - 2] - inst_sq_tx[Na - 3]
        wa_rx[np.where(abs(inst_sq_rx) <= (sq_rx + theta_rx / 2))] = 1

        aPattern = wa_tx * wa_rx
        rh = rh_ms + rh_bs
        datal += aPattern * np.exp(-2j * np.pi * rh / wl)
    return datal


# --- broadside monostatic geometry (so getRawData1D has tracks to run on) ---
def broadside_scene(r0, H, v, wl, prf, Na, beam_band_frac=0.9):
    """One point target at the scene centre, broadside, monostatic linear track.

    Azimuth = x; the platform flies along x at height H; the target is on the
    ground at the cross-range that puts it at slant range r0. The beamwidth
    theta is set so the illuminated Doppler band is ~beam_band_frac * prf.
    Returns everything getRawData1D needs."""
    y0 = np.sqrt(r0**2 - H**2)                       # cross-range of the target
    ptg = np.array([[0.0, y0, 0.0]])                 # [1,3] at azimuth cell Na/2
    t_rel = (np.arange(Na) - Na // 2) / prf
    ptx = np.stack([v * t_rel, np.zeros(Na), np.full(Na, H)], axis=1)  # [Na,3]
    vtx = np.full(Na, v)
    ta = np.arange(Na) / prf
    beta_max = np.arcsin(beam_band_frac * (prf / 2) * wl / (2 * v))
    theta = 2 * beta_max                             # full beamwidth [rad]
    return dict(ptgs=ptg, ptx=ptx, prx=ptx, vtx=vtx, vrx=vtx, ta=ta,
                sq=0.0, theta=theta, wl=wl, prf=prf)


def pipeline_point_line(r0, H, v, wl, prf, Na):
    """The azimuth line of one broadside point target, produced by the repo's
    getRawData1D. This is exactly what SATA receives (1 channel, range done)."""
    g = broadside_scene(r0, H, v, wl, prf, Na)
    return getRawData1D(g["ptgs"], g["ptx"], g["prx"], g["vtx"], g["vrx"],
                        g["ta"], g["sq"], g["sq"], g["theta"], g["theta"],
                        g["wl"], g["prf"])
