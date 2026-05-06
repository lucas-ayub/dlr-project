import numpy as np

def getRawData1D(ptgs, ptx, prx, vtx, vrx, ta,
                 sq_tx, sq_rx, theta_tx, theta_rx, wl, prf):
    """
    Simulate 1D bistatic SAR raw data in azimuth for one receiver channel.

    For each point target, the function computes the transmitter and receiver
    range histories, applies simple rectangular antenna visibility windows based
    on instantaneous squint angles, and adds the corresponding complex phase
    history to the raw azimuth signal.

    The simulated phase uses the total bistatic path:

        rh(t_a) = r_tx(t_a) + r_rx(t_a)

    and the signal model:

        s(t_a) = exp(-j 2*pi*rh(t_a)/wl)

    Parameters
    ----------
    ptgs : ndarray of shape (Np, 3)
        Point target positions. Each row is [x, y, z] in meters.
    ptx : ndarray of shape (Na, 3)
        Transmitter positions over azimuth time.
    prx : ndarray of shape (Na, 3)
        Receiver positions for one channel over azimuth time.
    vtx : ndarray of shape (Na,)
        Transmitter velocity magnitude over azimuth time [m/s].
    vrx : ndarray of shape (Na,)
        Receiver velocity magnitude over azimuth time [m/s].
    ta : ndarray of shape (Na,)
        Azimuth slow-time vector [s].
    sq_tx : float
        Transmitter squint angle limit or center parameter [rad].
    sq_rx : float
        Receiver squint angle limit or center parameter [rad].
    theta_tx : float
        Transmitter azimuth beamwidth [rad].
    theta_rx : float
        Receiver azimuth beamwidth [rad].
    wl : float
        Radar wavelength [m].
    prf : float
        Pulse repetition frequency [Hz].

    Returns
    -------
    datal : ndarray of shape (Na,)
        Simulated complex raw azimuth signal for one receiver channel.
    """

    Na = len(ta)
    Np = len(ptgs[:, 0])

    datal = np.zeros(Na, dtype=np.complex128)

    for p_idx in range(Np):
        inst_sq_tx = np.zeros(Na)
        inst_sq_rx = np.zeros(Na)
        wa_tx = np.zeros(Na)
        wa_rx = np.zeros(Na)

        rh_tx = np.linalg.norm(ptx - ptgs[p_idx], axis=1)
        inst_sq_tx[:-1] = np.arcsin(np.diff(rh_tx) * prf / vtx[1:])
        inst_sq_tx[-1] = 2 * inst_sq_tx[-2] - inst_sq_tx[-3]
        wa_tx[np.abs(inst_sq_tx) <= (sq_tx + theta_tx / 2)] = 1

        rh_rx = np.linalg.norm(prx - ptgs[p_idx], axis=1)
        inst_sq_rx[:-1] = np.arcsin(np.diff(rh_rx) * prf / vrx[1:])
        inst_sq_rx[-1] = 2 * inst_sq_rx[-2] - inst_sq_rx[-3]
        wa_rx[np.abs(inst_sq_rx) <= (sq_rx + theta_rx / 2)] = 1

        rh = rh_tx + rh_rx
        datal += (wa_tx * wa_rx) * np.exp(-2j * np.pi * rh / wl)

    return datal