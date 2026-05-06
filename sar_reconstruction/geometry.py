import numpy as np

def build_geometry(Na, ta, v, H, r0, b_at, b_xt, h0=0.0):
    """
    Build the transmitter/receiver geometry for a simplified bistatic or
    multistatic SAR scenario.

    The geometry assumes:
        - azimuth direction along the x-axis;
        - cross-track direction along the y-axis;
        - height/vertical direction along the z-axis;
        - transmitter moving as Tx(t) = (v t, 0, H);
        - receiver i moving as Rx_i(t) = (v t - b_at[i], b_xt[i], H);
        - one point target located at (0, y0, h0), where y0 is chosen so that
          the transmitter slant range at ta = 0 is equal to r0.

    Parameters
    ----------
    Na : int
        Number of azimuth samples.
    ta : ndarray of shape (Na,)
        Azimuth slow-time vector [s].
    v : float
        Platform velocity in azimuth direction [m/s].
    H : float
        Platform altitude [m].
    r0 : float
        Reference transmitter slant range at ta = 0 [m].
    b_at : array_like of shape (Nrx,)
        Along-track baselines for each receiver [m].
    b_xt : array_like of shape (Nrx,)
        Cross-track baselines for each receiver [m].
    h0 : float, optional
        Target height [m]. Default is 0.0.

    Returns
    -------
    sceneMid : ndarray of shape (1, 3)
        Target position array [[x0, y0, h0]].
    ptx : ndarray of shape (Na, 3)
        Transmitter positions for each azimuth time.
    prx : ndarray of shape (Nrx, Na, 3)
        Receiver positions for each channel and azimuth time.
    vtx : ndarray of shape (Na,)
        Transmitter velocity magnitude for each azimuth sample.
    vrx : ndarray of shape (Nrx, Na)
        Receiver velocity magnitude for each channel and azimuth sample.

    Raises
    ------
    ValueError
        If b_at and b_xt do not have the same length.
    """
    b_at = np.asarray(b_at, dtype=np.float64)
    b_xt = np.asarray(b_xt, dtype=np.float64)

    Nrx = len(b_at)

    if len(b_xt) != Nrx:
        raise ValueError("b_at and b_xt must have the same length.")

    x0 = 0.0
    y0 = np.sqrt(r0**2 - (H - h0)**2)

    sceneMid = np.array([[x0, y0, h0]], dtype=np.float64)

    ptx = np.zeros((Na, 3), dtype=np.float64)
    ptx[:, 0] = v * ta
    ptx[:, 1] = 0.0
    ptx[:, 2] = H

    vtx = v * np.ones(Na, dtype=np.float64)

    prx = np.zeros((Nrx, Na, 3), dtype=np.float64)
    vrx = np.zeros((Nrx, Na), dtype=np.float64)

    for ii in range(Nrx):
        prx[ii, :, 0] = v * ta - b_at[ii]
        prx[ii, :, 1] = b_xt[ii]
        prx[ii, :, 2] = H
        vrx[ii, :] = v

    return sceneMid, ptx, prx, vtx, vrx
