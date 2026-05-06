import numpy as np

def build_geometry(Na, ta, v, H, r0, b_at, b_xt, h0=0.0):
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
