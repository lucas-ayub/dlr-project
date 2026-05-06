import numpy as np

def getRawData1D(ptgs, ptx, prx, vtx, vrx, ta,
                 sq_tx, sq_rx, theta_tx, theta_rx, wl, prf):

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