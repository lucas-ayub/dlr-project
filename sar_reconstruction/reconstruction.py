import numpy as np

def GetCoeffNu(ptg, ptx, prx, vtx, vrx, pax, vax,
               prf, wl, ta, sq_tx, sq_rx,
               theta_tx, theta_rx,
               N_time=2, dN_time=2):

    rhT = np.sqrt(np.sum((ptx - ptg[np.newaxis, :]) ** 2, axis=1))
    inst_sqT = np.arcsin(np.gradient(rhT, 1 / prf) / vtx)
    valid_idxT = np.where(abs(inst_sqT) <= (sq_tx + theta_tx / 2))[0]

    rhA = np.sqrt(np.sum((pax - ptg[np.newaxis, :]) ** 2, axis=1))
    inst_sqA = np.arcsin(np.gradient(rhA, 1 / prf) / vax)
    valid_idxA = np.where(abs(inst_sqA) <= (sq_tx + theta_tx / 2))[0]

    if valid_idxT.all() != valid_idxA.all():
        print("Transmitter and active sensor valid pixels do not match!")

    rhR = np.sqrt(np.sum((prx - ptg[np.newaxis, :]) ** 2, axis=1))
    inst_sqR = np.arcsin(np.gradient(rhR, 1 / prf) / vrx)
    valid_idxR = np.where(abs(inst_sqR) <= (sq_rx + theta_rx / 2))[0]

    taCommon = np.intersect1d(ta[valid_idxT], ta[valid_idxR])
    idx_com = np.nonzero(np.in1d(ta, taCommon))[0]

    rhMS = 2 * rhT[idx_com]
    rhBS = (rhA + rhR)[idx_com]

    fi_ms = -1 / wl * np.diff(2 * rhT[idx_com]) * prf
    fi_bs = -1 / wl * np.diff(rhT[idx_com] + rhR[idx_com]) * prf

    f_max = np.min([abs(np.max(fi_ms)), abs(np.max(fi_bs))])
    f_min = -np.min([abs(np.min(fi_ms)), abs(np.min(fi_bs))])

    vld_ms = np.where((fi_ms < f_max) & (fi_ms > f_min))[0]
    vld_bs = np.where((fi_bs < f_max) & (fi_bs > f_min))[0]

    rh_ms = rhMS[vld_ms]
    rh_bs = rhBS[vld_bs]
    ta_ms = taCommon[vld_ms]
    ta_bs = taCommon[vld_bs]

    idx_ms = np.argmin(rh_ms)
    idx_bs = np.argmin(rh_bs)

    np_r = np.min([len(rh_ms[idx_ms:]), len(rh_bs[idx_bs:])])
    np_l = np.min([len(rh_ms[:idx_ms]), len(rh_bs[:idx_bs])])

    rh_ms = rh_ms[idx_ms - np_l:idx_ms + np_r]
    ta_ms = ta_ms[idx_ms - np_l:idx_ms + np_r]
    rh_bs = rh_bs[idx_bs - np_l:idx_bs + np_r]
    ta_bs = ta_bs[idx_bs - np_l:idx_bs + np_r]

    idx_ms = np.argmin(rh_ms)
    idx_bs = np.argmin(rh_bs)

    if idx_ms != idx_bs:
        print("Minimum indices are not same!")

    tbc_ms = ta_ms[idx_ms]
    tbc_bs = ta_bs[idx_bs]

    c_time = np.polyfit(ta_ms - tbc_ms, rh_ms, N_time)[::-1]
    dc_time = np.polyfit(ta_bs - tbc_bs, rh_bs - rh_ms, dN_time)[::-1]

    C0 = (
        dc_time[0]
        + c_time[1] ** 2 / (4 * c_time[2])
        - (c_time[1] + dc_time[1]) ** 2 / (4 * (c_time[2] + dc_time[2]))
    )

    C1 = (
        (c_time[2] * dc_time[1] - c_time[1] * dc_time[2])
        / (2 * c_time[2] * (c_time[2] + dc_time[2]))
    )

    C2 = dc_time[2] / (4 * c_time[2] * (c_time[2] + dc_time[2]))

    Dt = tbc_bs - tbc_ms

    return C0, C1, C2, Dt


def GetInversionFilters(Hf):
    Na_ch = Hf.shape[0]
    Nsb = Hf.shape[1]
    Nrx = Hf.shape[2]

    iHf = np.empty([Na_ch * Nsb, Nrx], dtype=np.complex64)

    id_mat = np.identity(Nrx) * Nrx
    Ti = np.repeat(id_mat[np.newaxis, :, :], Na_ch, axis=0)

    for jj in range(Nsb):
        rhs = Ti[:, :, jj].reshape([Na_ch, Nrx])
        iHf[jj * Na_ch:(jj + 1) * Na_ch, :] = np.linalg.solve(Hf, rhs)

    return iHf


def ReconstructSignalNumeri(data_ch, prfCh, wl, sceneMid, ta,
                            ptx, prx, vtx, vrx, pax, vax,
                            sq_tx, sq_rx, theta_tx, theta_rx,
                            ve, abw,
                            zeroOutBw=False):

    Nrx, Na_ch, Nr = np.shape(data_ch)

    Nsb = Nrx
    prfFinal = prfCh * Nrx
    Na = Na_ch * Nrx

    fsub = -prfFinal / 2 + np.arange(Na_ch) * prfCh / Na_ch

    srec = np.zeros([Na, Nr], dtype=np.complex64)

    Nsh = int(Na_ch / 2)
    if Nrx % 2 == 0:
        Nsh = 0

    for kk in range(Nrx):
        data_ch[kk, :, :] = np.roll(
            np.fft.fft(data_ch[kk, :, :], axis=0),
            Nsh,
            axis=0,
        )

    hf = np.zeros([Na_ch, Nsb, Nrx], dtype=np.complex64)

    C0 = np.zeros(Nrx)
    C1 = np.zeros(Nrx)
    C2 = np.zeros(Nrx)
    Dt = np.zeros(Nrx)

    for mm in range(Nr):

        print("\nCHANNEL COEFFICIENTS")
        print("====================")

        for kk in range(Nrx):
            C0[kk], C1[kk], C2[kk], Dt[kk] = GetCoeffNu(
                sceneMid[:, mm],
                ptx,
                prx[kk, :, :],
                vtx,
                vrx[kk, :],
                pax,
                vax,
                prfFinal,
                wl,
                ta,
                sq_tx,
                sq_rx[kk],
                theta_tx,
                theta_rx[kk],
            )

            print(
                f"ch {kk}: "
                f"C0={C0[kk]:+.6e}, "
                f"-C1+Dt={(-C1[kk] + Dt[kk]):+.6e}, "
                f"C2={C2[kk]:+.6e}, "
                f"Dt={Dt[kk]:+.6e}"
            )

        for jj in range(Nsb):
            fa_jj = fsub + jj * prfCh

            for ii in range(Nrx):
                hf[:, jj, ii] = np.exp(
                    -2j * np.pi * (
                        C0[ii] / wl
                        + (-C1[ii] + Dt[ii]) * fa_jj
                        + C2[ii] * wl * fa_jj ** 2
                    )
                )

        iHf = GetInversionFilters(hf)

        for kk in range(Nsb):
            for jj in range(Nrx):
                srec[kk * Na_ch:(kk + 1) * Na_ch, mm] += (
                    data_ch[jj, :, mm]
                    * iHf[kk * Na_ch:(kk + 1) * Na_ch, jj]
                )

    if zeroOutBw:
        fa = -prfFinal / 2 + np.arange(Na) * prfFinal / Na

        abw_idx = np.concatenate(
            (
                np.where(fa < -abw / 2)[0],
                np.where(fa > abw / 2)[0],
            ),
            axis=0,
        )

        srec[abw_idx, :] = 0

    srec = np.fft.ifft(np.roll(srec, int(Na / 2), axis=0), axis=0)

    return srec
