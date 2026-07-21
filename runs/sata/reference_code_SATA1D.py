import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
matplotlib.pyplot.ion()

#Performs SATA(subaperture topography - and aperture - dependent motion compensation)
def sata_for_topography(data, delta_C0_array, rref, prf, Nsb, v, squint, wl, r, inverse=False, sata_osf=1):
    """
    data_ch[1,:],  prf, vs, sq_tx, wl , np.array([np.min(rh_ref)/2])
        data: [naz, nrg]
        pos: [naz, 4] the real sensor positions
        ref: [naz, 4] reference sensor positions
        dem: [nrg, naz]
        dem_smoothed: [nrg, naz] :smoothed version of DEM used in second order MoCo
        squint: the processing squint [rad]
        wl:
        v
        r: the range vector
        direction: left/right looking, default-> left direction = -1
        norcmc: SATA before RCMC
        sata_osf = SATA with oversampling
    """

    dimx = len(data)

    #-----------------------------------------------------------------------------
    # STEP 1: SATA pre - computations
    #-----------------------------------------------------------------------------

    #SATA specific
    #computing sub - aperture optimum size
    deltax = np.sqrt(wl * rref / 2) #this formula comes from the best resolution of SATA. AskPau for demonstration.
    Tsubeff = np.round(deltax * prf / v * 0.5 / Nsb) * 2
    #block sizes0
    Nzp = int(2 ** np.ceil(np.log10(Tsubeff) / np.log10(2)) * sata_osf)
    Tovl = int(Tsubeff * 0.5) #overlap
    Tsub = Tsubeff * 0.5 #assuming 50 % overlap
    Nsub = int(dimx // Tsub)
    marg_az = np.round(0.5 * (1 + Nzp - Tsubeff))

    print('')
    print(f'SATA: resolution in the accommodation of the topography at mid range: {deltax:.2f}')
    print(f'SATA: sub-aperture length: {Tsubeff}, zero padded to {Nzp:.2f} with margin {marg_az:.2f}')

    if (Tsubeff <= 2):
        print('SATA: sub-aperture length very small, no correction needed')
        return

    #weighting(50 % assumed)
    rightweight = np.arange(Tovl) / (Tovl - 1)
    leftweight = rightweight[::-1]

    #frequency axis of the subaperture
    fc = 2 * v / wl *np.sin(squint)
    pfc = np.round(np.mod(fc, prf) * Nzp / prf) #Doppler centroid in pixels
    dfc = np.round(fc * Nzp / prf) * prf / Nzp #Doppler centroid rounded to sample resolution
    fsub = Nsb*(np.arange(Nzp) * prf / Nzp / Nsb - prf * 0.5 / Nsb) + dfc # frequency axis in sub - aperture
    fsub[0] = prf * 0.5 + dfc  #correcting small error due to previous line
    fsub = np.roll(fsub, +int(Nzp / 2 + pfc)) #shifting to be aligned with data
    # angles
    betasub = np.arcsin(wl * fsub / (2 * v))
    betasub = betasub

    azpos = r * (np.tan(betasub) - np.tan(squint)) / v * prf

    #-----------------------------------------------------------------------------
    # STEP 2: loop to perform SATA
    #-----------------------------------------------------------------------------
    temp = np.zeros(int(2 * Tsub), complex)
    for m in range(Nsub):
        if m == Nsub-1:
            posix = (m + 0.5) * Tsub
            aux = data[int(m * Tsub):]
            Tsubeff = int(dimx - m * Tsub)
        else:
            posix = (m + 1) * Tsub
            # in_lim = (posix + [-Nzp / 2, Nzp / 2 - 1]).astype(int)
            # if in_lim[0] < 0: in_lim[0] = 0
            # if in_lim[1] > dimx: in_lim[1] = dimx
            # aux = np.roll(data[in_lim[0]:in_lim[1]], int(in_lim[0] - m * Tsub))
            aux = data[int(m * Tsub) : int((m+2) * Tsub)]

        #zero padding
        if Nzp != Tsubeff:
            aux = np.concatenate((aux, np.zeros([int(Nzp - Tsubeff)], complex)), axis=0)

        aux = np.fft.fft(aux)
        # plt.figure(3), plt.plot(abs(np.fft.fft(data[in_lim[0]:in_lim[1]])))

        # obtaining heights
        # checking positions do not lie outside the image
        posaux = (azpos + dimx/2 + (posix - dimx/2)).astype(int)
        posaux[posaux < 0] = 0
        posaux[posaux > dimx - 1] = dimx - 1
        #
        # posaux2 = (azpos + (m + 1) * Tsub).astype(int)
        # posaux2[posaux2 < 0] = 0
        # posaux2[posaux2 > dimx - 1] = dimx - 1

        # print(m, posaux)
        if not np.all(delta_C0_array[posaux]==0):
            # plt.close('all')
            # plt.figure(1), plt.plot(delta_C0_array[posaux])
            # plt.figure(2), plt.plot(abs(np.fft.fft(data[int(m * Tsub) : int((m+2) * Tsub)])))
            print(m, Nsub//2,'SATA for PT')
        if m == Nsub//2-1:
            print('halfway')
        # obtaining final correction
        ph = -2 * np.pi / wl * delta_C0_array[posaux]
        ph[np.where(np.isnan(ph))] = 0
        ph[np.where(np.isinf(ph))] = 0

        # applyingcorrection
        if inverse: aux *= np.exp(-1j * ph)
        else: aux *= np.exp(1j * ph)
        aux = np.fft.ifft(aux, axis=0)

        #assuming 50 % overlap


        if m == 0: data[0:int(Tsub)] = aux[:int(Tsub)]
        else:
            # if m == Nsub//2-1:
            #     print('halfway')
            aux[:int(Tsub)] = temp[int(Tsub):int(2*Tsub)] * leftweight + aux[:int(Tsub)] * rightweight
            data[int((m - 1) * Tsub):m * int(Tsub)] = temp[:int(Tsub)]
            temp = aux
        temp = aux

        # if m == 0: data[0:int(Tsub)] = temp[:int(Tsub)]
        # else: data[int((m - 1) * Tsub):m * int(Tsub)] = temp[:int(Tsub)]
        # temp = aux[:int(2*Tsub)]


#    last block
    data[int((m-1) * Tsub):] = temp[:int(dimx - (m-1) * Tsub)]

    # print
    # print, 'SATA: maximum peak-to-peak phase correction: ', strtrim(!radeg * max_ph, 1), ' deg'
    # min_osf = sata_osf + ceil(2 * (0.5 * max_ph /np.pi-marg_az) * (sata_osf / Nzp))
    # if (min_osf gt sata_osf) then begin
    # print, 'WARNING: margin is insufficient for corrections of this magnitude!'
    # print, 'WARNING: image probably contains artefacts.'
    # print, 'WARNING: increase sata_osf from ', strtrim(sata_osf, 1), ' to ', strtrim(min_osf, 1), ' to avoid this.'

    return data


def sata(data, pos, ref, dem, demsmoothed, rref, prf, v, squint, wl , r, direction=-1, norcmc=True,inverse=False, sata_osf=None):
    """
        data: [naz, nrg]
        pos: [naz, 4] the real sensor positions
        ref: [naz, 4] reference sensor positions
        dem: [nrg, naz]
        dem_smoothed: [nrg, naz] :smoothed version of DEM used in second order MoCo
        squint: the processing squint [rad]
        wl:
        v
        r: the range vector
        direction: left/right looking, default-> left direction = -1
        norcmc: SATA before RCMC
        sata_osf = SATA with oversampling
    """

    dimx, dimr = np.shape(data)
    rps = (r[dimr - 1] - r[0]) / (dimr - 1)

    #-----------------------------------------------------------------------------
    # STEP 1: SATA pre - computations
    #-----------------------------------------------------------------------------
    dy = ref[:, 2]-pos[:, 2]# moco y
    dz = ref[:, 3]-pos[:, 3]# moco z

    #SATA specific
    #computing sub - aperture optimum size
    deltax = np.sqrt(wl *rref / 2) #this formula comes from the best resolution of SATA. AskPau for demonstration.
    Tsubeff = np.round(deltax * prf / v * 0.5) * 2
    #block sizes
    if not sata_osf:
        sata_osf = 1
    Nzp = 2 ** np.ceil(np.log10(Tsubeff) / np.log10(2)) * sata_osf
    Tovl = int(Tsubeff * 0.5) #overlap
    Tovlhalf = int(Tovl * 0.5)
    Tsub = int(Tsubeff * 0.5) #assuming 50 % overlap)
    Nsub = int(dimx // Tsub)
    marg_az = np.round(0.5 * (1 + Nzp - Tsubeff))

    print('')
    print(f'SATA: resolution in the accommodation of the topography at mid range: {deltax:.2f}')
    print(f'SATA: sub-aperture length: {Tsubeff}, zero padded to {Nzp:.2f} with margin {marg_az:.2f}')

    if (Tsubeff <= 2):
        print('SATA: sub-aperture length very small, no correction needed')
    return

    #weighting(50 % assumed)
    rightweight = np.arange(Tovl) / (Tovl - 1)
    # rightweight = (rightweight > 0) < 1
    leftweight = rightweight[::-1]

    # rightweight = rightweight[:, np.newaxis]
    # leftweight = leftweight[:, np.newaxis]

    #frequency axis of the subaperture
    fc = 2 * v / wl *np.sin(squint)
    pfc = np.round(np.mod(fc, prf) * Nzp / prf) #Doppler centroid in pixels
    dfc = np.round(fc * Nzp / prf) * prf / Nzp #Doppler centroid rounded to sample resolution
    fsub = np.arange(Nzp) * prf / Nzp - prf * 0.5 + dfc # frequency axis in sub - aperture
    fsub[0] = prf * 0.5 + dfc  #correcting small error due to previous line
    fsub = np.roll(fsub, +int(Nzp / 2 + pfc)) #shifting to be aligned with data
    # angles
    betasub = np.sin(wl *fsub / (2 * v))
    betasub = betasub[:,np.newaxis]

    rm = r[np.newaxis,:]
    azpos = rm * (np.tan(betasub) - np.tan(squint)) / v * prf
    posvector = np.arange(dimr).repeat(Nzp).reshape([dimr,Nzp]).T

    if norcmc:
        #correct DEM lookup tables if no RCMC preformed...
        azpos *= np.cos(betasub)
        posrange = rm * np.cos(betasub)
        for m in range(Nzp):
            cs = CubicSpline(r, np.arange(dimr))
            posvector[m, :] = cs(posrange[m, :])
        posvector[posvector < 0] = 0
        posvector[posvector > dimr - 1] = dimr - 1
        #range indices into zero - doppler DEM
        h_ind = (r * np.cos(squint) - r[0]) / rps
        h_ind[h_ind < 0] = 0
        h_ind[h_ind > dimr - 1] = dimr - 1

    rmtanbetasub2 = (rm * np.tan(betasub)) ** 2
    rm_cosbetasub = rm / np.cos(betasub)

    #for non - rcm corrected data...
    rmsinbetasub2 = (rm * np.sin(betasub)) ** 2
    rm_invcos2 = (rm * np.cos(betasub)) ** 2
    r_cossq2 = (r * np.cos(squint)) ** 2
    r_sinsq2 = (r * np.sin(squint)) ** 2

    #-----------------------------------------------------------------------------
    # STEP 2: loop to3perform3SATA
    #-----------------------------------------------------------------------------
    max_ph = 0.0
    temp = np.array([2 * Tsub, dimr], complex)

    for m in range(Nsub):
        if m == Nsub-1:
            posix = (m + 0.5) * Tsub
            aux = data[m * Tsub:,:]
            Tsubeff = dimx - m * Tsub
        else:
            posix = (m + 1) * Tsub
            in_lim = posix + [-Nzp / 2, Nzp / 2 - 1]
            in_lim -= in_lim[0] < 0
            in_lim[1] <= dimx - 1
            aux = np.roll(data[in_lim[0]:in_lim[1],:], int(in_lim[0] - m * Tsub), axis=0)

        #zero padding
        Csubeff = np.shape(aux)[0]
        if Nzp != Csubeff:
            aux = np.array([aux, np.array([Nzp - Tsubeff, dimr], complex)])

        aux = np.fft.fft(aux, axis=0)

        # obtaining heights
        # checking positions do not lie outside the image
        posaux = int(azpos + (m + 1) * Tsub)
        posaux[posaux < 0] = 0
        posaux[posaux > dimr - 1] = dimr - 1

        # altitude above ground
        heights = ref[posix, 3] - dem[posaux * dimr + posvector]

        # computing correction applied with smoothed DEM during second order MoCo

        if norcmc:
            h0 = ref[posix, 3] - demsmoothed[h_ind, posix]

        # computing correction applied with smoothed DEM during second order MoCo
            y0 = np.sqrt(r_cossq2 - h0 ** 2) - direction * dy[posix]
            rreal = np.sqrt((h0 - dz[posix]) ** 2 + y0 ** 2 + r_sinsq2)
            corrsmootheddem = (rreal - r)[:, np.newaxis]

            # computing newcorrection with DEM
            y0 = np.sqrt(rm_invcos2 - heights ** 2) - direction * dy[posix]
            rreal = np.sqrt((heights - dz[posix]) ** 2 + y0 ** 2 + rmsinbetasub2)
            dr = (rreal - rm)
        else:
            h0 = ref[posix, 3] - demsmoothed[:, posix]

            # computingcorrection applied with smoothed DEM during second order MoCo
            y0 = np.sqrt(r ** 2 - h0 ** 2) - direction * dy[posix]
            rreal = np.sqrt((h0 - dz[posix]) ** 2 + y0 ** 2 + (r * np.tan(squint)) ** 2)
            dr = (rreal - r / np.cos(squint))
            corrsmootheddem = dr[:,np.newaxis]

            # computing new correction with DEM
            y0 = np.sqrt(rm ** 2 - heights ** 2) - direction * dy[posix]
            rreal = np.sqrt((heights - dz[posix]) ** 2 + y0 ** 2 + rmtanbetasub2)
            dr = (rreal - rm_cosbetasub)

        # obtaining final correction
        ph = 4 * np.pi / ql * (dr-corrsmootheddem)
        ph[np.where(np.isnan(ph))] = 0
        ph[np.where(np.isinf(ph))] = 0
        # # posaux = np.where(finite(ph, /nan) or finite(ph, / inf), nn)
        # if nn > 0: ph[posaux] = 0
        # max_ph >= 2 * max(np.mean(abs(ph), axis=0))

        # applyingcorrection
        if inverse: aux *= exp(-1j * ph)
        else: aux *= exp(1j * ph)
        aux = np.fft.ifft(aux, axis=0)

        #assuming 50 % overlap
        aux[:Tsub - 1,:] = temp[Tsub:, :] * leftweight + aux[:Tsub - 1, :] * rightweight

        if m == 0: data[0:Tsub - 1, :] = temp[:Tsub - 1, :]
        else: data[(m - 1) * Tsub:m * Tsub - 1,:] = temp[:Tsub - 1,:]
        temp = aux

#    last block
    data[(m - 1) * Tsub:,:] = temp[:dimx - (m - 1) * Tsub - 1,:]

    # print
    # print, 'SATA: maximum peak-to-peak phase correction: ', strtrim(!radeg * max_ph, 1), ' deg'
    # min_osf = sata_osf + ceil(2 * (0.5 * max_ph /np.pi-marg_az) * (sata_osf / Nzp))
    # if (min_osf gt sata_osf) then begin
    # print, 'WARNING: margin is insufficient for corrections of this magnitude!'
    # print, 'WARNING: image probably contains artefacts.'
    # print, 'WARNING: increase sata_osf from ', strtrim(sata_osf, 1), ' to ', strtrim(min_osf, 1), ' to avoid this.'

    return data
