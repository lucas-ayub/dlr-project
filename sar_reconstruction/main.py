import os
import numpy as np
import matplotlib.pyplot as plt

from signal_model import getRawData1D
from geometry import build_geometry
from reconstruction import ReconstructSignalNumeri
from utils import zoom1Dpeak
from plotting import plot_results, plot_delta_models_all_channels


def main():
    # ============================================================
    # Parameters
    # ============================================================


    wl = 0.25

    v = 7408.5313923924796
    ve = v

    r0 = 700e3
    H = 514e3
    h0 = 0.0

    da = 24 * wl
    Tint = (ve / da) / 2.0 / ve**2 * wl * r0
    
    
    #===================================
    #            NIDA CASE
    #===================================
    # Nrx = 5
    # dx = 11.0
    # b_at = dx * np.arange(Nrx) / Nrx
    # b_at -= b_at[Nrx // 2]
    # b_xt = np.zeros(Nrx)
    
    b_at = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    b_xt = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
    
    b_at = np.array([-4.4, -2.2, 0.0, 2.2, 4.4])
    b_xt = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    Nrx = len(b_at)   

    prf = 1500.0
    abw = prf / 1.2
    PRF_op = prf / Nrx

    acq_time = 2 * Tint

    divfac = 1024
    Na = int(np.ceil(acq_time * prf / Nrx / divfac) * Nrx * divfac)
    Na_ch = int(Na / Nrx)

    ta = (np.arange(Na) - Na * 0.5) / prf

    sq_tx = 0.0
    La = 2 * da

    theta_tx = wl / La
    theta_rx = wl / La * np.ones(Nrx)
    sq_rx = np.zeros(Nrx, dtype=np.float64)

    # ============================================================
    # Geometry
    # ============================================================

    sceneMid, ptx, prx, vtx, vrx = build_geometry(
    Na=Na,
    ta=ta,
    v=v,
    H=H,
    r0=r0,
    b_at=b_at,
    b_xt=b_xt,
    )

    pax = ptx.copy()
    vax = vtx.copy()

    # ============================================================
    # Reference signal
    # ============================================================

    sref = getRawData1D(
        sceneMid,
        ptx,
        ptx,
        vtx,
        vtx,
        ta,
        sq_tx,
        sq_tx,
        theta_tx,
        theta_tx,
        wl,
        prf,
    )

    # ============================================================
    # Multichannel signal
    # ============================================================

    s_channel = np.zeros((Nrx, Na_ch), dtype=np.complex128)

    for ii in range(Nrx):
        sig_i = getRawData1D(
            sceneMid,
            ptx,
            prx[ii, :, :],
            vtx,
            vrx[ii, :],
            ta,
            sq_tx,
            sq_rx[ii],
            theta_tx,
            theta_rx[ii],
            wl,
            prf,
        )

        s_channel[ii, :] = sig_i[::Nrx]

    # ============================================================
    # NIDA numerical reconstruction
    # ============================================================

    srec = ReconstructSignalNumeri(
        s_channel.reshape((Nrx, Na_ch, 1)),
        PRF_op,
        wl,
        sceneMid.reshape((3, 1)),
        ta,
        ptx,
        prx,
        vtx,
        vrx,
        pax,
        vax,
        sq_tx,
        sq_rx,
        theta_tx,
        theta_rx,
        ve * np.ones(Na_ch),
        abw,
        True,
    )

    srec = srec.flatten()

    # ============================================================
    # IRF / correlation
    # ============================================================

    fa = np.roll((np.arange(Na) / Na - 0.5) * prf, int(Na / 2))

    abw_idx = np.concatenate(
        (
            np.where(fa < -0.5 * abw)[0],
            np.where(fa > 0.5 * abw)[0],
        ),
        axis=0,
    )

    srefF = np.roll(
        np.fft.ifft(np.fft.fft(sref) * np.conjugate(np.fft.fft(sref))),
        int(Na / 2),
    )

    srecF = np.roll(
        np.fft.ifft(np.fft.fft(srec) * np.conjugate(np.fft.fft(sref))),
        int(Na / 2),
    )

    N = 16
    zpf = 64

    taz = (np.arange(2 * N * zpf) - N * zpf) / prf / zpf

    u_ref = zoom1Dpeak(srefF, N, zpf)
    u_rec = zoom1Dpeak(srecF, N, zpf)

    # ============================================================
    # Plot
    # ============================================================

    plot_results(
        ta=ta,
        taz=taz,
        sref=sref,
        srec=srec,
        srefF=srefF,
        srecF=srecF,
        u_ref=u_ref,
        u_rec=u_rec,
        fa=fa,
        abw=abw,
        abw_idx=abw_idx,
    )
    
    plot_delta_models_all_channels(
        ta=ta,
        ptx=ptx,
        prx=prx,
        sceneMid=sceneMid,
        wl=wl,
    )

    # ============================================================
    # SAVE FIGURES BY FOLDER
    # ============================================================

    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

    dirs = {
        "delta_r": os.path.join(out_root, "delta_r"),
        "delta_phi": os.path.join(out_root, "delta_phi"),
        "general": os.path.join(out_root, "general"),
    }

    for folder in dirs.values():
        os.makedirs(folder, exist_ok=True)

    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)

        title = ""
        if fig._suptitle is not None:
            title = fig._suptitle.get_text().lower()

        axes_titles = " ".join(
            ax.get_title().lower()
            for ax in fig.axes
            if ax.get_title()
        )

        all_titles = title + " " + axes_titles

        if "delta r" in all_titles or "δr" in all_titles:
            folder = dirs["delta_r"]
            prefix = "delta_r"
        elif "delta phi" in all_titles or "δφ" in all_titles or "phase error" in all_titles:
            folder = dirs["delta_phi"]
            prefix = "delta_phi"
        else:
            folder = dirs["general"]
            prefix = "general"

        fig.tight_layout()
        filename = os.path.join(folder, f"{prefix}_figure_{i + 1}.png")
        fig.savefig(filename, dpi=180)
        print(f"saved: {filename}")

    print("end")


if __name__ == "__main__":
    main()
