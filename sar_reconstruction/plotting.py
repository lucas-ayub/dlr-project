import numpy as np
import matplotlib.pyplot as plt


def db_norm(x, eps=1e-12):
    """
    Convert a signal magnitude to normalized decibels.

    Parameters
    ----------
    x : array_like
        Input signal.
    eps : float, optional
        Small value used to avoid division by zero and log of zero.

    Returns
    -------
    x_db : ndarray
        Normalized magnitude of x in dB, with peak approximately at 0 dB.
    """
    return 20 * np.log10(np.abs(x) / (np.max(np.abs(x)) + eps) + eps)

def plot_results(
    ta, taz, sref, srec, srefF, srecF,
    u_ref, u_rec, fa, abw, abw_idx,
    case_title=None, b_at=None, b_xt=None,
):
    """
    Plot the main numerical reconstruction results.

    The function generates:
        1. A combined 2x2 summary figure;
        2. Individual larger figures for each diagnostic plot.

    The plotted diagnostics are:
        - azimuth signal envelope;
        - full impulse response / correlation;
        - zoomed impulse response main lobe;
        - residual Doppler-domain phase error.

    Parameters
    ----------
    ta : ndarray
        Azimuth slow-time vector [s].
    taz : ndarray
        Centered/zoomed azimuth time vector around the IRF peak [s].
    sref : ndarray
        Reference azimuth signal in time domain.
    srec : ndarray
        Reconstructed azimuth signal in time domain.
    srefF : ndarray
        Reference focused or correlation-domain signal.
    srecF : ndarray
        Reconstructed focused or correlation-domain signal.
    u_ref : ndarray
        Zoomed reference IRF around the main lobe.
    u_rec : ndarray
        Zoomed reconstructed IRF around the main lobe.
    fa : ndarray
        Azimuth/Doppler frequency vector [Hz].
    abw : float
        Processed azimuth bandwidth [Hz].
    abw_idx : ndarray or array_like of bool/int
        Indices or mask of Doppler samples outside the useful bandwidth.
        These samples are set to zero in the residual phase plot.
    """
    dph = np.angle(np.fft.fft(srecF) * np.conj(np.fft.fft(srefF)), deg=True)
    dph[abw_idx] = 0

    # Combined 2x2 figure
    plt.figure(figsize=(12, 9))

    if case_title is None:
        suptitle = "Numerical Reconstruction"
    else:
        suptitle = case_title

        if b_at is not None and b_xt is not None:
            b_at_str = np.array2string(np.asarray(b_at), precision=1, separator=", ")
            b_xt_str = np.array2string(np.asarray(b_xt), precision=1, separator=", ")

            suptitle += (
                "\n"
                rf"$b_{{at}}$ = {b_at_str} m"
                " | "
                rf"$b_{{xt}}$ = {b_xt_str} m"
            )

    plt.suptitle(suptitle, fontsize=14)

    plt.subplot(221)
    plt.title("1. Azimuth signal envelope")
    plt.plot(ta, np.abs(sref), label="Reference")
    plt.plot(ta, np.abs(srec), label="Reconstructed")
    plt.xlabel("Azimuth time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()

    plt.subplot(222)
    plt.title("2. Full IRF / correlation")
    plt.plot(ta, db_norm(srefF), label="Reference IRF")
    plt.plot(ta, db_norm(srecF), label="Reconstructed IRF")
    plt.xlabel("Azimuth time [s]")
    plt.ylabel("Normalized amplitude [dB]")
    plt.ylim([-100, 0])
    plt.grid()
    plt.legend()

    plt.subplot(223)
    plt.title("3. Zoomed IRF main lobe")
    plt.plot(taz * 1e3, db_norm(u_ref), label="Reference zoomed IRF")
    plt.plot(taz * 1e3, db_norm(u_rec), label="Reconstructed zoomed IRF")
    plt.xlabel("Azimuth time offset [ms]")
    plt.ylabel("Normalized amplitude [dB]")
    plt.grid()
    plt.legend()

    plt.subplot(224)
    plt.title("4. Residual phase error in Doppler")
    plt.plot(fa, dph, label=r"$\angle(S_{rec}S_{ref}^{*})$")
    plt.axvline(x=abw / 2, color="r", linestyle="-.", label="+ bandwidth")
    plt.axvline(x=-abw / 2, color="r", linestyle="-.", label="- bandwidth")
    plt.xlabel("Doppler frequency [Hz]")
    plt.ylabel("Residual phase [deg]")
    plt.grid()
    plt.legend()

    # Individual large figures
    plt.figure(figsize=(12, 6))
    plt.title("Azimuth signal envelope")
    plt.plot(ta, np.abs(sref), label="Reference")
    plt.plot(ta, np.abs(srec), label="Reconstructed")
    plt.xlabel("Azimuth time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()

    plt.figure(figsize=(12, 6))
    plt.title("Full IRF / correlation")
    plt.plot(ta, db_norm(srefF), label="Reference IRF")
    plt.plot(ta, db_norm(srecF), label="Reconstructed IRF")
    plt.xlabel("Azimuth time [s]")
    plt.ylabel("Normalized amplitude [dB]")
    plt.ylim([-100, 0])
    plt.grid()
    plt.legend()

    plt.figure(figsize=(12, 6))
    plt.title("Zoomed IRF main lobe")
    plt.plot(taz * 1e3, db_norm(u_ref), label="Reference zoomed IRF")
    plt.plot(taz * 1e3, db_norm(u_rec), label="Reconstructed zoomed IRF")
    plt.xlabel("Azimuth time offset [ms]")
    plt.ylabel("Normalized amplitude [dB]")
    plt.grid()
    plt.legend()

    plt.figure(figsize=(12, 6))
    plt.title("Residual phase error in Doppler")
    plt.plot(fa, dph, label=r"$\angle(S_{rec}S_{ref}^{*})$")
    plt.axvline(x=abw / 2, color="r", linestyle="-.", label="+ bandwidth")
    plt.axvline(x=-abw / 2, color="r", linestyle="-.", label="- bandwidth")
    plt.xlabel("Doppler frequency [Hz]")
    plt.ylabel("Residual phase [deg]")
    plt.grid()
    plt.legend()



def plot_delta_r_all_channels(ta, ptx, prx, sceneMid):
    """
    Plot the bistatic range deviation Delta r for all receiver channels.

    For each channel, the function computes:

        r_tx(t_a) = ||p_tx(t_a) - p_target||

        r_rx,i(t_a) = ||p_rx,i(t_a) - p_target||

        r_bi,i(t_a) = 0.5 * (r_tx(t_a) + r_rx,i(t_a))

        Delta r_i(t_a) = r_bi,i(t_a) - r_ref(t_a)

    where r_ref is taken as the transmitter range r_tx. It also fits a
    second-order polynomial to Delta r_i(t_a) and plots the fitted curve.

    Parameters
    ----------
    ta : ndarray of shape (Na,)
        Azimuth slow-time vector [s].
    ptx : ndarray of shape (Na, 3)
        Transmitter positions along azimuth.
    prx : ndarray of shape (Nrx, Na, 3)
        Receiver positions for each channel along azimuth.
    sceneMid : ndarray of shape (1, 3)
        Target position array. The first row is used as the point target.
    """
    ptg = sceneMid[0]

    r_tx = np.linalg.norm(ptx - ptg[None, :], axis=1)
    r_ref = r_tx

    plt.figure(figsize=(12, 6))
    plt.title(r"All channels: $\Delta r(t_a)$ and quadratic fits")

    for ch in range(prx.shape[0]):
        r_rx = np.linalg.norm(prx[ch] - ptg[None, :], axis=1)
        r_bi = 0.5 * (r_tx + r_rx)

        delta_r = r_bi - r_ref

        coeff = np.polyfit(ta, delta_r, 2)
        delta_r_fit = np.polyval(coeff, ta)

        plt.plot(ta, delta_r, label=fr"ch {ch} $\Delta r$")
        plt.plot(ta, delta_r_fit, "--", label=fr"ch {ch} fit")

    plt.xlabel("Azimuth time [s]")
    plt.ylabel(r"$\Delta r$ [m]")
    plt.grid()
    plt.legend()


def plot_delta_phi_H_all_channels(fa, abw, wl, coeffs):
    """
    Plot the Doppler-domain residual phase model used to build Hf.

    For each channel, the plotted phase is:

        Delta phi_H(f_a) = 2*pi * [
            C0 / wl
            + (-C1 + Dt) * f_a
            + C2 * wl * f_a^2
        ]

    This is the phase term used in the reconstruction transfer function Hf.

    Parameters
    ----------
    fa : ndarray
        Azimuth/Doppler frequency vector [Hz].
    abw : float
        Processed azimuth bandwidth [Hz].
    wl : float
        Radar wavelength [m].
    coeffs : dict
        Dictionary containing the channel-dependent coefficient arrays:
            - "C0"
            - "C1"
            - "C2"
            - "Dt"

        Each entry must have length Nrx.
    """
    C0 = coeffs["C0"]
    C1 = coeffs["C1"]
    C2 = coeffs["C2"]
    Dt = coeffs["Dt"]

    Nrx = len(C0)

    plt.figure(figsize=(12, 6))
    plt.title(r"All channels: $\Delta \phi_H(f_a)$ used in Hf")

    for ch in range(Nrx):
        delta_phi_H = 2 * np.pi * (
            C0[ch] / wl
            + (-C1[ch] + Dt[ch]) * fa
            + C2[ch] * wl * fa**2
        )

        plt.plot(fa, delta_phi_H, label=f"ch {ch}")

    plt.axvline(x=abw / 2, color="r", linestyle="-.", label="+ bandwidth")
    plt.axvline(x=-abw / 2, color="r", linestyle="-.", label="- bandwidth")
    plt.xlabel("Doppler frequency [Hz]")
    plt.ylabel(r"$\Delta \phi_H(f_a)$ [rad]")
    plt.grid()
    plt.legend()
