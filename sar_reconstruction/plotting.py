import numpy as np
import matplotlib.pyplot as plt


def db_norm(x, eps=1e-12):
    return 20 * np.log10(np.abs(x) / (np.max(np.abs(x)) + eps) + eps)


def plot_results(
    ta, taz, sref, srec, srefF, srecF,
    u_ref, u_rec, fa, abw, abw_idx,
):
    dph = np.angle(np.fft.fft(srecF) * np.conj(np.fft.fft(srefF)), deg=True)
    dph[abw_idx] = 0

    # Combined 2x2 figure
    plt.figure(figsize=(12, 9))
    plt.suptitle("NIDA Numerical Reconstruction", fontsize=15)

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
