import numpy as np

def zoom1Dpeak(s, N, zpf, Ndc=0):
    """
    Extract and oversample a centered window around the peak of a 1D signal.

    The function:
        1. finds the peak of |s|;
        2. circularly shifts the signal so the peak is centered;
        3. extracts a window of length 2*N around the peak;
        4. transforms it to frequency domain;
        5. zero-pads the spectrum by a factor zpf;
        6. returns the oversampled time-domain signal.

    Parameters
    ----------
    s : ndarray of shape (Na,)
        Input complex 1D signal.
    N : int
        Half-window size around the centered peak. The extracted window has
        length 2*N.
    zpf : int
        Zero-padding factor used for oversampling.
    Ndc : int, optional
        Doppler/DC shift correction applied before and after zero-padding.

    Returns
    -------
    s_zoom : ndarray of shape (2*N*zpf,)
        Oversampled complex signal around the original peak.
    """
    Na = len(s)
    ii = np.argmax(abs(s))

    sc = np.roll(s, int(0.5 * Na - ii))
    sc = sc[int(0.5 * Na - N):int(0.5 * Na + N)]

    sc = np.roll(np.fft.fft(sc), int(-Ndc))

    s2 = np.zeros(N * 2 * zpf, dtype=np.complex128)
    s2[:N] = sc[:N]
    s2[-N:] = sc[N:]

    return np.fft.ifft(np.roll(s2, int(Ndc)))