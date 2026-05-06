import numpy as np

def zoom1Dpeak(s, N, zpf, Ndc=0):
    Na = len(s)
    ii = np.argmax(abs(s))

    sc = np.roll(s, int(0.5 * Na - ii))
    sc = sc[int(0.5 * Na - N):int(0.5 * Na + N)]

    sc = np.roll(np.fft.fft(sc), int(-Ndc))

    s2 = np.zeros(N * 2 * zpf, dtype=np.complex128)
    s2[:N] = sc[:N]
    s2[-N:] = sc[N:]

    return np.fft.ifft(np.roll(s2, int(Ndc)))