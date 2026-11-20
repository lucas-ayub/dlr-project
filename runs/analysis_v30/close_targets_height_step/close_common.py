"""Shared setup for close_targets_height_step: re-uses joint_height_dense/common.py (same system, generators,
focusing, joint reconstruction and verified ambiguity metric) and adds local metrics for close target pairs."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util as _iu                                           # noqa: E402
_spec = _iu.spec_from_file_location("jhd_common", os.path.join(HERE, "..", "joint_height_dense", "common.py"))
_c = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_c)
S, relative, simulate, amb_new, nmse_db = _c.S, _c.relative, _c.simulate, _c.amb_new, _c.nmse_db
import numpy as np                                                     # noqa: E402

CACHE = os.path.join(HERE, "cache")
FIG = os.path.join(HERE, "fig")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def local_profile(r, key, x0, x1, os_=16, pad=40):
    """|line| upsampled x os_ (zero padding in frequency of a window around [x0, x1]); returns x [m], |y| re. the
    reference peak (linear)."""
    t, line, vs = r["t"], r[key], r["vs"]
    dt = t[1] - t[0]
    i0 = int(np.floor((x0 / vs - t[0]) / dt)) - pad
    i1 = int(np.ceil((x1 / vs - t[0]) / dt)) + pad
    n = i1 - i0
    n += n % 2
    seg = line[i0:i0 + n]
    U = np.fft.fftshift(np.fft.fft(seg))
    Up = np.zeros(n * os_, complex)
    c = Up.size // 2
    Up[c - n // 2:c + n // 2] = U
    s = np.abs(np.fft.ifft(np.fft.ifftshift(Up))) * os_
    x = (t[i0] + np.arange(Up.size) * dt / os_) * vs
    return x, s / np.abs(r["ref"]).max()


def pair_metrics(r, key, xs, cell=5.77):
    """Per target: peak level [dB re. reference peak] and position error [m] (only when the targets are more than 2
    cells apart, else NaN); for each neighbouring pair: dip between the two peaks [dB re. the weaker peak]."""
    xs = np.sort(np.asarray(xs, float))
    x, s = local_profile(r, key, xs[0] - 30.0, xs[-1] + 30.0)
    pk, pos = [], []
    for k, xt in enumerate(xs):
        gap = min([abs(xt - q) for q in xs if q != xt] or [1e9])
        w = min(0.5 * gap, 6.0)
        sel = np.abs(x - xt) <= w
        j = np.argmax(np.where(sel, s, -1))
        pk.append(20 * np.log10(s[j] + 1e-15))
        pos.append(float(x[j] - xt) if gap > 2 * cell else np.nan)
    dips = []
    for a, b in zip(xs[:-1], xs[1:]):
        sel = (x >= a) & (x <= b)
        lo = s[sel].min()
        pa = s[np.abs(x - a) <= min(0.5 * (b - a), 6.0)].max()
        pb = s[np.abs(x - b) <= min(0.5 * (b - a), 6.0)].max()
        dips.append(20 * np.log10(lo / min(pa, pb) + 1e-15))
    return pk, pos, dips


def local_err(r, key, xs, half=200.0):
    """Normalised error vs the reference on the stretch [min(xs) - half, max(xs) + half] [dB]."""
    x = r["t"] * r["vs"]
    sel = (x >= min(xs) - half) & (x <= max(xs) + half)
    y, ref = r[key][sel], r["ref"][sel]
    return float(20 * np.log10(np.linalg.norm(y - ref) / np.linalg.norm(ref)))
