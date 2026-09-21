# -*- coding: utf-8 -*-
"""
Reconstruction error budget -- figure for the cache written by run_check.py.

Four panels, all normalised to the monostatic reference:

(a) WHERE the error is: |S_rec - S_ref| / |S_ref| over the 2-D spectrum
    (range frequency x Doppler frequency), no SATA.  Diverging colour map
    centred on 0 dB: red = the error is larger than the signal itself,
    blue = smaller.
(b) The same ratio against Doppler frequency (RMS over range frequency), for
    no SATA and SATA per sub-band, with the in-band RMS in the legend.  Above
    0 dB a phase-difference map arg(S_rec S_ref*) is meaningless -- the error
    phasor is as long as the signal, so the angle can be anything.
(c) The azimuth impulse response over the whole slow-time axis, with the
    expected ambiguity positions t_k = k PRF_op / Ka marked.
(d) The main lobe, FFT-interpolated so its shape can actually be read.

Every number in the figure (geometry, Sigma, peaks, in-band ratios) is read
from the cache -- nothing is hard-coded, so re-running run_check.py with a
different geometry updates the figure consistently.

Run from ``sar_reconstruction/``::

    python -m sata2d.run_check && python -m sata2d.plot_esr
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import TwoSlopeNorm                       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
C_NO, C_SUB, C_REF = "#C62828", "#2E7D32", "#222222"


def _get(d, key, default=None):
    return d[key] if key in d.files else default


def smooth(y, n=41):
    """Running mean, for a readable trend on top of the raw (noisy) curve."""
    k = np.ones(n) / n
    return np.convolve(np.pad(y, n // 2, mode="edge"), k, mode="valid")


def interp_lobe(x, pk, half, zp=16):
    """FFT zero-padding interpolation of ``x[pk-half : pk+half]``."""
    w = x[pk - half:pk + half]
    W = np.fft.fftshift(np.fft.fft(w))
    n = len(w)
    P = np.zeros(n * zp, complex)
    P[(n * zp - n) // 2:(n * zp + n) // 2] = W
    return np.fft.ifft(np.fft.ifftshift(P)) * zp


def main():
    d = np.load(os.path.join(OUT, "cache_check.npz"))
    fa, fr = d["fa"], d["fr"]
    abw, prf, ve, pk = float(d["abw"]), float(d["prf"]), float(d["ve"]), int(d["pk"])
    mag = d["mag_ref"]
    valid = mag > 0.02 * mag.max()

    # --- facts for the labels (fallbacks keep old caches plottable) ----------
    Nrx = int(_get(d, "Nrx", 4))
    bxt = _get(d, "bxt")
    bat = _get(d, "bat")
    h = float(_get(d, "height", np.nan))
    sigma = float(_get(d, "sigma", np.nan))
    dpca_res = float(_get(d, "dpca_res", np.nan))
    inb = np.abs(fa) < abw / 2
    esr_no = float(_get(d, "esr_no", np.mean(d["esr_fa_no"][inb])))
    esr_sub = float(_get(d, "esr_sub", np.mean(d["esr_fa_sub"][inb])))
    amp_no, amp_sub = _get(d, "esr_amp_no"), _get(d, "esr_amp_sub")

    irf = {"mono": d["irf_mono"], "no": d["irf_no"], "sub": d["irf_sub"]}
    nrm = np.abs(irf["mono"]).max()
    pct = {k: 100 * np.abs(v).max() / nrm for k, v in irf.items()}

    def db(v):
        return 20 * np.log10(np.abs(v) / nrm + 1e-20)

    Na = len(fa)
    t = (np.arange(Na) - pk) / prf

    # --- figure --------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 10.5), dpi=150)
    arr = ""
    if bxt is not None:
        arr = (f"$b_{{xt}}$ = [{', '.join(f'{b:.0f}' for b in bxt)}] m"
               + (f",  $b_{{at}}$ step = {np.diff(bat).mean():.0f} m" if bat is not None
                  and len(bat) > 1 else ""))
        if np.isfinite(dpca_res):
            arr += "  (on DPCA)" if dpca_res < 1e-6 else "  (OFF the DPCA condition)"
    fig.suptitle(
        "Reconstruction error budget -- single target, 2-D exact reference\n"
        f"$N_{{rx}}$ = {Nrx}  |  {arr}  |  $h$ = {h:.0f} m  |  "
        f"channel spread $\\Sigma$ = {sigma:.2f} cycles  |  "
        f"PRF = {prf:.0f} Hz, $B_a$ = {abw:.0f} Hz",
        fontsize=11)

    # (a) 2-D error map ------------------------------------------------------
    a = ax[0, 0]
    img = np.where(valid, d["esr2d_no"], np.nan)
    im = a.imshow(img, origin="lower", aspect="auto", cmap="RdBu_r",
                  norm=TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20),
                  extent=[fr.min() / 1e6, fr.max() / 1e6, fa.min(), fa.max()])
    a.set_facecolor("#DDDDDD")
    cb = fig.colorbar(im, ax=a, extend="both")
    cb.set_label(r"error / signal  $|S_{rec}-S_{ref}|\,/\,|S_{ref}|$  [dB]")
    cb.ax.text(1.0, 1.04, "error > signal", transform=cb.ax.transAxes,
               ha="center", fontsize=7.5, color=C_NO)
    cb.ax.text(1.0, -0.07, "error < signal", transform=cb.ax.transAxes,
               ha="center", fontsize=7.5, color="#1F4E79")
    for s in (+1, -1):
        a.axhline(s * abw / 2, color="k", ls="--", lw=0.9)
    a.text(fr.min() / 1e6 + 0.05, abw / 2 + 15, r"$+B_a/2$ (processed band edge)",
           va="bottom", ha="left", fontsize=8)
    a.text(fr.min() / 1e6 + 0.05, -abw / 2 - 15, r"$-B_a/2$", va="top", ha="left",
           fontsize=8)
    a.set_xlabel("range frequency $f_r$ [MHz]")
    a.set_ylabel("azimuth (Doppler) frequency $f_a$ [Hz]")
    a.set_title("(a) where the error is -- 2-D spectrum, no SATA\n"
                "white: error as large as the signal  |  grey: masked (no signal)",
                fontsize=10)

    # (b) error vs Doppler ---------------------------------------------------
    b = ax[0, 1]
    for key, c, lab, val, amp in (("no", C_NO, "no SATA", esr_no, amp_no),
                                  ("sub", C_SUB, "SATA per sub-band", esr_sub, amp_sub)):
        y = d[f"esr_fa_{key}"]
        b.plot(fa, np.where(inb, y, np.nan), color=c, lw=0.4, alpha=0.30)
        extra = f", phase removed {float(amp):+.1f} dB" if amp is not None else ""
        b.plot(fa, np.where(inb, smooth(y), np.nan), color=c, lw=1.8,
               label=f"{lab}:  in-band RMS {val:+.1f} dB{extra}")
    b.axhline(0, color="k", lw=1.0)
    b.text(-990, 0.6, "error = signal (0 dB)", fontsize=8, va="bottom")
    b.axvspan(-1000, -abw / 2, color="0.88", zorder=0)
    b.axvspan(abw / 2, 1000, color="0.88", zorder=0)
    b.text(-(abw / 2 + 1000) / 2, -21, "outside\nprocessed\nband", ha="center",
           fontsize=8, color="0.35")
    b.text((abw / 2 + 1000) / 2, -21, "outside\nprocessed\nband", ha="center",
           fontsize=8, color="0.35")
    b.set_xlim(-1000, 1000)
    b.set_ylim(-25, 12)
    b.set_xlabel("azimuth (Doppler) frequency $f_a$ [Hz]")
    b.set_ylabel("error / signal [dB]  (RMS over $f_r$)")
    b.set_title("(b) the same, against Doppler frequency\n"
                "thin: raw per bin, thick: running mean", fontsize=10)
    b.grid(alpha=.3)
    b.legend(fontsize=8, loc="upper right", framealpha=0.95)

    # (c) full azimuth IRF with ambiguities ----------------------------------
    c = ax[1, 0]
    for key, col, lw, lab, z in (
            ("mono", "0.55", 0.5, "monostatic reference  (100 %)", 1),
            ("no", C_NO, 0.6, f"no SATA  (peak {pct['no']:.1f} %)", 2),
            ("sub", C_SUB, 0.6, f"SATA per sub-band  (peak {pct['sub']:.1f} %)", 3)):
        c.plot(t, db(irf[key]), color=col, lw=lw, label=lab, zorder=z)
    vs, wl, r0, prf_op = (_get(d, k) for k in ("vs", "wl", "r0", "prf_op"))
    if vs is not None:
        ka = 2 * float(vs) ** 2 / (float(wl) * float(r0))
        for k in (-2, -1, 1, 2):
            tk = k * float(prf_op) / ka
            c.axvline(tk, color="0.4", ls=":", lw=0.9)
            c.text(tk, 1.5, f"k={k:+d}", ha="center", va="bottom", fontsize=8,
                   color="0.3")
    c.set_ylim(-100, 8)
    c.set_xlabel("azimuth time $t_a$ [s]")
    c.set_ylabel("impulse response [dB re. monostatic peak]")
    c.set_title("(c) azimuth impulse response, full slow-time axis\n"
                r"dotted: expected ambiguities $t_k = k\,\mathrm{PRF}_{op}/K_a$",
                fontsize=10)
    sec = c.secondary_xaxis("top", functions=(lambda s: s * ve / 1e3,
                                               lambda x: x * 1e3 / ve))
    sec.set_xlabel("ground azimuth $x = v_e t_a$ [km]")
    c.grid(alpha=.3)
    c.legend(fontsize=8, loc="upper left", framealpha=0.95, bbox_to_anchor=(0.0, 0.93))

    # (d) interpolated main lobe ---------------------------------------------
    e = ax[1, 1]
    half, zp = 64, 16
    xm = (np.arange(2 * half * zp) - half * zp) / zp / prf * ve       # [m]
    sel = np.abs(xm) <= 40
    for key, col, ls, lab in (("mono", C_REF, "--", "monostatic reference"),
                              ("no", C_NO, "-", "no SATA"),
                              ("sub", C_SUB, "-", "SATA per sub-band")):
        z = interp_lobe(irf[key], pk, half, zp)
        e.plot(xm[sel], db(z)[sel], color=col, ls=ls, lw=1.3, label=lab)
    for key, col in (("no", C_NO), ("sub", C_SUB)):
        lv = 20 * np.log10(pct[key] / 100)
        e.axhline(lv, color=col, lw=0.7, ls=":")
        e.text(39, lv + 0.6, f"{lv:+.1f} dB = {pct[key]:.1f} %", ha="right",
               fontsize=8, color=col)
    e.set_xlim(-40, 40)
    e.set_ylim(-45, 4)
    e.set_xlabel("ground azimuth $x$ [m]")
    e.set_ylabel("impulse response [dB re. monostatic peak]")
    e.set_title("(d) main lobe, FFT-interpolated x16\n"
                "peak level = how much of the target was recovered", fontsize=10)
    sec2 = e.secondary_xaxis("top", functions=(lambda x: x / ve * 1e3,
                                                lambda s: s * ve / 1e3))
    sec2.set_xlabel("azimuth time $t_a$ [ms]")
    e.grid(alpha=.3)
    e.legend(fontsize=8, loc="upper left", framealpha=0.95)

    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5, w_pad=2.0)
    o = os.path.join(OUT, "error_budget.png")
    fig.savefig(o, dpi=150, bbox_inches="tight")
    print("written", o)


if __name__ == "__main__":
    main()