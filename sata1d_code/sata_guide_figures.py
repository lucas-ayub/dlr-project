"""
Figures of the step-by-step guide to the SATA 1D kernel (reference: sata_for_topography in
runs/sata/reference_code_SATA1D.py, and the original IDL sata of sata.txt). Signals come from the sata1d
signal model. Writes in --out:

    g01_target_signal.png      signal of one target and its Doppler history
    g02_why_subapertures.png   three targets overlap in time and in the spectrum, not in the time-Doppler plane
    g03_one_subaperture.png    one sub-aperture: every target is a tone; its spectrum is a coarse image
    g04_length_tradeoff.png    length of the sub-aperture: frequency resolution against Doppler drift
    g05_zero_padding.png       zero padding samples the same spectrum on a finer grid
    g06_ruler.png              Doppler and azimuth offset of every FFT bin (reference formulas)
    g07_block_walkthrough.png  one block through every step of the loop
    g08_per_target.png         what the kernel does to the signal of every target taken alone
    g09_stitching.png          overlap and margins: four ways of cutting and joining the blocks
    g10_table_halfwidth.png    width of the region of the table that holds a point target
    g11_recon_scans.png        inside the reconstruction: zero-padding factor, table width, table offset

    python sata_guide_figures.py --out ../results/sata1d/figures/sata_guide
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sata1d as S
import sata_review_figures as R

XS, COL, LAB = R.XS, R.COL, R.LAB
HALF = 1000.0          # half-width of the region of the table that holds the elevated target [m]


def kernel(data, dC0, wl, r, prf, v, osf=1, inverse=True, overlap=True, margin="zeros", keep=None):
    """The loop of the kernel with its options made explicit.
    overlap : True = hop of half a sub-aperture and linear cross-fade (reference); False = blocks side by side.
    margin  : "zeros" = the block is completed with zeros up to Nzp (reference 1D);
              "data"  = the FFT takes Nzp real samples centred on the block and only the centre is kept (original IDL).
    keep    : index of one block whose intermediate arrays are returned."""
    dimx = len(data)
    _, Tsubeff, Tsub, Nzp = R.sizes(wl, r, prf, v, osf)
    _, _, azpos = R.ruler(wl, r, prf, v, Nzp)
    hop = Tsub if overlap else Tsubeff
    up = np.arange(Tsub) / (Tsub - 1.0)
    w = np.r_[up, up[::-1]] if overlap else np.ones(Tsubeff)
    out = np.zeros(dimx, complex); norm = np.zeros(dimx); kept = None
    for m in range((dimx - Tsubeff) // hop + 1):
        i0 = m * hop; posix = i0 + Tsubeff // 2
        if margin == "data":
            lo = posix - Nzp // 2
            if lo < 0 or lo + Nzp > dimx:
                continue
            S0 = np.fft.fft(data[lo:lo + Nzp])
        else:
            S0 = np.fft.fft(data[i0:i0 + Tsubeff], Nzp)
        posaux = np.clip((azpos + posix).astype(int), 0, dimx - 1)
        ph = -2 * np.pi / wl * dC0[posaux]
        S1 = S0 * np.exp(-1j * ph if inverse else 1j * ph)
        full = np.fft.ifft(S1)
        y = full[Nzp // 2 - Tsub:Nzp // 2 + Tsub] if margin == "data" else full[:Tsubeff]
        out[i0:i0 + Tsubeff] += w * y; norm[i0:i0 + Tsubeff] += w
        if keep is not None and m == keep:
            kept = dict(i0=i0, posix=posix, x=data[i0:i0 + Tsubeff].copy(), S0=S0, posaux=posaux, tab=dC0[posaux], ph=ph, S1=S1, full=full, y=y)
    ok = norm > 1e-9
    out[ok] /= norm[ok]; out[~ok] = data[~ok]
    return out, kept


def table(cfg, dx, value, half_m, prf, taper_m=0.0):
    """delta_C0 table on the grid of a line sampled at prf: value within +- half_m of the target at dx, zero elsewhere.
    taper_m > 0 replaces the two abrupt edges by smooth (raised-cosine) ramps of that length, centred on the edges."""
    n = int(round(cfg.Na * prf / cfg.prf))
    xc = (np.arange(n) - n / 2) * cfg.system.vs / prf - cfg.scene.x0
    if taper_m <= 0:
        return xc, np.where(np.abs(xc - dx) <= half_m, value, 0.0)
    u = np.clip((half_m - np.abs(xc - dx)) / taper_m + 0.5, 0.0, 1.0)
    return xc, value * 0.5 * (1 - np.cos(np.pi * u))


def scene():
    cfg, tr = R.setup()
    parts = [R.one_target(cfg, tr, j) for j in range(3)]
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    tj = [(cfg.scene.x0 + x) / cfg.system.vs for x in XS]
    return cfg, tr, parts, t, tj


def central_block(cfg, osf=1):
    """Index of the block whose centre is closest to t = 0, and the sizes."""
    _, Tsubeff, Tsub, Nzp = R.sizes(cfg.system.wl, cfg.scene.r0, cfg.prf, cfg.system.vs, osf)
    m0 = int(round((cfg.Na / 2 - Tsub) / Tsub))
    return m0, m0 * Tsub, Tsubeff, Tsub, Nzp


def pos_of(f, cfg):
    return cfg.scene.r0 * np.tan(np.arcsin(cfg.system.wl * f / (2 * cfg.system.vs)))


# ---------------------------------------------------------------------------------------------------------------
def g01(cfg, parts, t, tj, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    s = parts[1]; t0 = tj[1]; Ka = -2 * v ** 2 / (wl * r0)
    on = np.abs(s) > 0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.4))
    z = np.abs(t - t0) <= 0.16
    a1.plot(t[z] - t0, s[z].real, color="tab:blue", lw=1.0, label="real part")
    a1.plot(t[z] - t0, s[z].imag, color="tab:orange", lw=1.0, alpha=0.8, label="imaginary part")
    a1.set_xlabel("time from the closest approach of the target [s]"); a1.set_ylabel("signal (amplitude 1)")
    a1.set_title("Signal of one target: slow near the closest approach, faster away from it", fontsize=10.5)
    a1.grid(alpha=0.3); a1.legend(fontsize=9, loc="upper right"); a1.set_ylim(-1.45, 1.45)
    fi = np.diff(np.unwrap(np.angle(s[on]))) * prf / (2 * np.pi)
    tt = t[on][:-1] - t0
    a2.plot(tt[::25], fi[::25], "o", ms=3, color="tab:blue", label="measured on the signal (phase slope)")
    a2.plot(tt, Ka * tt, color="k", lw=1.0, label="$f = -(2v^2/\\lambda r)\\,(t - t_j)$,  slope %.0f Hz/s" % Ka)
    a2.axhline(0, color="0.6", lw=0.7); a2.axvline(0, color="0.6", lw=0.7)
    a2.text(-0.22, 560, "before the closest approach the target is\nahead of the platform: positive Doppler", fontsize=9, ha="center", va="center")
    a2.text(0.42, 300, "after it, the target is behind\nthe platform: negative Doppler", fontsize=9, ha="center", va="center")
    a2.set_xlabel("time from the closest approach of the target [s]"); a2.set_ylabel("Doppler frequency [Hz]")
    a2.set_title("Doppler history: a straight line while the target is inside the beam (%.2f s)" % (tt[-1] - tt[0]), fontsize=10.5)
    a2.grid(alpha=0.3); a2.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)
    print("g01: illumination %.3f s = %.2f km of flight, Doppler from %.0f to %.0f Hz, slope %.1f Hz/s" % (tt[-1] - tt[0], (tt[-1] - tt[0]) * v / 1e3, fi.max(), fi.min(), Ka))


def g02(cfg, parts, t, tj, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.5))
    for j, (p, c) in enumerate(zip(parts, COL)):
        axs[0].fill_between(t, 0, np.abs(p), color=c, alpha=0.25, lw=0)
        axs[0].plot(t, np.abs(p), color=c, lw=1.5, label="target at x = %s m" % LAB(XS[j]))
        f = np.fft.fftshift(np.fft.fftfreq(cfg.Na, 1 / prf)); A = np.abs(np.fft.fftshift(np.fft.fft(p)))
        axs[1].plot(f, A / A.max(), color=c, lw=1.4, ls=("-", "--", ":")[j], label="target at x = %s m" % LAB(XS[j]))
    axs[0].set_xlim(-1.7, 1.7); axs[0].set_ylim(0, 1.45); axs[0].set_xlabel("azimuth time [s]"); axs[0].set_ylabel("amplitude of the signal of each target")
    axs[0].set_title("In time: the three targets overlap\n(each one is in the beam for 2.08 s)", fontsize=10.5); axs[0].legend(fontsize=8.5, loc="upper center", ncol=1); axs[0].grid(alpha=0.3)
    axs[1].set_xlim(-1000, 1000); axs[1].set_ylim(0, 1.45); axs[1].set_xlabel("Doppler frequency [Hz]"); axs[1].set_ylabel("amplitude of the spectrum of each target")
    axs[1].set_title("In the spectrum of the whole line: they overlap too\n(each one fills the whole Doppler band)", fontsize=10.5); axs[1].legend(fontsize=8.5, loc="upper center"); axs[1].grid(alpha=0.3)
    _, spec, _, centre = R.sata_reference(sum(parts), np.zeros(cfg.Na), wl, r0, prf, v, r0, osf=2, apply=False)
    _, _, _, Nzp = R.sizes(wl, r0, prf, v, 2)
    f, _, _ = R.ruler(wl, r0, prf, v, Nzp); o = np.argsort(f)
    tc = (centre - cfg.Na / 2) / prf
    A = np.abs(spec[:, o]); A = 20 * np.log10(A / A.max() + 1e-6)
    axs[2].imshow(A.T, origin="lower", aspect="auto", extent=[tc[0], tc[-1], f[o][0], f[o][-1]], cmap="magma", vmin=-35, vmax=0)
    for x, c, tjj in zip(XS, COL, tj):
        axs[2].plot(tjj, 0, "o", ms=7, mfc=c, mec="w", mew=1.0)
    axs[2].set_xlim(-1.7, 1.7); axs[2].set_xlabel("time of the short piece of signal [s]"); axs[2].set_ylabel("Doppler frequency inside the piece [Hz]")
    axs[2].set_title("Short pieces of signal, one spectrum per piece:\nthe three targets are three separate lines", fontsize=10.5)
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)


def g03(cfg, parts, t, tj, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    m0, i0, Tsubeff, Tsub, _ = central_block(cfg)
    tc = t[i0 + Tsub]; n = np.arange(Tsubeff); tm = 1e3 * n / prf
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.15]))
    fexp = []
    for j, (p, c) in enumerate(zip(parts, COL)):
        blk = p[i0:i0 + Tsubeff]
        fj = 2 * v / (wl * r0) * (cfg.scene.x0 + XS[j] - v * tc); fexp.append(fj)
        axs[0].plot(tm, blk.real + 2.6 * (1 - j), color=c, lw=1.3)
        axs[0].text(tm[-1] + 0.6, 2.6 * (1 - j), "x = %s m\n%+.0f Hz" % (LAB(XS[j]), fj), color=c, fontsize=8.5, va="center")
    axs[0].set_xlim(0, tm[-1] + 12); axs[0].set_yticks([]); axs[0].set_xlabel("time inside the sub-aperture [ms]")
    axs[0].set_title("Each target alone, inside one sub-aperture\n(real part): almost a pure tone", fontsize=10.5); axs[0].grid(alpha=0.3, axis="x")
    blk = sum(parts)[i0:i0 + Tsubeff]
    axs[1].plot(tm, blk.real, color="0.15", lw=1.2, label="real part"); axs[1].plot(tm, blk.imag, color="0.6", lw=1.0, label="imaginary part")
    axs[1].set_xlabel("time inside the sub-aperture [ms]"); axs[1].set_title("What the radar records: the sum of the three\n(%d complex samples)" % Tsubeff, fontsize=10.5)
    axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8.5, loc="upper right"); axs[1].set_ylim(-3.6, 4.6)
    Nf = 4096; f = np.fft.fftshift(np.fft.fftfreq(Nf, 1 / prf)); A = np.abs(np.fft.fftshift(np.fft.fft(blk, Nf))) / Tsubeff
    axs[2].plot(f, A, color="0.15", lw=1.3)
    for fj, c, x in zip(fexp, COL, XS):
        axs[2].axvline(fj, color=c, lw=1.1, ls="--"); axs[2].text(fj, 1.12, "%+.0f Hz" % fj, color=c, ha="center", fontsize=8.5)
    axs[2].set_xlim(-480, 480); axs[2].set_ylim(0, 1.25); axs[2].set_xlabel("Doppler frequency [Hz]"); axs[2].set_ylabel("amplitude of the FFT of the sub-aperture")
    sec = axs[2].secondary_xaxis("top", functions=(lambda fq: pos_of(fq, cfg), lambda x: 2 * v / wl * np.sin(np.arctan(x / r0))))
    sec.set_xlabel("azimuth position relative to the centre of the sub-aperture [m]", fontsize=9)
    axs[2].grid(alpha=0.3)
    fig.suptitle("One sub-aperture of %d samples (%.0f ms, %.0f m of flight): the FFT separates the targets by their Doppler, that is, by their position" % (Tsubeff, 1e3 * Tsubeff / prf, Tsubeff * v / prf), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(png, dpi=170); plt.close(fig)
    print("g03: block of %d samples starting at %d, centre t = %.4f s, expected Doppler of the targets [Hz]:" % (Tsubeff, i0, tc), np.round(fexp, 1))


def g04(cfg, parts, t, tj, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    s = parts[1]; ic = cfg.Na // 2 + int(round(tj[1] * prf)); Nf = 16384
    f = np.fft.fftshift(np.fft.fftfreq(Nf, 1 / prf)); x = pos_of(f, cfg)
    opt = np.sqrt(wl * r0 / 2)

    def width(L):
        A = np.abs(np.fft.fftshift(np.fft.fft(s[ic - L // 2:ic + L // 2], Nf))); A /= A.max()
        k = np.flatnonzero(A >= 1 / np.sqrt(2))
        return A, x[k[-1]] - x[k[0]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 4.7))
    for L, c, lab in ((20, "tab:orange", "too short: the FFT cannot tell nearby frequencies apart"), (80, "tab:green", "the length the reference uses"),
                      (320, "tab:purple", "too long: the Doppler of the target drifts inside the block")):
        A, w = width(L)
        a1.plot(x, A, color=c, lw=1.5, label="%d samples = %.0f m of flight: %s (width %.0f m)" % (L, L * v / prf, lab, w))
    a1.axhline(1 / np.sqrt(2), color="0.6", lw=0.7, ls=":"); a1.text(-1950, 0.73, "$-3$ dB", fontsize=8, color="0.4")
    a1.set_xlim(-2000, 2000); a1.set_ylim(0, 1.12); a1.grid(alpha=0.3); a1.legend(fontsize=8.6, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    a1.set_xlabel("azimuth position given by the Doppler of the bin [m]"); a1.set_ylabel("spectrum of the sub-aperture (normalised)")
    a1.set_title("One target seen through sub-apertures of three lengths", fontsize=10.5)
    Ls = np.unique(np.round(np.geomspace(8, 800, 60) / 2).astype(int) * 2)
    W = np.array([width(L)[1] for L in Ls]); Lm = Ls * v / prf
    a2.loglog(Lm, W, "o-", ms=3.5, color="k", lw=1.0, label="measured width at $-3$ dB")
    a2.loglog(Lm, 0.886 * wl * r0 / (2 * Lm), color="tab:orange", lw=1.2, ls="--", label="frequency resolution: $0.886\\,\\lambda r / 2L$")
    a2.loglog(Lm, Lm, color="tab:purple", lw=1.2, ls="--", label="Doppler drift inside the block: $L$")
    a2.axvline(opt, color="tab:green", lw=1.3); a2.text(opt * 1.06, 2200, "$\\sqrt{\\lambda r/2}$ = %.0f m" % opt, color="tab:green", fontsize=9)
    a2.set_xlabel("length of the sub-aperture $L$ [m of flight]"); a2.set_ylabel("width of the response of one target [m]")
    a2.set_title("Width of the response against the length of the sub-aperture", fontsize=10.5); a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(); fig.savefig(png, dpi=170, bbox_inches="tight"); plt.close(fig)
    k = int(np.argmin(W))
    print("g04: optimum sqrt(wl r / 2) = %.1f m; narrowest measured response %.0f m at L = %.0f m; widths at 20/80/320 samples: %s m" % (
        opt, W[k], Lm[k], ", ".join("%.0f" % width(L)[1] for L in (20, 80, 320))))


def g05(cfg, parts, t, tj, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    m0, i0, Tsubeff, Tsub, _ = central_block(cfg)
    blk = sum(parts)[i0:i0 + Tsubeff]
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    Nf = 8192; f = np.fft.fftshift(np.fft.fftfreq(Nf, 1 / prf))
    ax.plot(pos_of(f, cfg), np.abs(np.fft.fftshift(np.fft.fft(blk, Nf))) / Tsubeff, color="0.6", lw=1.0, label="the spectrum of the block (very fine grid, for reference)")
    out = []
    for osf, c, mk, ms in ((1, "tab:red", "o", 7), (4, "tab:blue", ".", 6)):
        Nzp = int(2 ** np.ceil(np.log2(Tsubeff)) * osf); f = np.fft.fftshift(np.fft.fftfreq(Nzp, 1 / prf))
        dx = pos_of(prf / Nzp, cfg); out.append((osf, Nzp, dx))
        ax.plot(pos_of(f, cfg), np.abs(np.fft.fftshift(np.fft.fft(blk, Nzp))) / Tsubeff, mk, ms=ms, color=c, mfc="none" if osf == 1 else c,
                label="sata_osf = %d: FFT of %d points, one bin every %.0f m" % (osf, Nzp, dx))
    for x, c in zip(XS, COL):
        ax.axvline(x, color=c, lw=0.9, ls="--")
    ax.set_xlim(-3600, 3600); ax.set_ylim(0, 1.45); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper center")
    ax.set_xlabel("azimuth position given by the Doppler of the bin [m]"); ax.set_ylabel("amplitude of the FFT of the block")
    ax.set_title("Zero padding: the block still has %d samples; the FFT only reads its spectrum at more points" % Tsubeff, fontsize=10.5)
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)
    print("g05: bins:", ["osf %d -> Nzp %d, %.1f m per bin" % q for q in out])


def g06(cfg, png):
    """The frequency axis and the azimuth offsets exactly as the reference writes them, for one channel (PRF_op)."""
    wl, r, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.PRF_op
    Nsb, squint, Nzp = 1, 0.0, 32
    fc = 2 * v / wl * np.sin(squint)
    pfc = np.round(np.mod(fc, prf) * Nzp / prf)
    dfc = np.round(fc * Nzp / prf) * prf / Nzp
    fsub = Nsb * (np.arange(Nzp) * prf / Nzp / Nsb - prf * 0.5 / Nsb) + dfc
    fsub[0] = prf * 0.5 + dfc
    fsub = np.roll(fsub, +int(Nzp / 2 + pfc))
    betasub = np.arcsin(wl * fsub / (2 * v))
    azpos = r * (np.tan(betasub) - np.tan(squint)) / v * prf
    k = np.arange(Nzp)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 4.6))
    cols = ["tab:blue" if q > 0 else "tab:red" if q < 0 else "k" for q in fsub]; cols[Nzp // 2] = "tab:purple"
    for a, yv in ((a1, fsub), (a2, azpos)):
        a.vlines(k, 0, yv, colors=cols, lw=1.3); a.scatter(k, yv, c=cols, s=22, zorder=3)
        a.axhline(0, color="0.5", lw=0.7); a.grid(alpha=0.3); a.set_xlabel("index of the FFT bin, in the order the FFT returns them")
    a1.set_ylabel("Doppler frequency given to the bin, fsub [Hz]"); a1.set_ylim(-265, 285); a2.set_ylabel("azimuth offset of the bin, azpos [cells of the line]")
    a1.set_title("Step 1 of the ruler: bin $\\rightarrow$ Doppler frequency", fontsize=10.5)
    a2.set_title("Steps 2 and 3: Doppler $\\rightarrow$ angle $\\rightarrow$ offset in cells from the block centre", fontsize=10.5)
    a1.text(0.2, 262, "bin 0: zero Doppler\n(straight below the block centre)", fontsize=8.5, va="top")
    a1.text(8.0, -110, "bins 1 to %d: positive Doppler\n(ground ahead of the centre)" % (Nzp // 2 - 1), color="tab:blue", fontsize=8.5, ha="center", va="center")
    a1.text(25.0, 60, "bins %d to %d: negative Doppler\n(ground behind the centre)" % (Nzp // 2 + 1, Nzp - 1), color="tab:red", fontsize=8.5, ha="center", va="center")
    a1.annotate("bin %d: the edge $\\pm$PRF/2,\nlabelled $+$PRF/2" % (Nzp // 2), (Nzp // 2, fsub[Nzp // 2]), xytext=(19.0, 175), color="tab:purple", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="tab:purple"))
    sec = a2.secondary_yaxis("right", functions=(lambda c: c * v / prf / 1e3, lambda x: x * 1e3 * prf / v)); sec.set_ylabel("the same offset [km]")
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)
    print("g06: channel at %.0f Hz, Nzp = %d: bin spacing %.3f Hz = %.2f cells = %.1f m; azpos from %.1f to %.1f cells; fsub[Nzp/2] = %+.1f Hz" % (
        prf, Nzp, prf / Nzp, azpos[1], azpos[1] * v / prf, azpos.min(), azpos.max(), fsub[Nzp // 2]))


def bad_line(cfg, parts, dval):
    """The three targets, the first one with the phase error exp(-j 2 pi dval / wl)."""
    return parts[0] * np.exp(-2j * np.pi * dval / cfg.system.wl) + parts[1] + parts[2]


def g07(cfg, parts, t, tj, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    dval = wl / 4; osf = 2
    xc, tab = table(cfg, XS[0], dval, HALF, prf)
    m0, i0, Tsubeff, Tsub, Nzp = central_block(cfg, osf)
    _, K = kernel(bad_line(cfg, parts, dval), tab, wl, r0, prf, v, osf=osf, keep=m0)
    f, _, azpos = R.ruler(wl, r0, prf, v, Nzp); o = np.argsort(f); xb = azpos[o] * v / prf
    fig, axs = plt.subplots(2, 3, figsize=(17, 8.6)); a = axs.ravel()
    n = np.arange(Tsubeff)
    a[0].plot(n, K["x"].real, color="tab:blue", lw=1.2, label="real part"); a[0].plot(n, K["x"].imag, color="tab:orange", lw=1.0, label="imaginary part")
    a[0].set_xlabel("sample inside the block"); a[0].set_title("(a) the block cut from the line: %d complex samples" % Tsubeff, fontsize=10.5); a[0].legend(fontsize=8.5, loc="upper right"); a[0].set_ylim(-3.6, 4.8)
    a[1].plot(xb, np.abs(K["S0"][o]) / Tsubeff, "-", color="0.15", lw=1.2); a[1].plot(xb, np.abs(K["S0"][o]) / Tsubeff, ".", color="0.15", ms=4)
    a[1].set_title("(b) after zero padding and FFT: %d bins, each one a position on the ground" % Nzp, fontsize=10.5); a[1].set_ylabel("amplitude of the bin")
    a[2].plot(xb, 1e3 * K["tab"][o], ".-", color="tab:red", lw=1.0, ms=4)
    a[2].set_title("(c) value of the table read by each bin, $\\Delta C_0[\\mathrm{posaux}]$", fontsize=10.5); a[2].set_ylabel("$\\Delta C_0$ [mm]"); a[2].set_ylim(-5, 1e3 * dval * 1.25)
    rot = np.degrees(np.angle(K["S1"][o] * np.conj(K["S0"][o])))
    a[3].plot(xb, rot, ".-", color="tab:purple", lw=1.0, ms=4, label="rotation of the bin [deg]")
    a[3].set_ylim(-10, 115); a[3].set_ylabel("phase of (bin after / bin before) [deg]")
    b3 = a[3].twinx(); b3.plot(xb, np.abs(K["S1"][o]) / np.maximum(np.abs(K["S0"][o]), 1e-30), color="0.5", lw=1.0, ls="--"); b3.set_ylim(0, 1.3); b3.set_ylabel("amplitude after / before", color="0.4")
    a[3].set_title("(d) the multiplication: bins that look at the elevated region turn 90$^\\circ$;\nno amplitude changes (dashed)", fontsize=10.5)
    for k in (1, 2, 3):
        a[k].axvspan(XS[0] - HALF, XS[0] + HALF, color="tab:red", alpha=0.10, lw=0)
        for x, c in zip(XS, COL):
            a[k].axvline(x, color=c, lw=0.9, ls="--")
        a[k].set_xlim(-4500, 4500); a[k].set_xlabel("azimuth position that the bin looks at [m]")
    nn = np.arange(Nzp)
    before = np.r_[np.abs(K["x"]), np.zeros(Nzp - Tsubeff)]
    a[4].semilogy(nn, np.maximum(before, 1e-6), color="0.15", lw=1.2, label="before (block, then zeros)")
    a[4].semilogy(nn, np.maximum(np.abs(K["full"]), 1e-6), color="tab:blue", lw=1.0, label="after the inverse FFT")
    a[4].axvspan(0, Tsubeff - 1, color="tab:green", alpha=0.12, lw=0); a[4].text(Tsubeff / 2, 2.2e-4, "kept", color="tab:green", ha="center", fontsize=9)
    a[4].text((Nzp + Tsubeff) / 2, 2.2e-4, "discarded", color="0.35", ha="center", fontsize=9)
    leak = np.sum(np.abs(K["full"][Tsubeff:]) ** 2) / np.sum(np.abs(K["full"]) ** 2)
    a[4].set_ylim(1e-4, 30); a[4].set_xlabel("sample of the inverse FFT (%d points)" % Nzp); a[4].set_ylabel("amplitude")
    a[4].set_title("(e) back in time: %.1f %% of the energy moved into the padded part" % (100 * leak), fontsize=10.5); a[4].legend(fontsize=8.5, loc="upper right")
    a[5].plot(n, K["x"].real, color="0.15", lw=1.2, label="before"); a[5].plot(n, K["y"].real, color="tab:blue", lw=1.2, ls="--", label="after")
    a[5].set_xlabel("sample inside the block"); a[5].set_title("(f) the block before and after (real part)", fontsize=10.5); a[5].legend(fontsize=8.5, loc="upper right"); a[5].set_ylim(-3.6, 4.8)
    for ax in a:
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)
    print("g07: block %d (samples %d to %d, centre posix = %d), Nzp = %d; bins that read a non-zero table value: %d of %d; energy moved into the padded part: %.2f %%" % (
        m0, K["i0"], K["i0"] + Tsubeff - 1, K["posix"], Nzp, np.count_nonzero(K["tab"]), Nzp, 100 * leak))


def g08(cfg, parts, t, tj, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    dval = wl / 4
    _, tab = table(cfg, XS[0], dval, HALF, prf)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 4.5))
    rows = []
    for j, (p, c) in enumerate(zip(parts, COL)):
        out, _ = kernel(p, tab, wl, r0, prf, v, osf=4)
        on = np.abs(p) > 0; ratio = out[on] / p[on]; tt = t[on] - tj[j]
        a1.plot(tt, np.degrees(np.angle(ratio)), color=c, lw=1.2, label="target at x = %s m%s" % (LAB(XS[j]), " (inside the region of the table)" if j == 0 else ""))
        a2.plot(tt, 20 * np.log10(np.abs(ratio)), color=c, lw=1.2)
        mid = np.abs(tt) < 0.8
        rows.append((XS[j], np.degrees(np.angle(np.mean(ratio[mid]))), 20 * np.log10(np.abs(ratio[mid])).min(), 20 * np.log10(np.abs(ratio[mid])).max()))
    a1.set_ylabel("phase of (output / input) [deg]"); a1.set_ylim(-25, 120); a1.legend(fontsize=8.5, loc="center right")
    a1.set_title("Phase that the kernel gives to the signal of each target", fontsize=10.5)
    _, _, Tsub, _ = R.sizes(wl, r0, prf, v, 4)
    for q in np.arange(-0.1, 0.1001, Tsub / prf):
        a2.axvline(q, color="0.75", lw=0.6, zorder=0)
    a2.set_xlim(-0.1, 0.1)
    a2.set_ylabel("amplitude of (output / input) [dB]"); a2.set_ylim(-1.2, 1.2)
    a2.set_title("Amplitude, zoom on 0.2 s: a ripple of a few tenths of dB\n(grey lines: one hop between blocks, %.0f ms)" % (1e3 * Tsub / prf), fontsize=10.5)
    for ax in (a1, a2):
        ax.grid(alpha=0.3, axis="y" if ax is a2 else "both"); ax.set_xlabel("time from the closest approach of each target [s]")
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)
    print("g08: target x, mean phase of output/input [deg], amplitude range [dB] for |t| < 0.8 s:")
    for r_ in rows:
        print("   %+6.0f  %+7.2f   %+.3f .. %+.3f" % r_)


def focus_stats(cfg, tr, parts, t, line):
    """Focused line and, for every target, phase and amplitude relative to the ideal line; highest level away from the peaks."""
    v = cfg.system.vs
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    F = lambda s_: np.roll(np.fft.ifft(np.fft.fft(s_) * np.conj(np.fft.fft(sref))), cfg.Na // 2)
    ideal, foc = F(sum(parts)), F(line); pk = np.abs(ideal).max()
    rows = []
    for x in XS:
        i = int(np.argmin(np.abs(t - x / v))); i = i - 3 + int(np.argmax(np.abs(ideal[i - 3:i + 4])))
        rows.append((np.degrees(np.angle(foc[i] / ideal[i])), 20 * np.log10(abs(foc[i]) / abs(ideal[i]))))
    away = np.all([np.abs(t - x / v) > 0.03 for x in XS], axis=0) & (np.abs(t) < 0.9)
    return np.abs(foc) / pk, np.abs(ideal) / pk, rows, 20 * np.log10(np.abs(foc[away]).max() / pk)


def g09(cfg, tr, parts, t, tj, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    dval = wl / 4
    _, tab = table(cfg, XS[0], dval, HALF, prf); bad = bad_line(cfg, parts, dval)
    variants = [("(a) blocks side by side, no overlap", dict(osf=4, overlap=False)),
                ("(b) 50 % overlap and cross-fade, FFT of the bare block (sata_osf = 1)", dict(osf=1)),
                ("(c) 50 % overlap and cross-fade, zero padding (sata_osf = 4): the reference", dict(osf=4)),
                ("(d) 50 % overlap and cross-fade, margins of real data (original IDL, sata_osf = 1)", dict(osf=1, margin="data"))]
    fig, axs = plt.subplots(2, 2, figsize=(15, 8.0), sharex=True, sharey=True)
    print("g09: variant | elevated target: phase [deg], amplitude [dB] | neighbour at x = 0: phase [deg] | highest level away from the peaks [dB]")
    for ax, (name, kw) in zip(axs.ravel(), variants):
        out, _ = kernel(bad, tab, wl, r0, prf, v, **kw)
        foc, ideal, rows, spur = focus_stats(cfg, tr, parts, t, out)
        ax.plot(t, 20 * np.log10(foc + 1e-9), color="tab:blue", lw=0.9, label="after the kernel")
        ax.plot(t, 20 * np.log10(ideal + 1e-9), color="0.25", lw=0.9, label="ideal line (no error)")
        ax.set_title("%s\nhighest level away from the three peaks: %.1f dB" % (name, spur), fontsize=10)
        ax.set_xlim(-0.75, 0.75); ax.set_ylim(-70, 6); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="upper right")
        print("   %-82s %+7.2f %+6.3f | %+6.2f | %6.1f" % (name, rows[0][0], rows[0][1], rows[1][0], spur))
    for ax in axs[-1]:
        ax.set_xlabel("azimuth time [s]")
    for ax in axs[:, 0]:
        ax.set_ylabel("focused amplitude [dB]")
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)
    for kw in (dict(osf=2), dict(osf=8), dict(osf=2, margin="data"), dict(osf=4, margin="data"), dict(osf=1, overlap=False)):
        out, _ = kernel(bad, tab, wl, r0, prf, v, **kw)
        _, _, rows, spur = focus_stats(cfg, tr, parts, t, out)
        print("   %-82s %+7.2f %+6.3f | %+6.2f | %6.1f" % ("extra: " + str(kw), rows[0][0], rows[0][1], rows[1][0], spur))


def g10(cfg, tr, parts, t, tj, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    dval = wl / 4; bad = bad_line(cfg, parts, dval); opt = np.sqrt(wl * r0 / 2)
    halves = np.array([100, 200, 309, 450, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000.0])
    res, res_s = [], []
    for hw in halves:
        for store, taper in ((res, 0.0), (res_s, 600.0)):
            _, tab = table(cfg, XS[0], dval, hw, prf, taper)
            out, _ = kernel(bad, tab, wl, r0, prf, v, osf=4)
            _, _, rows, spur = focus_stats(cfg, tr, parts, t, out)
            store.append((rows[0][0], rows[0][1], rows[1][0], spur))
    res, res_s = np.array(res), np.array(res_s)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 4.5))
    a1.plot(halves, res[:, 3], "o-", color="k", lw=1.2, label="region with abrupt edges")
    a1.plot(halves, res_s[:, 3], "s--", color="tab:blue", lw=1.2, label="edges replaced by smooth ramps of 600 m")
    a1.legend(fontsize=8.5, loc="lower left")
    a1.set_ylabel("highest level away from the three peaks [dB]"); a1.set_title("Artefacts left by the kernel", fontsize=10.5)
    a2.plot(halves, np.abs(res[:, 0]), "o-", color="tab:red", lw=1.2, label="elevated target: phase error left (it started at 90$^\\circ$)")
    a2.plot(halves, np.abs(res[:, 2]), "s-", color="tab:green", lw=1.2, label="neighbour 2000 m away: phase error created")
    a2.set_ylabel("phase error of the peak [deg]"); a2.set_yscale("log"); a2.set_ylim(4e-3, 300); a2.legend(fontsize=8.5, loc="upper center"); a2.set_title("Phase of the peaks", fontsize=10.5)
    for ax in (a1, a2):
        ax.axvline(opt, color="tab:green", lw=1.0, ls=":"); ax.axvline(1000, color="0.4", lw=1.0, ls=":")
        ax.grid(alpha=0.3, which="both"); ax.set_xlabel("half-width of the region of the table around the elevated target [m]")
    a1.set_ylim(min(res[:, 3].min(), res_s[:, 3].min()) - 6, res[:, 3].max() + 9)
    a1.text(opt + 20, res[:, 3].max() + 8, "resolution of one\nsub-aperture (309 m)", color="tab:green", fontsize=8.5, va="top")
    a1.text(1020, res[:, 3].max() + 8, "half the distance\nto the neighbour", color="0.3", fontsize=8.5, va="top")
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)
    print("g10: half-width [m] | elevated: phase [deg], amplitude [dB] | neighbour phase [deg] | highest level [dB]")
    for hw, r_, q_ in zip(halves, res, res_s):
        print("   %6.0f | %+7.2f %+6.3f | %+6.2f | %6.1f   || smooth edges: %+7.2f %+6.3f | %+6.2f | %6.1f" % (hw, *r_, *q_))


def g11(png):
    from tx_rx_channel_cases import make_case
    from run_one_case import ambiguity_level
    targets = [(-1500.0, 1000.0), (1500.0, 0.0)]; bxt = [0.0, 10.0, 18.0, 30.0]
    cfg = make_case(targets, bxt); tr = S.build_platform_tracks(cfg)
    wl, r0, v, prf, prfc, N = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf, cfg.PRF_op, cfg.Nrx
    dC0 = np.asarray(S.delta_C0_per_target(cfg, tr)); ch = S.generate_channels(cfg, tr)
    kern = R.load_reference_kernel()
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    F = lambda s_: np.abs(np.roll(np.fft.ifft(np.fft.fft(s_) * np.conj(np.fft.fft(sref))), cfg.Na // 2))
    t = (np.arange(cfg.Na) - cfg.Na // 2) / prf; dt = prfc * wl * r0 / (2 * v ** 2); xs = np.array([x for x, _ in targets])
    ref = F(S.generate_reference(cfg, tr)); pk = ref.max()

    def run(osf=4, half=60, copies=True, offset=0, inverse=True):
        tabs, _ = R.channel_tables(cfg, dC0, half, copies)
        tabs = np.roll(tabs, offset, axis=1)
        e = F(S.reconstruct(cfg, tr, np.array([kern(ch[i].copy(), tabs[i], r0, prfc, 1, v, 0.0, wl, r0, inverse=inverse, sata_osf=osf) for i in range(N)])))
        i = int(np.argmin(np.abs(t - xs[0] / v)))
        return 20 * np.log10(e[i - 4:i + 5].max() / ref[i - 4:i + 5].max()), ambiguity_level(t, e, pk, xs, v, dt)

    e0 = F(S.reconstruct(cfg, tr, ch)); i = int(np.argmin(np.abs(t - xs[0] / v)))
    none = (20 * np.log10(e0[i - 4:i + 5].max() / ref[i - 4:i + 5].max()), ambiguity_level(t, e0, pk, xs, v, dt))
    scans = [("zero-padding factor sata_osf", [1, 2, 4, 8], lambda q: run(osf=q)),
             ("half-width of the regions of the table [cells of 15.4 m]", [5, 10, 20, 30, 45, 60, 75, 90], lambda q: run(half=q)),
             ("offset of the table relative to the data [cells of 15.4 m]", [0, 10, 20, 40, 60, 80, 120, 200], lambda q: run(offset=q))]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
    print("g11: no SATA: peak %.2f dB, highest ambiguity %.1f dB" % none)
    for ax, (name, vals, fn) in zip(axs, scans):
        res = np.array([fn(q) for q in vals])
        ax.plot(vals, res[:, 1], "o-", color="k", lw=1.2, label="highest ambiguity")
        ax.plot(vals, res[:, 0], "s--", color="tab:blue", lw=1.2, label="main-lobe peak of the elevated target")
        ax.axhline(none[1], color="tab:red", lw=1.0, ls=":"); ax.text(vals[-1] if ax is not axs[2] else vals[1], none[1] + (1.0 if ax is not axs[2] else -3.6), "highest ambiguity with no SATA", color="tab:red", fontsize=8.5, ha="right" if ax is not axs[2] else "left")
        ax.set_xlabel(name); ax.grid(alpha=0.3); ax.set_ylim(-62, 4)
        print("   " + name + ":", ", ".join("%g -> peak %.2f / amb %.1f" % (q, a_, b_) for q, (a_, b_) in zip(vals, res)))
    axs[0].set_ylabel("[dB re. reference peak]"); axs[0].legend(fontsize=8.5, loc="center right"); axs[0].set_xscale("log", base=2); axs[0].set_xticks([1, 2, 4, 8]); axs[0].set_xticklabels(["1", "2", "4", "8"])
    fig.suptitle("SATA whole band on every channel, then the reconstruction (table with the targets and their folded-Doppler copies); one setting changed at a time", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(png, dpi=170); plt.close(fig)
    o1, o8 = (kern(ch[3].copy(), R.channel_tables(cfg, dC0, 60, True)[0][3], r0, prfc, 1, v, 0.0, wl, r0, inverse=True, sata_osf=q) for q in (1, 8))
    print("   channel 3 after the kernel, sata_osf = 1 against 8: max |difference| / max |signal| = %.2e" % (np.abs(o1 - o8).max() / np.abs(o8).max()))
    print("   wrong sign (direct branch with the same table): peak %.2f dB, highest ambiguity %.1f dB" % run(inverse=False))
    print("   table with the targets only (no copies): peak %.2f dB, highest ambiguity %.1f dB" % run(copies=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sata_guide")
    ap.add_argument("--only", nargs="*", default=None, help="figure numbers to make, e.g. --only 7 9")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg, tr, parts, t, tj = scene()
    P = lambda name: os.path.join(a.out, name)
    jobs = {1: lambda: g01(cfg, parts, t, tj, P("g01_target_signal.png")), 2: lambda: g02(cfg, parts, t, tj, P("g02_why_subapertures.png")),
            3: lambda: g03(cfg, parts, t, tj, P("g03_one_subaperture.png")), 4: lambda: g04(cfg, parts, t, tj, P("g04_length_tradeoff.png")),
            5: lambda: g05(cfg, parts, t, tj, P("g05_zero_padding.png")), 6: lambda: g06(cfg, P("g06_ruler.png")),
            7: lambda: g07(cfg, parts, t, tj, P("g07_block_walkthrough.png")), 8: lambda: g08(cfg, parts, t, tj, P("g08_per_target.png")),
            9: lambda: g09(cfg, tr, parts, t, tj, P("g09_stitching.png")), 10: lambda: g10(cfg, tr, parts, t, tj, P("g10_table_halfwidth.png")),
            11: lambda: g11(P("g11_recon_scans.png"))}
    for k, job in jobs.items():
        if a.only is None or str(k) in a.only:
            job()
    print("wrote the figures in", a.out)


if __name__ == "__main__":
    main()
